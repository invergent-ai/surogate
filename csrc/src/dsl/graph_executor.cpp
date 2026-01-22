// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Graph executor (DSL-driven).

#include "dsl/graph_executor.h"
#include "dsl/autodiff.h"
#include "dsl/dsl_runtime.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include "modules/fp8_scaling_state.h"
#include "modules/fp8_scaling_config.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/lora/lora_run_state.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/allocator.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "kernels/kernels.h"

namespace dsl {
namespace {

constexpr std::string_view kSavedPrefix = "saved.";

bool env_enabled(const char* name) {
    if (!name || !*name) {
        return false;
    }
    const char* value = std::getenv(name);
    if (!value) {
        return false;
    }
    return std::string_view(value) != "0" && std::string_view(value) != "false";
}

struct ExecState {
    DslRunState& rs;
    DslParamStore& weights;
    DslGradStore& grads;
    const modules::ModelConfig& config;
    long B = 0;
    long T = 0;
    ShapeEnv shape_env{};
    const std::unordered_map<std::string, std::string>* view_sources = nullptr;
    const std::unordered_map<std::string, std::string>* view_sources_rev = nullptr;

    std::unordered_map<std::string, Tensor> tensors;
    std::unordered_set<std::string> zero_tensors;
    std::vector<Tensor> temps;

};

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::optional<std::string> base_param_from_grad(std::string_view name) {
    if (!starts_with(name, "d_")) {
        return std::nullopt;
    }
    std::string base(name.substr(2));
    const std::string_view accum_tag = "_accum_";
    const std::string_view from_tag = "_from_";
    std::size_t pos = std::string::npos;
    std::size_t pos_accum = base.find(accum_tag);
    std::size_t pos_from = base.find(from_tag);
    if (pos_accum != std::string::npos) {
        pos = pos_accum;
    }
    if (pos_from != std::string::npos) {
        if (pos == std::string::npos || pos_from < pos) {
            pos = pos_from;
        }
    }
    if (pos != std::string::npos) {
        base = base.substr(0, pos);
    }
    return base;
}

bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name);

bool infer_block_tensor_shape(const ExecState& st, std::string_view name, std::vector<long>& shape) {
    int layer_idx = -1;
    std::string field;
    if (!parse_block_param(name, layer_idx, field)) {
        return false;
    }
    const long B = st.B;
    const long T = st.T;
    const long C = st.config.HiddenSize;
    const long D = st.config.IntermediateSize;
    const long Hq = st.config.NumQueryHeads;
    const long Hkv = st.config.NumKeyValHeads;
    const long Hs = st.config.head_size();
    const long QKV = Hs * (Hq + 2 * Hkv);
    const long AttnDim = Hq * Hs;
    const long MUp = 2 * D;

    if (field == "ln1" || field == "ln2" || field == "res_att" || field == "res_ffn" || field == "att_out" || field == "mlp_down") {
        shape = {B, T, C};
        return true;
    }
    if (field == "mlp_up") {
        shape = {B, T, MUp};
        return true;
    }
    if (field == "swiglu") {
        shape = {B, T, D};
        return true;
    }
    if (field == "qkv" || field == "qkv_rope" || field == "qkv_flat" || field == "qkv_biased") {
        shape = {B, T, QKV};
        return true;
    }
    if (field == "att") {
        shape = {B, T, AttnDim};
        return true;
    }
    if (field == "q_rstd") {
        shape = {B, T, Hq};
        return true;
    }
    if (field == "k_rstd") {
        shape = {B, T, Hkv};
        return true;
    }
    if (field == "lse") {
        shape = {B, Hq, T};
        return true;
    }
    return false;
}

const AttrValue* find_attr(const AttrMap& attrs, std::string_view key) {
    auto it = attrs.find(std::string(key));
    if (it == attrs.end()) {
        return nullptr;
    }
    return &it->second;
}

std::optional<std::string> attr_string(const AttrValue& value) {
    if (auto v = std::get_if<std::string>(&value.value)) {
        return *v;
    }
    return std::nullopt;
}

[[maybe_unused]] std::optional<long> attr_int(const AttrValue& value) {
    if (auto v = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<long>(*v);
    }
    if (auto v = std::get_if<double>(&value.value)) {
        return static_cast<long>(*v);
    }
    return std::nullopt;
}

std::optional<double> attr_double(const AttrValue& value) {
    if (auto v = std::get_if<double>(&value.value)) {
        return *v;
    }
    if (auto v = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<double>(*v);
    }
    return std::nullopt;
}

[[maybe_unused]] std::optional<bool> attr_bool(const AttrValue& value) {
    if (auto v = std::get_if<bool>(&value.value)) {
        return *v;
    }
    return std::nullopt;
}

void augment_shape_env(ShapeEnv& env, const AttrMap& config) {
    auto get_long = [&](std::string_view key) -> std::optional<long> {
        auto it = config.find(std::string(key));
        if (it == config.end()) return std::nullopt;
        if (auto v = std::get_if<std::int64_t>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        if (auto v = std::get_if<double>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        return std::nullopt;
    };

    auto d_model = get_long("d_model");
    if (!d_model) {
        d_model = get_long("hidden_size");
    }
    auto num_q = get_long("num_query_heads");
    if (!num_q) {
        num_q = get_long("num_attention_heads");
    }
    auto num_kv = get_long("num_kv_heads");
    if (!num_kv) {
        num_kv = get_long("num_key_value_heads");
    }
    auto head_size = get_long("head_size");
    if (!head_size) {
        head_size = get_long("head_dim");
    }
    auto d_ff = get_long("d_ff");
    if (!d_ff) {
        d_ff = get_long("intermediate_size");
    }
    auto vocab = get_long("vocab_size");
    if (!vocab) {
        vocab = get_long("vocab");
    }

    if (d_model) {
        env.values.emplace("C", *d_model);
    }
    if (num_q) {
        env.values.emplace("Hq", *num_q);
    }
    if (num_kv) {
        env.values.emplace("Hkv", *num_kv);
    } else if (num_q) {
        env.values.emplace("Hkv", *num_q);
    }
    long Hq = env.values.count("Hq") ? env.values.at("Hq") : 0;
    long Hkv = env.values.count("Hkv") ? env.values.at("Hkv") : 0;
    long C = env.values.count("C") ? env.values.at("C") : 0;
    if (!head_size && Hq > 0 && C > 0) {
        head_size = C / Hq;
    }
    if (head_size) {
        env.values.emplace("D", *head_size);
    }
    if (d_ff) {
        env.values.emplace("M", *d_ff);
        env.values.emplace("MUp", 2 * (*d_ff));
    }
    if (vocab) {
        env.values.emplace("V", *vocab);
    }
    if (Hq > 0 && head_size) {
        env.values.emplace("AttnDim", Hq * (*head_size));
    }
    if (head_size && Hq > 0 && Hkv > 0) {
        env.values.emplace("QKV", (Hq + 2 * Hkv) * (*head_size));
    }
}

std::vector<long> resolve_attr_shape(const AttrValue& value, const ShapeEnv& env) {
    const auto* list_ptr = std::get_if<AttrValue::ListPtr>(&value.value);
    if (!list_ptr || !*list_ptr) {
        throw std::runtime_error("DSL graph executor: shape attr is not a list");
    }
    std::vector<long> shape;
    shape.reserve((*list_ptr)->size());
    for (const auto& item : **list_ptr) {
        if (auto v = std::get_if<std::int64_t>(&item.value)) {
            shape.push_back(static_cast<long>(*v));
            continue;
        }
        if (auto v = std::get_if<double>(&item.value)) {
            shape.push_back(static_cast<long>(*v));
            continue;
        }
        if (auto v = std::get_if<std::string>(&item.value)) {
            shape.push_back(resolve_dim(Dim::computed(*v), env));
            continue;
        }
        throw std::runtime_error("DSL graph executor: unsupported shape attr item");
    }
    return shape;
}

[[maybe_unused]] ETensorDType resolve_dtype(const TensorInfo& info, ETensorDType fallback) {
    if (info.dtype.has_value()) {
        return info.dtype.value();
    }
    return fallback;
}

Tensor view_tensor(const Tensor& src, const std::vector<long>& shape) {
    if (shape.size() > MAX_TENSOR_DIM) {
        throw std::runtime_error("DSL graph executor: view rank too large");
    }
    Tensor out = src;
    out.Rank = static_cast<int>(shape.size());
    for (int i = 0; i < out.Rank; ++i) {
        out.Sizes[i] = shape[i];
    }
    for (int i = out.Rank; i < MAX_TENSOR_DIM; ++i) {
        out.Sizes[i] = 1;
    }
    return out;
}

bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name);

std::string tensor_shape_str(const Tensor& t) {
    std::string out = "[";
    for (int i = 0; i < t.Rank; ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(t.Sizes[i]);
    }
    out += "]";
    return out;
}

bool tensor_shape_matches(const Tensor& t, const std::vector<long>& shape) {
    if (t.Rank != static_cast<int>(shape.size())) {
        return false;
    }
    for (int i = 0; i < t.Rank; ++i) {
        if (t.Sizes[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

std::size_t shape_nelem(const std::vector<long>& shape) {
    std::size_t total = 1;
    for (long dim : shape) {
        total *= static_cast<std::size_t>(dim);
    }
    return total;
}

Tensor view_for_shape(const Tensor& src, const std::vector<long>& shape, const std::string& name) {
    if (shape.empty()) {
        return src;
    }
    if (shape_nelem(shape) != src.nelem()) {
        throw std::runtime_error("DSL graph executor: shape mismatch for tensor " + name);
    }
    return view_tensor(src, shape);
}

Tensor* resolve_block_activation_tensor(ExecState& st, const std::string& name, ETensorDType dtype,
                                        const std::vector<long>& shape) {
    int layer_idx = -1;
    std::string field;
    if (!parse_block_param(name, layer_idx, field)) {
        return nullptr;
    }

    auto map_tensor = [&](Tensor& base) -> Tensor* {
        if (!base.Data) {
            st.rs.temp_acquire(base);
            st.temps.push_back(base);
        }
        if (dtype != base.DType) {
            throw std::runtime_error("DSL graph executor: dtype mismatch for tensor " + name);
        }
        Tensor view = view_for_shape(base, shape, name);
        auto [it, inserted] = st.tensors.emplace(name, view);
        if (!inserted) {
            it->second = view;
        }
        return &it->second;
    };

    auto& acts = st.rs.simplified_acts(layer_idx);
    if (field == "ln1") return map_tensor(acts.ln1);
    if (field == "ln1_rstd") return map_tensor(acts.ln1_rstd);
    if (field == "ln2") return map_tensor(acts.ln2);
    if (field == "ln2_rstd") return map_tensor(acts.ln2_rstd);
    if (field == "q_rstd") return map_tensor(acts.q_rstd);
    if (field == "k_rstd") return map_tensor(acts.k_rstd);
    if (field == "qkv" || field == "qkv_flat" || field == "qkv_biased") {
        return map_tensor(acts.qkv);
    }
    if (field == "qkv_rope") {
        if (acts.qkv_rope.Data) {
            return map_tensor(acts.qkv_rope);
        }
        return map_tensor(acts.qkv);
    }
    if (field == "lse") return map_tensor(acts.lse);
    if (field == "att") return map_tensor(acts.att);
    if (field == "att_out" || field == "att_out_flat") return map_tensor(acts.att_out);
    if (field == "res_att") return map_tensor(acts.residual_att);
    if (field == "mlp_up" || field == "mlp_up_flat") return map_tensor(acts.mlp_up);
    if (field == "swiglu") return map_tensor(acts.swiglu);
    if (field == "mlp_down" || field == "mlp_down_flat") return map_tensor(acts.mlp_down);
    if (field == "res_ffn") {
        Tensor& res = st.rs.get_residual(layer_idx, st.rs.MainStream);
        return map_tensor(res);
    }

    return nullptr;
}

Tensor* resolve_block_activation_base(ExecState& st, const std::string& name) {
    int layer_idx = -1;
    std::string field;
    if (!parse_block_param(name, layer_idx, field)) {
        return nullptr;
    }

    auto map_tensor = [&](Tensor& base) -> Tensor* {
        if (!base.Data) {
            st.rs.temp_acquire(base);
            st.temps.push_back(base);
        }
        auto [it, inserted] = st.tensors.emplace(name, base);
        if (!inserted) {
            it->second = base;
        }
        return &it->second;
    };

    auto& acts = st.rs.simplified_acts(layer_idx);
    if (field == "ln1") return map_tensor(acts.ln1);
    if (field == "ln1_rstd") return map_tensor(acts.ln1_rstd);
    if (field == "ln2") return map_tensor(acts.ln2);
    if (field == "ln2_rstd") return map_tensor(acts.ln2_rstd);
    if (field == "q_rstd") return map_tensor(acts.q_rstd);
    if (field == "k_rstd") return map_tensor(acts.k_rstd);
    if (field == "qkv" || field == "qkv_flat" || field == "qkv_biased") {
        return map_tensor(acts.qkv);
    }
    if (field == "qkv_rope") {
        if (acts.qkv_rope.Data) {
            return map_tensor(acts.qkv_rope);
        }
        return map_tensor(acts.qkv);
    }
    if (field == "lse") return map_tensor(acts.lse);
    if (field == "att") return map_tensor(acts.att);
    if (field == "att_out" || field == "att_out_flat") return map_tensor(acts.att_out);
    if (field == "res_att") return map_tensor(acts.residual_att);
    if (field == "mlp_up" || field == "mlp_up_flat") return map_tensor(acts.mlp_up);
    if (field == "swiglu") return map_tensor(acts.swiglu);
    if (field == "mlp_down" || field == "mlp_down_flat") return map_tensor(acts.mlp_down);
    if (field == "res_ffn") {
        Tensor& res = st.rs.get_residual(layer_idx, st.rs.MainStream);
        return map_tensor(res);
    }

    return nullptr;
}

Tensor* resolve_block_gradient_tensor(ExecState& st, const std::string& name, ETensorDType dtype,
                                      const std::vector<long>& shape) {
    if (!starts_with(name, "d_")) {
        return nullptr;
    }
    const std::string base_name = name.substr(2);
    int layer_idx = -1;
    std::string field;
    if (!parse_block_param(base_name, layer_idx, field)) {
        return nullptr;
    }

    auto map_tensor = [&](Tensor& base) -> Tensor* {
        if (!base.Data) {
            st.rs.temp_acquire(base);
            st.temps.push_back(base);
        }
        if (dtype != base.DType) {
            throw std::runtime_error("DSL graph executor: dtype mismatch for tensor " + name);
        }
        Tensor view = view_for_shape(base, shape, name);
        auto [it, inserted] = st.tensors.emplace(name, view);
        if (!inserted) {
            it->second = view;
        }
        return &it->second;
    };

    auto& grads = st.rs.simplified_grads(layer_idx);
    if (field == "ln1") return map_tensor(grads.d_ln1);
    if (field == "qkv" || field == "qkv_rope") {
        return map_tensor(grads.d_qkv);
    }
    if (field == "att") return map_tensor(grads.d_att);
    if (field == "swiglu") {
        return map_tensor(grads.d_swiglu);
    }
    if (field == "mlp_up") {
        return map_tensor(grads.d_mlp_up);
    }
    if (field == "mlp_down" || field == "mlp_down_flat") {
        return map_tensor(grads.d_mlp_down);
    }
    if (field == "ln2") return map_tensor(grads.d_ln2);
    if (field == "res_att") return map_tensor(grads.d_res_att);
    if (field == "res_ffn") return map_tensor(grads.d_res_ffn);

    return nullptr;
}

Tensor* resolve_gradient_view_tensor(ExecState& st,
                                     const std::string& name,
                                     const std::unordered_map<std::string, Tensor>& saved) {
    if (!starts_with(name, "d_")) {
        return nullptr;
    }
    if (!st.view_sources_rev) {
        return nullptr;
    }
    const std::string base_in = name.substr(2);
    auto it_rev = st.view_sources_rev->find(base_in);
    if (it_rev == st.view_sources_rev->end()) {
        return nullptr;
    }
    const std::string base_out = it_rev->second;
    const std::string base_grad = "d_" + base_out;

    Tensor* base = nullptr;
    if (auto it = st.tensors.find(base_grad); it != st.tensors.end()) {
        base = &it->second;
    }
    if (!base) {
        return nullptr;
    }

    auto it_shape = saved.find(base_in);
    std::vector<long> shape;
    if (it_shape == saved.end()) {
        if (ends_with(base_in, "_flat") && base->Rank >= 2) {
            const long first = base->Sizes[0];
            const long second = base->Sizes[1];
            const long rest = static_cast<long>(base->nelem() / static_cast<std::size_t>(first * second));
            shape = {first * second, rest};
        } else {
            return nullptr;
        }
    } else {
        shape.assign(it_shape->second.Sizes.begin(),
                     it_shape->second.Sizes.begin() + it_shape->second.Rank);
    }
    Tensor view = view_for_shape(*base, shape, name);
    auto [ins_it, inserted] = st.tensors.emplace(name, view);
    if (!inserted) {
        ins_it->second = view;
    }
    return &ins_it->second;
}

Tensor& ensure_tensor(ExecState& st, const std::string& name, ETensorDType dtype, const std::vector<long>& shape) {
    auto it = st.tensors.find(name);
    if (it != st.tensors.end()) {
        return it->second;
    }
    if (Tensor* mapped = resolve_block_gradient_tensor(st, name, dtype, shape)) {
        return *mapped;
    }
    if (Tensor* mapped = resolve_block_activation_tensor(st, name, dtype, shape)) {
        return *mapped;
    }
    Tensor t = st.rs.temp_alloc(dtype, shape);
    st.temps.push_back(t);
    auto [ins_it, inserted] = st.tensors.emplace(name, t);
    (void)inserted;
    return ins_it->second;
}

Tensor& resolve_param_tensor(ExecState& st, const std::string& name);

Tensor& get_tensor(ExecState& st, const std::string& name, const std::unordered_map<std::string, Tensor>& saved) {
    // Check for explicit "saved." prefix first
    if (starts_with(name, kSavedPrefix)) {
        std::string key = std::string(name.substr(kSavedPrefix.size()));
        auto it = saved.find(key);
        if (it == saved.end()) {
            throw std::runtime_error("DSL graph executor: missing saved tensor " + key);
        }
        std::vector<long> shape(it->second.Sizes.begin(), it->second.Sizes.begin() + it->second.Rank);
        // Prefer a live tensor if it exists (e.g., recomputed activations), but validate shape.
        if (auto live_it = st.tensors.find(key); live_it != st.tensors.end()) {
            if (tensor_shape_matches(live_it->second, shape)) {
                return live_it->second;
            }
        }
        if (st.view_sources) {
            auto vs_it = st.view_sources->find(key);
            if (vs_it != st.view_sources->end()) {
                const std::string& src_name = vs_it->second;
                Tensor* base = nullptr;
                if (auto it_src = st.tensors.find(src_name); it_src != st.tensors.end()) {
                    base = &it_src->second;
                }
                if (!base) {
                    base = resolve_block_activation_base(st, src_name);
                }
                if (!base) {
                    auto it_src_saved = saved.find(src_name);
                    if (it_src_saved != saved.end()) {
                        base = const_cast<Tensor*>(&it_src_saved->second);
                    } else {
                        try {
                            base = &resolve_param_tensor(st, src_name);
                        } catch (...) {
                            base = nullptr;
                        }
                    }
                }
                if (base) {
                    Tensor view = view_for_shape(*base, shape, key);
                    auto [ins_it, inserted] = st.tensors.emplace(key, view);
                    if (!inserted) {
                        ins_it->second = view;
                    }
                    return ins_it->second;
                }
            }
        }
        if (Tensor* mapped = resolve_block_activation_tensor(st, key, it->second.DType, shape)) {
            return *mapped;
        }
        if (Tensor* mapped = resolve_block_gradient_tensor(st, key, it->second.DType, shape)) {
            return *mapped;
        }
        if (st.rs.ffn_temps_on_stack()) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(key, layer_idx, field)) {
                throw std::runtime_error(
                    "DSL graph executor: recompute_block active but saved tensor not mappable: " + key);
            }
        }
        return const_cast<Tensor&>(it->second);
    }
    // Check current tensors (params, intermediates, gradients)
    auto it = st.tensors.find(name);
    if (it != st.tensors.end()) {
        if (starts_with(name, "d_") && ends_with(name, "_flat") && it->second.Rank != 2) {
            if (Tensor* mapped = resolve_gradient_view_tensor(st, name, saved)) {
                return *mapped;
            }
        }
        return it->second;
    }
    if (Tensor* mapped = resolve_gradient_view_tensor(st, name, saved)) {
        return *mapped;
    }
    // Fall back to saved tensors (for forward intermediates used in backward)
    auto sit = saved.find(name);
    if (sit != saved.end()) {
        return const_cast<Tensor&>(sit->second);
    }
    // Try resolving as a parameter
    return resolve_param_tensor(st, name);
}

// Try to get a tensor by name, returning nullptr if not found (no throw)
Tensor* try_get_tensor(ExecState& st, const std::string& name, std::unordered_map<std::string, Tensor>& saved) {
    if (starts_with(name, kSavedPrefix)) {
        std::string key = std::string(name.substr(kSavedPrefix.size()));
        auto it = saved.find(key);
        if (it == saved.end()) {
            return nullptr;
        }
        std::vector<long> shape(it->second.Sizes.begin(), it->second.Sizes.begin() + it->second.Rank);
        if (auto it_live = st.tensors.find(key); it_live != st.tensors.end()) {
            if (tensor_shape_matches(it_live->second, shape)) {
                return &it_live->second;
            }
        }
        if (st.view_sources) {
            auto vs_it = st.view_sources->find(key);
            if (vs_it != st.view_sources->end()) {
                const std::string& src_name = vs_it->second;
                Tensor* base = nullptr;
                if (auto it_src = st.tensors.find(src_name); it_src != st.tensors.end()) {
                    base = &it_src->second;
                }
                if (!base) {
                    base = resolve_block_activation_base(st, src_name);
                }
                if (!base) {
                    auto it_src_saved = saved.find(src_name);
                    if (it_src_saved != saved.end()) {
                        base = &it_src_saved->second;
                    } else {
                        try {
                            base = &resolve_param_tensor(st, src_name);
                        } catch (...) {
                            base = nullptr;
                        }
                    }
                }
                if (base) {
                    Tensor view = view_for_shape(*base, shape, key);
                    auto [ins_it, inserted] = st.tensors.emplace(key, view);
                    if (!inserted) {
                        ins_it->second = view;
                    }
                    return &ins_it->second;
                }
            }
        }
        if (Tensor* mapped = resolve_block_activation_tensor(st, key, it->second.DType, shape)) {
            return mapped;
        }
        if (Tensor* mapped = resolve_block_gradient_tensor(st, key, it->second.DType, shape)) {
            return mapped;
        }
        if (st.rs.ffn_temps_on_stack()) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(key, layer_idx, field)) {
                return nullptr;
            }
        }
        return &it->second;
    }
    auto it = st.tensors.find(name);
    if (it != st.tensors.end()) {
        if (starts_with(name, "d_") && ends_with(name, "_flat") && it->second.Rank != 2) {
            if (Tensor* mapped = resolve_gradient_view_tensor(st, name, saved)) {
                return mapped;
            }
        }
        if (starts_with(name, "d_") && st.view_sources_rev) {
            const std::string base_in = name.substr(2);
            auto it_rev = st.view_sources_rev->find(base_in);
            if (it_rev != st.view_sources_rev->end()) {
                auto it_shape = saved.find(base_in);
                if (it_shape != saved.end()) {
                    std::vector<long> shape(it_shape->second.Sizes.begin(),
                                            it_shape->second.Sizes.begin() + it_shape->second.Rank);
                    if (!tensor_shape_matches(it->second, shape)) {
                        if (Tensor* mapped = resolve_gradient_view_tensor(st, name, saved)) {
                            return mapped;
                        }
                    }
                }
            }
        }
        return &it->second;
    }
    if (Tensor* mapped = resolve_gradient_view_tensor(st, name, saved)) {
        return mapped;
    }
    auto sit = saved.find(name);
    if (sit != saved.end()) {
        return &sit->second;
    }
    // Try resolving as a parameter
    try {
        return &resolve_param_tensor(st, name);
    } catch (...) {
        return nullptr;
    }
}

bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name) {
    auto dot = name.find('.');
    if (dot == std::string_view::npos) return false;
    auto prefix = name.substr(0, dot);
    auto rest = name.substr(dot + 1);

    // blocks[<idx>]
    if (starts_with(prefix, "blocks[")) {
        auto close = prefix.find(']');
        if (close == std::string_view::npos) return false;
        auto idx_str = prefix.substr(7, close - 7);
        try {
            layer_idx = std::stoi(std::string(idx_str));
        } catch (...) {
            return false;
        }
        param_name = std::string(rest);
        return true;
    }

    // blocks.<idx>
    if (starts_with(prefix, "blocks")) {
        auto idx_str = name.substr(dot + 1);
        auto dot2 = idx_str.find('.');
        if (dot2 == std::string_view::npos) return false;
        try {
            layer_idx = std::stoi(std::string(idx_str.substr(0, dot2)));
        } catch (...) {
            return false;
        }
        param_name = std::string(idx_str.substr(dot2 + 1));
        return true;
    }

    return false;
}

std::optional<modules::MatmulOp> matmul_op_from_weight(std::string_view name, int& layer_idx) {
    std::string field;
    if (!parse_block_param(name, layer_idx, field)) {
        return std::nullopt;
    }
    if (field == "qkv_weight") return modules::MatmulOp::QKV;
    if (field == "out_weight") return modules::MatmulOp::AttnOut;
    if (field == "mlp_up_weight") return modules::MatmulOp::MLPUp;
    if (field == "mlp_down_weight") return modules::MatmulOp::MLPDown;
    return std::nullopt;
}

bool allow_quant_layer(const RuntimeOptions& options, const modules::ModelConfig& config, int layer_idx) {
    if (layer_idx < 0) return false;
    const int skip_first = options.RecipeOptions.skip_quant_first_layers;
    const int skip_last = options.RecipeOptions.skip_quant_last_layers;
    if (skip_first > 0 && layer_idx < skip_first) return false;
    if (skip_last > 0 && layer_idx >= static_cast<int>(config.NumLayers) - skip_last) return false;
    return true;
}

Tensor* fp8_forward_buffer(DslRunState& rs, modules::MatmulOp op) {
    if (!rs.has_fp8_forward()) return nullptr;
    auto& q = rs.fp8_forward_quants();
    switch (op) {
        case modules::MatmulOp::QKV:
            return &q.ln1;
        case modules::MatmulOp::MLPUp:
            return &q.ln2;
        case modules::MatmulOp::AttnOut:
            return &q.att;
        case modules::MatmulOp::MLPDown:
            return &q.swiglu;
        default:
            return nullptr;
    }
}

Tensor* fp8_grad_buffer(DslRunState& rs, modules::MatmulOp op) {
    if (!rs.has_fp8_hybrid_backward()) return nullptr;
    auto& q = rs.simplified_quant_grads();
    switch (op) {
        case modules::MatmulOp::QKV:
            return &q.d_qkv;
        case modules::MatmulOp::MLPUp:
            return &q.d_mlp_up;
        case modules::MatmulOp::AttnOut:
            return &q.d_res_att;
        case modules::MatmulOp::MLPDown:
            return &q.d_res_ffn;
        default:
            return nullptr;
    }
}

int fp8_quantizer_index(const DslRunState& rs, modules::MatmulOp op, int layer_idx) {
    if (!rs.has_fp8_delayed_scaling()) return -1;
    switch (op) {
        case modules::MatmulOp::QKV:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_LN1);
        case modules::MatmulOp::MLPUp:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_LN2);
        case modules::MatmulOp::AttnOut:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_ATT);
        case modules::MatmulOp::MLPDown:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_SWIGLU);
        default:
            return -1;
    }
}

Tensor& resolve_param_tensor(ExecState& st, const std::string& name) {
    if (name.find("rope_freqs") != std::string::npos) {
        auto& freqs = st.rs.non_block_activations().freq_cis;
        if (!freqs.Data) {
            throw std::runtime_error("DSL graph executor: RoPE frequencies not allocated");
        }
        return freqs;
    }

    if (st.weights.has(name)) {
        return st.weights.get(name);
    }
    if ((name == "embeddings" || name == "embed_tokens") && st.weights.has("embedding")) {
        return st.weights.get("embedding");
    }
    if ((name == "final_norm_weight" || name == "norm") && st.weights.has("final_norm")) {
        return st.weights.get("final_norm");
    }
    if (name == "lm_head_weight" && st.weights.has("lm_head")) {
        return st.weights.get("lm_head");
    }

    throw std::runtime_error("DSL graph executor: unknown param " + name);
}

// Resolve view shape from either "shape" or "shape_like" attribute
std::vector<long> resolve_view_shape(
    const Operation& op,
    const ShapeEnv& env,
    ExecState& st,
    std::unordered_map<std::string, Tensor>& saved) {
    // Check for shape_like attribute (used by autodiff to reference tensor shape)
    auto* shape_like_attr = find_attr(op.attrs, "shape_like");
    if (shape_like_attr) {
        if (auto ref_name = attr_string(*shape_like_attr)) {
            // Try to get shape from referenced tensor (may be saved or intermediate)
            Tensor* ref = try_get_tensor(st, *ref_name, saved);
            if (ref) {
                return std::vector<long>(ref->Sizes.begin(), ref->Sizes.begin() + ref->Rank);
            }
            // If the reference tensor isn't available, try to infer shape from input tensor
            // This handles cases like backward view for logits where logits_flat isn't saved
            // For view backward: d_logits [B,T,V] -> d_logits_flat [B*T,V]
            // The input (d_logits) shape can be flattened to get output shape
            if (!op.inputs.empty()) {
                Tensor* input = try_get_tensor(st, op.inputs[0], saved);
                if (input && input->Rank >= 2) {
                    // Assume flattening first (Rank-1) dimensions and keeping last
                    // This is a heuristic for the common [B,T,V] -> [B*T,V] case
                    long flat_dim = 1;
                    for (int i = 0; i < input->Rank - 1; ++i) {
                        flat_dim *= input->Sizes[i];
                    }
                    return {flat_dim, input->Sizes[input->Rank - 1]};
                }
            }
            throw std::runtime_error("DSL graph executor: view shape_like reference not found: " + *ref_name);
        }
        throw std::runtime_error("DSL graph executor: view shape_like attr must be a string");
    }

    auto* shape_attr = find_attr(op.attrs, "shape");
    if (!shape_attr) {
        throw std::runtime_error("DSL graph executor: view missing shape or shape_like attr");
    }
    return resolve_attr_shape(*shape_attr, env);
}

EMMTranspose parse_transpose(const AttrMap& attrs) {
    auto attr = find_attr(attrs, "transpose");
    if (!attr) {
        return EMMTranspose::NN;
    }
    if (auto s = attr_string(*attr)) {
        if (*s == "NN") return EMMTranspose::NN;
        if (*s == "NT") return EMMTranspose::NT;
        if (*s == "TN") return EMMTranspose::TN;
        if (*s == "TT") return EMMTranspose::TT;
    }
    throw std::runtime_error("DSL graph executor: invalid transpose attr");
}

EMMTranspose swap_transpose(EMMTranspose mode) {
    // Row-major GEMM mapping to column-major: swap A/B, swap M/N, and swap transpose flags.
    // NN -> NN, NT -> TN, TN -> NT, TT -> TT.
    switch (mode) {
        case EMMTranspose::NN:
            return EMMTranspose::NN;
        case EMMTranspose::NT:
            return EMMTranspose::TN;
        case EMMTranspose::TN:
            return EMMTranspose::NT;
        case EMMTranspose::TT:
            return EMMTranspose::TT;
    }
    return EMMTranspose::NN;
}

void matmul_dims(const Tensor& a, const Tensor& b, EMMTranspose mode, int& M, int& N, int& K) {
    if (a.Rank != 2 || b.Rank != 2) {
        throw std::runtime_error("DSL graph executor: matmul expects rank-2 tensors");
    }
    const long a0 = a.Sizes[0];
    const long a1 = a.Sizes[1];
    const long b0 = b.Sizes[0];
    const long b1 = b.Sizes[1];
    const bool transA = (mode == EMMTranspose::TN || mode == EMMTranspose::TT);
    const bool transB = (mode == EMMTranspose::NT || mode == EMMTranspose::TT);
    const long a_rows = transA ? a1 : a0;
    const long a_cols = transA ? a0 : a1;
    const long b_rows = transB ? b1 : b0;
    const long b_cols = transB ? b0 : b1;
    if (a_cols != b_rows) {
        throw std::runtime_error("DSL graph executor: matmul dimension mismatch");
    }
    M = static_cast<int>(a_rows);
    N = static_cast<int>(b_cols);
    K = static_cast<int>(a_cols);
}

bool is_required_op(const Operation& op, const std::unordered_set<std::string>& needed) {
    for (const auto& out : op.outputs) {
        if (needed.count(out) > 0) {
            return true;
        }
    }
    return false;
}

std::vector<char> compute_required_ops(const Graph& graph, const std::vector<std::string>& outputs) {
    std::unordered_set<std::string> needed(outputs.begin(), outputs.end());
    std::vector<char> required(graph.operations.size(), 0);
    for (std::size_t idx = graph.operations.size(); idx-- > 0;) {
        const auto& op = graph.operations[idx];
        if (!is_required_op(op, needed)) {
            continue;
        }
        required[idx] = 1;
        for (const auto& inp : op.inputs) {
            needed.insert(inp);
        }
    }
    return required;
}

void free_temps(ExecState& st) {
    for (auto it = st.temps.rbegin(); it != st.temps.rend(); ++it) {
        st.rs.temp_free(*it);
    }
    st.temps.clear();
}

void reduce_loss(DslRunState& rs, long B, long T, NCCLCommunicator& comm) {
    deterministic_sum(rs.Losses.template get<float>(), rs.Losses.template get<float>(), B * T, rs.MainStream);
    comm.reduce_loss(rs.Losses.template get<float>(), rs.MainStream);
    CUDA_CHECK(cudaMemcpyAsync(rs.LossHost, rs.Losses.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost, rs.MainStream));
}

void add_bias_tensor(Tensor& out, const Tensor& bias, int B, int T, int OC, cudaStream_t stream) {
    if (out.DType != bias.DType) {
        throw std::runtime_error("DSL graph executor: bias_add dtype mismatch");
    }
    if (out.DType == ETensorDType::BF16) {
        add_bias(out.get<nv_bfloat16>(), bias.get<nv_bfloat16>(), B, T, OC, stream);
        return;
    }
    if (out.DType == ETensorDType::FP32) {
        add_bias(out.get<float>(), bias.get<float>(), B, T, OC, stream);
        return;
    }
    throw std::runtime_error("DSL graph executor: bias_add unsupported dtype");
}

template<typename Function>
void trace_or_execute_cuda_graph_with_stack(Function&& function,
                                            cudaStream_t stream,
                                            cudaGraphExec_t& instance,
                                            bool enabled,
                                            DeviceMemoryStack& stack,
                                            DeviceMemoryStack::Checkpoint& checkpoint) {
    if (!enabled) {
        function();
        return;
    }

    if (instance != nullptr) {
        stack.restore(checkpoint);
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        return;
    }

    checkpoint = stack.checkpoint();

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    function();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
}

Tensor recompute_lora_rmsnorm(modules::LoRARunState& lora_rs, const Tensor& residual, const Tensor& weight,
                              float eps, int B, int T, int C, cudaStream_t stream) {
    if (!lora_rs.recompute_ln.Data || !lora_rs.recompute_rstd.Data) {
        throw std::runtime_error("DSL graph executor: LoRA recompute buffers not allocated");
    }
    rmsnorm_forward(lora_rs.recompute_ln, lora_rs.recompute_rstd,
                    residual, weight, nullptr, eps, B, T, C, stream);
    return lora_rs.recompute_ln;
}

}  // namespace

GraphExecutor::GraphExecutor(const Module& module,
                             DslRunState& run_state,
                             DslParamStore& weights,
                             DslGradStore& grads,
                             const modules::ModelConfig& config,
                             const RuntimeOptions& options,
                             const GraphExecutorOptions& exec_options)
    : mModule(module),
      mRunState(run_state),
      mWeights(weights),
      mGrads(grads),
      mConfig(config),
      mOptions(options),
      mForward(module.forward ? &module.forward.value() : nullptr),
      mBackward(nullptr) {
    mGraphsEnabled = options.UseCudaGraphs;
    mBackwardGraphsEnabled = mGraphsEnabled;
    init(exec_options);
}

void GraphExecutor::set_lora_state(const modules::ModularLoRAConfig* config,
                                   modules::ModularLoRAWeightsManager* weights,
                                   modules::ModularLoRAGradsManager* grads,
                                   modules::LoRARunState* run_state) {
    mLoRAConfig = config;
    mLoRAWeights = weights;
    mLoRAGrads = grads;
    mLoRARunState = run_state;
}

void GraphExecutor::reset_cuda_graphs() {
    if (mForwardGraph) {
        (void)cudaGraphExecDestroy(mForwardGraph);
        mForwardGraph = nullptr;
    }
    for (auto& g : mBackwardGraph) {
        if (g) {
            (void)cudaGraphExecDestroy(g);
            g = nullptr;
        }
    }
}

void GraphExecutor::init(const GraphExecutorOptions& options) {
    if (!mForward) {
        throw std::runtime_error("DSL graph executor: module missing forward graph");
    }

    // Check if module has explicit backward graph
    if (mModule.backward.has_value()) {
        mBackward = &mModule.backward.value();
    } else if (options.auto_backward) {
        // Derive backward graph automatically using autodiff
        DeriveBackwardOptions derive_opts;
        derive_opts.loss_name = options.loss_name;
        derive_opts.auto_save = true;
        derive_opts.accumulate_grads = true;

        try {
            mDerivedBackward = derive_backward_graph(*mForward, derive_opts);
            mBackward = &mDerivedBackward.value();

            // Merge auto-computed saves with forward.save, but filter out:
            // 1. Graph outputs (they're re-computed during backward by classifier)
            // 2. Tensors produced by ops that depend on lm_head (not available in full=false mode)
            std::unordered_set<std::string> save_set(mForward->save.begin(), mForward->save.end());
            for (const auto& s : mDerivedBackward->save) {
                save_set.insert(s);
            }
            // Remove graph outputs - they're handled by the classifier
            for (const auto& [name, _] : mForward->outputs) {
                save_set.erase(name);
            }
            // Also remove tensors that are produced by ops that come after the last save-able tensor
            // (i.e., tensors that depend on lm_head which isn't available during training forward)
            // For now, we specifically exclude "logits_flat" as it's produced by the lm_head matmul
            save_set.erase("logits_flat");
            save_set.erase("logits");
            mSaveList.assign(save_set.begin(), save_set.end());

            if (options.debug_print_backward) {
                std::cerr << "[Autodiff] Derived backward graph with "
                          << mDerivedBackward->operations.size() << " operations\n";
                for (const auto& op : mDerivedBackward->operations) {
                    std::cerr << "  " << op.kernel_type << ": [";
                    for (size_t i = 0; i < op.inputs.size(); ++i) {
                        if (i > 0) std::cerr << ", ";
                        std::cerr << op.inputs[i];
                    }
                    std::cerr << "] -> [";
                    for (size_t i = 0; i < op.outputs.size(); ++i) {
                        if (i > 0) std::cerr << ", ";
                        std::cerr << op.outputs[i];
                    }
                    std::cerr << "]\n";
                }
                std::cerr << "[Autodiff] Auto-computed saves: ";
                for (const auto& s : mSaveList) {
                    std::cerr << s << " ";
                }
                std::cerr << "\n";
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("DSL graph executor: autodiff failed: ") + e.what());
        }
    }

    mViewSources.clear();
    mViewSourcesReverse.clear();
    mEmbeddingOutputs.clear();
    if (mForward) {
        for (const auto& op : mForward->operations) {
            const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
            if ((op_type == "view" || op_type == "reshape") && !op.outputs.empty() && !op.inputs.empty()) {
                const std::string& out = op.outputs.at(0);
                const std::string& in = op.inputs.at(0);
                mViewSources.emplace(out, in);
                mViewSourcesReverse.emplace(in, out);
            }
            if (op_type == "embedding" && !op.outputs.empty()) {
                mEmbeddingOutputs.push_back(op.outputs.at(0));
            }
        }
    }

    if (!mBackward) {
        throw std::runtime_error(
            "DSL graph executor: module missing backward graph (set auto_backward=true to derive automatically)");
    }

    auto is_noncapturable_op = [&](const Operation& op) {
        const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
        const bool is_embedding_bwd = (op_type == "embedding_backward" || op_type == "encoder_backward"
                                       || op.name == "embedding_backward" || op.name == "encoder_backward");
        if (!is_embedding_bwd) {
            return false;
        }
        for (const auto& out : op.outputs) {
            if (out.empty()) {
                continue;
            }
            if (auto param_name = base_param_from_grad(out)) {
                if (mWeights.has(*param_name) && mWeights.is_trainable(*param_name)) {
                    return true;
                }
            }
        }
        return false;
    };

    if (mBackward) {
        const auto& ops = mBackward->operations;
        const std::size_t op_count = ops.size();
        std::vector<char> noncapturable(op_count, 0);
        bool has_noncapturable = false;

        for (std::size_t idx = 0; idx < op_count; ++idx) {
            if (is_noncapturable_op(ops[idx])) {
                noncapturable[idx] = 1;
                has_noncapturable = true;
            }
        }

        if (has_noncapturable && op_count > 1) {
            std::unordered_map<std::string, std::size_t> producer;
            producer.reserve(op_count * 2);
            for (std::size_t idx = 0; idx < op_count; ++idx) {
                for (const auto& out : ops[idx].outputs) {
                    if (!out.empty()) {
                        producer[out] = idx;
                    }
                }
            }

            auto is_param_grad = [&](const std::string& name) {
                if (auto base = base_param_from_grad(name)) {
                    return mWeights.has(*base);
                }
                return false;
            };

            std::vector<char> core(op_count, 0);
            for (std::size_t idx = 0; idx < op_count; ++idx) {
                bool has_output = false;
                bool all_param_grads = true;
                for (const auto& out : ops[idx].outputs) {
                    if (out.empty()) {
                        continue;
                    }
                    has_output = true;
                    if (!is_param_grad(out)) {
                        all_param_grads = false;
                        break;
                    }
                }
                if (!has_output || !all_param_grads) {
                    core[idx] = 1;
                }
            }

            std::vector<std::size_t> stack;
            stack.reserve(op_count);
            for (std::size_t idx = 0; idx < op_count; ++idx) {
                if (core[idx]) {
                    stack.push_back(idx);
                }
            }
            while (!stack.empty()) {
                std::size_t idx = stack.back();
                stack.pop_back();
                for (const auto& inp : ops[idx].inputs) {
                    if (inp.empty()) {
                        continue;
                    }
                    auto it = producer.find(inp);
                    if (it == producer.end()) {
                        continue;
                    }
                    std::size_t prod_idx = it->second;
                    if (!core[prod_idx]) {
                        core[prod_idx] = 1;
                        stack.push_back(prod_idx);
                    }
                }
            }

            std::size_t tail_count = 0;
            for (std::size_t idx = 0; idx < op_count; ++idx) {
                if (!core[idx]) {
                    ++tail_count;
                }
            }

            if (tail_count > 0 && tail_count < op_count) {
                Graph* mutable_backward = mDerivedBackward ? &mDerivedBackward.value() : nullptr;
                if (!mutable_backward) {
                    mReorderedBackward = *mBackward;
                    mutable_backward = &mReorderedBackward.value();
                    mBackward = mutable_backward;
                }
                auto& mutable_ops = mutable_backward->operations;
                std::vector<Operation> reordered;
                reordered.reserve(op_count);
                for (std::size_t idx = 0; idx < op_count; ++idx) {
                    if (core[idx]) {
                        reordered.push_back(mutable_ops[idx]);
                    }
                }
                for (std::size_t idx = 0; idx < op_count; ++idx) {
                    if (!core[idx]) {
                        reordered.push_back(mutable_ops[idx]);
                    }
                }
                mutable_ops.swap(reordered);
            }
        }
    }

    // Backward CUDA graphs are not compatible with ops that sync on other streams.
    // If we encounter such ops, capture only the prefix and run the tail uncaptured.
    mBackwardGraphCapturable = true;
    mBackwardGraphCut = mBackward ? mBackward->operations.size() : 0;
    if (mBackward) {
        for (std::size_t idx = 0; idx < mBackward->operations.size(); ++idx) {
            if (is_noncapturable_op(mBackward->operations[idx])) {
                mBackwardGraphCapturable = false;
                mBackwardGraphCut = idx;
                break;
            }
        }
    }
    mBackwardGraphsEnabled = mGraphsEnabled && mBackwardGraphCut > 0;

    // If we didn't derive backward (using explicit backward from module), use forward.save
    if (mSaveList.empty()) {
        mSaveList = mForward->save;
    }
}

unsigned int GraphExecutor::next_rng_seed() {
    return static_cast<unsigned int>(mRng());
}

void GraphExecutor::prime_fp8_weight_cache(const std::vector<char>& required) {
    if (!mOptions.TrainingRecipe || !mRunState.has_fp8_forward()) {
        return;
    }
    if (!mForward) {
        return;
    }
    const auto& config = mConfig;
    for (std::size_t idx = 0; idx < mForward->operations.size(); ++idx) {
        if (!required.empty() && !required[idx]) {
            continue;
        }
        const auto& op = mForward->operations[idx];
        const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
        if (op_type != "matmul" && op_type != "matmul_bias") {
            continue;
        }
        if (op.inputs.size() < 2) {
            continue;
        }
        int layer_idx = -1;
        auto op_kind = matmul_op_from_weight(op.inputs.at(1), layer_idx);
        if (!op_kind.has_value()) {
            continue;
        }
        if (!allow_quant_layer(mOptions, config, layer_idx)) {
            continue;
        }
        const std::string& weight_name = op.inputs.at(1);
        if (!mWeights.has(weight_name)) {
            continue;
        }
        Tensor& weight = mWeights.get(weight_name);
        (void)get_fp8_cached_weight(weight_name, weight, mRunState.MainStream);
    }
}

const Tensor* GraphExecutor::get_fp8_cached_weight(const std::string& name, Tensor& weight, cudaStream_t stream) {
    const bool debug_cache = env_enabled("SUROGATE_DEBUG_DSL_FP8_CACHE");
    if (!mRunState.has_fp8_forward()) {
        return nullptr;
    }
    if (weight.DType == ETensorDType::FP8_E4M3) {
        return &weight;
    }
    if (!mWeights.has(name) || mWeights.is_trainable(name)) {
        return nullptr;
    }
    auto it = mFP8WeightCache.find(name);
    if (it == mFP8WeightCache.end()) {
        FP8WeightCacheEntry entry{};
        std::vector<long> shape(weight.Sizes.begin(), weight.Sizes.begin() + weight.Rank);
        if (debug_cache) {
            fprintf(stderr, "[DSL TRACE] fp8_cache alloc name=%s dtype=%s rank=%d\n",
                    name.c_str(), dtype_to_str(weight.DType), weight.Rank);
            fflush(stderr);
        }
        entry.weight = mRunState.Allocator->allocate(ETensorDType::FP8_E4M3,
                                                     ("fp8_cache_" + name).c_str(),
                                                     EAllocationType::ON_DEVICE,
                                                     shape);
        entry.stats = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                    ("fp8_cache_" + name + "_stats").c_str(),
                                                    EAllocationType::ON_DEVICE,
                                                    {2L});
        entry.weight.Stats = entry.stats.get<float>();
        auto [insert_it, _] = mFP8WeightCache.emplace(name, std::move(entry));
        it = insert_it;
    }

    // Quantize BF16/FP32 weight to FP8 cache once (static weights only).
    if (!it->second.initialized) {
        if (weight.DType == ETensorDType::BF16 || weight.DType == ETensorDType::FP32) {
            const long N = static_cast<long>(weight.nelem());
            if (N > 0) {
                if (debug_cache) {
                    fprintf(stderr, "[DSL TRACE] fp8_cache quant name=%s elems=%ld\n", name.c_str(), N);
                    fflush(stderr);
                }
                abs_max(it->second.weight.abs_max(), weight, N, mRunState.DeviceProp, stream);
                if (debug_cache) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                quantize_with_abs_max(it->second.weight, it->second.weight.scale(),
                                      weight, it->second.weight.abs_max(),
                                      N, mRunState.DeviceProp, stream);
                if (debug_cache) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }
        }
        it->second.initialized = true;
    }

    return &it->second.weight;
}

std::vector<std::byte> GraphExecutor::rng_state() const {
    std::stringstream tmp;
    static_cast<std::ostream&>(tmp) << mRng;
    auto view = tmp.rdbuf()->view();
    std::vector<std::byte> state;
    state.reserve(view.size());
    std::transform(view.begin(), view.end(), std::back_inserter(state),
                   [](char c) { return static_cast<std::byte>(c); });
    return state;
}

void GraphExecutor::set_rng_state(const std::vector<std::byte>& state) {
    std::stringstream tmp;
    tmp.write(reinterpret_cast<const char*>(state.data()), state.size());
    static_cast<std::istream&>(tmp) >> mRng;
}

void GraphExecutor::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    auto& rs = mRunState;

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
    }

    if (micro_step == 0) {
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        if (rs.has_fp8_delayed_scaling()) {
            if (auto* fp8_state = rs.get_fp8_scaling_state()) {
                if (!mFP8ScalingInitialized) {
                    fp8_state->reset(rs.MainStream);
                    mFP8ScalingInitialized = true;
                }
                fp8_state->zero_recorded_amaxes(rs.MainStream);
            }
        }
        rs.reset_moe_stats();
    }

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
        }
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, rs.MainStream));
    }

    execute_forward_graph(B, T, comm, /*full=*/false);

    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
    CUDA_CHECK(cudaEventRecord(rs.ForwardDone, rs.MainStream));
}

float GraphExecutor::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    auto& rs = mRunState;

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = false;
    }

    if (micro_step == 0) {
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        if (rs.has_fp8_delayed_scaling()) {
            if (auto* fp8_state = rs.get_fp8_scaling_state()) {
                if (!mFP8ScalingInitialized) {
                    fp8_state->reset(rs.MainStream);
                    mFP8ScalingInitialized = true;
                }
                fp8_state->zero_recorded_amaxes(rs.MainStream);
            }
        }
        rs.reset_moe_stats();
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
        }
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, rs.MainStream));
    }

    execute_forward_graph(B, T, comm, /*full=*/false);

    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
    CUDA_CHECK(cudaEventRecord(rs.ForwardDone, rs.MainStream));

    fill_zero(rs.Losses, rs.MainStream);
    fill_zero(rs.ValidTokenCount, rs.MainStream);
    fill_zero(rs.CorrectCount, rs.MainStream);

    const std::size_t target_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
    if (targets.Device == -1) {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyDeviceToDevice));
    }

    run_classifier(B, T, comm, /*grad_accum_steps=*/1, micro_step, /*compute_accuracy=*/true);

    reduce_loss(rs, B, T, comm);
    comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    comm.all_reduce_sum_int(rs.CorrectCount.template get<int>(), /*n=*/1, rs.MainStream);

    CUDA_CHECK(cudaMemcpyAsync(rs.NormHost, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(cudaMemcpyAsync(rs.AccuracyHost, rs.CorrectCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(cudaDeviceSynchronize());

    int valid_tokens = *reinterpret_cast<int*>(rs.NormHost);
    int correct_tokens = *reinterpret_cast<int*>(rs.AccuracyHost);
    if (valid_tokens > 0) {
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, comm.world_size()));
        *rs.LossHost /= avg_valid;
        *rs.AccuracyHost = (static_cast<float>(correct_tokens) / static_cast<float>(valid_tokens)) * 100.0f;
    } else {
        *rs.LossHost = 0.0f;
        *rs.AccuracyHost = 0.0f;
    }

    rs.temp_free(rs.non_block_activations().output);

    return *rs.LossHost;
}

void GraphExecutor::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    auto& rs = mRunState;
    auto& grads = mGrads;
    const auto& config = mConfig;
    rs.GradAccumSteps = std::max(1, grad_accum_steps);
    rs.WorldSize = std::max(1, comm.world_size());

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    mLastInputsCpu = inputs;

    // Copy targets to device (side stream).
    {
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.BackwardDone, 0));
        const std::size_t target_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice, rs.side_stream()));
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, rs.side_stream()));
    }

    if (micro_step == 0) {
        fill_zero(rs.Losses, rs.MainStream);
        fill_zero(rs.ValidTokenCount, rs.MainStream);
        grads.start_micro_step(rs.side_stream(), micro_step, grad_accum_steps);
        CUDA_CHECK(cudaEventRecord(rs.side_stream_event(), rs.side_stream()));
    } else {
        grads.start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    // Zero d_ln_final.
    fill_zero(rs.non_block_gradients().d_ln_final, rs.MainStream);
    if (config.NumLayers > 0) {
        fill_zero(rs.simplified_grads(config.NumLayers - 1).d_res_ffn, rs.MainStream);
    }

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
        mLoRAGrads->start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    run_classifier(B, T, comm, grad_accum_steps, micro_step, /*compute_accuracy=*/false);

    const bool last_step = micro_step == grad_accum_steps - 1;
    if (last_step) {
        reduce_loss(rs, B, T, comm);
        comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    }

    execute_backward_graph(B, T, comm, grad_accum_steps, micro_step);

    grads.end_micro_step(rs.MainStream, comm);
    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads) {
        mLoRAGrads->end_micro_step(rs.MainStream, comm);
    }
    CUDA_CHECK(cudaEventRecord(rs.BackwardDone, rs.MainStream));
    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
}

void GraphExecutor::run_classifier(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step, bool compute_accuracy) {
    auto& rs = mRunState;
    const auto& config = mConfig;
    (void)comm;
    if (mOptions.LMHeadChunks != 1) {
        throw std::runtime_error("DSL graph executor: lmhead_chunks > 1 not supported yet");
    }

    const size_t V = config.VocabSize;
    const size_t Vp = config.VocabSize;
    const float d_loss = compute_accuracy ? 1.0f : (1.0f / static_cast<float>(B * T * grad_accum_steps));

    rs.temp_acquire(rs.non_block_activations().output);

    // Ensure targets and gradient zeroing are visible on main stream.
    CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.TransferDone, 0));
    if (!compute_accuracy && micro_step == 0) {
        CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.side_stream_event(), 0));
    }

    Tensor lnf_flat = view_tensor(rs.non_block_activations().ln_final, {B * T, config.HiddenSize});
    Tensor logits = rs.non_block_activations().output;

    // Row-major logits = lnf @ lm_head.T -> map to column-major backend by swapping A/B,
    // swapping M/N, and swapping transpose flags.
    Tensor& lm_head = mWeights.has("lm_head")
        ? mWeights.get("lm_head")
        : mWeights.get("lm_head_weight");

    matmul(logits, lm_head, lnf_flat,
           std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
           static_cast<int>(V), static_cast<int>(B * T), static_cast<int>(config.HiddenSize),
           swap_transpose(EMMTranspose::NT), false, rs.MainStream);

    Tensor tgt = rs.Targets;
    tgt.Sizes[0] = static_cast<long>(B * T);
    tgt.Rank = 1;

    Tensor losses = rs.Losses;
    losses.Sizes[0] = static_cast<long>(B * T);
    losses.Rank = 1;

    if (compute_accuracy) {
        fused_classifier(logits, losses, d_loss, tgt,
                         &rs.ValidTokenCount, &rs.CorrectCount,
                         static_cast<int>(B * T), static_cast<int>(V), static_cast<int>(Vp), true, rs.MainStream);
    } else {
        fused_classifier(logits, losses, d_loss, tgt,
                         &rs.ValidTokenCount,
                         static_cast<int>(B * T), static_cast<int>(V), static_cast<int>(Vp), true, rs.MainStream);
    }
}

void GraphExecutor::execute_forward_graph(long B, long T, NCCLCommunicator& comm, bool full) {
    if (!mForward) {
        throw std::runtime_error("DSL graph executor: missing forward graph");
    }

    auto& rs = mRunState;
    auto& weights = mWeights;
    const auto& config = mConfig;
    const bool trace_ops = env_enabled("SUROGATE_DEBUG_DSL_TRACE");
    (void)comm;

    ExecState st{
        .rs = rs,
        .weights = weights,
        .grads = mGrads,
        .config = config,
        .B = B,
        .T = T,
        .shape_env = make_shape_env(mModule, B, T),
        .view_sources = &mViewSources,
        .view_sources_rev = &mViewSourcesReverse,
    };
    augment_shape_env(st.shape_env, mModule.config);

    // Bind known inputs.
    st.tensors.emplace("token_ids", rs.Inputs);
    st.tensors.emplace("position_ids", rs.PositionIDs);
    st.tensors.emplace("x0", rs.non_block_activations().encoded);

    const bool use_graphs = mGraphsEnabled;
    if (use_graphs && (mGraphB != B || mGraphT != T)) {
        reset_cuda_graphs();
        mGraphB = B;
        mGraphT = T;
    }
    const bool capturing = use_graphs && mForwardGraph == nullptr;
    if (!use_graphs || capturing) {
        mSaved.clear();
    }

    std::vector<char> required;
    if (!full) {
        std::vector<std::string> needed = mSaveList;
        for (const auto& kv : mForward->outputs) {
            needed.push_back(kv.first);
        }
        required = compute_required_ops(*mForward, needed);
    }

    if (capturing) {
        prime_fp8_weight_cache(required);
    }

    auto run_ops = [&]() {
    std::unordered_map<int, DeviceMemoryStack::Checkpoint> layer_checkpoints;
    std::unordered_map<int, std::size_t> layer_temp_marks;
    auto maybe_start_layer = [&](const Operation& op) {
        for (const auto& out_name : op.outputs) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(out_name, layer_idx, field)) {
                if (layer_checkpoints.find(layer_idx) == layer_checkpoints.end()) {
                    layer_checkpoints[layer_idx] = rs.Stack.checkpoint();
                    layer_temp_marks[layer_idx] = st.temps.size();
                }
                break;
            }
        }
    };
    auto finish_layer = [&](int layer_idx) {
        auto cp_it = layer_checkpoints.find(layer_idx);
        if (cp_it == layer_checkpoints.end()) {
            return;
        }
        rs.Stack.restore(cp_it->second);
        auto mark_it = layer_temp_marks.find(layer_idx);
        if (mark_it != layer_temp_marks.end()) {
            if (st.temps.size() > mark_it->second) {
                st.temps.resize(mark_it->second);
            }
        }
        if (rs.ffn_temps_on_stack()) {
            auto& acts = rs.simplified_acts(layer_idx);
            acts.mlp_up.Data = nullptr;
            acts.swiglu.Data = nullptr;
        }
        rs.scratch().cudnn_workspace.Data = nullptr;
        layer_checkpoints.erase(layer_idx);
        layer_temp_marks.erase(layer_idx);
    };

    for (std::size_t idx = 0; idx < mForward->operations.size(); ++idx) {
        if (!full && !required[idx]) {
            continue;
        }
        const auto& op = mForward->operations[idx];
        const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
        if (trace_ops) {
            fprintf(stderr, "[DSL TRACE] fwd op %zu/%zu id=%s type=%s\n",
                    idx, mForward->operations.size(), op.id.c_str(), op_type.c_str());
            fflush(stderr);
        }
        maybe_start_layer(op);

        if (op_type == "embedding") {
            Tensor& token_ids = get_tensor(st, op.inputs.at(0), mSaved);
            const std::string& emb_name = op.inputs.size() > 1 ? op.inputs.at(1) : "embedding";
            Tensor& emb = resolve_param_tensor(st, emb_name);
            const std::string& out_name = op.outputs.at(0);
            if (st.tensors.find(out_name) == st.tensors.end()) {
                st.tensors.emplace(out_name, rs.non_block_activations().encoded);
            }
            Tensor& out = st.tensors.at(out_name);
            encoder_forward(out, token_ids, emb, std::nullopt,
                            static_cast<int>(B), static_cast<int>(T),
                            config.HiddenSize, config.VocabSize, rs.MainStream);
            continue;
        }

        if (op_type == "zeros") {
            auto* shape_attr = find_attr(op.attrs, "shape");
            if (!shape_attr) {
                throw std::runtime_error("DSL graph executor: zeros missing shape attr");
            }
            auto shape = resolve_attr_shape(*shape_attr, st.shape_env);
            ETensorDType dtype = ETensorDType::BF16;
            if (auto* dtype_attr = find_attr(op.attrs, "dtype")) {
                if (auto s = attr_string(*dtype_attr)) {
                    dtype = dtype_from_str(*s);
                }
            }
            Tensor& out = ensure_tensor(st, op.outputs.at(0), dtype, shape);
            fill_zero(out, rs.MainStream);
            st.zero_tensors.insert(op.outputs.at(0));
            continue;
        }

        if (op_type == "fused_residual_rmsnorm") {
            Tensor& residual_in = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& input = get_tensor(st, op.inputs.at(1), mSaved);
            const std::string& weight_name = op.inputs.at(2);
            Tensor& weight = resolve_param_tensor(st, weight_name);

            const bool is_final = (weight_name == "final_norm" || weight_name == "final_norm_weight" || weight_name == "norm");

            Tensor* residual_out_ptr = nullptr;
            Tensor* y_ptr = nullptr;
            Tensor* rstd_ptr = nullptr;
            if (is_final) {
                residual_out_ptr = &rs.get_final_residual();
                y_ptr = &rs.non_block_activations().ln_final;
                rstd_ptr = &rs.non_block_activations().ln_final_rstd;
            } else {
                residual_out_ptr = &ensure_tensor(st, op.outputs.at(0), input.DType, {B, T, config.HiddenSize});
                y_ptr = &ensure_tensor(st, op.outputs.at(1), input.DType, {B, T, config.HiddenSize});
                rstd_ptr = &ensure_tensor(st, op.outputs.at(2), ETensorDType::FP32, {B, T});
            }
            Tensor& residual_out = *residual_out_ptr;
            Tensor& y = *y_ptr;
            Tensor& rstd = *rstd_ptr;

            double eps = config.RmsNormEps;
            if (auto* eps_attr = find_attr(op.attrs, "eps")) {
                if (auto v = attr_double(*eps_attr)) {
                    eps = *v;
                }
            }
            fused_residual_rmsnorm_forward(residual_out, y, rstd, residual_in, input, weight, nullptr,
                                           static_cast<float>(eps), static_cast<int>(B * T),
                                           config.HiddenSize, rs.MainStream);

            st.tensors.emplace(op.outputs.at(0), residual_out);
            st.tensors.emplace(op.outputs.at(1), y);
            st.tensors.emplace(op.outputs.at(2), rstd);
            continue;
        }

        if (op_type == "view") {
            Tensor& src = get_tensor(st, op.inputs.at(0), mSaved);
            auto shape = resolve_view_shape(op, st.shape_env, st, mSaved);
            Tensor view = view_tensor(src, shape);
            auto it = st.tensors.find(op.outputs.at(0));
            if (it != st.tensors.end()) {
                it->second = view;
            } else {
                st.tensors.emplace(op.outputs.at(0), view);
            }
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(op.outputs.at(0), layer_idx, field) && field == "mlp_down") {
                finish_layer(layer_idx);
            }
            continue;
        }

        if (op_type == "add") {
            Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
            if (a.DType != b.DType || a.nelem() != b.nelem()) {
                throw std::runtime_error("DSL graph executor: add op shape or dtype mismatch");
            }
            std::vector<long> shape(a.Sizes.begin(), a.Sizes.begin() + a.Rank);
            Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, shape);
            vector_add_sr(out, a, b, 1.0f, static_cast<long>(a.nelem()), 0, rs.MainStream);
            continue;
        }

        if (op_type == "matmul" || op_type == "matmul_bias") {
            Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
            if (a.Rank != 2 || b.Rank != 2) {
                throw std::runtime_error(
                    "DSL graph executor: matmul expects rank-2 tensors (op=" + op.id +
                    ", a=" + op.inputs.at(0) + " rank=" + std::to_string(a.Rank) +
                    " shape=" + tensor_shape_str(a) +
                    ", b=" + op.inputs.at(1) + " rank=" + std::to_string(b.Rank) +
                    " shape=" + tensor_shape_str(b) + ")"
                );
            }
            EMMTranspose mode = parse_transpose(op.attrs);
            int M = 0, N = 0, K = 0;
            matmul_dims(a, b, mode, M, N, K);
            std::vector<long> shape{M, N};
            Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, shape);
            std::optional<Tensor> bias;
            if (op_type == "matmul_bias" && op.inputs.size() > 2 && !op.inputs.at(2).empty()) {
                bias = get_tensor(st, op.inputs.at(2), mSaved);
            }
            bool used_recipe = false;
            if (mOptions.TrainingRecipe && mode == EMMTranspose::NT && a.Sizes[0] == B * T) {
                const recipes::Recipe& recipe = *mOptions.TrainingRecipe;
                int layer_idx = -1;
                auto op_kind = matmul_op_from_weight(op.inputs.at(1), layer_idx);
                const bool allow_quant = op_kind.has_value() && allow_quant_layer(mOptions, config, layer_idx);
                const modules::MatmulOp matmul_op = op_kind.value_or(modules::MatmulOp::LMHead);

                if (env_enabled("SUROGATE_DEBUG_DSL_MATMUL")) {
                    fprintf(stderr,
                            "[DSL TRACE] fwd matmul op=%s layer=%d allow_quant=%d a=%s b=%s out=%s M=%d N=%d K=%d\n",
                            op.inputs.at(1).c_str(), layer_idx, allow_quant ? 1 : 0,
                            dtype_to_str(a.DType), dtype_to_str(b.DType), dtype_to_str(out.DType), M, N, K);
                    fflush(stderr);
                }

                modules::MatmulContext ctx;
                ctx.out = &out;
                ctx.inp = &a;
                ctx.weight = &b;
                ctx.bias = bias ? &*bias : nullptr;
                ctx.B = static_cast<int>(B);
                ctx.T = static_cast<int>(T);
                ctx.C_in = K;
                ctx.C_out = N;
                ctx.run_state = &rs;
                ctx.stream = rs.MainStream;
                ctx.layer_idx = layer_idx;
                ctx.op = matmul_op;
                ctx.allow_fp8 = allow_quant;
                ctx.allow_fp4 = allow_quant;
                if (allow_quant) {
                    ctx.inp_quant = fp8_forward_buffer(rs, matmul_op);
                    ctx.cached_weight = get_fp8_cached_weight(op.inputs.at(1), b, rs.MainStream);
                    ctx.delayed_quantizer_idx = fp8_quantizer_index(rs, matmul_op, layer_idx);
                    if (env_enabled("SUROGATE_DEBUG_DSL_MATMUL")) {
                        fprintf(stderr,
                                "[DSL TRACE] fwd matmul quant inp_q=%p weight_q=%p dq_idx=%d\n",
                                ctx.inp_quant ? ctx.inp_quant->Data : nullptr,
                                ctx.cached_weight ? ctx.cached_weight->Data : nullptr,
                                ctx.delayed_quantizer_idx);
                        fflush(stderr);
                    }
                }

                recipe.forward_matmul(ctx);
                used_recipe = true;
            }
            if (!used_recipe) {
                // DSL matmul uses row-major semantics; map to column-major backend by swapping A/B,
                // swapping M/N, and swapping transpose flags.
                EMMTranspose mode_col = swap_transpose(mode);
                matmul(out, b, a, bias, nullptr, nullptr,
                       rs.CublasLtHandle, rs.CuBlasWorkspace,
                       N, M, K, mode_col, false, rs.MainStream);
            }

            if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAWeights && mLoRARunState) {
                int layer_idx = -1;
                std::string field;
                if (parse_block_param(op.inputs.at(1), layer_idx, field)) {
                    auto& acts = rs.simplified_acts(layer_idx);
                    auto& lora_rs = *mLoRARunState;
                    auto& lora_block = mLoRAWeights->get_block(layer_idx, rs.MainStream);
                    const int Bv = static_cast<int>(B);
                    const int Tv = static_cast<int>(T);
                    const int C = static_cast<int>(config.HiddenSize);
                    const int D = static_cast<int>(config.IntermediateSize);
                    const int Hq = static_cast<int>(config.NumQueryHeads);
                    const int Hkv = static_cast<int>(config.NumKeyValHeads);
                    const int Hs = static_cast<int>(config.head_size());
                    const int rank = mLoRAConfig->rank;
                    const int BT = Bv * Tv;
                    const float scaling = mLoRAConfig->scaling();
                    const float dropout = mLoRAConfig->dropout;
                    const bool training = lora_rs.is_training;

                    auto dropout_seed = [&](int proj_type) -> unsigned int {
                        return lora_rs.dropout_base_seed
                               + static_cast<unsigned int>(layer_idx) * 1000000u
                               + static_cast<unsigned int>(proj_type) * 100000u
                               + static_cast<unsigned int>(lora_rs.micro_step) * 10000u;
                    };

                    if (field == "qkv_weight") {
                        if (lora_block.attention.q.has_value()) {
                            modules::detail::apply_lora_contribution(
                                acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                lora_rs.intermediate, lora_rs.slice,
                                scaling, dropout, dropout_seed(0), training,
                                BT, C, Hq * Hs, rank,
                                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                        }
                        if (lora_block.attention.k.has_value()) {
                            modules::detail::apply_lora_contribution(
                                acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                lora_rs.intermediate, lora_rs.slice,
                                scaling, dropout, dropout_seed(1), training,
                                BT, C, Hkv * Hs, rank,
                                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                        }
                        if (lora_block.attention.v.has_value()) {
                            modules::detail::apply_lora_contribution(
                                acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                lora_rs.intermediate, lora_rs.slice,
                                scaling, dropout, dropout_seed(2), training,
                                BT, C, Hkv * Hs, rank,
                                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                        }
                    } else if (field == "out_weight") {
                        if (lora_block.attention.o.has_value()) {
                            modules::detail::apply_lora_contribution(
                                acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                lora_rs.intermediate, lora_rs.slice,
                                scaling, dropout, dropout_seed(3), training,
                                BT, Hq * Hs, C, rank,
                                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                        }
                    } else if (field == "mlp_up_weight") {
                        if (lora_block.mlp.up.has_value()) {
                            modules::detail::apply_lora_contribution(
                                acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                lora_rs.intermediate, lora_rs.slice,
                                scaling, dropout, dropout_seed(4), training,
                                BT, C, D, rank,
                                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                        }
                        if (lora_block.mlp.gate.has_value()) {
                            modules::detail::apply_lora_contribution(
                                acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                lora_rs.intermediate, lora_rs.slice,
                                scaling, dropout, dropout_seed(5), training,
                                BT, C, D, rank,
                                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                        }
                    } else if (field == "mlp_down_weight") {
                        if (lora_block.mlp.down.has_value()) {
                            modules::detail::apply_lora_contribution(
                                acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                lora_rs.intermediate, lora_rs.slice,
                                scaling, dropout, dropout_seed(6), training,
                                BT, D, C, rank,
                                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                        }
                    }
                }
            }
            continue;
        }

        if (op_type == "bias_add") {
            Tensor& x = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& bias = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& out = ensure_tensor(st, op.outputs.at(0), x.DType,
                                        {x.Sizes[0], x.Sizes[1], x.Sizes[2]});
            const std::size_t bytes = static_cast<std::size_t>(x.nelem()) * get_dtype_size(x.DType);
            CUDA_CHECK(cudaMemcpyAsync(out.Data, x.Data, bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
            add_bias_tensor(out, bias, static_cast<int>(x.Sizes[0]), static_cast<int>(x.Sizes[1]),
                            static_cast<int>(x.Sizes[2]), rs.MainStream);
            continue;
        }

        if (op_type == "swiglu") {
            Tensor& inp = get_tensor(st, op.inputs.at(0), mSaved);
            // Input shape (B, T, 2*D) -> output (B, T, D)
            const long D = inp.Sizes[2] / 2;
            Tensor& out = ensure_tensor(st, op.outputs.at(0), inp.DType, {inp.Sizes[0], inp.Sizes[1], D});
            swiglu_forward(out, inp, nullptr, static_cast<int>(inp.Sizes[0]),
                           static_cast<int>(inp.Sizes[1]), static_cast<int>(D), rs.MainStream);
            continue;
        }

        if (op_type == "qkv_qk_norm_rope") {
            Tensor& qkv = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& q_norm = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& k_norm = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor& freqs = get_tensor(st, op.inputs.at(3), mSaved);
            Tensor& pos_ids = get_tensor(st, op.inputs.at(4), mSaved);

            const int Hq = static_cast<int>(config.NumQueryHeads);
            const int Hkv = static_cast<int>(config.NumKeyValHeads);
            const int Hs = static_cast<int>(config.head_size());
            const int qkv_channels = Hs * (Hq + 2 * Hkv);

            Tensor& q_rstd = ensure_tensor(st, op.outputs.at(1), ETensorDType::FP32, {B, T, Hq});
            Tensor& k_rstd = ensure_tensor(st, op.outputs.at(2), ETensorDType::FP32, {B, T, Hkv});

            double eps = config.RmsNormEps;
            if (auto* eps_attr = find_attr(op.attrs, "eps")) {
                if (auto v = attr_double(*eps_attr)) {
                    eps = *v;
                }
            }

            // Match modular path: fused QK norm + RoPE when supported, otherwise fall back to separate ops.
            Tensor qkv_view = (qkv.Rank == 4)
                ? view_tensor(qkv, {B, T, qkv_channels})
                : qkv;
            int rotary_dim = Hs;
            if (auto* rd_attr = find_attr(op.attrs, "rotary_dim")) {
                if (auto v = attr_int(*rd_attr)) {
                    rotary_dim = static_cast<int>(*v);
                } else if (auto s = attr_string(*rd_attr)) {
                    rotary_dim = static_cast<int>(resolve_dim(Dim::symbolic(*s), st.shape_env));
                }
            }
            const bool rope_fusable = (rotary_dim > 0)
                && ((Hs % 2) == 0)
                && (((Hs / 2) % 32) == 0)
                && (freqs.Rank >= 2)
                && (freqs.Sizes[1] >= Hs)
                && (qkv_view.Rank == 3);
            if (rope_fusable) {
                qkv_qk_norm_rope_forward(qkv_view, q_rstd, k_rstd, q_norm, k_norm,
                                         freqs, reinterpret_cast<int*>(pos_ids.Data),
                                         static_cast<float>(eps),
                                         static_cast<int>(B), static_cast<int>(T),
                                         Hq, Hkv, Hs, rs.MainStream);
            } else {
                const int q_rows = Hq * Hs;
                qkv_head_rmsnorm_forward(qkv_view, q_rstd, q_norm,
                                         static_cast<float>(eps),
                                         static_cast<int>(B), static_cast<int>(T),
                                         qkv_channels, Hq, Hs, 0, rs.MainStream);
                qkv_head_rmsnorm_forward(qkv_view, k_rstd, k_norm,
                                         static_cast<float>(eps),
                                         static_cast<int>(B), static_cast<int>(T),
                                         qkv_channels, Hkv, Hs, q_rows, rs.MainStream);
                rope_forward(qkv, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                             static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
            }
            st.tensors.emplace(op.outputs.at(0), qkv);
            st.tensors.emplace(op.outputs.at(1), q_rstd);
            st.tensors.emplace(op.outputs.at(2), k_rstd);
            continue;
        }

        if (op_type == "rope") {
            Tensor& qkv = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& freqs = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& pos_ids = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor& out = ensure_tensor(st, op.outputs.at(0), qkv.DType, {qkv.Sizes[0], qkv.Sizes[1], qkv.Sizes[2], qkv.Sizes[3]});

            int Hq = static_cast<int>(config.NumQueryHeads);
            int Hkv = static_cast<int>(config.NumKeyValHeads);
            int Hs = static_cast<int>(config.head_size());
            int rotary_dim = Hs;
            if (auto* rd_attr = find_attr(op.attrs, "rotary_dim")) {
                if (auto v = attr_int(*rd_attr)) {
                    rotary_dim = static_cast<int>(*v);
                } else if (auto s = attr_string(*rd_attr)) {
                    rotary_dim = static_cast<int>(resolve_dim(Dim::symbolic(*s), st.shape_env));
                }
            }

            rope_forward(out, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                         static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
            continue;
        }

        if (op_type == "flash_attention" || op_type == "flash_attention_qkv") {
            Tensor& qkv = get_tensor(st, op.inputs.at(0), mSaved);
            const int Hq = static_cast<int>(config.NumQueryHeads);
            const int Hkv = static_cast<int>(config.NumKeyValHeads);
            const int Hs = static_cast<int>(config.head_size());

            Tensor& out = ensure_tensor(st, op.outputs.at(0), qkv.DType, {B, T, Hq, Hs});
            Tensor& lse = ensure_tensor(st, op.outputs.at(1), ETensorDType::FP32, {B, Hq, T});

            // Ensure workspace allocated
            if (!rs.scratch().cudnn_workspace.Data) {
                rs.temp_acquire(rs.scratch().cudnn_workspace);
                st.temps.push_back(rs.scratch().cudnn_workspace);
            }

            attention_forward_cudnn(out, lse, qkv, rs.scratch().cudnn_workspace,
                                    rs.CudnnHandle, static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs,
                                    rs.MainStream);
            continue;
        }

        throw std::runtime_error("DSL graph executor: unsupported forward op " + op.name);
    }

    // Save requested tensors for backward (uses auto-computed list if autodiff was used).
    for (const auto& name : mSaveList) {
        auto it = st.tensors.find(name);
        if (it != st.tensors.end()) {
            mSaved[name] = it->second;
        } else if (name == "token_ids") {
            mSaved[name] = rs.Inputs;
        } else {
            throw std::runtime_error("DSL graph executor: missing save tensor " + name);
        }
    }

    free_temps(st);
    };

    trace_or_execute_cuda_graph_with_stack(run_ops, rs.MainStream, mForwardGraph, use_graphs, rs.Stack, mForwardCheckpoint);
}

void GraphExecutor::execute_backward_graph(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    if (!mBackward) {
        throw std::runtime_error("DSL graph executor: missing backward graph");
    }

    auto& rs = mRunState;
    auto& weights = mWeights;
    auto& grads = mGrads;
    const auto& config = mConfig;
    const bool trace_ops = env_enabled("SUROGATE_DEBUG_DSL_TRACE");
    (void)comm;

    ExecState st{
        .rs = rs,
        .weights = weights,
        .grads = grads,
        .config = config,
        .B = B,
        .T = T,
        .shape_env = make_shape_env(mModule, B, T),
        .view_sources = &mViewSources,
        .view_sources_rev = &mViewSourcesReverse,
    };
    augment_shape_env(st.shape_env, mModule.config);

    const bool recompute_block = mOptions.RecomputeBlock;
    const bool recompute_att = mOptions.RecomputeAtt || recompute_block;
    const bool recompute_qkv = mOptions.RecomputeQKV || recompute_att;
    const bool recompute_ffn = mOptions.RecomputeFFN || recompute_block;
    const bool recompute_swiglu = mOptions.RecomputeSwiGLu || recompute_ffn;
    const bool recompute_rmsnorm = mOptions.RecomputeRMSNorm || recompute_block;
    const bool recompute_ln1 = recompute_rmsnorm || recompute_att || recompute_block;
    const bool recompute_ln2 = recompute_rmsnorm || recompute_ffn || recompute_block;
    const bool recompute_any = recompute_ln1 || recompute_qkv || recompute_att || recompute_ln2 || recompute_ffn || recompute_swiglu;

    std::vector<char> recomputed;
    if (recompute_any && !recompute_block && config.NumLayers > 0) {
        recomputed.assign(static_cast<size_t>(config.NumLayers), 0);
    }
    int last_recompute_layer = -1;

    auto recompute_layer = [&](int layer_idx) {
        if (!recompute_any) return;
        if (layer_idx < 0 || layer_idx >= config.NumLayers) return;
        if (recompute_block) {
            if (layer_idx == last_recompute_layer) {
                return;
            }
            last_recompute_layer = layer_idx;
        } else {
            if (recomputed.empty() || recomputed[layer_idx]) return;
            recomputed[layer_idx] = 1;
        }

        const int Bv = static_cast<int>(B);
        const int Tv = static_cast<int>(T);
        const int C = static_cast<int>(config.HiddenSize);
        const int D = static_cast<int>(config.IntermediateSize);
        const int Hq = static_cast<int>(config.NumQueryHeads);
        const int Hkv = static_cast<int>(config.NumKeyValHeads);
        const int Hs = static_cast<int>(config.head_size());
        const int qkv_channels = Hs * (Hq + 2 * Hkv);
        const int att_dim = Hq * Hs;
        const int MUp = 2 * D;

        auto& acts = rs.simplified_acts(layer_idx);
        auto ensure_act = [&](Tensor& t) {
            if (!t.Data) {
                rs.temp_acquire(t);
                st.temps.push_back(t);
            }
        };

        Tensor& res_ffn = rs.get_residual(layer_idx, rs.MainStream);
        if (recompute_ln1) {
            ensure_act(acts.ln1);
            ensure_act(acts.ln1_rstd);
            Tensor& ln1_weight = resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].ln1_weight");
            if (recompute_block) {
                // res_ffn already holds the residual stream for this layer in the DSL path.
                rmsnorm_forward(acts.ln1, acts.ln1_rstd, res_ffn, ln1_weight, nullptr,
                                config.RmsNormEps, Bv, Tv, C, rs.MainStream);
            } else {
                Tensor residual_in;
                Tensor x_in;
                if (layer_idx == 0) {
                    Tensor zero = rs.temp_alloc(acts.ln1.DType, {B, T, C});
                    st.temps.push_back(zero);
                    fill_zero(zero, rs.MainStream);
                    residual_in = zero;
                    x_in = rs.non_block_activations().encoded;
                } else {
                    residual_in = rs.simplified_acts(layer_idx - 1).residual_att;
                    x_in = rs.simplified_acts(layer_idx - 1).mlp_down;
                }
                fused_residual_rmsnorm_forward(
                    res_ffn, acts.ln1, acts.ln1_rstd,
                    residual_in, x_in, ln1_weight, nullptr,
                    config.RmsNormEps, static_cast<int>(B * T), C, rs.MainStream);
            }
        }

        auto dsl_matmul = [&](Tensor& out, const Tensor& a_in, const Tensor& b_in, EMMTranspose mode,
                              std::optional<Tensor> bias,
                              std::string_view weight_name,
                              std::optional<modules::MatmulOp> op_kind) {
            int M = 0, N = 0, K = 0;
            matmul_dims(a_in, b_in, mode, M, N, K);
            bool used_recipe = false;
            if (mOptions.TrainingRecipe && mode == EMMTranspose::NT && a_in.Sizes[0] == B * T) {
                const recipes::Recipe& recipe = *mOptions.TrainingRecipe;
                const int layer_idx_local = layer_idx;
                const bool allow_quant = op_kind.has_value() && allow_quant_layer(mOptions, config, layer_idx_local);
                const modules::MatmulOp matmul_op = op_kind.value_or(modules::MatmulOp::LMHead);

                modules::MatmulContext ctx;
                ctx.out = &out;
                ctx.inp = const_cast<Tensor*>(&a_in);
                ctx.weight = const_cast<Tensor*>(&b_in);
                ctx.bias = bias ? &*bias : nullptr;
                ctx.B = static_cast<int>(B);
                ctx.T = static_cast<int>(T);
                ctx.C_in = K;
                ctx.C_out = N;
                ctx.run_state = &rs;
                ctx.stream = rs.MainStream;
                ctx.layer_idx = layer_idx_local;
                ctx.op = matmul_op;
                ctx.allow_fp8 = allow_quant;
                ctx.allow_fp4 = allow_quant;
                if (allow_quant) {
                    ctx.inp_quant = fp8_forward_buffer(rs, matmul_op);
                    ctx.cached_weight = get_fp8_cached_weight(std::string(weight_name), *ctx.weight, rs.MainStream);
                    ctx.delayed_quantizer_idx = -1;  // Recompute uses JIT scaling
                }
                recipe.forward_matmul(ctx);
                used_recipe = true;
            }
            if (!used_recipe) {
                EMMTranspose mode_col = swap_transpose(mode);
                matmul(out, b_in, a_in, bias, nullptr, nullptr,
                       rs.CublasLtHandle, rs.CuBlasWorkspace,
                       N, M, K, mode_col, false, rs.MainStream);
            }
        };

        if (recompute_qkv) {
            ensure_act(acts.qkv);
            Tensor& qkv_weight = resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].qkv_weight");
            Tensor ln1_flat = view_tensor(acts.ln1, {B * T, C});
            Tensor qkv_flat = view_tensor(acts.qkv, {B * T, qkv_channels});
            std::string bias_name = "blocks[" + std::to_string(layer_idx) + "].qkv_bias";
            std::optional<Tensor> qkv_bias;
            if (mWeights.has(bias_name)) {
                qkv_bias = resolve_param_tensor(st, bias_name);
            }
            dsl_matmul(qkv_flat, ln1_flat, qkv_weight, EMMTranspose::NT, qkv_bias,
                       "blocks[" + std::to_string(layer_idx) + "].qkv_weight",
                       modules::MatmulOp::QKV);

            if (config.UseQKNorm) {
                ensure_act(acts.q_rstd);
                ensure_act(acts.k_rstd);
                Tensor qkv_view = view_tensor(acts.qkv, {B, T, qkv_channels});
                const bool rope_fusable = ((Hs % 2) == 0)
                    && (((Hs / 2) % 32) == 0)
                    && (rs.non_block_activations().freq_cis.Rank >= 2)
                    && (rs.non_block_activations().freq_cis.Sizes[1] >= Hs);
                if (rope_fusable) {
                    qkv_qk_norm_rope_forward(qkv_view, acts.q_rstd, acts.k_rstd,
                                             resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].q_norm_weight"),
                                             resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].k_norm_weight"),
                                             rs.non_block_activations().freq_cis,
                                             reinterpret_cast<int*>(rs.PositionIDs.Data),
                                             static_cast<float>(config.RmsNormEps),
                                             Bv, Tv, Hq, Hkv, Hs, rs.MainStream);
                } else {
                    qkv_head_rmsnorm_forward(qkv_view, acts.q_rstd,
                                             resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].q_norm_weight"),
                                             static_cast<float>(config.RmsNormEps),
                                             Bv, Tv, qkv_channels, Hq, Hs, 0, rs.MainStream);
                    qkv_head_rmsnorm_forward(qkv_view, acts.k_rstd,
                                             resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].k_norm_weight"),
                                             static_cast<float>(config.RmsNormEps),
                                             Bv, Tv, qkv_channels, Hkv, Hs, Hq * Hs, rs.MainStream);
                    Tensor qkv_rope = (acts.qkv.Rank == 4)
                        ? acts.qkv
                        : view_tensor(acts.qkv, {B, T, Hq + 2 * Hkv, Hs});
                    rope_forward(qkv_rope, qkv_rope, rs.non_block_activations().freq_cis,
                                 reinterpret_cast<int*>(rs.PositionIDs.Data), nullptr,
                                 Bv, Tv, Hq, Hkv, Hs, Hs, rs.MainStream);
                }
            } else {
                Tensor qkv_rope = (acts.qkv.Rank == 4)
                    ? acts.qkv
                    : view_tensor(acts.qkv, {B, T, Hq + 2 * Hkv, Hs});
                rope_forward(qkv_rope, qkv_rope, rs.non_block_activations().freq_cis,
                             reinterpret_cast<int*>(rs.PositionIDs.Data), nullptr,
                             Bv, Tv, Hq, Hkv, Hs, Hs, rs.MainStream);
            }
        }

        if (recompute_att) {
            ensure_act(acts.att);
            ensure_act(acts.lse);
            Tensor qkv_view = (acts.qkv.Rank == 4)
                ? acts.qkv
                : view_tensor(acts.qkv, {B, T, Hq + 2 * Hkv, Hs});
            if (!rs.scratch().cudnn_workspace.Data) {
                rs.temp_acquire(rs.scratch().cudnn_workspace);
                st.temps.push_back(rs.scratch().cudnn_workspace);
            }
            Tensor att_out = view_tensor(acts.att, {B, T, Hq, Hs});
            Tensor lse_view = view_tensor(acts.lse, {B, Hq, T});
            attention_forward_cudnn(att_out, lse_view, qkv_view, rs.scratch().cudnn_workspace,
                                    rs.CudnnHandle, Bv, Tv, Hq, Hkv, Hs, rs.MainStream);

            if (recompute_block || recompute_ln2) {
                ensure_act(acts.att_out);
                Tensor& out_weight = resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].out_weight");
                Tensor att_flat = view_tensor(acts.att, {B * T, att_dim});
                Tensor att_out_flat = view_tensor(acts.att_out, {B * T, C});
                dsl_matmul(att_out_flat, att_flat, out_weight, EMMTranspose::NT, std::nullopt,
                           "blocks[" + std::to_string(layer_idx) + "].out_weight",
                           modules::MatmulOp::AttnOut);
            }
        }

        if (recompute_ln2) {
            ensure_act(acts.ln2);
            ensure_act(acts.ln2_rstd);
            Tensor& ln2_weight = resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].ln2_weight");
            fused_residual_rmsnorm_forward(
                acts.residual_att, acts.ln2, acts.ln2_rstd,
                res_ffn, acts.att_out, ln2_weight, nullptr,
                config.RmsNormEps, static_cast<int>(B * T), C, rs.MainStream);
        }

        if (recompute_ffn) {
            ensure_act(acts.mlp_up);
            Tensor& mlp_up_weight = resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].mlp_up_weight");
            Tensor ln2_flat = view_tensor(acts.ln2, {B * T, C});
            Tensor mlp_up_flat = view_tensor(acts.mlp_up, {B * T, MUp});
            dsl_matmul(mlp_up_flat, ln2_flat, mlp_up_weight, EMMTranspose::NT, std::nullopt,
                       "blocks[" + std::to_string(layer_idx) + "].mlp_up_weight",
                       modules::MatmulOp::MLPUp);
        }

        if (recompute_swiglu) {
            ensure_act(acts.swiglu);
            swiglu_forward(acts.swiglu, acts.mlp_up, nullptr,
                           Bv, Tv, D, rs.MainStream);
        }

        // NOTE: mlp_down output is not required for backward (only swiglu + weights are needed),
        // so skip recomputing it to save a large matmul in recompute_block mode.
    };

    // Bind forward inputs needed by backward ops (e.g., rope uses position_ids).
    st.tensors.emplace("token_ids", rs.Inputs);
    st.tensors.emplace("position_ids", rs.PositionIDs);

    // Bind d_logits (produced by classifier) as [B, T, V] view.
    Tensor logits_view = view_tensor(rs.non_block_activations().output, {B, T, config.VocabSize});
    st.tensors.emplace("d_logits", logits_view);

    // Bind views for d_xF_flat / d_xF to write into d_ln_final.
    Tensor d_ln_final_flat = view_tensor(rs.non_block_gradients().d_ln_final, {B * T, config.HiddenSize});
    st.tensors.emplace("d_xF_flat", d_ln_final_flat);
    st.tensors.emplace("d_xF", rs.non_block_gradients().d_ln_final);

    // Bind embedding output gradients to the non-block d_embeddings buffer.
    for (const auto& name : mEmbeddingOutputs) {
        st.tensors.emplace("d_" + name, rs.non_block_gradients().d_embeddings);
    }

    // Bind gradient outputs for parameters.
    std::unordered_set<std::string> accumulate_tensors;

    auto bind_param_grad = [&](const std::string& param_name) {
        if (param_name.find("rope_freqs") != std::string::npos) {
            return;
        }
        bool accumulate = false;
        Tensor* grad_tensor = grads.get_param_grad(param_name, accumulate);
        if (!grad_tensor) {
            return;
        }
        std::string grad_name = "d_" + param_name;
        st.tensors.emplace(grad_name, *grad_tensor);
        if (accumulate) {
            accumulate_tensors.insert(grad_name);
        }
    };

    for (const auto& kv : mForward->params) {
        bind_param_grad(kv.first);
    }

    std::unordered_map<int, DeviceMemoryStack::Checkpoint> layer_checkpoints;
    std::unordered_map<int, std::size_t> layer_temp_marks;
    auto extract_layer = [&](const std::string& name, int& layer_idx) -> bool {
        std::string_view view{name};
        if (starts_with(view, "d_")) {
            view = view.substr(2);
        }
        if (starts_with(view, kSavedPrefix)) {
            view = view.substr(kSavedPrefix.size());
        }
        std::string field;
        return parse_block_param(view, layer_idx, field);
    };
    auto maybe_start_layer = [&](const Operation& op) {
        for (const auto& name : op.inputs) {
            int layer_idx = -1;
            if (extract_layer(name, layer_idx)) {
                if (layer_checkpoints.find(layer_idx) == layer_checkpoints.end()) {
                    layer_checkpoints[layer_idx] = rs.Stack.checkpoint();
                    layer_temp_marks[layer_idx] = st.temps.size();
                }
                return;
            }
        }
        for (const auto& name : op.outputs) {
            int layer_idx = -1;
            if (extract_layer(name, layer_idx)) {
                if (layer_checkpoints.find(layer_idx) == layer_checkpoints.end()) {
                    layer_checkpoints[layer_idx] = rs.Stack.checkpoint();
                    layer_temp_marks[layer_idx] = st.temps.size();
                }
                return;
            }
        }
    };
    auto finish_layer = [&](int layer_idx) {
        auto cp_it = layer_checkpoints.find(layer_idx);
        if (cp_it == layer_checkpoints.end()) {
            return;
        }
        rs.Stack.restore(cp_it->second);
        auto mark_it = layer_temp_marks.find(layer_idx);
        if (mark_it != layer_temp_marks.end()) {
            if (st.temps.size() > mark_it->second) {
                st.temps.resize(mark_it->second);
            }
        }
        if (rs.large_bwd_temps_on_stack()) {
            auto& grads_layer = rs.simplified_grads(layer_idx);
            grads_layer.d_qkv.Data = nullptr;
            grads_layer.d_mlp_up.Data = nullptr;
            grads_layer.d_swiglu.Data = nullptr;
        }
        rs.scratch().cudnn_workspace.Data = nullptr;
        layer_checkpoints.erase(layer_idx);
        layer_temp_marks.erase(layer_idx);
    };

    const bool use_graphs = mBackwardGraphsEnabled && mBackwardGraphCut > 0;
    if (use_graphs && (mGraphB != B || mGraphT != T)) {
        reset_cuda_graphs();
        mGraphB = B;
        mGraphT = T;
    }
    const int graph_idx = (micro_step > 0) ? 1 : 0;
    const bool capturing = use_graphs && mBackwardGraph[graph_idx] == nullptr;

    auto run_ops_range = [&](std::size_t begin, std::size_t end) {
        end = std::min(end, mBackward->operations.size());
        for (std::size_t op_idx = begin; op_idx < end; ++op_idx) {
            const auto& op = mBackward->operations[op_idx];
            const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
            if (trace_ops) {
                fprintf(stderr, "[DSL TRACE] bwd op %zu/%zu id=%s type=%s\n",
                        op_idx, mBackward->operations.size(), op.id.c_str(), op_type.c_str());
                fflush(stderr);
            }
            maybe_start_layer(op);
            if (recompute_any) {
                int layer_idx = -1;
                bool found = false;
                for (const auto& name : op.inputs) {
                    if (extract_layer(name, layer_idx)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    for (const auto& name : op.outputs) {
                        if (extract_layer(name, layer_idx)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (found) {
                    recompute_layer(layer_idx);
                }
            }

            if (op_type == "view") {
                Tensor& src = get_tensor(st, op.inputs.at(0), mSaved);
                auto shape = resolve_view_shape(op, st.shape_env, st, mSaved);
                Tensor view = view_tensor(src, shape);
                auto it = st.tensors.find(op.outputs.at(0));
                if (it != st.tensors.end()) {
                    it->second = view;
                } else {
                    st.tensors.emplace(op.outputs.at(0), view);
                }
                continue;
            }

            if (op_type == "add") {
                Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
                Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
                if (a.DType != b.DType || a.nelem() != b.nelem()) {
                    throw std::runtime_error("DSL graph executor: add op shape or dtype mismatch");
                }
                std::vector<long> shape(a.Sizes.begin(), a.Sizes.begin() + a.Rank);
                Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, shape);
                vector_add_sr(out, a, b, 1.0f, static_cast<long>(a.nelem()), 0, rs.MainStream);
                continue;
            }

            if (op_type == "matmul_backward") {
                Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
                Tensor& a = get_tensor(st, op.inputs.at(1), mSaved);
                Tensor& b = get_tensor(st, op.inputs.at(2), mSaved);

                const std::string& dA_name = (op.outputs.size() > 0) ? op.outputs.at(0) : "";
                const std::string& dB_name = (op.outputs.size() > 1) ? op.outputs.at(1) : "";

                Tensor* dA_ptr = nullptr;
                Tensor* dB_ptr = nullptr;
                if (!dA_name.empty()) {
                    dA_ptr = &ensure_tensor(st, dA_name, a.DType,
                                            {a.Sizes[0], a.Sizes[1]});
                }
                if (!dB_name.empty()) {
                    dB_ptr = &ensure_tensor(st, dB_name, b.DType,
                                            {b.Sizes[0], b.Sizes[1]});
                }

                if (!dA_ptr && !dB_ptr) {
                    continue;
                }

                EMMTranspose mode = parse_transpose(op.attrs);
                bool used_recipe = false;

                // Determine op kind for quant buffers (only for NT layout).
                int layer_idx = -1;
                auto op_kind = matmul_op_from_weight(op.inputs.at(2), layer_idx);
                const bool allow_quant = op_kind.has_value() && allow_quant_layer(mOptions, config, layer_idx);
                const modules::MatmulOp matmul_op = op_kind.value_or(modules::MatmulOp::LMHead);

                bool do_accumulate = dB_ptr && (accumulate_tensors.count(dB_name) > 0);
                bool skip_weight_grad = true;
                if (dB_ptr) {
                    if (auto base_name = base_param_from_grad(dB_name)) {
                        if (mWeights.has(*base_name) && mWeights.is_trainable(*base_name)) {
                            skip_weight_grad = false;
                        }
                    }
                }

                if (mOptions.TrainingRecipe && mode == EMMTranspose::NT && a.Sizes[0] == B * T) {
                    const recipes::Recipe& recipe = *mOptions.TrainingRecipe;

                    Tensor dA_tmp{};
                    Tensor dB_tmp{};
                    Tensor* dA_use = dA_ptr;
                    Tensor* dB_use = dB_ptr;
                    if (!dA_use) {
                        dA_tmp = rs.temp_alloc(a.DType, {a.Sizes[0], a.Sizes[1]});
                        st.temps.push_back(dA_tmp);
                        dA_use = &dA_tmp;
                    }
                    if (!dB_use) {
                        dB_tmp = rs.temp_alloc(b.DType, {b.Sizes[0], b.Sizes[1]});
                        st.temps.push_back(dB_tmp);
                        dB_use = &dB_tmp;
                    }

                    Tensor* weight_ptr = &b;
                    if (allow_quant) {
                        if (const Tensor* cached = get_fp8_cached_weight(op.inputs.at(2), b, rs.MainStream)) {
                            weight_ptr = const_cast<Tensor*>(cached);
                        }
                    }

                    modules::MatmulContext ctx;
                    ctx.dinp = dA_use;
                    ctx.dweight = dB_use;
                    ctx.dout = &d_out;
                    ctx.inp = &a;
                    ctx.weight = weight_ptr;
                    ctx.B = static_cast<int>(B);
                    ctx.T = static_cast<int>(T);
                    ctx.C_in = static_cast<int>(a.Sizes[1]);
                    ctx.C_out = static_cast<int>(b.Sizes[0]);
                    ctx.run_state = &rs;
                    ctx.stream = rs.MainStream;
                    ctx.layer_idx = layer_idx;
                    ctx.op = matmul_op;
                    ctx.accumulate = do_accumulate;
                    ctx.skip_weight_grad = skip_weight_grad || !dB_ptr;
                    ctx.allow_fp8 = allow_quant;
                    ctx.allow_fp4 = allow_quant;
                    if (allow_quant) {
                        ctx.dout_quant = fp8_grad_buffer(rs, matmul_op);
                        if (!ctx.dout_quant || !ctx.dout_quant->Data) {
                            ctx.allow_fp8 = false;
                        }
                    }

                    recipe.backward_matmul(ctx);
                    used_recipe = true;
                }

                if (!used_recipe) {
                    // Fallback: emulate matmul backward with explicit matmuls.
                    EMMTranspose mode_dA = EMMTranspose::NN;
                    EMMTranspose mode_dB = EMMTranspose::NN;
                    switch (mode) {
                        case EMMTranspose::NN:
                            mode_dA = EMMTranspose::NT;
                            mode_dB = EMMTranspose::TN;
                            break;
                        case EMMTranspose::NT:
                            mode_dA = EMMTranspose::NN;
                            mode_dB = EMMTranspose::TN;
                            break;
                        case EMMTranspose::TN:
                            mode_dA = EMMTranspose::NT;
                            mode_dB = EMMTranspose::NN;
                            break;
                        case EMMTranspose::TT:
                            mode_dA = EMMTranspose::TT;
                            mode_dB = EMMTranspose::TT;
                            break;
                    }

                    if (dA_ptr) {
                        int M = 0, N = 0, K = 0;
                        matmul_dims(d_out, b, mode_dA, M, N, K);
                        EMMTranspose mode_col = swap_transpose(mode_dA);
                        matmul(*dA_ptr, b, d_out, std::nullopt, nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               N, M, K, mode_col, false, rs.MainStream);
                    }
                    if (dB_ptr && !skip_weight_grad) {
                        int M = 0, N = 0, K = 0;
                        matmul_dims(d_out, a, mode_dB, M, N, K);
                        EMMTranspose mode_col = swap_transpose(mode_dB);
                        matmul(*dB_ptr, a, d_out, std::nullopt, nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               N, M, K, mode_col, do_accumulate, rs.MainStream);
                    }
                }

                if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAWeights && mLoRAGrads && mLoRARunState) {
                    int lora_layer_idx = -1;
                    std::string field;
                    if (parse_block_param(op.inputs.at(2), lora_layer_idx, field)) {
                        if (field == "qkv_weight" || field == "out_weight" ||
                            field == "mlp_up_weight" || field == "mlp_down_weight") {
                            auto& acts = rs.simplified_acts(lora_layer_idx);
                            auto& grads_layer = rs.simplified_grads(lora_layer_idx);
                            auto& lora_rs = *mLoRARunState;
                            auto& lora_block = mLoRAWeights->get_block(lora_layer_idx, rs.MainStream);
                            bool lora_accum = false;
                            auto& lora_grads = mLoRAGrads->get_block_full(lora_layer_idx, rs.MainStream, comm, lora_accum);
                            lora_accum = lora_accum || (do_accumulate && !skip_weight_grad);

                            const int Bv = static_cast<int>(B);
                            const int Tv = static_cast<int>(T);
                            const int C = static_cast<int>(config.HiddenSize);
                            const int D = static_cast<int>(config.IntermediateSize);
                            const int Hq = static_cast<int>(config.NumQueryHeads);
                            const int Hkv = static_cast<int>(config.NumKeyValHeads);
                            const int Hs = static_cast<int>(config.head_size());
                            const int rank = mLoRAConfig->rank;
                            const float scaling = mLoRAConfig->scaling();
                            const float dropout = mLoRAConfig->dropout;
                            const bool training = lora_rs.is_training;
                            const int BT = Bv * Tv;
                            const std::string block_prefix = "blocks[" + std::to_string(lora_layer_idx) + "].";

                            auto grad_tensor = [&](const std::string& field) -> Tensor& {
                                return get_tensor(st, "d_" + block_prefix + field, mSaved);
                            };

                            auto dropout_seed = [&](int proj_type) -> unsigned int {
                                return lora_rs.dropout_base_seed
                                       + static_cast<unsigned int>(lora_layer_idx) * 1000000u
                                       + static_cast<unsigned int>(proj_type) * 100000u
                                       + static_cast<unsigned int>(lora_rs.micro_step) * 10000u;
                            };

                            if (field == "qkv_weight") {
                                Tensor ln1_input;
                                if (mOptions.RecomputeLoRA) {
                                    Tensor& res_ffn = rs.get_residual(lora_layer_idx, rs.MainStream);
                                    Tensor& ln1_weight = resolve_param_tensor(st, "blocks[" + std::to_string(lora_layer_idx) + "].ln1_weight");
                                    ln1_input = recompute_lora_rmsnorm(lora_rs, res_ffn, ln1_weight,
                                                                       config.RmsNormEps, Bv, Tv, C, rs.MainStream);
                                } else {
                                    ln1_input = acts.ln1;
                                }

                                Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
                                modules::LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};

                                if (lora_block.attention.q.has_value() && lora_grads.attention.q.has_value()) {
                                    dA_q = lora_grads.attention.q->A;
                                    dB_q = lora_grads.attention.q->B;
                                    lora_q = *lora_block.attention.q;
                                }
                                if (lora_block.attention.k.has_value() && lora_grads.attention.k.has_value()) {
                                    dA_k = lora_grads.attention.k->A;
                                    dB_k = lora_grads.attention.k->B;
                                    lora_k = *lora_block.attention.k;
                                }
                                if (lora_block.attention.v.has_value() && lora_grads.attention.v.has_value()) {
                                    dA_v = lora_grads.attention.v->A;
                                    dB_v = lora_grads.attention.v->B;
                                    lora_v = *lora_block.attention.v;
                                }

                                modules::detail::backward_lora_qkv_fused(
                                    dA_q, dB_q,
                                    dA_k, dB_k,
                                    dA_v, dB_v,
                                    grads_layer.d_ln1,
                                    grads_layer.d_qkv,
                                    ln1_input,
                                    lora_q, lora_k, lora_v,
                                    scaling,
                                    dropout,
                                    dropout_seed(0), dropout_seed(1), dropout_seed(2),
                                    training,
                                    BT,
                                    C,
                                    Hq * Hs,
                                    Hkv * Hs,
                                    rank,
                                    lora_accum,
                                    lora_rs.intermediate,
                                    lora_rs.intermediate2,
                                    lora_rs.slice,
                                    rs.CublasLtHandle,
                                    rs.CuBlasWorkspace,
                                    rs.MainStream);
                            } else if (field == "out_weight") {
                                if (lora_block.attention.o.has_value() && lora_grads.attention.o.has_value()) {
                                    modules::detail::backward_lora_layer(
                                        lora_grads.attention.o->A, lora_grads.attention.o->B,
                                        grads_layer.d_att,
                                        grads_layer.d_res_att, 0,
                                        acts.att,
                                        lora_block.attention.o->A, lora_block.attention.o->B,
                                        scaling,
                                        dropout, dropout_seed(3), training,
                                        lora_rs.intermediate, lora_rs.slice,
                                        BT, Hq * Hs, C, rank, lora_accum,
                                        rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                                }
                            } else if (field == "mlp_up_weight") {
                                Tensor ln2_input;
                                if (mOptions.RecomputeLoRA) {
                                    Tensor& ln2_weight = resolve_param_tensor(st, "blocks[" + std::to_string(lora_layer_idx) + "].ln2_weight");
                                    ln2_input = recompute_lora_rmsnorm(lora_rs, acts.residual_att, ln2_weight,
                                                                       config.RmsNormEps, Bv, Tv, C, rs.MainStream);
                                } else {
                                    ln2_input = acts.ln2;
                                }

                                Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
                                modules::LoRALayerWeights<Tensor> lora_up{}, lora_gate{};

                                if (lora_block.mlp.up.has_value() && lora_grads.mlp.up.has_value()) {
                                    dA_up = lora_grads.mlp.up->A;
                                    dB_up = lora_grads.mlp.up->B;
                                    lora_up = *lora_block.mlp.up;
                                }
                                if (lora_block.mlp.gate.has_value() && lora_grads.mlp.gate.has_value()) {
                                    dA_gate = lora_grads.mlp.gate->A;
                                    dB_gate = lora_grads.mlp.gate->B;
                                    lora_gate = *lora_block.mlp.gate;
                                }

                                Tensor& d_mlp_up = grad_tensor("mlp_up");

                                modules::detail::backward_lora_mlp_up_gate_fused(
                                    dA_up, dB_up,
                                    dA_gate, dB_gate,
                                    grads_layer.d_ln2,
                                    d_mlp_up,
                                    ln2_input,
                                    lora_up, lora_gate,
                                    scaling,
                                    dropout,
                                    dropout_seed(4), dropout_seed(5), training,
                                    BT,
                                    C,
                                    D,
                                    rank,
                                    lora_accum,
                                    lora_rs.intermediate,
                                    lora_rs.intermediate2,
                                    lora_rs.slice,
                                    rs.CublasLtHandle,
                                    rs.CuBlasWorkspace,
                                    rs.MainStream);
                            } else if (field == "mlp_down_weight") {
                                if (lora_block.mlp.down.has_value() && lora_grads.mlp.down.has_value()) {
                                    Tensor& d_swiglu = grad_tensor("swiglu");
                                    modules::detail::backward_lora_layer(
                                        lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                                        d_swiglu,
                                        grads_layer.d_res_ffn, 0,
                                        acts.swiglu,
                                        lora_block.mlp.down->A, lora_block.mlp.down->B,
                                        scaling,
                                        dropout, dropout_seed(6), training,
                                        lora_rs.intermediate, lora_rs.slice,
                                        BT, D, C, rank, lora_accum,
                                        rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                                }
                            }
                        }
                    }
                }
                continue;
            }

            if (op_type == "matmul" || op_type == "matmul_bias") {
                if (op.outputs.at(0).empty()) {
                    continue;
                }
                Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
                Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
                if (a.Rank != 2 || b.Rank != 2) {
                    throw std::runtime_error(
                        "DSL graph executor: matmul expects rank-2 tensors (op=" + op.id +
                        ", a=" + op.inputs.at(0) + " rank=" + std::to_string(a.Rank) +
                        " shape=" + tensor_shape_str(a) +
                        ", b=" + op.inputs.at(1) + " rank=" + std::to_string(b.Rank) +
                        " shape=" + tensor_shape_str(b) + ")"
                    );
                }
                EMMTranspose mode = parse_transpose(op.attrs);
                int M = 0, N = 0, K = 0;
                matmul_dims(a, b, mode, M, N, K);
                const std::string& out_name = op.outputs.at(0);
                bool do_accumulate = accumulate_tensors.count(op.outputs.at(0)) > 0;
                std::optional<Tensor> bias;
                if (op_type == "matmul_bias" && op.inputs.size() > 2 && !op.inputs.at(2).empty()) {
                    bias = get_tensor(st, op.inputs.at(2), mSaved);
                }
                bool skip_weight_grad = false;
                bool is_param_grad = false;
                if (auto param_name = base_param_from_grad(out_name)) {
                    if (mWeights.has(*param_name)) {
                        is_param_grad = true;
                        if (!mWeights.is_trainable(*param_name)) {
                            skip_weight_grad = true;
                        }
                    }
                }
                // DSL matmul uses row-major semantics; map to column-major backend by swapping A/B,
                // swapping M/N, and swapping transpose flags.
                EMMTranspose mode_col = swap_transpose(mode);
                if (!skip_weight_grad) {
                    std::vector<long> shape{M, N};
                    Tensor& out = ensure_tensor(st, out_name, a.DType, shape);
                    matmul(out, b, a, bias, nullptr, nullptr,
                           rs.CublasLtHandle, rs.CuBlasWorkspace,
                           N, M, K, mode_col, do_accumulate, rs.MainStream);
                }

                if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAWeights && mLoRAGrads && mLoRARunState) {
                    if (auto base_name = base_param_from_grad(out_name)) {
                        int layer_idx = -1;
                        std::string field;
                        if (parse_block_param(*base_name, layer_idx, field)) {
                            auto& acts = rs.simplified_acts(layer_idx);
                            auto& grads_layer = rs.simplified_grads(layer_idx);
                            auto& lora_rs = *mLoRARunState;
                            auto& lora_block = mLoRAWeights->get_block(layer_idx, rs.MainStream);
                            bool lora_accum = false;
                            auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, rs.MainStream, comm, lora_accum);
                            lora_accum = lora_accum || (do_accumulate && !skip_weight_grad && is_param_grad);

                            const int Bv = static_cast<int>(B);
                            const int Tv = static_cast<int>(T);
                            const int C = static_cast<int>(config.HiddenSize);
                            const int D = static_cast<int>(config.IntermediateSize);
                            const int Hq = static_cast<int>(config.NumQueryHeads);
                            const int Hkv = static_cast<int>(config.NumKeyValHeads);
                            const int Hs = static_cast<int>(config.head_size());
                            const int rank = mLoRAConfig->rank;
                            const float scaling = mLoRAConfig->scaling();
                            const float dropout = mLoRAConfig->dropout;
                            const bool training = lora_rs.is_training;
                            const int BT = Bv * Tv;
                            const std::string block_prefix = "blocks[" + std::to_string(layer_idx) + "].";

                            auto grad_tensor = [&](const std::string& field) -> Tensor& {
                                return get_tensor(st, "d_" + block_prefix + field, mSaved);
                            };

                            auto dropout_seed = [&](int proj_type) -> unsigned int {
                                return lora_rs.dropout_base_seed
                                       + static_cast<unsigned int>(layer_idx) * 1000000u
                                       + static_cast<unsigned int>(proj_type) * 100000u
                                       + static_cast<unsigned int>(lora_rs.micro_step) * 10000u;
                            };

                            if (field == "qkv_weight") {
                                Tensor ln1_input;
                                if (mOptions.RecomputeLoRA) {
                                    Tensor& res_ffn = rs.get_residual(layer_idx, rs.MainStream);
                                    Tensor& ln1_weight = resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].ln1_weight");
                                    ln1_input = recompute_lora_rmsnorm(lora_rs, res_ffn, ln1_weight,
                                                                       config.RmsNormEps, Bv, Tv, C, rs.MainStream);
                                } else {
                                    ln1_input = acts.ln1;
                                }

                                Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
                                modules::LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};

                                if (lora_block.attention.q.has_value() && lora_grads.attention.q.has_value()) {
                                    dA_q = lora_grads.attention.q->A;
                                    dB_q = lora_grads.attention.q->B;
                                    lora_q = *lora_block.attention.q;
                                }
                                if (lora_block.attention.k.has_value() && lora_grads.attention.k.has_value()) {
                                    dA_k = lora_grads.attention.k->A;
                                    dB_k = lora_grads.attention.k->B;
                                    lora_k = *lora_block.attention.k;
                                }
                                if (lora_block.attention.v.has_value() && lora_grads.attention.v.has_value()) {
                                    dA_v = lora_grads.attention.v->A;
                                    dB_v = lora_grads.attention.v->B;
                                    lora_v = *lora_block.attention.v;
                                }

                                modules::detail::backward_lora_qkv_fused(
                                    dA_q, dB_q,
                                    dA_k, dB_k,
                                    dA_v, dB_v,
                                    grads_layer.d_ln1,
                                    grads_layer.d_qkv,
                                    ln1_input,
                                    lora_q, lora_k, lora_v,
                                    scaling,
                                    dropout,
                                    dropout_seed(0), dropout_seed(1), dropout_seed(2),
                                    training,
                                    BT,
                                    C,
                                    Hq * Hs,
                                    Hkv * Hs,
                                    rank,
                                    lora_accum,
                                    lora_rs.intermediate,
                                    lora_rs.intermediate2,
                                    lora_rs.slice,
                                    rs.CublasLtHandle,
                                    rs.CuBlasWorkspace,
                                    rs.MainStream);
                            } else if (field == "out_weight") {
                                if (lora_block.attention.o.has_value() && lora_grads.attention.o.has_value()) {
                                    modules::detail::backward_lora_layer(
                                        lora_grads.attention.o->A, lora_grads.attention.o->B,
                                        grads_layer.d_att,
                                        grads_layer.d_res_att, 0,
                                        acts.att,
                                        lora_block.attention.o->A, lora_block.attention.o->B,
                                        scaling,
                                        dropout, dropout_seed(3), training,
                                        lora_rs.intermediate, lora_rs.slice,
                                        BT, Hq * Hs, C, rank, lora_accum,
                                        rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                                }
                            } else if (field == "mlp_up_weight") {
                                Tensor ln2_input;
                                if (mOptions.RecomputeLoRA) {
                                    Tensor& ln2_weight = resolve_param_tensor(st, "blocks[" + std::to_string(layer_idx) + "].ln2_weight");
                                    ln2_input = recompute_lora_rmsnorm(lora_rs, acts.residual_att, ln2_weight,
                                                                       config.RmsNormEps, Bv, Tv, C, rs.MainStream);
                                } else {
                                    ln2_input = acts.ln2;
                                }

                                Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
                                modules::LoRALayerWeights<Tensor> lora_up{}, lora_gate{};

                                if (lora_block.mlp.up.has_value() && lora_grads.mlp.up.has_value()) {
                                    dA_up = lora_grads.mlp.up->A;
                                    dB_up = lora_grads.mlp.up->B;
                                    lora_up = *lora_block.mlp.up;
                                }
                                if (lora_block.mlp.gate.has_value() && lora_grads.mlp.gate.has_value()) {
                                    dA_gate = lora_grads.mlp.gate->A;
                                    dB_gate = lora_grads.mlp.gate->B;
                                    lora_gate = *lora_block.mlp.gate;
                                }

                                Tensor& d_mlp_up = grad_tensor("mlp_up");

                                modules::detail::backward_lora_mlp_up_gate_fused(
                                    dA_up, dB_up,
                                    dA_gate, dB_gate,
                                    grads_layer.d_ln2,
                                    d_mlp_up,
                                    ln2_input,
                                    lora_up, lora_gate,
                                    scaling,
                                    dropout,
                                    dropout_seed(4), dropout_seed(5), training,
                                    BT,
                                    C,
                                    D,
                                    rank,
                                    lora_accum,
                                    lora_rs.intermediate,
                                    lora_rs.intermediate2,
                                    lora_rs.slice,
                                    rs.CublasLtHandle,
                                    rs.CuBlasWorkspace,
                                    rs.MainStream);
                            } else if (field == "mlp_down_weight") {
                                if (lora_block.mlp.down.has_value() && lora_grads.mlp.down.has_value()) {
                                    Tensor& d_swiglu = grad_tensor("swiglu");
                                    modules::detail::backward_lora_layer(
                                        lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                                        d_swiglu,
                                        grads_layer.d_res_ffn, 0,
                                        acts.swiglu,
                                        lora_block.mlp.down->A, lora_block.mlp.down->B,
                                        scaling,
                                        dropout, dropout_seed(6), training,
                                        lora_rs.intermediate, lora_rs.slice,
                                        BT, D, C, rank, lora_accum,
                                        rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
                                }
                            }
                        }
                    }
                }
                continue;
            }

        if (op_type == "bias_add_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            if (!op.outputs.at(0).empty()) {
                st.tensors.emplace(op.outputs.at(0), d_out);
            }
            int Bv = 1;
            int Tv = 1;
            int OC = 1;
            if (d_out.Rank == 2) {
                Bv = static_cast<int>(d_out.Sizes[0]);
                Tv = 1;
                OC = static_cast<int>(d_out.Sizes[1]);
            } else {
                Bv = static_cast<int>(d_out.Sizes[0]);
                Tv = static_cast<int>(d_out.Sizes[1]);
                OC = static_cast<int>(d_out.Sizes[2]);
            }
            const bool bias_trainable = mWeights.has(op.inputs.at(1)) && mWeights.is_trainable(op.inputs.at(1));
            if (bias_trainable && op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
                Tensor& d_bias = ensure_tensor(st, op.outputs.at(1), d_out.DType, {static_cast<long>(OC)});
                const bool do_accumulate = accumulate_tensors.count(op.outputs.at(1)) > 0;
                const int scratch_bytes = get_bias_backward_scratch_size(d_out.DType, OC, rs.DeviceProp);
                Tensor scratch = rs.temp_alloc(ETensorDType::FP32, {static_cast<long>(scratch_bytes / sizeof(float))});
                st.temps.push_back(scratch);
                if (do_accumulate) {
                    Tensor tmp = rs.temp_alloc(d_out.DType, {static_cast<long>(OC)});
                    st.temps.push_back(tmp);
                    backward_bias(tmp, d_out, nullptr, nullptr, scratch,
                                  Bv, Tv, OC, rs.DeviceProp, rs.MainStream);
                    vector_add_sr(d_bias, d_bias, tmp, 1.0f, static_cast<long>(d_bias.nelem()), 0,
                                  rs.MainStream);
                } else {
                    backward_bias(d_bias, d_out, nullptr, nullptr, scratch,
                                  Bv, Tv, OC, rs.DeviceProp, rs.MainStream);
                }
            }
            continue;
        }

        if (op_type == "swiglu_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& inp = get_tensor(st, op.inputs.at(1), mSaved);
            if (op.outputs.at(0).empty()) {
                continue;
            }
            const long D = inp.Sizes[2] / 2;
            Tensor& d_inp = ensure_tensor(st, op.outputs.at(0), inp.DType, {inp.Sizes[0], inp.Sizes[1], inp.Sizes[2]});
            float* abs_max_ptr = rs.has_grad_quants()
                ? rs.simplified_quant_grads().d_mlp_up.abs_max()
                : nullptr;
            swiglu_backward(d_inp, d_out, inp, abs_max_ptr,
                            static_cast<int>(inp.Sizes[0]), static_cast<int>(inp.Sizes[1]),
                            static_cast<int>(D), rs.MainStream);
            continue;
        }

        if (op_type == "rope_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& freqs = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& pos_ids = get_tensor(st, op.inputs.at(2), mSaved);
            if (op.outputs.at(0).empty()) {
                continue;
            }
            Tensor& d_inp = ensure_tensor(st, op.outputs.at(0), d_out.DType,
                                          {d_out.Sizes[0], d_out.Sizes[1], d_out.Sizes[2], d_out.Sizes[3]});
            int Hq = static_cast<int>(config.NumQueryHeads);
            int Hkv = static_cast<int>(config.NumKeyValHeads);
            int Hs = static_cast<int>(config.head_size());
            int rotary_dim = Hs;
            if (auto* rd_attr = find_attr(op.attrs, "rotary_dim")) {
                if (auto v = attr_int(*rd_attr)) {
                    rotary_dim = static_cast<int>(*v);
                } else if (auto s = attr_string(*rd_attr)) {
                    rotary_dim = static_cast<int>(resolve_dim(Dim::symbolic(*s), st.shape_env));
                }
            }
            float* abs_max_ptr = rs.has_grad_quants()
                ? rs.simplified_quant_grads().d_qkv.abs_max()
                : nullptr;
            rope_backward(d_inp, d_out, freqs, reinterpret_cast<int*>(pos_ids.Data), abs_max_ptr,
                          static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
            continue;
        }

        if (op_type == "qkv_qk_norm_rope_backward") {
            Tensor& d_qkv = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& qkv = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& q_norm = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor& k_norm = get_tensor(st, op.inputs.at(3), mSaved);
            Tensor& q_rstd = get_tensor(st, op.inputs.at(4), mSaved);
            Tensor& k_rstd = get_tensor(st, op.inputs.at(5), mSaved);
            Tensor& freqs = get_tensor(st, op.inputs.at(6), mSaved);
            Tensor& pos_ids = get_tensor(st, op.inputs.at(7), mSaved);

            const int Hq = static_cast<int>(config.NumQueryHeads);
            const int Hkv = static_cast<int>(config.NumKeyValHeads);
            const int Hs = static_cast<int>(config.head_size());
            const int qkv_channels = Hs * (Hq + 2 * Hkv);

            Tensor d_qkv_view = (d_qkv.Rank == 4)
                ? view_tensor(d_qkv, {B, T, qkv_channels})
                : d_qkv;
            Tensor qkv_view = (qkv.Rank == 4)
                ? view_tensor(qkv, {B, T, qkv_channels})
                : qkv;

            int rotary_dim = Hs;
            if (auto* rd_attr = find_attr(op.attrs, "rotary_dim")) {
                if (auto v = attr_int(*rd_attr)) {
                    rotary_dim = static_cast<int>(*v);
                } else if (auto s = attr_string(*rd_attr)) {
                    rotary_dim = static_cast<int>(resolve_dim(Dim::symbolic(*s), st.shape_env));
                }
            }

            const bool q_norm_trainable = mWeights.has(op.inputs.at(2)) && mWeights.is_trainable(op.inputs.at(2));
            const bool k_norm_trainable = mWeights.has(op.inputs.at(3)) && mWeights.is_trainable(op.inputs.at(3));
            const bool rope_fusable = (rotary_dim > 0)
                && ((Hs % 2) == 0)
                && (((Hs / 2) % 32) == 0)
                && (freqs.Rank >= 2)
                && (freqs.Sizes[1] >= Hs);
            float* d_qkv_abs_max = rs.has_grad_quants()
                ? rs.simplified_quant_grads().d_qkv.abs_max()
                : nullptr;

            if (rope_fusable) {
                // Fused QK norm + RoPE backward (matches modular path).
                qkv_head_rmsnorm_rope_backward_dx(
                    d_qkv_view, qkv_view, q_norm, q_rstd,
                    freqs, reinterpret_cast<int*>(pos_ids.Data),
                    static_cast<int>(B), static_cast<int>(T), qkv_channels, Hq, Hs, 0,
                    rs.MainStream);
                qkv_head_rmsnorm_rope_backward_dx(
                    d_qkv_view, qkv_view, k_norm, k_rstd,
                    freqs, reinterpret_cast<int*>(pos_ids.Data),
                    static_cast<int>(B), static_cast<int>(T), qkv_channels, Hkv, Hs, Hq * Hs,
                    rs.MainStream);
                if (q_norm_trainable && op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
                    Tensor& d_q_norm = ensure_tensor(st, op.outputs.at(1), q_norm.DType, {Hs});
                    const bool acc = accumulate_tensors.count(op.outputs.at(1)) > 0;
                    qkv_head_rmsnorm_rope_backward_dweight(
                        d_q_norm, d_qkv_view, qkv_view, q_norm,
                        freqs, reinterpret_cast<int*>(pos_ids.Data),
                        static_cast<int>(B), static_cast<int>(T), qkv_channels, Hq, Hs, 0,
                        acc, rs.MainStream);
                }
                if (k_norm_trainable && op.outputs.size() > 2 && !op.outputs.at(2).empty()) {
                    Tensor& d_k_norm = ensure_tensor(st, op.outputs.at(2), k_norm.DType, {Hs});
                    const bool acc = accumulate_tensors.count(op.outputs.at(2)) > 0;
                    qkv_head_rmsnorm_rope_backward_dweight(
                        d_k_norm, d_qkv_view, qkv_view, k_norm,
                        freqs, reinterpret_cast<int*>(pos_ids.Data),
                        static_cast<int>(B), static_cast<int>(T), qkv_channels, Hkv, Hs, Hq * Hs,
                        acc, rs.MainStream);
                }
            } else {
                // Undo RoPE on gradients and activations (in-place) before QK RMSNorm backward.
                rope_backward(d_qkv, d_qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), d_qkv_abs_max,
                              static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
                rope_backward(qkv, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                              static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);

                if (q_norm_trainable && op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
                    Tensor& d_q_norm = ensure_tensor(st, op.outputs.at(1), q_norm.DType, {Hs});
                    const bool acc = accumulate_tensors.count(op.outputs.at(1)) > 0;
                    qkv_head_rmsnorm_backward_dweight(
                        d_q_norm, d_qkv_view, qkv_view, q_norm,
                        static_cast<int>(B), static_cast<int>(T), qkv_channels, Hq, Hs, 0,
                        acc, rs.MainStream);
                }
                if (k_norm_trainable && op.outputs.size() > 2 && !op.outputs.at(2).empty()) {
                    Tensor& d_k_norm = ensure_tensor(st, op.outputs.at(2), k_norm.DType, {Hs});
                    const bool acc = accumulate_tensors.count(op.outputs.at(2)) > 0;
                    qkv_head_rmsnorm_backward_dweight(
                        d_k_norm, d_qkv_view, qkv_view, k_norm,
                        static_cast<int>(B), static_cast<int>(T), qkv_channels, Hkv, Hs, Hq * Hs,
                        acc, rs.MainStream);
                }

                // Backward dx for Q/K RMSNorm (in-place on d_qkv_view).
                qkv_head_rmsnorm_backward_dx(
                    d_qkv_view, qkv_view, q_norm, q_rstd,
                    static_cast<int>(B), static_cast<int>(T), qkv_channels, Hq, Hs, 0,
                    rs.MainStream);
                qkv_head_rmsnorm_backward_dx(
                    d_qkv_view, qkv_view, k_norm, k_rstd,
                    static_cast<int>(B), static_cast<int>(T), qkv_channels, Hkv, Hs, Hq * Hs,
                    rs.MainStream);
            }
            if (!op.outputs.at(0).empty()) {
                st.tensors.emplace(op.outputs.at(0), d_qkv);
            }
            if (d_qkv_abs_max && rope_fusable) {
                abs_max(d_qkv_abs_max, d_qkv, d_qkv.nelem(), rs.DeviceProp, rs.MainStream);
            }
            continue;
        }

        if (op_type == "flash_attention_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& out = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& lse = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor& qkv = get_tensor(st, op.inputs.at(3), mSaved);
            if (op.outputs.at(0).empty()) {
                continue;
            }
            Tensor& d_qkv = ensure_tensor(st, op.outputs.at(0), qkv.DType,
                                          {qkv.Sizes[0], qkv.Sizes[1], qkv.Sizes[2], qkv.Sizes[3]});

            const int Hq = static_cast<int>(config.NumQueryHeads);
            const int Hkv = static_cast<int>(config.NumKeyValHeads);
            const int Hs = static_cast<int>(config.head_size());
            if (!rs.scratch().cudnn_workspace.Data) {
                rs.temp_acquire(rs.scratch().cudnn_workspace);
                st.temps.push_back(rs.scratch().cudnn_workspace);
            }
            attention_backward_cudnn(d_qkv, lse, out, d_out, qkv,
                                     rs.scratch().cudnn_workspace, rs.CudnnHandle,
                                     static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rs.MainStream);
            continue;
        }

        if (op_type == "zeros") {
            auto* shape_attr = find_attr(op.attrs, "shape");
            if (!shape_attr) {
                throw std::runtime_error("DSL graph executor: zeros missing shape attr");
            }
            auto shape = resolve_attr_shape(*shape_attr, st.shape_env);
            ETensorDType dtype = ETensorDType::BF16;
            if (auto* dtype_attr = find_attr(op.attrs, "dtype")) {
                if (auto s = attr_string(*dtype_attr)) {
                    dtype = dtype_from_str(*s);
                }
            }
            Tensor& out = ensure_tensor(st, op.outputs.at(0), dtype, shape);
            fill_zero(out, rs.MainStream);
            st.zero_tensors.insert(op.outputs.at(0));
            continue;
        }

        if (op_type == "fused_residual_rmsnorm_backward" || op.name == "fused_residual_rmsnorm_backward") {
            Tensor& d_y = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& residual_out = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor* d_residual_next = nullptr;
            if (!op.inputs.at(1).empty()) {
                d_residual_next = &get_tensor(st, op.inputs.at(1), mSaved);
            }

            Tensor& weight = resolve_param_tensor(st, op.inputs.at(3));

            Tensor& d_input = ensure_tensor(st, op.outputs.at(1), d_y.DType, {B, T, config.HiddenSize});
            Tensor* d_residual_out = nullptr;
            if (!op.outputs.at(0).empty()) {
                d_residual_out = &ensure_tensor(st, op.outputs.at(0), d_y.DType, {B, T, config.HiddenSize});
            }

            Tensor* d_residual = d_residual_next;
            if (!d_residual) {
                std::string zero_name = op.id + "_dres_zero";
                Tensor& d_residual_zero = ensure_tensor(st, zero_name, d_y.DType, {B, T, config.HiddenSize});
                fill_zero(d_residual_zero, rs.MainStream);
                d_residual = &d_residual_zero;
            }

            const bool weight_trainable = mWeights.has(op.inputs.at(3)) && mWeights.is_trainable(op.inputs.at(3));
            Tensor* d_weight = nullptr;
            bool skip_weight_grad = true;
            if (weight_trainable && op.outputs.size() > 2 && !op.outputs.at(2).empty()) {
                d_weight = &ensure_tensor(st, op.outputs.at(2), weight.DType, {config.HiddenSize});
                skip_weight_grad = false;
            }
            Tensor dummy_weight{};
            if (!d_weight) {
                dummy_weight = rs.temp_alloc(weight.DType, {config.HiddenSize});
                st.temps.push_back(dummy_weight);
                d_weight = &dummy_weight;
            }

            const bool skip_weight = skip_weight_grad;
            float* abs_max_ptr = nullptr;
            if (rs.has_grad_quants()) {
                bool use_res_att = false;
                if (op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
                    use_res_att = op.outputs.at(1).find("res_att") != std::string::npos;
                }
                if (!use_res_att) {
                    use_res_att = op.inputs.at(3).find("ln2_weight") != std::string::npos;
                }
                abs_max_ptr = use_res_att
                    ? rs.simplified_quant_grads().d_res_att.abs_max()
                    : rs.simplified_quant_grads().d_res_ffn.abs_max();
            }
            rmsnorm_backward(d_input, *d_weight, rs.scratch().rmsnorm_scratch,
                             *d_residual, d_y, residual_out, weight,
                             get_tensor(st, op.inputs.at(4), mSaved),
                             abs_max_ptr,
                             static_cast<int>(B), static_cast<int>(T), config.HiddenSize,
                             rs.DeviceProp, rs.MainStream,
                             skip_weight);
            if (d_residual_out && op.outputs.at(0) != op.outputs.at(1)) {
                CUDA_CHECK(cudaMemcpyAsync(d_residual_out->Data, d_input.Data, d_input.bytes(),
                                           cudaMemcpyDeviceToDevice, rs.MainStream));
            }
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(op.inputs.at(3), layer_idx, field) && field == "ln1_weight") {
                finish_layer(layer_idx);
            }
            continue;
        }

        if (op_type == "embedding_backward" || op.name == "embedding_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            if (op.outputs.empty() || op.outputs.at(0).empty()) {
                continue;
            }
            auto it = st.tensors.find(op.outputs.at(0));
            if (it == st.tensors.end()) {
                continue;
            }
            Tensor& d_emb = it->second;
            encoder_backward(d_emb,
                             rs.scratch().encoder_bwd_scratch,
                             rs.scratch().encoder_bwd_indices,
                             rs.scratch().encoder_bwd_info,
                             d_out,
                             rs.Inputs,
                             mLastInputsCpu,
                             static_cast<int>(B), static_cast<int>(T), config.HiddenSize,
                             next_rng_seed(),
                             rs.MainStream,
                             rs.side_stream_event(),
                             rs.side_stream());
            continue;
        }

        throw std::runtime_error("DSL graph executor: unsupported backward op " + op.name);
        }
    };

    if (use_graphs) {
        auto run_graph_ops = [&]() { run_ops_range(0, mBackwardGraphCut); };
        trace_or_execute_cuda_graph_with_stack(run_graph_ops, rs.MainStream, mBackwardGraph[graph_idx], use_graphs, rs.Stack, mBackwardCheckpoint[graph_idx]);
        if (mBackwardGraphCut < mBackward->operations.size()) {
            run_ops_range(mBackwardGraphCut, mBackward->operations.size());
        }
    } else {
        run_ops_range(0, mBackward->operations.size());
    }

    if (!use_graphs || capturing) {
        free_temps(st);
        rs.temp_free(rs.non_block_activations().output);
    }
}

}  // namespace dsl
