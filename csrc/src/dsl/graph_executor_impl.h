// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Graph executor (DSL-driven).

#include "dsl/autodiff.h"

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

#include "modules/model/modular_model.h"
#include "modules/gradient_manager.h"
#include "modules/lora/lora_model_core.h"
#include "modules/lora/lora_model_state_management.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/lora/lora_utils.h"
#include "modules/model/modular_model_fwd.h"
#include "modules/run_state.h"
#include "modules/weights/weight_manager.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "kernels/kernels.h"

namespace dsl {
namespace {

template<typename Block> using Model = modules::ModularTransformerModel<Block>;
template<typename Block> using RunState = modules::ModularRunState<Block>;
template<typename Block> using WeightManager = modules::ModularWeightManager<Block>;
template<typename Block> using GradManager = modules::ModularGradientManager<Block>;

constexpr std::string_view kSavedPrefix = "saved.";

template<typename Block>
struct ExecState {
    RunState<Block>& rs;
    WeightManager<Block>& weights;
    const modules::ModelConfig& config;
    NCCLCommunicator& comm;
    long B = 0;
    long T = 0;
    ShapeEnv shape_env{};

    std::unordered_map<std::string, Tensor> tensors;
    std::unordered_set<std::string> zero_tensors;
    std::vector<Tensor> temps;

    // Block weight cache
    int current_block_layer = -1;
    bool block_loaded = false;

    // Non-block weight cache flags
    bool embeddings_loaded = false;
    bool final_norm_loaded = false;
    bool lm_head_loaded = false;

    // Block gradient cache
    std::unordered_map<int, typename Block::Gradients*> block_grads;
    std::unordered_map<int, bool> block_grad_accumulate;

};

template<typename Gradients>
Tensor* get_ln1_weight_grad(Gradients& grads) {
    if constexpr (requires(Gradients g) { g.ln1_grads.d_weight; }) {
        return &grads.ln1_grads.d_weight;
    } else if constexpr (requires(Gradients g) { g.ln1.d_weight; }) {
        return &grads.ln1.d_weight;
    } else {
        return nullptr;
    }
}

template<typename Gradients>
Tensor* get_ln2_weight_grad(Gradients& grads) {
    if constexpr (requires(Gradients g) { g.ln2_grads.d_weight; }) {
        return &grads.ln2_grads.d_weight;
    } else if constexpr (requires(Gradients g) { g.ln2.d_weight; }) {
        return &grads.ln2.d_weight;
    } else {
        return nullptr;
    }
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
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

template<typename Block>
Tensor* resolve_block_activation_tensor(ExecState<Block>& st, const std::string& name, ETensorDType dtype,
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

template<typename Block>
Tensor* resolve_block_gradient_tensor(ExecState<Block>& st, const std::string& name, ETensorDType dtype,
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

template<typename Block>
Tensor& ensure_tensor(ExecState<Block>& st, const std::string& name, ETensorDType dtype, const std::vector<long>& shape) {
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

template<typename Block>
Tensor& resolve_param_tensor(ExecState<Block>& st, const std::string& name);

template<typename Block>
Tensor& get_tensor(ExecState<Block>& st, const std::string& name, const std::unordered_map<std::string, Tensor>& saved) {
    // Check for explicit "saved." prefix first
    if (starts_with(name, kSavedPrefix)) {
        std::string key = std::string(name.substr(kSavedPrefix.size()));
        auto it = saved.find(key);
        if (it == saved.end()) {
            throw std::runtime_error("DSL graph executor: missing saved tensor " + key);
        }
        return const_cast<Tensor&>(it->second);
    }
    // Check current tensors (params, intermediates, gradients)
    auto it = st.tensors.find(name);
    if (it != st.tensors.end()) {
        return it->second;
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
template<typename Block>
Tensor* try_get_tensor(ExecState<Block>& st, const std::string& name, std::unordered_map<std::string, Tensor>& saved) {
    if (starts_with(name, kSavedPrefix)) {
        std::string key = std::string(name.substr(kSavedPrefix.size()));
        auto it = saved.find(key);
        return it != saved.end() ? &it->second : nullptr;
    }
    auto it = st.tensors.find(name);
    if (it != st.tensors.end()) {
        return &it->second;
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

template<typename Block>
void ensure_block_loaded(ExecState<Block>& st, int layer_idx) {
    if (st.block_loaded && st.current_block_layer == layer_idx) {
        return;
    }
    if (st.block_loaded) {
        st.weights.release_block(st.current_block_layer, st.rs.MainStream);
        st.block_loaded = false;
    }
    st.weights.gather_block(layer_idx, st.comm, st.rs.MainStream);
    st.current_block_layer = layer_idx;
    st.block_loaded = true;
}

template<typename Block>
Tensor& resolve_block_weight(ExecState<Block>& st, int layer_idx, const std::string& param) {
    ensure_block_loaded(st, layer_idx);
    auto& block = st.weights.get_block(layer_idx, st.rs.MainStream);
    if (param == "ln1_weight") return block.ln1.weight;
    if (param == "ln2_weight") return block.ln2.weight;
    if (param == "qkv_weight") return block.attention.qkv_weight;
    if (param == "qkv_bias") {
        if (!block.attention.qkv_bias.has_value()) {
            throw std::runtime_error("DSL graph executor: missing qkv_bias for layer " + std::to_string(layer_idx));
        }
        return block.attention.qkv_bias.value();
    }
    if (param == "out_weight") return block.attention.out_weight;
    if (param == "q_norm_weight") {
        if (!block.attention.q_norm_weight.has_value()) {
            throw std::runtime_error("DSL graph executor: missing q_norm_weight for layer " + std::to_string(layer_idx));
        }
        return block.attention.q_norm_weight.value();
    }
    if (param == "k_norm_weight") {
        if (!block.attention.k_norm_weight.has_value()) {
            throw std::runtime_error("DSL graph executor: missing k_norm_weight for layer " + std::to_string(layer_idx));
        }
        return block.attention.k_norm_weight.value();
    }
    if (param == "rope_freqs") {
        auto& freqs = st.rs.non_block_activations().freq_cis;
        if (freqs.Data) {
            return freqs;
        }
        return block.attention.rope_freqs;
    }
    if (param == "mlp_up_weight") return block.mlp_up_weight;
    if (param == "mlp_down_weight") return block.mlp_down_weight;

    throw std::runtime_error("DSL graph executor: unknown block param " + param);
}

template<typename Block>
Tensor& resolve_param_tensor(ExecState<Block>& st, const std::string& name) {
    if (name == "embedding" || name == "embeddings" || name == "embed_tokens") {
        if (!st.embeddings_loaded) {
            st.weights.gather_embeddings(st.comm, st.rs.MainStream);
            st.embeddings_loaded = true;
        }
        return st.weights.get_embeddings(st.rs.MainStream);
    }
    if (name == "final_norm" || name == "final_norm_weight" || name == "norm") {
        if (!st.final_norm_loaded) {
            st.weights.gather_final_norm(st.comm, st.rs.MainStream);
            st.final_norm_loaded = true;
        }
        return st.weights.get_final_norm(st.rs.MainStream);
    }
    if (name == "lm_head" || name == "lm_head_weight") {
        if (!st.lm_head_loaded) {
            st.weights.gather_lm_head(st.comm, st.rs.MainStream);
            st.lm_head_loaded = true;
        }
        return st.weights.get_lm_head(st.rs.MainStream);
    }
    int layer_idx = -1;
    std::string param;
    if (parse_block_param(name, layer_idx, param)) {
        return resolve_block_weight(st, layer_idx, param);
    }
    throw std::runtime_error("DSL graph executor: unknown param " + name);
}

// Resolve view shape from either "shape" or "shape_like" attribute
template<typename Block>
std::vector<long> resolve_view_shape(
    const Operation& op,
    const ShapeEnv& env,
    ExecState<Block>& st,
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

template<typename Block>
void free_temps(ExecState<Block>& st) {
    for (auto it = st.temps.rbegin(); it != st.temps.rend(); ++it) {
        st.rs.temp_free(*it);
    }
    st.temps.clear();
}

template<typename Block>
void reduce_loss(RunState<Block>& rs, long B, long T, NCCLCommunicator& comm) {
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

Tensor recompute_lora_rmsnorm(modules::LoRARunState& lora_rs, const Tensor& residual, const Tensor& weight,
                              float eps, int B, int T, int C, cudaStream_t stream) {
    if (!lora_rs.recompute_ln.Data || !lora_rs.recompute_rstd.Data) {
        throw std::runtime_error("DSL graph executor: LoRA recompute buffers not allocated");
    }
    rmsnorm_forward(lora_rs.recompute_ln, lora_rs.recompute_rstd,
                    residual, weight, nullptr, eps, B, T, C, stream);
    return lora_rs.recompute_ln;
}

template<typename Block>
modules::ForwardHook make_lora_forward_hook(Model<Block>& model,
                                           modules::ModularLoRAModel<Block>& lora,
                                           bool is_training) {
    lora.lora_run_state().is_training = is_training;

    return [&model, &lora](int layer_idx, cudaStream_t stream,
                                                       modules::ForwardHookPoint point, void* context) {
        (void)context;
        auto& lora_cfg = lora.lora_config();
        auto& lora_weights = lora.lora_weights();
        auto& lora_rs = lora.lora_run_state();
        const auto& cfg = model.config();
        auto& rs = model.run_state();
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = lora_cfg.rank;
        const float scaling = lora_cfg.scaling();
        const float dropout = lora_cfg.dropout;
        const bool training = lora_rs.is_training;

        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            return lora_rs.dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(lora_rs.micro_step) * 10000u;
        };

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = lora_weights.get_block(layer_idx, stream);
        const int BT = B * T;

        switch (point) {
            case modules::ForwardHookPoint::AfterQKVProjection: {
                if (lora_block.attention.q.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(0), training,
                                                             BT, C, Hq * Hs, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(1), training,
                                                             BT, C, Hkv * Hs, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(2), training,
                                                             BT, C, Hkv * Hs, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(3), training,
                                                             BT, Hq * Hs, C, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(4), training,
                                                             BT, C, D, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(5), training,
                                                             BT, C, D, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(6), training,
                                                             BT, D, C, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            default:
                break;
        }
    };
}

}  // namespace

template<typename Block>
GraphExecutorImpl<Block>::GraphExecutorImpl(const Module& module, Model<Block>& backend)
    : mModule(module),
      mBackend(backend),
      mForward(module.forward ? &module.forward.value() : nullptr),
      mBackward(nullptr) {
    init(GraphExecutorOptions{});
}

template<typename Block>
GraphExecutorImpl<Block>::GraphExecutorImpl(const Module& module, Model<Block>& backend, const GraphExecutorOptions& options)
    : mModule(module),
      mBackend(backend),
      mForward(module.forward ? &module.forward.value() : nullptr),
      mBackward(nullptr) {
    init(options);
}

template<typename Block>
void GraphExecutorImpl<Block>::set_lora_model(modules::ModularLoRAModel<Block>* lora_model) {
    mLoRA = lora_model;
}

template<typename Block>
void GraphExecutorImpl<Block>::init(const GraphExecutorOptions& options) {
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

    if (!mBackward) {
        throw std::runtime_error(
            "DSL graph executor: module missing backward graph (set auto_backward=true to derive automatically)");
    }

    // If we didn't derive backward (using explicit backward from module), use forward.save
    if (mSaveList.empty()) {
        mSaveList = mForward->save;
    }

    // Debug helper: force-save QKV tensors for recompute comparisons.
    if (env_enabled("SUROGATE_DEBUG_RECOMPUTE_COMPARE") &&
        env_enabled("SUROGATE_DEBUG_RECOMPUTE_COMPARE_ATTN_ONLY")) {
        int debug_layer = -1;
        if (const char* v = std::getenv("SUROGATE_DEBUG_RECOMPUTE_LAYER")) {
            debug_layer = std::atoi(v);
        }
        auto add_save = [&](const std::string& name) {
            if (std::find(mSaveList.begin(), mSaveList.end(), name) == mSaveList.end()) {
                mSaveList.push_back(name);
            }
        };
        if (debug_layer >= 0) {
            const std::string prefix = "blocks[" + std::to_string(debug_layer) + "].";
            add_save(prefix + "qkv");
            add_save(prefix + "qkv_rope");
        } else {
            for (int l = 0; l < mConfig.NumLayers; ++l) {
                const std::string prefix = "blocks[" + std::to_string(l) + "].";
                add_save(prefix + "qkv");
                add_save(prefix + "qkv_rope");
            }
        }
    }
}

template<typename Block>
unsigned int GraphExecutorImpl<Block>::next_rng_seed() {
    auto state = mBackend.rng_state();
    std::stringstream ss;
    ss.write(reinterpret_cast<const char*>(state.data()), static_cast<std::streamsize>(state.size()));
    std::minstd_rand rng;
    ss >> rng;
    unsigned int seed = static_cast<unsigned int>(rng());
    std::stringstream out;
    out << rng;
    auto view = out.rdbuf()->view();
    std::vector<std::byte> updated;
    updated.reserve(view.size());
    std::transform(view.begin(), view.end(), std::back_inserter(updated),
                   [](char c) { return static_cast<std::byte>(c); });
    mBackend.set_rng_state(updated);
    return seed;
}

template<typename Block>
void GraphExecutorImpl<Block>::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    mSaved.clear();

    if (micro_step == 0) {
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        weights.invalidate();
        if (rs.has_fp8_delayed_scaling()) {
            auto& fp8_state = rs.fp8_scaling_state();
            if (!mFP8ScalingInitialized) {
                fp8_state.reset(rs.MainStream);
                mFP8ScalingInitialized = true;
            }
            fp8_state.zero_recorded_amaxes(rs.MainStream);
        }
        rs.reset_moe_stats();
    }

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    modules::ForwardHook lora_hook;
    const modules::ForwardHook* hook_ptr = nullptr;
    if (mLoRA && mLoRA->lora_enabled()) {
        mLoRA->ensure_lora_state(comm, static_cast<int>(B), static_cast<int>(T));
        if (mLoRA->qlora_enabled() && micro_step == 0) {
            mLoRA->invalidate_qlora_cache();
        }
        auto& lora_rs = mLoRA->lora_run_state();
        lora_rs.micro_step = micro_step;
        lora_hook = make_lora_forward_hook(model, *mLoRA, /*is_training=*/true);
        hook_ptr = &lora_hook;
    }

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

    execute_forward_graph(B, T, comm, /*full=*/false, hook_ptr);

    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
    CUDA_CHECK(cudaEventRecord(rs.ForwardDone, rs.MainStream));
}

template<typename Block>
float GraphExecutorImpl<Block>::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    mSaved.clear();

    if (micro_step == 0) {
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        weights.invalidate();
        if (rs.has_fp8_delayed_scaling()) {
            auto& fp8_state = rs.fp8_scaling_state();
            if (!mFP8ScalingInitialized) {
                fp8_state.reset(rs.MainStream);
                mFP8ScalingInitialized = true;
            }
            fp8_state.zero_recorded_amaxes(rs.MainStream);
        }
        rs.reset_moe_stats();
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    modules::ForwardHook lora_hook;
    const modules::ForwardHook* hook_ptr = nullptr;
    if (mLoRA && mLoRA->lora_enabled()) {
        mLoRA->ensure_lora_state(comm, static_cast<int>(B), static_cast<int>(T));
        auto& lora_rs = mLoRA->lora_run_state();
        lora_rs.micro_step = micro_step;
        lora_hook = make_lora_forward_hook(model, *mLoRA, /*is_training=*/false);
        hook_ptr = &lora_hook;
    }

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

    execute_forward_graph(B, T, comm, /*full=*/false, hook_ptr);

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

    if (mLmHeadCached) {
        model.weights_manager().release_lm_head(rs.MainStream);
        mLmHeadCached = false;
    }
    rs.temp_free(rs.non_block_activations().output);

    return *rs.LossHost;
}

template<typename Block>
void GraphExecutorImpl<Block>::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& grads = model.grads();
    const auto& config = model.config();
    rs.GradAccumSteps = std::max(1, grad_accum_steps);
    rs.WorldSize = std::max(1, comm.world_size());

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    mLastInputsCpu = inputs;

    modules::BackwardHook lora_hook;
    const modules::BackwardHook* hook_ptr = nullptr;
    if (mLoRA && mLoRA->lora_enabled()) {
        auto* lora = mLoRA;
        lora->ensure_lora_state(comm, static_cast<int>(B), static_cast<int>(T));
        auto& lora_rs = lora->lora_run_state();
        lora_rs.micro_step = micro_step;
        lora_rs.is_training = true;
        lora->lora_grads().start_micro_step(rs.MainStream, micro_step, grad_accum_steps);

        lora_hook = [&, lora](int layer_idx, bool accumulate, cudaStream_t stream,
                        modules::BackwardHookPoint point, void* context) {
            (void)context;
            auto& lora_cfg = lora->lora_config();
            auto& lora_weights = lora->lora_weights();
            auto& lora_grads_mgr = lora->lora_grads();
            auto& lora_rs = lora->lora_run_state();
            const auto& cfg = model.config();
            auto& rs = model.run_state();
            const int B = (int)rs.B;
            const int T = (int)rs.T;
            const int rank = lora_cfg.rank;
            const float dropout = lora_cfg.dropout;
            const bool is_training = lora_rs.is_training;
            const int micro_step = lora_rs.micro_step;

            auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                return lora_rs.dropout_base_seed
                       + static_cast<unsigned int>(layer_idx) * 1000000u
                       + static_cast<unsigned int>(proj_type) * 100000u
                       + static_cast<unsigned int>(micro_step) * 10000u;
            };

            auto& a = rs.simplified_acts(layer_idx);
            auto& da = rs.simplified_grads(layer_idx);
            auto& lora_block = lora_weights.get_block(layer_idx, stream);

            switch (point) {
                case modules::BackwardHookPoint::AfterQKVBackward: {
                    const int C = (int)cfg.HiddenSize;
                    const int Hq = (int)cfg.NumQueryHeads;
                    const int Hkv = (int)cfg.NumKeyValHeads;
                    const int Hs = (int)cfg.head_size();

                    bool lora_accum = false;
                    auto& lora_grads = lora_grads_mgr.get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;

                    Tensor ln1_input;
                    if (rs.config().recompute_lora) {
                        Tensor& residual = rs.get_residual(layer_idx, stream);
                        auto& block_weights = model.weights_manager().get_block(layer_idx, stream);
                        ln1_input = recompute_lora_rmsnorm(lora_rs, residual, block_weights.ln1.weight,
                                                           cfg.RmsNormEps, B, T, C, stream);
                    } else {
                        ln1_input = a.ln1;
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
                        da.d_ln1,
                        da.d_qkv,
                        ln1_input,
                        lora_q, lora_k, lora_v,
                        lora_cfg.scaling(),
                        dropout,
                        get_dropout_seed(0), get_dropout_seed(1), get_dropout_seed(2), is_training,
                        B * T,
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
                        stream);

                    lora_grads_mgr.notify_block(layer_idx, stream, comm);
                } break;
                case modules::BackwardHookPoint::AfterAttnOutBackward: {
                    const int C = (int)cfg.HiddenSize;
                    const int Hq = (int)cfg.NumQueryHeads;
                    const int Hs = (int)cfg.head_size();

                    if (!lora_block.attention.o.has_value()) {
                        break;
                    }

                    bool lora_accum = false;
                    auto& lora_grads = lora_grads_mgr.get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;
                    if (!lora_grads.attention.o.has_value()) {
                        break;
                    }

                    const unsigned int dropout_seed =
                        lora_rs.dropout_base_seed
                        + static_cast<unsigned int>(layer_idx) * 1000000u
                        + 3u * 100000u
                        + static_cast<unsigned int>(micro_step) * 10000u;

                    Tensor x = a.att;
                    Tensor dL_dy = da.d_res_att;

                    modules::detail::backward_lora_layer(
                        lora_grads.attention.o->A, lora_grads.attention.o->B,
                        da.d_att,
                        dL_dy, 0,
                        x,
                        lora_block.attention.o->A, lora_block.attention.o->B,
                        lora_cfg.scaling(),
                        dropout, dropout_seed, is_training,
                        lora_rs.intermediate, lora_rs.slice,
                        B * T, Hq * Hs, C, rank, lora_accum,
                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                } break;
                case modules::BackwardHookPoint::AfterMLPUpBackward: {
                    const int C = (int)cfg.HiddenSize;
                    const int D = (int)cfg.IntermediateSize;

                    bool lora_accum = false;
                    auto& lora_grads = lora_grads_mgr.get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;

                    Tensor ln2_input;
                    if (rs.config().recompute_lora) {
                        auto& block_weights = model.weights_manager().get_block(layer_idx, stream);
                        ln2_input = recompute_lora_rmsnorm(lora_rs, a.residual_att, block_weights.ln2.weight,
                                                           cfg.RmsNormEps, B, T, C, stream);
                    } else {
                        ln2_input = a.ln2;
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

                    modules::detail::backward_lora_mlp_up_gate_fused(
                        dA_up, dB_up,
                        dA_gate, dB_gate,
                        da.d_ln2,
                        da.d_mlp_up,
                        ln2_input,
                        lora_up, lora_gate,
                        lora_cfg.scaling(),
                        dropout,
                        get_dropout_seed(4), get_dropout_seed(5), is_training,
                        B * T,
                        C,
                        D,
                        rank,
                        lora_accum,
                        lora_rs.intermediate,
                        lora_rs.intermediate2,
                        lora_rs.slice,
                        rs.CublasLtHandle,
                        rs.CuBlasWorkspace,
                        stream);
                } break;
                case modules::BackwardHookPoint::AfterMLPDownBackward: {
                    const int C = (int)cfg.HiddenSize;
                    const int D = (int)cfg.IntermediateSize;

                    if (!lora_block.mlp.down.has_value()) {
                        break;
                    }

                    bool lora_accum = false;
                    auto& lora_grads = lora_grads_mgr.get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;
                    if (!lora_grads.mlp.down.has_value()) {
                        break;
                    }

                    const unsigned int dropout_seed =
                        lora_rs.dropout_base_seed
                        + static_cast<unsigned int>(layer_idx) * 1000000u
                        + 6u * 100000u
                        + static_cast<unsigned int>(micro_step) * 10000u;

                    Tensor x = a.swiglu;
                    // Prefer the direct upstream gradient from MLP down (d_mlp_down). In recompute
                    // modes, d_res_ffn may be stale/aliased by stack temporaries before the hook runs.
                    Tensor dL_dy = da.d_mlp_down.Data ? da.d_mlp_down : da.d_res_ffn;

                    modules::detail::backward_lora_layer(
                        lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                        da.d_swiglu,
                        dL_dy, 0,
                        x,
                        lora_block.mlp.down->A, lora_block.mlp.down->B,
                        lora_cfg.scaling(),
                        dropout, dropout_seed, is_training,
                        lora_rs.intermediate, lora_rs.slice,
                        B * T, D, C, rank, lora_accum,
                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                } break;
                default:
                    break;
            }
        };

        hook_ptr = &lora_hook;
    }

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

    run_classifier(B, T, comm, grad_accum_steps, micro_step, /*compute_accuracy=*/false);

    const bool last_step = micro_step == grad_accum_steps - 1;
    if (last_step) {
        reduce_loss(rs, B, T, comm);
        comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    }

    execute_backward_graph(B, T, comm, grad_accum_steps, micro_step, hook_ptr);

    grads.end_micro_step(rs.MainStream, comm);
    if (mLoRA && mLoRA->lora_enabled()) {
        mLoRA->lora_grads().end_micro_step(rs.MainStream, comm);
    }
    CUDA_CHECK(cudaEventRecord(rs.BackwardDone, rs.MainStream));
    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
}

template<typename Block>
void GraphExecutorImpl<Block>::run_classifier(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step, bool compute_accuracy) {
    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    const auto& options = model.options();
    const auto& config = model.config();
    if (options.lmhead_chunks != 1) {
        throw std::runtime_error("DSL graph executor: lmhead_chunks > 1 not supported yet");
    }

    const size_t V = config.VocabSize;
    const size_t Vp = config.VocabSize;
    const float d_loss = compute_accuracy ? 1.0f : (1.0f / static_cast<float>(B * T * grad_accum_steps));

    if (!mLmHeadCached) {
        weights.gather_lm_head(comm, rs.side_stream());
        mLmHeadCached = true;
    }
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
    matmul(logits, weights.get_lm_head(rs.MainStream), lnf_flat,
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

template<typename Block>
void GraphExecutorImpl<Block>::execute_forward_graph(long B, long T, NCCLCommunicator& comm, bool full,
                                          const modules::ForwardHook* hook) {
    if (!mForward) {
        throw std::runtime_error("DSL graph executor: missing forward graph");
    }

    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    const auto& config = model.config();
    (void)hook;

    ExecState<Block> st{
        .rs = rs,
        .weights = weights,
        .config = config,
        .comm = comm,
        .B = B,
        .T = T,
        .shape_env = make_shape_env(mModule, B, T),
    };
    augment_shape_env(st.shape_env, mModule.config);

    // Bind known inputs.
    st.tensors.emplace("token_ids", rs.Inputs);
    st.tensors.emplace("position_ids", rs.PositionIDs);
    st.tensors.emplace("x0", rs.non_block_activations().encoded);

    if (full) {
        weights.gather_lm_head(comm, rs.side_stream());
        st.tensors.emplace("lm_head", weights.get_lm_head(rs.MainStream));
    }

    std::vector<char> required;
    if (!full) {
        std::vector<std::string> needed = mSaveList;
        for (const auto& kv : mForward->outputs) {
            needed.push_back(kv.first);
        }
        required = compute_required_ops(*mForward, needed);
    }

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
        // Note: cudnn_workspace is persistently allocated, don't clear
        layer_checkpoints.erase(layer_idx);
        layer_temp_marks.erase(layer_idx);
    };

    for (std::size_t idx = 0; idx < mForward->operations.size(); ++idx) {
        if (!full && !required[idx]) {
            continue;
        }
        const auto& op = mForward->operations[idx];
        const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
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
            EMMTranspose mode = parse_transpose(op.attrs);
            int M = 0, N = 0, K = 0;
            matmul_dims(a, b, mode, M, N, K);
            std::vector<long> shape{M, N};
            Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, shape);
            std::optional<Tensor> bias;
            if (op_type == "matmul_bias" && op.inputs.size() > 2 && !op.inputs.at(2).empty()) {
                bias = get_tensor(st, op.inputs.at(2), mSaved);
            }
            // DSL matmul uses row-major semantics; map to column-major backend by swapping A/B,
            // swapping M/N, and swapping transpose flags.
            EMMTranspose mode_col = swap_transpose(mode);
            matmul(out, b, a, bias, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   N, M, K, mode_col, false, rs.MainStream);
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

            // Match modular path: Q/K RMSNorm then RoPE (both in-place on qkv).
            Tensor qkv_view = (qkv.Rank == 4)
                ? view_tensor(qkv, {B, T, qkv_channels})
                : qkv;
            const int q_rows = Hq * Hs;

            qkv_head_rmsnorm_forward(qkv_view, q_rstd, q_norm,
                                     static_cast<float>(eps),
                                     static_cast<int>(B), static_cast<int>(T),
                                     qkv_channels, Hq, Hs, 0, rs.MainStream);
            qkv_head_rmsnorm_forward(qkv_view, k_rstd, k_norm,
                                     static_cast<float>(eps),
                                     static_cast<int>(B), static_cast<int>(T),
                                     qkv_channels, Hkv, Hs, q_rows, rs.MainStream);

            int rotary_dim = Hs;
            if (auto* rd_attr = find_attr(op.attrs, "rotary_dim")) {
                if (auto v = attr_int(*rd_attr)) {
                    rotary_dim = static_cast<int>(*v);
                } else if (auto s = attr_string(*rd_attr)) {
                    rotary_dim = static_cast<int>(resolve_dim(Dim::symbolic(*s), st.shape_env));
                }
            }
            rope_forward(qkv, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                         static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
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
    if (full) {
        weights.release_lm_head(rs.MainStream);
    }
    if (st.block_loaded) {
        weights.release_block(st.current_block_layer, rs.MainStream);
    }
    if (st.embeddings_loaded) {
        weights.release_embeddings(rs.MainStream);
    }
    if (st.final_norm_loaded) {
        weights.release_final_norm(rs.MainStream);
    }
    if (st.lm_head_loaded) {
        weights.release_lm_head(rs.MainStream);
    }
}

template<typename Block>
void GraphExecutorImpl<Block>::execute_backward_graph(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step,
                                           const modules::BackwardHook* hook) {
    if (!mBackward) {
        throw std::runtime_error("DSL graph executor: missing backward graph");
    }

    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    auto& grads = model.grads();
    const auto& config = model.config();
    const bool lora_only = rs.is_lora_only_mode();
    (void)hook;

    ExecState<Block> st{
        .rs = rs,
        .weights = weights,
        .config = config,
        .comm = comm,
        .B = B,
        .T = T,
        .shape_env = make_shape_env(mModule, B, T),
    };
    augment_shape_env(st.shape_env, mModule.config);

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

    // Bind gradient outputs for parameters.
    std::unordered_set<std::string> accumulate_tensors;
    std::unordered_set<int> block_layers_used;
    bool embeddings_used = false;
    bool final_norm_used = false;
    bool lm_head_used = false;

    auto bind_block_grad = [&](int layer_idx, const std::string& param) -> Tensor* {
        auto it = st.block_grads.find(layer_idx);
        if (it == st.block_grads.end()) {
            bool accumulate = false;
            auto& block = grads.get_block_full(layer_idx, rs.MainStream, comm, accumulate);
            st.block_grads.emplace(layer_idx, &block);
            st.block_grad_accumulate.emplace(layer_idx, accumulate);
            block_layers_used.insert(layer_idx);
            it = st.block_grads.find(layer_idx);
        }
        auto* block_grads = it->second;
        if (!block_grads) return nullptr;

        if (param == "ln1_weight") {
            return get_ln1_weight_grad(*block_grads);
        }
        if (param == "ln2_weight") {
            return get_ln2_weight_grad(*block_grads);
        }
        if (param == "qkv_weight") return &block_grads->attention_grads.d_qkv_weight;
        if (param == "qkv_bias") {
            if (block_grads->attention_grads.d_qkv_bias.has_value()) {
                return &block_grads->attention_grads.d_qkv_bias.value();
            }
            return nullptr;
        }
        if (param == "out_weight") return &block_grads->attention_grads.d_out_weight;
        if (param == "q_norm_weight") {
            if (block_grads->attention_grads.d_q_norm_weight.has_value()) {
                return &block_grads->attention_grads.d_q_norm_weight.value();
            }
            return nullptr;
        }
        if (param == "k_norm_weight") {
            if (block_grads->attention_grads.d_k_norm_weight.has_value()) {
                return &block_grads->attention_grads.d_k_norm_weight.value();
            }
            return nullptr;
        }
        if (param == "mlp_up_weight") return &block_grads->d_mlp_up_weight;
        if (param == "mlp_down_weight") return &block_grads->d_mlp_down_weight;
        return nullptr;
    };

    auto bind_param_grad = [&](const std::string& param_name) {
        if (param_name.find("rope_freqs") != std::string::npos) {
            return;
        }
        std::string grad_name = "d_" + param_name;
        bool accumulate = false;
        Tensor* grad_tensor = nullptr;

        int layer_idx = -1;
        std::string param;
        if (parse_block_param(param_name, layer_idx, param)) {
            grad_tensor = bind_block_grad(layer_idx, param);
            auto acc_it = st.block_grad_accumulate.find(layer_idx);
            if (acc_it != st.block_grad_accumulate.end()) {
                accumulate = acc_it->second;
            }
        } else if (param_name == "embedding" || param_name == "embeddings" || param_name == "embed_tokens") {
            if (!lora_only) {
                grad_tensor = &grads.get_embeddings_full(rs.MainStream, comm, accumulate);
                embeddings_used = true;
            }
        } else if (param_name == "final_norm" || param_name == "final_norm_weight" || param_name == "norm") {
            grad_tensor = &grads.get_final_norm_full(rs.MainStream, comm, accumulate);
            final_norm_used = true;
        } else if (param_name == "lm_head" || param_name == "lm_head_weight") {
            if (!lora_only) {
                grad_tensor = &grads.get_lm_head_full(rs.MainStream, comm, accumulate);
                lm_head_used = true;
            }
        }

        if (grad_tensor) {
            st.tensors.emplace(grad_name, *grad_tensor);
            if (accumulate) {
                accumulate_tensors.insert(grad_name);
            }
        }
    };

    for (const auto& kv : mForward->params) {
        bind_param_grad(kv.first);
    }

    if (mLmHeadCached) {
        st.tensors.emplace("lm_head", weights.get_lm_head(rs.MainStream));
    }

    std::unordered_map<int, DeviceMemoryStack::Checkpoint> layer_checkpoints;
    std::unordered_map<int, std::size_t> layer_temp_marks;
    auto extract_layer = [&](const std::string& name, int& layer_idx) -> bool {
        std::string_view view{name};
        if (starts_with(view, "d_")) {
            view = view.substr(2);
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
        // Note: cudnn_workspace is persistently allocated, don't clear
        layer_checkpoints.erase(layer_idx);
        layer_temp_marks.erase(layer_idx);
    };

    for (const auto& op : mBackward->operations) {
        const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
        maybe_start_layer(op);

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
                dA_ptr = &ensure_tensor(st, dA_name, a.DType, {a.Sizes[0], a.Sizes[1]});
            }
            if (!dB_name.empty()) {
                dB_ptr = &ensure_tensor(st, dB_name, b.DType, {b.Sizes[0], b.Sizes[1]});
            }
            if (!dA_ptr && !dB_ptr) {
                continue;
            }

            EMMTranspose mode = parse_transpose(op.attrs);
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
            if (dB_ptr) {
                bool do_accumulate = accumulate_tensors.count(dB_name) > 0;
                int M = 0, N = 0, K = 0;
                matmul_dims(d_out, a, mode_dB, M, N, K);
                EMMTranspose mode_col = swap_transpose(mode_dB);
                matmul(*dB_ptr, a, d_out, std::nullopt, nullptr, nullptr,
                       rs.CublasLtHandle, rs.CuBlasWorkspace,
                       N, M, K, mode_col, do_accumulate, rs.MainStream);
            }
            continue;
        }

        if (op_type == "matmul" || op_type == "matmul_bias") {
            if (op.outputs.at(0).empty()) {
                continue;
            }
            Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
            EMMTranspose mode = parse_transpose(op.attrs);
            int M = 0, N = 0, K = 0;
            matmul_dims(a, b, mode, M, N, K);
            std::vector<long> shape{M, N};
            Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, shape);
            bool do_accumulate = accumulate_tensors.count(op.outputs.at(0)) > 0;
            std::optional<Tensor> bias;
            if (op_type == "matmul_bias" && op.inputs.size() > 2 && !op.inputs.at(2).empty()) {
                bias = get_tensor(st, op.inputs.at(2), mSaved);
            }
            // DSL matmul uses row-major semantics; map to column-major backend by swapping A/B,
            // swapping M/N, and swapping transpose flags.
            EMMTranspose mode_col = swap_transpose(mode);
            matmul(out, b, a, bias, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   N, M, K, mode_col, do_accumulate, rs.MainStream);
            continue;
        }

        if (op_type == "bias_add_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            if (!op.outputs.at(0).empty()) {
                st.tensors.emplace(op.outputs.at(0), d_out);
            }
            if (op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
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
                Tensor& d_bias = ensure_tensor(st, op.outputs.at(1), d_out.DType, {static_cast<long>(OC)});
                Tensor& scratch = rs.scratch().matmul_bias_scratch;
                backward_bias(d_bias, d_out, nullptr, nullptr, scratch,
                              Bv, Tv, OC, rs.DeviceProp, rs.MainStream);
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
            swiglu_backward(d_inp, d_out, inp, nullptr,
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
            rope_backward(d_inp, d_out, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
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

            // Undo RoPE on gradients and activations (in-place) before QK RMSNorm backward.
            rope_backward(d_qkv, d_qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                          static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
            rope_backward(qkv, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                          static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);

            // Weight grads
            if (op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
                Tensor& d_q_norm = ensure_tensor(st, op.outputs.at(1), q_norm.DType, {Hs});
                const bool acc = accumulate_tensors.count(op.outputs.at(1)) > 0;
                qkv_head_rmsnorm_backward_dweight(
                    d_q_norm, d_qkv_view, qkv_view, q_norm,
                    static_cast<int>(B), static_cast<int>(T), qkv_channels, Hq, Hs, 0,
                    acc, rs.MainStream);
            }
            if (op.outputs.size() > 2 && !op.outputs.at(2).empty()) {
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
            if (!op.outputs.at(0).empty()) {
                st.tensors.emplace(op.outputs.at(0), d_qkv);
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
            if (!op.outputs.at(0).empty()) {
                st.tensors[op.outputs.at(0)] = d_input;
            }

            Tensor* d_residual = d_residual_next;
            if (!d_residual) {
                std::string zero_name = op.id + "_dres_zero";
                Tensor& d_residual_zero = ensure_tensor(st, zero_name, d_y.DType, {B, T, config.HiddenSize});
                fill_zero(d_residual_zero, rs.MainStream);
                d_residual = &d_residual_zero;
            }

            Tensor* d_weight = nullptr;
            bool skip_weight_grad = true;
            if (op.outputs.size() > 2 && !op.outputs.at(2).empty()) {
                d_weight = &ensure_tensor(st, op.outputs.at(2), weight.DType, {config.HiddenSize});
                skip_weight_grad = false;
            }
            Tensor dummy_weight{};
            if (!d_weight) {
                dummy_weight = rs.temp_alloc(weight.DType, {config.HiddenSize});
                st.temps.push_back(dummy_weight);
                d_weight = &dummy_weight;
            }

            const bool skip_weight = skip_weight_grad || rs.is_lora_only_mode();
            rmsnorm_backward(d_input, *d_weight, rs.scratch().rmsnorm_scratch,
                             *d_residual, d_y, residual_out, weight,
                             get_tensor(st, op.inputs.at(4), mSaved),
                             rs.has_grad_quants() ? rs.simplified_quant_grads().d_res_ffn.abs_max() : nullptr,
                             static_cast<int>(B), static_cast<int>(T), config.HiddenSize,
                             rs.DeviceProp, rs.MainStream,
                             skip_weight);
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(op.inputs.at(3), layer_idx, field) && field == "ln1_weight") {
                finish_layer(layer_idx);
            }
            continue;
        }

        if (op_type == "embedding_backward" || op.name == "embedding_backward") {
            if (lora_only) {
                continue;
            }
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            bool accumulate = false;
            Tensor& d_emb = grads.get_embeddings_full(rs.MainStream, comm, accumulate);
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
            grads.notify_embeddings(rs.MainStream, comm);
            continue;
        }

        throw std::runtime_error("DSL graph executor: unsupported backward op " + op.name);
    }

    free_temps(st);
    for (int layer_idx : block_layers_used) {
        grads.notify_block(layer_idx, rs.MainStream, comm);
    }
    if (embeddings_used) {
        grads.notify_embeddings(rs.MainStream, comm);
    }
    if (final_norm_used) {
        grads.notify_final_norm(rs.MainStream, comm);
    }
    if (lm_head_used) {
        grads.notify_lm_head(rs.MainStream, comm);
    }
    if (st.block_loaded) {
        weights.release_block(st.current_block_layer, rs.MainStream);
    }
    if (st.embeddings_loaded) {
        weights.release_embeddings(rs.MainStream);
    }
    if (st.final_norm_loaded) {
        weights.release_final_norm(rs.MainStream);
    }
    if (st.lm_head_loaded) {
        weights.release_lm_head(rs.MainStream);
    }
    if (mLmHeadCached) {
        weights.release_lm_head(rs.MainStream);
        mLmHeadCached = false;
    }
    rs.temp_free(rs.non_block_activations().output);
}

}  // namespace dsl
