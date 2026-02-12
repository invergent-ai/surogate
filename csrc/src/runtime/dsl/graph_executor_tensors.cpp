// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tensor resolution functions for DSL Graph executor.

#include "runtime/dsl/graph_executor_tensors.h"

#include <stdexcept>
#include <string>

#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "runtime/dsl/ir.h"
#include "utilities/tensor.h"

namespace dsl {

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
    if (field == "ln1_flat") return map_tensor(acts.ln1);
    if (field == "ln2_flat") return map_tensor(acts.ln2);
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
    if (field == "att" || field == "att_flat") return map_tensor(acts.att);
    if (field == "att_out" || field == "att_out_flat") return map_tensor(acts.att_out);
    if (field == "res_att") return map_tensor(acts.residual_att);
    if (field == "mlp_up" || field == "mlp_up_flat") return map_tensor(acts.mlp_up);
    if (field == "swiglu") return map_tensor(acts.swiglu);
    if (field == "mlp_down" || field == "mlp_down_flat") return map_tensor(acts.mlp_down);
    if (field == "res_ffn" || field == "res_in") {
        Tensor& res = st.rs.get_residual(layer_idx, st.rs.MainStream);
        return map_tensor(res);
    }

    return nullptr;
}

Tensor* block_activation_base_ptr(DslRunState& rs, int layer_idx, const std::string& field) {
    auto& acts = rs.simplified_acts(layer_idx);
    if (field == "ln1" || field == "ln1_flat") return &acts.ln1;
    if (field == "ln1_rstd") return &acts.ln1_rstd;
    if (field == "ln2" || field == "ln2_flat") return &acts.ln2;
    if (field == "ln2_rstd") return &acts.ln2_rstd;
    if (field == "q_rstd") return &acts.q_rstd;
    if (field == "k_rstd") return &acts.k_rstd;
    if (field == "qkv" || field == "qkv_flat" || field == "qkv_biased") return &acts.qkv;
    if (field == "qkv_rope") {
        return acts.qkv_rope.Data ? &acts.qkv_rope : &acts.qkv;
    }
    if (field == "lse") return &acts.lse;
    if (field == "att" || field == "att_flat") return &acts.att;
    if (field == "att_out" || field == "att_out_flat") return &acts.att_out;
    if (field == "res_att") return &acts.residual_att;
    if (field == "mlp_up" || field == "mlp_up_flat") return &acts.mlp_up;
    if (field == "swiglu") return &acts.swiglu;
    if (field == "mlp_down" || field == "mlp_down_flat") return &acts.mlp_down;
    if (field == "res_ffn" || field == "res_in") {
        return &rs.get_residual(layer_idx, rs.MainStream);
    }
    return nullptr;
}

Tensor* resolve_recomputed_block_tensor(ExecState& st, const std::string& name) {
    if (!st.recomputed_layers) {
        return nullptr;
    }
    int layer_idx = -1;
    std::string field;
    if (!parse_block_param(name, layer_idx, field)) {
        return nullptr;
    }
    if (layer_idx < 0 || layer_idx >= static_cast<int>(st.config.NumLayers)) {
        return nullptr;
    }
    if (st.recomputed_layers->empty() || (*st.recomputed_layers)[layer_idx] == 0) {
        return nullptr;
    }
    std::vector<long> shape;
    if (!infer_block_tensor_shape(st, name, shape)) {
        return nullptr;
    }
    Tensor* base = block_activation_base_ptr(st.rs, layer_idx, field);
    if (!base || !base->Data) {
        return nullptr;
    }
    return resolve_block_activation_tensor(st, name, base->DType, shape);
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
    if (field == "ln1" || field == "ln1_flat") return map_tensor(acts.ln1);
    if (field == "ln1_rstd") return map_tensor(acts.ln1_rstd);
    if (field == "ln2" || field == "ln2_flat") return map_tensor(acts.ln2);
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
    if (field == "att" || field == "att_flat") return map_tensor(acts.att);
    if (field == "att_out" || field == "att_out_flat") return map_tensor(acts.att_out);
    if (field == "res_att") return map_tensor(acts.residual_att);
    if (field == "mlp_up" || field == "mlp_up_flat") return map_tensor(acts.mlp_up);
    if (field == "swiglu") return map_tensor(acts.swiglu);
    if (field == "mlp_down" || field == "mlp_down_flat") return map_tensor(acts.mlp_down);
    if (field == "res_ffn" || field == "res_in") {
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
    if (field == "ln1" || field == "ln1_flat") return map_tensor(grads.d_ln1);
    if (field == "qkv" || field == "qkv_rope" || field == "qkv_flat" || field == "qkv_biased") {
        return map_tensor(grads.d_qkv);
    }
    if (field == "att" || field == "att_flat") return map_tensor(grads.d_att);
    if (field == "att_out" || field == "att_out_flat") {
        // Separate storage: d_att_out is the gradient of the O-proj output,
        // d_res_att is the gradient of the residual input to the block.
        // In standard transformers (res_att = x + att_out) they're equal;
        // in hybrid architectures (Nemotron-H) they're independent.
        return map_tensor(grads.d_att_out);
    }
    if (field == "swiglu") {
        return map_tensor(grads.d_swiglu);
    }
    if (field == "mlp_up") {
        return map_tensor(grads.d_mlp_up);
    }
    if (field == "mlp_down" || field == "mlp_down_flat") {
        return map_tensor(grads.d_mlp_down);
    }
    if (field == "ln2" || field == "ln2_flat") return map_tensor(grads.d_ln2);
    if (field == "res_att") return map_tensor(grads.d_res_att);
    if (field == "res_ffn" || field == "res_in") return map_tensor(grads.d_res_ffn);

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
    if (Tensor* recomputed = resolve_recomputed_block_tensor(st, name)) {
        return *recomputed;
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
        if (it->second.Data != nullptr) {
            return &it->second;
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

}  // namespace dsl
