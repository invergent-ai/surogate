// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tensor resolution functions for DSL Graph executor.

#include "runtime/executor/graph_executor_tensors.h"

#include <stdexcept>
#include <string>

#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/tensor_role.h"
#include "runtime/dsl/tensor_slot_dispatch.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/dsl/ir.h"
#include "utilities/tensor.h"

namespace dsl {

Tensor* resolve_block_activation_tensor(ExecState& st,
                                        const std::string& name,
                                        ETensorDType dtype,
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
        Tensor view = base;
        if (!shape.empty()) {
            view = view_for_shape(base, shape, name);
        }
        auto [it, inserted] = st.tensors.emplace(name, view);
        if (!inserted) {
            it->second = view;
        }
        return &it->second;
    };

    // Dispatch via the shared slot helpers — every field alias (`ln1_flat`,
    // `qkv_biased`, etc.) is in the name→slot table, and block_activation_ptr
    // handles the qkv_rope fallback internally.
    const TensorSlot slot = builtin_slot_from_name(field);
    if (Tensor* t = block_activation_ptr(st.rs, layer_idx, slot)) {
        return map_tensor(*t);
    }
    return nullptr;
}

Tensor* block_activation_base_ptr(DslRunState& rs, int layer_idx, const std::string& field) {
    return block_activation_ptr(rs, layer_idx, builtin_slot_from_name(field));
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

    // Same dispatch as resolve_block_activation_tensor but without shape/dtype
    // view wrapping — callers here pre-infer the shape and bind directly.
    if (Tensor* t = block_activation_ptr(st.rs, layer_idx, builtin_slot_from_name(field))) {
        return map_tensor(*t);
    }
    return nullptr;
}

Tensor* resolve_block_gradient_tensor(ExecState& st,
                                      const std::string& name,
                                      ETensorDType dtype,
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
        Tensor view = base;
        if (!shape.empty()) {
            view = view_for_shape(base, shape, name);
        }
        auto [it, inserted] = st.tensors.emplace(name, view);
        if (!inserted) {
            it->second = view;
        }
        return &it->second;
    };

    // Look up the gradient slot. The name→slot table already encodes the
    // activation→gradient mapping (qkv_rope → BlockDQKV, att_flat → BlockDAtt,
    // etc.), so the same field name as the activation side maps to the
    // correct BlockD* slot. Separate d_att_out / d_res_att storage is
    // handled by block_gradient_ptr — in standard transformers the two
    // are aliased, in hybrid they're independent buffers.
    const std::string grad_field = "d_" + field;
    if (Tensor* t = block_gradient_ptr(st.rs, layer_idx, builtin_slot_from_name(grad_field))) {
        return map_tensor(*t);
    }
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
        shape.assign(it_shape->second.Sizes.begin(), it_shape->second.Sizes.begin() + it_shape->second.Rank);
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
    Tensor t = st.rs.temp_alloc(dtype, shape, name.c_str());
    st.temps.push_back(t);
    auto [ins_it, inserted] = st.tensors.emplace(name, t);
    (void)inserted;
    return ins_it->second;
}

Tensor& resolve_param_tensor(ExecState& st, const std::string& name) {
    if (tensor_role_is_rope_name(name)) {
        auto& freqs = st.rs.rope_freqs(name);
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
                throw std::runtime_error("DSL graph executor: recompute_block active but saved tensor not mappable: " +
                                         key);
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
std::vector<long> resolve_view_shape(const Operation& op,
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
