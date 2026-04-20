// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/tensor_slot_dispatch.h"

#include <string>
#include <string_view>
#include <utility>

#include "runtime/dsl/dsl_run_state.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "utilities/tensor.h"

namespace dsl {

// Forward-declared to avoid pulling graph_compiler.h / graph_executor_utils.h
// into this TU — both headers already depend on the slot-registry surface.
std::string strip_ssa_suffix(const std::string& field);
bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name);

TensorSlot
resolve_block_slot(const std::string& name, int* out_layer_idx, bool* out_is_flat_view, std::string* out_base_field) {
    int layer_idx = -1;
    std::string field;
    if (!parse_block_param(name, layer_idx, field)) {
        return TensorSlot::Mapped;
    }
    std::string base_field = strip_ssa_suffix(field);
    const bool is_flat = (base_field.size() >= 5 && base_field.compare(base_field.size() - 5, 5, "_flat") == 0);
    const TensorSlot slot = builtin_slot_from_name(base_field);

    if (out_layer_idx) *out_layer_idx = layer_idx;
    if (out_is_flat_view) *out_is_flat_view = is_flat;
    if (out_base_field) *out_base_field = std::move(base_field);
    return slot;
}

Tensor* block_activation_ptr(DslRunState& rs, int layer_idx, TensorSlot slot) {
    if (layer_idx < 0) {
        return nullptr;
    }
    auto& acts = rs.simplified_acts(layer_idx);
    switch (slot) {
        case TensorSlot::BlockLN1: return &acts.ln1;
        case TensorSlot::BlockLN1RSTD: return &acts.ln1_rstd;
        case TensorSlot::BlockLN2: return &acts.ln2;
        case TensorSlot::BlockLN2RSTD: return &acts.ln2_rstd;
        case TensorSlot::BlockQRSTD: return &acts.q_rstd;
        case TensorSlot::BlockKRSTD: return &acts.k_rstd;
        case TensorSlot::BlockQKV: return &acts.qkv;
        case TensorSlot::BlockQKVRoPE:
            // Non-QK-norm attention applies RoPE in-place on acts.qkv, leaving
            // acts.qkv_rope unallocated. Fall back to acts.qkv so the saved
            // reference still resolves to the post-RoPE tensor.
            return acts.qkv_rope.Data ? &acts.qkv_rope : &acts.qkv;
        case TensorSlot::BlockLSE: return &acts.lse;
        case TensorSlot::BlockAtt: return &acts.att;
        case TensorSlot::BlockAttOut: return &acts.att_out;
        case TensorSlot::BlockResidualAtt: return &acts.residual_att;
        case TensorSlot::BlockMLPUp: return &acts.mlp_up;
        case TensorSlot::BlockSwiGLU: return &acts.swiglu;
        case TensorSlot::BlockMLPDown: return &acts.mlp_down;
        case TensorSlot::BlockHOut: return &acts.h_out;
        case TensorSlot::BlockResidualFFN: return &rs.get_residual(layer_idx, rs.MainStream);
        // MoE activations — populated only when NumExperts > 0. When the
        // model has no experts these return pointers to empty Tensors (their
        // `.Data` is nullptr), which callers should treat as "unavailable"
        // the same way they already do for any lazily-allocated slot.
        case TensorSlot::BlockRouterLogits: return &acts.router_logits;
        case TensorSlot::BlockRouterProbs: return &acts.router_probs;
        case TensorSlot::BlockRoutingWeights: return &acts.routing_weights;
        case TensorSlot::BlockRoutingIndices: return &acts.routing_indices;
        case TensorSlot::BlockPermutedInput: return &acts.permuted_input;
        case TensorSlot::BlockScatterIndices: return &acts.scatter_indices;
        case TensorSlot::BlockExpertGateUp: return &acts.expert_gate_up;
        case TensorSlot::BlockExpertAct: return &acts.expert_act;
        case TensorSlot::BlockExpertDown: return &acts.expert_down;
        case TensorSlot::BlockMoeOut: return &acts.moe_out;
        default: return nullptr;
    }
}

Tensor* block_gradient_ptr(DslRunState& rs, int layer_idx, TensorSlot slot) {
    if (layer_idx < 0) {
        return nullptr;
    }
    auto& grads = rs.simplified_grads(layer_idx);
    switch (slot) {
        case TensorSlot::BlockDLN1: return &grads.d_ln1;
        case TensorSlot::BlockDLN2: return &grads.d_ln2;
        case TensorSlot::BlockDQKV: return &grads.d_qkv;
        case TensorSlot::BlockDAtt: return &grads.d_att;
        case TensorSlot::BlockDAttOut: return &grads.d_att_out;
        case TensorSlot::BlockDMLPUp: return &grads.d_mlp_up;
        case TensorSlot::BlockDSwiGLU: return &grads.d_swiglu;
        case TensorSlot::BlockDMLPDown: return &grads.d_mlp_down;
        case TensorSlot::BlockDHOut: return &grads.d_h_out;
        case TensorSlot::BlockDResAtt: return &grads.d_res_att;
        case TensorSlot::BlockDResFFN: return &grads.d_res_ffn;
        default: return nullptr;
    }
}

Tensor* global_activation_ptr(DslRunState& rs, TensorSlot slot) {
    switch (slot) {
        case TensorSlot::TokenIDs: return &rs.Inputs;
        case TensorSlot::PositionIDs: return &rs.PositionIDs;
        case TensorSlot::Encoded: return &rs.non_block_activations().encoded;
        case TensorSlot::LNFinal: return &rs.non_block_activations().ln_final;
        case TensorSlot::LNFinalRSTD: return &rs.non_block_activations().ln_final_rstd;
        case TensorSlot::FinalResidual: return &rs.get_final_residual();
        // FreqCis intentionally returns nullptr — callers must pass the full
        // name to `rs.rope_freqs(name)` to disambiguate per-block RoPE variants
        // (e.g., Gemma4 has both sliding and full RoPE tables).
        case TensorSlot::FreqCis: return nullptr;
        default: return nullptr;
    }
}

}  // namespace dsl
