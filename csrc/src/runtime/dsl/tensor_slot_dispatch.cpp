// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/tensor_slot_dispatch.h"

#include <array>
#include <cstddef>
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

namespace {

// Member-pointer tables: one entry per TensorSlot enum value. `nullptr` means
// the slot isn't a block-activation / block-gradient field (globals, params,
// MoE side-channels, etc.). Member pointers are compile-time constants and
// survive struct copies — the table doesn't need re-initialization when
// DslRunState resizes or copies its per-layer vectors.
//
// kSlotCount is sized to fit every enum value; must be updated if the enum
// grows. The static_assert below catches mismatches at compile time.
constexpr std::size_t kSlotCount = static_cast<std::size_t>(TensorSlot::Mapped) + 1;

using ActMemberPtr = Tensor modules::SimplifiedLayerActivations::*;
using GradMemberPtr = Tensor modules::SimplifiedLayerGradients::*;

constexpr auto make_activation_table() {
    std::array<ActMemberPtr, kSlotCount> t{};
    t[static_cast<std::size_t>(TensorSlot::BlockLN1)] = &modules::SimplifiedLayerActivations::ln1;
    t[static_cast<std::size_t>(TensorSlot::BlockLN1RSTD)] = &modules::SimplifiedLayerActivations::ln1_rstd;
    t[static_cast<std::size_t>(TensorSlot::BlockLN2)] = &modules::SimplifiedLayerActivations::ln2;
    t[static_cast<std::size_t>(TensorSlot::BlockLN2RSTD)] = &modules::SimplifiedLayerActivations::ln2_rstd;
    t[static_cast<std::size_t>(TensorSlot::BlockQRSTD)] = &modules::SimplifiedLayerActivations::q_rstd;
    t[static_cast<std::size_t>(TensorSlot::BlockKRSTD)] = &modules::SimplifiedLayerActivations::k_rstd;
    t[static_cast<std::size_t>(TensorSlot::BlockQKV)] = &modules::SimplifiedLayerActivations::qkv;
    // BlockQKVRoPE handled specially (conditional fallback to acts.qkv).
    t[static_cast<std::size_t>(TensorSlot::BlockLSE)] = &modules::SimplifiedLayerActivations::lse;
    t[static_cast<std::size_t>(TensorSlot::BlockAtt)] = &modules::SimplifiedLayerActivations::att;
    t[static_cast<std::size_t>(TensorSlot::BlockAttOut)] = &modules::SimplifiedLayerActivations::att_out;
    t[static_cast<std::size_t>(TensorSlot::BlockResidualAtt)] = &modules::SimplifiedLayerActivations::residual_att;
    t[static_cast<std::size_t>(TensorSlot::BlockMLPUp)] = &modules::SimplifiedLayerActivations::mlp_up;
    t[static_cast<std::size_t>(TensorSlot::BlockSwiGLU)] = &modules::SimplifiedLayerActivations::swiglu;
    t[static_cast<std::size_t>(TensorSlot::BlockMLPDown)] = &modules::SimplifiedLayerActivations::mlp_down;
    t[static_cast<std::size_t>(TensorSlot::BlockHOut)] = &modules::SimplifiedLayerActivations::h_out;
    // BlockResidualFFN handled specially (managed residual, not a raw field).
    t[static_cast<std::size_t>(TensorSlot::BlockRouterLogits)] = &modules::SimplifiedLayerActivations::router_logits;
    t[static_cast<std::size_t>(TensorSlot::BlockRouterProbs)] = &modules::SimplifiedLayerActivations::router_probs;
    t[static_cast<std::size_t>(TensorSlot::BlockRoutingWeights)] =
        &modules::SimplifiedLayerActivations::routing_weights;
    t[static_cast<std::size_t>(TensorSlot::BlockRoutingIndices)] =
        &modules::SimplifiedLayerActivations::routing_indices;
    t[static_cast<std::size_t>(TensorSlot::BlockPermutedInput)] = &modules::SimplifiedLayerActivations::permuted_input;
    t[static_cast<std::size_t>(TensorSlot::BlockScatterIndices)] =
        &modules::SimplifiedLayerActivations::scatter_indices;
    t[static_cast<std::size_t>(TensorSlot::BlockExpertGateUp)] = &modules::SimplifiedLayerActivations::expert_gate_up;
    t[static_cast<std::size_t>(TensorSlot::BlockExpertAct)] = &modules::SimplifiedLayerActivations::expert_act;
    t[static_cast<std::size_t>(TensorSlot::BlockExpertDown)] = &modules::SimplifiedLayerActivations::expert_down;
    t[static_cast<std::size_t>(TensorSlot::BlockMoeOut)] = &modules::SimplifiedLayerActivations::moe_out;
    return t;
}

constexpr auto make_gradient_table() {
    std::array<GradMemberPtr, kSlotCount> t{};
    t[static_cast<std::size_t>(TensorSlot::BlockDLN1)] = &modules::SimplifiedLayerGradients::d_ln1;
    t[static_cast<std::size_t>(TensorSlot::BlockDLN2)] = &modules::SimplifiedLayerGradients::d_ln2;
    t[static_cast<std::size_t>(TensorSlot::BlockDQKV)] = &modules::SimplifiedLayerGradients::d_qkv;
    t[static_cast<std::size_t>(TensorSlot::BlockDAtt)] = &modules::SimplifiedLayerGradients::d_att;
    t[static_cast<std::size_t>(TensorSlot::BlockDAttOut)] = &modules::SimplifiedLayerGradients::d_att_out;
    t[static_cast<std::size_t>(TensorSlot::BlockDMLPUp)] = &modules::SimplifiedLayerGradients::d_mlp_up;
    t[static_cast<std::size_t>(TensorSlot::BlockDSwiGLU)] = &modules::SimplifiedLayerGradients::d_swiglu;
    t[static_cast<std::size_t>(TensorSlot::BlockDMLPDown)] = &modules::SimplifiedLayerGradients::d_mlp_down;
    t[static_cast<std::size_t>(TensorSlot::BlockDHOut)] = &modules::SimplifiedLayerGradients::d_h_out;
    t[static_cast<std::size_t>(TensorSlot::BlockDResAtt)] = &modules::SimplifiedLayerGradients::d_res_att;
    t[static_cast<std::size_t>(TensorSlot::BlockDResFFN)] = &modules::SimplifiedLayerGradients::d_res_ffn;
    return t;
}

constexpr auto kActivationSlotTable = make_activation_table();
constexpr auto kGradientSlotTable = make_gradient_table();

}  // namespace

Tensor* block_activation_ptr(DslRunState& rs, int layer_idx, TensorSlot slot) {
    if (layer_idx < 0) return nullptr;
    const auto idx = static_cast<std::size_t>(slot);
    if (idx >= kSlotCount) return nullptr;

    // Special cases: BlockQKVRoPE falls back to acts.qkv when qkv_rope isn't
    // allocated (non-QK-norm path does in-place RoPE). BlockResidualFFN is
    // owned by the managed residual stream, not a direct struct field.
    auto& acts = rs.simplified_acts(layer_idx);
    if (slot == TensorSlot::BlockQKVRoPE) {
        return acts.qkv_rope.Data ? &acts.qkv_rope : &acts.qkv;
    }
    if (slot == TensorSlot::BlockResidualFFN) {
        return &rs.get_residual(layer_idx, rs.MainStream);
    }
    ActMemberPtr member = kActivationSlotTable[idx];
    return member ? &(acts.*member) : nullptr;
}

Tensor* block_gradient_ptr(DslRunState& rs, int layer_idx, TensorSlot slot) {
    if (layer_idx < 0) return nullptr;
    const auto idx = static_cast<std::size_t>(slot);
    if (idx >= kSlotCount) return nullptr;
    GradMemberPtr member = kGradientSlotTable[idx];
    if (!member) return nullptr;
    return &(rs.simplified_grads(layer_idx).*member);
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
