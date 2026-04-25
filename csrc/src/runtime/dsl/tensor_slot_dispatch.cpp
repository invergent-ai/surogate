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
#include "runtime/executor/compiled_ops.h"
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

constexpr bool is_block_activation_slot(TensorSlot s) {
    switch (s) {
        case TensorSlot::BlockLN1:
        case TensorSlot::BlockLN1RSTD:
        case TensorSlot::BlockLN2:
        case TensorSlot::BlockLN2RSTD:
        case TensorSlot::BlockQRSTD:
        case TensorSlot::BlockKRSTD:
        case TensorSlot::BlockQKV:
        case TensorSlot::BlockQKVRoPE:
        case TensorSlot::BlockLSE:
        case TensorSlot::BlockAtt:
        case TensorSlot::BlockAttOut:
        case TensorSlot::BlockResidualAtt:
        case TensorSlot::BlockMLPUp:
        case TensorSlot::BlockSwiGLU:
        case TensorSlot::BlockMLPDown:
        case TensorSlot::BlockResidualFFN:
        case TensorSlot::BlockHOut:
        case TensorSlot::BlockRouterLogits:
        case TensorSlot::BlockRouterProbs:
        case TensorSlot::BlockRoutingWeights:
        case TensorSlot::BlockRoutingIndices:
        case TensorSlot::BlockPermutedInput:
        case TensorSlot::BlockScatterIndices:
        case TensorSlot::BlockExpertGateUp:
        case TensorSlot::BlockExpertAct:
        case TensorSlot::BlockExpertDown:
        case TensorSlot::BlockMoeOut: return true;
        default: return false;
    }
}

constexpr bool is_block_gradient_slot(TensorSlot s) {
    switch (s) {
        case TensorSlot::BlockDLN1:
        case TensorSlot::BlockDLN2:
        case TensorSlot::BlockDQKV:
        case TensorSlot::BlockDAtt:
        case TensorSlot::BlockDAttOut:
        case TensorSlot::BlockDMLPUp:
        case TensorSlot::BlockDSwiGLU:
        case TensorSlot::BlockDMLPDown:
        case TensorSlot::BlockDHOut:
        case TensorSlot::BlockDResAtt:
        case TensorSlot::BlockDResFFN: return true;
        default: return false;
    }
}

}  // namespace

Tensor* block_activation_ptr(DslRunState& rs, int layer_idx, TensorSlot slot) {
    if (layer_idx < 0) return nullptr;
    if (!is_block_activation_slot(slot)) return nullptr;

    // mTensors[tid] is the source of truth for block activations. Special
    // cases: managed residual stream and in-place QKVRoPE fallback route
    // directly to their owners. Returns nullptr when no executor is active.
    if (Tensor* t = rs.active_executor_slot(layer_idx, slot)) {
        return t;
    }
    if (slot == TensorSlot::BlockResidualFFN) {
        return &rs.get_residual(layer_idx, rs.MainStream);
    }
    if (slot == TensorSlot::BlockQKVRoPE) {
        return rs.active_executor_slot(layer_idx, TensorSlot::BlockQKV);
    }
    return nullptr;
}

Tensor* block_gradient_ptr(DslRunState& rs, int layer_idx, TensorSlot slot) {
    if (layer_idx < 0) return nullptr;
    if (!is_block_gradient_slot(slot)) return nullptr;
    // mTensors[tid] is the source of truth for gradient activations.
    // Arena-backed tids are seeded by populate_bwd_stack_bindings at bwd
    // entry; Stack-backed gradient tids get Data via temp_acquire on first
    // use. Returns nullptr when no executor is active.
    if (auto* exec = rs.active_executor()) {
        return exec->executor_tid_slot_binding(layer_idx, slot);
    }
    return nullptr;
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
