// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Single source of truth for mapping a TensorSlot enum value to the
// corresponding Tensor buffer inside DslRunState. Callers that need to
// resolve "this slot at this layer -> which Tensor*?" should use these
// helpers instead of open-coding per-field strcmp dispatch.
//
// This replaces the repeated switch-on-name-strings that appeared in
// try_get_tensor_fuzzy, resolve_source, resolve_block_activation_tensor,
// block_activation_base_ptr, resolve_block_gradient_tensor, and
// resolve_block_activation_base. Each of those had an independently-maintained
// table of fields -> RunState members; any one of them falling out of sync
// with the others has historically produced silent-corruption bugs (see the
// `res_attn`, `mlp_x_flat`, and matmul_backward aliasing incidents).
//
// Usage:
//   int lid;
//   std::string field;
//   if (parse_block_param(name, lid, field)) {
//       auto slot = builtin_slot_from_name(strip_ssa_suffix(field));
//       if (auto* t = block_activation_ptr(rs, lid, slot)) { ... }
//       if (auto* t = block_gradient_ptr(rs, lid, slot))   { ... }
//   }
//   if (auto* t = global_activation_ptr(rs, builtin_slot_from_name(name))) { ... }
//
// To extend: when a new TensorSlot enum value is added, updating the switch
// in ONE of these helpers (the one whose RunState group holds the buffer)
// is all that is required. No other callsite needs to change.

#ifndef SUROGATE_SRC_DSL_TENSOR_SLOT_DISPATCH_H
#define SUROGATE_SRC_DSL_TENSOR_SLOT_DISPATCH_H

#include "runtime/dsl/tensor_slot.h"

struct Tensor;  // forward declared in utilities/tensor.h (global namespace)

namespace dsl {

class DslRunState;

/// Return a pointer to the block-scope activation buffer for (`layer_idx`, `slot`),
/// or `nullptr` if the slot isn't a block-scope activation (global/gradient/unknown).
///
/// Handles the `BlockQKVRoPE` case transparently: falls back to `acts.qkv` when the
/// separate `qkv_rope` buffer isn't allocated (non-QK-norm path does in-place RoPE).
/// For `BlockResidualFFN` this acquires the managed residual stream for the layer
/// on `rs.MainStream`.
Tensor* block_activation_ptr(DslRunState& rs, int layer_idx, TensorSlot slot);

/// Return a pointer to the per-layer gradient buffer for (`layer_idx`, `slot`),
/// or `nullptr` if the slot isn't a block-scope gradient.
///
/// Callers with a tensor name that's NOT itself a `d_*` gradient (e.g.,
/// `blocks[N].d_h_out` where the `d_` is part of the field, not the prefix)
/// should still pass the gradient slot enum value — this helper dispatches
/// purely on `TensorSlot`, not on name shape.
Tensor* block_gradient_ptr(DslRunState& rs, int layer_idx, TensorSlot slot);

/// Return a pointer to the global (non-block) activation buffer for `slot`, or
/// `nullptr` if the slot isn't a global activation. Covers Encoded, LNFinal,
/// LNFinalRSTD, FinalResidual, FreqCis, TokenIDs, PositionIDs.
///
/// For FreqCis we do not receive the original name here; callers that need to
/// resolve ambiguous qualified names like `blocks[N].rope_freqs` should call
/// `rs.rope_freqs(name)` directly after a positive FreqCis classification.
Tensor* global_activation_ptr(DslRunState& rs, TensorSlot slot);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_TENSOR_SLOT_DISPATCH_H
