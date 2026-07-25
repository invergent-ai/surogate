// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Per-layer block-dimension derivation from IR param shapes. Shared by
// dsl_model.cpp (runtime config: rope tables, LoRA sizing) and
// graph_compiler.cpp (per-layer shape resolution) so the two never drift.

#ifndef SUROGATE_SRC_DSL_PER_LAYER_DIMS_H
#define SUROGATE_SRC_DSL_PER_LAYER_DIMS_H

#include <vector>

#include "runtime/dsl/dsl_runtime_config.h"
#include "runtime/dsl/ir.h"

namespace dsl {

/// Populate per-layer dims from the forward graph's param shapes.
///
/// ``dims`` is resized to ``num_layers`` and every entry is first filled with
/// the caller-supplied global defaults, then overridden from concrete param
/// shapes in three passes:
///  1. out_weight (authoritative for attn_dim; head_size = cols / global_hq),
///     mlp_down_weight / mlp_gate_weight (intermediate / mlp_up)
///  2. qkv_weight / self_attn_q_weight (qkv_channels; head_size fallback)
///  3. separate-projection attention (Laguna) and MLP fixups:
///     q_norm_weight rows give the exact per-layer head size (dividing
///     out_weight cols by the *global* head count is wrong for layers that
///     override the query head count), k/g projections give kv_dim/gate_dim,
///     mlp_up_weight rows are authoritative for mlp_up (2*M when gate+up are
///     fused), and layers with separate K/V projections derive qkv_channels
///     as attn_dim + 2*kv_dim.
void derive_per_layer_block_dims(const Graph& graph,
                                 int num_layers,
                                 long global_hq,
                                 long default_hkv,
                                 long default_hs,
                                 long default_dff,
                                 std::vector<BlockTypeDims>& dims);

/// True when any layer's dims differ from layer 0 in a field that affects
/// shape resolution (head_size / qkv_channels / attn_dim / kv_dim / gate_dim
/// / intermediate). Callers clear the dims vector when this is false so
/// homogeneous models keep using the global config.
[[nodiscard]] bool per_layer_dims_vary(const std::vector<BlockTypeDims>& dims);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_PER_LAYER_DIMS_H
