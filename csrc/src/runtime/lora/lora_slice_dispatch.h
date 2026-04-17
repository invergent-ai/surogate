// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Dispatch helpers that apply LoRA adapters inline after a matmul,
// driven by ``CompiledAttrs::lora_slices``. Inline application avoids
// the per-op hook callback dispatch used by the earlier design.

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_SLICE_DISPATCH_H
#define SUROGATE_SRC_MODULES_LORA_LORA_SLICE_DISPATCH_H

#include <stdexcept>
#include <string>
#include <vector>

#include <cublasLt.h>

#include "runtime/dsl/graph_compiler.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_grads_manager.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/lora/lora_types.h"
#include "runtime/lora/lora_weights_manager.h"
#include "utilities/comm.h"
#include "utilities/tensor.h"

namespace modules {
namespace detail {

inline unsigned compute_dropout_seed(const LoRARunState& rs, int layer_idx, const dsl::LoRASlice& slice) {
    return rs.dropout_base_seed + static_cast<unsigned>(layer_idx) * 1000000u +
           lora_target_seed_key(slice.id, slice.name) * 100000u + static_cast<unsigned>(rs.micro_step) * 10000u;
}

/// Best-effort human-readable label for a slice (prefers raw name when
/// available; otherwise the canonical enum value as a number). Only
/// invoked on error or for unknown targets.
inline std::string slice_label(const dsl::LoRASlice& slice) {
    if (!slice.name.empty()) return slice.name;
    return "id=" + std::to_string(static_cast<int>(slice.id));
}

/// Throws with the failing slice + layer on malformed bounds. Silently
/// skipping would produce a model that appears to train but whose LoRA
/// contribution is nil — very hard to diagnose later.
inline int
resolve_and_validate_slice_size(const dsl::LoRASlice& slice, int total_out, int layer_idx, const char* op_label) {
    const int size = slice.size > 0 ? slice.size : (total_out - slice.offset);
    if (size <= 0 || slice.offset < 0 || slice.offset + size > total_out) {
        throw std::invalid_argument(
            std::string(op_label) + ": LoRA slice '" + slice_label(slice) + "' at layer " + std::to_string(layer_idx) +
            " is out of bounds (offset=" + std::to_string(slice.offset) + ", size=" + std::to_string(size) +
            ", total_out=" + std::to_string(total_out) + "). Check the LoRATarget declaration in the DSL module.");
    }
    return size;
}

/// @brief Accumulate LoRA forward contributions into a matmul output.
///
/// ``input_2d`` / ``output_2d`` must be 2-D ``[BT, in]`` / ``[BT, out]``
/// views — use ``dsl::flatten_bt`` at the call site for ``[B, T, ...]``
/// tensors. Grouped slices (MoE routed experts) are dispatched by the
/// grouped-GEMM operators instead and are skipped here.
inline void apply_lora_slices_forward(const std::vector<dsl::LoRASlice>& slices,
                                      int layer_idx,
                                      Tensor input_2d,
                                      Tensor output_2d,
                                      int BT,
                                      ModularLoRAWeightsManager* weights,
                                      const ModularLoRAConfig* config,
                                      LoRARunState* run_state,
                                      cublasLtHandle_t handle,
                                      Tensor& workspace,
                                      cudaStream_t stream) {
    if (slices.empty() || weights == nullptr || config == nullptr || run_state == nullptr) return;
    if (!config->enabled()) return;

    auto& block = weights->get_block(layer_idx, stream);

    const int rank = config->rank;
    const float scaling = config->scaling();
    const bool is_training = run_state->is_training;
    const float dropout = is_training ? config->dropout : 0.0f;
    const int in_features = static_cast<int>(input_2d.Sizes[input_2d.Rank - 1]);
    const int total_out = static_cast<int>(output_2d.Sizes[output_2d.Rank - 1]);

    for (const auto& slice : slices) {
        if (slice.grouped) continue;
        // Validate bounds before the lookup: even if the target isn't
        // allocated at runtime (user didn't enable it), a declaration that
        // doesn't fit the matmul's output shape is a DSL bug that must
        // surface loudly rather than silently no-op.
        const int size = resolve_and_validate_slice_size(slice, total_out, layer_idx, "apply_lora_slices_forward");
        auto* lora = get_layer_weight_by_target(block, slice.id);
        if (!lora || !lora->has_value()) continue;

        const unsigned seed = compute_dropout_seed(*run_state, layer_idx, slice);
        apply_lora_contribution(output_2d,
                                slice.offset,
                                input_2d,
                                *lora,
                                run_state->intermediate,
                                run_state->slice,
                                scaling,
                                dropout,
                                seed,
                                is_training,
                                BT,
                                in_features,
                                size,
                                rank,
                                handle,
                                workspace,
                                stream);
    }
}

/// @brief Accumulate LoRA dA/dB gradients and (optionally) LoRA's
/// contribution to dx. Mirrors ``apply_lora_slices_forward``'s layout.
inline void apply_lora_slices_backward(const std::vector<dsl::LoRASlice>& slices,
                                       int layer_idx,
                                       Tensor input_2d,
                                       Tensor d_output_2d,
                                       Tensor dx_2d,
                                       int BT,
                                       ModularLoRAWeightsManager* weights,
                                       ModularLoRAGradsManager* grads,
                                       const ModularLoRAConfig* config,
                                       LoRARunState* run_state,
                                       NCCLCommunicator& comm,
                                       bool accumulate_op,
                                       cublasLtHandle_t handle,
                                       Tensor& workspace,
                                       cudaStream_t stream) {
    if (slices.empty() || weights == nullptr || grads == nullptr || config == nullptr || run_state == nullptr) return;
    if (!config->enabled()) return;

    auto& block = weights->get_block(layer_idx, stream);
    bool accum_base = false;
    auto& grad_block = grads->get_block_full(layer_idx, stream, comm, accum_base);
    const bool lora_accum = accum_base || accumulate_op;

    const int rank = config->rank;
    const float scaling = config->scaling();
    const bool is_training = run_state->is_training;
    const float dropout = is_training ? config->dropout : 0.0f;
    const int in_features = static_cast<int>(input_2d.Sizes[input_2d.Rank - 1]);
    const int total_out = static_cast<int>(d_output_2d.Sizes[d_output_2d.Rank - 1]);

    for (const auto& slice : slices) {
        if (slice.grouped) continue;
        // Validate bounds eagerly (see apply_lora_slices_forward for rationale).
        const int size = resolve_and_validate_slice_size(slice, total_out, layer_idx, "apply_lora_slices_backward");
        auto* lora = get_layer_weight_by_target(block, slice.id);
        if (!lora || !lora->has_value()) continue;
        auto* lora_grad = get_layer_weight_by_target(grad_block, slice.id);
        if (!lora_grad || !lora_grad->has_value()) continue;

        const unsigned seed = compute_dropout_seed(*run_state, layer_idx, slice);
        backward_lora_layer(lora_grad->A,
                            lora_grad->B,
                            dx_2d,
                            d_output_2d,
                            slice.offset,
                            input_2d,
                            lora->A,
                            lora->B,
                            scaling,
                            dropout,
                            seed,
                            is_training,
                            run_state->intermediate,
                            run_state->slice,
                            BT,
                            in_features,
                            size,
                            rank,
                            lora_accum,
                            handle,
                            workspace,
                            stream,
                            /*skip_dx=*/!dx_2d.Data);
    }
}

}  // namespace detail
}  // namespace modules

#endif  // SUROGATE_SRC_MODULES_LORA_LORA_SLICE_DISPATCH_H
