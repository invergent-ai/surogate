// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_MOE_TYPES_H
#define SUROGATE_SRC_MODULES_MOE_MOE_TYPES_H

#include "utilities/tensor.h"

#include <algorithm>
#include <set>
#include <vector>
#include <cuda_runtime.h>

namespace modules {

// Forward declaration for MoEGroupedContext
struct SelectiveExpertInfo;

/**
 * @brief Context for MoE expert group computation
 *
 * This struct is passed to hooks during MoE block execution to enable
 * third-party modules (like LoRA) to perform specialized grouped computation.
 */
struct MoEGroupedContext {
    // Input/State tensors (from base model)
    const Tensor* expert_offsets;   ///< (num_experts + 1) int
    const Tensor* permuted_input;    ///< (total_tokens, C)
    const Tensor* expert_gate_up;    ///< (total_tokens, 2*D) output of base projection (Saved for backward)
    const Tensor* expert_outputs;    ///< (total_tokens, C) output of base down projection
    const Tensor* expert_indices;    ///< (BT, top_k) int - router-selected expert indices for each token

    // Gradient tensors (during backward)
    const Tensor* d_expert_outputs;  ///< (total_tokens, C) gradient w.r.t expert outputs
    const Tensor* d_expert_gate_up;  ///< (total_tokens, 2*D) gradient w.r.t gate_up
    Tensor* d_permuted_input;        ///< (total_tokens, C) gradient w.r.t permuted input

    const int* host_offsets;        ///< Cached expert offsets on host
    int num_experts;
    int top_k;
    int total_tokens;

    // For selective expert dequantization - hook can populate this
    SelectiveExpertInfo* selection_info = nullptr;  ///< Filled by hook for index remapping

    // Optional: flag to indicate if the hook handled the computation
    bool handled = false;
};

/**
 * @brief Selection info for selective expert dequantization
 *
 * When using selective expert dequantization, only the experts that are
 * actually selected by the router are dequantized. This struct holds the
 * information needed to map between global expert indices and compact
 * buffer indices.
 */
struct SelectiveExpertInfo {
    /// Unique expert indices that were selected across all tokens (sorted, deduplicated)
    /// Size: num_active_experts (typically much smaller than total experts)
    std::vector<int> active_experts;

    /// Maps global expert index -> compact buffer index (-1 if not active)
    /// Size: num_total_experts
    std::vector<int> expert_to_compact;

    /// Number of unique experts selected (active_experts.size())
    int num_active = 0;

    /// Total number of experts in the model
    int num_total = 0;

    /// Whether selective dequantization is enabled
    bool enabled = false;

    /**
     * @brief Build the selection info from router output
     *
     * @param expert_indices Device tensor (BT, top_k) of selected expert indices
     * @param num_experts Total number of experts in the model
     * @param stream CUDA stream for D2H copy
     */
    void build_from_router_output(const Tensor& expert_indices, int num_experts, cudaStream_t stream);

    /**
     * @brief Get the compact index for a global expert index
     * @return Compact index, or -1 if expert is not active
     */
    int get_compact_index(int global_expert_idx) const {
        if (!enabled || global_expert_idx < 0 || global_expert_idx >= num_total) {
            return -1;
        }
        return expert_to_compact[global_expert_idx];
    }

    /**
     * @brief Check if an expert is active (selected by router)
     */
    bool is_active(int global_expert_idx) const {
        return get_compact_index(global_expert_idx) >= 0;
    }

    /**
     * @brief Reset to disabled state
     */
    void reset() {
        active_experts.clear();
        expert_to_compact.clear();
        num_active = 0;
        num_total = 0;
        enabled = false;
    }
};

// ============================================================================
// Inline Implementation
// ============================================================================

inline void SelectiveExpertInfo::build_from_router_output(
    const Tensor& expert_indices, int num_experts, cudaStream_t stream)
{
    // Get dimensions
    const int BT = expert_indices.Sizes[0];
    const int top_k = expert_indices.Sizes[1];
    const int total_selections = BT * top_k;

    // Copy expert indices from device to host
    std::vector<int> host_indices(total_selections);
    cudaMemcpyAsync(host_indices.data(), expert_indices.get<int>(),
                    total_selections * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Find unique experts using a set
    std::set<int> unique_set(host_indices.begin(), host_indices.end());

    // Build the active experts list (sorted)
    active_experts.assign(unique_set.begin(), unique_set.end());
    num_active = static_cast<int>(active_experts.size());
    num_total = num_experts;

    // Build the reverse mapping: global expert index -> compact index
    expert_to_compact.assign(num_experts, -1);
    for (int compact_idx = 0; compact_idx < num_active; ++compact_idx) {
        int global_idx = active_experts[compact_idx];
        if (global_idx >= 0 && global_idx < num_experts) {
            expert_to_compact[global_idx] = compact_idx;
        }
    }

    enabled = true;
}

/**
 * @brief Context for router LoRA forward hook
 *
 * This struct is passed to the AfterRouterProjection hook to allow
 * LoRA to add its contribution to the router logits before softmax.
 */
struct MoERouterContext {
    Tensor* logits;           ///< (B*T, num_experts) router logits to modify in-place
    const Tensor* input;      ///< (B*T, hidden_size) input to router (ln2 output)
    int num_experts;
    int hidden_size;
    bool handled = false;     ///< Set to true if hook handled the computation
};

/**
 * @brief Context for router LoRA backward hook
 *
 * This struct is passed to the AfterRouterBackward hook to allow
 * LoRA to compute gradients for the router's lora_A and lora_B matrices.
 *
 * The router LoRA forward computes: logits += scaling * (input @ A^T @ B^T)
 * The backward computes:
 *   d_lora_B = d_logits^T @ intermediate   where intermediate = input @ A^T
 *   d_lora_A = B^T @ d_logits^T @ input
 */
struct MoERouterBackwardContext {
    const Tensor* d_logits;   ///< (B*T, num_experts) FP32 gradient w.r.t router logits
    const Tensor* input;      ///< (B*T, hidden_size) input to router (ln2 output, BF16)
    int num_experts;
    int hidden_size;
    int BT;                   ///< Batch * Sequence length (total tokens)
    bool handled = false;     ///< Set to true if hook handled the computation
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_MOE_TYPES_H
