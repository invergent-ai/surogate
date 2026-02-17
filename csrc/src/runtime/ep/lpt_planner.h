// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// LPT (Longest Processing Time) planner for LLEP load balancing.
// Runs per-layer, per-step to compute optimal expert→GPU assignments
// when routing imbalance exceeds the adaptive threshold.

#ifndef SUROGATE_SRC_RUNTIME_EP_LPT_PLANNER_H
#define SUROGATE_SRC_RUNTIME_EP_LPT_PLANNER_H

#include <vector>

namespace ep {

/// Describes a weight transfer between two GPUs.
struct WeightTransferEntry {
    int expert_id;      ///< Global expert ID
    int src_rank;       ///< Rank that owns the weight (native)
    int dst_rank;       ///< Rank that will receive the weight (helper)
};

/// Per-expert assignment: which GPU handles which tokens.
struct ExpertAssignment {
    int gpu_id;         ///< GPU that will process these tokens
    int token_start;    ///< Start token offset for this chunk
    int token_end;      ///< End token offset for this chunk
};

/// Complete LPT plan for a single layer.
struct LPTPlan {
    /// Per-expert assignments: expert_assignments[expert_id] = list of (gpu, start, end)
    std::vector<std::vector<ExpertAssignment>> expert_assignments;

    /// Global list of weight transfers needed
    std::vector<WeightTransferEntry> weight_transfers;

    /// Per-GPU final load (token count)
    std::vector<int> gpu_loads;

    /// For this rank: experts to send (expert_id, dst_rank)
    std::vector<std::pair<int, int>> weights_to_send;

    /// For this rank: experts to receive (expert_id, src_rank)
    std::vector<std::pair<int, int>> weights_to_receive;

    /// Primary GPU for each expert: expert_to_gpu[expert_id] = GPU that processes it.
    /// Derived from expert_assignments (GPU with most tokens for this expert).
    std::vector<int> expert_to_gpu;

    /// Whether this plan uses LPT (true) or standard balanced path (false)
    bool uses_lpt = false;
};

/// Compute the GPU load imbalance ratio under default expert assignment.
/// Returns max_load / mean_load (1.0 = perfectly balanced).
float compute_imbalance_ratio(
    const int* global_expert_counts,  ///< [num_experts] token counts per expert
    int num_experts,
    int ep_size,
    int num_local_experts);

/// Compute LPT plan for LLEP with weight spilling.
///
/// Implements the Largest Processing Time algorithm: sorts experts by load
/// (largest first), then assigns each expert to its native GPU if capacity
/// allows, otherwise spills tokens + weights to the least-loaded helper GPU.
///
/// @param global_expert_counts  [num_experts] global token counts (on host)
/// @param num_experts           Total number of experts
/// @param ep_size               Number of GPUs in EP group
/// @param ep_rank               This GPU's rank within EP group
/// @param num_local_experts     Experts per GPU (num_experts / ep_size)
/// @param max_tokens_factor     Capacity factor (default 1.1 = 10% overcommit)
/// @param min_tokens_per_gemm   Minimum chunk size to avoid tiny GEMMs
LPTPlan compute_lpt_plan(
    const int* global_expert_counts,
    int num_experts,
    int ep_size,
    int ep_rank,
    int num_local_experts,
    float max_tokens_factor = 1.1f,
    int min_tokens_per_gemm = 1024);

}  // namespace ep

#endif  // SUROGATE_SRC_RUNTIME_EP_LPT_PLANNER_H
