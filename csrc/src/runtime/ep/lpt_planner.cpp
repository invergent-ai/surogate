// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/ep/lpt_planner.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ep {

float compute_imbalance_ratio(
    const int* global_expert_counts,
    int num_experts,
    int ep_size,
    int num_local_experts) {

    // Compute per-GPU load under default assignment
    std::vector<long> gpu_loads(ep_size, 0);
    for (int e = 0; e < num_experts; ++e) {
        int gpu = e / num_local_experts;
        if (gpu < ep_size) {
            gpu_loads[gpu] += global_expert_counts[e];
        }
    }

    long max_load = *std::max_element(gpu_loads.begin(), gpu_loads.end());
    long total = std::accumulate(gpu_loads.begin(), gpu_loads.end(), 0L);
    float mean_load = static_cast<float>(total) / ep_size;

    if (mean_load <= 0.0f) return 1.0f;
    return static_cast<float>(max_load) / mean_load;
}

LPTPlan compute_lpt_plan(
    const int* global_expert_counts,
    int num_experts,
    int ep_size,
    int ep_rank,
    int num_local_experts,
    float max_tokens_factor,
    int min_tokens_per_gemm) {

    LPTPlan plan;
    plan.expert_assignments.resize(num_experts);
    plan.gpu_loads.resize(ep_size, 0);
    plan.uses_lpt = true;

    // Compute total tokens and balanced load
    long total_tokens = 0;
    for (int e = 0; e < num_experts; ++e) {
        total_tokens += global_expert_counts[e];
    }
    const long balanced_tokens = (ep_size > 0) ? total_tokens / ep_size : total_tokens;
    int max_tokens_per_gpu = (balanced_tokens > 0)
        ? static_cast<int>(max_tokens_factor * balanced_tokens)
        : static_cast<int>(total_tokens);
    max_tokens_per_gpu = std::max(max_tokens_per_gpu, 1);

    // Pre-compute native load per GPU
    std::vector<int> native_load(ep_size, 0);
    for (int e = 0; e < num_experts; ++e) {
        int gpu = e / num_local_experts;
        if (gpu < ep_size) {
            native_load[gpu] += global_expert_counts[e];
        }
    }

    // Track pending native load (decreases as we process experts in LPT order)
    std::vector<int> pending_native(native_load);
    // Track assigned load (increases as we assign experts)
    std::vector<int> assigned(ep_size, 0);

    // Sort experts by token count descending (LPT ordering)
    std::vector<std::pair<int, int>> sorted_experts(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        sorted_experts[e] = {e, global_expert_counts[e]};
    }
    std::sort(sorted_experts.begin(), sorted_experts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Effective load = assigned + pending native (accounts for future arrivals)
    auto effective_load = [&](int gpu) -> int {
        return assigned[gpu] + pending_native[gpu];
    };

    for (const auto& [expert_id, expert_tokens] : sorted_experts) {
        if (expert_tokens == 0) continue;

        const int native_gpu = expert_id / num_local_experts;

        // Remove from pending native (this expert is now being processed)
        pending_native[native_gpu] -= expert_tokens;

        const int native_eff = effective_load(native_gpu);
        const int native_available = max_tokens_per_gpu - native_eff;

        if (native_available >= expert_tokens) {
            // Case 1: Native GPU can handle all tokens
            plan.expert_assignments[expert_id].push_back(
                {native_gpu, 0, expert_tokens});
            assigned[native_gpu] += expert_tokens;

        } else if (native_available > 0) {
            // Case 2: Partial on native, spill rest to helper(s)
            int native_chunk = std::min(native_available, expert_tokens);
            plan.expert_assignments[expert_id].push_back(
                {native_gpu, 0, native_chunk});
            assigned[native_gpu] += native_chunk;

            int remaining = expert_tokens - native_chunk;
            int token_offset = native_chunk;

            while (remaining > 0) {
                // Find least-loaded helper by effective load
                int best_gpu = -1;
                int best_eff = std::numeric_limits<int>::max();
                for (int g = 0; g < ep_size; ++g) {
                    if (g == native_gpu) continue;
                    int eff = effective_load(g);
                    if (eff < best_eff) {
                        best_eff = eff;
                        best_gpu = g;
                    }
                }

                if (best_gpu < 0) {
                    // Only one GPU — force to native
                    auto& last = plan.expert_assignments[expert_id].back();
                    if (last.gpu_id == native_gpu) {
                        last.token_end += remaining;
                    } else {
                        plan.expert_assignments[expert_id].push_back(
                            {native_gpu, token_offset, token_offset + remaining});
                    }
                    assigned[native_gpu] += remaining;
                    break;
                }

                int helper_available = max_tokens_per_gpu - best_eff;
                int chunk = (helper_available > 0)
                    ? std::min(remaining, helper_available)
                    : remaining;

                // Skip tiny chunks unless it's all that's left
                if (chunk < min_tokens_per_gemm && remaining > chunk) {
                    chunk = remaining;
                }

                plan.expert_assignments[expert_id].push_back(
                    {best_gpu, token_offset, token_offset + chunk});
                assigned[best_gpu] += chunk;

                plan.weight_transfers.push_back(
                    {expert_id, native_gpu, best_gpu});

                token_offset += chunk;
                remaining -= chunk;
            }

        } else {
            // Case 3: Native at/over capacity, spill everything
            int remaining = expert_tokens;
            int token_offset = 0;

            // Collect and sort helpers by effective load
            std::vector<std::pair<int, int>> helpers;
            for (int g = 0; g < ep_size; ++g) {
                if (g == native_gpu) continue;
                helpers.push_back({g, effective_load(g)});
            }
            std::sort(helpers.begin(), helpers.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });

            for (const auto& [helper_gpu, helper_eff] : helpers) {
                if (remaining <= 0) break;

                int avail = max_tokens_per_gpu - helper_eff;
                if (avail <= 0) continue;

                int chunk = std::min(remaining, avail);
                if (chunk < min_tokens_per_gemm && remaining > chunk) {
                    continue;
                }

                plan.expert_assignments[expert_id].push_back(
                    {helper_gpu, token_offset, token_offset + chunk});
                assigned[helper_gpu] += chunk;

                plan.weight_transfers.push_back(
                    {expert_id, native_gpu, helper_gpu});

                token_offset += chunk;
                remaining -= chunk;
            }

            // Force remaining to least loaded helper if all at capacity
            if (remaining > 0) {
                int fallback_gpu = helpers.empty() ? native_gpu : helpers[0].first;
                plan.expert_assignments[expert_id].push_back(
                    {fallback_gpu, token_offset, token_offset + remaining});
                assigned[fallback_gpu] += remaining;

                if (fallback_gpu != native_gpu) {
                    plan.weight_transfers.push_back(
                        {expert_id, native_gpu, fallback_gpu});
                }
            }
        }
    }

    // Copy final loads
    plan.gpu_loads = assigned;

    // Build expert_to_gpu mapping (primary GPU = GPU with most tokens for each expert)
    plan.expert_to_gpu.resize(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        const auto& asgn = plan.expert_assignments[e];
        if (asgn.empty()) {
            plan.expert_to_gpu[e] = e / num_local_experts;  // default native
        } else {
            int best_gpu = asgn[0].gpu_id;
            int best_tokens = asgn[0].token_end - asgn[0].token_start;
            for (size_t i = 1; i < asgn.size(); ++i) {
                int tokens = asgn[i].token_end - asgn[i].token_start;
                if (tokens > best_tokens) {
                    best_tokens = tokens;
                    best_gpu = asgn[i].gpu_id;
                }
            }
            plan.expert_to_gpu[e] = best_gpu;
        }
    }

    // Extract per-rank send/receive lists
    for (const auto& wt : plan.weight_transfers) {
        if (wt.src_rank == ep_rank) {
            plan.weights_to_send.emplace_back(wt.expert_id, wt.dst_rank);
        }
        if (wt.dst_rank == ep_rank) {
            plan.weights_to_receive.emplace_back(wt.expert_id, wt.src_rank);
        }
    }

    return plan;
}

}  // namespace ep
