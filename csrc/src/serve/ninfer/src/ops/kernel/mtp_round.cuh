#pragma once

// Implements: include/ninfer/ops/mtp_round.h
// Match: request-major fixed K=1..5 autoregressive MTP round transition.

#include <cstdint>

namespace ninfer::ops {

__global__ void mtp_prepare_next_round_kernel(
    const std::int32_t* verify_ids, const std::int32_t* next_anchors, const std::int32_t* accepted,
    const std::int32_t* updated_frontiers, const std::int32_t* remaining_budgets,
    const std::int32_t* licensed_counts, const std::int32_t* rope_deltas,
    std::int32_t* alignment_ids, std::int32_t* next_extents, std::int32_t* ar_positions,
    std::int32_t* ar_rope_positions, std::int32_t* ar_valid_columns, std::int32_t k,
    std::int32_t ar_step_stride, std::int32_t max_context) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int a         = accepted[row];
    a             = a < 0 ? 0 : (a > k ? k : a);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        alignment_ids[row * T + j] = j < a ? verify_ids[row * T + j + 1] : next_anchors[row];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int licensed       = licensed_counts[row];
        const int remaining      = remaining_budgets[row] - licensed;
        const int budget_extent  = remaining > 1 ? remaining - 1 : 0;
        const int context_extent = max_context - updated_frontiers[row] - 1;
        int next                 = budget_extent < context_extent ? budget_extent : context_extent;
        next                     = next < 0 ? 0 : (next > k ? k : next);
        next_extents[row]        = next;
        const int steps          = k > 1 ? k - 1 : 1;
        for (int s = 0; s < steps; ++s) {
            const int offset          = s * ar_step_stride + row;
            const int position        = updated_frontiers[row] + s;
            ar_positions[offset]      = position;
            ar_rope_positions[offset] = position + rope_deltas[row];
            ar_valid_columns[offset]  = s + 1 < next ? 1 : 0;
        }
    }
}

} // namespace ninfer::ops
