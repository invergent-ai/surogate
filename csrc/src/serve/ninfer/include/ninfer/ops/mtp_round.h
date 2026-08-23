#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: mtp_prepare_next_round
 *
 * Math / indexing:
 *   For T=K+1 and each row b with A=accepted[b], L=licensed_counts[b]:
 *     alignment_ids[j,b] = verify_ids[j+1,b]  for 0<=j<A
 *                          next_anchors[b]     otherwise;
 *     remaining_after = max(remaining_budgets[b]-L,0);
 *     context_after   = max(max_context-updated_frontiers[b]-1,0);
 *     next_extents[b] = min(K,max(remaining_after-1,0),context_after);
 *     For S=max(K-1,1) and 0<=s<S:
 *       ar_positions[b,s]      = updated_frontiers[b]+s;
 *       ar_rope_positions[b,s] = ar_positions[b,s]+rope_deltas[b];
 *       ar_valid_columns[b,s]  = (s+1 < next_extents[b]).
 *
 * Logical shapes / effects:
 *   verify_ids/alignment_ids are distinct contiguous I32 [K+1,B]. ar_positions,
 *   ar_rope_positions, and ar_valid_columns are I32 [B,max(K-1,1)] with contiguous rows and one
 *   shared step stride at least B; this permits an exact-B prefix of a fixed-capacity frame. All
 *   other tensors are contiguous I32 [B]. B>=1, 1<=K<=5, 0<=accepted[b]<=K,
 *   licensed_counts[b]=accepted[b]+1, updated_frontiers and remaining_budgets are non-negative,
 *   and max_context is positive. The Op writes every output slot, including safe invalid-tail
 *   values. Inputs remain unchanged. No workspace or other state is used.
 */
void mtp_prepare_next_round(const Tensor& verify_ids, const Tensor& next_anchors,
                            const Tensor& accepted, const Tensor& updated_frontiers,
                            const Tensor& remaining_budgets, const Tensor& licensed_counts,
                            const Tensor& rope_deltas, Tensor& alignment_ids, Tensor& next_extents,
                            Tensor& ar_positions, Tensor& ar_rope_positions,
                            Tensor& ar_valid_columns, std::int32_t max_context,
                            cudaStream_t stream);

} // namespace ninfer::ops
