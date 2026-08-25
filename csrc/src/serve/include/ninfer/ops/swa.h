#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Host execution-resource promise for swa.
 *
 * positions[0,b] is row b's exact device-resident committed-context frontier. This envelope bounds
 * every row so a fixed launch can be captured and replayed without a host read.
 */
struct SwaContextExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

/**
 * Op: symmetric non-causal sliding-window grouped-query attention
 *
 * The fixed optimized geometry is D=128, Hq=32, Hkv=8, group=4, and window W=4096.
 * q/out are contiguous BF16 [128,32,W,B], query_k/query_v are contiguous BF16 [128,8,W,B],
 * positions is contiguous device I32 [W,B], valid_columns and lanes are contiguous device I32
 * [B]. Row b has V=valid_columns[b] live query columns with positions[i,b]=L[b]+i for i<V;
 * lanes[b] selects its cyclic-cache lane. Columns i>=V are an inert physical tail and produce
 * zero output.
 *
 * The read-only cyclic context contains committed absolute positions [max(0,L-4096),L), with
 * absolute position p stored at physical slot p mod 4096. Query K/V is a separate temporary
 * segment at positions [L,L+V). For every live query position p_i, admitted populated keys satisfy
 * abs(p_j-p_i)<4096. Thus distance 4095 is included, distance 4096 is excluded, and every query
 * row sees every live temporary query row from the same batch row. scale is 1/sqrt(128).
 *
 * Context and query K/V are unchanged. out is the only observable mutation and is completely
 * overwritten. The current optimized implementation domain is T=1..16 on sm_120a.
 *
 * The caller guarantees min_context <= L <= max_context, sequential nonnegative positions, and
 * that the cyclic context contains the declared live interval. The execution envelope may affect
 * finite launch selection and workspace capacity, never the admitted key set.
 */
void swa(const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& positions,
         const Tensor& valid_columns, const Tensor& lanes, float scale,
         const CyclicKVCacheLayerView& context, SwaContextExecutionEnvelope envelope,
         WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * Returns the transient arena capacity required for every T in the inclusive optimized interval.
 * The execution envelope is the fixed profile; invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t swa_workspace_capacity_bytes(SwaContextExecutionEnvelope envelope,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens,
                                                       std::int32_t batch_size);

} // namespace ninfer::ops
