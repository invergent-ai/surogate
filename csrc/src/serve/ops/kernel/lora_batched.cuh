#pragma once

// Per-token adapter selection: the delta for a round in which different tokens
// belong to different adapters.
//
// The single-adapter path multiplies by one A and one B. Serving many adapters
// at once cannot do that, because the round's tokens no longer share a weight
// matrix. Both halves therefore read the adapter *the token selected*, out of a
// stacked bank -- which is the shape vLLM's punica kernels use, and the reason
// the stacked bank is padded to (max_loras, max_rank): every slot has the same
// stride, so the token's index is the only thing that varies and the launch
// geometry is constant. A constant launch geometry is also what lets this be
// captured in a CUDA graph.
//
// A token whose adapter id is negative is a base-model token and contributes
// nothing. Mixing adapted and unadapted requests in one round is the normal
// case, not an edge case, so it costs a branch rather than a separate launch.


#include <cuda_bf16.h>
#include <cstdint>

namespace sinfer::ops {

/// low[r, t] = sum_k A[id[t], r, k] * x[k, t], with `a_stride` the elements
/// between adapter slots and `rank_stride` those between rows of one slot.
///
/// One warp per (token, rank row): rank is small (<= 64) and k is thousands, so
/// the reduction is over k and the parallelism is over the two small extents.
__global__ void lora_batched_shrink_kernel(const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ a_bank,
                                           const std::int32_t* __restrict__ ids,
                                           __nv_bfloat16* __restrict__ low, std::int32_t k,
                                           std::int32_t rank, std::int32_t tokens,
                                           std::int64_t a_stride,
                                           const std::int32_t* __restrict__ uniform) {
    const int token = static_cast<int>(blockIdx.y);
    const int row   = static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
    if (token >= tokens || row >= rank) { return; }
    // Prefill hands one slot for the whole round rather than a vector: every
    // column belongs to the same request. It arrives as a device cell rather than
    // a launch argument because a captured graph freezes its arguments -- a slot
    // passed by value would pin every later replay to whichever request was being
    // prefilled when the graph was captured.
    const int adapter = ids != nullptr ? ids[token] : (uniform != nullptr ? *uniform : -1);
    if (adapter < 0) {
        if (threadIdx.x == 0) { low[static_cast<std::int64_t>(token) * rank + row] = __float2bfloat16(0.0F); }
        return;
    }

    const __nv_bfloat16* a_row =
        a_bank + static_cast<std::int64_t>(adapter) * a_stride + static_cast<std::int64_t>(row) * k;
    const __nv_bfloat16* x_col = x + static_cast<std::int64_t>(token) * k;

    float sum = 0.0F;
    for (int i = static_cast<int>(threadIdx.x) * 2; i < k; i += 64) {
        // Two at a time: both operands are contiguous over k, so this is a clean
        // stream and the pair halves the loop's issue count.
        const float2 av = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(a_row + i));
        const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x_col + i));
        sum += av.x * xv.x + av.y * xv.y;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
    }
    if (threadIdx.x == 0) {
        low[static_cast<std::int64_t>(token) * rank + row] = __float2bfloat16(sum);
    }
}

/// out[n, t] += sum_r B[id[t], n, r] * low[r, t].
///
/// The accumulate is into the projection's own output, so this is the only place
/// the delta touches it: one read-modify-write per element, no separate add pass
/// and no second buffer.
__global__ void lora_batched_expand_kernel(const __nv_bfloat16* __restrict__ low,
                                           const __nv_bfloat16* __restrict__ b_bank,
                                           const std::int32_t* __restrict__ ids,
                                           __nv_bfloat16* __restrict__ out, std::int32_t n,
                                           std::int32_t rank, std::int32_t tokens,
                                           std::int64_t b_stride,
                                           const std::int32_t* __restrict__ uniform) {
    const int token = static_cast<int>(blockIdx.y);
    const int row   = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (token >= tokens || row >= n) { return; }
    const int adapter = ids != nullptr ? ids[token] : (uniform != nullptr ? *uniform : -1);
    if (adapter < 0) { return; }

    const __nv_bfloat16* b_row =
        b_bank + static_cast<std::int64_t>(adapter) * b_stride + static_cast<std::int64_t>(row) * rank;
    const __nv_bfloat16* low_col = low + static_cast<std::int64_t>(token) * rank;

    float sum = 0.0F;
    for (int r = 0; r < rank; ++r) {
        sum += __bfloat162float(b_row[r]) * __bfloat162float(low_col[r]);
    }
    __nv_bfloat16* cell = out + static_cast<std::int64_t>(token) * n + row;
    *cell               = __float2bfloat16(__bfloat162float(*cell) + sum);
}

} // namespace sinfer::ops
