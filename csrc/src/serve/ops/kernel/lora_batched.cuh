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

inline constexpr int kLoraShrinkThreads = 128;

__device__ __forceinline__ float lora_dot2(unsigned a_bits, unsigned x_bits) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a_bits));
    const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&x_bits));
    return a.x * x.x + a.y * x.y;
}

/// low[r, t] = sum_k A[id[t], r, k] * x[k, t], with `a_stride` the elements
/// between adapter slots.
///
/// One 128-thread block per (token, rank row). A single warp per row was
/// latency-bound at large k -- the 27b's down_proj is k=17408, and 64 of those
/// per token cost more than the model's own GEMMs -- so the row gets four warps
/// of memory parallelism and 16-byte loads (both operands are contiguous over
/// k). The reduction is a fixed shared-memory tree, never atomics: blockDim is
/// a constant, so the summation order is identical every launch and eager and
/// captured replays agree bit for bit.
__global__ void lora_batched_shrink_kernel(const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ a_bank,
                                           const std::int32_t* __restrict__ ids,
                                           __nv_bfloat16* __restrict__ low, std::int32_t k,
                                           std::int32_t rank, std::int32_t tokens,
                                           std::int64_t a_stride,
                                           const std::int32_t* __restrict__ uniform) {
    const int token = static_cast<int>(blockIdx.y);
    const int row   = static_cast<int>(blockIdx.x);
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
    if ((k & 7) == 0 && (reinterpret_cast<std::uintptr_t>(a_row) & 15U) == 0 &&
        (reinterpret_cast<std::uintptr_t>(x_col) & 15U) == 0) {
        const auto* a_vec = reinterpret_cast<const uint4*>(a_row);
        const auto* x_vec = reinterpret_cast<const uint4*>(x_col);
        const int vecs    = k >> 3;
        for (int i = static_cast<int>(threadIdx.x); i < vecs; i += kLoraShrinkThreads) {
            const uint4 a8 = a_vec[i];
            const uint4 x8 = x_vec[i];
            sum += lora_dot2(a8.x, x8.x) + lora_dot2(a8.y, x8.y) + lora_dot2(a8.z, x8.z) +
                   lora_dot2(a8.w, x8.w);
        }
    } else {
        for (int i = static_cast<int>(threadIdx.x); i < k; i += kLoraShrinkThreads) {
            sum += __bfloat162float(a_row[i]) * __bfloat162float(x_col[i]);
        }
    }

    __shared__ float partial[kLoraShrinkThreads];
    partial[threadIdx.x] = sum;
    __syncthreads();
#pragma unroll
    for (int stride = kLoraShrinkThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        low[static_cast<std::int64_t>(token) * rank + row] = __float2bfloat16(partial[0]);
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
