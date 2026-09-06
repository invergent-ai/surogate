#pragma once

// sinfer::ops -- head_linear kernel. One warp per output row of one head, over a tile of
// `kTokens` columns; the head's slice of the activation tile is staged in shared memory once
// per block and read by every warp. Only the launcher includes this.

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace sinfer::ops::detail {

constexpr int kHeadLinearThreads = 256;
constexpr int kHeadLinearWarps   = kHeadLinearThreads / 32;
/// W8G32: an int8 code and one FP16 scale per group of this many columns.
constexpr int kHeadLinearGroup = 32;

/// grid = (ceil(n / warps), heads, ceil(T / kTokens)); dynamic shared = kTokens * k * 2 bytes.
///
/// A lane walks the columns `lane, lane + 32, ...`, so a warp's loads of a weight row and of
/// the staged activation are contiguous, and every lane's column lands in group `col / 32`
/// which is the same for all 32 lanes of one step -- one scale read per step, broadcast.
template <int kTokens>
__global__ __launch_bounds__(kHeadLinearThreads) void head_linear_kernel(
    const __nv_bfloat16* __restrict__ x, const std::int8_t* __restrict__ codes,
    const __half* __restrict__ scales, int heads, int n, int k, int code_stride,
    int scale_stride, int tokens, float out_scale, __nv_bfloat16* __restrict__ out) {
    extern __shared__ __nv_bfloat16 tile[]; // [kTokens][k]
    const int head                = static_cast<int>(blockIdx.y);
    const int t0                  = static_cast<int>(blockIdx.z) * kTokens;
    const int nt                  = min(kTokens, tokens - t0);
    const std::int64_t x_stride   = static_cast<std::int64_t>(heads) * k;
    const std::int64_t out_stride = static_cast<std::int64_t>(heads) * n;
    const std::int64_t x_head     = static_cast<std::int64_t>(head) * k;

    // The tile rows past `nt` are zeroed rather than skipped: every warp then reads a defined
    // value for every `kTokens` slot and only the store is guarded.
    for (int i = static_cast<int>(threadIdx.x); i < kTokens * k; i += kHeadLinearThreads) {
        const int tt = i / k;
        const int kk = i - tt * k;
        tile[i]      = tt < nt ? x[(t0 + tt) * x_stride + x_head + kk] : __float2bfloat16_rn(0.0F);
    }
    __syncthreads();

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int row_in_head = static_cast<int>(blockIdx.x) * kHeadLinearWarps + warp;
    if (row_in_head >= n) { return; }
    const std::int64_t row  = static_cast<std::int64_t>(head) * n + row_in_head;
    const std::int8_t* wrow = codes + row * code_stride;
    const __half* srow      = scales + row * scale_stride;

    float acc[kTokens];
#pragma unroll
    for (int tt = 0; tt < kTokens; ++tt) { acc[tt] = 0.0F; }
    for (int kk = lane; kk < k; kk += 32) {
        const float wv = static_cast<float>(wrow[kk]) * __half2float(srow[kk / kHeadLinearGroup]);
#pragma unroll
        for (int tt = 0; tt < kTokens; ++tt) {
            acc[tt] = fmaf(wv, __bfloat162float(tile[tt * k + kk]), acc[tt]);
        }
    }
#pragma unroll
    for (int tt = 0; tt < kTokens; ++tt) {
        const float sum = warp_reduce_sum(acc[tt]);
        if (lane == 0 && tt < nt) {
            out[(t0 + tt) * out_stride + row] = __float2bfloat16_rn(sum * out_scale);
        }
    }
}

} // namespace sinfer::ops::detail
