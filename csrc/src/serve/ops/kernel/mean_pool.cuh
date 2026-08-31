#pragma once

// sinfer::ops — mean_pool kernel: out[h] = mean over tokens of x[h, t].
// One block per hidden row; the block strides the token axis and reduces in
// FP32. Included only by its launcher. See docs/op-development.md §6.

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops {

inline constexpr int kMeanPoolBlock = 256;

// `scale` is 1/count for a mean and 1.0 for a running sum; `accumulate` adds
// into out rather than overwriting, so a chunked prefill can pool across calls.
__launch_bounds__(kMeanPoolBlock) __global__
    void mean_pool_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
                          std::int32_t hidden, std::int32_t count, float scale, bool accumulate) {
    const std::int32_t row = blockIdx.x;
    if (row >= hidden) { return; }

    float sum = 0.0f;
    for (std::int32_t t = threadIdx.x; t < count; t += kMeanPoolBlock) {
        sum += __bfloat162float(x[static_cast<std::int64_t>(t) * hidden + row]);
    }

    __shared__ float partial[kMeanPoolBlock];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kMeanPoolBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float value = partial[0] * scale;
        out[row] = __float2bfloat16(accumulate ? __bfloat162float(out[row]) + value : value);
    }
}

} // namespace sinfer::ops
