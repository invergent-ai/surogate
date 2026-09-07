#pragma once

// sinfer::ops — masked softmax over one encoder score matrix.
//
// Scores arrive column-major as s[key, query]: the key axis is contiguous, which
// is the axis the softmax runs over, so each block reads one stride-1 run. One
// block per (head, query); the block reduces max and sum in FP32 and writes the
// probabilities as BF16 for the second GEMM.
//
// Included only by its launcher. See docs/op-development.md §6.

#include <cuda_bf16.h>

#include <cfloat>
#include <cstdint>

namespace sinfer::ops {

inline constexpr int kEncoderSoftmaxBlock = 256;

__device__ __forceinline__ float block_reduce(float value, float* shared, bool take_max) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kEncoderSoftmaxBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = shared[threadIdx.x + stride];
            shared[threadIdx.x] =
                take_max ? fmaxf(shared[threadIdx.x], other) : shared[threadIdx.x] + other;
        }
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

/// `window` 0 admits every key; positive W admits abs(query - key) < W.
__launch_bounds__(kEncoderSoftmaxBlock) __global__
    void encoder_softmax_kernel(const float* __restrict__ scores, __nv_bfloat16* __restrict__ probs,
                                std::int32_t tokens, std::int32_t window, float scale) {
    const std::int64_t matrix = static_cast<std::int64_t>(blockIdx.y) * tokens * tokens;
    const std::int32_t query  = blockIdx.x;
    const float* row          = scores + matrix + static_cast<std::int64_t>(query) * tokens;
    __nv_bfloat16* out        = probs + matrix + static_cast<std::int64_t>(query) * tokens;

    __shared__ float shared[kEncoderSoftmaxBlock];

    // The admitted band. A global layer takes everything; a local one takes a
    // symmetric window, which is what bidirectional attention means -- not the
    // causal half-window.
    const std::int32_t lo = window > 0 ? max(0, query - window + 1) : 0;
    const std::int32_t hi = window > 0 ? min(tokens, query + window) : tokens;

    float local_max = -FLT_MAX;
    for (std::int32_t key = lo + threadIdx.x; key < hi; key += kEncoderSoftmaxBlock) {
        local_max = fmaxf(local_max, row[key] * scale);
    }
    const float row_max = block_reduce(local_max, shared, true);

    float local_sum = 0.0f;
    for (std::int32_t key = lo + threadIdx.x; key < hi; key += kEncoderSoftmaxBlock) {
        local_sum += __expf(row[key] * scale - row_max);
    }
    const float row_sum = block_reduce(local_sum, shared, false);
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;

    // Masked-out keys must be written, not skipped: the buffer is scratch and
    // the second GEMM reads the whole row.
    for (std::int32_t key = threadIdx.x; key < tokens; key += kEncoderSoftmaxBlock) {
        const float value =
            (key >= lo && key < hi) ? __expf(row[key] * scale - row_max) * inv_sum : 0.0f;
        out[key] = __float2bfloat16(value);
    }
}

} // namespace sinfer::ops
