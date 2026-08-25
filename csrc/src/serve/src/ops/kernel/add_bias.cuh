#pragma once

// Implements: include/ninfer/ops/add_bias.h
// Match: contiguous BF16 [D,C]. Aligned cache-sized registered domains use
// 16-byte channel packs with bias reuse across columns; large domains retain a
// BF16x2 stream, and scalar indexing covers odd/unaligned storage.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kAddBiasPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 add_bias_pair(__nv_bfloat162 value, __nv_bfloat162 bias) {
    return __floats2bfloat162_rn(__low2float(value) + __low2float(bias),
                                 __high2float(value) + __high2float(bias));
}

template <int Block, int RowsPerBlock>
__launch_bounds__(Block) __global__
    void add_bias_bf16x8_kernel(const Bf16x8Pack* bias, Bf16x8Pack* x, std::int32_t packs,
                                std::int32_t rows) {
    const int pack = static_cast<int>(blockIdx.x) * Block + static_cast<int>(threadIdx.x);
    if (pack >= packs) { return; }
    const Bf16x8Pack bv = load_vec<Bf16x8Pack>(bias + pack);
    const int first_row = static_cast<int>(blockIdx.y) * RowsPerBlock;
#pragma unroll
    for (int item = 0; item < RowsPerBlock; ++item) {
        const int row = first_row + item;
        if (row >= rows) { return; }
        const std::int64_t offset = static_cast<std::int64_t>(row) * packs + pack;
        Bf16x8Pack value          = load_vec<Bf16x8Pack>(x + offset);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            value.pair[pair] = add_bias_pair(value.pair[pair], bv.pair[pair]);
        }
        store_vec(x + offset, value);
    }
}

template <int Block>
__launch_bounds__(Block) __global__
    void add_bias_bf16x2_kernel(const __nv_bfloat162* bias, __nv_bfloat162* x, std::int32_t pairs,
                                std::int32_t rows) {
    const std::int32_t first =
        (static_cast<std::int32_t>(blockIdx.x) * Block + static_cast<std::int32_t>(threadIdx.x)) *
        kAddBiasPairsPerThread;
    for (std::int32_t row = static_cast<std::int32_t>(blockIdx.y); row < rows;
         row += static_cast<std::int32_t>(gridDim.y)) {
        const std::int64_t base = static_cast<std::int64_t>(row) * pairs;
#pragma unroll
        for (int item = 0; item < kAddBiasPairsPerThread; ++item) {
            const std::int32_t pair = first + item;
            if (pair < pairs) { x[base + pair] = add_bias_pair(x[base + pair], bias[pair]); }
        }
    }
}

__global__ void add_bias_kernel(const __nv_bfloat16* bias, __nv_bfloat16* x, std::int32_t d,
                                std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float value = __bfloat162float(x[i]) + __bfloat162float(bias[i % d]);
        x[i]              = __float2bfloat16_rn(value);
    }
}

} // namespace ninfer::ops
