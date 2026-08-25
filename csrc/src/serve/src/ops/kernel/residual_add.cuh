#pragma once

// Implements: include/ninfer/ops/residual_add.h
// Match: contiguous BF16 inputs. Registered aligned/eight-element domains use
// one 16-byte pack per thread; BF16x2 and scalar routes preserve correctness for
// smaller alignments and odd tails.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kResidualAddPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 residual_add_pair(__nv_bfloat162 y, __nv_bfloat162 x) {
    const float r0 = __low2float(x) + __low2float(y);
    const float r1 = __high2float(x) + __high2float(y);
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void residual_add_scalar_kernel(const __nv_bfloat16* y, __nv_bfloat16* x,
                                           std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(y[i]));
    }
}

__launch_bounds__(256) __global__
    void residual_add_bf16x8_kernel(const Bf16x8Pack* y, Bf16x8Pack* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const Bf16x8Pack yv = load_vec<Bf16x8Pack>(y + i);
        Bf16x8Pack xv       = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            xv.pair[pair] = residual_add_pair(yv.pair[pair], xv.pair[pair]);
        }
        store_vec(x + i, xv);
    }
}

__launch_bounds__(256) __global__
    void residual_add_bf16x2_kernel(const __nv_bfloat16* y, __nv_bfloat16* x, std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kResidualAddPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* y2 = reinterpret_cast<const __nv_bfloat162*>(y);
    auto* x2       = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t j = tid * kResidualAddPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 y0 = y2[j];
        const __nv_bfloat162 x0 = x2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 y1  = y2[j + 1];
            const __nv_bfloat162 x1  = x2[j + 1];
            const __nv_bfloat162 y2v = y2[j + 2];
            const __nv_bfloat162 x2v = x2[j + 2];
            const __nv_bfloat162 y3  = y2[j + 3];
            const __nv_bfloat162 x3  = x2[j + 3];
            x2[j]                    = residual_add_pair(y0, x0);
            x2[j + 1]                = residual_add_pair(y1, x1);
            x2[j + 2]                = residual_add_pair(y2v, x2v);
            x2[j + 3]                = residual_add_pair(y3, x3);
        } else {
            x2[j] = residual_add_pair(y0, x0);
            if (j + 1 < n2) { x2[j + 1] = residual_add_pair(y2[j + 1], x2[j + 1]); }
            if (j + 2 < n2) { x2[j + 2] = residual_add_pair(y2[j + 2], x2[j + 2]); }
        }
    }

    if (tid == 0 && (n & 1) != 0) {
        const std::int64_t i = n - 1;
        x[i]                 = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(y[i]));
    }
}

} // namespace ninfer::ops
