#pragma once

// Implements: include/ninfer/ops/gelu.h
// Match: contiguous BF16 storage. Exact-erf and tanh formulas share aligned
// BF16x8/BF16x2/scalar storage routes but remain separate compile-time math.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGeluPairsPerThread = 4;

template <bool TanhApprox>
__device__ __forceinline__ float gelu_one(float value) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    constexpr float kSqrt2Pi  = 0.79788456080286535588f;
    if constexpr (TanhApprox) {
        return 0.5f * value *
               (1.0f + tanhf(kSqrt2Pi * (value + 0.044715f * value * value * value)));
    }
    return 0.5f * value * (1.0f + erff(value * kInvSqrt2));
}

template <bool TanhApprox>
__device__ __forceinline__ __nv_bfloat162 gelu_pair(__nv_bfloat162 input) {
    return __floats2bfloat162_rn(gelu_one<TanhApprox>(__low2float(input)),
                                 gelu_one<TanhApprox>(__high2float(input)));
}

template <bool TanhApprox, int Block>
__launch_bounds__(Block) __global__ void gelu_bf16x8_kernel(Bf16x8Pack* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        Bf16x8Pack value = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            value.pair[pair] = gelu_pair<TanhApprox>(value.pair[pair]);
        }
        store_vec(x + i, value);
    }
}

template <bool TanhApprox, int Block>
__launch_bounds__(Block) __global__
    void gelu_bf16x2_kernel(__nv_bfloat162* x, std::int64_t pairs, __nv_bfloat16* scalar,
                            bool has_scalar) {
    const std::int64_t first =
        (static_cast<std::int64_t>(blockIdx.x) * Block + threadIdx.x) * kGeluPairsPerThread;
#pragma unroll
    for (int item = 0; item < kGeluPairsPerThread; ++item) {
        const std::int64_t pair = first + item;
        if (pair < pairs) { x[pair] = gelu_pair<TanhApprox>(x[pair]); }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && has_scalar) {
        *scalar = __float2bfloat16_rn(gelu_one<TanhApprox>(__bfloat162float(*scalar)));
    }
}

template <bool TanhApprox>
__global__ void gelu_kernel(__nv_bfloat16* x, std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(gelu_one<TanhApprox>(__bfloat162float(x[i])));
    }
}

} // namespace ninfer::ops
