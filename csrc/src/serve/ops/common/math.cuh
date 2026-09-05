#pragma once

#include "ops/common/math.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace sinfer::ops {

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

/// SwiGLU with the clamp a checkpoint may have been trained under.
///
/// GLM-5.3 bounds both halves before the product -- the gate from above only, the linear half
/// from both sides -- so a value that saturated in training saturates here too. `limit <= 0`
/// is the unclamped product every other mixture computes, and compiles to exactly that when
/// the limit is a compile-time constant.
__device__ __forceinline__ float swiglu_clamped(float gate, float up, float limit) {
    if (limit > 0.0f) {
        gate = fminf(gate, limit);
        up   = fminf(fmaxf(up, -limit), limit);
    }
    return silu(gate) * up;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplus(float x) { return (x > 20.0f) ? x : log1pf(expf(x)); }

__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
    std::uint32_t out;
    const std::uint32_t lo_bits = __float_as_uint(lo);
    const std::uint32_t hi_bits = __float_as_uint(hi);
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(hi_bits), "r"(lo_bits));
    return out;
}

__device__ __forceinline__ float2 bf16x2_to_float2(__nv_bfloat162 value) {
    return __bfloat1622float2(value);
}

__device__ __forceinline__ float2 bf16x2_bits_to_float2(std::uint32_t bits) {
    return bf16x2_to_float2(load_vec<__nv_bfloat162>(&bits));
}

__device__ __forceinline__ __half2 half2_from_bits(std::uint32_t bits) {
    return load_vec<__half2>(&bits);
}

} // namespace sinfer::ops
