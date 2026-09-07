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

/// The tanh approximation of GELU, which is what `gelu_pytorch_tanh` names and what every
/// Gemma is trained with. Not interchangeable with the erf form at this precision: the two
/// differ by ~1e-3 around |x| = 2, which is where a gated activation spends most of its mass.
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    const float inner            = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/// One expert's gated activation: the gate through `Activation`, times the linear half, with
/// the clamp a checkpoint may have been trained under.
///
/// The kind is a compile-time constant of the registered geometry, so this collapses to the one
/// activation that mixture uses. SiLU is what every routed mixture here ran until Gemma 4, whose
/// experts are GELU-gated like the rest of the model -- serving them through SiLU is serving a
/// different function, not a rounding difference.
__device__ __forceinline__ float gated_clamped(float gate, float up, float limit,
                                               GatedActivation activation) {
    if (limit > 0.0f) {
        gate = fminf(gate, limit);
        up   = fminf(fmaxf(up, -limit), limit);
    }
    const float gated = activation == GatedActivation::GeluTanh ? gelu_tanh(gate) : silu(gate);
    return gated * up;
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
