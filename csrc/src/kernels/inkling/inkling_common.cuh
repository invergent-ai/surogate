// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer kernels: device helpers shared by the forward / backward kernels.
//
// Forward kernels reproduce the serving engine's (and vLLM's sm_120 reference's) fp32 operations instruction for
// instruction: the same approximate PTX (rsqrt.approx, ex2.approx, rcp.approx, tanh.approx), IEEE division where
// Triton uses div_rn, and NO FMA contraction where Triton compiles with enable_fp_fusion=False -- written with the
// explicit __fmul_rn / __fadd_rn / fmaf intrinsics because nvcc contracts a*b+c by default (-fmad=true).
// Backward kernels use exact fp32 derivatives (accurate expf / tanhf).

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace inkling {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr float kFp32Max = 3.4028234663852886e38f;

__device__ __forceinline__ float rsqrt_approx(float x) {  // non-FTZ, as cute.math.rsqrt / ops/norm.py rsqrt
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float ex2_approx_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float rcp_approx(float x) {
    float y;
    asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float rcp_approx_ftz(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float tanh_approx(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float lg2_approx(float x) {
    float y;
    asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

/// `ops/attention.py` `_sigmoid` (gate / sink): rcp.approx(ex2.approx(-x * log2e) + 1), no FMA.
__device__ __forceinline__ float sigmoid_gate_sink(float x) {
    return rcp_approx(__fadd_rn(ex2_approx(__fmul_rn(-x, kLog2e)), 1.0f));
}
/// `ops/activation.py` `_sigmoid_approx` (SiTU): mul.rn by -log2e (0fBFB8AA3B), ex2.approx.ftz, add.rn 1, rcp.approx.ftz.
__device__ __forceinline__ float sigmoid_situ(float x) {
    return rcp_approx_ftz(__fadd_rn(ex2_approx_ftz(__fmul_rn(x, __int_as_float(0xBFB8AA3B))), 1.0f));
}
/// `ops/activation.py` `_div_rn(x, 25)`: product with the rounded reciprocal + one FMA correction (bitwise div_rn).
__device__ __forceinline__ float div_rn25(float x) {
    const float r = 1.0f / 25.0f;
    float q = __fmul_rn(x, r);
    const float residual = fmaxf(fmaf(q, 25.0f, -x), -kFp32Max);
    return fmaf(-residual, r, q);
}
/// `_softcap(x, CAP)` = CAP * tanh.approx(div_rn(x, CAP)), CAP in {4, 25}.
__device__ __forceinline__ float softcap4(float x) {
    return __fmul_rn(4.0f, tanh_approx(__fmul_rn(x, 0.25f)));
}
__device__ __forceinline__ float softcap25(float x) {
    return __fmul_rn(25.0f, tanh_approx(div_rn25(x)));
}

__device__ __forceinline__ float sigmoid_exact(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float bf16_round(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}
/// fp32 -> e4m3fn (RNE, satfinite: the engine's `cvt.rn.satfinite.e4m3x2.f32` KV store) -> fp32 (exact).
__device__ __forceinline__ float e4m3_round(float x) {
    const __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    const __half_raw h = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3);
    return __half2float(__half(h));
}

template <typename T>
__device__ __forceinline__ float ld(const T* p, long i);
template <>
__device__ __forceinline__ float ld<float>(const float* p, long i) {
    return p[i];
}
template <>
__device__ __forceinline__ float ld<nv_bfloat16>(const nv_bfloat16* p, long i) {
    return __bfloat162float(p[i]);
}

/// Warp sum with a fixed xor tree (deterministic for a fixed lane assignment).
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/// Block sum of `n` values per thread (n <= kMaxVals) in a fixed order: warp xor trees, then warp 0 sums the warp
/// partials in warp order. `scratch` holds >= 32 * n floats. Every thread gets the totals. blockDim.x % 32 == 0.
template <int kMaxVals>
__device__ __forceinline__ void block_sum(float (&v)[kMaxVals], int n, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int i = 0; i < n; ++i) v[i] = warp_sum(v[i]);
    __syncthreads();
    if (lane == 0)
        for (int i = 0; i < n; ++i) scratch[warp * n + i] = v[i];
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        float s = 0.f;
        for (int w = 0; w < warps; ++w) s += scratch[w * n + i];
        v[i] = s;
    }
    __syncthreads();
}

}  // namespace inkling
