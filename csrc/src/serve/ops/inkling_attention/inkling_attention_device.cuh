#pragma once

// Device helpers shared by the Inkling attention kernels. Every arithmetic
// helper here reproduces one operation of vLLM's Inkling2 sm_120 kernels exactly as they issue it
// (attention design notes, section 1): explicit PTX for the approximate instructions, `__f*_rn`
// where the reference compiles with `enable_fp_fusion=False` (nvcc contracts `a * b + c` by default).

#include "api/ops/inkling_attention.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace sinfer::ops::inkling_detail {

using Geo = InklingAttentionGeometry;

inline constexpr float kLog2e = 1.4426950408889634F;
inline constexpr float kLn2   = 0.6931471805599453F;
// Python's `(width + 1) ** -0.5` and `(2 * width + 1) ** -0.5` for width 4, passed as fp32.
inline constexpr float kStaticConvScale  = 0.44721359549995793F; // 5 ** -0.5
inline constexpr float kDynamicConvScale = 0.33333333333333333F; // 9 ** -0.5

// KV pages: the engine's paged layout (paged_kv_address.cuh) at head dim 128, 8 KV heads.
inline constexpr int kPageTokens = 64;
inline constexpr int kPageShift  = 6;

__device__ __forceinline__ std::int64_t kv_page_offset(std::int32_t physical_page, int kv_head,
                                                       int slot) {
    return (static_cast<std::int64_t>(physical_page) * Geo::kKvHeads + kv_head) *
               (kPageTokens * Geo::kHeadDim) +
           static_cast<std::int64_t>(slot) * Geo::kHeadDim;
}

// ---- approximate instructions, as the reference issues them -------------------------------------

/// `ex2.approx.f32` (non-FTZ): Triton's tl.exp2 and the reference's `_sigmoid`.
__device__ __forceinline__ float ptx_ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
/// `ex2.approx.ftz.f32`: the split combine's `_ex2`.
__device__ __forceinline__ float ptx_ex2_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float ptx_rcp(float x) {
    float y;
    asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
/// `lg2.approx.f32`: Triton's tl.log2.
__device__ __forceinline__ float ptx_lg2(float x) {
    float y;
    asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
/// `lg2.approx.ftz.f32`: the split combine's `_lg2`.
__device__ __forceinline__ float ptx_lg2_ftz(float x) {
    float y;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
/// `rsqrt.approx.f32` (non-FTZ): the reference's `rsqrt` (ops/norm.py).
__device__ __forceinline__ float ptx_rsqrt(float x) {
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
/// `div.full.f32`: Triton's default fp32 division (unverified assumption, DESIGN section 1 A5).
__device__ __forceinline__ float ptx_div_full(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

/// m16n8k8 bf16 MMA (fp32 accumulate): a0 / a1 = rows gid / gid + 8, k 2 lid .. + 1; b0 = k 2 lid .. + 1, n gid.
__device__ __forceinline__ void mma_bf16_k8(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
                                            unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

/// m16n8k16 bf16 MMA (fp32 accumulate) WITHOUT `volatile`: the same instruction as ops::mma_bf16 (same bits), but the
/// compiler may interleave independent chains around it (the volatile form keeps every MMA in program order).
__device__ __forceinline__ void mma_bf16_nv(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
                                            unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/// Ask L2 to fetch [ptr, ptr + bytes) (bytes a multiple of 16; no registers, no shared memory, no completion to wait for).
__device__ __forceinline__ void prefetch_l2_bulk(const void* ptr, unsigned bytes) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr), "r"(bytes) : "memory");
#else
    // cp.async.bulk is sm_90+; below it the hint is dropped (the sm_89 cubin of the serve build must compile).
    (void)ptr;
    (void)bytes;
#endif
}

/// Non-coherent global loads the compiler may not move (a register ring's distance stays as written).
__device__ __forceinline__ uint2 ldg_pinned_v2(const void* ptr) {
    uint2 v;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr));
    return v;
}
__device__ __forceinline__ uint4 ldg_pinned_v4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

/// The reference's `_sigmoid` (ops/attention.py): rcp.approx(ex2.approx(-x * log2 e) + 1).
__device__ __forceinline__ float ref_sigmoid(float x) {
    return ptx_rcp(__fadd_rn(ptx_ex2(__fmul_rn(-x, kLog2e)), 1.0F));
}

/// bf16 round trip of an fp32 value (RNE).
__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

/// Four fp32 values to four unit-scale e4m3 codes (RNE, satfinite), packed little-endian.
__device__ __forceinline__ std::uint32_t e4m3x4(float a, float b, float c, float d) {
    const std::uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const std::uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return lo | (hi << 16);
}

/// Eight e4m3 codes (an int2) widened exactly to eight bf16 (an int4): e4m3 -> f16 (cvt.rn.f16x2.e4m3x2,
/// as the reference's `_load_fp8_kv`) -> bf16, both exact.
__device__ __forceinline__ int4 e4m3x8_to_bf16x8(int2 raw) {
    const std::uint32_t words[2] = {static_cast<std::uint32_t>(raw.x), static_cast<std::uint32_t>(raw.y)};
    std::uint32_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint16_t pair = static_cast<std::uint16_t>(words[i >> 1] >> ((i & 1) * 16));
        const __half2_raw h      = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
        const __half2 hv         = *reinterpret_cast<const __half2*>(&h);
        packed[i]                = pack_bf16x2(__low2float(hv), __high2float(hv));
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

/// Sum of squares of a warp's 32 lanes x N values: each lane adds its own squares left to right, then a
/// xor butterfly (every lane ends with the same bits). Products and sums rounded separately, as the
/// reference's `tl.sum(x * x)` with fusion off. Width 16 splits the warp into two independent halves.
template <int Width, int N>
__device__ __forceinline__ float sum_squares(const float (&x)[N]) {
    float s = __fmul_rn(x[0], x[0]);
#pragma unroll
    for (int i = 1; i < N; ++i) { s = __fadd_rn(s, __fmul_rn(x[i], x[i])); }
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        s = __fadd_rn(s, __shfl_xor_sync(kFullWarpMask, s, offset, Width));
    }
    return s;
}

/// The L2-clip factor of `_l2_clip` / `_kv_conv_tile`: min(threshold / (||x|| + eps), 1), ||x|| = ss * rsqrt(ss).
__device__ __forceinline__ float l2_clip_factor(float ss, float threshold) {
    const float norm = ss > 0.0F ? __fmul_rn(ss, ptx_rsqrt(ss)) : 0.0F;
    return fminf(__fdiv_rn(threshold, __fadd_rn(norm, Geo::kClipEps)), 1.0F);
}

} // namespace sinfer::ops::inkling_detail
