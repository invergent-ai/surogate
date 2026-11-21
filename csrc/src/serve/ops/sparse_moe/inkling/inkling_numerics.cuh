#pragma once

// Inkling-2-Mini MoE numerics: every rounding point of the vLLM image's sm_120 kernels
// (ref/src/vllm/models/inkling2/nvidia/ops/{moe,nvfp4,mxfp8,activation,moe_sm120}.py), written
// with explicit round-to-nearest intrinsics so neither nvcc's default FMA contraction nor any
// target flag can change a bit. The reference kernels run with `enable_fp_fusion=False`.
// Design: rl/engine-inkling-2026-11-21/lanes/d/DESIGN-moe.md section 1.

#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/sparse_moe/inkling/inkling_moe_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cfloat>
#include <cstdint>

namespace sinfer::ops::detail::inkling {

// fp32 words of the reference's constants (a Python double literal rounded to fp32 where Triton
// multiplies it into an fp32 tensor). Checked against the hex words in the op test.
inline constexpr float kInv1536   = 0x1.555556p-11F; // 0x3A2AAAAB, `1.0 / 1536.0`
inline constexpr float kOneSixth  = 0x1.555556p-3F;  // 0x3E2AAAAB, `1.0 / 6.0`
inline constexpr float kInv448    = 0x1.24924Ap-9F;  // 0x3B124925, `1.0 / 448.0`
inline constexpr float kInv25     = 0x1.47AE14p-5F;  // 0x3D23D70A, `1.0 / 25.0`
inline constexpr float kRouteScale = 8.0F;
inline constexpr float kTwoPowMinus127 = 0x1p-127F;  // 0x00400000 (subnormal)
inline constexpr float kTwoPow127      = 0x1p127F;

// ---------------------------------------------------------------------------------------------
// Small conversions
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float bf16_bits_to_f32(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ std::uint16_t f32_to_bf16_bits(float value) {
    const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&rounded);
}

/// fp32 -> e4m3fn, round to nearest even, saturating (Triton's `.to(tl.float8e4nv)` on sm_89+:
/// `cvt.rn.satfinite.e4m3x2.f32`).
__device__ __forceinline__ std::uint8_t f32_to_e4m3(float value) {
    return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

/// e4m3fn byte -> fp32 (exact).
__device__ __forceinline__ float e4m3_to_f32(std::uint8_t code) {
    const __half_raw half = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
    return __half2float(__half(half));
}

/// The two e4m3fn bytes of `pair` (low byte first) -> fp32 (exact).
__device__ __forceinline__ float2 e4m3x2_to_f32(std::uint16_t pair) {
    const __half2_raw halves =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
    return __half22float2(__half2(halves));
}

/// E8M0 exponent -> fp32 scale as `ops/moe_sm120.py` `_dequant_e4m3` builds it: e == 0 is the
/// subnormal 2^-127 (0x00400000), otherwise e << 23.
__device__ __forceinline__ float e8m0_to_f32(std::uint32_t exponent) {
    return __uint_as_float(exponent == 0U ? 0x00400000U : (exponent << 23));
}

/// The bf16 of one MXFP8 element: `(e4m3.to(fp32) * scale).to(bf16)`.
__device__ __forceinline__ __nv_bfloat162 mxfp8_dequant_pair(std::uint16_t codes, float scale) {
    const float2 values = e4m3x2_to_f32(codes);
    return __floats2bfloat162_rn(__fmul_rn(values.x, scale), __fmul_rn(values.y, scale));
}

// ---------------------------------------------------------------------------------------------
// e2m1 (NVFP4 values)
// ---------------------------------------------------------------------------------------------
/// `ops/nvfp4.py` `_e2m1_pair`: the e2m1x2 byte of (lo, hi) (lo in the low nibble) and both
/// values rounded to e2m1 (`cvt.rn.satfinite.e2m1x2.f32`, decoded back by `cvt.rn.f16x2.e2m1x2`).
__device__ __forceinline__ std::uint32_t e2m1_pair(float lo, float hi, float& value_lo,
                                                   float& value_hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    std::uint32_t byte;
    asm("{ .reg .b8 q; .reg .b32 h; .reg .b16 l, u; "
        "cvt.rn.satfinite.e2m1x2.f32 q, %4, %3; cvt.u32.u8 %0, q; "
        "cvt.rn.f16x2.e2m1x2 h, q; mov.b32 {l, u}, h; "
        "cvt.f32.f16 %1, l; cvt.f32.f16 %2, u; }"
        : "=r"(byte), "=f"(value_lo), "=f"(value_hi)
        : "f"(lo), "f"(hi));
    return byte;
#else
    // The reference's software path (`ops/nvfp4.py` `_e2m1`, Hopper): the same thresholds.
    const auto encode = [](float x, float& value) -> std::uint32_t {
        const float a      = fabsf(x);
        const std::uint32_t code = a <= 0.25F  ? 0U
                                   : a < 0.75F ? 1U
                                   : a <= 1.25F ? 2U
                                   : a < 1.75F ? 3U
                                   : a <= 2.5F ? 4U
                                   : a < 3.5F  ? 5U
                                   : a <= 5.0F ? 6U
                                               : 7U;
        constexpr float kValues[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
        const std::uint32_t sign = __float_as_uint(x) >> 31;
        value                    = sign != 0U ? -kValues[code] : kValues[code];
        return code | (sign << 3);
    };
    const std::uint32_t low  = encode(lo, value_lo);
    const std::uint32_t high = encode(hi, value_hi);
    return low | (high << 4);
#endif
}

// ---------------------------------------------------------------------------------------------
// NVFP4 four-over-six (`ops/nvfp4.py` `_quantize_columns` / `_quantize_block`)
// ---------------------------------------------------------------------------------------------
struct Nvfp4RowScales {
    float encoder; ///< min(div_rn(1536, amax), FLT_MAX)
    float decoder; ///< div_rn(1, encoder)
    float outer;   ///< amax * (1/1536)
};

__device__ __forceinline__ Nvfp4RowScales nvfp4_row_scales(float amax) {
    Nvfp4RowScales s;
    s.encoder = fminf(__fdiv_rn(1536.0F, amax), FLT_MAX);
    s.decoder = __fdiv_rn(1.0F, s.encoder);
    s.outer   = __fmul_rn(amax, kInv1536);
    return s;
}

struct Nvfp4Block {
    std::uint64_t codes; ///< byte k = columns (2k, 2k+1), low nibble = even column
    std::uint8_t scale;  ///< e4m3fn
};

/// One candidate of `_candidate`: packed codes and the pairwise-adjacent error sum.
__device__ __forceinline__ std::uint64_t nvfp4_candidate(const float (&x)[16], float inverse,
                                                         float scale, float outer, float& error) {
    std::uint64_t packed = 0;
    float pair_error[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float low = 0.0F;
        float high = 0.0F;
        const std::uint32_t byte =
            e2m1_pair(__fmul_rn(x[2 * k], inverse), __fmul_rn(x[2 * k + 1], inverse), low, high);
        packed |= static_cast<std::uint64_t>(byte & 0xFFU) << (8 * k);
        const float error_low  = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(low, scale), outer), x[2 * k]));
        const float error_high =
            fabsf(__fsub_rn(__fmul_rn(__fmul_rn(high, scale), outer), x[2 * k + 1]));
        pair_error[k] = __fadd_rn(error_low, error_high);
    }
    const float quad0  = __fadd_rn(pair_error[0], pair_error[1]);
    const float quad1  = __fadd_rn(pair_error[2], pair_error[3]);
    const float quad2  = __fadd_rn(pair_error[4], pair_error[5]);
    const float quad3  = __fadd_rn(pair_error[6], pair_error[7]);
    const float octet0 = __fadd_rn(quad0, quad1);
    const float octet1 = __fadd_rn(quad2, quad3);
    error              = __fadd_rn(octet0, octet1);
    return packed;
}

/// One 16-column block of a row with row scales `row`.
__device__ __forceinline__ Nvfp4Block nvfp4_quantize_block_4o6(const float (&x)[16],
                                                               const Nvfp4RowScales& row) {
    float block_max = fabsf(x[0]);
#pragma unroll
    for (int j = 1; j < 16; ++j) { block_max = fmaxf(block_max, fabsf(x[j])); }
    const std::uint8_t code6 =
        f32_to_e4m3(fminf(__fmul_rn(block_max, __fmul_rn(row.encoder, kOneSixth)), 448.0F));
    const std::uint8_t code4 =
        f32_to_e4m3(fminf(__fmul_rn(block_max, __fmul_rn(row.encoder, 0.25F)), 448.0F));
    const float scale6   = e4m3_to_f32(code6);
    const float scale4   = e4m3_to_f32(code4);
    const float inverse6 = fminf(__fdiv_rn(1.0F, __fmul_rn(scale6, row.decoder)), FLT_MAX);
    const float inverse4 = fminf(__fdiv_rn(1.0F, __fmul_rn(scale4, row.decoder)), FLT_MAX);
    float error6         = 0.0F;
    float error4         = 0.0F;
    const std::uint64_t packed6 = nvfp4_candidate(x, inverse6, scale6, row.outer, error6);
    const std::uint64_t packed4 = nvfp4_candidate(x, inverse4, scale4, row.outer, error4);
    const bool choose4          = error4 < error6;
    return {choose4 ? packed4 : packed6, choose4 ? code4 : code6};
}

// ---------------------------------------------------------------------------------------------
// MXFP8 (`ops/mxfp8.py` `_mxfp8_block`)
// ---------------------------------------------------------------------------------------------
/// The E8M0 exponent of a 32-block with max |x| `amax`, and the multiplier applied before the e4m3
/// rounding.
__device__ __forceinline__ std::uint32_t mxfp8_block_exponent(float amax, float& inverse) {
    const float raw = __fmul_rn(amax, kInv448);
    // `((bits(raw) + 0x7FFFFF) & 0x7F800000) >> 23` (raw >= 0; unsigned so a NaN cannot overflow).
    std::int32_t exponent =
        static_cast<std::int32_t>(((__float_as_uint(raw) + 0x007FFFFFU) & 0x7F800000U) >> 23);
    exponent = raw <= kTwoPowMinus127 ? 0 : min(exponent, 254);
    if (raw != raw) { exponent = 255; }
    const float scale = fmaxf(__uint_as_float(static_cast<std::uint32_t>(exponent) << 23),
                              kTwoPowMinus127);
    inverse = exponent == 0 ? kTwoPow127 : __frcp_rn(scale);
    if (exponent == 255) { inverse = 0.0F; }
    return static_cast<std::uint32_t>(exponent);
}

/// `clamp(x * inverse, +-448)` (NaN propagates) -> e4m3fn.
__device__ __forceinline__ std::uint8_t mxfp8_encode(float x, float inverse) {
    float value = __fmul_rn(x, inverse);
    if (value == value) { value = fminf(fmaxf(value, -448.0F), 448.0F); }
    return f32_to_e4m3(value);
}

// ---------------------------------------------------------------------------------------------
// Softcapped SwiGLU with the routing weight (`ops/activation.py` `_situ`)
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float situ_sigmoid_approx(float x) {
    float y;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 %0, t; }"
        : "=f"(y)
        : "f"(x));
    return y;
}

__device__ __forceinline__ float situ_tanh_approx(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

/// `_softcap(x, 4.0)`: `_div_rn(x, 4)` is the exact product with 0.25.
__device__ __forceinline__ float situ_softcap4(float x) {
    return __fmul_rn(4.0F, situ_tanh_approx(__fmul_rn(x, 0.25F)));
}

/// `_softcap(x, 25.0)`: `_div_rn(x, 25)` = product with the rounded reciprocal and one FMA
/// correction step whose residual is clamped for infinite x.
__device__ __forceinline__ float situ_softcap25(float x) {
    float q              = __fmul_rn(x, kInv25);
    const float residual = fmaxf(__fmaf_rn(q, 25.0F, -x), -FLT_MAX);
    q                    = __fmaf_rn(-residual, kInv25, q);
    return __fmul_rn(25.0F, situ_tanh_approx(q));
}

/// y = bf16(((softcap4(g) * sigmoid(g)) * softcap25(u)) * gamma), g / u the bf16 gate / up.
__device__ __forceinline__ __nv_bfloat16 situ(float gate, float up, float gamma) {
    const float sigmoid = situ_sigmoid_approx(gate);
    const float g       = situ_softcap4(gate);
    const float u       = situ_softcap25(up);
    return __float2bfloat16_rn(__fmul_rn(__fmul_rn(__fmul_rn(g, sigmoid), u), gamma));
}

// ---------------------------------------------------------------------------------------------
// Router (`ops/moe.py` `_selection_keys`, `_log_sigmoid`, `_routing_weights`, `_sum8`)
// ---------------------------------------------------------------------------------------------
/// Order-preserving bits of sigmoid(logit) + bias in the high word, (E - expert) in the low 16
/// bits: the larger key wins, ties go to the lower expert id.
__device__ __forceinline__ std::uint64_t router_key(float logit, float bias, int expert) {
    const float sigmoid   = __fdiv_rn(1.0F, __fadd_rn(1.0F, expf(-logit)));
    const float selection = __fadd_rn(sigmoid, bias);
    std::uint32_t bits    = __float_as_uint(selection);
    bits ^= (bits & 0x80000000U) != 0U ? 0xFFFFFFFFU : 0x80000000U;
    return (static_cast<std::uint64_t>(bits) << 16) |
           static_cast<std::uint64_t>(kExperts - expert);
}

__device__ __forceinline__ int router_key_expert(std::uint64_t key) {
    return kExperts - static_cast<int>(key & 0xFFFFU);
}

__device__ __forceinline__ float router_log_sigmoid(float logit) {
    return __fsub_rn(fminf(logit, 0.0F), log1pf(expf(-fabsf(logit))));
}

/// Routing weights of the 8 selected experts in slot order from their log-sigmoids.
__device__ __forceinline__ void router_weights(const float (&log_probs)[kTopK],
                                               float (&weights)[kTopK]) {
    float maximum = log_probs[0];
#pragma unroll
    for (int s = 1; s < kTopK; ++s) { maximum = fmaxf(maximum, log_probs[s]); }
    float shifted[kTopK];
#pragma unroll
    for (int s = 0; s < kTopK; ++s) { shifted[s] = expf(__fsub_rn(log_probs[s], maximum)); }
    const float total = __fadd_rn(__fadd_rn(__fadd_rn(shifted[0], shifted[2]),
                                            __fadd_rn(shifted[1], shifted[3])),
                                  __fadd_rn(__fadd_rn(shifted[4], shifted[6]),
                                            __fadd_rn(shifted[5], shifted[7])));
    const float log_sum_exp = __fadd_rn(maximum, logf(total));
#pragma unroll
    for (int s = 0; s < kTopK; ++s) {
        weights[s] = __fmul_rn(expf(__fsub_rn(log_probs[s], log_sum_exp)), kRouteScale);
    }
}

// ---------------------------------------------------------------------------------------------
// Warp helpers
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__device__ __forceinline__ std::uint64_t warp_max_u64(std::uint64_t value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const std::uint64_t other = __shfl_xor_sync(0xFFFFFFFFU, value, offset);
        value                     = other > value ? other : value;
    }
    return value;
}

/// Fixed-tree warp sum (xor butterfly): the same association on every call.
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

} // namespace sinfer::ops::detail::inkling
