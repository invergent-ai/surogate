// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#pragma once

#include "ops/linear/ggml/ggml_iq_tables.h"

#include <cuda_fp16.h>

#include <cstdint>

namespace sinfer::ops::detail::ggml {

// Bit casts and exp2 that compile the same for the host pass of a `__host__ __device__` decoder.
__host__ __device__ __forceinline__ float __uint_as_float_portable(std::uint32_t bits) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(bits);
#else
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}
__host__ __device__ __forceinline__ float exp2f_portable(int e) {
    // 2^e for -7 <= e <= 8: build the float directly from its exponent field.
    return __uint_as_float_portable(static_cast<std::uint32_t>(e + 127) << 23);
}

// Superblock geometry shared by every K-quant: 256 values, 12 bytes of 6-bit sub-scales.
inline constexpr int QK_K         = 256;
inline constexpr int K_SCALE_SIZE = 12;
// Q8_0: not a K-quant at all -- 32 values with one scale, no superblock and no sub-scales. A
// K_M quant leaves the attention, GDN and shared-expert projections in it, so serving it is what
// keeps those off the dequantise path.
inline constexpr int QK8_0 = 32;
inline constexpr int QR8_0 = 1;
inline constexpr int QI8_0 = QK8_0 / (4 * QR8_0);
inline constexpr int VDR_Q8_0_Q8_1_MMVQ = 2;

// The int8 activation block the vec-dots consume: 32 values, (d, sum) as half2.
inline constexpr int QK8_1 = 32;
inline constexpr int QR8_1 = 1;
inline constexpr int QI8_1 = QK8_1 / (4 * QR8_1);

inline constexpr int QR2_K = 4; inline constexpr int QI2_K = QK_K / (4 * QR2_K);
inline constexpr int QR3_K = 4; inline constexpr int QI3_K = QK_K / (4 * QR3_K);
inline constexpr int QR4_K = 2; inline constexpr int QI4_K = QK_K / (4 * QR4_K);
inline constexpr int QR5_K = 2; inline constexpr int QI5_K = QK_K / (4 * QR5_K);
inline constexpr int QR6_K = 2; inline constexpr int QI6_K = QK_K / (4 * QR6_K);

struct block_q8_0 {
    __half d;          // scale
    int8_t qs[QK8_0];  // quants
};
static_assert(sizeof(block_q8_0) == sizeof(__half) + QK8_0, "wrong q8_0 block size/padding");

// Q4_1 / Q5_1: 32 values with a scale and an additive minimum, x = d*q + m. Not K-quants and
// not repackable into W8 either -- W8 carries a scale and has nowhere to put the min. A
// quantiser writes these where the reduction axis is not a multiple of 256, so no superblock
// fits a row; a MoE down projection whose expert width is 640 is exactly that case.
inline constexpr int QK4_1 = 32;
inline constexpr int QR4_1 = 2;
inline constexpr int QI4_1 = QK4_1 / (4 * QR4_1);
inline constexpr int VDR_Q4_1_Q8_1_MMVQ = 2;

inline constexpr int QK5_1 = 32;
inline constexpr int QR5_1 = 2;
inline constexpr int QI5_1 = QK5_1 / (4 * QR5_1);
inline constexpr int VDR_Q5_1_Q8_1_MMVQ = 2;

struct block_q4_1 {
    __half2 dm;                 // d (scale), m (minimum)
    uint8_t qs[QK4_1 / 2];      // nibbles
};
static_assert(sizeof(block_q4_1) == 2 * sizeof(__half) + QK4_1 / 2, "wrong q4_1 block size/padding");

struct block_q5_1 {
    __half2 dm;                 // d (scale), m (minimum)
    uint8_t qh[4];              // fifth bit of each quant
    uint8_t qs[QK5_1 / 2];      // low nibbles
};
static_assert(sizeof(block_q5_1) == 2 * sizeof(__half) + 4 + QK5_1 / 2,
              "wrong q5_1 block size/padding");

// Q4_0 / Q5_0: 32 values, one scale, and codes biased by half their range -- `w = d*(q - 8)`
// for four bits, `d*(q - 16)` for five. No minimum and no table; the plainest of the family.
inline constexpr int QK4_0 = 32;
inline constexpr int QR4_0 = 2;
inline constexpr int QI4_0 = QK4_0 / (4 * QR4_0);
inline constexpr int VDR_Q4_0_Q8_1_MMVQ = 2;

inline constexpr int QK5_0 = 32;
inline constexpr int QR5_0 = 2;
inline constexpr int QI5_0 = QK5_0 / (4 * QR5_0);
inline constexpr int VDR_Q5_0_Q8_1_MMVQ = 2;

struct block_q4_0 {
    __half d;
    uint8_t qs[QK4_0 / 2];
};
static_assert(sizeof(block_q4_0) == sizeof(__half) + QK4_0 / 2, "wrong q4_0 block size/padding");

struct block_q5_0 {
    __half d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
};
static_assert(sizeof(block_q5_0) == sizeof(__half) + 4 + QK5_0 / 2,
              "wrong q5_0 block size/padding");

// IQ4_NL: 32 values, one scale, four-bit codes that index a sixteen-entry table of int8 levels
// rather than standing for their own magnitude. The levels are spaced non-linearly -- closer
// together near zero, where weights actually live -- which is what buys the accuracy over Q4_0
// at the same width.
inline constexpr int QK4_NL = 32;
inline constexpr int QR4_NL = 2;
inline constexpr int QI4_NL = QK4_NL / (4 * QR4_NL);
inline constexpr int VDR_IQ4_NL_Q8_1_MMVQ = 2;

/// ggml-common.h `kvalues_iq4nl`.
// Global rather than __constant__: read uniformly as four words by the byte-permute lookup
// and per value by the dequantisers, and a divergent constant read serialises.
__device__ static const int8_t kIq4nlValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

struct block_iq4_nl {
    __half d;                   // scale
    uint8_t qs[QK4_NL / 2];     // four-bit table indices
};
static_assert(sizeof(block_iq4_nl) == sizeof(__half) + QK4_NL / 2,
              "wrong iq4_nl block size/padding");

struct block_q8_1 {
    __half2 ds;          // d (scale), s (sum of the 32 unquantised values)
    int8_t  qs[QK8_1];   // quants
};
static_assert(sizeof(block_q8_1) == 2 * sizeof(__half) + QK8_1, "wrong q8_1 block size/padding");

// 2-bit: 16 blocks of 16, x = d*sc*q - dmin*m; 2.625 bits per weight.
struct block_q2_K {
    uint8_t scales[QK_K / 16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K / 4];      // quants
    __half2 dm;                // d: super-block scale for the scales, dmin: for the mins
};
static_assert(sizeof(block_q2_K) == 2 * sizeof(__half) + QK_K / 16 + QK_K / 4, "wrong q2_K block size/padding");

// 3-bit: 16 blocks of 16, x = d*sc*q; 3.4375 bits per weight.
struct block_q3_K {
    uint8_t hmask[QK_K / 8]; // quants - high bit
    uint8_t qs[QK_K / 4];    // quants - low 2 bits
    uint8_t scales[12];      // scales, quantized with 6 bits
    __half  d;               // super-block scale
};
static_assert(sizeof(block_q3_K) == sizeof(__half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// 4-bit: 8 blocks of 32, x = d*sc*q - dmin*m; 4.5 bits per weight.
struct block_q4_K {
    __half2 dm;                   // d: super-block scale for the scales, dmin: for the mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];         // 4-bit quants
};
static_assert(sizeof(block_q4_K) == 2 * sizeof(__half) + K_SCALE_SIZE + QK_K / 2, "wrong q4_K block size/padding");

// 5-bit: 8 blocks of 32, x = d*sc*q - dmin*m; 5.5 bits per weight.
struct block_q5_K {
    __half2 dm;                   // d, dmin as above
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K / 8];         // quants, high bit
    uint8_t qs[QK_K / 2];         // quants, low 4 bits
};
static_assert(sizeof(block_q5_K) == 2 * sizeof(__half) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8, "wrong q5_K block size/padding");

// 6-bit: 16 blocks of 16, x = d*sc*q; 6.5625 bits per weight.
struct block_q6_K {
    uint8_t ql[QK_K / 2];      // quants, lower 4 bits
    uint8_t qh[QK_K / 4];      // quants, upper 2 bits
    int8_t  scales[QK_K / 16]; // scales, quantized with 8 bits
    __half  d;                 // super-block scale
};
static_assert(sizeof(block_q6_K) == sizeof(__half) + QK_K / 16 + 3 * QK_K / 4, "wrong q6_K block size/padding");

// ---------------------------------------------------------------------------------------------
// The importance-matrix ("IQ") family. A block is still 256 values, but a value is not a code
// times a scale: it is an index into a codebook of eight-value rows (the grids in
// ggml_iq_tables.h) plus a sign bit, with a 4-bit scale per 32 or per 16. What varies by type
// is how many bits the index gets and where the signs live.
inline constexpr int QR2_XXS = 4; inline constexpr int QI2_XXS = QK_K / (4 * QR2_XXS);
inline constexpr int QR2_XS  = 4; inline constexpr int QI2_XS  = QK_K / (4 * QR2_XS);
inline constexpr int QR2_S   = 4; inline constexpr int QI2_S   = QK_K / (4 * QR2_S);
inline constexpr int QR3_XXS = 4; inline constexpr int QI3_XXS = QK_K / (4 * QR3_XXS);
inline constexpr int QR3_S   = 4; inline constexpr int QI3_S   = QK_K / (4 * QR3_S);
inline constexpr int QR1_S   = 8; inline constexpr int QI1_S   = QK_K / (4 * QR1_S);
inline constexpr int QR1_M   = 8; inline constexpr int QI1_M   = QK_K / (4 * QR1_M);
inline constexpr int QR4_XS  = 2; inline constexpr int QI4_XS  = QK_K / (4 * QR4_XS);
inline constexpr int VDR_IQ2_XXS_Q8_1_MMVQ = 2;
inline constexpr int VDR_IQ2_XS_Q8_1_MMVQ  = 2;
inline constexpr int VDR_IQ2_S_Q8_1_MMVQ   = 2;
inline constexpr int VDR_IQ3_XXS_Q8_1_MMVQ = 2;
inline constexpr int VDR_IQ3_S_Q8_1_MMVQ   = 2;
inline constexpr int VDR_IQ1_S_Q8_1_MMVQ   = 1;
inline constexpr int VDR_IQ1_M_Q8_1_MMVQ   = 1;
inline constexpr int VDR_IQ4_XS_Q8_1_MMVQ  = 4;
inline constexpr int IQ3S_N_SCALE = QK_K / 64;
// The 1-bit types carry a per-block offset from -1 by this much, its sign chosen per 32.
inline constexpr float IQ1S_DELTA = 0.125F;
inline constexpr float IQ1M_DELTA = 0.125F;

// 2.0625 bits: eight 8-bit grid indices per 32 values, seven sign bits per eight values and a
// 4-bit scale packed into the same 32-bit word.
struct block_iq2_xxs {
    __half   d;
    uint16_t qs[QK_K / 8];
};
static_assert(sizeof(block_iq2_xxs) == sizeof(__half) + QK_K / 4, "wrong iq2_xxs block size/padding");

// 2.3125 bits: 9-bit grid indices with the seven sign bits alongside, 4-bit scales per 16.
struct block_iq2_xs {
    __half   d;
    uint16_t qs[QK_K / 8];
    uint8_t  scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_xs) == sizeof(__half) + QK_K / 4 + QK_K / 32, "wrong iq2_xs block size/padding");

// 2.5 bits: 10-bit grid indices (8 in qs, 2 in qh), a full sign byte per eight values.
struct block_iq2_s {
    __half  d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_s) == sizeof(__half) + QK_K / 4 + QK_K / 16, "wrong iq2_s block size/padding");

// 3.0625 bits: 8-bit indices into a grid of four-value rows, signs and scale packed per 32.
struct block_iq3_xxs {
    __half  d;
    uint8_t qs[3 * QK_K / 8];
};
static_assert(sizeof(block_iq3_xxs) == sizeof(__half) + 3 * QK_K / 8, "wrong iq3_xxs block size/padding");

// 3.4375 bits: 9-bit indices (8 in qs, 1 in qh), a full sign byte per eight, 4-bit scales per 32.
struct block_iq3_s {
    __half  d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[IQ3S_N_SCALE];
};
static_assert(sizeof(block_iq3_s) == sizeof(__half) + 13 * QK_K / 32 + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

// 1.5625 bits: 11-bit indices into the 1-bit grid (8 in qs, 3 in qh), a 3-bit scale and the
// delta sign per 32, all inside qh's sixteen bits.
struct block_iq1_s {
    __half   d;
    uint8_t  qs[QK_K / 8];
    uint16_t qh[QK_K / 32];
};
static_assert(sizeof(block_iq1_s) == sizeof(__half) + QK_K / 8 + QK_K / 16, "wrong iq1_s block size/padding");

// 1.75 bits: the same grid, 3-bit scales per 16 and the block scale spread over the top nibbles
// of the four scale words -- there is no `d` field, iq1m_scale_t reassembles it.
struct block_iq1_m {
    uint8_t qs[QK_K / 8];
    uint8_t qh[QK_K / 16];
    uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq1_m) == QK_K / 8 + QK_K / 16 + QK_K / 32, "wrong iq1_m block size/padding");
union iq1m_scale_t {
    __half   f16;
    uint16_t u16;
};

// 4.25 bits: IQ4_NL's sixteen-level table over a 256-value superblock, 6-bit scales per 32
// split between scales_l (low nibbles) and scales_h (two bits each).
struct block_iq4_xs {
    __half   d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K / 64];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_iq4_xs) == sizeof(__half) + sizeof(uint16_t) + QK_K / 64 + QK_K / 2,
              "wrong iq4_xs block size/padding");

// ---------------------------------------------------------------------------------------------
// The ternary types (BitNet): values in {-1, 0, 1} times one scale per 256. TQ1_0 packs five
// trits per byte in base three; TQ2_0 spends two bits per value. llama.cpp serves neither on
// CUDA, so their vec-dots below are ours.
struct block_tq1_0 {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qh[QK_K / 64];                   // 4 elements per byte
    __half  d;
};
static_assert(sizeof(block_tq1_0) == sizeof(__half) + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5,
              "wrong tq1_0 block size/padding");
struct block_tq2_0 {
    uint8_t qs[QK_K / 4]; // 2 bits per element
    __half  d;
};
static_assert(sizeof(block_tq2_0) == sizeof(__half) + QK_K / 4, "wrong tq2_0 block size/padding");

// ---------------------------------------------------------------------------------------------
// The microscaling floats: four-bit E2M1 values (the sixteen-entry kFp4Values table, in
// half-units) under a shared exponent. MXFP4 is 32 values under one E8M0 exponent byte; NVFP4
// is 64 values in four sub-blocks of 16, each under a UE4M3 scale byte.
inline constexpr int QK_MXFP4 = 32;
inline constexpr int QR_MXFP4 = 2;
inline constexpr int QI_MXFP4 = QK_MXFP4 / (4 * QR_MXFP4);
inline constexpr int VDR_MXFP4_Q8_1_MMVQ = 2;
struct block_mxfp4 {
    uint8_t e; // E8M0
    uint8_t qs[QK_MXFP4 / 2];
};
static_assert(sizeof(block_mxfp4) == 1 + QK_MXFP4 / 2, "wrong mxfp4 block size/padding");

inline constexpr int QK_NVFP4     = 64;
inline constexpr int QK_NVFP4_SUB = 16;
inline constexpr int QR_NVFP4     = 2;
inline constexpr int QI_NVFP4     = QK_NVFP4 / (4 * QR_NVFP4);
inline constexpr int VDR_NVFP4_Q8_1_MMVQ = 4;
struct block_nvfp4 {
    uint8_t d[QK_NVFP4 / QK_NVFP4_SUB]; // UE4M3 scales, one per 16-value sub-block
    uint8_t qs[QK_NVFP4 / 2];           // packed 4-bit E2M1 values
};
static_assert(sizeof(block_nvfp4) == QK_NVFP4 / QK_NVFP4_SUB + QK_NVFP4 / 2, "wrong nvfp4 block size/padding");

// ---------------------------------------------------------------------------------------------
// The plain low-bit types: one scale, codes that stand for themselves. Q1_0 is a sign bit per
// value over 128; Q2_0 is two bits per value over 64, code - 1 in {-1, 0, 1, 2}.
inline constexpr int QK1_0 = 128;
inline constexpr int QR1_0 = 1;
inline constexpr int QI1_0 = QK1_0 / 32;
inline constexpr int VDR_Q1_0_Q8_1_MMVQ = 1;
struct block_q1_0 {
    __half  d;
    uint8_t qs[QK1_0 / 8];
};
static_assert(sizeof(block_q1_0) == sizeof(__half) + QK1_0 / 8, "wrong q1_0 block size/padding");

inline constexpr int QK2_0 = 64;
inline constexpr int QR2_0 = 1;
inline constexpr int QI2_0 = QK2_0 / 32;
inline constexpr int VDR_Q2_0_Q8_1_MMVQ = 1;
struct block_q2_0 {
    __half  d;
    uint8_t qs[QK2_0 / 4];
};
static_assert(sizeof(block_q2_0) == sizeof(__half) + QK2_0 / 4, "wrong q2_0 block size/padding");

// ---------------------------------------------------------------------------------------------
// Scalar decoders shared by the vec-dot, the dequantiser and the eight-value MoE decode, host
// and device alike, so the three consumers of a block cannot disagree about a value.

/// E8M0 exponent byte to float, halved: kFp4Values holds the E2M1 magnitudes doubled, so the
/// scale carries the half. x < 2 are the two smallest denormals; NaN (0xFF) is not handled,
/// as in ggml.
__host__ __device__ __forceinline__ float e8m0_to_fp32_half(std::uint8_t x) {
    const std::uint32_t bits = x < 2 ? (0x00200000u << x) : (static_cast<std::uint32_t>(x - 1) << 23);
    return __uint_as_float_portable(bits);
}

/// UE4M3 (unsigned, bias 7) to float, halved for the same reason; 0 and the NaN code decode to
/// zero, as ggml's CPU path does.
__host__ __device__ __forceinline__ float ue4m3_to_fp32_half(std::uint8_t x) {
    if (x == 0 || x == 0x7F) { return 0.0F; }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    const float raw = exp == 0 ? static_cast<float>(man) * (1.0F / 512.0F)
                               : (1.0F + static_cast<float>(man) * 0.125F) * exp2f_portable(exp - 7);
    return raw * 0.5F;
}

/// TQ1_0: value `v` of the block as a trit in {-1, 0, 1}. Five trits share a byte in base
/// three (3^5 = 243): the first 240 values are five passes over the 48 `qs` bytes -- 32 bytes,
/// then 16 -- and the last 16 are four passes over the four `qh` bytes.
__host__ __device__ __forceinline__ int tq1_0_trit(const block_tq1_0& x, int v) {
    constexpr std::uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    std::uint8_t q;
    if (v < 160) {
        q = static_cast<std::uint8_t>(x.qs[v & 31] * pow3[v >> 5]);
    } else if (v < 240) {
        const int r = v - 160;
        q = static_cast<std::uint8_t>(x.qs[32 + (r & 15)] * pow3[r >> 4]);
    } else {
        const int r = v - 240;
        q = static_cast<std::uint8_t>(x.qh[r & 3] * pow3[r >> 2]);
    }
    return static_cast<int>((static_cast<std::uint16_t>(q) * 3) >> 8) - 1;
}

/// TQ2_0: value `v` as a code in {-1, 0, 1, 2}. Two bits per value; the 64 `qs` bytes are two
/// runs of 32, each byte holding four values 32 apart.
__host__ __device__ __forceinline__ int tq2_0_code(const block_tq2_0& x, int v) {
    const int byte  = (v >> 7) * 32 + (v & 31);
    const int shift = ((v >> 5) & 3) * 2;
    return static_cast<int>((x.qs[byte] >> shift) & 3) - 1;
}

/// IQ1_M carries its block scale as the top nibble of each of its four 16-bit scale words.
__host__ __device__ __forceinline__ __half iq1m_block_scale(const block_iq1_m& x) {
    const std::uint16_t* sc = reinterpret_cast<const std::uint16_t*>(x.scales);
    iq1m_scale_t scale;
    scale.u16 = static_cast<std::uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                                           ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    return scale.f16;
}

} // namespace sinfer::ops::detail::ggml
