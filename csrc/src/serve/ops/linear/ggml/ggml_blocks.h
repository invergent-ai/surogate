// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace sinfer::ops::detail::ggml {

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

} // namespace sinfer::ops::detail::ggml
