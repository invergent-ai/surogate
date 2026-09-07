// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#pragma once

// Weight decode for the shapes that consume a K-quant as *floats* rather than through the
// integer vec-dot: the sparse-MoE bodies, which multiply a BF16 activation directly. Each lane
// owns eight consecutive values of a 256-value superblock, which every K-quant layout supports
// exactly because eight divides both sub-block sizes (32 for Q4_K/Q5_K, 16 for Q6_K) -- a run of
// eight never straddles a scale boundary or a nibble half.
//
// Host as well as device: the CPU expert path decodes the same blocks with the same
// arithmetic, so a routed expert answers alike whether the slot cache had it or the host
// computed it. One decoder, not two that drift.

#include "ops/linear/ggml/ggml_blocks.h"
#include "ops/linear/ggml/ggml_mmvq.h"
#include "ops/linear/ggml/ggml_dequant.cuh"

#include <cstdint>

namespace sinfer::ops::detail::ggml {

/// Eight consecutive values starting at `lane * 8` of superblock `ib`.
template <GgmlType type>
__host__ __device__ __forceinline__ void decode_eight(const void* blocks, std::int64_t ib, int lane,
                                             float (&w)[8]);

/// Q2_K: sixteen sub-blocks of sixteen, `x = d*sc*q - dmin*m` with sc and m four bits each.
/// Eight consecutive values share one sub-block, because eight divides sixteen.
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q2_K>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q2_K* x = static_cast<const block_q2_K*>(blocks) + ib;
    const int v0 = lane * 8;
    const int n  = v0 >> 7;          // which 128-value half
    const int r  = v0 & 127;
    const int j  = r >> 5;           // which of the four 32-value stripes
    const int l0 = r & 31;
    const int is = 8 * n + (l0 >> 4) + 2 * j;
    const float d  = __low2float(x->dm) * static_cast<float>(x->scales[is] & 0xF);
    const float mo = __high2float(x->dm) * static_cast<float>(x->scales[is] >> 4);
    const std::uint8_t* q = x->qs + 32 * n + l0;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        w[l] = d * static_cast<float>((q[l] >> (2 * j)) & 3) - mo;
    }
}

/// Q3_K: the same stripe geometry with a 6-bit scale split across two bytes and the third bit
/// of each quant living inverted in `hmask`.
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q3_K>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q3_K* x = static_cast<const block_q3_K*>(blocks) + ib;
    const int v0  = lane * 8;
    const int n   = v0 >> 7;
    const int r   = v0 & 127;
    const int j   = r >> 5;
    const int l0  = r & 31;
    const int is0 = l0 >> 4;
    const int is  = 8 * n + 2 * j + is0;
    const std::int8_t us =
        is < 4    ? static_cast<std::int8_t>((x->scales[is] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4))
        : is < 8  ? static_cast<std::int8_t>((x->scales[is] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4))
        : is < 12 ? static_cast<std::int8_t>((x->scales[is - 8] >> 4) | (((x->scales[is] >> 4) & 3) << 4))
                  : static_cast<std::int8_t>((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
    const float dl        = __half2float(x->d) * static_cast<float>(us - 32);
    const std::uint8_t m  = static_cast<std::uint8_t>(1 << (4 * n + j));
    const std::uint8_t* q = x->qs + 32 * n;
    const std::uint8_t* hm = x->hmask;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int code = static_cast<int>((q[l0 + l] >> (2 * j)) & 3) - ((hm[l0 + l] & m) ? 0 : 4);
        w[l]           = dl * static_cast<float>(code);
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q4_K>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q4_K* x = static_cast<const block_q4_K*>(blocks) + ib;
    const int v0 = lane * 8;
    const int il = v0 >> 6;          // which 64-value quarter
    const int o  = v0 & 63;
    const int hi = o >> 5;           // low nibbles for the first 32, high for the rest
    const int r  = o & 31;
    std::uint8_t sc = 0;
    std::uint8_t m  = 0;
    get_scale_min_k4(2 * il + hi, x->scales, sc, m);
    const float d = __low2float(x->dm) * sc;
    const float mo = __high2float(x->dm) * m;
    const std::uint8_t* q = x->qs + 32 * il + r;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int code = hi ? (q[l] >> 4) : (q[l] & 0xF);
        w[l]           = d * code - mo;
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q5_K>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q5_K* x = static_cast<const block_q5_K*>(blocks) + ib;
    const int v0 = lane * 8;
    const int il = v0 >> 6;
    const int o  = v0 & 63;
    const int hi = o >> 5;
    const int r  = o & 31;
    std::uint8_t sc = 0;
    std::uint8_t m  = 0;
    get_scale_min_k4(2 * il + hi, x->scales, sc, m);
    const float d  = __low2float(x->dm) * sc;
    const float mo = __high2float(x->dm) * m;
    const std::uint8_t* q  = x->qs + 32 * il + r;
    const std::uint8_t* qh = x->qh + r;
    const std::uint8_t bit = static_cast<std::uint8_t>(1u << (2 * il + hi));
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int code = (hi ? (q[l] >> 4) : (q[l] & 0xF)) + ((qh[l] & bit) ? 16 : 0);
        w[l]           = d * code - mo;
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q6_K>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q6_K* x = static_cast<const block_q6_K*>(blocks) + ib;
    const int v0  = lane * 8;
    const int ip  = v0 >> 7;         // which 128-value half
    const int rem = v0 & 127;
    const int t   = rem >> 5;        // which of the four interleaved stripes
    const int il  = rem & 31;
    const float d = __half2float(x->d) * x->scales[8 * ip + il / 16 + 2 * t];
    const std::uint8_t* ql = x->ql + 64 * ip + il + 32 * (t & 1);
    const std::uint8_t* qh = x->qh + 32 * ip + il;
    const int shift = 2 * t;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int low  = (t < 2) ? (ql[l] & 0xF) : (ql[l] >> 4);
        const int code = low | (((qh[l] >> shift) & 3) << 4);
        w[l]           = d * (code - 32);
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::F16>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    // Like Q8_0: 32 values, so four lanes cover one block. There is no scale to apply.
    const block_f16* x = static_cast<const block_f16*>(blocks) + ib;
#pragma unroll
    for (int l = 0; l < 8; ++l) { w[l] = __half2float(x->qs[lane * 8 + l]); }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q8_0>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    // A Q8_0 block is 32 values, so four lanes cover one rather than thirty-two: `ib` is the
    // block and `lane` its eighth-of-a-block, exactly as the callers already index.
    const block_q8_0* x = static_cast<const block_q8_0*>(blocks) + ib;
    const float d       = __half2float(x->d);
#pragma unroll
    for (int l = 0; l < 8; ++l) { w[l] = d * static_cast<float>(x->qs[lane * 8 + l]); }
}

/// The plain 32-value blocks. Four lanes cover one, and eight consecutive values never
/// straddle the block's single scale, so each lane reads one nibble half of eight bytes.
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q4_0>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q4_0* x = static_cast<const block_q4_0*>(blocks) + ib;
    const int base      = lane * 8;
    const bool high     = base >= QK4_0 / 2;
    const int off       = high ? base - QK4_0 / 2 : base;
    const float d       = __half2float(x->d);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int q = high ? (x->qs[off + l] >> 4) : (x->qs[off + l] & 0x0F);
        w[l]        = d * static_cast<float>(q - 8);
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q4_1>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q4_1* x = static_cast<const block_q4_1*>(blocks) + ib;
    const int base      = lane * 8;
    const bool high     = base >= QK4_1 / 2;
    const int off       = high ? base - QK4_1 / 2 : base;
    const float2 dm     = __half22float2(x->dm);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int q = high ? (x->qs[off + l] >> 4) : (x->qs[off + l] & 0x0F);
        w[l]        = dm.x * static_cast<float>(q) + dm.y;
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q5_0>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q5_0* x = static_cast<const block_q5_0*>(blocks) + ib;
    const int base      = lane * 8;
    const bool high     = base >= QK5_0 / 2;
    const int off       = high ? base - QK5_0 / 2 : base;
    const float d       = __half2float(x->d);
    std::uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int low  = high ? (x->qs[off + l] >> 4) : (x->qs[off + l] & 0x0F);
        const int bit  = static_cast<int>((qh >> (base + l)) & 1u) << 4;
        w[l]           = d * static_cast<float>((low | bit) - 16);
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q5_1>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q5_1* x = static_cast<const block_q5_1*>(blocks) + ib;
    const int base      = lane * 8;
    const bool high     = base >= QK5_1 / 2;
    const int off       = high ? base - QK5_1 / 2 : base;
    const float2 dm     = __half22float2(x->dm);
    std::uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int low = high ? (x->qs[off + l] >> 4) : (x->qs[off + l] & 0x0F);
        const int bit = static_cast<int>((qh >> (base + l)) & 1u) << 4;
        w[l]          = dm.x * static_cast<float>(low | bit) + dm.y;
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::IQ4_NL>(const void* blocks, std::int64_t ib,
                                                               int lane, float (&w)[8]) {
    const block_iq4_nl* x = static_cast<const block_iq4_nl*>(blocks) + ib;
    const int base        = lane * 8;
    const bool high       = base >= QK4_NL / 2;
    const int off         = high ? base - QK4_NL / 2 : base;
    const float d         = __half2float(x->d);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int code = high ? (x->qs[off + l] >> 4) : (x->qs[off + l] & 0x0F);
        w[l]           = d * static_cast<float>(kIq4nlValues[code]);
    }
}

/// The sparse-MoE codec seam: a 256-value group is one superblock, so a warp's 32 lanes cover it
/// with eight values each. `high` and `scales` are unused -- a superblock carries its own.
// ---------------------------------------------------------------------------------------------
// The IQ family: a lane's eight values are exactly one grid row (IQ2, IQ1) or two four-value
// rows (IQ3), so this is llama.cpp's dequantiser with (il, ib) read off the lane: values
// 32*ib + 8*il, i.e. ib = lane / 4, il = lane % 4.
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::IQ2_XXS>(const void* blocks, std::int64_t ib,
                                                                int lane, float (&w)[8]) {
    const block_iq2_xxs& x = static_cast<const block_iq2_xxs*>(blocks)[ib];
    const int il = lane % 4, b = lane / 4;
    const std::uint16_t* q2   = x.qs + 4 * b;
    const std::uint8_t* aux8  = reinterpret_cast<const std::uint8_t*>(q2);
    const std::uint8_t* grid  = reinterpret_cast<const std::uint8_t*>(kIq2xxsGrid + aux8[il]);
    const std::uint32_t aux32 = q2[2] | (static_cast<std::uint32_t>(q2[3]) << 16);
    const float d             = __half2float(x.d) * (0.5F + (aux32 >> 28)) * 0.25F;
    const std::uint8_t signs  = kIq2xsSigns[(aux32 >> 7 * il) & 127];
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * grid[j] * ((signs & kIq2xsMask[j]) ? -1.0F : 1.0F); }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::IQ2_XS>(const void* blocks, std::int64_t ib,
                                                               int lane, float (&w)[8]) {
    const block_iq2_xs& x = static_cast<const block_iq2_xs*>(blocks)[ib];
    const int il = lane % 4, b = lane / 4;
    const std::uint16_t* q2  = x.qs + 4 * b;
    const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(kIq2xsGrid + (q2[il] & 511));
    const float d            = __half2float(x.d) * (0.5F + ((x.scales[b] >> 4 * (il / 2)) & 0xF)) * 0.25F;
    const std::uint8_t signs = kIq2xsSigns[q2[il] >> 9];
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * grid[j] * ((signs & kIq2xsMask[j]) ? -1.0F : 1.0F); }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::IQ2_S>(const void* blocks, std::int64_t ib,
                                                              int lane, float (&w)[8]) {
    const block_iq2_s& x = static_cast<const block_iq2_s*>(blocks)[ib];
    const int il = lane % 4, b = lane / 4;
    const std::uint8_t* grid = reinterpret_cast<const std::uint8_t*>(
        kIq2sGrid + (x.qs[4 * b + il] | ((x.qh[b] << (8 - 2 * il)) & 0x300)));
    const float d            = __half2float(x.d) * (0.5F + ((x.scales[b] >> 4 * (il / 2)) & 0xF)) * 0.25F;
    const std::uint8_t signs = x.qs[QK_K / 8 + 4 * b + il];
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * grid[j] * ((signs & kIq2xsMask[j]) ? -1.0F : 1.0F); }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::IQ3_XXS>(const void* blocks, std::int64_t ib,
                                                                int lane, float (&w)[8]) {
    const block_iq3_xxs& x = static_cast<const block_iq3_xxs*>(blocks)[ib];
    const int il = lane % 4, b = lane / 4;
    const std::uint8_t* q3    = x.qs + 8 * b;
    const std::uint16_t* gas  = reinterpret_cast<const std::uint16_t*>(x.qs + QK_K / 4) + 2 * b;
    const std::uint8_t* grid1 = reinterpret_cast<const std::uint8_t*>(kIq3xxsGrid + q3[2 * il + 0]);
    const std::uint8_t* grid2 = reinterpret_cast<const std::uint8_t*>(kIq3xxsGrid + q3[2 * il + 1]);
    const std::uint32_t aux32 = gas[0] | (static_cast<std::uint32_t>(gas[1]) << 16);
    const float d             = __half2float(x.d) * (0.5F + (aux32 >> 28)) * 0.5F;
    const std::uint8_t signs  = kIq2xsSigns[(aux32 >> 7 * il) & 127];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j + 0] = d * grid1[j] * ((signs & kIq2xsMask[j + 0]) ? -1.0F : 1.0F);
        w[j + 4] = d * grid2[j] * ((signs & kIq2xsMask[j + 4]) ? -1.0F : 1.0F);
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::IQ3_S>(const void* blocks, std::int64_t ib,
                                                              int lane, float (&w)[8]) {
    const block_iq3_s& x = static_cast<const block_iq3_s*>(blocks)[ib];
    const int il = lane % 4, b = lane / 4;
    const std::uint8_t* qs    = x.qs + 8 * b;
    const std::uint8_t* grid1 = reinterpret_cast<const std::uint8_t*>(kIq3sGrid + (qs[2 * il + 0] | ((x.qh[b] << (8 - 2 * il)) & 256)));
    const std::uint8_t* grid2 = reinterpret_cast<const std::uint8_t*>(kIq3sGrid + (qs[2 * il + 1] | ((x.qh[b] << (7 - 2 * il)) & 256)));
    const float d             = __half2float(x.d) * (1 + 2 * ((x.scales[b / 2] >> 4 * (b % 2)) & 0xF));
    const std::uint8_t signs  = x.signs[4 * b + il];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j + 0] = d * grid1[j] * ((signs & kIq2xsMask[j + 0]) ? -1.0F : 1.0F);
        w[j + 4] = d * grid2[j] * ((signs & kIq2xsMask[j + 4]) ? -1.0F : 1.0F);
    }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::IQ1_S>(const void* blocks, std::int64_t ib,
                                                              int lane, float (&w)[8]) {
    const block_iq1_s& x = static_cast<const block_iq1_s*>(blocks)[ib];
    const int il = lane % 4, b = lane / 4;
    const float delta = (x.qh[b] & 0x8000) ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d     = __half2float(x.d) * (2 * ((x.qh[b] >> 12) & 7) + 1);
    std::uint32_t grid32[2];
    const std::int8_t* q = reinterpret_cast<const std::int8_t*>(grid32);
    grid32[0] = kIq1sGridGpu[x.qs[4 * b + il] | (((x.qh[b] >> 3 * il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0F0F0F0F;
    grid32[0] &= 0x0F0F0F0F;
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * (q[j] + delta); }
}

template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::IQ1_M>(const void* blocks, std::int64_t ib,
                                                              int lane, float (&w)[8]) {
    const block_iq1_m& x = static_cast<const block_iq1_m*>(blocks)[ib];
    const int il = lane % 4, b = lane / 4;
    const std::uint16_t* sc = reinterpret_cast<const std::uint16_t*>(x.scales);
    const int ib16          = 2 * b + il / 2;
    const float d           = __half2float(iq1m_block_scale(x)) * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta       = (x.qh[2 * b + il / 2] & (0x08 << 4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    std::uint32_t grid32[2];
    const std::int8_t* q = reinterpret_cast<const std::int8_t*>(grid32);
    grid32[0] = kIq1sGridGpu[x.qs[4 * b + il] | (((x.qh[2 * b + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0F0F0F0F;
    grid32[0] &= 0x0F0F0F0F;
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * (q[j] + delta); }
}

/// IQ4_XS: eight consecutive values are eight low nibbles or eight high nibbles of one 16-byte
/// run, under that run's 6-bit scale.
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::IQ4_XS>(const void* blocks, std::int64_t ib,
                                                               int lane, float (&w)[8]) {
    const block_iq4_xs& x = static_cast<const block_iq4_xs*>(blocks)[ib];
    const int v0   = lane * 8;
    const int b    = v0 / 32;          // the 32-value run
    const int r    = v0 % 32;          // 0, 8 (low nibbles) or 16, 24 (high nibbles)
    const bool high = r >= 16;
    const std::uint8_t* q4 = x.qs + 16 * b + (high ? r - 16 : r);
    const float d = __half2float(x.d) *
                    ((((x.scales_l[b / 2] >> 4 * (b % 2)) & 0xF) | (((x.scales_h >> 2 * b) & 3) << 4)) - 32);
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * kIq4nlValues[high ? (q4[j] >> 4) : (q4[j] & 0xF)]; }
}

/// The ternary pair, through the scalar decoders.
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::TQ1_0>(const void* blocks, std::int64_t ib,
                                                              int lane, float (&w)[8]) {
    const block_tq1_0& x = static_cast<const block_tq1_0*>(blocks)[ib];
    const float d        = __half2float(x.d);
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * static_cast<float>(tq1_0_trit(x, lane * 8 + j)); }
}
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::TQ2_0>(const void* blocks, std::int64_t ib,
                                                              int lane, float (&w)[8]) {
    const block_tq2_0& x = static_cast<const block_tq2_0*>(blocks)[ib];
    const float d        = __half2float(x.d);
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * static_cast<float>(tq2_0_code(x, lane * 8 + j)); }
}

/// MXFP4: a 32-value block, four lanes; the low nibbles are the first sixteen values.
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::MXFP4>(const void* blocks, std::int64_t ib,
                                                              int lane, float (&w)[8]) {
    const block_mxfp4& x = static_cast<const block_mxfp4*>(blocks)[ib];
    const int base  = lane * 8;
    const bool high = base >= QK_MXFP4 / 2;
    const int off   = high ? base - QK_MXFP4 / 2 : base;
    const float d   = e8m0_to_fp32_half(x.e);
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * kFp4Values[high ? (x.qs[off + j] >> 4) : (x.qs[off + j] & 0x0F)]; }
}

/// NVFP4: a 64-value block, eight lanes, two per 16-value sub-block (low nibbles, then high).
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::NVFP4_GGML>(const void* blocks, std::int64_t ib,
                                                              int lane, float (&w)[8]) {
    const block_nvfp4& x = static_cast<const block_nvfp4*>(blocks)[ib];
    const int sub   = lane / 2;
    const bool high = (lane % 2) != 0;
    const std::uint8_t* qs = x.qs + sub * (QK_NVFP4_SUB / 2);
    const float d          = ue4m3_to_fp32_half(x.d[sub]);
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = d * kFp4Values[high ? (qs[j] >> 4) : (qs[j] & 0x0F)]; }
}

/// Q1_0: 128 values, sixteen lanes, one byte of sign bits each. Q2_0: 64 values, eight lanes,
/// two bytes each.
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q1_0>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q1_0& x = static_cast<const block_q1_0*>(blocks)[ib];
    const float d       = __half2float(x.d);
    const std::uint8_t bits = x.qs[lane];
#pragma unroll
    for (int j = 0; j < 8; ++j) { w[j] = ((bits >> j) & 1) ? d : -d; }
}
template <>
__host__ __device__ __forceinline__ void decode_eight<GgmlType::Q2_0>(const void* blocks, std::int64_t ib,
                                                             int lane, float (&w)[8]) {
    const block_q2_0& x = static_cast<const block_q2_0*>(blocks)[ib];
    const float d       = __half2float(x.d);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int v = lane * 8 + j;
        w[j] = d * static_cast<float>(((x.qs[v / 4] >> ((v % 4) * 2)) & 0x03) - 1);
    }
}

template <GgmlType type>
struct GgmlMoeCodec {
    static constexpr bool kGateRowsFirst = true;
    /// The stored block: a K-quant superblock is 256 values and a warp covers it with eight
    /// each; Q8_0 is 32, so four lanes do. The body's packed loop derives its lane split from
    /// this, so both fall out of the same code.
    static constexpr int kGroupK = block_values(type);
    /// `ggml-blocks-v1` stores K exactly -- a superblock format carries its scales inside the
    /// block and the layout requires K to be a whole number of blocks -- so a stored row is as
    /// wide as the math reads. The row-split codecs pad to 128 and say so.
    static constexpr int kStoredKAlignment = 1;
    // Both projections take the body's generic packed-word8 loop, the one written against
    // load_eight; the two specialised D3 shapes assume a plane layout this format does not have.
    static constexpr bool kPackedWord8         = true;
    static constexpr bool kD3PackedWord8       = false;
    static constexpr bool kD3SingleValuePerLane = false;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t*, const std::uint8_t*,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        decode_eight<type>(codes, group_index, lane_in_group, weights);
    }
};

/// Thirty-two consecutive values starting at `group * 32`, for a caller that owns a whole
/// group rather than eight of them. A 32-value block is one group; a superblock is eight.
template <GgmlType type>
__host__ __device__ __forceinline__ void decode_group_32(const void* blocks, std::int64_t group,
                                                float (&v)[32]) {
    constexpr int kValues = block_values(type);
    const std::int64_t base = group * 32;
    const std::int64_t ib   = base / kValues;
    const int lane0         = static_cast<int>((base % kValues) / 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float w[8];
        decode_eight<type>(blocks, ib, lane0 + j, w);
#pragma unroll
        for (int l = 0; l < 8; ++l) { v[j * 8 + l] = w[l]; }
    }
}

} // namespace sinfer::ops::detail::ggml
