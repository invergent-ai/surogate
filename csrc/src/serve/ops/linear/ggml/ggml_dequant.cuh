// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#pragma once

#include "ops/linear/ggml/ggml_blocks.h"
#include "ops/linear/ggml/ggml_mmvq.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace sinfer::ops::detail::ggml {

// llama.cpp's own thread layout for the K-quant dequantisers: 32 threads for Q4_K, 64 for the
// other four (convert.cu launches them so, and each thread writes its fixed share of a block).
// Getting this wrong leaves part of every block unwritten.
template <GgmlType type>
__host__ __device__ constexpr int dequant_threads() {
    // llama.cpp launches Q4_K's dequantiser with 32 threads and the other K-quants with 64; a
    // Q8_0 block is 32 values, one per lane.
    if constexpr (type == GgmlType::Q8_0) { return QK8_0; }
    if constexpr (type == GgmlType::Q4_1) { return QK4_1; }
    if constexpr (type == GgmlType::Q5_1) { return QK5_1; }
    if constexpr (type == GgmlType::IQ4_NL) { return QK4_NL; }
    if constexpr (type == GgmlType::Q4_0) { return QK4_0; }
    if constexpr (type == GgmlType::Q5_0) { return QK5_0; }
    return type == GgmlType::Q4_K ? 32 : 64;
}

template <typename dst_t> __device__ __forceinline__ dst_t cast_to(float v);
template <> __device__ __forceinline__ float cast_to<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cast_to<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q2_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[ib].qs[32*n + l];
    dst_t * y = yy + 128*n;

    float dall = __low2half(x[ib].dm);
    float dmin = __high2half(x[ib].dm);
    y[l+ 0] = cast_to<dst_t>(dall * (x[ib].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[ib].scales[is+0] >> 4));
    y[l+32] = cast_to<dst_t>(dall * (x[ib].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[ib].scales[is+2] >> 4));
    y[l+64] = cast_to<dst_t>(dall * (x[ib].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[ib].scales[is+4] >> 4));
    y[l+96] = cast_to<dst_t>(dall * (x[ib].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[ib].scales[is+6] >> 4));
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = tid/4;
    const int64_t t = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(tid%4);
    const int64_t n = t / 4;
    const int64_t j = t - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is+0] >> 4) & 3) << 4) :
                          (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[ib].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + 128*n + 32*j;
    const uint8_t * q = x[ib].qs + 32*n;
    const uint8_t * hm = x[ib].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = cast_to<dst_t>(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}

static inline __host__ __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;

    // assume 32 threads
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + 64*il + n*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * q = x[ib].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cast_to<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l +32] = cast_to<dst_t>(d2 * (q[l] >>  4) - m2);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + 64*il + 2*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = cast_to<dst_t>(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = cast_to<dst_t>(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = cast_to<dst_t>(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = cast_to<dst_t>(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}

/// One value per lane: a Q8_0 block is 32 quants and a single scale.
template <typename dst_t>
__device__ __forceinline__ void dequantize_q8_0(const void* blocks, std::int64_t ib, dst_t* out,
                                                int tid) {
    const block_q8_0* x = static_cast<const block_q8_0*>(blocks) + ib;
    if (tid < QK8_0) {
        out[tid] = cast_to<dst_t>(__half2float(x->d) * static_cast<float>(x->qs[tid]));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + 128*ip + il;

    const float d = x[ib].d;

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t   qh = x[ib].qh[32*ip + il];
    const int8_t  * sc = x[ib].scales + is;

    y[ 0] = cast_to<dst_t>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = cast_to<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = cast_to<dst_t>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = cast_to<dst_t>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

/// One value per lane: a Q4_1 block is 32 nibbles with a scale and an additive minimum.
template <typename dst_t>
__device__ __forceinline__ void dequantize_q4_1(const void* blocks, std::int64_t ib, dst_t* out,
                                                int tid) {
    const block_q4_1* x = static_cast<const block_q4_1*>(blocks) + ib;
    if (tid < QK4_1) {
        const float2 dm = __half22float2(x->dm);
        // The low nibbles hold the first sixteen values and the high nibbles the rest, which is
        // the order the vec-dot reads them in.
        const int q = (tid < QK4_1 / 2) ? (x->qs[tid] & 0x0F) : (x->qs[tid - QK4_1 / 2] >> 4);
        out[tid]    = cast_to<dst_t>(dm.x * static_cast<float>(q) + dm.y);
    }
}

/// One value per lane: a Q5_1 block, the fifth bit of each quant living in a separate word.
template <typename dst_t>
__device__ __forceinline__ void dequantize_q5_1(const void* blocks, std::int64_t ib, dst_t* out,
                                                int tid) {
    const block_q5_1* x = static_cast<const block_q5_1*>(blocks) + ib;
    if (tid < QK5_1) {
        const float2 dm = __half22float2(x->dm);
        std::uint32_t qh;
        memcpy(&qh, x->qh, sizeof(qh));
        const int low  = (tid < QK5_1 / 2) ? (x->qs[tid] & 0x0F) : (x->qs[tid - QK5_1 / 2] >> 4);
        const int high = static_cast<int>((qh >> tid) & 1u) << 4;
        out[tid]       = cast_to<dst_t>(dm.x * static_cast<float>(low | high) + dm.y);
    }
}

/// One value per lane: Q4_0 and Q5_0, whose codes are biased by half their range.
template <typename dst_t>
__device__ __forceinline__ void dequantize_q4_0(const void* blocks, std::int64_t ib, dst_t* out,
                                                int tid) {
    const block_q4_0* x = static_cast<const block_q4_0*>(blocks) + ib;
    if (tid < QK4_0) {
        const int q = (tid < QK4_0 / 2) ? (x->qs[tid] & 0x0F) : (x->qs[tid - QK4_0 / 2] >> 4);
        out[tid]    = cast_to<dst_t>(__half2float(x->d) * static_cast<float>(q - 8));
    }
}

template <typename dst_t>
__device__ __forceinline__ void dequantize_q5_0(const void* blocks, std::int64_t ib, dst_t* out,
                                                int tid) {
    const block_q5_0* x = static_cast<const block_q5_0*>(blocks) + ib;
    if (tid < QK5_0) {
        std::uint32_t qh;
        memcpy(&qh, x->qh, sizeof(qh));
        const int low  = (tid < QK5_0 / 2) ? (x->qs[tid] & 0x0F) : (x->qs[tid - QK5_0 / 2] >> 4);
        const int high = static_cast<int>((qh >> tid) & 1u) << 4;
        out[tid]       = cast_to<dst_t>(__half2float(x->d) * static_cast<float>((low | high) - 16));
    }
}

/// One value per lane: an IQ4_NL block, each nibble a table index rather than a magnitude.
template <typename dst_t>
__device__ __forceinline__ void dequantize_iq4_nl(const void* blocks, std::int64_t ib, dst_t* out,
                                                  int tid) {
    const block_iq4_nl* x = static_cast<const block_iq4_nl*>(blocks) + ib;
    if (tid < QK4_NL) {
        const int code = (tid < QK4_NL / 2) ? (x->qs[tid] & 0x0F) : (x->qs[tid - QK4_NL / 2] >> 4);
        out[tid] = cast_to<dst_t>(__half2float(x->d) * static_cast<float>(kIq4nlValues[code]));
    }
}

/// One 256-value superblock `ib` of a K-quant array into `out`, by the calling CTA's threads.
template <GgmlType type, typename dst_t>
__device__ __forceinline__ void dequantize_superblock(const void* blocks, std::int64_t ib,
                                                      dst_t* out, int tid) {
    if constexpr (type == GgmlType::Q2_K) { dequantize_q2_K(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::Q3_K) { dequantize_q3_K(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::Q4_K) { dequantize_q4_K(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::Q5_K) { dequantize_q5_K(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::Q6_K) { dequantize_q6_K(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::Q8_0) { dequantize_q8_0(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::Q4_1) { dequantize_q4_1(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::Q5_1) { dequantize_q5_1(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::IQ4_NL) { dequantize_iq4_nl(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::Q4_0) { dequantize_q4_0(blocks, ib, out, tid); }
    if constexpr (type == GgmlType::Q5_0) { dequantize_q5_0(blocks, ib, out, tid); }
}

} // namespace sinfer::ops::detail::ggml
