// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#pragma once

#include "ops/linear/ggml/ggml_blocks.h"

#include <cuda_fp16.h>

#include <cstdint>

namespace sinfer::ops::detail::ggml {

static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;

    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    return x32;
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

#define VDR_Q2_K_Q8_1_MMVQ 1

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const __half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * __dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

#define VDR_Q3_K_Q8_1_MMVQ 1

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
}

#define VDR_Q4_K_Q8_1_MMVQ 2

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const __half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q5_K_Q8_1_MMVQ 2

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const __half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = __dp4a(v0i, u[2*i+0], __dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+0], __dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

#define VDR_Q6_K_Q8_1_MMVQ 1

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

// Q4_0 / Q5_0 against the int8 activation block. The bias is the affine term again in disguise:
// `w = d*(q - c)` gives `d*dy*(sum(q*u) - c*sum(u))`, and the activation-quant sum is taken
// exactly with dp4a rather than read from the fp16 sum, for the reason given below.
__device__ __forceinline__ float vec_dot_q4_0_q8_1(const void* __restrict__ vbq,
                                                   const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_0* bq = static_cast<const block_q4_0*>(vbq) + kbx;
    int sumi             = 0;
    int sumu             = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        const int v   = get_int_b2(bq->qs, iqs + i);
        const int u0  = get_int_b4(bq8_1->qs, iqs + i);
        const int u1  = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
        sumi          = __dp4a((v >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi          = __dp4a((v >> 4) & 0x0F0F0F0F, u1, sumi);
        sumu          = __dp4a(0x01010101, u0, sumu);
        sumu          = __dp4a(0x01010101, u1, sumu);
    }
    return __half2float(bq->d) * __half2float(__low2half(bq8_1->ds)) *
           static_cast<float>(sumi - 8 * sumu);
}

__device__ __forceinline__ float vec_dot_q5_0_q8_1(const void* __restrict__ vbq,
                                                   const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_0* bq = static_cast<const block_q5_0*>(vbq) + kbx;
    int sumi             = 0;
    int sumu             = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        const int vl = get_int_b2(bq->qs, iqs + i);
        const int vh = get_int_b2(bq->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i);
        const int u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);

        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = __dp4a(vi0, u0, sumi);

        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = __dp4a(vi1, u1, sumi);

        sumu = __dp4a(0x01010101, u0, sumu);
        sumu = __dp4a(0x01010101, u1, sumu);
    }
    return __half2float(bq->d) * __half2float(__low2half(bq8_1->ds)) *
           static_cast<float>(sumi - 16 * sumu);
}

// A sixteen-entry int8 table looked up for eight nibbles at once with byte permutes, in
// registers: the table's four words are read uniformly and `__byte_perm` selects with the
// three low bits of each nibble, the fourth bit choosing between the two halves. The obvious
// form -- eight per-lane reads of the table -- serialises thirty-two ways on a divergent
// address and made the IQ4_XS GEMV twelve times slower than Q4_K's (232 us against 19 on the
// 27B's shapes) until it was measured. llama.cpp's `get_int_from_table_16`.
__device__ __forceinline__ int2 table16_levels(const int q4, const std::int8_t* table) {
    const std::uint32_t* table32 = reinterpret_cast<const std::uint32_t*>(table);
    std::uint32_t tmp[2];
    const std::uint32_t low_high = (0x32103210u | ((static_cast<std::uint32_t>(q4) & 0x88888888u) >> 1));
#pragma unroll
    for (std::uint32_t i = 0; i < 2; ++i) {
        const std::uint32_t shift = 16 * i;
        const std::uint32_t low   = __byte_perm(table32[0], table32[1], static_cast<std::uint32_t>(q4) >> shift);
        const std::uint32_t high  = __byte_perm(table32[2], table32[3], static_cast<std::uint32_t>(q4) >> shift);
        tmp[i] = __byte_perm(low, high, low_high >> shift);
    }
    return make_int2(static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x6420)),
                     static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x7531)));
}

// IQ4_NL against the int8 activation block. Four nibbles at a time become four int8 levels
// through the table, and from there it is the same dp4a dot every other type does; the table
// lookup is the only thing between this and Q4_0.
__device__ __forceinline__ int2 iq4_nl_levels(int q4) { return table16_levels(q4, kIq4nlValues); }

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq,
                                                     const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_nl* bq = static_cast<const block_iq4_nl*>(vbq) + kbx;
    int sumi               = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMVQ; ++l) {
        // `qs` sits at a two-byte offset behind the scale, so the reads are two-byte aligned.
        const int2 v = iq4_nl_levels(get_int_b2(bq->qs, iqs + l));
        sumi         = __dp4a(v.x, get_int_b4(bq8_1->qs, iqs + l), sumi);
        sumi         = __dp4a(v.y, get_int_b4(bq8_1->qs, iqs + l + QI4_NL), sumi);
    }
    return __half2float(bq->d) * __half2float(__low2half(bq8_1->ds)) * static_cast<float>(sumi);
}

// Q4_1 / Q5_1 against the int8 activation block. The affine term is what makes these different
// from every other type here: `w = d*q + m`, so with the activation `y = dy*u` the dot is
// `d*dy * sum(q*u)  +  m*dy * sum(u)`.
//
// The second sum is over the activation *quants*, taken exactly with dp4a against 0x01010101 --
// the same way the K-quants take theirs. llama.cpp instead reads the sum of the unquantised
// activations that `quantize_q8_1` parks in the high half of `ds`; that costs nothing extra but
// sums a different set of numbers than the first term does, and the mismatch shows up as ~4e-4
// relative error against an oracle that dequantises. Two more dp4a buy exactness.
template <int vdr>
__device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(const int* v, const int* u,
                                                        const __half2& dm4, const __half2& ds8) {
    int sumi = 0;
    int sumu = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi          = __dp4a(vi0, u[2 * i + 0], sumi);
        sumi          = __dp4a(vi1, u[2 * i + 1], sumi);
        sumu          = __dp4a(0x01010101, u[2 * i + 0], sumu);
        sumu          = __dp4a(0x01010101, u[2 * i + 1], sumu);
    }
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    return ds8f.x * (dm4f.x * static_cast<float>(sumi) + dm4f.y * static_cast<float>(sumu));
}

template <int vdr>
__device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(const int* vl, const int* vh, const int* u,
                                                        const __half2& dm5, const __half2& ds8) {
    int sumi = 0;
    int sumu = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;   // low nibbles
        vi0 |= (vh[i] << 4) & 0x00000010;      // qh bit 0 -> byte 0, bit 4
        vi0 |= (vh[i] << 11) & 0x00001000;     // qh bit 1 -> byte 1, bit 4
        vi0 |= (vh[i] << 18) & 0x00100000;     // qh bit 2 -> byte 2, bit 4
        vi0 |= (vh[i] << 25) & 0x10000000;     // qh bit 3 -> byte 3, bit 4
        sumi = __dp4a(vi0, u[2 * i + 0], sumi);

        int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;   // high nibbles
        vi1 |= (vh[i] >> 12) & 0x00000010;     // fifth bit of value 16
        vi1 |= (vh[i] >> 5) & 0x00001000;      // ... of value 17
        vi1 |= (vh[i] << 2) & 0x00100000;      // ... of value 18
        vi1 |= (vh[i] << 9) & 0x10000000;      // ... of value 19
        sumi = __dp4a(vi1, u[2 * i + 1], sumi);
        sumu = __dp4a(0x01010101, u[2 * i + 0], sumu);
        sumu = __dp4a(0x01010101, u[2 * i + 1], sumu);
    }
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    return ds8f.x * (dm5f.x * static_cast<float>(sumi) + dm5f.y * static_cast<float>(sumu));
}

__device__ __forceinline__ float vec_dot_q4_1_q8_1(const void* __restrict__ vbq,
                                                   const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_1* bq4_1 = static_cast<const block_q4_1*>(vbq) + kbx;
    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2 * VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]             = get_int_b4(bq4_1->qs, iqs + i);
        u[2 * i + 0]     = get_int_b4(bq8_1->qs, iqs + i);
        u[2 * i + 1]     = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
    }
    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

__device__ __forceinline__ float vec_dot_q5_1_q8_1(const void* __restrict__ vbq,
                                                   const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_1* bq5_1 = static_cast<const block_q5_1*>(vbq) + kbx;
    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int u[2 * VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]        = get_int_b4(bq5_1->qs, iqs + i);
        vh[i]        = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2 * i + 0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
    }
    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

/// Q8_0 against the quantised activation: a plain dp4a dot, the two scales multiplied out.
/// Its `qs` sits at a two-byte offset inside the block, so the reads are the two-byte-aligned
/// accessor rather than the four-byte one the K-quants use.
static __device__ __forceinline__ float vec_dot_q8_0_q8_1(const void* __restrict__ vbq,
                                                          const block_q8_1* __restrict__ bq8_1,
                                                          const int& kbx, const int& iqs) {
    const block_q8_0* bq8_0 = static_cast<const block_q8_0*>(vbq) + kbx;
    int sumi                = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        sumi = __dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    }
    return __half2float(bq8_0->d) * __half2float(__low2half(bq8_1->ds)) * static_cast<float>(sumi);
}

/// F16 against the quantised activation. There is no integer product to accumulate — the
/// weight is already a number — so this is a plain float FMA over the four values a lane owns.
/// Exact in the weight (half to float loses nothing); the only error is the activation's own
/// int8-per-32, which every other type served from a GGUF pays too.
__device__ __forceinline__ float vec_dot_f16_q8_1(const void* __restrict__ vbq,
                                                  const block_q8_1* __restrict__ bq8_1,
                                                  const int& kbx, const int& iqs) {
    const block_f16* bf16 = static_cast<const block_f16*>(vbq) + kbx;
    float sum             = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += __half2float(bf16->qs[4 * iqs + i]) *
               static_cast<float>(bq8_1->qs[4 * iqs + i]);
    }
    return __half2float(__low2half(bq8_1->ds)) * sum;
}

__device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_b4(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq + kbx;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_b2(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

// ---------------------------------------------------------------------------------------------
// The importance-matrix family. A grid row is eight (IQ2) or four (IQ3) 2- or 3-bit magnitudes
// packed one per byte; the seven sign bits of an eight-value group expand to an xor mask through
// `unpack_ksigns`, whose eighth bit is the parity of the other seven.

__device__ __forceinline__ std::uint32_t unpack_ksigns(const std::uint8_t v) {
    const std::uint32_t p = __popc(v) & 1u;
    const std::uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

/// Sixteen-entry table lookup for eight nibbles at once: the even nibbles' levels in `.x`, the
/// odd nibbles' in `.y`, as two int8x4 words for dp4a. `__byte_perm` selects with three bits;
/// the fourth bit picks between the table's two halves.

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq,
                                                      const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = static_cast<const block_iq2_xxs*>(vbq) + kbx;
    const int q2                = get_int_b2(bq2->qs, iqs);
    const std::uint8_t* aux8    = reinterpret_cast<const std::uint8_t*>(&q2);
    const std::uint32_t aux32   = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos      = reinterpret_cast<const uint2*>(kIq2xxsGrid)[aux8[k0 / 2]];
        const std::uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0     = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi             = __dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1     = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi             = __dp4a(grid1, u1, sumi);
    }
    // The block scale is (2*s + 1)/8 with s the four-bit field. llama.cpp evaluates it in
    // integer arithmetic (`sumi * ls / 8`), which truncates; the exact form costs one multiply
    // and lands on the dequantised reference.
    const int ls  = (aux32 >> 27) | 1;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * static_cast<float>(sumi * ls) * 0.125F;
}

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq,
                                                     const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq2_xs* bq2 = static_cast<const block_iq2_xs*>(vbq) + kbx;
    const int2 q2_packed      = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const std::uint16_t* q2   = reinterpret_cast<const std::uint16_t*>(&q2_packed);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos      = reinterpret_cast<const uint2*>(kIq2xsGrid)[q2[l0 / 2] & 0x1FF];
        const std::uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0     = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1     = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, u0, sumi0);
            sumi0 = __dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = __dp4a(grid_l, u0, sumi1);
            sumi1 = __dp4a(grid_h, u1, sumi1);
        }
    }
    // exact (2*ls + 1)/8 per half, rather than llama.cpp's truncating integer form
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * static_cast<float>(sumi0 * (2 * ls0 + 1) + sumi1 * (2 * ls1 + 1)) * 0.125F;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq,
                                                    const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = static_cast<const block_iq2_s*>(vbq) + kbx;
    const int qs_packed              = get_int_b2(bq2->qs, iqs / 2);
    const std::uint8_t* qs           = reinterpret_cast<const std::uint8_t*>(&qs_packed);
    const int qh                     = bq2->qh[iqs / 2];
    const int signs_packed_32        = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const std::uint8_t* signs_packed = reinterpret_cast<const std::uint8_t*>(&signs_packed_32);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = reinterpret_cast<const int*>(kIq2sGrid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed[l0 / 2] & 0x03) << 7) | ((signs_packed[l0 / 2] & 0x0C) << 21), 0);
        const int signs1 = __vcmpne4(((signs_packed[l0 / 2] & 0x30) << 3) | ((signs_packed[l0 / 2] & 0xC0) << 17), 0);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0     = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1     = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, u0, sumi0);
            sumi0 = __dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = __dp4a(grid_l, u0, sumi1);
            sumi1 = __dp4a(grid_h, u1, sumi1);
        }
    }
    // exact (2*ls + 1)/8 per half, rather than llama.cpp's truncating integer form
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * static_cast<float>(sumi0 * (2 * ls0 + 1) + sumi1 * (2 * ls1 + 1)) * 0.125F;
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq,
                                                      const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = static_cast<const block_iq3_xxs*>(vbq) + kbx;
    const int2 q3_packed      = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const std::uint8_t* q3    = reinterpret_cast<const std::uint8_t*>(&q3_packed);
    const std::uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos       = make_int2(kIq3xxsGrid[q3[l0 + 0]], kIq3xxsGrid[q3[l0 + 1]]);
        const std::uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0     = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1     = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi             = __dp4a(grid_l, u0, sumi);
        sumi             = __dp4a(grid_h, u1, sumi);
    }
    const int ls  = aux32 >> 28; // the scale is (2*ls + 1)/4, evaluated exactly
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * static_cast<float>(sumi * (2 * ls + 1)) * 0.25F;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq,
                                                    const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = static_cast<const block_iq3_s*>(vbq) + kbx;
    const int2 qs_packed             = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const std::uint8_t* qs           = reinterpret_cast<const std::uint8_t*>(&qs_packed);
    const int qh                     = bq3->qh[iqs / 2];
    const int signs_packed_32        = get_int_b2(bq3->signs, iqs / 2);
    const std::uint8_t* signs_packed = reinterpret_cast<const std::uint8_t*>(&signs_packed_32);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(kIq3sGrid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        kIq3sGrid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed[l0 / 2] & 0x03) << 7) | ((signs_packed[l0 / 2] & 0x0C) << 21), 0);
        const int signs1 = __vcmpne4(((signs_packed[l0 / 2] & 0x30) << 3) | ((signs_packed[l0 / 2] & 0xC0) << 17), 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0     = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1     = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi             = __dp4a(grid_l, u0, sumi);
        sumi             = __dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

// The 1-bit pair. A grid row is eight values in {-1, 0, 1} plus a per-32 (IQ1_S) or per-16
// (IQ1_M) offset delta; the dot needs the activation's sum too, which q8_1 carries in ds.y.
__device__ __forceinline__ float vec_dot_iq1_s_q8_1(const void* __restrict__ vbq,
                                                    const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_s* bq1 = static_cast<const block_iq1_s*>(vbq) + kbx;
    const int qs_packed    = get_int_b2(bq1->qs, iqs);
    const std::uint8_t* qs = reinterpret_cast<const std::uint8_t*>(&qs_packed);
    const int qh           = bq1->qh[iqs];
    int sumi = 0;
    int sumy = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid  = kIq1sGridGpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0    = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1    = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi            = __dp4a(grid0, u0, sumi);
        sumi            = __dp4a(grid1, u1, sumi);
        // the activation's sum from its int8 codes, as IQ1_M does: q8_1's stored sum is of the
        // unquantised values, and with codes in {0, 1, 2} the delta term is not small next to
        // the main one, so that rounding showed as 5e-3 against the dequantised reference
        sumy            = __dp4a(u0, 0x01010101, sumy);
        sumy            = __dp4a(u1, 0x01010101, sumy);
    }
    const float d1q   = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0F + IQ1S_DELTA - (qh & 0x8000) * (2.0F * IQ1S_DELTA / 0x8000);
    return d1q * __low2float(bq8_1[iqs].ds) * (static_cast<float>(sumi) + delta * static_cast<float>(sumy));
}

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq,
                                                    const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_m* bq1 = static_cast<const block_iq1_m*>(vbq) + kbx;
    const int qs_packed    = get_int_b4(bq1->qs, iqs);
    const std::uint8_t* qs = reinterpret_cast<const std::uint8_t*>(&qs_packed);
    int sumi[2]   = {0, 0};
    float sumf[2] = {0.0F, 0.0F};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl   = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid  = kIq1sGridGpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0    = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1    = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4]    = __dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4]    = __dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0F + IQ1M_DELTA - (qhl & 0x08) * (2.0F * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy     = __dp4a(u0, 0x01010101, sumy);
        sumy     = __dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const std::uint16_t* sc = reinterpret_cast<const std::uint16_t*>(bq1->scales);
    const float d = __half2float(iq1m_block_scale(*bq1)) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

/// IQ4_XS: IQ4_NL's table over a superblock, one 6-bit scale per 32 split across two fields.
__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq,
                                                     const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = static_cast<const block_iq4_xs*>(vbq) + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v     = iq4_nl_levels(aux_q4);
        const int u0     = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1     = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi             = __dp4a(v.x, u0, sumi);
        sumi             = __dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// ---------------------------------------------------------------------------------------------
// The microscaling floats: E2M1 nibbles through the doubled table, the shared scale halved.
__device__ __forceinline__ float vec_dot_mxfp4_q8_1(const void* __restrict__ vbq,
                                                    const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_mxfp4* bq4 = static_cast<const block_mxfp4*>(vbq) + kbx;
    const int* q8          = reinterpret_cast<const int*>(bq8_1->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        const int2 v     = table16_levels(aux_q4, kFp4Values);
        sumi             = __dp4a(v.x, q8[l + 0], sumi);
        sumi             = __dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = e8m0_to_fp32_half(bq4->e) * __low2float(bq8_1->ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_nvfp4_q8_1(const void* __restrict__ vbq,
                                                    const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_nvfp4* bq4 = static_cast<const block_nvfp4*>(vbq) + kbx;
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < VDR_NVFP4_Q8_1_MMVQ / 2; ++i) {
        const int iqs0 = iqs + 2 * i;
        const int iqs1 = iqs0 + 1;
        const int is   = iqs0 >> 1;
        const int2 v0  = table16_levels(get_int_b4(bq4->qs, iqs0), kFp4Values);
        const int2 v1  = table16_levels(get_int_b4(bq4->qs, iqs1), kFp4Values);
        const block_q8_1* bq8 = bq8_1 + (is >> 1);
        const int i8          = ((is & 1) << 2);
        int sumi = __dp4a(v0.x, get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = __dp4a(v0.y, get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = __dp4a(v1.x, get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = __dp4a(v1.y, get_int_b4(bq8->qs, i8 + 3), sumi);
        const float d = ue4m3_to_fp32_half(bq4->d[is]) * __low2float(bq8->ds);
        sum += d * static_cast<float>(sumi);
    }
    return sum;
}

// ---------------------------------------------------------------------------------------------
// The plain 1- and 2-bit blocks: `__byte_perm` spreads the packed bits to one int8 per byte.
__device__ __forceinline__ float vec_dot_q1_0_q8_1(const void* __restrict__ vbq,
                                                   const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q1_0* bq1_0 = static_cast<const block_q1_0*>(vbq) + kbx;
    // 128 values under one scale; iqs picks one of four 32-value chunks, each a q8_1 block
    const float d1            = __half2float(bq1_0->d);
    const std::int16_t* qs    = reinterpret_cast<const std::int16_t*>(bq1_0->qs) + iqs * 2;
    const block_q8_1* bq8     = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int q  = qs[j];
        const int u0 = get_int_b4(bq8->qs, j * 4 + 0);
        const int u1 = get_int_b4(bq8->qs, j * 4 + 1);
        const int u2 = get_int_b4(bq8->qs, j * 4 + 2);
        const int u3 = get_int_b4(bq8->qs, j * 4 + 3);
        // unpack crumbs into nibble indices, nibbles into bytes, then unshuffle
        const int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0);
        const int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2);
        const int s0 = __byte_perm(0x01FF, 0x01FF, n0 >> 0);
        const int s1 = __byte_perm(0x01FF, 0x01FF, n1 >> 0);
        const int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
        const int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
        const int v0 = __byte_perm(s0, s1, 0x5410);
        const int v1 = __byte_perm(s0, s1, 0x7632);
        const int v2 = __byte_perm(s2, s3, 0x5410);
        const int v3 = __byte_perm(s2, s3, 0x7632);
        sumi = __dp4a(v0, u0, sumi);
        sumi = __dp4a(v1, u1, sumi);
        sumi = __dp4a(v2, u2, sumi);
        sumi = __dp4a(v3, u3, sumi);
    }
    return d1 * __low2float(bq8->ds) * sumi;
}

__device__ __forceinline__ float vec_dot_q2_0_q8_1(const void* __restrict__ vbq,
                                                   const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_0* bq2_0 = static_cast<const block_q2_0*>(vbq) + kbx;
    // 64 values under one scale; iqs picks one of two 32-value chunks
    const float d2         = __half2float(bq2_0->d);
    const std::int16_t* qs = reinterpret_cast<const std::int16_t*>(bq2_0->qs) + iqs * 4;
    const block_q8_1* bq8  = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q  = qs[j];
        const int u  = get_int_b4(bq8->qs, j * 2 + 0);
        const int v  = get_int_b4(bq8->qs, j * 2 + 1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = __dp4a(u, qx, sumi);
        sumi = __dp4a(v, qy, sumi);
    }
    return d2 * __low2float(bq8->ds) * sumi;
}

// ---------------------------------------------------------------------------------------------
// The ternary pair. llama.cpp serves neither on CUDA, so these are ours: `iqs` picks one of the
// eight 32-value runs of the superblock, the run is decoded to int8 codes through the shared
// scalar decoder and dotted against its q8_1 block. Correct rather than fast -- no GGUF on the
// board carries them.
__device__ __forceinline__ float vec_dot_tq1_0_q8_1(const void* __restrict__ vbq,
                                                    const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_tq1_0& x = static_cast<const block_tq1_0*>(vbq)[kbx];
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int v0 = 32 * iqs + 4 * j;
        const char4 c = make_char4(static_cast<signed char>(tq1_0_trit(x, v0 + 0)), static_cast<signed char>(tq1_0_trit(x, v0 + 1)),
                                   static_cast<signed char>(tq1_0_trit(x, v0 + 2)), static_cast<signed char>(tq1_0_trit(x, v0 + 3)));
        sumi = __dp4a(*reinterpret_cast<const int*>(&c), get_int_b4(bq8_1[iqs].qs, j), sumi);
    }
    return __half2float(x.d) * __low2float(bq8_1[iqs].ds) * sumi;
}

__device__ __forceinline__ float vec_dot_tq2_0_q8_1(const void* __restrict__ vbq,
                                                    const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_tq2_0& x = static_cast<const block_tq2_0*>(vbq)[kbx];
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int v0 = 32 * iqs + 4 * j;
        const char4 c = make_char4(static_cast<signed char>(tq2_0_code(x, v0 + 0)), static_cast<signed char>(tq2_0_code(x, v0 + 1)),
                                   static_cast<signed char>(tq2_0_code(x, v0 + 2)), static_cast<signed char>(tq2_0_code(x, v0 + 3)));
        sumi = __dp4a(*reinterpret_cast<const int*>(&c), get_int_b4(bq8_1[iqs].qs, j), sumi);
    }
    return __half2float(x.d) * __low2float(bq8_1[iqs].ds) * sumi;
}

} // namespace sinfer::ops::detail::ggml
