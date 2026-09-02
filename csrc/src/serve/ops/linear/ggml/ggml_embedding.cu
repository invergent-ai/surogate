// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/ggml/ggml_embedding.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace sinfer::ops::detail::ggml {
namespace {

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

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
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

// One CTA per (token, superblock) expands one 256-value block of the row the token id
// selects, straight into that token's column of out [hidden, tokens]. The dequantisers keep
// llama.cpp's thread layout: 32 threads for Q4_K, 64 for the other four (convert.cu launches
// them so, and each thread writes its fixed share of the block).
template <GgmlType type>
__host__ __device__ constexpr int gather_threads() { return type == GgmlType::Q4_K ? 32 : 64; }

template <GgmlType type>
__global__ void gather_kernel(const void* __restrict__ table, const std::int32_t* __restrict__ ids,
                              __nv_bfloat16* __restrict__ out, const int vocab, const int hidden) {
    const int token = blockIdx.x;
    const int block = blockIdx.y; // superblock along hidden
    const int id    = ids[token];
    __nv_bfloat16* column = out + static_cast<std::size_t>(token) * hidden + block * QK_K;
    if (id < 0 || id >= vocab) {
        // Out-of-range ids read as zero rather than as memory past the table.
        for (int i = threadIdx.x; i < QK_K; i += gather_threads<type>()) {
            column[i] = __float2bfloat16(0.0f);
        }
        return;
    }
    const std::int64_t ib = static_cast<std::int64_t>(id) * (hidden / QK_K) + block;
    if constexpr (type == GgmlType::Q2_K) { dequantize_q2_K(table, ib, column, threadIdx.x); }
    if constexpr (type == GgmlType::Q3_K) { dequantize_q3_K(table, ib, column, threadIdx.x); }
    if constexpr (type == GgmlType::Q4_K) { dequantize_q4_K(table, ib, column, threadIdx.x); }
    if constexpr (type == GgmlType::Q5_K) { dequantize_q5_K(table, ib, column, threadIdx.x); }
    if constexpr (type == GgmlType::Q6_K) { dequantize_q6_K(table, ib, column, threadIdx.x); }
}

template <GgmlType type>
void launch(const void* table, const std::int32_t* ids, std::int32_t tokens, std::int32_t vocab,
            std::int32_t hidden, __nv_bfloat16* out, cudaStream_t stream) {
    const dim3 grid(tokens, hidden / QK_K);
    gather_kernel<type><<<grid, gather_threads<type>(), 0, stream>>>(table, ids, out, vocab, hidden);
}

} // namespace

void embedding_gather_launch(GgmlType type, const void* table, std::int32_t vocab, std::int32_t hidden,
                             const std::int32_t* ids, std::int32_t tokens, __nv_bfloat16* out,
                             cudaStream_t stream) {
    if (table == nullptr || ids == nullptr || out == nullptr || vocab <= 0 || hidden <= 0 ||
        (hidden % QK_K) != 0 || tokens <= 0) {
        throw std::invalid_argument("ggml embedding: table [vocab, hidden] with hidden a multiple of 256");
    }
    switch (type) {
    case GgmlType::Q2_K: launch<GgmlType::Q2_K>(table, ids, tokens, vocab, hidden, out, stream); return;
    case GgmlType::Q3_K: launch<GgmlType::Q3_K>(table, ids, tokens, vocab, hidden, out, stream); return;
    case GgmlType::Q4_K: launch<GgmlType::Q4_K>(table, ids, tokens, vocab, hidden, out, stream); return;
    case GgmlType::Q5_K: launch<GgmlType::Q5_K>(table, ids, tokens, vocab, hidden, out, stream); return;
    case GgmlType::Q6_K: launch<GgmlType::Q6_K>(table, ids, tokens, vocab, hidden, out, stream); return;
    }
    throw std::invalid_argument("ggml embedding: unknown GGML type");
}

void ggml_embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    require_ggml_weight(table, "ggml embedding");
    if (ids.dtype != DType::I32 || out.dtype != DType::BF16 || !ids.is_contiguous() ||
        !out.is_contiguous() || ids.ne[0] <= 0 || ids.ne[1] != 1 || out.ne[0] != table.k ||
        out.ne[1] != ids.ne[0] || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("ggml embedding: ids I32 [T], out BF16 [hidden, T]");
    }
    embedding_gather_launch(ggml_type_for(table.qtype), table.qdata, table.n, table.k,
                            static_cast<const std::int32_t*>(ids.data), ids.ne[0],
                            static_cast<__nv_bfloat16*>(out.data), stream);
}

} // namespace sinfer::ops::detail::ggml
