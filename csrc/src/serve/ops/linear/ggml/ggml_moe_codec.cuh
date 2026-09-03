// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#pragma once

// Weight decode for the shapes that consume a K-quant as *floats* rather than through the
// integer vec-dot: the sparse-MoE bodies, which multiply a BF16 activation directly. Each lane
// owns eight consecutive values of a 256-value superblock, which every K-quant layout supports
// exactly because eight divides both sub-block sizes (32 for Q4_K/Q5_K, 16 for Q6_K) -- a run of
// eight never straddles a scale boundary or a nibble half.

#include "ops/linear/ggml/ggml_blocks.h"
#include "ops/linear/ggml/ggml_dequant.cuh"

#include <cstdint>

namespace sinfer::ops::detail::ggml {

/// Eight consecutive values starting at `lane * 8` of superblock `ib`.
template <GgmlType type>
__device__ __forceinline__ void decode_eight(const void* blocks, std::int64_t ib, int lane,
                                             float (&w)[8]);

template <>
__device__ __forceinline__ void decode_eight<GgmlType::Q4_K>(const void* blocks, std::int64_t ib,
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
__device__ __forceinline__ void decode_eight<GgmlType::Q5_K>(const void* blocks, std::int64_t ib,
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
__device__ __forceinline__ void decode_eight<GgmlType::Q6_K>(const void* blocks, std::int64_t ib,
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

/// The sparse-MoE codec seam: a 256-value group is one superblock, so a warp's 32 lanes cover it
/// with eight values each. `high` and `scales` are unused -- a superblock carries its own.
template <GgmlType type>
struct GgmlMoeCodec {
    static constexpr bool kGateRowsFirst = true;
    static constexpr int kGroupK         = QK_K;
    static constexpr bool kPackedWord8   = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t*, const std::uint8_t*,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        decode_eight<type>(codes, group_index, lane_in_group, weights);
    }
};

} // namespace sinfer::ops::detail::ggml
