#pragma once

#include "ops/linear/ggml/ggml_prefill_codec.cuh"

#include <type_traits>

namespace sinfer::ops::detail::ggml {

// One warp projects a row. Lanes decode a whole 256-value superblock in
// parallel, accumulating two independent K partitions. Prefill uses the same
// partitions and final reduction tree, preserving batch-invariant rounding
// without staging a mostly empty matrix tile for each decode token.
template <class Codec, int Columns>
__device__ __forceinline__ void
dense_k_dots(const typename Codec::Block* __restrict__ w, const std::int8_t* __restrict__ x,
             const __half2* __restrict__ scales, int k, float (&acc)[Columns]) {
    constexpr int width = Codec::kScaleWidth;
    constexpr int lanes = width / 8;
    const int lane      = threadIdx.x & 31;
    const int group     = lane / lanes;
    const int offset    = (lane % lanes) * 8;
    static_assert(Columns == 1 || Columns == 2);
#pragma unroll
    for (int c = 0; c < Columns; ++c) { acc[c] = 0.0f; }
#pragma unroll 2
    for (int block = 0; block < k / QK_K; ++block) {
        const auto& b = w[block];
        unsigned q0, q1;
        float2 pair{};
        float ws = 0;
        if constexpr (Codec::kHasMin) {
            const auto* qs  = reinterpret_cast<const unsigned*>(b.qs + 32 * (group / 2) + offset);
            const int shift = 4 * (group & 1);
            q0              = (qs[0] >> shift) & 0x0f0f0f0fu;
            q1              = (qs[1] >> shift) & 0x0f0f0f0fu;
            if constexpr (std::is_same_v<Codec, GgmlQ5KPrefill>) {
                const auto* qh = reinterpret_cast<const unsigned*>(b.qh + offset);
                q0 |= ((qh[0] >> group) & 0x01010101u) << 4;
                q1 |= ((qh[1] >> group) & 0x01010101u) << 4;
            }
            pair = Codec::scale_pair(reinterpret_cast<const std::uint8_t*>(&b), group);
        } else {
            const int half   = group / 8;
            const int stripe = (group % 8) / 2;
            const int pos    = (group & 1) * 16 + offset;
            const int shift  = 4 * (stripe / 2);
            const auto* ql   = b.ql + 64 * half + 32 * (stripe & 1) + pos;
            const auto* qh   = b.qh + 32 * half + pos;
            q0               = GgmlQ6KPrefill::minus_32(
                ((GgmlQ6KPrefill::load_word(ql) >> shift) & 0x0f0f0f0fu) |
                (((GgmlQ6KPrefill::load_word(qh) >> (2 * stripe)) & 0x03030303u) << 4));
            q1 = GgmlQ6KPrefill::minus_32(
                ((GgmlQ6KPrefill::load_word(ql + 4) >> shift) & 0x0f0f0f0fu) |
                (((GgmlQ6KPrefill::load_word(qh + 4) >> (2 * stripe)) & 0x03030303u) << 4));
            ws = __half2float(b.d) * static_cast<float>(b.scales[group]);
        }
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            float scale, value;
            const auto* xv =
                reinterpret_cast<const int*>(x + int64_t(c) * k + block * QK_K + lane * 8);
            if constexpr (Codec::kHasMin) {
                int dot =
                    __dp4a(static_cast<int>(q1), xv[1], __dp4a(static_cast<int>(q0), xv[0], 0));
                int sum = __dp4a(0x01010101, xv[1], __dp4a(0x01010101, xv[0], 0));
                dot += __shfl_xor_sync(0xffffffffu, dot, 1, lanes);
                dot += __shfl_xor_sync(0xffffffffu, dot, 2, lanes);
                sum += __shfl_xor_sync(0xffffffffu, sum, 1, lanes);
                sum += __shfl_xor_sync(0xffffffffu, sum, 2, lanes);
                scale = __low2float(scales[int64_t(c) * (k / 32) + block * 8 + group]);
                value =
                    __fmaf_rn(pair.x, static_cast<float>(dot), -pair.y * static_cast<float>(sum));
            } else {
                int dot =
                    __dp4a(static_cast<int>(q1), xv[1], __dp4a(static_cast<int>(q0), xv[0], 0));
                dot += __shfl_xor_sync(0xffffffffu, dot, 1, lanes);
                scale = ws * __low2float(scales[int64_t(c) * (k / 32) + block * 8 + group / 2]);
                value = static_cast<float>(dot);
            }
#pragma unroll
            for (int g = 0; g < QK_K / width / 2; ++g) {
                const int source = (group % 2 + g * 2) * lanes;
                const float s    = __shfl_sync(0xffffffffu, scale, source);
                const float v    = __shfl_sync(0xffffffffu, value, source);
                acc[c]           = __fmaf_rn(s, v, acc[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < Columns; ++c) { acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], lanes); }
}

template <class Codec>
__device__ __forceinline__ float dense_k_dot(const typename Codec::Block* __restrict__ w,
                                             const std::int8_t* __restrict__ x,
                                             const __half2* __restrict__ scales, int k) {
    float acc[1];
    dense_k_dots<Codec, 1>(w, x, scales, k, acc);
    return acc[0];
}

template <class Codec>
__global__ __launch_bounds__(32) void dense_decode_kernel(
    const typename Codec::Block* __restrict__ weights, const std::int8_t* __restrict__ codes,
    const __half2* __restrict__ ds, int rows, int k, __nv_bfloat16* out, bool accumulate) {
    const int row = blockIdx.x, token = blockIdx.y;
    if (row >= rows) { return; }
    const float acc = dense_k_dot<Codec>(weights + static_cast<std::int64_t>(row) * (k / QK_K),
                                         codes + static_cast<std::int64_t>(token) * k,
                                         ds + static_cast<std::int64_t>(token) * (k / 32), k);
    if ((threadIdx.x & 31) == 0) {
        const auto i = static_cast<std::int64_t>(token) * rows + row;
        out[i]       = __float2bfloat16_rn(acc + (accumulate ? __bfloat162float(out[i]) : 0.0f));
    }
}

} // namespace sinfer::ops::detail::ggml
