#pragma once

#include "ops/linear/ggml/ggml_moe_codec.cuh"
#include "ops/linear/ggml/ggml_vecdot.cuh"

namespace sinfer::ops::detail::ggml {

// The stored integer/codebook value and its affine transform remain separate
// until after the integer dot product. No weight requantization is involved.
template <GgmlType> struct DenseFormat;
#define SINFER_DENSE_FORMAT(Type, BlockType, Width, Bias) \
    template <> struct DenseFormat<GgmlType::Type> { \
        using Block = BlockType; \
        static constexpr int kWidth = Width; \
        static constexpr bool kBias = Bias; \
    };
SINFER_DENSE_FORMAT(Q2_K, block_q2_K, 16, true)
SINFER_DENSE_FORMAT(Q3_K, block_q3_K, 16, false)
SINFER_DENSE_FORMAT(Q8_0, block_q8_0, 32, false)
SINFER_DENSE_FORMAT(Q4_0, block_q4_0, 32, false)
SINFER_DENSE_FORMAT(Q4_1, block_q4_1, 32, true)
SINFER_DENSE_FORMAT(Q5_0, block_q5_0, 32, false)
SINFER_DENSE_FORMAT(Q5_1, block_q5_1, 32, true)
SINFER_DENSE_FORMAT(IQ4_NL, block_iq4_nl, 32, false)
SINFER_DENSE_FORMAT(IQ2_XXS, block_iq2_xxs, 32, false)
SINFER_DENSE_FORMAT(IQ2_XS, block_iq2_xs, 16, false)
SINFER_DENSE_FORMAT(IQ2_S, block_iq2_s, 16, false)
SINFER_DENSE_FORMAT(IQ3_XXS, block_iq3_xxs, 32, false)
SINFER_DENSE_FORMAT(IQ3_S, block_iq3_s, 32, false)
SINFER_DENSE_FORMAT(IQ1_S, block_iq1_s, 32, false)
SINFER_DENSE_FORMAT(IQ1_M, block_iq1_m, 16, false)
SINFER_DENSE_FORMAT(IQ4_XS, block_iq4_xs, 32, false)
SINFER_DENSE_FORMAT(TQ1_0, block_tq1_0, 32, false)
SINFER_DENSE_FORMAT(TQ2_0, block_tq2_0, 32, false)
SINFER_DENSE_FORMAT(MXFP4, block_mxfp4, 32, false)
SINFER_DENSE_FORMAT(NVFP4_GGML, block_nvfp4, 16, false)
SINFER_DENSE_FORMAT(Q1_0, block_q1_0, 32, false)
SINFER_DENSE_FORMAT(Q2_0, block_q2_0, 32, false)
#undef SINFER_DENSE_FORMAT

struct DenseAffineEight {
    unsigned lo = 0, hi = 0;
    float scale = 0, bias = 0;
    __device__ __forceinline__ void signs(unsigned bits) {
        const unsigned repeated = bits * 0x01010101u;
        const unsigned s0 = __vcmpne4(repeated & 0x08040201u, 0);
        const unsigned s1 = __vcmpne4(repeated & 0x80402010u, 0);
        lo = __vsub4(lo ^ s0, s0);
        hi = __vsub4(hi ^ s1, s1);
    }
    __device__ __forceinline__ void set(int i, int code) {
        const unsigned byte = static_cast<unsigned>(code) & 255u;
        if (i < 4) { lo |= byte << (8 * i); }
        else { hi |= byte << (8 * (i - 4)); }
    }
};

template <GgmlType Type>
__device__ __forceinline__ DenseAffineEight dense_codes_eight(const void* blocks, int value) {
    constexpr int values = block_values(Type);
    const auto& b = static_cast<const typename DenseFormat<Type>::Block*>(blocks)[value / values];
    const int lane = (value % values) / 8;
    DenseAffineEight out;
    if constexpr (Type == GgmlType::Q2_K || Type == GgmlType::Q3_K) {
        const int v = lane * 8, half = v / 128, stripe = (v % 128) / 32, pos = v % 32;
        const int group = 8 * half + 2 * stripe + pos / 16;
        if constexpr (Type == GgmlType::Q2_K) {
            out.scale = __low2float(b.dm) * (b.scales[group] & 15);
            out.bias = -__high2float(b.dm) * (b.scales[group] >> 4);
            unsigned lo, hi;
            load_eight_bytes<2>(b.qs + 32 * half + pos, lo, hi);
            out.lo = (lo >> (2 * stripe)) & 0x03030303u;
            out.hi = (hi >> (2 * stripe)) & 0x03030303u;
        } else {
            const int s = group < 4 ? (b.scales[group] & 15) | (((b.scales[group + 8] >> 0) & 3) << 4)
                : group < 8 ? (b.scales[group] & 15) | (((b.scales[group + 4] >> 2) & 3) << 4)
                : group < 12 ? (b.scales[group - 8] >> 4) | (((b.scales[group] >> 4) & 3) << 4)
                : (b.scales[group - 8] >> 4) | (((b.scales[group - 4] >> 6) & 3) << 4);
            out.scale = __half2float(b.d) * (s - 32);
            unsigned lo, hi, h0, h1;
            load_eight_bytes<2>(b.qs + 32 * half + pos, lo, hi);
            load_eight_bytes<2>(b.hmask + pos, h0, h1);
            const unsigned mask = 0x01010101u << (4 * half + stripe);
            out.lo = __vsub4((lo >> (2 * stripe)) & 0x03030303u,
                             __vcmpeq4(h0 & mask, 0) & 0x04040404u);
            out.hi = __vsub4((hi >> (2 * stripe)) & 0x03030303u,
                             __vcmpeq4(h1 & mask, 0) & 0x04040404u);
        }
    } else if constexpr (Type == GgmlType::Q8_0) {
        out.scale = __half2float(b.d);
        load_eight_bytes<2>(reinterpret_cast<const std::uint8_t*>(b.qs) + lane * 8, out.lo, out.hi);
    } else if constexpr (Type == GgmlType::Q4_0 || Type == GgmlType::Q4_1 ||
                         Type == GgmlType::Q5_0 || Type == GgmlType::Q5_1 || Type == GgmlType::IQ4_NL) {
        constexpr bool five = Type == GgmlType::Q5_0 || Type == GgmlType::Q5_1;
        constexpr bool bias = DenseFormat<Type>::kBias;
        if constexpr (bias) { out.scale = __low2float(b.dm); out.bias = __high2float(b.dm); }
        else { out.scale = __half2float(b.d); }
        unsigned high = 0;
        if constexpr (five) { memcpy(&high, b.qh, sizeof(high)); }
        const int offset = (lane & 1) * 8, shift = 4 * (lane / 2);
        unsigned lo, hi;
        load_eight_bytes<2>(b.qs + offset, lo, hi);
        if constexpr (Type == GgmlType::IQ4_NL) {
            const int2 l = table16_levels(lo, kIq4nlValues), h = table16_levels(hi, kIq4nlValues);
            out.lo = shift ? l.y : l.x;
            out.hi = shift ? h.y : h.x;
        } else {
            out.lo = (lo >> shift) & 0x0f0f0f0fu;
            out.hi = (hi >> shift) & 0x0f0f0f0fu;
            if constexpr (five) {
                const auto expand = [](unsigned h) {
                    return ((h << 4) & 0x10u) | ((h << 11) & 0x1000u) |
                           ((h << 18) & 0x100000u) | ((h << 25) & 0x10000000u);
                };
                out.lo |= expand(high >> (lane * 8));
                out.hi |= expand(high >> (lane * 8 + 4));
            }
            if constexpr (!bias) {
                out.lo = __vsub4(out.lo, five ? 0x10101010u : 0x08080808u);
                out.hi = __vsub4(out.hi, five ? 0x10101010u : 0x08080808u);
            }
        }
    } else if constexpr (Type == GgmlType::IQ2_XXS || Type == GgmlType::IQ2_XS || Type == GgmlType::IQ2_S) {
        const int group = lane / 4, il = lane % 4;
        const std::uint8_t* grid;
        unsigned signs;
        if constexpr (Type == GgmlType::IQ2_XXS) {
            const auto* q = b.qs + 4 * group;
            const unsigned aux = q[2] | (static_cast<unsigned>(q[3]) << 16);
            grid = reinterpret_cast<const std::uint8_t*>(kIq2xxsGrid + reinterpret_cast<const std::uint8_t*>(q)[il]);
            signs = kIq2xsSigns[(aux >> (7 * il)) & 127];
            out.scale = __half2float(b.d) * (0.5f + (aux >> 28)) * 0.25f;
        } else if constexpr (Type == GgmlType::IQ2_XS) {
            const auto q = b.qs[lane];
            grid = reinterpret_cast<const std::uint8_t*>(kIq2xsGrid + (q & 511));
            signs = kIq2xsSigns[q >> 9];
            out.scale = __half2float(b.d) * (0.5f + ((b.scales[group] >> (4 * (il / 2))) & 15)) * 0.25f;
        } else {
            grid = reinterpret_cast<const std::uint8_t*>(kIq2sGrid + (b.qs[lane] | ((b.qh[group] << (8 - 2 * il)) & 0x300)));
            signs = b.qs[QK_K / 8 + lane];
            out.scale = __half2float(b.d) * (0.5f + ((b.scales[group] >> (4 * (il / 2))) & 15)) * 0.25f;
        }
        const auto packed = *reinterpret_cast<const uint2*>(grid);
        out.lo = packed.x; out.hi = packed.y;
        out.signs(signs);
    } else if constexpr (Type == GgmlType::IQ3_XXS || Type == GgmlType::IQ3_S) {
        const int group = lane / 4, il = lane % 4;
        const std::uint8_t *grid0, *grid1;
        unsigned signs;
        if constexpr (Type == GgmlType::IQ3_XXS) {
            const auto* gas = reinterpret_cast<const std::uint16_t*>(b.qs + QK_K / 4) + 2 * group;
            const unsigned aux = gas[0] | (static_cast<unsigned>(gas[1]) << 16);
            grid0 = reinterpret_cast<const std::uint8_t*>(kIq3xxsGrid + b.qs[2 * lane]);
            grid1 = reinterpret_cast<const std::uint8_t*>(kIq3xxsGrid + b.qs[2 * lane + 1]);
            signs = kIq2xsSigns[(aux >> (7 * il)) & 127];
            out.scale = __half2float(b.d) * (0.5f + (aux >> 28)) * 0.5f;
        } else {
            grid0 = reinterpret_cast<const std::uint8_t*>(kIq3sGrid + (b.qs[2 * lane] | ((b.qh[group] << (8 - 2 * il)) & 256)));
            grid1 = reinterpret_cast<const std::uint8_t*>(kIq3sGrid + (b.qs[2 * lane + 1] | ((b.qh[group] << (7 - 2 * il)) & 256)));
            signs = b.signs[lane];
            out.scale = __half2float(b.d) * (1 + 2 * ((b.scales[group / 2] >> (4 * (group % 2))) & 15));
        }
        out.lo = *reinterpret_cast<const unsigned*>(grid0);
        out.hi = *reinterpret_cast<const unsigned*>(grid1);
        out.signs(signs);
    } else if constexpr (Type == GgmlType::IQ1_S || Type == GgmlType::IQ1_M) {
        const int group = lane / 4, il = lane % 4;
        unsigned index;
        int delta;
        if constexpr (Type == GgmlType::IQ1_S) {
            index = b.qs[lane] | (((b.qh[group] >> (3 * il)) & 7) << 8);
            delta = (b.qh[group] & 0x8000) ? -1 : 1;
            out.scale = __half2float(b.d) * (2 * ((b.qh[group] >> 12) & 7) + 1) * 0.125f;
        } else {
            const auto* sc = reinterpret_cast<const std::uint16_t*>(b.scales);
            const int ib16 = lane / 2;
            index = b.qs[lane] | (((b.qh[lane / 2] >> (4 * (il % 2))) & 7) << 8);
            delta = (b.qh[lane / 2] & (0x08 << (4 * (il % 2)))) ? -1 : 1;
            out.scale = __half2float(iq1m_block_scale(b)) * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 7) + 1) * 0.125f;
        }
        const unsigned grid = kIq1sGridGpu[index];
        const auto levels = [delta](unsigned q) {
            const unsigned centered = __vsub4(q & 0x0f0f0f0fu, 0x01010101u);
            return __vadd4((centered & 0x1f1f1f1fu) << 3,
                           delta < 0 ? 0xffffffffu : 0x01010101u);
        };
        out.lo = levels(grid); out.hi = levels(grid >> 4);
    } else if constexpr (Type == GgmlType::IQ4_XS) {
        const int group = lane / 4, pos = (lane % 2) * 8;
        const int scale = ((b.scales_l[group / 2] >> (4 * (group % 2))) & 15) |
                          (((b.scales_h >> (2 * group)) & 3) << 4);
        out.scale = __half2float(b.d) * (scale - 32);
        unsigned lo, hi;
        load_eight_bytes<8>(b.qs + 16 * group + pos, lo, hi);
        const int2 l = table16_levels(lo, kIq4nlValues), h = table16_levels(hi, kIq4nlValues);
        out.lo = lane % 4 >= 2 ? l.y : l.x;
        out.hi = lane % 4 >= 2 ? h.y : h.x;
    } else if constexpr (Type == GgmlType::TQ1_0 || Type == GgmlType::TQ2_0) {
        out.scale = __half2float(b.d);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if constexpr (Type == GgmlType::TQ1_0) { out.set(j, tq1_0_trit(b, lane * 8 + j)); }
            else { out.set(j, tq2_0_code(b, lane * 8 + j)); }
        }
    } else if constexpr (Type == GgmlType::MXFP4 || Type == GgmlType::NVFP4_GGML) {
        int offset, shift;
        if constexpr (Type == GgmlType::MXFP4) {
            out.scale = e8m0_to_fp32_half(b.e);
            offset = (lane % 2) * 8;
            shift = (lane / 2) * 4;
        } else {
            out.scale = ue4m3_to_fp32_half(b.d[lane / 2]);
            offset = (lane / 2) * 8;
            shift = (lane % 2) * 4;
        }
        unsigned lo, hi;
        load_eight_bytes<Type == GgmlType::MXFP4 ? 1 : 2>(b.qs + offset, lo, hi);
        const int2 l = table16_levels(lo, kFp4Values), h = table16_levels(hi, kFp4Values);
        out.lo = shift ? l.y : l.x;
        out.hi = shift ? h.y : h.x;
    } else if constexpr (Type == GgmlType::Q1_0 || Type == GgmlType::Q2_0) {
        out.scale = __half2float(b.d);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if constexpr (Type == GgmlType::Q1_0) { out.set(j, ((b.qs[lane] >> j) & 1) ? 1 : -1); }
            else { out.set(j, ((b.qs[(lane * 8 + j) / 4] >> (2 * (j % 4))) & 3) - 1); }
        }
    }
    return out;
}

} // namespace sinfer::ops::detail::ggml
