#include "ops/linear/ggml/ggml_host_decode.h"

#include "ops/linear/ggml/ggml_moe_codec.cuh"
#include "api/ops/expert_slot_cache.h" // kQ5GroupBytes

#include <cmath>
#include <cstring>

namespace sinfer::ops {
namespace {

/// One row, group by group. `decode_group_32` is the same function the gather calls on the
/// device -- compiled here for the host, not reimplemented for it.
template <detail::ggml::GgmlType type>
void decode_row(const void* blocks, std::int64_t k, std::int8_t* codes, std::uint16_t* scales) {
    const std::int64_t groups = k / 32;
    for (std::int64_t g = 0; g < groups; ++g) {
        float value[32];
        detail::ggml::decode_group_32<type>(blocks, g, value);
        float amax = 0.0F;
        for (float v : value) { amax = std::fmax(amax, std::fabs(v)); }
        const float scale = amax / 127.0F;
        const float inv   = scale > 0.0F ? 1.0F / scale : 0.0F;
        for (int i = 0; i < 32; ++i) {
            const float q = std::nearbyint(value[i] * inv);
            codes[g * 32 + i] =
                static_cast<std::int8_t>(q < -127.0F ? -127.0F : (q > 127.0F ? 127.0F : q));
        }
        scales[g] = __half_as_ushort(__float2half_rn(scale));
    }
}

template <detail::ggml::GgmlType type>
void decode_row_float(const void* blocks, std::int64_t k, float* out) {
    const std::int64_t groups = k / 32;
    for (std::int64_t g = 0; g < groups; ++g) {
        float value[32];
        detail::ggml::decode_group_32<type>(blocks, g, value);
        for (int i = 0; i < 32; ++i) { out[g * 32 + i] = value[i]; }
    }
}

std::uint16_t half_bits(float v) { return __half_as_ushort(__float2half_rn(v)); }

/// Sixteen code bytes of one Q4G32AM group from a 32-value nibble source: value v is the
/// nibble `pick(v)`, and the group packs values 2i and 2i+1 into byte i.
template <class Pick>
void pack_group(std::uint8_t* out, Pick pick) {
    for (int i = 0; i < 16; ++i) {
        out[i] = static_cast<std::uint8_t>((pick(2 * i) & 0xF) | ((pick(2 * i + 1) & 0xF) << 4));
    }
}

void q4_K_row_to_q4g32am(const void* blocks, std::int64_t k, std::uint8_t* codes,
                         std::uint16_t* scales, std::uint16_t* mins) {
    const auto* x = static_cast<const detail::ggml::block_q4_K*>(blocks);
    for (std::int64_t b = 0; b < k / detail::ggml::QK_K; ++b) {
        const float d    = __half2float(__low2half(x[b].dm));
        const float dmin = __half2float(__high2half(x[b].dm));
        for (int is = 0; is < 8; ++is) {
            std::uint8_t sc = 0, m = 0;
            detail::ggml::get_scale_min_k4(is, x[b].scales, sc, m);
            const std::int64_t g  = b * 8 + is;
            scales[g]             = half_bits(d * static_cast<float>(sc));
            mins[g]               = half_bits(-(dmin * static_cast<float>(m)));
            // Sub-blocks come in pairs of 32 bytes: the even one in the low nibbles, the odd
            // one in the high nibbles, value v in byte v of the pair.
            const std::uint8_t* q = x[b].qs + 32 * (is / 2);
            const int shift       = (is & 1) != 0 ? 4 : 0;
            pack_group(codes + g * 16, [&](int v) { return q[v] >> shift; });
        }
    }
}

void q4_0_row_to_q4g32am(const void* blocks, std::int64_t k, std::uint8_t* codes,
                         std::uint16_t* scales, std::uint16_t* mins) {
    const auto* x = static_cast<const detail::ggml::block_q4_0*>(blocks);
    for (std::int64_t g = 0; g < k / 32; ++g) {
        const float d = __half2float(x[g].d);
        scales[g]     = half_bits(d);
        mins[g]       = half_bits(-8.0F * d); // value = d * (q - 8)
        const std::uint8_t* q = x[g].qs;
        pack_group(codes + g * 16, [&](int v) { return v < 16 ? q[v] : q[v - 16] >> 4; });
    }
}

void q4_1_row_to_q4g32am(const void* blocks, std::int64_t k, std::uint8_t* codes,
                         std::uint16_t* scales, std::uint16_t* mins) {
    const auto* x = static_cast<const detail::ggml::block_q4_1*>(blocks);
    for (std::int64_t g = 0; g < k / 32; ++g) {
        scales[g] = half_bits(__half2float(__low2half(x[g].dm)));
        mins[g]   = half_bits(__half2float(__high2half(x[g].dm))); // value = d * q + m
        const std::uint8_t* q = x[g].qs;
        pack_group(codes + g * 16, [&](int v) { return v < 16 ? q[v] : q[v - 16] >> 4; });
    }
}

/// Twenty code bytes of one Q5G32AM group: the low nibbles packed pairwise, then the fifth
/// bits as a 32-bit word whose bit v is value v's.
template <class Pick>
void pack_group_q5(std::uint8_t* out, Pick low, std::uint32_t high) {
    pack_group(out, low);
    std::memcpy(out + 16, &high, sizeof(high));
}

void q5_K_row_to_q5g32am(const void* blocks, std::int64_t k, std::uint8_t* codes,
                         std::uint16_t* scales, std::uint16_t* mins) {
    const auto* x = static_cast<const detail::ggml::block_q5_K*>(blocks);
    for (std::int64_t b = 0; b < k / detail::ggml::QK_K; ++b) {
        const float d    = __half2float(__low2half(x[b].dm));
        const float dmin = __half2float(__high2half(x[b].dm));
        for (int is = 0; is < 8; ++is) {
            std::uint8_t sc = 0, m = 0;
            detail::ggml::get_scale_min_k4(is, x[b].scales, sc, m);
            const std::int64_t g  = b * 8 + is;
            scales[g]             = half_bits(d * static_cast<float>(sc));
            mins[g]               = half_bits(-(dmin * static_cast<float>(m)));
            // Low nibbles as Q4_K's: sub-block pairs of 32 bytes, the even one low, the odd
            // one high. Fifth bits: bit `is` of qh[v] for value v of sub-block `is`.
            const std::uint8_t* q = x[b].qs + 32 * (is / 2);
            const int shift       = (is & 1) != 0 ? 4 : 0;
            std::uint32_t high    = 0;
            for (int v = 0; v < 32; ++v) {
                high |= static_cast<std::uint32_t>((x[b].qh[v] >> is) & 1U) << v;
            }
            pack_group_q5(codes + g * kQ5GroupBytes, [&](int v) { return q[v] >> shift; }, high);
        }
    }
}

void q5_0_row_to_q5g32am(const void* blocks, std::int64_t k, std::uint8_t* codes,
                         std::uint16_t* scales, std::uint16_t* mins) {
    const auto* x = static_cast<const detail::ggml::block_q5_0*>(blocks);
    for (std::int64_t g = 0; g < k / 32; ++g) {
        const float d = __half2float(x[g].d);
        scales[g]     = half_bits(d);
        mins[g]       = half_bits(-16.0F * d); // value = d * (q - 16)
        std::uint32_t high = 0;
        std::memcpy(&high, x[g].qh, sizeof(high));
        const std::uint8_t* q = x[g].qs;
        pack_group_q5(codes + g * kQ5GroupBytes, [&](int v) { return v < 16 ? q[v] : q[v - 16] >> 4; },
                      high);
    }
}

void q5_1_row_to_q5g32am(const void* blocks, std::int64_t k, std::uint8_t* codes,
                         std::uint16_t* scales, std::uint16_t* mins) {
    const auto* x = static_cast<const detail::ggml::block_q5_1*>(blocks);
    for (std::int64_t g = 0; g < k / 32; ++g) {
        scales[g] = half_bits(__half2float(__low2half(x[g].dm)));
        mins[g]   = half_bits(__half2float(__high2half(x[g].dm))); // value = d * q + m
        std::uint32_t high = 0;
        std::memcpy(&high, x[g].qh, sizeof(high));
        const std::uint8_t* q = x[g].qs;
        pack_group_q5(codes + g * kQ5GroupBytes, [&](int v) { return v < 16 ? q[v] : q[v - 16] >> 4; },
                      high);
    }
}

} // namespace

bool ggml_row_to_q5g32am(QType type, const void* blocks, std::int64_t k, std::uint8_t* codes,
                         std::uint16_t* scales, std::uint16_t* mins) noexcept {
    if (blocks == nullptr || codes == nullptr || scales == nullptr || mins == nullptr || k <= 0 ||
        k % 32 != 0) {
        return false;
    }
    switch (type) {
    case QType::Q5_K:
        if (k % detail::ggml::QK_K != 0) { return false; }
        q5_K_row_to_q5g32am(blocks, k, codes, scales, mins);
        return true;
    case QType::Q5_0:
        q5_0_row_to_q5g32am(blocks, k, codes, scales, mins);
        return true;
    case QType::Q5_1:
        q5_1_row_to_q5g32am(blocks, k, codes, scales, mins);
        return true;
    default:
        return false;
    }
}

bool ggml_row_to_q4g32am(QType type, const void* blocks, std::int64_t k, std::uint8_t* codes,
                         std::uint16_t* scales, std::uint16_t* mins) noexcept {
    if (blocks == nullptr || codes == nullptr || scales == nullptr || mins == nullptr || k <= 0 ||
        k % 32 != 0) {
        return false;
    }
    switch (type) {
    case QType::Q4_K:
        if (k % detail::ggml::QK_K != 0) { return false; }
        q4_K_row_to_q4g32am(blocks, k, codes, scales, mins);
        return true;
    case QType::Q4_0:
        q4_0_row_to_q4g32am(blocks, k, codes, scales, mins);
        return true;
    case QType::Q4_1:
        q4_1_row_to_q4g32am(blocks, k, codes, scales, mins);
        return true;
    default:
        return false;
    }
}

bool ggml_decode_row_float(QType type, const void* blocks, std::int64_t k, float* out) noexcept {
    using T = detail::ggml::GgmlType;
    if (blocks == nullptr || out == nullptr || k <= 0 || k % 32 != 0) { return false; }
    switch (type) {
#define SINFER_HOST_DECODE_FLOAT_CASE(NAME)                                                        \
    case QType::NAME:                                                                              \
        decode_row_float<T::NAME>(blocks, k, out);                                                 \
        return true;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_HOST_DECODE_FLOAT_CASE)
#undef SINFER_HOST_DECODE_FLOAT_CASE
    default:
        return false;
    }
}

bool ggml_decode_row_w8(QType type, const void* blocks, std::int64_t k, std::int8_t* codes,
                        std::uint16_t* scales) noexcept {
    using T = detail::ggml::GgmlType;
    if (blocks == nullptr || codes == nullptr || scales == nullptr || k <= 0 || k % 32 != 0) {
        return false;
    }
    switch (type) {
#define SINFER_HOST_DECODE_CASE(NAME)                                                              \
    case QType::NAME:                                                                              \
        decode_row<T::NAME>(blocks, k, codes, scales);                                             \
        return true;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_HOST_DECODE_CASE)
#undef SINFER_HOST_DECODE_CASE
    default:
        return false;
    }
}

std::int64_t ggml_row_bytes(QType type, std::int64_t k) noexcept {
    using T = detail::ggml::GgmlType;
    if (k <= 0) { return 0; }
    switch (type) {
#define SINFER_HOST_ROW_BYTES_CASE(NAME)                                                           \
    case QType::NAME: {                                                                            \
        constexpr std::int64_t values = detail::ggml::block_values(T::NAME);                       \
        constexpr std::int64_t bytes  = detail::ggml::block_bytes(T::NAME);                        \
        return k % values == 0 ? k / values * bytes : 0;                                           \
    }
        SINFER_GGML_FOR_EACH_TYPE(SINFER_HOST_ROW_BYTES_CASE)
#undef SINFER_HOST_ROW_BYTES_CASE
    default:
        return 0;
    }
}

} // namespace sinfer::ops
