#include "ops/linear/ggml/ggml_host_decode.h"

#include "ops/linear/ggml/ggml_moe_codec.cuh"

#include <cmath>

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

} // namespace

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
