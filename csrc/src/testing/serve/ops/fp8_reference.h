#pragma once

#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace sinfer::test {

// Independent scalar E4M3FN oracle: enumerate the finite positive values and
// choose the nearest code, resolving exact ties by an even significand.
inline float fp8_reference_value(std::uint8_t code) {
    if ((code & 127) == 127) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    const float magnitude =
        exponent ? std::ldexp(1.F + mantissa / 8.F, exponent - 7) : std::ldexp(static_cast<float>(mantissa), -9);
    return std::copysign(magnitude, code & 128 ? -1.F : 1.F);
}

inline std::uint8_t fp8_reference_code(float value) {
    static const auto values = [] {
        std::array<float, 127> out{};
        for (int code = 0; code < 127; ++code) {
            out[code] = fp8_reference_value(code);
        }
        return out;
    }();
    const auto sign = static_cast<std::uint8_t>(std::signbit(value) ? 128 : 0);
    if (std::isnan(value)) {
        return sign | 127;
    }
    const float magnitude = std::abs(value);
    const auto high = std::lower_bound(values.begin(), values.end(), magnitude);
    if (high == values.end()) {
        return sign | 126;
    }
    const int upper = static_cast<int>(high - values.begin());
    if (upper == 0) {
        return sign;
    }
    const float below = magnitude - values[upper - 1], above = values[upper] - magnitude;
    return sign | static_cast<std::uint8_t>(below < above || (below == above && (upper & 1)) ? upper - 1 : upper);
}

template <typename Code>
inline std::vector<Code> kv_reference_bits(const std::vector<float>& values) {
    std::vector<Code> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if constexpr (std::is_same_v<Code, std::uint8_t>) {
            out[i] = fp8_reference_code(values[i]);
        } else {
            out[i] = f32_to_bf16(values[i]);
        }
    }
    return out;
}

template <typename Code>
inline void round_kv_reference(std::vector<float>& values) {
    for (auto& value : values) {
        if constexpr (std::is_same_v<Code, std::uint8_t>) {
            value = fp8_reference_value(fp8_reference_code(value));
        } else {
            value = bf16_to_f32(f32_to_bf16(value));
        }
    }
}

template <typename Code>
inline constexpr DType kv_reference_dtype = std::is_same_v<Code, std::uint8_t> ? DType::FP8_E4M3FN : DType::BF16;

}  // namespace sinfer::test
