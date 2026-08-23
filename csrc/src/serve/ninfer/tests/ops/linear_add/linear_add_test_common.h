#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::linear_add {

enum class WeightFormat : std::uint8_t {
    BF16,
    Q5G64F16S,
    W8G32F16S,
};

struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    std::span<const std::int32_t> route_starts;
    std::span<const std::int32_t> route_interiors;
};

bool cuda_available();

int run_shape(std::string_view label, WeightFormat format, const ShapeCase& shape);

} // namespace ninfer::test::linear_add
