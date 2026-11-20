#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace sinfer::test::linear_swiglu {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
};

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ActivationCompute activation_compute;
};

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases);

/// ops::linear_swiglu_preactivation_delta against linear_swiglu: a zero delta must give the same bits, a random delta
/// the fp64 oracle of silu(gate + dg) * (up + du); `expect_route` false checks that the delta form is refused.
int run_preactivation_delta_profile(std::string_view label, const Profile& profile,
                                    std::span<const std::int32_t> token_cases, bool expect_route);

} // namespace sinfer::test::linear_swiglu
