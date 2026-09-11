#pragma once

#include "api/ops/linear.h"
#include "ops/linear/w8/w8_launch.h"

#include <cstdint>

namespace sinfer::ops::detail {

W8Launch select_w8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
W8Launch select_w8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);
bool w8_uses_stable_accumulation(std::int32_t n, std::int32_t k);

void w8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

} // namespace sinfer::ops::detail
