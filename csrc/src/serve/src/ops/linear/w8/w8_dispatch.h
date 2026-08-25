#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/w8/w8_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

W8Launch select_w8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
W8Launch select_w8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void w8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

} // namespace ninfer::ops::detail
