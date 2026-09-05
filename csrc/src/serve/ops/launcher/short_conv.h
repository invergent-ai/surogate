#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void short_conv_launch(const Tensor& bcx, const Tensor& taps, Tensor& state, Tensor& out,
                       std::int32_t channels, cudaStream_t stream);

} // namespace sinfer::ops::detail
