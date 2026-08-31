#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void add_bias_launch(const Tensor& bias, Tensor& x, cudaStream_t stream);

} // namespace sinfer::ops::detail
