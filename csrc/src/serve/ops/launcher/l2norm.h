#pragma once

// ninfer::ops::detail - private launch prototype for l2norm.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void l2norm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
