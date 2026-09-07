#pragma once

// sinfer::ops::detail - private launch prototype for l2norm.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void l2norm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops::detail
