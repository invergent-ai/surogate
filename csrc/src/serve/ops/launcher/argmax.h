#pragma once

// sinfer::ops::detail - private launch prototype for argmax.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

} // namespace sinfer::ops::detail
