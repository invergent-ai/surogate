#pragma once

// sinfer::ops::detail - private launch prototypes for mask_columns.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void mask_columns_zero_launch(Tensor& matrix, const Tensor& valid_columns, cudaStream_t stream);

} // namespace sinfer::ops::detail
