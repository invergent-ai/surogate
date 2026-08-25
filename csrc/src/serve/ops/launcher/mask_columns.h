#pragma once

// ninfer::ops::detail - private launch prototypes for mask_columns.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void mask_columns_zero_launch(Tensor& matrix, const Tensor& valid_columns, cudaStream_t stream);

} // namespace ninfer::ops::detail
