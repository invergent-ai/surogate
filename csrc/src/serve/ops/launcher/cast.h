#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void cast_fp32_to_bf16_launch(const Tensor& source, Tensor& destination, cudaStream_t stream);

void cast_bf16_to_fp32_launch(const Tensor& source, Tensor& destination, cudaStream_t stream);
} // namespace sinfer::ops::detail
