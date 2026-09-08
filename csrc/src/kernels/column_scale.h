#pragma once

#include "utilities/tensor.h"
#include <cuda_runtime.h>

// Per-channel multiplication and its scale gradient, with FP32 reduction.
void column_scale(Tensor& out, const Tensor& data, const Tensor& scale, long rows, long columns, cudaStream_t stream);
void column_scale_gradient(Tensor& out, const Tensor& grad, const Tensor& data, long rows, long columns, cudaStream_t stream);
