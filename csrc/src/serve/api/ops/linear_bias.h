#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime.h>

namespace sinfer::ops {
// out[N,T] = weight[N,K] * x[K,T] + bias[N]. x, bias and out are contiguous BF16,
// with the same positive dimensions/alignment as linear. Inputs and out must not
// overlap. BF16 weights accumulate the dot product and bias in FP32 before the
// output rounds to BF16. Other supported A16 weight formats round the projection
// before adding the bias. Uses the engine's existing GEMM workspace.
void linear_bias(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                 cudaStream_t stream);
} // namespace sinfer::ops
