#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void launch_fp8_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_fp8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_fp8_vocabulary_a16_mma(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream);

/// The shape-generic route: out[n, T] = W . x for any (n, k) the registered table does not hold,
/// at any width, over BF16 activations; `accumulate` adds into `out` instead of overwriting it,
/// which is the residual form. K must be a whole number of 32 values.
void launch_fp8_generic(const Tensor& x, const Weight& weight, Tensor& out, bool accumulate,
                        cudaStream_t stream);

} // namespace sinfer::ops::detail
