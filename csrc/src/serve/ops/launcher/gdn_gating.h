#pragma once

// sinfer::ops::detail - private launch prototype for gdn_gating.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                       Tensor& g, Tensor& beta, cudaStream_t stream);

void kda_gating_launch(const Tensor& a, const Tensor& b, const Tensor& A_log,
                       const Tensor& dt_bias, float lower_bound, Tensor& g, Tensor& beta,
                       cudaStream_t stream);

} // namespace sinfer::ops::detail
