#pragma once

// sinfer::ops::detail -- private launch prototype for head_linear. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail {

// Host entry; assumes the wrapper validated everything. `n` and `k` are one head's matrix.
void head_linear_launch(const Tensor& x, const Weight& w, std::int32_t heads, std::int32_t n,
                        std::int32_t k, float out_scale, Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops::detail
