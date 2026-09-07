#pragma once

// sinfer::ops::detail — private launch prototype for mean_pool. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void mean_pool_launch(const Tensor& x, int count, bool accumulate, Tensor& out,
                      cudaStream_t stream);

} // namespace sinfer::ops::detail
