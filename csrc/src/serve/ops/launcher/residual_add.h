#pragma once

// sinfer::ops::detail - private launch prototype for residual_add. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void residual_add_launch(const Tensor& y, Tensor& x, cudaStream_t stream);

} // namespace sinfer::ops::detail
