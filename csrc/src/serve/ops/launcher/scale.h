#pragma once

// sinfer::ops::detail — private launch prototype for scale. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void scale_launch(Tensor& x, float factor, cudaStream_t stream);

} // namespace sinfer::ops::detail
