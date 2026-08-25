#pragma once

// ninfer::ops::detail — private launch prototype for silu_mul. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void silu_and_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
