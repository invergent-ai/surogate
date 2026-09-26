#pragma once

// sinfer::ops::detail — private launch prototype for gelu_mul. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void gelu_and_mul_launch(const Tensor& gate, const Tensor& up, bool tanh_approx, Tensor& out,
                         cudaStream_t stream, bool round_gate);

// Host entry for the fused-plane form; `gate_up` is [2K, T], `out` [K, T], K even.
void gelu_and_mul_fused_launch(const Tensor& gate_up, bool tanh_approx, Tensor& out,
                               cudaStream_t stream);

} // namespace sinfer::ops::detail
