#pragma once

// sinfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void rope_interleaved_launch(const Tensor& positions, int rotary_dim, float theta,
                             int height_pairs, int width_pairs, Tensor& q, Tensor& k,
                             cudaStream_t stream);

void rope_launch(const Tensor& positions, int rotary_dim, int active_pairs, float theta,
                 Tensor& q, Tensor& k, cudaStream_t stream);

void rope_single_launch(const Tensor& positions, int rotary_dim, int active_pairs, float theta,
                        Tensor& x, cudaStream_t stream);

} // namespace sinfer::ops::detail
