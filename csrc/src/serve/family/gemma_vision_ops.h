#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace sinfer::family::gemma_vision {
// Positions use the common axis-major [y,x] control layout. Tables store [x,y] axes.
void position_add(const Tensor& table, const Tensor& positions, Tensor& x, cudaStream_t stream);
void spatial_rope(const Tensor& positions, float theta, Tensor& q, Tensor& k, cudaStream_t stream);
void pool(const Tensor& x, int merge_unit, float scale, const Tensor* bias, const Tensor* gain,
          Tensor& out, cudaStream_t stream);
void clamp(const Tensor& x, const Tensor& bounds, int offset, Tensor& out, cudaStream_t stream);
} // namespace sinfer::family::gemma_vision
