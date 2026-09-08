#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void vision_pos_embed_add_launch(const Tensor& table, const Tensor& indices, const Tensor& weights,
                                 Tensor& x, cudaStream_t stream);

void siglip2_pos_embed_add_launch(const Tensor& table, int side, int height, int width, int merge,
                                  Tensor& x, cudaStream_t stream);

} // namespace sinfer::ops::detail
