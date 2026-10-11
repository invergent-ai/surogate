#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail {

void dflash_conv_launch(const Tensor& input, const Tensor& dynamic, const Tensor& base,
                        std::int32_t stage, std::int32_t width, const Tensor* residual, Tensor& out,
                        std::int32_t groups, cudaStream_t stream);

void dflash_select_top_k_launch(const Tensor& logits, std::int32_t domain, Tensor& ids,
                                Tensor& values, cudaStream_t stream);

void dflash_select_walk_launch(const Tensor& anchors, const Tensor& ids, const Tensor& values,
                               const Tensor& hidden, const Tensor& predecessor,
                               const Tensor& successor, std::int32_t positions, Tensor& drafts,
                               cudaStream_t stream);

} // namespace sinfer::ops::detail
