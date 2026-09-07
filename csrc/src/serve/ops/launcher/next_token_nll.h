#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void next_token_nll_launch(const Tensor& logits, const Tensor& targets, Tensor& out,
                           Tensor* argmax, std::int32_t token_domain, cudaStream_t stream);

} // namespace sinfer::ops::detail
