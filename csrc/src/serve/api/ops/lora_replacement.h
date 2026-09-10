#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace sinfer::ops {
void lora_replace_linear(const Tensor& x, Tensor& out, const void* const* weights,
                         const Tensor& slots, const std::int32_t* uniform, cudaStream_t stream);
void lora_replace_embedding(const Tensor& ids, Tensor& out, const void* const* weights,
                            const Tensor& slots, const std::int32_t* uniform, cudaStream_t stream);
} // namespace sinfer::ops
