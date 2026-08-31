#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

/// low[rank, T] = A[ids[t]] · x[:, t], per token.
void lora_batched_shrink_launch(const Tensor& x, const void* a_bank, const Tensor& ids, Tensor& low,
                                std::int32_t k, std::int32_t rank, std::int64_t a_stride,
                                cudaStream_t stream);

/// out[n, T] += B[ids[t]] · low[:, t], per token.
void lora_batched_expand_launch(const Tensor& low, const void* b_bank, const Tensor& ids,
                                Tensor& out, std::int32_t n, std::int32_t rank,
                                std::int64_t b_stride, cudaStream_t stream);

} // namespace sinfer::ops::detail
