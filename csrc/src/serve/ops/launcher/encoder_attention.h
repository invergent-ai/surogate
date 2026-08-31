#pragma once

// sinfer::ops::detail — private launch prototype for encoder_attention. Included by the
// wrapper (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void encoder_attention_launch(const Tensor& qkv, std::int32_t q_heads, std::int32_t head_dim,
                              std::int32_t window, float scale, Tensor& out, void* workspace,
                              cudaStream_t stream);

void encoder_attention_prewarm();

} // namespace sinfer::ops::detail
