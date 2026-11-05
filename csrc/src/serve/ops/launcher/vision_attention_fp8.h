#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

/// Whether the current device runs the e4m3 vision attention (compute capability 12.x: its P V
/// product reads V with ldmatrix .trans .b8) at this head dim (64 or 72).
[[nodiscard]] bool vision_attention_fp8_supported(std::int32_t head_dim);

/// Bytes of scratch the prepared operands of `heads` x `patches` take; 0 for an unsupported
/// head dim.
[[nodiscard]] std::size_t vision_attention_fp8_scratch_bytes(std::int32_t head_dim,
                                                             std::int32_t heads,
                                                             std::int32_t patches);

/// Prepares Q, K, V into `scratch` and runs the e4m3 attention; `tiles` as
/// vision_attention_launch's (null for one segment).
void vision_attention_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& cu_seqlens, Tensor* tiles, DeviceSpan scratch,
                                 Tensor& out, cudaStream_t stream, float scale);

} // namespace sinfer::ops::detail
