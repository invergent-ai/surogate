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

/// Whether vision_attention_fp8_launch can apply this rotation itself, bit for bit as ops::rope
/// would: where ops::rope takes its fixed Vision2D / Vision2D64 kernel -- 16 query and key heads,
/// two position axes, the whole head rotated, theta 10000. (Other head counts take its generic
/// kernel, whose frequencies differ from the fixed table in the last bit.)
[[nodiscard]] bool vision_attention_fp8_rope_supported(std::int32_t head_dim, std::int32_t heads,
                                                       std::int32_t rotary_dim, float theta,
                                                       std::int32_t axes);

/// Prepares Q, K, V into `scratch` and runs the e4m3 attention; `tiles` as
/// vision_attention_launch's (null for one segment). With `rope_positions` ([P, 2] I32, see
/// vision_attention_fp8_rope_supported) Q and K are rotated on the way into e4m3 instead of
/// having been rotated in place.
void vision_attention_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& cu_seqlens, Tensor* tiles, DeviceSpan scratch,
                                 Tensor& out, cudaStream_t stream, float scale,
                                 const Tensor* rope_positions = nullptr);

} // namespace sinfer::ops::detail
