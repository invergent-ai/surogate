#pragma once

// Identity-free Qwen3.6 family runtime helper.

#include "core/arena.h"
#include "core/tensor.h"
#include <api/family/mtp_alignment.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace sinfer::family::detail {

// Adds intermediate visual features at the selected prompt columns. Indices are ordered
// and unique; contiguous image/frame runs share one residual-add launch.
void add_visual_embeddings(Tensor& residual, const Tensor& features,
                           std::span<const std::int32_t> indices, cudaStream_t stream);

// Composes the generic scatter Op from the family-provided shifted-window interpretation.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const family::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, cudaStream_t stream);

} // namespace sinfer::family::detail
