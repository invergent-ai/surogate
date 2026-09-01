#pragma once

// Identity-free Qwen3.6 family runtime helper.

#include "core/arena.h"
#include "core/tensor.h"
#include <api/family/mtp_alignment.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace sinfer::family::detail {

// Composes the generic scatter Op from the family-provided shifted-window interpretation.
void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const family::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, cudaStream_t stream);

} // namespace sinfer::family::detail
