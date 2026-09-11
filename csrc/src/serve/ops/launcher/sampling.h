#pragma once

// sinfer::ops::detail - private launch prototype for sample.

#include "core/tensor.h"
#include "api/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void sampling_update_greedy_targets_launch(const Tensor& logits, Tensor& targets,
                                    std::int32_t token_domain, const SamplingConfig* configs,
                                    cudaStream_t stream);

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream);

[[nodiscard]] std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain,
                                                         std::int32_t columns);

} // namespace sinfer::ops::detail
