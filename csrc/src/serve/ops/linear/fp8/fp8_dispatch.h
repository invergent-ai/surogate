#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "api/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

/// Whether the row-scaled FP8 route can serve this weight shape: one of the registered
/// geometries with its own production schedules, or any other shape through the generic route
/// (fp8_generic.cu). Only a K that is not a whole number of 32 values is refused -- which a plan
/// may ask before it asks for workspace.
[[nodiscard]] bool fp8_linear_serves(std::int32_t output_rows, std::int32_t input_rows) noexcept;

[[nodiscard]] std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream);

} // namespace sinfer::ops::detail
