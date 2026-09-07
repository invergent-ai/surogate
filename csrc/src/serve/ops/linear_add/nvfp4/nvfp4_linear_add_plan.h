#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "api/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream);

/// Whether `tokens` takes the wide (cuBLASLt / CUTLASS) GEMM, whose operand is the tiled-scale
/// layout, and that GEMM over an operand a caller quantised itself.
[[nodiscard]] bool nvfp4_linear_add_w4a4_wide(const Weight& weight, std::int32_t tokens);
void nvfp4_linear_add_w4a4_wide_gemm(const Weight& weight, Nvfp4W4a4Workspace workspace,
                                     Tensor& residual, std::int32_t tokens, cudaStream_t stream);

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace sinfer::ops::detail
