#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "api/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Largest output segment of the FP8 attention projection (the query rows): the cuBLASLt
// route stages one segment at a time. The launcher asserts it against the output layout.
constexpr std::int32_t kFp8AttnInputCublasLtStagingRows = 6144;

[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
