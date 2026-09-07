#pragma once

// surogate vendor patch (PATCHES.md #17): W8A8-int IMMA dispatchers
// (split2/split4/residual direct-write forms; workspace = act-quant only).

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/linear/w8a8/w8a8_act_quant.h"

namespace sinfer::ops::detail {

void w8a8_gemm_split2(const Tensor& x, const Weight& weight, Tensor& first, Tensor& second,
                      WorkspaceArena& workspace, cudaStream_t stream);

void w8a8_gemm_split4(const Tensor& x, const Weight& weight, Tensor& query, Tensor& key,
                      Tensor& gate, Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);

void w8a8_gemm_residual(const Tensor& x, const Weight& weight, Tensor& residual_out,
                        WorkspaceArena& workspace, cudaStream_t stream);

} // namespace sinfer::ops::detail
