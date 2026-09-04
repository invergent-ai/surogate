// Block-scaled FP8: E4M3 codes with one FP32 scale per 128x128 block -- Hugging Face's
// fine-grained FP8 (`weight_scale_inv`), DeepSeek's recipe. The scale varies along K, so
// the row-scaled route's factorisation (GEMM with unit scales, scales applied after) does not
// hold, and this cuBLASLt admits only scalar FP8 scales on sm_120; the routes here are the
// engine's own, runtime-shaped: a tensor-core tile that applies the block scale once per 128
// of K, and a decode GEMV on exact BF16 activations. Activations for the tile are quantised
// per token per 128, the recipe's own convention.
#pragma once

#include "core/tensor.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::fp8_block {

[[nodiscard]] bool is_fp8_block_qtype(QType qtype) noexcept;

/// Throws unless `w` is a block-scaled FP8 weight: QuantLayout::Fp8Block128, n and k whole
/// 128-blocks, a scale grid.
void require_fp8_block_weight(const Weight& w, const char* op);

/// Bytes of activation planes the tile route stages for `tokens` columns of k rows.
[[nodiscard]] std::size_t workspace_bytes(std::int32_t k, std::int32_t tokens) noexcept;
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          std::int32_t max_tokens);

/// The row range `[row_begin, row_begin + rows)` as a weight of its own; `row_begin` a
/// multiple of 128 so the scale grid splits with it.
[[nodiscard]] Weight weight_rows(const Weight& w, std::int32_t row_begin, std::int32_t rows);

/// out[n, T] = W . x. `workspace` may be null: the engine-slot scratch stands in.
void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena* workspace,
            cudaStream_t stream);
/// residual[n, T] += W . x.
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena* workspace,
                cudaStream_t stream);
/// out[rows, T] = W[row_begin : row_begin + rows, :] . x -- what the fused projections split
/// their parents with.
void project_rows(const Tensor& x, const Weight& w, std::int32_t row_begin, Tensor& out,
                  WorkspaceArena* workspace, cudaStream_t stream);

} // namespace sinfer::ops::detail::fp8_block
