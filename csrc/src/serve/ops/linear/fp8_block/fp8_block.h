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
#include "api/ops/linear.h"

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
/// What linear_projections over row ranges of a `parent_rows` parent needs to run them as one
/// launch on Hopper (a staging plane on narrow rounds); an arena with only
/// linear_workspace_capacity_bytes runs them one by one.
[[nodiscard]] std::size_t projections_workspace_capacity_bytes(std::int32_t parent_rows,
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

/// Whether a round of `tokens` columns through `w` runs on quantised activations -- past the
/// GEMV's widths for this weight, which read the exact BF16 activation instead (to four tokens;
/// on Hopper a tall block-FP8 weight leaves the GEMV past two, see gemv_serves).
[[nodiscard]] bool quantizes_activations(const Weight& w, std::int32_t tokens) noexcept;
/// residual[n, T] += W . (silu(gate) * up), from `packed` [2k, T]: each token's k gate rows, then
/// its k up rows (linear_swiglu's packed plane). The activation goes straight into W's quantised
/// operand and is never written as BF16; each value is rounded to BF16 first, as silu_mul's output
/// is, so the result is the unfused pair's. `limit` > 0 clamps as silu_mul does. Only at widths
/// quantizes_activations admits; the workspace need is linear_workspace_capacity_bytes(n, k, T).
void swiglu_linear_add(const Tensor& packed, const Weight& w, Tensor& residual, float limit,
                       WorkspaceArena* workspace, cudaStream_t stream);

/// Each projection's row range into its output. Consecutive ranges of one parent, in order (up
/// to four), run as one launch.
void linear_projections(const Tensor& x, std::span<const LinearProjection> projections,
                        WorkspaceArena* workspace, cudaStream_t stream);

} // namespace sinfer::ops::detail::fp8_block
