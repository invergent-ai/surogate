#pragma once

#include "ops/linear/ggml/ggml_mmvq.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::ggml {

/// Token count from which generic formats use wider token tiles. Overridable with
/// SUROGATE_GGML_WIDE_MIN_TOKENS; changing it does not change projection arithmetic.
/// Q4_K/Q5_K/Q6_K select their specialized tiles independently.
std::int32_t wide_min_tokens() noexcept;

/// Workspace bytes for `tokens` columns of a [rows, k] GGML weight: int8 activation
/// codes and scales. F16 uses the original activation and leaves this scratch unused.
std::size_t linear_workspace_bytes(std::int32_t rows, std::int32_t k, std::int32_t tokens) noexcept;

/// out[rows, tokens] = W · x. x BF16 [k, tokens], out BF16 [rows, tokens], both contiguous.
/// `scratch` holds at least linear_workspace_bytes(rows, k, tokens), 16-byte aligned.
void linear_launch(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                   const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* out, void* scratch,
                   std::size_t scratch_bytes, cudaStream_t stream);

/// K-quant projection consuming activation planes already produced by quantize_q8_1_planes.
void linear_prequantized_launch(GgmlType type, const void* blocks, std::int32_t rows,
                                std::int32_t k, std::int32_t tokens, __nv_bfloat16* out,
                                void* scratch, std::size_t scratch_bytes, cudaStream_t stream);

/// residual[rows, tokens] += W · x: the same routes accumulating into the residual.
void linear_add_launch(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                       const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* residual,
                       void* scratch, std::size_t scratch_bytes, cudaStream_t stream);

/// out[k, tokens] with each 32-row group `d` of x moved to group `group_map[d]`: the
/// activation rearranged to the column order a weight was stored in. `group_map` is on the
/// device, k / 32 entries, a permutation.
void permute_column_groups_launch(const __nv_bfloat16* x, std::int32_t k, std::int32_t tokens,
                                  const std::int32_t* group_map, __nv_bfloat16* out,
                                  cudaStream_t stream);

/// Same as linear_launch, fp32 output (tests and fp32 consumers); GEMV route only.
void linear_launch_f32(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                       const __nv_bfloat16* x, std::int32_t tokens, float* out, void* scratch,
                       std::size_t scratch_bytes, cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
