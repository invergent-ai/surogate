#pragma once

#include "ops/linear/ggml/ggml_mmvq.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::ggml {

/// Workspace this route needs for `tokens` columns: the block_q8_1 activation planes.
std::size_t linear_workspace_bytes(std::int32_t k, std::int32_t tokens) noexcept;

/// out[n, tokens] = W · x for any token count. x BF16 [k, tokens], out BF16 [n, tokens], both
/// contiguous. `scratch` holds at least linear_workspace_bytes(k, tokens), 16-byte aligned.
/// Columns beyond 8 run as successive mmvq launches (each re-reads the weight); MMQ replaces
/// that for prefill widths.
void linear_launch(GgmlType type, const void* blocks, std::int32_t n, std::int32_t k,
                   const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* out,
                   void* scratch, std::size_t scratch_bytes, cudaStream_t stream);

/// residual[n, tokens] += W · x: the same route accumulating into the residual.
void linear_add_launch(GgmlType type, const void* blocks, std::int32_t n, std::int32_t k,
                       const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* residual,
                       void* scratch, std::size_t scratch_bytes, cudaStream_t stream);

/// Same, fp32 output (tests and fp32 consumers).
void linear_launch_f32(GgmlType type, const void* blocks, std::int32_t n, std::int32_t k,
                       const __nv_bfloat16* x, std::int32_t tokens, float* out, void* scratch,
                       std::size_t scratch_bytes, cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
