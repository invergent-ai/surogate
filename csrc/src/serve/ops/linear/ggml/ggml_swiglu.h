#pragma once

#include "core/arena.h"
#include "ops/linear/ggml/ggml_linear.h"

namespace sinfer::ops::detail::ggml {

bool swiglu_decode_admits(GgmlType gate, GgmlType up, std::int32_t rows, std::int32_t k,
                          std::int32_t tokens) noexcept;

/// Both projections retain their BF16 rounding before SwiGLU. Scratch contains
/// only the shared activation planes; no gate/up output planes are materialized.
void swiglu_decode_launch(GgmlType gate_type, const void* gate, GgmlType up_type, const void* up,
                          std::int32_t rows, std::int32_t k, const __nv_bfloat16* x,
                          std::int32_t tokens, __nv_bfloat16* out, void* scratch,
                          std::size_t scratch_bytes, cudaStream_t stream);

/// Returns false without allocating or writing when the weight/batch needs the
/// general route. Supports contiguous parents, typed segments and input maps.
bool ggml_swiglu_decode(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                        cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
