#pragma once

#include "core/arena.h"
#include "ops/linear/ggml/ggml_linear.h"

namespace sinfer::ops::detail::ggml {

bool swiglu_decode_admits(GgmlType gate, GgmlType up, std::int32_t rows, std::int32_t k,
                          std::int32_t tokens) noexcept;

bool swiglu_prefill_admits(GgmlType gate, GgmlType up, std::int32_t rows, std::int32_t k,
                           std::int32_t tokens) noexcept;

void swiglu_prefill_launch(GgmlType type, const void* gate, const void* up,
                           std::int32_t rows, std::int32_t k, const __nv_bfloat16* x,
                           std::int32_t tokens, __nv_bfloat16* out, void* scratch,
                           std::size_t scratch_bytes, cudaStream_t stream);

/// Capacity for known half formats; includes every decode fallback in the interval.
std::size_t swiglu_workspace_capacity_bytes(GgmlType gate, GgmlType up, std::int32_t rows,
                                            std::int32_t k, std::int32_t first, std::int32_t last);

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

/// Decode, fused K-quant prefill, or K-quant projections sharing activation quantization.
bool ggml_swiglu(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                 cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
