#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/linear/ggml/ggml_mmvq.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::ggml {

bool is_ggml_qtype(QType qtype) noexcept;
GgmlType ggml_type_for(QType qtype);

/// Throws unless `w` is a GGML weight in QuantLayout::GgmlBlocks with complete
/// blocks along each row and positive dimensions.
void require_ggml_weight(const Weight& w, const char* op);

/// Scratch for the int8 activation planes when the caller has no arena to offer: an
/// engine-slot buffer grown on first use, which happens during the pre-capture warm-up (the
/// same window the derived quant planes allocate in). Never grows inside a capture; a call
/// that would need to throws instead.
void* scratch_for(std::size_t bytes, cudaStream_t stream);
/// Bytes the engine-slot scratch currently holds, for the graph allowance to exclude.
std::size_t scratch_bytes() noexcept;

/// y = W · x for a GGML weight. `workspace` may be null (see scratch_for).
void ggml_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena* workspace,
                 cudaStream_t stream);
/// residual += W · x.
void ggml_linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena* workspace,
                     cudaStream_t stream);
/// The row range as a GGML weight of its own: its segment's format when the parent mixes
/// formats, its own otherwise. Refuses a range that straddles two segments.
Weight ggml_weight_rows(const Weight& w, std::int32_t row_begin, std::int32_t rows);

/// out[rows, T] = W[row_begin : row_begin + rows, :] · x — a row range of the parent, straight
/// into the caller's tensor. What the fused projections split their parents with.
void ggml_project_rows(const Tensor& x, const Weight& w, std::int32_t row_begin, Tensor& out,
                       WorkspaceArena* workspace, cudaStream_t stream);

std::size_t ggml_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                std::int32_t input_rows,
                                                std::int32_t max_tokens);

} // namespace sinfer::ops::detail::ggml
