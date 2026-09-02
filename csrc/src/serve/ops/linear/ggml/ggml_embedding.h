#pragma once

#include "core/tensor.h"
#include "ops/linear/ggml/ggml_mmvq.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail::ggml {

/// out[hidden, tokens] = rows ids[t] of a K-quant table [vocab, hidden], dequantised.
void embedding_gather_launch(GgmlType type, const void* table, std::int32_t vocab, std::int32_t hidden,
                             const std::int32_t* ids, std::int32_t tokens, __nv_bfloat16* out,
                             cudaStream_t stream);

/// The ops::embedding branch for a GgmlBlocks table.
void ggml_embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
