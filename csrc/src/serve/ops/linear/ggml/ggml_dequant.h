#pragma once

#include "ops/linear/ggml/ggml_mmvq.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail::ggml {

/// Expands `rows` consecutive rows of a K-quant weight into BF16 [rows, k], k contiguous --
/// the layout the BF16 cuBLASLt route reads a weight in. `blocks` points at the first row's
/// superblocks; k must be a multiple of 256.
void dequantize_rows_launch(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                            __nv_bfloat16* out, cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
