#pragma once

#include "ops/linear/ggml/ggml_blocks.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::ggml {

/// Bytes of block_q8_1 storage for `tokens` activations of `k` values each (k % 32 == 0).
inline std::size_t q8_1_bytes(std::int32_t k, std::int32_t tokens) noexcept {
    return static_cast<std::size_t>(tokens) * static_cast<std::size_t>(k / QK8_1) * sizeof(block_q8_1);
}

/// Quantises x [k, tokens] (k contiguous, BF16) to block_q8_1 [tokens][k/32]: per 32 values
/// d = amax/127, q = round(x/d), and (d, sum of the 32 unquantised values) stored as half2.
/// This is the activation side every K-quant vec-dot consumes.
void quantize_q8_1_launch(const __nv_bfloat16* x, std::int32_t k, std::int32_t tokens,
                          block_q8_1* out, cudaStream_t stream);


/// The same numbers as `quantize_q8_1_launch`, written as two planes instead of interleaved
/// blocks: `codes` is [tokens][k] int8 and `ds` is [tokens][k/32] half2 of (d, sum). A tiled
/// GEMM stages its activation operand with sixteen-byte asynchronous copies, which a 36-byte
/// block cannot be the source of; a codes plane whose rows are whole multiples of sixteen can.
void quantize_q8_1_planes_launch(const __nv_bfloat16* x, std::int32_t k, std::int32_t tokens,
                                 std::int8_t* codes, __half2* ds, cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
