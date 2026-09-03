#pragma once

#include "ops/linear/ggml/ggml_mmvq.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::ggml {

/// Routed-expert GEMV, llama.cpp's `mul_mat_vec_q` with its id indirection (`mul_mat_id`).
///
/// `blocks` is one expert's weight after another: [experts][rows][k/256] superblocks, so expert
/// e begins at `e * rows * (k/256) * block_bytes(type)`. `ids` maps a (token, slot) pair to the
/// expert that token routed to that slot: `ids[slot + token * ids_stride]`. Each launch covers
/// every token and slot, writing `out[(token * slots + slot) * rows + row]`.
///
/// `y` holds the tokens' activations quantised to int8 per 32 (`quantize_q8_1_launch`), token
/// t at `y + t * (k / 32)`.
void moe_gemv_launch(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                     const block_q8_1* y, const std::int32_t* ids, std::int32_t tokens,
                     std::int32_t slots, std::int32_t ids_stride, float* out, cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
