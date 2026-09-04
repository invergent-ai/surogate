#pragma once

#include "ops/linear/ggml/ggml_blocks.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail::ggml {

enum class GgmlType : std::uint8_t { Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_1, Q5_1 };

/// Values per stored block: 256 for every K-quant, 32 for Q8_0, which has no superblock.
__host__ __device__ constexpr std::int32_t block_values(GgmlType type) noexcept {
    return (type == GgmlType::Q8_0 || type == GgmlType::Q4_1 || type == GgmlType::Q5_1)
               ? QK8_0
               : QK_K;
}

__host__ __device__ constexpr std::int32_t block_bytes(GgmlType type) noexcept {
    switch (type) {
    case GgmlType::Q2_K: return sizeof(block_q2_K);
    case GgmlType::Q3_K: return sizeof(block_q3_K);
    case GgmlType::Q4_K: return sizeof(block_q4_K);
    case GgmlType::Q5_K: return sizeof(block_q5_K);
    case GgmlType::Q6_K: return sizeof(block_q6_K);
    case GgmlType::Q8_0: return sizeof(block_q8_0);
    case GgmlType::Q4_1: return sizeof(block_q4_1);
    case GgmlType::Q5_1: return sizeof(block_q5_1);
    }
    return 0;
}
constexpr const char* type_name(GgmlType type) noexcept {
    switch (type) {
    case GgmlType::Q2_K: return "Q2_K";
    case GgmlType::Q3_K: return "Q3_K";
    case GgmlType::Q4_K: return "Q4_K";
    case GgmlType::Q5_K: return "Q5_K";
    case GgmlType::Q6_K: return "Q6_K";
    case GgmlType::Q8_0: return "Q8_0";
    case GgmlType::Q4_1: return "Q4_1";
    case GgmlType::Q5_1: return "Q5_1";
    }
    return "?";
}

/// The widest column count one mmvq launch handles; wider problems are chunked by the caller.
inline constexpr std::int32_t kMmvqMaxColumns = 8;

/// out[n, tokens] = W[n, k] · x (or += with Accumulate) for tokens <= 8 columns, W in native GGML superblocks
/// (`blocks` = n rows × k/256 blocks, verbatim file bytes) and x already quantised to
/// block_q8_1 [tokens][k/32]. `DstT` is float (tests, fp32 consumers) or __nv_bfloat16.
template <typename DstT, bool Accumulate = false>
void mmvq_launch(GgmlType type, const void* blocks, std::int32_t n, std::int32_t k,
                 const block_q8_1* y, std::int32_t tokens, DstT* out, cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
