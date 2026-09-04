// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/ggml/ggml_dequant.cuh"
#include "ops/linear/ggml/ggml_embedding.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace sinfer::ops::detail::ggml {
namespace {

// One CTA per (token, superblock) expands one 256-value block of the row the token id
// selects, straight into that token's column of out [hidden, tokens]. The dequantisers keep
// llama.cpp's thread layout: 32 threads for Q4_K, 64 for the other four (convert.cu launches
// them so, and each thread writes its fixed share of the block).
template <GgmlType type>
__global__ void gather_kernel(const void* __restrict__ table, const std::int32_t* __restrict__ ids,
                              __nv_bfloat16* __restrict__ out, const int vocab, const int hidden) {
    const int token = blockIdx.x;
    const int block = blockIdx.y; // superblock along hidden
    const int id    = ids[token];
    __nv_bfloat16* column =
        out + static_cast<std::size_t>(token) * hidden + block * block_values(type);
    if (id < 0 || id >= vocab) {
        // Out-of-range ids read as zero rather than as memory past the table.
        for (int i = threadIdx.x; i < block_values(type); i += dequant_threads<type>()) {
            column[i] = __float2bfloat16(0.0f);
        }
        return;
    }
    const std::int64_t ib =
        static_cast<std::int64_t>(id) * (hidden / block_values(type)) + block;
    dequantize_superblock<type>(table, ib, column, threadIdx.x);
}

template <GgmlType type>
void launch(const void* table, const std::int32_t* ids, std::int32_t tokens, std::int32_t vocab,
            std::int32_t hidden, __nv_bfloat16* out, cudaStream_t stream) {
    const dim3 grid(tokens, hidden / block_values(type));
    gather_kernel<type><<<grid, dequant_threads<type>(), 0, stream>>>(table, ids, out, vocab, hidden);
}

} // namespace

void embedding_gather_launch(GgmlType type, const void* table, std::int32_t vocab, std::int32_t hidden,
                             const std::int32_t* ids, std::int32_t tokens, __nv_bfloat16* out,
                             cudaStream_t stream) {
    if (table == nullptr || ids == nullptr || out == nullptr || vocab <= 0 || hidden <= 0 ||
        (hidden % block_values(type)) != 0 || tokens <= 0) {
        throw std::invalid_argument("ggml embedding: table [vocab, hidden] with hidden a multiple of 256");
    }
    switch (type) {
#define SINFER_EMBED_CASE(NAME)                                                                    \
    case GgmlType::NAME: launch<GgmlType::NAME>(table, ids, tokens, vocab, hidden, out, stream); return;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_EMBED_CASE)
#undef SINFER_EMBED_CASE
    }
    throw std::invalid_argument("ggml embedding: unknown GGML type");
}

void ggml_embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    require_ggml_weight(table, "ggml embedding");
    if (ids.dtype != DType::I32 || out.dtype != DType::BF16 || !ids.is_contiguous() ||
        !out.is_contiguous() || ids.ne[0] <= 0 || ids.ne[1] != 1 || out.ne[0] != table.k ||
        out.ne[1] != ids.ne[0] || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("ggml embedding: ids I32 [T], out BF16 [hidden, T]");
    }
    embedding_gather_launch(ggml_type_for(table.qtype), table.qdata, table.n, table.k,
                            static_cast<const std::int32_t*>(ids.data), ids.ne[0],
                            static_cast<__nv_bfloat16*>(out.data), stream);
}

} // namespace sinfer::ops::detail::ggml
