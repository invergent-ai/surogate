#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

namespace sinfer::ops {

// GLM's NoPE pooled indexer: affine LayerNorm keys, independent per-channel
// pooling gates plus positional bias, and signed weights of rectified head scores.
struct GlmIndexerGeometry {
    std::int32_t head_dim = 0;
    std::int32_t heads = 0;
    std::int32_t block = 0;
    std::int32_t top_k = 0;
    float norm_epsilon = 1e-6F;
};

// FP32 weight [K,N], BF16 activation [K,T], FP32 output [N,T]. Dimension zero
// is contiguous. Keeps selection controls in FP32 and accepts every positive T.
void glm_indexer_project(const Tensor& weight, const Tensor& x, Tensor& out, cudaStream_t stream);

// Cache plane is FP32 [3*D,page_tokens,1,pages]: normalized keys, raw pooling
// gates, and pooled keys. Raw components remain intact when a pool completes,
// allowing truncation and replay of an incomplete pool without a special undo.
// Positions are contiguous within each sequence; columns_per_row partitions T
// columns by table_rows. Optional valid_columns [sequences] suppresses padding.
void glm_indexer_append(const Tensor& keys, const Tensor& gates, const Tensor& norm,
    const Tensor& bias, const Tensor& ape, const Tensor& positions, const Tensor& table_rows,
    const Tensor& valid_columns, std::int32_t columns_per_row, const GlmIndexerGeometry& geometry,
    PagedKVBatchLayerView cache, cudaStream_t stream);

// FP32 queries [D,H,T], signed head weights [H,T]. Select top_k/block complete
// pools, breaking exact ties by ascending pool index; always retain the current
// incomplete tail. Per-query causal masks are I32 [ceil(ceil(keys/block)/32),T].
// Scores use bounded, tiled scratch rather than allocating T*keys for long prefills.
void glm_indexer_select(const Tensor& queries, const Tensor& head_weights,
    const Tensor& positions, const Tensor& table_rows, const Tensor& valid_columns,
    std::int32_t columns_per_row, const GlmIndexerGeometry& geometry, PagedKVBatchLayerView cache,
    std::int32_t keys, WorkspaceArena& workspace, Tensor& mask, cudaStream_t stream);
std::size_t glm_indexer_select_workspace_capacity_bytes(std::int32_t rows, std::int32_t keys,
    const GlmIndexerGeometry& geometry);

} // namespace sinfer::ops
