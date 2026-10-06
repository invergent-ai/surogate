#pragma once

// sinfer::ops::detail - private launch prototypes for gqa_attention policies.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "api/ops/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

enum class GqaAttentionRoute { SmallT, ChunkedSmallT, Prompt };

struct GqaSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    GqaBlockMask selection{}; // QSA sparse selection; a null `words` is the dense path
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
    /// Causal sliding window, zero for unbounded. See GqaExecutionEnvelope.
    std::int32_t sliding_window = 0;
    /// Batched lanes only: each lane's first column in q, pos and out (I32, batch_size), in place
    /// of column_begin + lane * full_width -- lanes of several sequences at their own columns.
    const Tensor* lane_columns  = nullptr;
};

// Splits the launcher will use for this shape, which sizes the partial buffers
// it writes. The head counts must be the served shape's: they select the same
// registered geometry the launcher selects.
std::int32_t gqa_attention_split_capacity(std::int32_t head_dim, std::int32_t q_heads,
                                          std::int32_t kv_heads,
                                          std::int32_t tokens, DType cache_dtype,
                                          GqaExecutionEnvelope envelope);

bool gqa_attention_uses_small_t(std::int32_t tokens);

// Widest small-T step this head geometry serves: the lane step packs
// width x group query rows and holds at most 64, so a group of twelve
// (24 query heads over 2 KV heads) steps 5 tokens, the others 6.
std::int32_t gqa_attention_small_t_max_width(std::int32_t q_heads, std::int32_t kv_heads);

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t kv_heads,
                                              std::int32_t width, std::int32_t batch_size,
                                              GqaExecutionEnvelope envelope);

const char* gqa_attention_route_name(GqaAttentionRoute route);

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale,
                                  PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                  std::int32_t column_begin, std::int32_t width,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream,
                                  GqaBlockMask selection = {});

// Prompt tiles of several sequences in one launch: lane i covers `width` columns of q, positions
// and out from lane_columns[i] and reads table row table_rows[i]. BF16/FP8 caches only; the
// queries' keys must already be in the cache.
void gqa_attention_cached_lanes_small_t_launch(const Tensor& q, const Tensor& positions,
                                               const Tensor& valid_columns,
                                               const Tensor& table_rows,
                                               const Tensor& lane_columns, float scale,
                                               PagedKVBatchLayerView cache,
                                               GqaExecutionEnvelope envelope, std::int32_t width,
                                               Tensor& partial_acc, Tensor& partial_m,
                                               Tensor& partial_l, Tensor& out,
                                               cudaStream_t stream);

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream, GqaBlockMask selection = {});

// `sliding_window` is deliberately required and deliberately ahead of the defaulted
// `selection`: a window that defaults to 0 reads as "unbounded" but means "the caller
// forgot", and the two are indistinguishable at the call site. Making it positional
// turns an unforwarded window into a compile error instead of silent wrong attention.
void gqa_attention_cached_batch_small_t_launch(const Tensor& q, const Tensor& pos,
                                               const Tensor& valid_columns,
                                               const Tensor& table_rows, float scale,
                                               PagedKVBatchLayerView cache,
                                               GqaExecutionEnvelope envelope,
                                               std::int32_t column_begin, std::int32_t width,
                                               Tensor& partial_acc, Tensor& partial_m,
                                               Tensor& partial_l, Tensor& out,
                                               cudaStream_t stream, GqaBlockMask selection = {});

void gqa_attention_prompt_cached_launch(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        float scale, PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream, std::int32_t sliding_window,
                                        GqaBlockMask selection = {});

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream, std::int32_t sliding_window,
                                 GqaBlockMask selection = {});

// Prepare query tiles that share one sequence's paged history without host metadata reads.
void gqa_query_tile_metadata(const Tensor& parent_valid, const Tensor& parent_row,
                             int parent_width, int begin, int tile_width,
                             Tensor& valid, Tensor& rows, cudaStream_t stream);

void gqa_kv_append_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                const Tensor& valid_columns, const Tensor& table_rows,
                                PagedKVBatchLayerView cache, cudaStream_t stream);

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream);

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream, std::int32_t sliding_window,
                                           GqaBlockMask selection = {});

// The route for shapes with no tuned kernel (gqa_attention_generic.cu): any query group over a
// registered (head dim, KV heads) pair, every batch row and column in one launch, each query's
// history split across CTAs while the columns alone leave the device idle. Splits depend on the
// query-head and column counts and the device, never on the history; one under --batch-invariant.
[[nodiscard]] int gqa_generic_attention_splits(std::int32_t q_heads, std::int32_t columns,
                                               std::uint32_t max_visible_keys);
[[nodiscard]] std::size_t gqa_generic_attention_workspace_bytes(std::int32_t head_dim,
                                                                std::int32_t q_heads,
                                                                std::int32_t columns,
                                                                std::int32_t splits);
[[nodiscard]] bool gqa_generic_attention_serves(std::int32_t head_dim, DType cache_dtype);

void gqa_generic_attention_launch(const Tensor& q, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  float scale, const PagedKVBatchLayerView& cache,
                                  std::int32_t sliding_window, std::int32_t splits,
                                  DeviceSpan workspace, Tensor& out, cudaStream_t stream);
void gqa_generic_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& cache, std::int32_t sliding_window,
                                  std::int32_t splits, DeviceSpan workspace, Tensor& out,
                                  cudaStream_t stream);
void gqa_generic_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  const PagedKVBatchLayerView& cache, cudaStream_t stream);

} // namespace sinfer::ops::detail
