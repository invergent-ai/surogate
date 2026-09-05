#pragma once

// sinfer::ops::detail - private launch prototypes for gqa_attention policies.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "api/ops/gqa_attention.h"

#include <cuda_runtime.h>

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

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream, GqaBlockMask selection = {});

// `sliding_window` is deliberately required and deliberately ahead of the defaulted
// `selection`: a window that defaults to 0 reads as "unbounded" but means "the caller
// forgot", and the two are indistinguishable at the call site. Making it positional
// turns an unforwarded window into a compile error instead of silent wrong attention.
void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream, std::int32_t sliding_window,
                                 GqaBlockMask selection = {});

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream);

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream, std::int32_t sliding_window,
                                           GqaBlockMask selection = {});

} // namespace sinfer::ops::detail
