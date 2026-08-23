#pragma once

#include "ninfer/ops/kv_cache_append_prefix.h"

namespace ninfer::ops::detail {

struct KVCacheAppendPrefixPlan {
    std::int32_t tokens;
    std::int32_t min_count;
    std::int32_t max_count;
};

[[nodiscard]] KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope);

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& table_rows,
                                   PagedKVBatchLayerView cache, const KVCacheAppendPrefixPlan& plan,
                                   cudaStream_t stream);
void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& lanes,
                                   CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream);

} // namespace ninfer::ops::detail
