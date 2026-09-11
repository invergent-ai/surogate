#include "ops/launcher/kv_cache_append_prefix.h"

#include "core/device.h"
#include "ops/kernel/kv_cache_append_prefix.cuh"

#include <cstdint>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

constexpr int kBlock = 256;

void validate_plan(const Tensor& k, const KVCacheAppendPrefixPlan& plan) {
    if (plan.tokens != k.ne[2] || plan.min_count < 0 || plan.max_count < plan.min_count ||
        plan.max_count > plan.tokens) {
        throw std::invalid_argument("kv_cache_append_prefix: inconsistent plan");
    }
}

template <typename CacheT>
void launch_paged(const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& counts,
                  const Tensor& table_rows, PagedKVBatchLayerView cache,
                  const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    validate_plan(k, plan);
    if (plan.max_count == 0) return;
    auto* cache_k       = static_cast<CacheT*>(cache.k_pages.data);
    auto* cache_v       = static_cast<CacheT*>(cache.v_pages.data);
    const auto* input_k = static_cast<const __nv_bfloat16*>(k.data);
    const auto* input_v = static_cast<const __nv_bfloat16*>(v.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    const auto* count   = static_cast<const std::int32_t*>(counts.data);
    const auto* rows    = static_cast<const std::int32_t*>(table_rows.data);
    const auto* tables  = static_cast<const std::int32_t*>(cache.block_tables.data);

    const dim3 grid(1 + (plan.max_count - 1) / 4, k.ne[3], 1);
    kv_cache_append_prefix_paged_kernel<<<grid, kBlock, 0, stream>>>(
        input_k, input_v, pos, count, rows, cache_k, cache_v, tables, cache.k_pages.ne[2],
        cache.block_tables.ne[0], plan.min_count, plan.max_count, plan.tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CacheT>
void launch_cyclic(const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& counts,
                   const Tensor& lanes, CyclicKVCacheLayerView cache,
                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    validate_plan(k, plan);
    if (plan.max_count == 0) return;
    auto* cache_k       = static_cast<CacheT*>(cache.k.data);
    auto* cache_v       = static_cast<CacheT*>(cache.v.data);
    const auto* input_k = static_cast<const __nv_bfloat16*>(k.data);
    const auto* input_v = static_cast<const __nv_bfloat16*>(v.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    const auto* count   = static_cast<const std::int32_t*>(counts.data);
    const auto* lane    = static_cast<const std::int32_t*>(lanes.data);
    const int padded    = static_cast<int>(cache.padded_capacity);

    const dim3 grid(1 + (plan.max_count - 1) / 4, k.ne[3], 1);
    kv_cache_append_prefix_cyclic_kernel<<<grid, kBlock, 0, stream>>>(
        input_k, input_v, pos, count, lane, cache_k, cache_v, plan.min_count, plan.max_count,
        plan.tokens, padded);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (tokens < 1) {
        throw std::invalid_argument("kv_cache_append_prefix plan: T must be positive");
    }
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument("kv_cache_append_prefix plan: invalid execution envelope");
    }
    return {
        .tokens    = tokens,
        .min_count = static_cast<std::int32_t>(envelope.min_count),
        .max_count = static_cast<std::int32_t>(envelope.max_count),
    };
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& table_rows,
                                   PagedKVBatchLayerView cache, const KVCacheAppendPrefixPlan& plan,
                                   cudaStream_t stream) {
    if (cache.dtype == DType::FP8_E4M3FN) {
        launch_paged<std::uint8_t>(k, v, positions, counts, table_rows, cache, plan, stream);
    } else {
        launch_paged<__nv_bfloat16>(k, v, positions, counts, table_rows, cache, plan, stream);
    }
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& lanes,
                                   CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    if (cache.k.dtype == DType::FP8_E4M3FN) {
        launch_cyclic<std::uint8_t>(k, v, positions, counts, lanes, cache, plan, stream);
    } else {
        launch_cyclic<__nv_bfloat16>(k, v, positions, counts, lanes, cache, plan, stream);
    }
}

} // namespace sinfer::ops::detail
