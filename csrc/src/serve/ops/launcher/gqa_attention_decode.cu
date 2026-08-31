// sinfer::ops - split-KV GQA small-T launcher and unified route dispatcher.
//
// The launcher template itself lives in gqa_attention_decode_launch.cuh and is
// instantiated one geometry per translation unit (gqa_attention_decode_*.cu).
// The `extern template` declarations below keep this file from instantiating them
// a second time, which is what made it the build's critical path.
#include "ops/launcher/gqa_attention_decode_launch.cuh"

namespace sinfer::ops::detail {

extern template void gqa_attention_small_t_launch_for<Gqa256_24q2, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
extern template void gqa_attention_small_t_launch_for<Gqa256_24q2, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
extern template void gqa_attention_small_t_launch_for<Gqa27Geometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
extern template void gqa_attention_small_t_launch_for<Gqa27Geometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
extern template void gqa_attention_small_t_launch_for<Gqa08Geometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
extern template void gqa_attention_small_t_launch_for<Gqa08Geometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
extern template void gqa_attention_small_t_launch_for<Gqa4BGeometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
extern template void gqa_attention_small_t_launch_for<Gqa4BGeometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
extern template void gqa_attention_small_t_launch_for<Gqa35Geometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
extern template void gqa_attention_small_t_launch_for<Gqa35Geometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);


bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t gqa_attention_small_t_max_width(std::int32_t q_heads, std::int32_t kv_heads) {
    if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("gqa_attention small-T width: unsupported head geometry");
    }
    const std::int32_t group = q_heads / kv_heads;
    const std::int32_t rows  = 64 / group;
    return rows < 6 ? rows : 6;
}

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 6 || (cache_dtype != DType::BF16 && cache_dtype != DType::I8 &&
         cache_dtype != DType::FP8_E4M3FN) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("gqa_attention split capacity: invalid profile");
    }
    if (q_heads == Gqa27Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa27Geometry>(envelope, tokens, cache_dtype);
    }
    if (q_heads == Gqa35Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa35Geometry>(envelope, tokens, cache_dtype);
    }
    if (q_heads == Gqa08Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa08Geometry>(envelope, tokens, cache_dtype);
    }
    throw std::invalid_argument("gqa_attention split capacity: unsupported head geometry");
}

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale,
                                  PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                  std::int32_t column_begin, std::int32_t width,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream, GqaBlockMask selection) {
    const GqaAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                               static_cast<const __nv_bfloat16*>(v.data)};
    const GqaSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .selection     = selection,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (q.ne[1] == Gqa256_24q2::QHeads && cache.num_kv_heads == Gqa256_24q2::KVHeads) {
        gqa_attention_small_t_launch_for<Gqa256_24q2>(q, input, pos, scale, cache, invocation,
                                                      envelope, partial_acc, partial_m, partial_l,
                                                      out, stream);
        return;
    }
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m, partial_l,
                                                        out, stream);
        return;
    }
    if (q.ne[1] == Gqa08Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa08Geometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m, partial_l,
                                                        out, stream);
        return;
    }
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b shares QHeads 16 with
    // the 35B; the cache resolves the pair.
    if (cache.num_kv_heads == Gqa4BGeometry::KVHeads) {
        gqa_attention_small_t_launch_for<Gqa4BGeometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m,
                                                        partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
}

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream, GqaBlockMask selection) {
    const GqaCachedInput input{};
    const GqaSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .selection     = selection,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    const PagedKVBatchLayerView batch_cache = single_row_batch_view(cache);
    if (q.ne[1] == Gqa256_24q2::QHeads && batch_cache.num_kv_heads == Gqa256_24q2::KVHeads) {
        gqa_attention_small_t_launch_for<Gqa256_24q2>(q, input, pos, scale, batch_cache,
                                                      invocation, envelope, partial_acc,
                                                      partial_m, partial_l, out, stream);
        return;
    }
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, batch_cache,
                                                        invocation, envelope, partial_acc,
                                                        partial_m, partial_l, out, stream);
        return;
    }
    if (q.ne[1] == Gqa08Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa08Geometry>(q, input, pos, scale, batch_cache,
                                                        invocation, envelope, partial_acc,
                                                        partial_m, partial_l, out, stream);
        return;
    }
    // surogate vendor patch (PATCHES.md #18): see above — the cache resolves
    // the 16-query pair.
    if (batch_cache.num_kv_heads == Gqa4BGeometry::KVHeads) {
        gqa_attention_small_t_launch_for<Gqa4BGeometry>(q, input, pos, scale, batch_cache,
                                                        invocation, envelope, partial_acc,
                                                        partial_m, partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, batch_cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
}

} // namespace sinfer::ops::detail
