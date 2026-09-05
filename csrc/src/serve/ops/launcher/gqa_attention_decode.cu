// sinfer::ops - split-KV GQA small-T launcher and unified route dispatcher.
//
// The launcher template itself lives in gqa_attention_decode_launch.cuh and is
// instantiated one geometry per translation unit (gqa_attention_decode_*.cu).
// The `extern template` declarations below keep this file from instantiating them
// a second time, which is what made it the build's critical path. They are
// generated from the geometry registry, so a shape registered without its
// translation unit fails at link rather than silently landing on another shape.
#include "ops/launcher/gqa_attention_decode_launch.cuh"

namespace sinfer::ops::detail {

#define SINFER_GQA_EXTERN_DECODE(Name)                                                             \
    extern template void gqa_attention_small_t_launch_for<Name, GqaAppendInput>(                   \
        const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,                \
        const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,      \
        cudaStream_t);                                                                             \
    extern template void gqa_attention_small_t_launch_for<Name, GqaCachedInput>(                   \
        const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,                \
        const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,      \
        cudaStream_t);
SINFER_GQA_FOR_EACH_GEOMETRY(SINFER_GQA_EXTERN_DECODE)
#undef SINFER_GQA_EXTERN_DECODE

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t gqa_attention_small_t_max_width(std::int32_t q_heads, std::int32_t kv_heads) {
    if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("gqa_attention small-T width: unsupported head geometry");
    }
    const std::int32_t group = q_heads / kv_heads;
    const std::int32_t rows  = 64 / group;
    return rows < 6 ? rows : 6;
}

std::int32_t gqa_attention_split_capacity(std::int32_t head_dim, std::int32_t q_heads,
                                          std::int32_t kv_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 6 || (cache_dtype != DType::BF16 && cache_dtype != DType::I8 &&
         cache_dtype != DType::FP8_E4M3FN) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("gqa_attention split capacity: invalid profile");
    }
    // Sizes the split buffers the launcher writes, so it has to resolve the same
    // geometry the launcher will: the split count follows DecodeSplitScale, and
    // two shapes sharing a query count need not share it.
    return gqa_dispatch_geometry(head_dim, q_heads, kv_heads, [&]<typename Geometry>() {
        return gqa_small_t_launch_capacity<Geometry>(envelope, tokens, cache_dtype);
    });
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
        .sliding_window = envelope.sliding_window,
    };
    gqa_dispatch_geometry(q.ne[0], q.ne[1], cache.num_kv_heads, [&]<typename Geometry>() {
        gqa_attention_small_t_launch_for<Geometry>(q, input, pos, scale, cache, invocation,
                                                   envelope, partial_acc, partial_m, partial_l,
                                                   out, stream);
    });
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
        .sliding_window = envelope.sliding_window,
    };
    const PagedKVBatchLayerView batch_cache = single_row_batch_view(cache);
    gqa_dispatch_geometry(q.ne[0], q.ne[1], batch_cache.num_kv_heads, [&]<typename Geometry>() {
        gqa_attention_small_t_launch_for<Geometry>(q, input, pos, scale, batch_cache, invocation,
                                                   envelope, partial_acc, partial_m, partial_l,
                                                   out, stream);
    });
}

} // namespace sinfer::ops::detail
