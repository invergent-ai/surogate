// sinfer::ops - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/launcher/gqa_attention.h"
#include "ops/kernel/func_attribute.cuh"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstddef>
#include <type_traits>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail {
namespace {

// A query-count-independent fallback over the same paged cache representation.
// FP32 online softmax avoids storing a score matrix or allocating scratch memory.
template <typename CacheT>
__device__ float generic_cache_value(const CacheT* data, const __half* scales,
                                     std::int64_t offset, std::int64_t scale_offset) {
    if constexpr (std::is_same_v<CacheT, __nv_bfloat16>) {
        return __bfloat162float(data[offset]);
    } else if constexpr (std::is_same_v<CacheT, std::uint8_t>) {
        __nv_fp8_e4m3 value;
        value.__x = data[offset];
        return static_cast<float>(value);
    } else {
        return static_cast<float>(data[offset]) * __half2float(scales[scale_offset]);
    }
}

template <typename CacheT, typename Metadata>
__global__ void gqa_attention_generic_kernel(
    const __nv_bfloat16* q, const CacheT* keys, const CacheT* values,
    const __half* key_scales, const __half* value_scales, const std::int32_t* positions,
    Metadata metadata, int dim, int queries, int kv_heads, int width, float scale,
    __nv_bfloat16* out, GqaBlockMask selection) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    const int lane = threadIdx.x;
    const std::int64_t output_base = (static_cast<std::int64_t>(token) * queries + head) * dim;
    if (token >= metadata.valid_tokens(width)) {
        for (int d = lane; d < dim; d += 32) { out[output_base + d] = __float2bfloat16(0.0F); }
        return;
    }
    const int kv_head = head / (queries / kv_heads);
    const int last = positions[0] + token;
    const int first = metadata.window > 0 ? max(0, last + 1 - metadata.window) : 0;
    const auto* table = metadata.block_table();
    float result[16] = {};
    float maximum = -CUDART_INF_F, denominator = 0.0F;
    for (int key = first; key <= selection.last_key(last); ++key) {
        if (selection.words != nullptr) {
            const int block = key / selection.block;
            if (((selection.words[static_cast<std::int64_t>(token) * selection.stride +
                                   (block >> 5)] >> (block & 31)) & 1U) == 0) { continue; }
        }
        const auto cell = (static_cast<std::int64_t>(table[key / kPagedKVPageSize]) * kv_heads +
                           kv_head) * kPagedKVPageSize + key % kPagedKVPageSize;
        float dot = 0.0F;
        for (int d = lane; d < dim; d += 32) {
            const float value = generic_cache_value(keys, key_scales, cell * dim + d,
                                                     cell * (dim / 64) + d / 64);
            dot = fmaf(__bfloat162float(q[output_base + d]), value, dot);
        }
        for (int delta = 16; delta > 0; delta /= 2) {
            dot += __shfl_xor_sync(0xffffffffU, dot, delta);
        }
        const float score = dot * scale;
        const float next = fmaxf(maximum, score);
        const float previous_weight = expf(maximum - next);
        const float weight = expf(score - next);
        denominator = denominator * previous_weight + weight;
        for (int d = lane; d < dim; d += 32) {
            const float value = generic_cache_value(values, value_scales, cell * dim + d,
                                                     cell * (dim / 64) + d / 64);
            result[d / 32] = result[d / 32] * previous_weight + weight * value;
        }
        maximum = next;
    }
    for (int d = lane; d < dim; d += 32) {
        out[output_base + d] = __float2bfloat16(denominator > 0 ? result[d / 32] / denominator : 0);
    }
}

template <typename Visitor>
void dispatch_prompt_geometry(int dim, int queries, int kv, Visitor&& visitor) {
    if (gqa_shape_is_registered(dim, queries, kv)) {
        gqa_dispatch_geometry(dim, queries, kv, visitor);
    } else {
        gqa_dispatch_kv_geometry(dim, kv, visitor);
    }
}

Tensor row_tensor(const Tensor& tensor, int row, int elements) {
    if (tensor.data == nullptr) { return tensor; }
    return Tensor(static_cast<std::byte*>(tensor.data) +
                      static_cast<std::size_t>(row) * elements * dtype_size(tensor.dtype),
                  tensor.dtype, {elements});
}
Tensor batch_tensor(const Tensor& tensor, int batch) {
    return Tensor(static_cast<std::byte*>(tensor.data) +
                      static_cast<std::size_t>(batch) * tensor.nb[3], tensor.dtype,
                  {tensor.ne[0], tensor.ne[1], tensor.ne[2]});
}

// A key's int8 group scales are copied as a single cp.async, so they must be a
// legal transfer width. With kGqaKvQuantGroup at 64 that means a head carrying at
// least two groups -- head dim 128 or wider. A 64-wide head carries one group,
// two bytes, which no cp.async width covers, so this kernel cannot be compiled
// for it at all. Shapes outside the set refuse at the dispatch rather than
// blocking the build; their bf16 and e4m3 caches are unaffected.
template <typename Geometry>
inline constexpr bool kGqaI8PrefillRegistered =
    // Its arena is sized for a head of at most 256; at 512 it is past what the card opts
    // in to, so a wider head refuses an INT8 cache by name rather than failing at load.
    Geometry::HeadDim <= 256 &&
    (Geometry::HeadDim / kGqaKvQuantGroup) * static_cast<int>(sizeof(__half)) >= 4;

template <typename Geometry>
void refuse_i8_prefill() {
    throw std::invalid_argument("gqa_attention prefill: int8 KV is not served for head dim " +
                                std::to_string(Geometry::HeadDim) + " (its group scales are " +
                                std::to_string((Geometry::HeadDim / kGqaKvQuantGroup) *
                                               static_cast<int>(sizeof(__half))) +
                                " bytes, which is not a cp.async transfer width)");
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                               float scale, const CacheView& cache,
                                               Metadata metadata, Tensor& out,
                                               cudaStream_t stream,
                                               GqaBlockMask selection = {}) {
    if (q.ne[1] != Geometry::QHeads) {
        if (selection.words && cache.dtype == DType::I8) {
            throw std::invalid_argument(
                "gqa_attention: the QSA selection is not served over an int8 KV cache");
        }
        const auto launch = [&]<typename CacheT>() {
            gqa_attention_generic_kernel<CacheT, Metadata><<<dim3(q.ne[1], q.ne[2]), 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const CacheT*>(cache.k_pages.data),
                static_cast<const CacheT*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                q.ne[0], q.ne[1], cache.num_kv_heads, q.ne[2], scale,
                static_cast<__nv_bfloat16*>(out.data), selection);
        };
        if (cache.dtype == DType::BF16) { launch.template operator()<__nv_bfloat16>(); }
        else if (cache.dtype == DType::FP8_E4M3FN) { launch.template operator()<std::uint8_t>(); }
        else { launch.template operator()<std::int8_t>(); }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // A selection rides on the cache dtype rather than replacing it: the BF16 and e4m3 prefill
    // kernels are one template over the cache type, and the block mask is orthogonal to it. The
    // int8 kernel is a different one and has no sparse specialization, so it says so by name.
    if (selection.words != nullptr && cache.dtype == DType::I8) {
        throw std::invalid_argument(
            "gqa_attention: the QSA selection is not served over an int8 KV cache");
    }
    if (selection.words != nullptr && selection.block != 4) {
        throw std::invalid_argument("gqa_attention: unregistered QSA block size");
    }
    // Both dtype-specialized kernels size their arena from the geometry's head
    // dimension; at 256 both exceed the default 48 KiB dynamic-smem ceiling, and
    // raising the limit for a kernel that would fit anyway is harmless.
    constexpr int kSmemBytes   = kGqaPrefillSmemBytes<Geometry::HeadDim>;
    constexpr int kI8SmemBytes = kGqaPrefillI8SmemBytes<Geometry::HeadDim>;
    CUDA_CHECK(::sinfer::ops::set_func_attribute_per_device(gqa_attention_prefill_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    if constexpr (kGqaI8PrefillRegistered<Geometry>) {
        CUDA_CHECK(::sinfer::ops::set_func_attribute_per_device(
            gqa_attention_prefill_i8_kernel<Geometry, Metadata>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kI8SmemBytes));
    }

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (cache.dtype == DType::I8) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        if constexpr (!kGqaI8PrefillRegistered<Geometry>) {
            refuse_i8_prefill<Geometry>();
        } else
        gqa_attention_prefill_i8_kernel<Geometry, Metadata>
            <<<attention_grid, kGqaPrefillI8Threads, kI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens, selection);
    } else if (cache.dtype == DType::FP8_E4M3FN) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        if (selection.words != nullptr) {
            CUDA_CHECK(::sinfer::ops::set_func_attribute_per_device(
                gqa_attention_prefill_bf16_kernel<Geometry, Metadata, std::uint8_t, true, 4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
            gqa_attention_prefill_bf16_kernel<Geometry, Metadata, std::uint8_t, true, 4>
                <<<attention_grid, kGqaPrefillThreads, kSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::uint8_t*>(cache_k.data),
                    static_cast<const std::uint8_t*>(cache_v.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens, selection);
        } else {
            CUDA_CHECK(::sinfer::ops::set_func_attribute_per_device(
                gqa_attention_prefill_bf16_kernel<Geometry, Metadata, std::uint8_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
            gqa_attention_prefill_bf16_kernel<Geometry, Metadata, std::uint8_t>
                <<<attention_grid, kGqaPrefillThreads, kSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::uint8_t*>(cache_k.data),
                    static_cast<const std::uint8_t*>(cache_v.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens, selection);
        }
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        if (selection.words != nullptr) {
            CUDA_CHECK(::sinfer::ops::set_func_attribute_per_device(
                gqa_attention_prefill_bf16_kernel<Geometry, Metadata, __nv_bfloat16, true, 4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
            gqa_attention_prefill_bf16_kernel<Geometry, Metadata, __nv_bfloat16, true, 4>
                <<<attention_grid, kGqaPrefillThreads, kSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(cache_k.data),
                    static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens, selection);
        } else {
            gqa_attention_prefill_bf16_kernel<Geometry, Metadata>
                <<<attention_grid, kGqaPrefillThreads, kSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(cache_k.data),
                    static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens, selection);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_kv_append_launch_for(const Tensor& k, const Tensor& v, const Tensor& positions,
                              CacheView cache, Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    if (cache.dtype == DType::I8) {
        Tensor& cache_k_scale    = cache.k_scale_pages;
        Tensor& cache_v_scale    = cache.v_scale_pages;
        constexpr int kFillBlock = 256;
        if (tokens >= 128 && Geometry::KVHeads == 2) {
            constexpr int kPageBlock     = 256;
            constexpr int kTokensPerTile = 8;
            const int max_tiles          = div_up(tokens + kTokensPerTile - 1, kTokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads),
                                 static_cast<unsigned>(kGqaKvQuantGroups<Geometry>));
            if constexpr (!kGqaI8PrefillRegistered<Geometry>) {
                refuse_i8_prefill<Geometry>();
            } else
            gqa_attention_prefill_fill_i8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kPageBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int kFillWarps = kFillBlock / 32;
            const std::int64_t fill_units =
                static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                kGqaKvQuantGroups<Geometry>;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
            if constexpr (!kGqaI8PrefillRegistered<Geometry>) {
                refuse_i8_prefill<Geometry>();
            } else
            gqa_attention_prefill_fill_i8_kernel<Geometry, Metadata>
                <<<fill_grid, kFillBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        constexpr int kBlock           = Geometry::KVHeads == 4 ? 128 : 96;
        constexpr int kFillVecElems    = 8;
        const std::int64_t kv_elements = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                         (Geometry::HeadDim / kFillVecElems);
        const int fill_grid =
            static_cast<int>(div_up(kv_elements, static_cast<std::int64_t>(kBlock)));
        if (cache.dtype == DType::FP8_E4M3FN) {
            gqa_attention_prefill_fill_bf16_kernel<Geometry, Metadata, std::uint8_t>
                <<<fill_grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data), tokens);
        } else {
            gqa_attention_prefill_fill_bf16_kernel<Geometry, Metadata>
                <<<fill_grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<__nv_bfloat16*>(cache_k.data),
                    static_cast<__nv_bfloat16*>(cache_v.data), tokens);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream, std::int32_t sliding_window,
                                           GqaBlockMask selection) {
    const GqaPrefillDirectMetadata metadata{
        .table  = static_cast<const std::int32_t*>(cache.block_table.data),
        .window = sliding_window};
    dispatch_prompt_geometry(q.ne[0], q.ne[1], cache.num_kv_heads, [&]<typename Geometry>() {
        gqa_attention_prompt_attention_launch_for<Geometry>(q, positions, scale, cache, metadata,
                                                            out, stream, selection);
    });
}

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    // Append depends on the head dimension and the KV head count only, so it
    // dispatches on those rather than on a full shape it does not have.
    gqa_dispatch_kv_geometry(k.ne[0], k.ne[1], [&]<typename Geometry>() {
        gqa_kv_append_launch_for<Geometry>(k, v, positions, cache, metadata, stream);
    });
}

/// The prompt route over a cache that is already populated: the same launch minus the append.
///
/// A layer that shares an earlier layer's keys and values reads them and writes nothing. It is
/// the append that is optional here, not the attention -- the metadata, the mask and the
/// geometry dispatch are the ones any layer uses.
void gqa_attention_prompt_cached_launch(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        float scale, PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream, std::int32_t sliding_window,
                                        GqaBlockMask selection) {
    if (q.ne[3] > 1 && !gqa_shape_is_registered(q.ne[0], q.ne[1], cache.num_kv_heads)) {
        for (int batch = 0; batch < q.ne[3]; ++batch) {
            Tensor q_row = batch_tensor(q, batch), out_row = batch_tensor(out, batch);
            const Tensor pos = row_tensor(positions, batch, q.ne[2]);
            const Tensor valid = row_tensor(valid_columns, batch, 1);
            const Tensor table = row_tensor(table_rows, batch, 1);
            GqaBlockMask mask = selection;
            if (mask.words) { mask.words += static_cast<std::int64_t>(batch) * q.ne[2] * mask.stride; }
            gqa_attention_prompt_cached_launch(q_row, pos, valid, table, scale, cache,
                                                out_row, stream, sliding_window, mask);
        }
        return;
    }
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
            .window       = sliding_window,
        };
        dispatch_prompt_geometry(q.ne[0], q.ne[1], cache.num_kv_heads, [&]<typename Geometry>() {
            gqa_attention_prompt_attention_launch_for<Geometry>(q, positions, scale, cache,
                                                                metadata, out, stream, selection);
        });
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream, std::int32_t sliding_window,
                                 GqaBlockMask selection) {
    if (q.ne[3] > 1 && !gqa_shape_is_registered(q.ne[0], q.ne[1], cache.num_kv_heads)) {
        for (int batch = 0; batch < q.ne[3]; ++batch) {
            Tensor q_row = batch_tensor(q, batch), out_row = batch_tensor(out, batch);
            const Tensor pos = row_tensor(positions, batch, q.ne[2]);
            const Tensor valid = row_tensor(valid_columns, batch, 1);
            const Tensor table = row_tensor(table_rows, batch, 1);
            GqaBlockMask mask = selection;
            if (mask.words) { mask.words += static_cast<std::int64_t>(batch) * q.ne[2] * mask.stride; }
            const Tensor k_row = batch_tensor(k, batch), v_row = batch_tensor(v, batch);
            gqa_attention_prompt_launch(q_row, k_row, v_row, pos, valid, table, scale, cache,
                                         out_row, stream, sliding_window, mask);
        }
        return;
    }
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
            .window       = sliding_window,
        };
        dispatch_prompt_geometry(q.ne[0], q.ne[1], cache.num_kv_heads, [&]<typename Geometry>() {
            gqa_kv_append_launch_for<Geometry>(k, v, positions, cache, metadata, stream);
            gqa_attention_prompt_attention_launch_for<Geometry>(q, positions, scale, cache,
                                                                metadata, out, stream, selection);
        });
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace sinfer::ops::detail
