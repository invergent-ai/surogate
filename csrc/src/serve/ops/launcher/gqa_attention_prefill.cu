// sinfer::ops - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/launcher/gqa_attention.h"
#include "ops/kernel/func_attribute.cuh"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail {
namespace {

// A key's int8 group scales are copied as a single cp.async, so they must be a
// legal transfer width. With kGqaKvQuantGroup at 64 that means a head carrying at
// least two groups -- head dim 128 or wider. A 64-wide head carries one group,
// two bytes, which no cp.async width covers, so this kernel cannot be compiled
// for it at all. Shapes outside the set refuse at the dispatch rather than
// blocking the build; their bf16 and e4m3 caches are unaffected.
template <typename Geometry>
inline constexpr bool kGqaI8PrefillRegistered =
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
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    if (selection.words != nullptr && cache.dtype != DType::BF16) {
        throw std::invalid_argument(
            "gqa_attention: the QSA selection needs a BF16 KV cache (the quantized kernels are dense)");
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
                static_cast<__nv_bfloat16*>(out.data), tokens);
    } else if (cache.dtype == DType::FP8_E4M3FN) {
        CUDA_CHECK(::sinfer::ops::set_func_attribute_per_device(
            gqa_attention_prefill_bf16_kernel<Geometry, Metadata, std::uint8_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        gqa_attention_prefill_bf16_kernel<Geometry, Metadata, std::uint8_t>
            <<<attention_grid, kGqaPrefillThreads, kSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::uint8_t*>(cache_k.data),
                static_cast<const std::uint8_t*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        if (selection.words != nullptr) {
            // QSA selection: only the 4-cell block is registered.
            if (selection.block != 4) {
                throw std::invalid_argument("gqa_attention: unregistered QSA block size");
            }
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
                    static_cast<__nv_bfloat16*>(out.data), tokens);
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
                                           cudaStream_t stream, GqaBlockMask selection) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    gqa_dispatch_geometry(q.ne[0], q.ne[1], cache.num_kv_heads, [&]<typename Geometry>() {
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

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream, GqaBlockMask selection) {
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        gqa_dispatch_geometry(q.ne[0], q.ne[1], cache.num_kv_heads, [&]<typename Geometry>() {
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
