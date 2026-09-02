// Shared definition of the small-T GQA decode launcher, so each supported head
// geometry can be instantiated in its own translation unit.
//
// This template pulls in the bf16, INT8 and FP8 decode kernels and is instantiated
// for five geometries and two cache-input types. Compiled together that is one
// ~5-minute `cicc` invocation and the critical path of every serve build; compiled
// one geometry per file they run in parallel. Nothing about the generated code
// changes — only how many of these instantiations share a translation unit.
#pragma once

// sinfer::ops - split-KV GQA small-T launcher and unified route dispatcher.
#include "ops/launcher/gqa_attention.h"
#include "ops/kernel/func_attribute.cuh"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "core/device.h" // CUDA_CHECK
#include "api/ops/gqa_attention.h"

#include <cstdint>
#include <string>
#include <cstdlib>
#include <stdexcept>


namespace sinfer::ops::detail {
namespace {

// Whether the int8 decode kernel has a warp route for this shape. That kernel
// splits a Br x HeadDim output across Wc/RowTiles consumer warps, each of which
// must own a whole number of eight-wide n-tiles; the warp counts below were tuned
// for the 256-wide heads whose query group is four to eight, and no other shape
// divides into them (a group of two leaves twenty-four consumer warps sharing one
// row tile; a group of twelve leaves one). Shapes outside it have no int8 decode
// kernel and say so at the dispatch rather than compiling one that would address
// its own output wrongly. Their bf16 and e4m3 caches are unaffected.
template <typename Geometry>
inline constexpr bool kGqaI8DecodeRegistered =
    Geometry::HeadDim == 256 && Geometry::GroupSize >= 4 && Geometry::GroupSize <= 8;

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t gqa_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::DecodeSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::DecodeSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::DecodeSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::DecodeSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::DecodeSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::DecodeSplitScale); }

    return (splits < Geometry::DecodeSplits) ? splits : Geometry::DecodeSplits;
}

template <typename Geometry>
std::int32_t gqa_small_t_split_count(std::int32_t window, std::int32_t tokens, DType kv_dtype) {
    // A 64-key default split just above a 32-key boundary makes the partial
    // kernel execute a nearly empty second tile. These short ranges instead
    // launch one 32-key tile per split; the larger CTAs keep the small grid busy.
    if (kv_dtype == DType::I8 && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::DecodeSplitScale);
    }
    if (kv_dtype == DType::I8 && tokens == 6 && window > 128 && window <= 160) {
        return div_up(window, 24 / Geometry::DecodeSplitScale);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (kv_dtype == DType::I8 && tokens == 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::DecodeSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::DecodeSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::DecodeSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return gqa_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t gqa_small_t_launch_capacity(GqaExecutionEnvelope envelope, std::int32_t tokens,
                                         DType dtype) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits =
            gqa_small_t_split_count<Geometry>(static_cast<std::int32_t>(window), tokens, dtype);
        capacity = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput, typename CacheT = __nv_bfloat16>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k = cache.k_pages;
    Tensor& cache_v = cache.v_pages;
    // bf16 kernel uses only static smem (no dynamic staging). A QSA selection instantiates the
    // sparse specialization; without one the dense kernel is unchanged.
    if (invocation.selection.words != nullptr) {
        if (invocation.selection.block != 4) {
            throw std::invalid_argument("gqa_attention: unregistered QSA block size");
        }
        gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                     Masked, CacheInput, CacheT, true, 4>
            <<<grid, kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<CacheT*>(cache_k.data),
                static_cast<CacheT*>(cache_v.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.width, invocation.full_width,
                invocation.column_begin, logical_capacity, invocation.sliding_window, scale,
                static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                invocation.selection);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                 Masked, CacheInput,
                                                 CacheT><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<CacheT*>(cache_k.data),
        static_cast<CacheT*>(cache_v.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
        logical_capacity, invocation.sliding_window, scale,
        static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * Geometry::HeadDim) : 0u;
        if constexpr (DynamicArena) {
            CUDA_CHECK(::sinfer::ops::set_func_attribute_per_device(
                gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     MultiBatch, Masked, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes)));
        }
        gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                             KeyBlock, DynamicArena, MultiBatch, Masked, CacheInput>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, invocation.sliding_window, scale,
        static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    if constexpr (TokenTile == 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if constexpr (Geometry::GroupSize == 4) {
            // surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b. RowTiles=2
            // here, so Wc must keep Wc/RowTiles in {2,4,8,16}.
            if (implementation_window > 128 && implementation_window <= 160) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 2054) {
                launch.template operator()<16, 1, 32, false>();
            } else if (implementation_window <= 8198) {
                launch.template operator()<16, 1, 64, true>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 4) {
            // surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b (RowTiles=2).
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

PagedKVBatchLayerView single_row_batch_view(const PagedKVLayerView& cache) {
    return {
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .dtype         = cache.dtype,
        .quant_group   = cache.quant_group,
    };
}

} // namespace


template <typename Geometry, typename CacheInput>
void gqa_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                      float scale, PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                      Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                      cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    auto splits = gqa_small_t_launch_capacity<Geometry>(envelope, invocation.width, cache.dtype);
    // Batch-aware clamp: the grid is (KVHeads, splits, batch), so the batch
    // dimension already supplies parallelism at multi-user widths. Splitting KV
    // further past a full wave only multiplies the partial-buffer traffic the
    // reducer then has to read back — at 64 lanes the old policy asked for ~10
    // splits and 15 waves of CTAs. Keep enough CTAs to fill the device twice
    // over and no more. The value stays constant per (envelope, batch), which
    // is what a captured decode graph requires.
    if (invocation.batch_size > 0) {
        // Target CTA count for the split policy, in units of a 170-SM wave.
        // Tunable so the wave count can be swept against a real load rather
        // than argued about.
        static const std::int32_t kTargetCtas = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_ATTN_WAVES");
            const int waves = raw != nullptr ? std::atoi(raw) : 2;
            return static_cast<std::int32_t>((waves > 0 ? waves : 2) * 170);
        }();
        const std::int32_t per_split       = Geometry::KVHeads * invocation.batch_size;
        const std::int32_t wanted          = per_split > 0 ? div_up(kTargetCtas, per_split) : splits;
        const std::int32_t floored         = wanted < 1 ? 1 : wanted;
        const std::int32_t occupancy_split = splits < floored ? splits : floored;
        // A split stages its page ids in a fixed 64-entry shared array, so it may
        // span at most 64 pages. The clamp above only ever lowers the split count,
        // and lowering it widens each split -- so a shape with many KV heads (the
        // clamp divides by them) can be pushed past the page budget and write off
        // the end of that array. Keep the count at or above what the budget needs,
        // still capped by the split capacity the partial buffers were sized for.
        // The tile allowance covers the leading partial tile a split may start on.
        constexpr std::int32_t kMaxPagesPerSplit = 64;
        constexpr std::int32_t kMaxTokenTile     = 128;
        constexpr std::int32_t kKeysPerSplitCap =
            kMaxPagesPerSplit * kPagedKVPageSize - kMaxTokenTile;
        const std::int32_t page_floor =
            implementation_window > 0 ? div_up(implementation_window, kKeysPerSplitCap) : 1;
        const std::int32_t needed = occupancy_split > page_floor ? occupancy_split : page_floor;
        splits                    = splits < needed ? splits : needed;
    }

    // BF16 keeps its row-tile warp count; INT8 selects its producer/consumer
    // geometry inside launch_tc_partial_i8.
#define SINFER_GQA_SMALL_T_DISPATCH(TOKENS, WARPS)                                                 \
    do {                                                                                           \
        const auto launch_profile = [&]<bool MultiBatch, bool Masked>() {                          \
            if (cache.dtype == DType::I8) {                                                        \
                if constexpr (!kGqaI8DecodeRegistered<Geometry>) {                                 \
                    throw std::invalid_argument(                                                   \
                        "gqa_attention: int8 KV is not served for head dim " +                     \
                        std::to_string(Geometry::HeadDim) + " with " +                             \
                        std::to_string(Geometry::QHeads) + " query heads over " +                  \
                        std::to_string(Geometry::KVHeads) + " KV heads");                          \
                } else {                                                                           \
                    launch_tc_partial_i8<Geometry, (TOKENS), MultiBatch, Masked>(                  \
                        q, input, pos, scale, cache, invocation, logical_capacity,                 \
                        implementation_window, splits, partial_acc, partial_m, partial_l, stream); \
                }                                                                                  \
            } else if (cache.dtype == DType::FP8_E4M3FN) {                                         \
                launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked,            \
                                       decltype(input), std::uint8_t>(                             \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            } else {                                                                               \
                launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(           \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            }                                                                                      \
        };                                                                                         \
        const bool masked = invocation.valid_columns != nullptr;                                   \
        if (invocation.batch_size == 1) {                                                          \
            if (masked) {                                                                          \
                launch_profile.template operator()<false, true>();                                 \
            } else {                                                                               \
                launch_profile.template operator()<false, false>();                                \
            }                                                                                      \
        } else if (masked) {                                                                       \
            launch_profile.template operator()<true, true>();                                      \
        } else {                                                                                   \
            launch_profile.template operator()<true, false>();                                     \
        }                                                                                          \
    } while (0)

    if (invocation.width * Geometry::GroupSize > 64) {
        throw std::invalid_argument(
            "gqa_attention: this head geometry serves at most 64 query rows per lane step (width " +
            std::to_string(invocation.width) + " x group " + std::to_string(Geometry::GroupSize) +
            ", batch " + std::to_string(invocation.batch_size) + ", full width " +
            std::to_string(invocation.full_width) + ", column begin " +
            std::to_string(invocation.column_begin) + ")");
    }
    switch (invocation.width) {
    case 1:
        SINFER_GQA_SMALL_T_DISPATCH(1, 2);
        break;
    case 2:
        SINFER_GQA_SMALL_T_DISPATCH(2, 4);
        break;
    case 3:
        SINFER_GQA_SMALL_T_DISPATCH(3, 4);
        break;
    case 4:
        SINFER_GQA_SMALL_T_DISPATCH(4, 4);
        break;
    case 5:
        SINFER_GQA_SMALL_T_DISPATCH(5, 4);
        break;
    case 6:
        SINFER_GQA_SMALL_T_DISPATCH(6, 4);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
#undef SINFER_GQA_SMALL_T_DISPATCH

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(Geometry::QHeads, div_up(Geometry::HeadDim, kDChunk),
                           invocation.width * invocation.batch_size);
    const auto launch_reduce = [&]<bool Int8, bool MultiBatch, bool Masked, bool Offset>() {
        gqa_attention_small_t_reduce_output_kernel<Geometry, kDChunk, Int8, MultiBatch, Masked,
                                                   Offset>
            <<<reduce_grid, kReduceBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, invocation.sliding_window,
                static_cast<__nv_bfloat16*>(out.data));
    };
    const bool masked         = invocation.valid_columns != nullptr;
    const auto launch_profile = [&]<bool Int8, bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0) {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, false>();
        } else {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, true>();
        }
    };
    const auto launch_for_dtype = [&]<bool Int8>() {
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_profile.template operator()<Int8, false, true>();
            } else {
                launch_profile.template operator()<Int8, false, false>();
            }
        } else if (masked) {
            launch_profile.template operator()<Int8, true, true>();
        } else {
            launch_profile.template operator()<Int8, true, false>();
        }
    };
    if (cache.dtype == DType::I8) {
        launch_for_dtype.template operator()<true>();
    } else {
        launch_for_dtype.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace sinfer::ops::detail
