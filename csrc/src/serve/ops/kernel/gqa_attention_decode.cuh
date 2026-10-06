#pragma once

// sinfer::ops - split-KV GQA small-T attention shared scaffolding. The bf16 and
// int8 partial kernels live in gqa_attention_decode_bf16.cuh and
// gqa_attention_decode_i8.cuh respectively; they are fully separate kernels (no
// shared body) so each KV format can be optimized independently. This header owns
// only what both share: layout constants, device helpers, and the split reducer.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <atomic>
#include <cstdint>

namespace sinfer::ops {

// Head dimension now lives on the geometry (see gqa_attention_geometry.cuh).

struct GqaAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct GqaCachedInput {
    static constexpr bool writes_cache = false;
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_cache_index(int physical_page, int kv_head, int d,
                                                        int page_offset) {
    return paged_kv_element_offset<Geometry::HeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                   page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(Geometry::HeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_new_index(int kv_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(Geometry::HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_acc_index(int q_head, int d, int token,
                                                              int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(Geometry::HeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_stat_index(int q_head, int token, int split,
                                                               int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool gqa_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < Geometry::KVHeads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize && q_head < Geometry::QHeads;
}

/// The first key any token of this tile may attend to.
///
/// A sliding window makes the oldest keys invisible to every token at once: the
/// earliest query in the tile admits keys from `first_pos - window + 1`, and
/// nothing below that can contribute to any of them. Splitting the range from
/// here, rather than from key zero, is what keeps a windowed layer's decode cost
/// bounded by its window instead of by the whole context.
///
/// The split kernels and the reducer must agree on the range exactly -- they
/// derive the active split count from it independently -- so both call this.
__device__ __forceinline__ int gqa_small_t_key_lo(int first_pos, int sliding_window) {
    if (sliding_window <= 0) { return 0; }
    const int lo = first_pos - sliding_window + 1;
    return lo > 0 ? lo : 0;
}

// Hopper (sm_90) uses narrower partitions past 4K keys. A split CTA is two warps that stage
// one 32-key tile at a time, so its time grows with its partition, and the widest partition
// is what the whole layer waits for: with the default scheme a 17K-key decode spends ~80 us
// per layer behind one 2048-key partition while most of the H100 sits idle. Partitions of
// 128 keys up to 16K and 256 up to 64K keep every CTA short, at about three times the split
// storage of the default scheme. Device code knows its own architecture; the host asks the
// current device, which is the one its launches and workspaces are for.
namespace detail {
inline bool gqa_host_device_uses_fine_partitions() {
    static std::atomic<int> cached[64] = {};  // 0 unknown, 1 default scheme, 2 Hopper
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) { return false; }
    int known = cached[device].load(std::memory_order_relaxed);
    if (known == 0) {
        int major = 0;
        int minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        known = major == 9 && minor == 0 ? 2 : 1;
        cached[device].store(known, std::memory_order_relaxed);
    }
    return known == 2;
}
} // namespace detail

__host__ __device__ inline bool gqa_fine_key_partitions() {
#if defined(__CUDA_ARCH__)
#    if __CUDA_ARCH__ == 900
    return true;
#    else
    return false;
#    endif
#else
    return detail::gqa_host_device_uses_fine_partitions();
#endif
}

// Partitions are anchored to absolute key positions, never to the current
// sequence length or batch. Wider partitions in the tail bound temporary
// storage at long contexts without regrouping a query when the context grows.
__host__ __device__ inline int gqa_key_partition(int key) {
    if (gqa_fine_key_partitions()) {
        if (key < 16384) { return key / 128; }
        if (key < 65536) { return 128 + (key - 16384) / 256; }
        return 320 + (key - 65536) / 2048;
    }
    if (key < 4096) { return key / 128; }
    if (key < 16384) { return 32 + (key - 4096) / 512; }
    return 56 + (key - 16384) / 2048;
}

__host__ __device__ inline int gqa_key_partition_begin(int partition) {
    if (gqa_fine_key_partitions()) {
        if (partition < 128) { return partition * 128; }
        if (partition < 320) { return 16384 + (partition - 128) * 256; }
        return 65536 + (partition - 320) * 2048;
    }
    if (partition < 32) { return partition * 128; }
    if (partition < 56) { return 4096 + (partition - 32) * 512; }
    return 16384 + (partition - 56) * 2048;
}

template <typename Geometry>
__device__ __forceinline__ int gqa_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::DecodeSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::DecodeSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::DecodeSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::DecodeSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::DecodeSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::DecodeSplits ? splits : Geometry::DecodeSplits;
}

template <typename Geometry, bool Int8>
__device__ __forceinline__ int gqa_small_t_active_splits(int window, int launch_capacity,
                                                         int tokens) {
    if (window <= 0) { return launch_capacity; }
    if constexpr (!Int8) { return min(launch_capacity, div_up(window, 128)); }
    int splits = 0;
    if constexpr (Int8) {
        if (tokens == 5 && window > 128 && window <= 512) {
            splits = div_up(window, 32 / Geometry::DecodeSplitScale);
        } else if (tokens == 6 && window > 128 && window <= 160) {
            splits = div_up(window, 24 / Geometry::DecodeSplitScale);
        } else if (tokens == 6 && window > 5000 && window <= 8198) {
            splits             = div_up(window, 192 / Geometry::DecodeSplitScale);
            constexpr int kMin = 4 * Geometry::DecodeSplitScale;
            constexpr int kMax = 42 * Geometry::DecodeSplitScale;
            splits             = splits > kMin ? splits : kMin;
            splits             = splits < kMax ? splits : kMax;
        } else {
            splits = gqa_small_t_default_splits<Geometry>(window);
        }
    } else {
        splits = gqa_small_t_default_splits<Geometry>(window);
    }
    return splits < launch_capacity ? splits : launch_capacity;
}

__device__ __forceinline__ int gqa_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int gqa_small_t_tc_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

// Signed int8 QK MMA, k=32 contraction. A = 16x32 s8 (4 regs/thread, 4 s8 each),
// B = 8x32 s8 col-major (2 regs/thread), D = 16x8 s32 (4 regs/thread). The A/B
// register byte layout is identical to the m16n8k16 bf16 fragments loaded by
// ldmatrix_x4/x2 over a d-contiguous int8 tile reinterpreted as
// b16 (two packed int8 per 16-bit lane), so the same ldmatrix helpers and XOR
// swizzle feed this MMA. The s32 accumulator layout matches the bf16 f32
// accumulator (c0/c1 -> row groupID, c2/c3 -> row groupID+8), so score
// consumption is unchanged; only per-64-group scale rescale differs.
template <typename Geometry>
__device__ __forceinline__ void gqa_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                         int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

template <typename Geometry, int DChunk, bool Int8, bool MultiBatch, bool Masked, bool Offset,
          bool LaneColumns = false>
__launch_bounds__(256) __global__ void gqa_attention_small_t_reduce_output_kernel(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, std::int32_t sliding_window, __nv_bfloat16* out,
    const std::int32_t* lane_columns = nullptr) {
    static_assert(DChunk > 0 && DChunk <= Geometry::HeadDim);
    static_assert(!LaneColumns || (MultiBatch && !Offset), "lane columns replace the batch stride");

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (LaneColumns) {
        positions += lane_columns[batch];
    } else {
        if constexpr (Offset) { positions += column_begin; }
        if constexpr (MultiBatch) { positions += batch * full_width; }
    }
    const int last_pos  = positions[tokens - 1];
    const int first_pos = positions[0];
    int output_column   = token;
    if constexpr (LaneColumns) {
        output_column += lane_columns[batch];
    } else {
        if constexpr (Offset) { output_column += column_begin; }
        if constexpr (MultiBatch) { output_column += batch * full_width; }
    }

    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * Geometry::HeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    // The same range the split kernels partitioned, derived the same way.
    const int key_lo = gqa_small_t_key_lo(first_pos, sliding_window);
    const int window = (last_pos + 1) - key_lo;
    const int active_split_count = Int8
        ? gqa_small_t_active_splits<Geometry, true>(window, split_count, tokens)
        : min(split_count, gqa_key_partition(last_pos) + 1 - gqa_key_partition(key_lo));

    __shared__ float reduce[256];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        local_m = fmaxf(local_m,
                        partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)]);
    }
    reduce[tid] = local_m;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]); }
        __syncthreads();
    }
    const float head_m = reduce[0];
    __syncthreads();

    if (head_m == -CUDART_INF_F) {
        const int d = d_start + tid;
        if (tid < DChunk && d < Geometry::HeadDim) {
            out[gqa_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(0.0f);
        }
        return;
    }

    // Each split's weight, computed once by the whole block instead of serially per output
    // value. The sums below still add the same terms in the same order, so the result is
    // bit-identical to the serial form; it just stops waiting on one load per split.
    constexpr int kStagedSplits = 512;
    __shared__ float split_l[kStagedSplits];
    __shared__ float split_weight[kStagedSplits];
    const bool staged = !Int8 && active_split_count <= kStagedSplits;
    if (staged) {
        for (int split = tid; split < active_split_count; split += blockDim.x) {
            const auto index = gqa_partial_stat_index<Geometry>(q_head, token, split, tokens);
            const float tile_l  = partial_l[index];
            split_l[split]      = tile_l;
            split_weight[split] = tile_l > 0.0f ? expf(partial_m[index] - head_m) : 0.0f;
        }
        __syncthreads();
    }

    // Keep the normalization sum in absolute key order. A sliding-window query
    // tile may add leading empty splits, which must not regroup the nonzero terms.
    if constexpr (!Int8) {
        if (tid == 0) {
            float sum = 0.0f;
            if (staged) {
                for (int split = 0; split < active_split_count; ++split) {
                    if (split_l[split] > 0.0f) { sum += split_l[split] * split_weight[split]; }
                }
            } else {
                for (int split = 0; split < active_split_count; ++split) {
                    const auto index = gqa_partial_stat_index<Geometry>(q_head, token, split, tokens);
                    if (partial_l[index] > 0.0f) {
                        sum += partial_l[index] * expf(partial_m[index] - head_m);
                    }
                }
            }
            reduce[0] = sum;
        }
        __syncthreads();
    } else {
        float local_l = 0.0f;
        for (int split = tid; split < active_split_count; split += blockDim.x) {
            const float tile_l =
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
            if (tile_l > 0.0f) {
                local_l +=
                    tile_l *
                    expf(partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] -
                         head_m);
            }
        }
        reduce[tid] = local_l;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) { reduce[tid] += reduce[tid + stride]; }
            __syncthreads();
        }
    }
    const float head_l = reduce[0];

    const int d = d_start + tid;
    if (tid >= DChunk || d >= Geometry::HeadDim) { return; }

    float numerator = 0.0f;
    if (head_l > 0.0f && staged) {
        // Eight independent loads ahead of the in-order accumulation.
        constexpr int kAhead = 8;
        int split = 0;
        for (; split + kAhead <= active_split_count; split += kAhead) {
            float acc[kAhead];
#pragma unroll
            for (int j = 0; j < kAhead; ++j) {
                acc[j] = split_l[split + j] > 0.0f
                    ? partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split + j, tokens)]
                    : 0.0f;
            }
#pragma unroll
            for (int j = 0; j < kAhead; ++j) {
                if (split_l[split + j] > 0.0f) { numerator += acc[j] * split_weight[split + j]; }
            }
        }
        for (; split < active_split_count; ++split) {
            if (split_l[split] <= 0.0f) { continue; }
            numerator +=
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] *
                split_weight[split];
        }
    } else if (head_l > 0.0f) {
        for (int split = 0; split < active_split_count; ++split) {
            const float tile_l =
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
            if (tile_l <= 0.0f) { continue; }
            const float weight = expf(
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] - head_m);
            numerator +=
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] *
                weight;
        }
    }
    bool valid = true;
    if constexpr (Masked) {
        int absolute_column = token;
        if constexpr (Offset) { absolute_column += column_begin; }
        valid = absolute_column < valid_columns[batch];
    }
    const float value = (valid && head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[gqa_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(value);
}

} // namespace sinfer::ops
