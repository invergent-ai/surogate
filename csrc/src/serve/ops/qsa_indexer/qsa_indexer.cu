#include "api/ops/qsa_indexer.h"

#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim  = 128; // the only registered indexer width
constexpr int kBlock    = 4;   // the only registered compress ratio
constexpr int kWarp     = 32;
constexpr int kPerLane  = kHeadDim / kWarp; // 4 values of the head per lane
constexpr int kWarps    = 4;
constexpr int kThreads  = kWarps * kWarp;

// The indexer plane holds one head of `kHeadDim` per cell.
__device__ __forceinline__ std::int64_t indexer_offset(const std::int32_t* block_table,
                                                       std::int32_t position) {
    return paged_kv_element_offset<kHeadDim, 1>(block_table, 0, position, 0);
}

// Same arithmetic as the engine's rope kernel (ops/kernel/rope.cuh): an accurate `powf` for the
// frequency and `sincosf` for the rotation — the fast intrinsics lose the low-frequency pairs,
// whose angle grows with the position.
__device__ __forceinline__ void rope_sincos(float position, int pair, int rotary_dim, float theta,
                                            float* sine, float* cosine) {
    const float frequency =
        powf(theta, -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim));
    sincosf(position * frequency, sine, cosine);
}

// One warp per new column: writes the token's raw key and, when the column completes a block,
// folds that block's `kBlock` raw keys into the block key kept at the block's first cell.
__global__ void qsa_append_kernel(const __nv_bfloat16* __restrict__ keys,
                                  const std::int32_t* __restrict__ positions,
                                  const std::int32_t* __restrict__ table_rows,
                                  const std::int32_t* __restrict__ block_tables,
                                  std::int32_t table_stride,
                                  const __nv_bfloat16* __restrict__ key_norm,
                                  __nv_bfloat16* __restrict__ plane, int tokens, int rotary_dim,
                                  float theta, float eps) {
    const int warp  = static_cast<int>(threadIdx.x) / kWarp;
    const int lane  = static_cast<int>(threadIdx.x) % kWarp;
    const int token = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (token >= tokens) { return; }
    const int position = positions[token];
    const int d0       = lane * kPerLane;
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_rows == nullptr ? 0 : table_rows[token]) *
                           table_stride;

    float raw[kPerLane];
    const __nv_bfloat16* src = keys + static_cast<std::int64_t>(token) * kHeadDim + d0;
    __nv_bfloat16* dst       = plane + indexer_offset(block_table, position) + d0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        raw[i] = __bfloat162float(src[i]);
        dst[i] = __float2bfloat16(raw[i]);
    }
    if ((position + 1) % kBlock != 0) { return; } // the block is still open

    // Every cell of the block is written: the earlier ones by earlier rounds or by earlier
    // columns of this launch, whose stores are ordered before this read by the fence.
    __threadfence();
    const int first = position - (kBlock - 1);
    float value[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) { value[i] = 0.0F; }
    for (int c = 0; c < kBlock; ++c) {
        const __nv_bfloat16* cell = plane + indexer_offset(block_table, first + c) + d0;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) { value[i] += __bfloat162float(cell[i]); }
    }
    float square = 0.0F;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        value[i] /= static_cast<float>(kBlock);
        square += value[i] * value[i];
    }
    square            = warp_reduce_sum(square);
    square            = __shfl_sync(0xFFFFFFFFU, square, 0);
    const float scale = rsqrtf(square / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        value[i] *= scale * __bfloat162float(key_norm[d0 + i]);
    }

    // Split-half NeoX rope at the block's first position. The partner of dimension d < half is
    // d + half, and both halves live inside this warp, so a shuffle fetches it.
    const int half = rotary_dim / 2;
    float partner[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int d    = d0 + i;
        const int mate = d < half ? d + half : d - half;
        float mate_value = 0.0F;
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
            const float candidate = __shfl_sync(0xFFFFFFFFU, value[j], mate / kPerLane);
            if (mate % kPerLane == j) { mate_value = candidate; }
        }
        partner[i] = mate_value;
    }
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int d = d0 + i;
        if (d >= rotary_dim) { continue; }
        const int pair = d < half ? d : d - half;
        float sin_a, cos_a;
        rope_sincos(static_cast<float>(first), pair, rotary_dim, theta, &sin_a, &cos_a);
        value[i] = d < half ? value[i] * cos_a - partner[i] * sin_a
                            : value[i] * cos_a + partner[i] * sin_a;
    }
    __nv_bfloat16* block_dst = plane + indexer_offset(block_table, first) + d0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) { block_dst[i] = __float2bfloat16(value[i]); }
}

// One CTA per query row: scores every complete block, finds the budget's threshold, and writes
// the row's block bitmask. One warp handles one block at a time.
template <int Heads>
__global__ void qsa_select_kernel(const __nv_bfloat16* __restrict__ q,
                                  const std::int32_t* __restrict__ positions,
                                  const std::int32_t* __restrict__ table_rows,
                                  const std::int32_t* __restrict__ block_tables,
                                  std::int32_t table_stride,
                                  const __nv_bfloat16* __restrict__ plane,
                                  std::uint32_t* __restrict__ mask, int words, int budget_blocks,
                                  float* __restrict__ scores, int score_stride) {
    const int row      = static_cast<int>(blockIdx.x);
    const int position = positions[row];
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid / kWarp;
    const int lane     = tid % kWarp;
    const int d0       = lane * kPerLane;
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_rows[row]) * table_stride;
    std::uint32_t* row_mask = mask + static_cast<std::int64_t>(row) * words;

    const int visible = position + 1;                    // cells 0..position are populated
    const int scored  = visible / kBlock;                // complete blocks
    const int touched = (visible + kBlock - 1) / kBlock; // complete blocks plus the open tail

    for (int w = tid; w < words; w += kThreads) { row_mask[w] = 0U; }
    __syncthreads();
    for (int b = scored + tid; b < touched; b += kThreads) { // the tail is always visible
        atomicOr(&row_mask[b >> 5], 1U << (b & 31));
    }
    if (scored <= budget_blocks) { // the budget covers everything: the selection is the identity
        for (int b = tid; b < scored; b += kThreads) {
            atomicOr(&row_mask[b >> 5], 1U << (b & 31));
        }
        return;
    }

    float qv[Heads][kPerLane];
#pragma unroll
    for (int h = 0; h < Heads; ++h) {
        const __nv_bfloat16* src =
            q + (static_cast<std::int64_t>(row) * Heads + h) * kHeadDim + d0;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) { qv[h][i] = __bfloat162float(src[i]); }
    }

    float* row_scores = scores + static_cast<std::int64_t>(row) * score_stride;
    for (int b = warp; b < scored; b += kWarps) {
        const __nv_bfloat16* k = plane + indexer_offset(block_table, b * kBlock) + d0;
        float kv[kPerLane];
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) { kv[i] = __bfloat162float(k[i]); }
        float total = 0.0F;
#pragma unroll
        for (int h = 0; h < Heads; ++h) {
            float dot = 0.0F;
#pragma unroll
            for (int i = 0; i < kPerLane; ++i) { dot += qv[h][i] * kv[i]; }
            dot = warp_reduce_sum(dot);
            total += fmaxf(dot, 0.0F); // rectified per head, as in the reference
        }
        if (lane == 0) { row_scores[b] = total; }
    }
    __syncthreads();

    // Threshold: the budget-th largest score. The scores are non-negative, so their bit patterns
    // order like the floats and 32 counting passes bisect the threshold exactly.
    __shared__ unsigned threshold;
    __shared__ int reduce[kWarps];
    if (tid == 0) { threshold = 0U; }
    __syncthreads();
    for (int bit = 31; bit >= 0; --bit) {
        const unsigned candidate = threshold | (1U << bit);
        int local                = 0;
        for (int b = tid; b < scored; b += kThreads) {
            if (__float_as_uint(row_scores[b]) >= candidate) { ++local; }
        }
        local = static_cast<int>(warp_reduce_sum(static_cast<float>(local)));
        if (lane == 0) { reduce[warp] = local; }
        __syncthreads();
        if (tid == 0) {
            int total = 0;
            for (int w = 0; w < kWarps; ++w) { total += reduce[w]; }
            if (total >= budget_blocks) { threshold = candidate; }
        }
        __syncthreads();
    }

    // Everything strictly above the threshold is in; blocks exactly on it fill the remaining
    // budget. Which of several equally scoring blocks gets in is unspecified — the reference
    // breaks those ties by cell index, we by arrival, and the scores are floats.
    __shared__ int ties_left;
    if (tid == 0) { ties_left = budget_blocks; }
    __syncthreads();
    for (int b = tid; b < scored; b += kThreads) {
        const unsigned bits = __float_as_uint(row_scores[b]);
        if (bits > threshold) {
            atomicSub(&ties_left, 1);
            atomicOr(&row_mask[b >> 5], 1U << (b & 31));
        }
    }
    __syncthreads();
    for (int b = tid; b < scored; b += kThreads) {
        if (__float_as_uint(row_scores[b]) != threshold) { continue; }
        if (atomicSub(&ties_left, 1) > 0) { atomicOr(&row_mask[b >> 5], 1U << (b & 31)); }
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa_indexer: ") + message); }
}

} // namespace

std::int32_t qsa_block_mask_words(std::int32_t keys, std::int32_t block) {
    require(keys >= 0 && block > 0, "mask geometry must be positive");
    const std::int32_t blocks = (keys + block - 1) / block;
    return (blocks + 31) / 32;
}

bool qsa_selection_is_dense(std::int32_t keys, const QsaIndexerGeometry& geometry) {
    require(geometry.block > 0 && geometry.top_k > 0, "geometry must be positive");
    return keys <= geometry.top_k + geometry.block - 1;
}

std::size_t qsa_indexer_select_workspace_capacity_bytes(std::int32_t rows, std::int32_t keys,
                                                        const QsaIndexerGeometry& geometry) {
    require(rows >= 0 && keys >= 0 && geometry.block > 0, "workspace geometry must be positive");
    const std::size_t blocks =
        static_cast<std::size_t>((keys + geometry.block - 1) / geometry.block);
    return static_cast<std::size_t>(rows) * blocks * sizeof(float) + 256U;
}

void qsa_indexer_append(const Tensor& keys, const Tensor& positions, const Tensor& table_rows,
                        const Tensor& key_norm, const QsaIndexerGeometry& geometry,
                        PagedKVBatchLayerView cache, cudaStream_t stream) {
    require(geometry.head_dim == kHeadDim && geometry.block == kBlock,
            "only the 128-wide, 4-cell indexer is registered");
    require(geometry.rotary_dim > 0 && geometry.rotary_dim <= kHeadDim &&
                (geometry.rotary_dim % 2) == 0,
            "rotary_dim must be even and within the head");
    require(keys.dtype == DType::BF16 && key_norm.dtype == DType::BF16, "keys must be BF16");
    require(positions.dtype == DType::I32, "positions must be I32");
    require(keys.ne[0] == kHeadDim, "keys must be [head_dim, T]");
    require(cache.indexer_pages.data != nullptr, "the cache carries no indexer plane");
    require(cache.block_tables.data != nullptr, "the cache view has no block tables");
    const int tokens = static_cast<int>(keys.ne[1]);
    if (tokens == 0) { return; }
    require(positions.ne[0] == tokens, "positions must match the columns");
    const int blocks = (tokens + kWarps - 1) / kWarps;
    require(table_rows.data == nullptr || table_rows.ne[0] == tokens,
            "table rows must match the columns");
    qsa_append_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(keys.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        static_cast<const __nv_bfloat16*>(key_norm.data),
        static_cast<__nv_bfloat16*>(cache.indexer_pages.data), tokens, geometry.rotary_dim,
        geometry.rope_theta, geometry.rms_eps);
}

void qsa_indexer_select(const Tensor& q, const Tensor& positions, const Tensor& table_rows,
                        const QsaIndexerGeometry& geometry, PagedKVBatchLayerView cache,
                        std::int32_t keys, WorkspaceArena& workspace, Tensor& mask,
                        cudaStream_t stream) {
    require(geometry.head_dim == kHeadDim && geometry.block == kBlock,
            "only the 128-wide, 4-cell indexer is registered");
    require(q.dtype == DType::BF16 && positions.dtype == DType::I32 &&
                table_rows.dtype == DType::I32 && mask.dtype == DType::I32,
            "select tensor dtypes are wrong");
    require(cache.indexer_pages.data != nullptr, "the cache carries no indexer plane");
    require(q.ne[0] == kHeadDim && q.ne[1] == geometry.heads, "q must be [head_dim, heads, rows]");
    const int rows = static_cast<int>(q.ne[2]);
    if (rows == 0) { return; }
    const int words = qsa_block_mask_words(keys, geometry.block);
    require(mask.ne[0] == words && mask.ne[1] == rows, "mask must be [words, rows]");
    require(positions.ne[0] == rows && table_rows.ne[0] == rows,
            "positions and table rows must match the query rows");
    const int blocks  = (keys + geometry.block - 1) / geometry.block;
    Tensor scores     = workspace.alloc(DType::FP32, {blocks, rows});
    const auto stride = static_cast<int>(blocks);
    const int budget  = geometry.top_k / geometry.block;
    const auto launch = [&](auto heads) {
        qsa_select_kernel<decltype(heads)::value><<<rows, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(table_rows.data),
            static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
            static_cast<const __nv_bfloat16*>(cache.indexer_pages.data),
            static_cast<std::uint32_t*>(mask.data), words, budget,
            static_cast<float*>(scores.data), stride);
    };
    switch (geometry.heads) {
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 8: launch(std::integral_constant<int, 8>{}); break;
    default: throw std::invalid_argument("qsa_indexer: unregistered indexer head count");
    }
}

} // namespace ninfer::ops
