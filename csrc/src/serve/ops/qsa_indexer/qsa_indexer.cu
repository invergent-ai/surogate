#include "api/ops/qsa_indexer.h"

#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace sinfer::ops {
namespace {

constexpr int kHeadDim  = 128; // the only registered indexer width
constexpr int kBlock    = 4;   // the only registered compress ratio
constexpr int kWarp     = 32;
constexpr int kPerLane  = kHeadDim / kWarp; // 4 values of the head per lane
constexpr int kWarps    = 4;
// Scratch the selection may use for block scores, whatever the context length. Small enough to
// stay in L2 between the scoring kernel that writes it and the cut that reads it four times.
constexpr std::int64_t kSelectScratchBytes = 16LL << 20;
constexpr int kThreads  = kWarps * kWarp;

// Scoring: eight lanes share one block key, two 16-byte loads each, and a warp step covers
// the sixteen pooled keys of one page.
constexpr int kScoreThreads = 128;
constexpr int kScoreLanes   = 8;
constexpr int kScoreGroups  = kWarp / kScoreLanes;          // blocks per warp per unrolled step
constexpr int kScoreDims    = kHeadDim / kScoreLanes;       // 16 dims of every head per lane
constexpr int kScoreUnroll  = 4;
constexpr int kWarpBlocks   = kScoreGroups * kScoreUnroll;  // 16
constexpr int kScoreStep    = kScoreThreads / kWarp * kWarpBlocks;
constexpr int kMinScoreSpan = 4 * kScoreStep;
static_assert(kWarpBlocks * kBlock == kPagedKVPageSize, "a warp step reads one page's pooled keys");
static_assert(kPagedKVPageSize * kHeadDim * (kBlock + 1) / kBlock + kPagedKVPageSize / kBlock * 4 * 2 <=
                  kPagedKVPageSize * kQsaIndexerStorageHeadDim,
              "raw keys, pooled keys and block positions fit in a page");

// The cut: radix selection of each row's budget-th largest score, 8 bits per pass.
constexpr int kCutThreads = 512;
constexpr int kCutWarps   = kCutThreads / kWarp;
constexpr int kCutUnroll  = 8; // loads in flight per thread: the cut is latency bound
constexpr int kRadixBits  = 8;
constexpr int kRadixBins  = 1 << kRadixBits;
constexpr int kLaneBins   = kRadixBins / kWarp;

// Raw and pooled keys occupy disjoint regions within each physical page.
template <bool Pooled = false>
__device__ __forceinline__ std::int64_t indexer_offset(const std::int32_t* block_table,
                                                       std::int32_t position) {
    const int cell = position % kPagedKVPageSize;
    const auto page = paged_kv_element_offset<kQsaIndexerStorageHeadDim, 1>(
        block_table, 0, position - cell, 0);
    return page + (Pooled ? kPagedKVPageSize * kHeadDim + (cell / kBlock) * kHeadDim
                          : cell * kHeadDim);
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

__device__ __forceinline__ std::int32_t* block_positions(__nv_bfloat16* plane,
                                                        const std::int32_t* table, int position) {
    const auto page = indexer_offset(table, position - position % kPagedKVPageSize);
    const auto offset = page + kPagedKVPageSize * kHeadDim * (kBlock + 1) / kBlock;
    return reinterpret_cast<std::int32_t*>(plane + offset) + (position % kPagedKVPageSize / kBlock) * 4;
}

// One warp per new column: writes the token's raw indexer key into its cell.
__global__ void qsa_append_raw_kernel(const __nv_bfloat16* __restrict__ keys,
                                      const std::int32_t* __restrict__ positions,
                                      const std::int32_t* __restrict__ table_rows,
                                      std::int32_t columns_per_row,
                                      const std::int32_t* __restrict__ block_tables,
                                      std::int32_t table_stride, __nv_bfloat16* __restrict__ plane,
                                      int tokens, const std::int32_t* rope_positions,
                                      std::int64_t rope_axis_stride, int rope_axes) {
    const int warp  = static_cast<int>(threadIdx.x) / kWarp;
    const int lane  = static_cast<int>(threadIdx.x) % kWarp;
    const int token = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (token >= tokens) { return; }
    const int d0                    = lane * kPerLane;
    const std::int32_t* block_table = block_tables + static_cast<std::int64_t>(
                                                         table_rows[token / columns_per_row]) *
                                                         table_stride;
    const __nv_bfloat16* src = keys + static_cast<std::int64_t>(token) * kHeadDim + d0;
    __nv_bfloat16* dst       = plane + indexer_offset(block_table, positions[token]) + d0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) { dst[i] = src[i]; }
    if (positions[token] % kBlock == 0 && lane < 4) {
        block_positions(plane, block_table, positions[token])[lane] = lane == 3 ? 0 :
            rope_positions[token + (rope_axes == 3 ? lane * rope_axis_stride : 0)];
    }
}

// One warp per new column: if the column completed a block, folds that block's raw keys into the
// separate block key — mean, RMSNorm, rope at the first position. A
// separate launch from the raw writes above: a block's cells can be written by warps of other
// CTAs, and only a launch boundary orders those against this read.
__global__ void qsa_append_fold_kernel(const std::int32_t* __restrict__ positions,
                                       const std::int32_t* __restrict__ table_rows,
                                       std::int32_t columns_per_row,
                                       const std::int32_t* __restrict__ block_tables,
                                       std::int32_t table_stride,
                                       const __nv_bfloat16* __restrict__ key_norm,
                                       __nv_bfloat16* __restrict__ plane, int tokens,
                                       int rotary_dim, float theta, float eps,
                                       int height_pairs, int width_pairs) {
    const int warp  = static_cast<int>(threadIdx.x) / kWarp;
    const int lane  = static_cast<int>(threadIdx.x) % kWarp;
    const int token = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (token >= tokens) { return; }
    const int position = positions[token];
    if ((position + 1) % kBlock != 0) { return; } // the block is still open
    const int d0                    = lane * kPerLane;
    const std::int32_t* block_table = block_tables + static_cast<std::int64_t>(
                                                         table_rows[token / columns_per_row]) *
                                                         table_stride;
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
    for (int i = 0; i < kPerLane; ++i) { value[i] *= scale * __bfloat162float(key_norm[d0 + i]); }

    // Split-half NeoX rope at the block's first position; the partner dimension lives in this
    // warp, so a shuffle fetches it.
    const int half = rotary_dim / 2;
    float partner[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int d      = d0 + i;
        const int mate   = d < half ? d + half : d - half;
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
        const int axis = pair % 3 == 1 && pair < 3 * height_pairs ? 1 :
                         pair % 3 == 2 && pair < 3 * width_pairs ? 2 : 0;
        const auto* axes = block_positions(plane, block_table, first);
        rope_sincos(static_cast<float>(axes[axis]), pair, rotary_dim, theta, &sin_a, &cos_a);
        value[i] = d < half ? value[i] * cos_a - partner[i] * sin_a
                            : value[i] * cos_a + partner[i] * sin_a;
    }
    __nv_bfloat16* block_dst = plane + indexer_offset<true>(block_table, first) + d0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) { block_dst[i] = __float2bfloat16(value[i]); }
}

// The first selection, kept as the control behind SUROGATE_SERVE_QSA_SELECT=bisect. One CTA per
// query row: scores every complete block, finds the budget's threshold, and writes the row's
// block bitmask. One warp handles one block at a time. A decode round has one row per sequence,
// so a long history was read by one SM.
template <int Heads>
__global__ void qsa_select_kernel(const __nv_bfloat16* __restrict__ q,
                                  const std::int32_t* __restrict__ positions,
                                  const std::int32_t* __restrict__ table_rows,
                                  std::int32_t columns_per_row, std::int32_t row_offset,
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
        block_tables +
        static_cast<std::int64_t>(table_rows[(row_offset + row) / columns_per_row]) * table_stride;
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
        const __nv_bfloat16* k = plane + indexer_offset<true>(block_table, b * kBlock) + d0;
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

// Scores a span of one row's complete blocks into `scores`. The spans spread a row over the GPU,
// so a decode round of a single query reads its long history on every SM. Eight lanes share one
// block key: lane `member` holds dims [8m, 8m+8) and [64+8m, 64+8m+8) of every head, which costs
// three shuffles per head and block where the control spends five.
template <int Heads>
__global__ void __launch_bounds__(kScoreThreads)
    qsa_score_kernel(const __nv_bfloat16* __restrict__ q,
                     const std::int32_t* __restrict__ positions,
                     const std::int32_t* __restrict__ table_rows, std::int32_t columns_per_row,
                     std::int32_t row_offset, const std::int32_t* __restrict__ block_tables,
                     std::int32_t table_stride, const __nv_bfloat16* __restrict__ plane,
                     int budget_blocks, int span, float* __restrict__ scores, int score_stride) {
    const int row    = static_cast<int>(blockIdx.y);
    const int scored = (positions[row] + 1) / kBlock;
    const int begin  = static_cast<int>(blockIdx.x) * span;
    if (scored <= budget_blocks || begin >= scored) { return; } // the cut writes the identity
    const int end    = min(begin + span, scored);
    const int lane   = static_cast<int>(threadIdx.x) % kWarp;
    const int warp   = static_cast<int>(threadIdx.x) / kWarp;
    const int group  = lane / kScoreLanes;
    const int member = lane % kScoreLanes;
    const std::int32_t* block_table =
        block_tables +
        static_cast<std::int64_t>(table_rows[(row_offset + row) / columns_per_row]) * table_stride;

    float qv[Heads][kScoreDims];
#pragma unroll
    for (int h = 0; h < Heads; ++h) {
        const __nv_bfloat16* src =
            q + (static_cast<std::int64_t>(row) * Heads + h) * kHeadDim + member * 8;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint4 packed = *reinterpret_cast<const uint4*>(src + half * (kHeadDim / 2));
            const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 v              = __bfloat1622float2(pairs[i]);
                qv[h][half * 8 + 2 * i]     = v.x;
                qv[h][half * 8 + 2 * i + 1] = v.y;
            }
        }
    }

    float* row_scores = scores + static_cast<std::int64_t>(row) * score_stride;
    // `begin` and `span` are whole steps, so every warp step starts a page.
    for (int first = begin + warp * kWarpBlocks; first < end; first += kScoreStep) {
        const __nv_bfloat16* page =
            plane + indexer_offset<true>(block_table, first * kBlock) + member * 8;
        uint4 keys[kScoreUnroll][2];
#pragma unroll
        for (int u = 0; u < kScoreUnroll; ++u) {
            const int slot           = u * kScoreGroups + group;
            const __nv_bfloat16* key = page + slot * kHeadDim;
            if (first + slot < end) {
                keys[u][0] = *reinterpret_cast<const uint4*>(key);
                keys[u][1] = *reinterpret_cast<const uint4*>(key + kHeadDim / 2);
            } else {
                keys[u][0] = keys[u][1] = make_uint4(0U, 0U, 0U, 0U);
            }
        }
#pragma unroll
        for (int u = 0; u < kScoreUnroll; ++u) {
            float kv[kScoreDims];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&keys[u][half]);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 v           = __bfloat1622float2(pairs[i]);
                    kv[half * 8 + 2 * i]     = v.x;
                    kv[half * 8 + 2 * i + 1] = v.y;
                }
            }
            float total = 0.0F;
#pragma unroll
            for (int h = 0; h < Heads; ++h) {
                float dot = 0.0F;
#pragma unroll
                for (int i = 0; i < kScoreDims; ++i) { dot += qv[h][i] * kv[i]; }
                dot = warp_sum<kScoreLanes>(dot);
                total += fmaxf(dot, 0.0F); // rectified per head, as in the reference
            }
            const int block = first + u * kScoreGroups + group;
            if (member == 0 && block < end) { row_scores[block] = total; }
        }
    }
}

// One CTA per query row. The scores are non-negative, so their bit patterns order like the
// floats, and four 8-bit radix passes find the budget-th largest exactly. Blocks above it are
// in; blocks equal to it fill the rest of the budget in block order, the reference's tie order.
__global__ void __launch_bounds__(kCutThreads)
    qsa_cut_kernel(const std::int32_t* __restrict__ positions, std::uint32_t* __restrict__ mask,
                   int words, int budget_blocks, const float* __restrict__ scores,
                   int score_stride) {
    const int row      = static_cast<int>(blockIdx.x);
    const int visible  = positions[row] + 1;
    const int scored   = visible / kBlock;
    const int touched  = (visible + kBlock - 1) / kBlock;
    const int tid      = static_cast<int>(threadIdx.x);
    const int lane     = tid % kWarp;
    const int warp     = tid / kWarp;
    const bool covered = scored <= budget_blocks; // the selection is the identity
    std::uint32_t* row_mask = mask + static_cast<std::int64_t>(row) * words;
    const auto* bits =
        reinterpret_cast<const unsigned*>(scores + static_cast<std::int64_t>(row) * score_stride);

    __shared__ unsigned histogram[kCutWarps][kRadixBins];
    __shared__ unsigned cut_prefix;
    __shared__ unsigned cut_rank;
    __shared__ int warp_ties[kCutWarps];

    unsigned threshold = 0U;
    unsigned ties_in   = 0U; // blocks equal to the threshold that the budget admits
    if (!covered) {
        unsigned prefix = 0U, decided = 0U, rank = static_cast<unsigned>(budget_blocks);
        for (int shift = 32 - kRadixBits; shift >= 0; shift -= kRadixBits) {
            for (int i = tid; i < kCutWarps * kRadixBins; i += kCutThreads) {
                histogram[i / kRadixBins][i % kRadixBins] = 0U;
            }
            __syncthreads();
            for (int base = 0; base < scored; base += kCutThreads * kCutUnroll) {
                unsigned value[kCutUnroll];
#pragma unroll
                for (int u = 0; u < kCutUnroll; ++u) {
                    const int b = base + u * kCutThreads + tid;
                    value[u]    = b < scored ? bits[b] : 0U;
                }
#pragma unroll
                for (int u = 0; u < kCutUnroll; ++u) {
                    const int b        = base + u * kCutThreads + tid;
                    const bool counted = b < scored && (value[u] & decided) == prefix;
                    const unsigned digit =
                        counted ? (value[u] >> shift) & (kRadixBins - 1) : kRadixBins;
                    // One shared atomic per distinct digit in the warp: the high digits of
                    // similar scores collide.
                    const unsigned peers = __match_any_sync(kFullWarpMask, digit);
                    if (counted && lane == __ffs(peers) - 1) {
                        atomicAdd(&histogram[warp][digit], static_cast<unsigned>(__popc(peers)));
                    }
                }
            }
            __syncthreads();
            for (int bin = tid; bin < kRadixBins; bin += kCutThreads) {
                unsigned total = 0U;
#pragma unroll
                for (int w = 0; w < kCutWarps; ++w) { total += histogram[w][bin]; }
                histogram[0][bin] = total;
            }
            __syncthreads();
            if (warp == 0) {
                // Lane l holds bins 255-8l down to 248-8l; the scan runs from the largest bin.
                unsigned count[kLaneBins];
                unsigned sum = 0U;
#pragma unroll
                for (int i = 0; i < kLaneBins; ++i) {
                    count[i] = histogram[0][kRadixBins - 1 - lane * kLaneBins - i];
                    sum += count[i];
                }
                unsigned running = sum;
#pragma unroll
                for (int offset = 1; offset < kWarp; offset <<= 1) {
                    const unsigned other = __shfl_up_sync(kFullWarpMask, running, offset);
                    if (lane >= offset) { running += other; }
                }
                const unsigned crossing = __ballot_sync(kFullWarpMask, running >= rank);
                if (lane == __ffs(crossing) - 1) {
                    unsigned above = running - sum;
#pragma unroll
                    for (int i = 0; i < kLaneBins; ++i) {
                        if (above + count[i] >= rank) {
                            const unsigned bin = kRadixBins - 1 - lane * kLaneBins - i;
                            cut_prefix         = prefix | (bin << shift);
                            cut_rank           = rank - above;
                            break;
                        }
                        above += count[i];
                    }
                }
            }
            __syncthreads();
            prefix = cut_prefix;
            rank   = cut_rank;
            decided |= static_cast<unsigned>(kRadixBins - 1) << shift;
        }
        threshold = prefix;
        ties_in   = rank;
    }

    // Each warp writes a contiguous run of words; lane l of word w reads block 32w + l.
    const int per_warp = (words + kCutWarps - 1) / kCutWarps;
    const int w_begin  = min(words, warp * per_warp);
    const int w_end    = min(words, w_begin + per_warp);
    const auto score_of = [&](int w) {
        const int b = w * kWarp + lane;
        return w < w_end && b < scored && !covered ? bits[b] : 0U;
    };
    unsigned tie_rank = 0U; // ties in the words before this one
    if (!covered) {
        unsigned ties = 0U;
        for (int w0 = w_begin; w0 < w_end; w0 += kCutUnroll) {
            unsigned value[kCutUnroll];
#pragma unroll
            for (int u = 0; u < kCutUnroll; ++u) { value[u] = score_of(w0 + u); }
#pragma unroll
            for (int u = 0; u < kCutUnroll; ++u) {
                const int b = (w0 + u) * kWarp + lane;
                ties += __popc(__ballot_sync(
                    kFullWarpMask, w0 + u < w_end && b < scored && value[u] == threshold));
            }
        }
        if (lane == 0) { warp_ties[warp] = static_cast<int>(ties); }
        __syncthreads();
        for (int w = 0; w < warp; ++w) { tie_rank += static_cast<unsigned>(warp_ties[w]); }
    }
    for (int w0 = w_begin; w0 < w_end; w0 += kCutUnroll) {
        unsigned value[kCutUnroll];
#pragma unroll
        for (int u = 0; u < kCutUnroll; ++u) { value[u] = score_of(w0 + u); }
#pragma unroll
        for (int u = 0; u < kCutUnroll; ++u) {
            const int w = w0 + u;
            if (w >= w_end) { break; } // uniform across the warp
            const int b         = w * kWarp + lane;
            const bool complete = b < scored;
            unsigned word =
                __ballot_sync(kFullWarpMask, complete && (covered || value[u] > threshold)) |
                __ballot_sync(kFullWarpMask, !complete && b < touched); // the open tail
            unsigned tied =
                __ballot_sync(kFullWarpMask, complete && !covered && value[u] == threshold);
            const unsigned count = static_cast<unsigned>(__popc(tied));
            const unsigned quota = ties_in > tie_rank ? ties_in - tie_rank : 0U;
            if (quota >= count) {
                word |= tied;
            } else {
                for (unsigned i = 0; i < quota; ++i) { // the lowest `quota` tied blocks
                    word |= tied & (0U - tied);
                    tied &= tied - 1U;
                }
            }
            tie_rank += count;
            if (lane == 0) { row_mask[w] = word; }
        }
    }
}

int multiprocessors() {
    int device = 0, sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return 1;
    }
    return std::max(sms, 1);
}

// Blocks per scoring CTA: enough CTAs to cover the GPU twice, each scoring at least four steps.
int score_span(int rows, std::int64_t blocks) {
    const std::int64_t wanted = std::max<std::int64_t>(1, (2 * multiprocessors() + rows - 1) / rows);
    const std::int64_t most   = std::max<std::int64_t>(1, (blocks + kMinScoreSpan - 1) / kMinScoreSpan);
    const std::int64_t splits = std::min(wanted, most);
    const std::int64_t span   = (blocks + splits - 1) / splits;
    return static_cast<int>((span + kScoreStep - 1) / kScoreStep * kScoreStep);
}

// One CTA per tile of query rows: the tile's mask words are OR-ed in shared memory, then each
// thread lists the set bits of a contiguous run of words from its place in the CTA's prefix
// sum, so the list ascends.
constexpr int kUnionThreads = 256;

__global__ void __launch_bounds__(kUnionThreads)
    qsa_tile_union_kernel(const std::uint32_t* __restrict__ mask, int words, int rows,
                          std::int32_t* __restrict__ blocks, int stride,
                          std::int32_t* __restrict__ counts) {
    extern __shared__ std::uint32_t union_words[];
    __shared__ int warp_counts[kUnionThreads / kWarp];
    const int tile  = static_cast<int>(blockIdx.x);
    const int first = tile * kQsaTileRows;
    const int last  = min(first + kQsaTileRows, rows);
    const int tid   = static_cast<int>(threadIdx.x);
    for (int w = tid; w < words; w += kUnionThreads) {
        std::uint32_t bits = 0U;
        for (int row = first; row < last; ++row) {
            bits |= mask[static_cast<std::int64_t>(row) * words + w];
        }
        union_words[w] = bits;
    }
    __syncthreads();
    const int per     = (words + kUnionThreads - 1) / kUnionThreads;
    const int w_begin = min(tid * per, words);
    const int w_end   = min(w_begin + per, words);
    int count = 0;
    for (int w = w_begin; w < w_end; ++w) { count += __popc(union_words[w]); }
    const int lane = tid % kWarp;
    const int warp = tid / kWarp;
    int inclusive  = count;
#pragma unroll
    for (int offset = 1; offset < kWarp; offset <<= 1) {
        const int below = __shfl_up_sync(kFullWarpMask, inclusive, offset);
        if (lane >= offset) { inclusive += below; }
    }
    if (lane == kWarp - 1) { warp_counts[warp] = inclusive; }
    __syncthreads();
    int slot = inclusive - count, total = 0;
    for (int w = 0; w < kUnionThreads / kWarp; ++w) {
        if (w < warp) { slot += warp_counts[w]; }
        total += warp_counts[w];
    }
    std::int32_t* list = blocks + static_cast<std::int64_t>(tile) * stride;
    for (int w = w_begin; w < w_end; ++w) {
        for (std::uint32_t bits = union_words[w]; bits != 0U; bits &= bits - 1U) {
            list[slot++] = w * kWarp + __ffs(static_cast<int>(bits)) - 1;
        }
    }
    if (tid == 0) { counts[tile] = total; }
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

std::int32_t qsa_select_row_tile(std::int32_t rows, std::int32_t keys,
                                 const QsaIndexerGeometry& geometry) noexcept {
    // The block scores are per query row, so the scratch would grow as rows x blocks: 512 MB for
    // a 2,048-column chunk over a 262k context. The rows are independent, so the selection runs
    // in tiles and the plan reserves one tile.
    const std::int64_t blocks = (keys + geometry.block - 1) / geometry.block;
    if (blocks <= 0) { return rows > 0 ? rows : 1; }
    const std::int64_t tile = kSelectScratchBytes / (blocks * static_cast<std::int64_t>(sizeof(float)));
    const std::int64_t clamped = std::max<std::int64_t>(1, std::min<std::int64_t>(tile, rows));
    return static_cast<std::int32_t>(clamped);
}

std::int32_t qsa_tile_union_stride(std::int32_t keys, std::int32_t block) {
    return qsa_block_mask_words(keys, block) * kWarp;
}

std::size_t qsa_tile_union_bytes(std::int32_t rows, std::int32_t keys, std::int32_t block) {
    const auto round_up = [](std::size_t bytes) { return (bytes + 255U) / 256U * 256U; };
    const auto tiles    = static_cast<std::size_t>((std::max(rows, 1) + kQsaTileRows - 1) / kQsaTileRows);
    return round_up(static_cast<std::size_t>(qsa_tile_union_stride(keys, block)) * tiles *
                    sizeof(std::int32_t)) +
           round_up(tiles * sizeof(std::int32_t));
}

void qsa_tile_union(const Tensor& mask, Tensor& blocks, Tensor& counts, cudaStream_t stream) {
    require(mask.dtype == DType::I32 && blocks.dtype == DType::I32 && counts.dtype == DType::I32,
            "tile union tensors must be I32");
    const int words = static_cast<int>(mask.ne[0]);
    const int rows  = static_cast<int>(mask.ne[1]);
    const int tiles = (rows + kQsaTileRows - 1) / kQsaTileRows;
    require(blocks.ne[0] == words * kWarp && blocks.ne[1] == tiles && counts.ne[0] == tiles,
            "tile union lists must be [words * 32, tiles] and counts [tiles]");
    if (tiles == 0) { return; }
    const auto smem = static_cast<int>(words * sizeof(std::uint32_t));
    require(smem <= 48 * 1024, "tile union: the history's mask row exceeds 48 KiB");
    qsa_tile_union_kernel<<<tiles, kUnionThreads, smem, stream>>>(
        static_cast<const std::uint32_t*>(mask.data), words, rows,
        static_cast<std::int32_t*>(blocks.data), static_cast<int>(blocks.ne[0]),
        static_cast<std::int32_t*>(counts.data));
}

std::size_t qsa_indexer_select_workspace_capacity_bytes(std::int32_t rows, std::int32_t keys,
                                                        const QsaIndexerGeometry& geometry) {
    require(rows >= 0 && keys >= 0 && geometry.block > 0, "workspace geometry must be positive");
    const std::size_t blocks =
        static_cast<std::size_t>((keys + geometry.block - 1) / geometry.block);
    const auto tile = static_cast<std::size_t>(qsa_select_row_tile(rows, keys, geometry));
    return tile * blocks * sizeof(float) + 256U;
}

void qsa_indexer_append(const Tensor& keys, const Tensor& positions, const Tensor& table_rows,
                        std::int32_t columns_per_row, const Tensor& key_norm,
                        const QsaIndexerGeometry& geometry, PagedKVBatchLayerView cache,
                        cudaStream_t stream, const Tensor& rope_positions) {
    require(geometry.head_dim == kHeadDim && geometry.block == kBlock,
            "only the 128-wide, 4-cell indexer is registered");
    require(geometry.rotary_dim > 0 && geometry.rotary_dim <= kHeadDim &&
                (geometry.rotary_dim % 2) == 0,
            "rotary_dim must be even and within the head");
    require(keys.dtype == DType::BF16 && key_norm.dtype == DType::BF16, "keys must be BF16");
    require(positions.dtype == DType::I32, "positions must be I32");
    require(keys.ne[0] == kHeadDim, "keys must be [head_dim, T]");
    require(cache.indexer_pages.data != nullptr, "the cache carries no indexer plane");
    require(cache.indexer_pages.dtype == DType::BF16 &&
                cache.indexer_pages.ne[0] == kQsaIndexerStorageHeadDim &&
                cache.indexer_pages.ne[1] == kPagedKVPageSize,
            "indexer pages must hold separate raw and pooled keys");
    require(cache.block_tables.data != nullptr, "the cache view has no block tables");
    const int tokens = static_cast<int>(keys.ne[1]);
    if (tokens == 0) { return; }
    require(positions.ne[0] == tokens, "positions must match the columns");
    const Tensor& rope = rope_positions.data ? rope_positions : positions;
    require(rope.dtype == DType::I32 && rope.ne[0] == tokens &&
                (rope.ne[1] == 1 || rope.ne[1] == 3) && rope.ne[2] == 1 && rope.ne[3] == 1 &&
                rope.nb[0] == sizeof(std::int32_t), "rope positions must be I32 [T] or [T,3]");
    require(geometry.mrope_height >= 0 && geometry.mrope_width >= 0 &&
                geometry.mrope_height <= (geometry.rotary_dim / 2 + 1) / 3 &&
                geometry.mrope_width <= geometry.rotary_dim / 6, "invalid mRoPE sections");
    const int blocks = (tokens + kWarps - 1) / kWarps;
    require(table_rows.data != nullptr && table_rows.dtype == DType::I32,
            "table rows must be a non-null I32 vector");
    require(columns_per_row > 0 && (tokens + columns_per_row - 1) / columns_per_row <=
                                       static_cast<std::int32_t>(table_rows.ne[0]),
            "table rows must cover every column's sequence");
    qsa_append_raw_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(keys.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(table_rows.data), columns_per_row,
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        static_cast<__nv_bfloat16*>(cache.indexer_pages.data), tokens,
        static_cast<const std::int32_t*>(rope.data), rope.nb[1] / sizeof(std::int32_t), rope.ne[1]);
    qsa_append_fold_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(table_rows.data), columns_per_row,
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        static_cast<const __nv_bfloat16*>(key_norm.data),
        static_cast<__nv_bfloat16*>(cache.indexer_pages.data), tokens, geometry.rotary_dim,
        geometry.rope_theta, geometry.rms_eps, geometry.mrope_height, geometry.mrope_width);
}

void qsa_indexer_select(const Tensor& q, const Tensor& positions, const Tensor& table_rows,
                        std::int32_t columns_per_row, const QsaIndexerGeometry& geometry,
                        PagedKVBatchLayerView cache, std::int32_t keys, WorkspaceArena& workspace,
                        Tensor& mask, cudaStream_t stream) {
    require(geometry.head_dim == kHeadDim && geometry.block == kBlock,
            "only the 128-wide, 4-cell indexer is registered");
    require(q.dtype == DType::BF16 && positions.dtype == DType::I32 &&
                table_rows.dtype == DType::I32 && mask.dtype == DType::I32,
            "select tensor dtypes are wrong");
    require(cache.indexer_pages.data != nullptr, "the cache carries no indexer plane");
    require(cache.indexer_pages.dtype == DType::BF16 &&
                cache.indexer_pages.ne[0] == kQsaIndexerStorageHeadDim &&
                cache.indexer_pages.ne[1] == kPagedKVPageSize,
            "indexer pages must hold separate raw and pooled keys");
    require(q.ne[0] == kHeadDim && q.ne[1] == geometry.heads, "q must be [head_dim, heads, rows]");
    const int rows = static_cast<int>(q.ne[2]);
    if (rows == 0) { return; }
    const int words = qsa_block_mask_words(keys, geometry.block);
    require(mask.ne[0] == words && mask.ne[1] == rows, "mask must be [words, rows]");
    require(positions.ne[0] == rows, "positions must match the query rows");
    require(table_rows.data != nullptr && columns_per_row > 0 &&
                (rows + columns_per_row - 1) / columns_per_row <=
                    static_cast<std::int32_t>(table_rows.ne[0]),
            "table rows must cover every query row's sequence");
    const int blocks  = (keys + geometry.block - 1) / geometry.block;
    const int tile    = qsa_select_row_tile(rows, keys, geometry);
    Tensor scores     = workspace.alloc(DType::FP32, {blocks, tile});
    const auto stride = static_cast<int>(blocks);
    const int budget  = geometry.top_k / geometry.block;
    // SUROGATE_SERVE_QSA_SELECT=bisect runs the one-CTA-per-row control.
    static const bool bisect = [] {
        const char* value = std::getenv("SUROGATE_SERVE_QSA_SELECT");
        return value != nullptr && std::string(value) == "bisect";
    }();
    const int span = score_span(std::min(tile, rows), blocks);
    // One tile of query rows at a time: the rows are independent, and the per-row scratch is
    // what would otherwise scale with the context length.
    const auto launch = [&](auto heads, int first, int count) {
        constexpr int kHeads = decltype(heads)::value;
        const auto* q_rows = static_cast<const __nv_bfloat16*>(q.data) +
                             static_cast<std::int64_t>(first) * kHeadDim * geometry.heads;
        const auto* row_positions = static_cast<const std::int32_t*>(positions.data) + first;
        const auto* tables        = static_cast<const std::int32_t*>(cache.block_tables.data);
        const auto* plane         = static_cast<const __nv_bfloat16*>(cache.indexer_pages.data);
        auto* row_mask = static_cast<std::uint32_t*>(mask.data) + static_cast<std::int64_t>(first) * words;
        auto* row_scores = static_cast<float*>(scores.data);
        if (bisect) {
            qsa_select_kernel<kHeads><<<count, kThreads, 0, stream>>>(
                q_rows, row_positions, static_cast<const std::int32_t*>(table_rows.data),
                columns_per_row, first, tables, cache.block_tables.ne[0], plane, row_mask, words,
                budget, row_scores, stride);
            return;
        }
        const dim3 grid((blocks + span - 1) / span, count);
        qsa_score_kernel<kHeads><<<grid, kScoreThreads, 0, stream>>>(
            q_rows, row_positions, static_cast<const std::int32_t*>(table_rows.data),
            columns_per_row, first, tables, cache.block_tables.ne[0], plane, budget, span,
            row_scores, stride);
        qsa_cut_kernel<<<count, kCutThreads, 0, stream>>>(row_positions, row_mask, words, budget,
                                                         row_scores, stride);
    };
    for (int first = 0; first < rows; first += tile) {
        const int count = std::min(tile, rows - first);
        switch (geometry.heads) {
        case 4: launch(std::integral_constant<int, 4>{}, first, count); break;
        case 8: launch(std::integral_constant<int, 8>{}, first, count); break;
        default: throw std::invalid_argument("qsa_indexer: unregistered indexer head count");
        }
    }
}

} // namespace sinfer::ops
