#pragma once

// One launch per adapter site: the shrink and the expand of up to three
// projections that share an input.
//
// The two-kernel path cost one launch for A·x and another for B·(A·x), per
// module -- q, k and v alone were six launches, and a round's delta work was
// dominated by kernel entry, not math: an armed round (adapters resident, no
// token selecting one) paid ~2 us per launch just to read an id and exit.
// Here the low vector lives in shared memory instead of a scratch tensor, the
// site is one launch, and q/k/v ride together because they read the same
// hidden column.
//
// Every block recomputes the low vectors it needs (stage 1) rather than
// syncing with other blocks: the redundant A and x reads all hit L2 -- a few
// megabytes at the largest site -- which is cheaper than a grid-wide barrier
// or a second launch. Reductions are a fixed warp-shuffle order and fp32
// accumulation, so the summation order is identical on every launch: eager
// calls and captured replays agree bit for bit.

#include "ops/kernel/lora_fused_limits.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace sinfer::ops {

inline constexpr int kLoraFusedThreads  = 128;
inline constexpr int kLoraFusedMaxPairs = kLoraFusedPairLimit;
inline constexpr int kLoraFusedMaxRank  = kLoraFusedRankLimit;

struct LoraFusedPair {
    const __nv_bfloat16* a = nullptr; ///< [slots, rank, k] stacked
    const __nv_bfloat16* b = nullptr; ///< [slots, n, rank] stacked
    __nv_bfloat16* out     = nullptr; ///< [n, tokens], accumulated in place
    std::int64_t a_stride  = 0;       ///< elements between adapter slots of a
    std::int64_t b_stride  = 0;       ///< elements between adapter slots of b
    std::int32_t n         = 0;
    std::int64_t out_stride = 0;
    std::int32_t rank      = 0;       ///< the bank's padded rank
    const float* gain = nullptr;
    const float* bias = nullptr;
    std::int32_t row_begin = 0;       ///< prefix offset in the concatenated rows
};

struct LoraFusedParams {
    LoraFusedPair pairs[kLoraFusedMaxPairs];
    const __nv_bfloat16* x    = nullptr; ///< [k, tokens], shared by every pair
    const std::int32_t* ids   = nullptr; ///< per-token adapter, or null
    const std::int32_t* uniform = nullptr; ///< device cell, when ids is null
    std::int32_t pair_count   = 0;
    std::int32_t total_rows   = 0;
    std::int32_t k            = 0;
    std::int32_t tokens       = 0;
};

__device__ __forceinline__ float lora_fused_dot2(unsigned a_bits, unsigned x_bits) {
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a_bits));
    const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&x_bits));
    return a.x * x.x + a.y * x.y;
}

__global__ __launch_bounds__(kLoraFusedThreads) void lora_fused_delta_kernel(LoraFusedParams p) {
    const int token = static_cast<int>(blockIdx.y);
    if (token >= p.tokens) { return; }
    const int adapter =
        p.ids != nullptr ? p.ids[token] : (p.uniform != nullptr ? *p.uniform : -1);
    // Uniform per block: every thread takes the same exit, so the base-model
    // case is one id read and out.
    if (adapter < 0) { return; }

    __shared__ float low[kLoraFusedMaxPairs][kLoraFusedMaxRank];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const __nv_bfloat16* x_col = p.x + static_cast<std::int64_t>(token) * p.k;
    const bool vector_k = (p.k & 7) == 0 &&
                          (reinterpret_cast<std::uintptr_t>(x_col) & 15U) == 0;

    int total_rank = 0;
#pragma unroll
    for (int q = 0; q < kLoraFusedMaxPairs; ++q) {
        if (q < p.pair_count) { total_rank += p.pairs[q].rank; }
    }
    // Stage 1: the low vectors, rows striped over the four warps.
    for (int rr = warp; rr < total_rank; rr += kLoraFusedThreads / 32) {
        int q = 0, r = rr;
        while (r >= p.pairs[q].rank) { r -= p.pairs[q].rank; ++q; }
        const LoraFusedPair& pair = p.pairs[q];
        const __nv_bfloat16* a_row = pair.a +
                                     static_cast<std::int64_t>(adapter) * pair.a_stride +
                                     static_cast<std::int64_t>(r) * p.k;
        float sum = 0.0F;
        if (vector_k && (reinterpret_cast<std::uintptr_t>(a_row) & 15U) == 0) {
            const auto* a_vec = reinterpret_cast<const uint4*>(a_row);
            const auto* x_vec = reinterpret_cast<const uint4*>(x_col);
            for (int i = lane; i < (p.k >> 3); i += 32) {
                const uint4 a8 = a_vec[i];
                const uint4 x8 = x_vec[i];
                sum += lora_fused_dot2(a8.x, x8.x) + lora_fused_dot2(a8.y, x8.y) +
                       lora_fused_dot2(a8.z, x8.z) + lora_fused_dot2(a8.w, x8.w);
            }
        } else {
            for (int i = lane; i < p.k; i += 32) {
                sum += __bfloat162float(a_row[i]) * __bfloat162float(x_col[i]);
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
        }
        if (lane == 0) { low[q][r] = sum; }
    }
    __syncthreads();

    // Stage 2: this block's slice of the concatenated output rows.
    const int row = static_cast<int>(blockIdx.x) * kLoraFusedThreads +
                    static_cast<int>(threadIdx.x);
    if (row >= p.total_rows) { return; }
    int q = 0;
    while (q + 1 < p.pair_count && row >= p.pairs[q + 1].row_begin) { ++q; }
    const LoraFusedPair& pair = p.pairs[q];
    const int local           = row - pair.row_begin;
    const __nv_bfloat16* b_row = pair.b +
                                 static_cast<std::int64_t>(adapter) * pair.b_stride +
                                 static_cast<std::int64_t>(local) * pair.rank;
    float acc = 0.0F;
    for (int r = 0; r < pair.rank; ++r) {
        acc += __bfloat162float(b_row[r]) * low[q][r];
    }
    __nv_bfloat16* cell = pair.out + static_cast<std::int64_t>(token) * pair.out_stride + local;
    const auto index = static_cast<std::int64_t>(adapter) * pair.n + local;
    const float gain = pair.gain ? 1.0F + pair.gain[index] : 1.0F;
    const float bias = pair.bias ? pair.bias[index] : 0.0F;
    *cell = __float2bfloat16((__bfloat162float(*cell) + acc) * gain + bias);
}

/// The split flavor of the same site, for geometries where the one-launch
/// kernel's redundancy is too expensive. Every stage-2 block of the fused
/// kernel recomputes stage 1, and that redundant traffic is
/// blocks x total_rank x k -- fine for a small site, megabytes per token for a
/// wide one. Here stage 1 runs once into scratch and stage 2 reads it back:
/// two launches, no redundancy. The launcher picks per site by that product,
/// so armed rounds keep the fewest launches where it is cheap and selected
/// tokens keep linear work where it is not.
__global__ __launch_bounds__(kLoraFusedThreads) void lora_split_shrink_kernel(
    LoraFusedParams p, __nv_bfloat16* __restrict__ low) {
    const int token = static_cast<int>(blockIdx.y);
    if (token >= p.tokens) { return; }
    const int adapter =
        p.ids != nullptr ? p.ids[token] : (p.uniform != nullptr ? *p.uniform : -1);
    const int rr = static_cast<int>(blockIdx.x);
    int total_rank = 0;
#pragma unroll
    for (int q = 0; q < kLoraFusedMaxPairs; ++q) {
        if (q < p.pair_count) { total_rank += p.pairs[q].rank; }
    }
    if (rr >= total_rank) { return; }
    if (adapter < 0) {
        if (threadIdx.x == 0) {
            low[static_cast<std::int64_t>(token) * total_rank + rr] = __float2bfloat16(0.0F);
        }
        return;
    }
    int q = 0, r = rr;
    while (r >= p.pairs[q].rank) { r -= p.pairs[q].rank; ++q; }
    const LoraFusedPair& pair  = p.pairs[q];
    const __nv_bfloat16* a_row = pair.a +
                                 static_cast<std::int64_t>(adapter) * pair.a_stride +
                                 static_cast<std::int64_t>(r) * p.k;
    const __nv_bfloat16* x_col = p.x + static_cast<std::int64_t>(token) * p.k;
    float sum = 0.0F;
    if ((p.k & 7) == 0 && (reinterpret_cast<std::uintptr_t>(a_row) & 15U) == 0 &&
        (reinterpret_cast<std::uintptr_t>(x_col) & 15U) == 0) {
        const auto* a_vec = reinterpret_cast<const uint4*>(a_row);
        const auto* x_vec = reinterpret_cast<const uint4*>(x_col);
        for (int i = static_cast<int>(threadIdx.x); i < (p.k >> 3); i += kLoraFusedThreads) {
            const uint4 a8 = a_vec[i];
            const uint4 x8 = x_vec[i];
            sum += lora_fused_dot2(a8.x, x8.x) + lora_fused_dot2(a8.y, x8.y) +
                   lora_fused_dot2(a8.z, x8.z) + lora_fused_dot2(a8.w, x8.w);
        }
    } else {
        for (int i = static_cast<int>(threadIdx.x); i < p.k; i += kLoraFusedThreads) {
            sum += __bfloat162float(a_row[i]) * __bfloat162float(x_col[i]);
        }
    }
    __shared__ float partial[kLoraFusedThreads];
    partial[threadIdx.x] = sum;
    __syncthreads();
#pragma unroll
    for (int stride = kLoraFusedThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        low[static_cast<std::int64_t>(token) * total_rank + rr] = __float2bfloat16(partial[0]);
    }
}

__global__ __launch_bounds__(kLoraFusedThreads) void lora_split_expand_kernel(
    LoraFusedParams p, const __nv_bfloat16* __restrict__ low) {
    const int token = static_cast<int>(blockIdx.y);
    if (token >= p.tokens) { return; }
    const int adapter =
        p.ids != nullptr ? p.ids[token] : (p.uniform != nullptr ? *p.uniform : -1);
    if (adapter < 0) { return; }
    const int row = static_cast<int>(blockIdx.x) * kLoraFusedThreads +
                    static_cast<int>(threadIdx.x);
    if (row >= p.total_rows) { return; }
    int total_rank = 0;
    int rank_begin[kLoraFusedMaxPairs];
#pragma unroll
    for (int q = 0; q < kLoraFusedMaxPairs; ++q) {
        rank_begin[q] = total_rank;
        if (q < p.pair_count) { total_rank += p.pairs[q].rank; }
    }
    int q = 0;
    while (q + 1 < p.pair_count && row >= p.pairs[q + 1].row_begin) { ++q; }
    const LoraFusedPair& pair    = p.pairs[q];
    const int local              = row - pair.row_begin;
    const __nv_bfloat16* b_row   = pair.b +
                                 static_cast<std::int64_t>(adapter) * pair.b_stride +
                                 static_cast<std::int64_t>(local) * pair.rank;
    const __nv_bfloat16* low_col = low + static_cast<std::int64_t>(token) * total_rank +
                                   rank_begin[q];
    float acc = 0.0F;
    for (int r = 0; r < pair.rank; ++r) {
        acc += __bfloat162float(b_row[r]) * __bfloat162float(low_col[r]);
    }
    __nv_bfloat16* cell = pair.out + static_cast<std::int64_t>(token) * pair.out_stride + local;
    const auto index = static_cast<std::int64_t>(adapter) * pair.n + local;
    const float gain = pair.gain ? 1.0F + pair.gain[index] : 1.0F;
    const float bias = pair.bias ? pair.bias[index] : 0.0F;
    *cell = __float2bfloat16((__bfloat162float(*cell) + acc) * gain + bias);
}

} // namespace sinfer::ops
