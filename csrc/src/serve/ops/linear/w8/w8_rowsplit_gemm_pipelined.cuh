#pragma once

// W8G32 RowSplit split-K MMA with a multi-stage cp.async pipeline, for the batch-consistent route
// (launch_w8_consistent): the arithmetic of w8_rowsplit_medium_t_splitk_kernel with four K-split
// warps and one N group -- each weight rounded once to BF16 from its FP16 group scale, each
// warp's K slices in the same mma order, the same split combine -- so every output is that
// kernel's bits. What differs is the loading. That kernel stages one 256-wide K group of codes and
// activations at a time and loads the group's scales in its MMA loop, so a CTA has at most 8 KiB
// in flight and waits twice per group; on an H100 a vocabulary head (151936 x 4096) read its
// weights at ~1.9 TB/s. Here `Stages` groups are in flight a CTA, scales included (a row's scales
// over one group are 16 contiguous bytes), and the CTA computes on the oldest while the rest land.
//
// `RowGroups` sets of four warps share a CTA's staged activations, each set over its own rows:
// a CTA stages its columns' activations for the whole K, so at 32 columns 32-row CTAs read
// twice the weight's bytes from L2, and 64-row CTAs once. A row's warp, K slices and mma order
// do not depend on the grouping.
//
// Shared memory is dynamic: Stages x (codes, activations, scales) of one group; after the K loop
// it holds the split partial sums.

#include "ops/linear/w8/w8_rowsplit_output.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

template <int TileCols, int Stages, int RowTiles, int RowGroups = 1>
struct W8PipelinedLayout {
    static constexpr int kKSplits    = 4;
    static constexpr int kTileK      = 64;
    static constexpr int kGroupK     = kKSplits * kTileK;
    static constexpr int kWarpRows   = 16 * RowTiles;
    static constexpr int kRowsPerCta = kWarpRows * RowGroups;
    static constexpr int kThreads    = kKSplits * RowGroups * 32;
    static constexpr int kScaleBytes = kGroupK / 32 * 2; // a row's FP16 scales over one group
    static constexpr std::size_t kCodeBytes  = std::size_t{kRowsPerCta} * kGroupK;
    static constexpr std::size_t kActBytes   = std::size_t{kKSplits} * TileCols * kTileK * 2;
    static constexpr std::size_t kScaleStage = std::size_t{kRowsPerCta} * kScaleBytes;
    static constexpr std::size_t kCodesAt    = 0;
    static constexpr std::size_t kActAt      = kCodesAt + Stages * kCodeBytes;
    static constexpr std::size_t kScalesAt   = kActAt + Stages * kActBytes;
    static constexpr std::size_t kBytes      = kScalesAt + Stages * kScaleStage;
    // The split partial sums (one float4 a lane a fragment a warp) fit in the stages.
    static_assert(std::size_t{kKSplits} * RowGroups * (TileCols / 8) * RowTiles * 32 * 16 <= kBytes);
    static_assert(kScaleBytes == 16 && kCodeBytes % 16 == 0 && kActBytes % 16 == 0);
    static_assert(kRowsPerCta <= kThreads);
};

template <int TileCols, int Stages, int RowTiles, int RowGroups = 1>
__global__ __launch_bounds__(W8PipelinedLayout<TileCols, Stages, RowTiles, RowGroups>::kThreads)
void w8_rowsplit_pipelined_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, W8ContiguousOutput output, int active_cols,
    int hidden) {
    using Layout               = W8PipelinedLayout<TileCols, Stages, RowTiles, RowGroups>;
    constexpr int KSplits      = Layout::kKSplits;
    constexpr int kTileK       = Layout::kTileK;
    constexpr int kGroupK      = Layout::kGroupK;
    constexpr int kRowsPerCta  = Layout::kRowsPerCta;
    constexpr int kThreads     = Layout::kThreads;
    constexpr int kWarpCols    = TileCols;
    constexpr int kNt          = kWarpCols / 8;
    constexpr int kFragments   = kNt * RowTiles;
    constexpr unsigned kMask   = 0xffffffffu;
    static_assert(RowTiles == 1 || RowTiles == 2);
    static_assert(kWarpCols % 8 == 0);
    static_assert(Stages >= 2 && Stages <= 6);
    static_assert(RowGroups == 1 || RowGroups == 2);

    extern __shared__ __align__(16) unsigned char smem[];
    using CodeTile = std::uint8_t[kRowsPerCta][kGroupK];
    using ActTile  = __nv_bfloat16[KSplits][kWarpCols * kTileK];
    auto* code_stages  = reinterpret_cast<CodeTile*>(smem + Layout::kCodesAt);
    auto* act_stages   = reinterpret_cast<ActTile*>(smem + Layout::kActAt);
    auto* scale_stages = smem + Layout::kScalesAt;

    const int kGroups      = hidden / kGroupK;
    const int column_begin = static_cast<int>(blockIdx.y) * TileCols;
    x += static_cast<std::int64_t>(column_begin) * hidden;
    active_cols = min(TileCols, active_cols - column_begin);

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int k_split    = warp % KSplits;
    const int warp_row0  = warp / KSplits * Layout::kWarpRows; // the warp's rows within the CTA
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int local_cols = active_cols <= 0 ? 0 : active_cols;
    const int cta_row0   = static_cast<int>(blockIdx.x) * kRowsPerCta;
    const int warp_koff  = k_split * kTileK;

    // One group into stage `buf`: the CTA's codes, scales and activations (a column's 256 K as
    // one K slice a split). A group past the end commits an empty group, so the wait counts hold.
    const auto stage = [&](int group_index, int buf) {
        if (group_index < kGroups) {
            const int group_k0    = group_index * kGroupK;
            constexpr int kChunks = kGroupK / 16;
            auto& code_shared     = code_stages[buf];
            for (int item = tid; item < kRowsPerCta * kChunks; item += kThreads) {
                const int row            = item / kChunks;
                const int chunk          = item - row * kChunks;
                const int swizzled_chunk = chunk ^ (row & 7);
                cp_async<16, Cache::cg>(&code_shared[row][swizzled_chunk * 16],
                                        codes + static_cast<std::int64_t>(cta_row0 + row) * hidden +
                                            group_k0 + chunk * 16);
            }
            if (tid < kRowsPerCta) {
                cp_async<16, Cache::cg>(
                    scale_stages + buf * Layout::kScaleStage + tid * Layout::kScaleBytes,
                    scales + (static_cast<std::int64_t>(cta_row0 + tid) * (hidden / 32) + group_k0 / 32) * 2);
            }
            constexpr int kActChunks = kGroupK / 8;
            for (int item = tid; item < local_cols * kActChunks; item += kThreads) {
                const int col   = item / kActChunks;
                const int chunk = item - col * kActChunks; // 8 activations
                const int split = chunk / (kTileK / 8);
                const int k8    = chunk - split * (kTileK / 8);
                auto* dst = &act_stages[buf][split][col * kTileK + w8_small_t_swizzle_64(col, k8 * 8)];
                cp_async<16, Cache::cg>(dst, &x[static_cast<std::int64_t>(col) * hidden + group_k0 + chunk * 8]);
            }
        }
        cp_commit();
    };

    const int b_rin  = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    float acc[kFragments][4];
#pragma unroll
    for (int ni = 0; ni < kFragments; ++ni) {
        acc[ni][0] = 0.0f;
        acc[ni][1] = 0.0f;
        acc[ni][2] = 0.0f;
        acc[ni][3] = 0.0f;
    }

#pragma unroll
    for (int s = 0; s < Stages - 1; ++s) { stage(s, s); }

    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int buf = group_index % Stages;
        stage(group_index + Stages - 1, (group_index + Stages - 1) % Stages);
        cp_wait<Stages - 1>();
        __syncthreads();

        const auto& code_shared = code_stages[buf];
        const auto& b_shared    = act_stages[buf];
        const unsigned char* scale_stage = scale_stages + buf * Layout::kScaleStage;
        // The row tiles' scales over this warp's K slice, then each K step's weight fragments for
        // every row tile, so one activation fragment feeds them all. Each accumulator still
        // takes its K steps in the medium-T kernel's order.
        unsigned top_scale_pairs[RowTiles], bot_scale_pairs[RowTiles];
#pragma unroll
        for (int mt = 0; mt < RowTiles; ++mt) {
            unsigned lane_scale_pair = 0;
            if (lid < 2) {
                const int scale_row = warp_row0 + mt * 16 + gid + lid * 8;
                lane_scale_pair     = *reinterpret_cast<const unsigned*>(
                    scale_stage + scale_row * Layout::kScaleBytes + k_split * 4);
            }
            top_scale_pairs[mt] = __shfl_sync(kMask, lane_scale_pair, lane & ~3);
            bot_scale_pairs[mt] = __shfl_sync(kMask, lane_scale_pair, (lane & ~3) + 1);
        }
        const auto load_code_pair = [&](int code_row, int col) {
            const int chunk  = (warp_koff + col) >> 4;
            const int offset = (chunk ^ (code_row & 7)) * 16 + (col & 15);
            return static_cast<unsigned>(
                *reinterpret_cast<const unsigned short*>(&code_shared[code_row][offset]));
        };

#pragma unroll
        for (int group = 0; group < 2; ++group) {
#pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                const int ks       = group * 2 + ki;
                const int code_col = ks * 16 + lid * 2;
                unsigned af[RowTiles][4];
#pragma unroll
                for (int mt = 0; mt < RowTiles; ++mt) {
                    const unsigned top_bits =
                        group == 0 ? top_scale_pairs[mt] & 0xffffu : top_scale_pairs[mt] >> 16;
                    const unsigned bot_bits =
                        group == 0 ? bot_scale_pairs[mt] & 0xffffu : bot_scale_pairs[mt] >> 16;
                    const float top_scale = __half2float(__ushort_as_half(top_bits));
                    const float bot_scale = __half2float(__ushort_as_half(bot_bits));
                    const int top_row     = warp_row0 + mt * 16 + gid;
                    af[mt][0] = w8_small_t_bf16_pair_from_s8(load_code_pair(top_row, code_col), top_scale);
                    af[mt][1] = w8_small_t_bf16_pair_from_s8(load_code_pair(top_row + 8, code_col), bot_scale);
                    af[mt][2] = w8_small_t_bf16_pair_from_s8(load_code_pair(top_row, code_col + 8), top_scale);
                    af[mt][3] = w8_small_t_bf16_pair_from_s8(load_code_pair(top_row + 8, code_col + 8), bot_scale);
                }
#pragma unroll
                for (int ni = 0; ni < kNt; ++ni) {
                    unsigned bf0, bf1;
                    const int br = ni * 8 + b_rin;
                    ldmatrix_x2(bf0, bf1,
                                smem_addr(&b_shared[k_split][br * kTileK +
                                                            w8_small_t_swizzle_64(br, ks * 16 + b_koff)]));
#pragma unroll
                    for (int mt = 0; mt < RowTiles; ++mt) {
                        auto& fragment = acc[mt * kNt + ni];
                        mma_bf16(fragment[0], fragment[1], fragment[2], fragment[3],
                                 af[mt][0], af[mt][1], af[mt][2], af[mt][3], bf0, bf1);
                    }
                }
            }
        }
        // The stage computed on is the one the next iteration's load overwrites.
        __syncthreads();
    }

    cp_wait<0>();
    __syncthreads();
    // The split combine of the medium-T kernel, among each row group's four warps, over the
    // stages: ((s0 + s1) + (s2 + s3)).
    auto* partial = reinterpret_cast<float*>(smem);
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int ni = 0; ni < kFragments; ++ni) {
            store_vec(partial + ((warp * kFragments + ni) * 32 + lane) * 4,
                      make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int ni = 0; ni < kFragments; ++ni) {
            const float4 partner =
                load_vec<float4>(partial + (((warp + 1) * kFragments + ni) * 32 + lane) * 4);
            acc[ni][0] += partner.x;
            acc[ni][1] += partner.y;
            acc[ni][2] += partner.z;
            acc[ni][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((warp * kFragments + ni) * 32 + lane) * 4,
                          make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
            }
        }
    }

    __syncthreads();
    if (k_split == 0) {
#pragma unroll
        for (int ni = 0; ni < kFragments; ++ni) {
#pragma unroll
            for (int split = 2; split < KSplits; split += 2) {
                const float4 partner =
                    load_vec<float4>(partial + (((warp + split) * kFragments + ni) * 32 + lane) * 4);
                acc[ni][0] += partner.x;
                acc[ni][1] += partner.y;
                acc[ni][2] += partner.z;
                acc[ni][3] += partner.w;
            }
        }
        const W8OutputTile output_tile = output.tile(cta_row0);
        const auto store               = [&](int row, int col, float value) {
            *output_tile.at(row, col + column_begin) = __float2bfloat16_rn(value);
        };
#pragma unroll
        for (int ni = 0; ni < kFragments; ++ni) {
            const int col0 = (ni % kNt) * 8 + 2 * lid;
            const int row0 = cta_row0 + warp_row0 + (ni / kNt) * 16 + gid;
            if (col0 < active_cols) {
                store(row0, col0, acc[ni][0]);
                store(row0 + 8, col0, acc[ni][2]);
            }
            if (col0 + 1 < active_cols) {
                store(row0, col0 + 1, acc[ni][1]);
                store(row0 + 8, col0 + 1, acc[ni][3]);
            }
        }
    }
}

} // namespace sinfer::ops::detail
