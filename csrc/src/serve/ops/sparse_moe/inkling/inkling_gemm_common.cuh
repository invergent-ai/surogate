#pragma once

// Pieces shared by the NVFP4 and MXFP8 grouped expert GEMMs of the Inkling MoE: finding a CTA's
// expert tile in the routing offsets, the adapter slot of a token, and the fused LoRA expand.

#include "ops/sparse_moe/inkling/inkling_moe_internal.h"
#include "ops/sparse_moe/inkling/inkling_numerics.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops::detail::inkling {

/// The expert tile `tile` when each expert's routed rows (offsets[e] .. offsets[e + 1]) are cut
/// into tiles of `TileRows`: writes {expert, row_begin, row_end} to `result` (expert -1 = past
/// the round's tiles). Every thread of the block calls it; one barrier inside.
template <int TileRows>
__device__ __forceinline__ void find_expert_tile(const std::int32_t* offsets, int tile,
                                                 int (&result)[3]) {
    if (threadIdx.x < 32) {
        const int lane = static_cast<int>(threadIdx.x);
        int bounds[5];
#pragma unroll
        for (int i = 0; i < 5; ++i) { bounds[i] = offsets[4 * lane + i]; }
        int tiles[4];
        int local = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            tiles[i] = (bounds[i + 1] - bounds[i] + TileRows - 1) / TileRows;
            local += tiles[i];
        }
        int inclusive = local;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xFFFFFFFFU, inclusive, offset);
            if (lane >= offset) { inclusive += other; }
        }
        const int total = __shfl_sync(0xFFFFFFFFU, inclusive, 31);
        if (lane == 0 && tile >= total) { result[0] = -1; }
        int before = inclusive - local;
        if (tile >= before && tile < inclusive) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (tile >= before && tile < before + tiles[i]) {
                    result[0] = 4 * lane + i;
                    result[1] = bounds[i] + (tile - before) * TileRows;
                    result[2] = min(result[1] + TileRows, bounds[i + 1]);
                }
                before += tiles[i];
            }
        }
    }
    __syncthreads();
}

/// The adapter slot of token `token` (negative = base token, no delta).
__device__ __forceinline__ int lora_slot_of(const LoraRound& lora, int token) {
    if (lora.table == nullptr) { return -1; }
    if (lora.token_slot != nullptr) { return lora.token_slot[token]; }
    return lora.uniform_slot != nullptr ? *lora.uniform_slot : -1;
}

/// delta = B[slot][n][:] . low[:] in fp32, r ascending (one fixed order for every round).
__device__ __forceinline__ float lora_expand(const LoraBank& bank, int slot, int n, const float* low) {
    const auto* b = static_cast<const __nv_bfloat16*>(bank.b) +
                    static_cast<std::int64_t>(slot) * bank.b_stride +
                    static_cast<std::int64_t>(n) * bank.rank;
    float delta = 0.0F;
    for (int r = 0; r < bank.rank; ++r) {
        delta = __fmaf_rn(__bfloat162float(b[r]), low[r], delta);
    }
    return delta;
}

/// The bank of `port` (0 gate, 1 up, 2 down) of expert `expert` in a `bind_lora_moe` table.
__device__ __forceinline__ const LoraBank& lora_bank(const LoraRound& lora, int expert, int port) {
    return lora.table[2 + 3 * expert + port];
}

/// Fused combine of the down GEMM (DESIGN-moe 2.1 v2). Called by the `Threads` threads that
/// stored this tile's down rows (columns [n_begin, n_begin + Columns) of the slots `slot_of(r)`,
/// r < rows; negative = no row), with `sync()` a barrier over exactly those threads and `done` /
/// `done_count` scratch in shared memory (done_count zero on entry; zero again on return). Each
/// row arrives on its token's counter for this column tile; the CTA that brings a counter to 8
/// sums the token's 8 rows in slot order, ((0 + d0) + d1) ... + d7 in fp32 (the combine
/// kernel's arithmetic: no atomics in the sum, no dependence on which CTA arrives last).
template <int Columns, int Threads, class SlotOf, class Sync>
__device__ __forceinline__ void combine_arrive(const GemmParams& p, int rows, int n_begin, SlotOf slot_of,
                                               int* done, int* done_count, int thread, Sync sync) {
    static_assert(Columns % 8 == 0 && kGateUpRows % Columns == 0);
    constexpr int kTilesPerToken = kGateUpRows / Columns;
    constexpr int kChunks        = Columns / 8;
    const int n_tile             = n_begin / Columns;
    __threadfence();
    sync();
    for (int r = thread; r < rows; r += Threads) {
        const int slot = slot_of(r);
        if (slot < 0) { continue; }
        const int token = slot >> 3;
        if (atomicAdd(p.arrivals + static_cast<std::int64_t>(token) * kArrivalTiles + n_tile, 1) == kTopK - 1) {
            done[atomicAdd(done_count, 1)] = token;
        }
    }
    static_assert(kTilesPerToken <= kArrivalTiles);
    sync();
    __threadfence();
    const int count = *done_count;
    for (int task = thread; task < count * kChunks; task += Threads) {
        const int token  = done[task / kChunks];
        const int column = n_begin + (task % kChunks) * 8;
        float acc[8]     = {};
#pragma unroll
        for (int s = 0; s < kTopK; ++s) {
            const uint4 v = __ldcg(reinterpret_cast<const uint4*>(
                p.out + (static_cast<std::int64_t>(token) * kTopK + s) * kHidden + column));
            const std::uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[2 * i]     = __fadd_rn(acc[2 * i], __uint_as_float(words[i] << 16));
                acc[2 * i + 1] = __fadd_rn(acc[2 * i + 1], __uint_as_float(words[i] & 0xFFFF0000U));
            }
        }
        std::uint32_t packed[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat162 pair = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
            packed[i]                 = *reinterpret_cast<const std::uint32_t*>(&pair);
        }
        *reinterpret_cast<uint4*>(p.combined + static_cast<std::int64_t>(token) * kHidden + column) =
            make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
    sync();
    if (thread == 0) { *done_count = 0; }
}

/// fp32 -> bf16 -> fp32: the reference's bf16 store of gate_up followed by the SiTU's load.
__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

} // namespace sinfer::ops::detail::inkling
