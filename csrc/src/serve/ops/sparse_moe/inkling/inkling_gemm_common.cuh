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

/// fp32 -> bf16 -> fp32: the reference's bf16 store of gate_up followed by the SiTU's load.
__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

} // namespace sinfer::ops::detail::inkling
