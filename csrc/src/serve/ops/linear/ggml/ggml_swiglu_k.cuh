#pragma once

#include "ops/linear/ggml/ggml_i8_tile.cuh"
#include "ops/linear/ggml/ggml_dense_decode.cuh"

namespace sinfer::ops::detail::ggml {

// Decode both token columns from each weight load. Each warp retains the
// format's two K accumulators independently for each token.
template <class Gate, class Up>
__global__ __launch_bounds__(64) void swiglu_k_dual_kernel(const uint8_t* gate, const uint8_t* up,
                                                           const int8_t* codes, const __half2* ds,
                                                           int rows, int k, __nv_bfloat16* out) {
    __shared__ __nv_bfloat16 halves[2][2];
    const int row = blockIdx.x, warp = threadIdx.x >> 5;
    const int column0 = blockIdx.y * 2;
    codes += int64_t(column0) * k;
    ds += int64_t(column0) * (k / 32);
    out += int64_t(column0) * rows;
    float acc[2];
    if (warp == 0) {
        dense_k_dots<Gate, 2>(reinterpret_cast<const typename Gate::Block*>(gate) +
                                  int64_t(row) * (k / QK_K),
                              codes, ds, k, acc);
    } else {
        dense_k_dots<Up, 2>(reinterpret_cast<const typename Up::Block*>(up) +
                                int64_t(row) * (k / QK_K),
                            codes, ds, k, acc);
    }
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
        for (int c = 0; c < 2; ++c) { halves[warp][c] = __float2bfloat16_rn(acc[c] + 0.0f); }
    }
    __syncthreads();
    if (threadIdx.x < 2) {
        const int column                  = threadIdx.x;
        out[int64_t(column) * rows + row] = __float2bfloat16_rn(swiglu_clamped(
            __bfloat162float(halves[0][column]), __bfloat162float(halves[1][column]), 0.0f));
    }
}

// A homogeneous pair uses 16 gate rows and 16 up rows in one tile. Keeping
// each half contiguous also avoids narrow partially-filled decode slowdowns.
template <class Codec, int TileRows = 32, int TileCols = 8>
__global__ __launch_bounds__((TileCols / 8) *
                             32) void swiglu_k_tile_kernel(const uint8_t* gate, const uint8_t* up,
                                                           const int8_t* codes, const __half2* ds,
                                                           int rows, int k, int tokens,
                                                           __nv_bfloat16* out) {
    static_assert(TileRows == 32 || TileRows == 64);
    constexpr int half_rows = TileRows / 2;
    __shared__ GgmlI8Smem<Codec, TileCols, TileRows> sm;
    const int row0 = blockIdx.x * half_rows;
    const int col0 = blockIdx.y * TileCols;
    auto row_base  = [&](int row) {
        return (row < half_rows ? gate : up) +
               int64_t(min(row0 + row % half_rows, rows - 1)) * (k / QK_K) * Codec::kBlockBytes;
    };
    auto act_row = [&](int column) { return int64_t(col0 + column); };
    float acc[TileRows / 16][4];
    ggml_i8_tile<Codec, TileCols / 8, TileCols, true, TileRows>(
        sm, row_base, act_row, codes, ds, min(TileCols, tokens - col0), k, acc);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int mi = 0; mi < half_rows / 16; ++mi)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row    = row0 + mi * 16 + (lane >> 2) + (e / 2) * 8;
            const int column = col0 + (threadIdx.x >> 5) * 8 + (lane & 3) * 2 + (e & 1);
            if (row < rows && column < tokens) {
                const float g = __bfloat162float(__float2bfloat16_rn(acc[mi][e] + 0.0f));
                const float u =
                    __bfloat162float(__float2bfloat16_rn(acc[mi + half_rows / 16][e] + 0.0f));
                out[int64_t(column) * rows + row] = __float2bfloat16_rn(swiglu_clamped(g, u, 0.0f));
            }
        }
}

// Mixed codecs can have different scale widths (Q6_K versus Q4_K/Q5_K).
// Project each with its own tile, retaining rounded gate values in registers
// while reusing shared staging memory for up. No global intermediate planes.
template <class Gate, class Up>
__global__ __launch_bounds__(32) void swiglu_k_mixed_tile_kernel(const uint8_t* gate,
                                                                 const uint8_t* up,
                                                                 const int8_t* codes,
                                                                 const __half2* ds, int rows, int k,
                                                                 int tokens, __nv_bfloat16* out) {
    using GateSmem       = GgmlI8Smem<Gate, 8, 16>;
    using UpSmem         = GgmlI8Smem<Up, 8, 16>;
    constexpr auto bytes = sizeof(GateSmem) > sizeof(UpSmem) ? sizeof(GateSmem) : sizeof(UpSmem);
    __shared__ __align__(16) uint8_t storage[bytes];
    const int row0 = blockIdx.x * 16;
    auto gate_row  = [&](int row) {
        return gate + int64_t(min(row0 + row, rows - 1)) * (k / QK_K) * Gate::kBlockBytes;
    };
    auto up_row = [&](int row) {
        return up + int64_t(min(row0 + row, rows - 1)) * (k / QK_K) * Up::kBlockBytes;
    };
    auto act_row = [&](int column) { return int64_t(column); };
    float g[1][4], u[1][4];
    ggml_i8_tile<Gate, 1, 8, true, 16>(*reinterpret_cast<GateSmem*>(storage), gate_row, act_row,
                                       codes, ds, tokens, k, g);
#pragma unroll
    for (int e = 0; e < 4; ++e) { g[0][e] = __bfloat162float(__float2bfloat16_rn(g[0][e] + 0.0f)); }
    // ggml_i8_tile drains its async copies and synchronizes before returning.
    ggml_i8_tile<Up, 1, 8, true, 16>(*reinterpret_cast<UpSmem*>(storage), up_row, act_row, codes,
                                     ds, tokens, k, u);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int row    = row0 + (lane >> 2) + (e / 2) * 8;
        const int column = (lane & 3) * 2 + (e & 1);
        if (row < rows && column < tokens) {
            const float uv = __bfloat162float(__float2bfloat16_rn(u[0][e] + 0.0f));
            out[int64_t(column) * rows + row] =
                __float2bfloat16_rn(swiglu_clamped(g[0][e], uv, 0.0f));
        }
    }
}

} // namespace sinfer::ops::detail::ggml
