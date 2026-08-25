#pragma once

// surogate vendor patch (PATCHES.md #21): NVFP4 prefill GEMM
// (mma.kind::mxf4nvf4.block_scale, sm_120a).
//
// Same staging skeleton as the IMMA/FP8 kernels — at the byte level the
// e2m1 m16n8k64 fragment equals the int8 m16n8k32 fragment (16 rows x 32
// bytes), so ldmatrix.x4/x2 carry over; a K-tile is 64 elements = 32
// bytes. Scales are applied IN HARDWARE from ue4m3 per-16 block-scale
// operands (mapping validated exactly against a CPU reference: sfa reg =
// 4 k-group bytes for weight row 8*(lane&1) + (lane>>2) of the 16-row
// tile; sfb reg = 4 bytes for token lane>>2), and the f32 accumulator
// lives in-core across the whole K. Only row_scale * token_scale remains
// in the tail. Measured 274-309 TF/s (-22% vs the FP8 folded kernel).

#include "ops/common/memory.cuh"
#include "ops/linear/w8a8/w8a8_imma_gemm.cuh"  // ldmatrix helpers, row maps, paired trait

#include <cstdint>

namespace ninfer::ops::detail {

struct W4Fp4Config {
    static constexpr int BM      = 64;
    static constexpr int BN      = 128;
    static constexpr int BKB     = 32;  // bytes per K-tile (64 e2m1)
    static constexpr int BKB_PAD = 48;  // 16B-aligned rows (ldmatrix requirement)
    static constexpr int WARPS_M = 2;
    static constexpr int WARPS_N = 8;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    // A 2-CTA residency cap forces <=64 registers and this kernel spills
    // under it (measured 1.9x on every base-config shape); one resident CTA
    // with full registers is faster.
    static constexpr int MINCTA  = 1;
};

struct W4Fp4WideConfig {
    static constexpr int BM      = 128;
    static constexpr int BN      = 128;
    static constexpr int BKB     = 32;
    static constexpr int BKB_PAD = 48;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 8;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int MINCTA  = 1;
};

__device__ __forceinline__ void w4fp4_mma_16n8k64(float& d0, float& d1, float& d2, float& d3,
                                                  unsigned a0, unsigned a1, unsigned a2,
                                                  unsigned a3, unsigned b0, unsigned b1,
                                                  unsigned sfa, unsigned sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1"
        ".f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
          "h"(static_cast<unsigned short>(0)), "h"(static_cast<unsigned short>(0)), "r"(sfb),
          "h"(static_cast<unsigned short>(0)), "h"(static_cast<unsigned short>(0)));
}

// codes: [rows, k/2]; sf: [rows, k/16] ue4m3; row_scales: [rows] f32.
// x_codes: [tokens, k/2]; x_sf: [tokens, k/16]; x_scales: [tokens] f32.
// `rows` counts LOGICAL rows (the row map translates code/sf/scale rows).
template <class RowMap, class Epilogue, class Cfg = W4Fp4Config>
__global__ __launch_bounds__(Cfg::THREADS, Cfg::MINCTA) void w4fp4_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ sf,
    const float* __restrict__ row_scales, const std::uint8_t* __restrict__ x_codes,
    const std::uint8_t* __restrict__ x_sf, const float* __restrict__ x_scales, int rows, int k,
    int tokens, RowMap row_map, Epilogue epilogue) {
    __shared__ std::uint8_t Ws[2][Cfg::BM * Cfg::BKB_PAD];
    __shared__ std::uint8_t Xs[2][Cfg::BN * Cfg::BKB_PAD];
    __shared__ std::uint8_t SFWs[2][Cfg::BM * 4];  // 4 ue4m3 k-groups per row
    __shared__ std::uint8_t SFXs[2][Cfg::BN * 4];

    const int m0     = static_cast<int>(blockIdx.x) * Cfg::BM;
    const int n0     = static_cast<int>(blockIdx.y) * Cfg::BN;
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int wm     = (warp % Cfg::WARPS_M) * 32;
    const int wn     = (warp / Cfg::WARPS_M) * 16;
    const int kb_row = k / 2;   // code bytes per row
    const int kg_row = k / 16;  // sf bytes per row

    const auto stage = [&](int kt, int slot) {
        const int kbase = kt * Cfg::BKB;
        for (int i = tid; i < Cfg::BM * (Cfg::BKB / 16); i += Cfg::THREADS) {
            const int local = i / (Cfg::BKB / 16);
            const int col   = (i % (Cfg::BKB / 16)) * 16;
            const int wrow  = row_map.weight_row(m0 + local);
            ninfer::ops::cp_async<16, ninfer::ops::Cache::cg>(
                &Ws[slot][local * Cfg::BKB_PAD + col],
                &codes[static_cast<std::int64_t>(wrow) * kb_row + kbase + col]);
        }
        for (int i = tid; i < Cfg::BN * (Cfg::BKB / 16); i += Cfg::THREADS) {
            const int token  = i / (Cfg::BKB / 16);
            const int col    = (i % (Cfg::BKB / 16)) * 16;
            const bool valid = n0 + token < tokens;
            const std::int64_t src =
                static_cast<std::int64_t>(valid ? n0 + token : 0) * kb_row + kbase + col;
            ninfer::ops::cp_async_zfill<16>(&Xs[slot][token * Cfg::BKB_PAD + col], &x_codes[src],
                                            valid ? 16 : 0);
        }
        // Block scales for this K-tile: 4 async bytes per row/token, in the
        // same commit group as the codes (a synchronous LDG here stalls the
        // stage and cost +16..40% on the measured shapes).
        for (int i = tid; i < Cfg::BM; i += Cfg::THREADS) {
            const int wrow = row_map.weight_row(m0 + i);
            ninfer::ops::cp_async<4>(&SFWs[slot][i * 4],
                                     sf + static_cast<std::int64_t>(wrow) * kg_row + kt * 4);
        }
        for (int i = tid; i < Cfg::BN; i += Cfg::THREADS) {
            const int token = n0 + i < tokens ? n0 + i : tokens - 1;
            ninfer::ops::cp_async<4>(&SFXs[slot][i * 4],
                                     x_sf + static_cast<std::int64_t>(token) * kg_row + kt * 4);
        }
    };

    float acc[2][2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[mi][ni][q] = 0.0f;

    const int nkt = k / 64;
    stage(0, 0);
    ninfer::ops::cp_commit();

    // SF rows for this lane (mapping validated single-mma-exact).
    const int sfa_frag_row = 8 * (lane & 1) + (lane >> 2);
    const int sfb_frag_tok = lane >> 2;

    for (int kt = 0; kt < nkt; ++kt) {
        const int slot = kt & 1;
        ninfer::ops::cp_wait<0>();
        __syncthreads();
        if (kt + 1 < nkt) {
            stage(kt + 1, slot ^ 1);
            ninfer::ops::cp_commit();
        }

#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int arow0 = wm + mi * 16;
            unsigned a[4];
            {
                const int local = arow0 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int cofs  = (lane >> 4) * 16;
                w8a8_ldmatrix_x4(a[0], a[1], a[2], a[3], &Ws[slot][local * Cfg::BKB_PAD + cofs]);
            }
            const unsigned sfa =
                reinterpret_cast<const unsigned*>(SFWs[slot])[arow0 + sfa_frag_row];
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int token0 = wn + ni * 8;
                unsigned b[2];
                {
                    const int token = token0 + (lane & 7);
                    const int cofs  = ((lane >> 3) & 1) * 16;
                    w8a8_ldmatrix_x2(b[0], b[1], &Xs[slot][token * Cfg::BKB_PAD + cofs]);
                }
                const unsigned sfb =
                    reinterpret_cast<const unsigned*>(SFXs[slot])[token0 + sfb_frag_tok];
                w4fp4_mma_16n8k64(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                                  a[0], a[1], a[2], a[3], b[0], b[1], sfa, sfb);
            }
        }
        __syncthreads();
    }

    if constexpr (W8A8EpilogueIsPaired<Epilogue>::value) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float value   = acc[mi][ni][q];
                    const float partner = __shfl_down_sync(0xffffffffu, value, 4);
                    const int frag_row  = lane >> 2;
                    if ((frag_row & 1) != 0) { continue; }
                    const int row   = m0 + wm + mi * 16 + frag_row + (q >= 2 ? 8 : 0);
                    const int token = n0 + wn + ni * 8 + (lane & 3) * 2 + (q & 1);
                    if (row < rows && token < tokens) {
                        const float xs = x_scales[token];
                        const float gate =
                            value * row_scales[row_map.weight_row(row)] * xs;
                        const float up =
                            partner * row_scales[row_map.weight_row(row + 1)] * xs;
                        epilogue.store_pair(row >> 1, token, gate, up);
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int row   = m0 + wm + mi * 16 + (lane >> 2) + (q >= 2 ? 8 : 0);
                    const int token = n0 + wn + ni * 8 + (lane & 3) * 2 + (q & 1);
                    if (row < rows && token < tokens) {
                        epilogue(row, token,
                                 acc[mi][ni][q] * row_scales[row_map.weight_row(row)] *
                                     x_scales[token]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
