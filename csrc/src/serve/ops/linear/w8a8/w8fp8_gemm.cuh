#pragma once

// surogate vendor patch (PATCHES.md #20): folded-scale FP8 prefill GEMM.
//
// Same staging structure as the W8A8-int IMMA kernel (2-stage cp.async,
// 80-byte staging stride, ldmatrix.x4/x2) but the operands are the derived
// e4m3 plane (group scales folded into per-row-renormalized weights, see
// w8fp8_plane.h) and e4m3 per-token activations. The mma accumulates in
// f16 ACROSS the whole 64-wide k-tile (bounded by 2*32*1.0*448 = 28672 <
// 65504, no clamping by construction), spills to f32 once per tile, and
// applies row_scale * x_scale once in the tail. Removing the per-group
// scale tail is the entire win: measured -12..16% versus the IMMA kernel
// at every prefill shape (the mma-rate swap alone measures 0).

#include "ops/common/memory.cuh"
#include "ops/linear/w8a8/w8a8_imma_gemm.cuh"  // configs, ldmatrix helpers, row maps, paired trait

#include <cuda_fp16.h>

#include <cstdint>

namespace sinfer::ops::detail {

__device__ __forceinline__ void w8fp8_mma_16n8k32_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                                         unsigned a1, unsigned a2, unsigned a3,
                                                         unsigned b0, unsigned b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // FP8 tensor cores require sm_89+; hosts gate on runtime CC before
    // launching (w8fp8_plane_enabled), so this body is unreachable there.
    (void)d0; (void)d1; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    __trap();
#endif
}

// codes: [weight_rows, k] e4m3 (derived plane); row_scales: [weight_rows] f32.
// x_codes: [tokens, k] e4m3; x_scales: [tokens] f32 (xmax / 448).
// `rows` counts LOGICAL output rows (the row map translates to plane rows).
template <class RowMap, class Epilogue, class Cfg = W8A8ImmaConfig>
__global__ __launch_bounds__(Cfg::THREADS, 2) void w8fp8_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const float* __restrict__ row_scales,
    const std::uint8_t* __restrict__ x_codes, const float* __restrict__ x_scales, int rows, int k,
    int tokens, RowMap row_map, Epilogue epilogue) {
    __shared__ std::uint8_t Ws[2][Cfg::BM * Cfg::BK_PAD];
    __shared__ std::uint8_t Xs[2][Cfg::BN * Cfg::BK_PAD];

    const int m0   = static_cast<int>(blockIdx.x) * Cfg::BM;
    const int n0   = static_cast<int>(blockIdx.y) * Cfg::BN;
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = (warp % Cfg::WARPS_M) * 32;
    const int wn   = (warp / Cfg::WARPS_M) * 16;

    const auto stage = [&](int kt, int slot) {
        const int kbase = kt * Cfg::BK;
        for (int i = tid; i < Cfg::BM * (Cfg::BK / 16); i += Cfg::THREADS) {
            const int local = i / (Cfg::BK / 16);
            const int col   = (i % (Cfg::BK / 16)) * 16;
            const int wrow  = row_map.weight_row(m0 + local);
            sinfer::ops::cp_async<16, sinfer::ops::Cache::cg>(
                &Ws[slot][local * Cfg::BK_PAD + col],
                &codes[static_cast<std::int64_t>(wrow) * k + kbase + col]);
        }
        for (int i = tid; i < Cfg::BN * (Cfg::BK / 16); i += Cfg::THREADS) {
            const int token = i / (Cfg::BK / 16);
            const int col   = (i % (Cfg::BK / 16)) * 16;
            const bool valid = n0 + token < tokens;
            const std::int64_t src =
                static_cast<std::int64_t>(valid ? n0 + token : 0) * k + kbase + col;
            sinfer::ops::cp_async_zfill<16>(&Xs[slot][token * Cfg::BK_PAD + col], &x_codes[src],
                                            valid ? 16 : 0);
        }
    };

    float acc[2][2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[mi][ni][q] = 0.0f;

    const int nkt = k / Cfg::BK;
    stage(0, 0);
    sinfer::ops::cp_commit();

    for (int kt = 0; kt < nkt; ++kt) {
        const int slot = kt & 1;
        sinfer::ops::cp_wait<0>();
        __syncthreads();
        if (kt + 1 < nkt) {
            stage(kt + 1, slot ^ 1);
            sinfer::ops::cp_commit();
        }

#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int arow0 = wm + mi * 16;
            unsigned d[2][2] = {{0u, 0u}, {0u, 0u}};
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                const int kb = group * 32;
                unsigned a[4];
                {
                    const int local = arow0 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int cofs  = kb + (lane >> 4) * 16;
                    w8a8_ldmatrix_x4(a[0], a[1], a[2], a[3],
                                     &Ws[slot][local * Cfg::BK_PAD + cofs]);
                }
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int token0 = wn + ni * 8;
                    unsigned b[2];
                    {
                        const int token = token0 + (lane & 7);
                        const int cofs  = kb + ((lane >> 3) & 1) * 16;
                        w8a8_ldmatrix_x2(b[0], b[1], &Xs[slot][token * Cfg::BK_PAD + cofs]);
                    }
                    w8fp8_mma_16n8k32_f16acc(d[ni][0], d[ni][1], a[0], a[1], a[2], a[3], b[0],
                                             b[1]);
                }
            }
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const __half2 lo = *reinterpret_cast<const __half2*>(&d[ni][0]);
                const __half2 hi = *reinterpret_cast<const __half2*>(&d[ni][1]);
                acc[mi][ni][0] += __low2float(lo);
                acc[mi][ni][1] += __high2float(lo);
                acc[mi][ni][2] += __low2float(hi);
                acc[mi][ni][3] += __high2float(hi);
            }
        }
        __syncthreads();
    }

    if constexpr (W8A8EpilogueIsPaired<Epilogue>::value) {
        // Paired tail (fused swiglu): even fragment rows are gate, +4 lanes
        // is the matching up row. Each half scales by ITS OWN row scale
        // before the pair joins (see W8A8SwigluPairRowMap).
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

} // namespace sinfer::ops::detail
