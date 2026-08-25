#pragma once

// surogate vendor patch (PATCHES.md #17): W8A8-int IMMA prefill GEMM.
//
// W8G32_F16S weights stay bit-exact (the codes ARE int8); activations are
// per-token int8 (w8a8_act_quant). mma.m16n8k32 consumes exactly one
// 32-value quantization group per instruction, so the per-group weight
// scale applies on the int32 group result before the FP32 accumulate and
// the per-token activation scale at the epilogue.
//
// Configuration measured on RTX 5090 (bench/ops/w8a8_imma_probe_bench):
// BM64 x BN128 x BK64, 16 warps (warp tile 32x16), 2-stage cp.async, and
// an 80-byte staging stride — banks step by 20 per row, making the
// ldmatrix.x4/x2 fragment loads conflict-free without XOR swizzles —
// reaching 132-144 TF/s at T >= 1024 (~1.5x the BF16 A16 family ceiling).
// Iterations that measured SLOWER and are deliberately absent: 4-stage
// staging (not latency-bound) and ldmatrix at 8 warps (occupancy-starved).
//
// RowMap chooses the weight/scale row for each logical output row (identity
// for plain/split outputs; the swiglu form maps the CTA's second half to the
// up-projection half of the fused gate_up weight). Epilogue receives the
// fully-scaled FP32 value per (logical row, token).

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct W8A8ImmaConfig {
    static constexpr int BM      = 64;
    static constexpr int BN      = 128;
    static constexpr int BK      = 64;  // two 32-value groups per k-tile
    static constexpr int BK_PAD  = 80;  // bank-conflict-free ldmatrix stride
    static constexpr int WARPS_M = 2;   // warp tile 32 x 16
    static constexpr int WARPS_N = 8;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
};

// Wide-row configuration: +10% at T >= ~1024 (200 TF/s at 1912-class on the
// K=2048 shapes); BM64 stays better below. Same warp tile (32x16).
struct W8A8ImmaWideConfig {
    static constexpr int BM      = 128;
    static constexpr int BN      = 128;
    static constexpr int BK      = 64;
    static constexpr int BK_PAD  = 80;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 8;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
};

inline constexpr std::int32_t kW8A8WideMinTokens = 1024;

struct W8A8IdentityRowMap {
    __device__ __forceinline__ int weight_row(int logical_row) const { return logical_row; }
};

// surogate vendor patch (PATCHES.md #19): gate/up interleave for the fused
// swiglu epilogue. Logical row 2i is gate row i, 2i+1 is up row i, so a pair
// lands 4 lanes apart in the mma fragment and one shfl joins them.
struct W8A8SwigluPairRowMap {
    int intermediate;
    __device__ __forceinline__ int weight_row(int logical_row) const {
        return (logical_row & 1) ? intermediate + (logical_row >> 1) : (logical_row >> 1);
    }
};

template <class Epilogue, class = void>
struct W8A8EpilogueIsPaired : std::false_type {};
template <class Epilogue>
struct W8A8EpilogueIsPaired<Epilogue, std::void_t<decltype(Epilogue::kPairedRows)>>
    : std::bool_constant<Epilogue::kPairedRows> {};

__device__ __forceinline__ void w8a8_imma_16n8k32(int& d0, int& d1, int& d2, int& d3, unsigned a0,
                                                  unsigned a1, unsigned a2, unsigned a3,
                                                  unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void w8a8_ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2,
                                                 unsigned& r3, const void* smem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void w8a8_ldmatrix_x2(unsigned& r0, unsigned& r1, const void* smem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

// codes: [weight_rows, k] int8 row-major (the W8 row-split code plane).
// scales: [weight_rows, k/32] fp16 row-major (the W8 scale plane).
// x_codes: [tokens, k] int8 (w8a8_act_quant layout); x_scales: [tokens] fp32.
// `rows` counts LOGICAL output rows (grid.x covers ceil(rows / BM)).
template <class RowMap, class Epilogue, class Cfg = W8A8ImmaConfig>
__global__ __launch_bounds__(Cfg::THREADS, 2) void w8a8_imma_gemm_kernel(
    const std::int8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const std::int8_t* __restrict__ x_codes, const float* __restrict__ x_scales, int rows, int k,
    int tokens, RowMap row_map, Epilogue epilogue) {
    __shared__ std::int8_t Ws[2][Cfg::BM * Cfg::BK_PAD];
    __shared__ std::int8_t Xs[2][Cfg::BN * Cfg::BK_PAD];
    __shared__ __half Ss[2][Cfg::BM * 2];

    const int m0   = static_cast<int>(blockIdx.x) * Cfg::BM;
    const int n0   = static_cast<int>(blockIdx.y) * Cfg::BN;
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = (warp % Cfg::WARPS_M) * 32;
    const int wn   = (warp / Cfg::WARPS_M) * 16;
    const int kg   = k / 32;

    const auto* scale_plane = reinterpret_cast<const __half*>(scales);

    const auto stage = [&](int kt, int slot) {
        const int kbase = kt * Cfg::BK;
        for (int i = tid; i < Cfg::BM * (Cfg::BK / 16); i += Cfg::THREADS) {
            const int local = i / (Cfg::BK / 16);
            const int col   = (i % (Cfg::BK / 16)) * 16;
            const int wrow  = row_map.weight_row(m0 + local);
            ninfer::ops::cp_async<16, ninfer::ops::Cache::cg>(
                &Ws[slot][local * Cfg::BK_PAD + col],
                &codes[static_cast<std::int64_t>(wrow) * k + kbase + col]);
        }
        for (int i = tid; i < Cfg::BN * (Cfg::BK / 16); i += Cfg::THREADS) {
            const int token = i / (Cfg::BK / 16);
            const int col   = (i % (Cfg::BK / 16)) * 16;
            // Out-of-range tokens zero-fill: no global read at all. A clamp
            // to the last valid row measured ~2x SLOWER for heavily-partial
            // last tiles (duplicate-source cp.async serializes).
            const bool valid = n0 + token < tokens;
            const std::int64_t src =
                static_cast<std::int64_t>(valid ? n0 + token : 0) * k + kbase + col;
            ninfer::ops::cp_async_zfill<16>(&Xs[slot][token * Cfg::BK_PAD + col], &x_codes[src],
                                            valid ? 16 : 0);
        }
        for (int i = tid; i < Cfg::BM * 2; i += Cfg::THREADS) {
            const int local = i >> 1;
            const int group = i & 1;
            const int wrow  = row_map.weight_row(m0 + local);
            Ss[slot][i] = scale_plane[static_cast<std::int64_t>(wrow) * kg + kt * 2 + group];
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
    ninfer::ops::cp_commit();

    for (int kt = 0; kt < nkt; ++kt) {
        const int slot = kt & 1;
        ninfer::ops::cp_wait<0>();
        __syncthreads();
        if (kt + 1 < nkt) {
            stage(kt + 1, slot ^ 1);
            ninfer::ops::cp_commit();
        }

#pragma unroll
        for (int group = 0; group < 2; ++group) {
            const int kb = group * 32;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int arow0 = wm + mi * 16;
                unsigned a[4];
                {
                    const int local = arow0 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int cofs  = kb + (lane >> 4) * 16;
                    w8a8_ldmatrix_x4(a[0], a[1], a[2], a[3],
                                     &Ws[slot][local * Cfg::BK_PAD + cofs]);
                }
                int d[2][4];
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int token0 = wn + ni * 8;
                    unsigned b[2];
                    {
                        const int token = token0 + (lane & 7);
                        const int cofs  = kb + ((lane >> 3) & 1) * 16;
                        w8a8_ldmatrix_x2(b[0], b[1], &Xs[slot][token * Cfg::BK_PAD + cofs]);
                    }
                    d[ni][0] = d[ni][1] = d[ni][2] = d[ni][3] = 0;
                    w8a8_imma_16n8k32(d[ni][0], d[ni][1], d[ni][2], d[ni][3], a[0], a[1], a[2],
                                      a[3], b[0], b[1]);
                }
                const float ws0 = __half2float(Ss[slot][(arow0 + (lane >> 2)) * 2 + group]);
                const float ws8 = __half2float(Ss[slot][(arow0 + (lane >> 2) + 8) * 2 + group]);
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    acc[mi][ni][0] += static_cast<float>(d[ni][0]) * ws0;
                    acc[mi][ni][1] += static_cast<float>(d[ni][1]) * ws0;
                    acc[mi][ni][2] += static_cast<float>(d[ni][2]) * ws8;
                    acc[mi][ni][3] += static_cast<float>(d[ni][3]) * ws8;
                }
            }
        }
        __syncthreads();
    }

    if constexpr (W8A8EpilogueIsPaired<Epilogue>::value) {
        // Paired tail: every thread computes and shuffles before any guard so
        // the shfl mask stays full. Even fragment rows hold gate, +4 lanes
        // holds the matching up row (see W8A8SwigluPairRowMap).
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
                        const float scale = x_scales[token];
                        epilogue.store_pair(row >> 1, token, value * scale, partner * scale);
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
                        epilogue(row, token, acc[mi][ni][q] * x_scales[token]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
