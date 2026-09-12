// The int8 tensor-core tile every K-quant prefill GEMM runs on: a 64-row weight tile in the
// format's own superblocks, staged and unpacked to int8 codes, against activations quantised to
// int8 per 32 with a (scale, sum) pair each -- what llama.cpp's MMQ consumes. The routing is
// the caller's: `row_base(local_row)` names a weight row's first superblock and `act_row(col)`
// the activation row a column reads. Dense projections retain the exact activation code sum
// so their affine correction uses the same quantised values and scale as decode.
#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/ggml/ggml_blocks.h"
#include "ops/linear/ggml/ggml_prefill_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace sinfer::ops::detail::ggml {

inline constexpr int kI8BM     = 64; // weight rows per tile
inline constexpr int kI8BK     = 64; // K per stage, one quarter of a superblock
inline constexpr int kI8Stages = 2;

__device__ __forceinline__ const std::uint8_t* align_down16(const std::uint8_t* p) {
    return reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(p) &
                                                 ~std::uintptr_t{15});
}

/// For a covering codec, the even offset below sixteen at which the tile's superblock starts:
/// what its readers add to the staged tile and header. Zero for every other codec.
template <class Codec>
__device__ __forceinline__ int cover_offset(const std::uint8_t* row_base, int tile) {
    if constexpr (Codec::kCover) {
        return static_cast<int>(reinterpret_cast<std::uintptr_t>(row_base + Codec::header_offset(tile)) & 15u);
    } else {
        (void)row_base; (void)tile;
        return 0;
    }
}

template <class Codec>
__device__ __forceinline__ void stage_ggml_tile(std::uint8_t* dst, const std::uint8_t* row_base,
                                                int tile, int unit) {
    if constexpr (Codec::kCpAsync) {
        const std::uint8_t* src = row_base + Codec::chunk_offset(tile, unit);
        if constexpr (Codec::kCover) { src = align_down16(src); }
        cp_async<16, Cache::cg>(dst + 16 * unit, src);
    } else {
        const auto* src = reinterpret_cast<const std::uint16_t*>(
            row_base + Codec::chunk_offset(tile, unit / 8));
        auto* out     = reinterpret_cast<std::uint16_t*>(dst + 16 * (unit / 8));
        out[unit % 8] = src[unit % 8];
    }
}

/// The superblock header of the tile's block, staged into the same double buffer as the body.
/// Holding all of a row's headers instead cost eight kilobytes of shared memory on a 2048-wide
/// gate/up -- enough to lose a block per SM -- and re-read them through an uncoalesced prologue
/// on every work item.
template <class Codec>
__device__ __forceinline__ void stage_ggml_header(std::uint8_t* dst, const std::uint8_t* row_base,
                                                  int tile, int unit, int row_tiles) {
    const std::uint8_t* src = row_base + Codec::header_offset(tile);
    if constexpr (Codec::kCpAsync) {
        if constexpr (Codec::kCover) {
            // The last block can end inside this aligned span. Do not depend
            // on an allocation padding the row: bound the final scalar loads.
            const int bytes = tile + kTilesPerBlock < row_tiles ? 16 :
                min(16, max(0, Codec::kHeaderPayloadBytes +
                               cover_offset<Codec>(row_base, tile) - 16 * unit));
            src = align_down16(src);
            if (bytes == 16) {
                cp_async<16, Cache::cg>(dst + 16 * unit, src + 16 * unit);
            } else {
                const auto* in = reinterpret_cast<const volatile std::uint16_t*>(src + 16 * unit);
                auto* out = reinterpret_cast<std::uint16_t*>(dst + 16 * unit);
#pragma unroll
                for (int i = 0; i < 8; ++i) { out[i] = 2 * i < bytes ? in[i] : 0; }
            }
        } else {
            cp_async<16, Cache::cg>(dst + 16 * unit, src + 16 * unit);
        }
    } else {
        reinterpret_cast<std::uint16_t*>(dst)[unit] =
            reinterpret_cast<const std::uint16_t*>(src)[unit];
    }
}


// the K-quant affine form summed against x, with the x sum taken from the quantiser rather than
// recomputed. A format whose scales cover sixteen values (Q6_K) runs two sixteen-deep MMAs per
// group, each against its own scale, and folds its symmetric offset into the codes so there is
// no min term. Per row the sub-block scales are decoded once per superblock into a shared table
// indexed [scale][row], since every warp of the block owns every row.
// ---------------------------------------------------------------------------------------------
constexpr int kI8Stride = 80; // int8 tile row stride; conflict-free ldmatrix on 64-byte rows

template <class Codec, int ExpertBN, int ExpertBM = kI8BM>
struct GgmlI8Smem {
    using Scale = std::conditional_t<Codec::kHasMin, float2, float>;
    alignas(16) std::uint8_t Cr[kI8Stages][ExpertBM * Codec::kTileStride];
    alignas(16) std::uint8_t Hdr[kI8Stages][ExpertBM * Codec::kHeaderStride];
    alignas(16) std::int8_t As[ExpertBM * kI8Stride];
    alignas(16) std::int8_t Xs[kI8Stages][ExpertBN * kI8Stride];
    alignas(16) Scale Sc[kI8Stages][Codec::kScaleGroups * ExpertBM];
    alignas(16) __half2 Ds[kI8Stages][ExpertBN * 2];
    std::uint8_t Off[kI8Stages][ExpertBM]; // a covering codec's per-row offset, by superblock parity
};

/// The K loop the gate/up and down kernels share. `row_base(local_row)` is a weight row's
/// first superblock, `act_row(local_col)` the activation row a tile column reads (a token for
/// gate/up, a packed column for down), and `acc` comes back holding this warp's 64 x 8 products.
template <class Codec, int ExpertWarps, int ExpertBN, bool ExactActivationSum = false,
          int ExpertBM = kI8BM, class RowBase, class ActRow>
__device__ __forceinline__ void ggml_i8_tile(GgmlI8Smem<Codec, ExpertBN, ExpertBM>& sm,
                                                    RowBase row_base, ActRow act_row,
                                                    const std::int8_t* __restrict__ codes,
                                                    const __half2* __restrict__ ds, int cols, int K,
                                             float (&acc)[ExpertBM / 16][4]) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int WarpCols      = ExpertBN / ExpertWarps;
    static_assert(WarpCols == 8, "one n8 fragment per warp");
    static_assert((ExpertBM == 16 || ExpertBM == 32 || ExpertBM == 64) && kI8BK == 64 && kI8Stages == 2);
    // K is a whole number of superblocks; the caller's dispatch guarantees it.
    {
    constexpr int TileUnits   = Codec::kCpAsync ? Codec::kTileBytes / 16 : Codec::kTileBytes / 2;
    constexpr int HeaderUnits = Codec::kCpAsync ? Codec::kHeaderBytes / 16 : Codec::kHeaderBytes / 2;
    const int KTiles          = K / kI8BK;
    using Scale               = typename GgmlI8Smem<Codec, ExpertBN, ExpertBM>::Scale;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
#pragma unroll
    for (int mi = 0; mi < ExpertBM / 16; ++mi) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { acc[mi][e] = 0.0f; }
    }

    auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * kI8BK;
        for (int item = tid; item < ExpertBN * 4; item += ExpertThreads) {
            const int col           = item >> 2;
            const int k16           = item & 3;
            const bool valid        = col < cols;
            const std::int64_t arow = valid ? act_row(col) : 0;
            cp_async_zfill<16, Cache::cg>(&sm.Xs[stage][col * kI8Stride + 16 * k16],
                                          codes + arow * K + k0 + 16 * k16, valid ? 16 : 0);
        }
        for (int col = tid; col < ExpertBN; col += ExpertThreads) {
            const bool valid        = col < cols;
            const std::int64_t arow = valid ? act_row(col) : 0;
            cp_async_zfill<8, Cache::ca>(&sm.Ds[stage][col * 2], ds + arow * (K / 32) + kt * 2,
                                         valid ? 8 : 0);
        }
        for (int item = tid; item < ExpertBM * TileUnits; item += ExpertThreads) {
            const int row  = item / TileUnits;
            const int unit = item - row * TileUnits;
            stage_ggml_tile<Codec>(&sm.Cr[stage][row * Codec::kTileStride], row_base(row), kt,
                                   unit);
        }
        if (kt % kTilesPerBlock == 0) {
            const int slot = (kt / kTilesPerBlock) & 1;
            for (int item = tid; item < ExpertBM * HeaderUnits; item += ExpertThreads) {
                const int row  = item / HeaderUnits;
                const int unit = item - row * HeaderUnits;
                stage_ggml_header<Codec>(&sm.Hdr[slot][row * Codec::kHeaderStride],
                                         row_base(row), kt, unit, KTiles);
                if constexpr (Codec::kCover) {
                    if (unit == 0) {
                        sm.Off[slot][row] = static_cast<std::uint8_t>(cover_offset<Codec>(row_base(row), kt));
                    }
                }
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < kI8Stages; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

#pragma unroll 1
    for (int kt = 0; kt < KTiles; ++kt) {
        const int stage = kt & 1;
        const int tib   = kt % kTilesPerBlock;
        const int slot  = (kt / kTilesPerBlock) & 1;
        cp_wait<kI8Stages - 1>();
        __syncthreads();
        if (tib == 0) {
            for (int item = tid; item < Codec::kScaleGroups * ExpertBM; item += ExpertThreads) {
                const int row     = item & (ExpertBM - 1);
                const int idx     = item / ExpertBM;
                const int off     = Codec::kCover ? sm.Off[slot][row] : 0;
                const float2 pair = Codec::scale_pair(&sm.Hdr[slot][row * Codec::kHeaderStride] + off, idx);
                if constexpr (Codec::kHasMin) {
                    sm.Sc[slot][idx * ExpertBM + row] = pair;
                } else {
                    sm.Sc[slot][idx * ExpertBM + row] = pair.x;
                }
            }
        }
        for (int item = tid; item < ExpertBM * 8; item += ExpertThreads) {
            const int row = item >> 3;
            const int q   = item & 7;
            unsigned lo   = 0;
            unsigned hi   = 0;
            const int off = Codec::kCover ? sm.Off[slot][row] : 0;
            Codec::unpack(&sm.Cr[stage][row * Codec::kTileStride] + off,
                          &sm.Hdr[slot][row * Codec::kHeaderStride] + off, tib, q, lo, hi);
            *reinterpret_cast<unsigned*>(&sm.As[row * kI8Stride + 4 * q])      = lo;
            *reinterpret_cast<unsigned*>(&sm.As[row * kI8Stride + 32 + 4 * q]) = hi;
        }
        __syncthreads();

        if (warp * WarpCols < cols) {
            const int col0   = warp * WarpCols + 2 * lid;
            const uint2 dsw0 = *reinterpret_cast<const uint2*>(&sm.Ds[stage][col0 * 2]);
            const uint2 dsw1 = *reinterpret_cast<const uint2*>(&sm.Ds[stage][(col0 + 1) * 2]);
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                unsigned b0 = 0;
                unsigned b1 = 0;
                {
                    const int token = warp * WarpCols + (lane & 7);
                    const int cofs  = 32 * g + ((lane >> 3) & 1) * 16;
                    ldmatrix_x2(b0, b1, smem_addr(&sm.Xs[stage][token * kI8Stride + cofs]));
                }
                const unsigned w0 = g ? dsw0.y : dsw0.x;
                const unsigned w1 = g ? dsw1.y : dsw1.x;
                float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(&w0));
                float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(&w1));
                if constexpr (ExactActivationSum && Codec::kHasMin) {
                    // Four lanes sum one token's 32 codes. Reconstruct with the stored
                    // scale: rounding d*sum(q) to FP16 independently of d leaves a false
                    // residual even for an affine block whose weights are exactly zero.
                    const int token = warp * WarpCols + (lane & 7);
                    const int* values = reinterpret_cast<const int*>(
                        &sm.Xs[stage][token * kI8Stride + 32 * g + (lane >> 3) * 8]);
                    int sum = __dp4a(0x01010101, values[1],
                                    __dp4a(0x01010101, values[0], 0));
                    sum += __shfl_xor_sync(0xffffffffu, sum, 8);
                    sum += __shfl_xor_sync(0xffffffffu, sum, 16);
                    x0.y = static_cast<float>(__shfl_sync(0xffffffffu, sum, 2 * lid));
                    x1.y = static_cast<float>(__shfl_sync(0xffffffffu, sum, 2 * lid + 1));
                }
#pragma unroll
                for (int mi = 0; mi < ExpertBM / 16; ++mi) {
                    unsigned a[4];
                    {
                        const int local = mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                        const int cofs  = 32 * g + (lane >> 4) * 16;
                        ldmatrix_x4(a[0], a[1], a[2], a[3],
                                    smem_addr(&sm.As[local * kI8Stride + cofs]));
                    }
                    const int r0 = mi * 16 + gid;
                    const int r1 = r0 + 8;
                    auto apply = [&](int d0, int d1, int d2, int d3, const Scale& s0,
                                     const Scale& s1) {
                        if constexpr (Codec::kHasMin && ExactActivationSum) {
                            // Cancel the weight's affine terms before applying the common
                            // activation scale, retaining the integer sum until this point.
                            acc[mi][0] += x0.x * __fmaf_rn(s0.x, static_cast<float>(d0), -s0.y * x0.y);
                            acc[mi][1] += x1.x * __fmaf_rn(s0.x, static_cast<float>(d1), -s0.y * x1.y);
                            acc[mi][2] += x0.x * __fmaf_rn(s1.x, static_cast<float>(d2), -s1.y * x0.y);
                            acc[mi][3] += x1.x * __fmaf_rn(s1.x, static_cast<float>(d3), -s1.y * x1.y);
                        } else if constexpr (Codec::kHasMin) {
                            acc[mi][0] += (s0.x * x0.x) * static_cast<float>(d0) - s0.y * x0.y;
                            acc[mi][1] += (s0.x * x1.x) * static_cast<float>(d1) - s0.y * x1.y;
                            acc[mi][2] += (s1.x * x0.x) * static_cast<float>(d2) - s1.y * x0.y;
                            acc[mi][3] += (s1.x * x1.x) * static_cast<float>(d3) - s1.y * x1.y;
                        } else {
                            acc[mi][0] += (s0 * x0.x) * static_cast<float>(d0);
                            acc[mi][1] += (s0 * x1.x) * static_cast<float>(d1);
                            acc[mi][2] += (s1 * x0.x) * static_cast<float>(d2);
                            acc[mi][3] += (s1 * x1.x) * static_cast<float>(d3);
                        }
                    };
                    if constexpr (Codec::kScaleWidth == 32) {
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        mma_s8(d0, d1, d2, d3, a[0], a[1], a[2], a[3], b0, b1);
                        const int idx = 2 * tib + g;
                        apply(d0, d1, d2, d3, sm.Sc[slot][idx * ExpertBM + r0],
                              sm.Sc[slot][idx * ExpertBM + r1]);
                    } else {
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            mma_s8_k16(d0, d1, d2, d3, a[2 * h], a[2 * h + 1], h ? b1 : b0);
                            const int idx = 4 * tib + 2 * g + h;
                            apply(d0, d1, d2, d3, sm.Sc[slot][idx * ExpertBM + r0],
                                  sm.Sc[slot][idx * ExpertBM + r1]);
                        }
                    }
                }
            }
        }

        __syncthreads();
        const int next = kt + kI8Stages;
        if (next < KTiles) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();
    __syncthreads();
    }
}


} // namespace sinfer::ops::detail::ggml
