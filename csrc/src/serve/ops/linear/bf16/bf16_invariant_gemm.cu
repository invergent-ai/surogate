#include "ops/linear/bf16/bf16_invariant_gemm.h"

#include "api/ops/batch_invariant.h"
#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/kernel/func_attribute.cuh"

#include <cuda_bf16.h>

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

namespace sinfer::ops {
namespace {
std::atomic<bool> g_batch_invariant{false};
} // namespace

void set_batch_invariant(bool enabled) noexcept {
    g_batch_invariant.store(enabled, std::memory_order_relaxed);
}

bool batch_invariant() noexcept { return g_batch_invariant.load(std::memory_order_relaxed); }

} // namespace sinfer::ops

namespace sinfer::ops::detail {
namespace {

// Every schedule shares kBlockK and the k16 MMA step; that, not the tile, fixes the reduction.
constexpr int kBlockK = 64;

template <int BlockRows, int BlockCols, int WarpRows, int WarpCols, int Stages>
struct InvariantSchedule {
    static constexpr int kBlockRows = BlockRows;
    static constexpr int kBlockCols = BlockCols;
    static constexpr int kWarpRows  = WarpRows;
    static constexpr int kWarpCols  = WarpCols;
    static constexpr int kStages    = Stages;
    static constexpr int kWarpsM    = BlockRows / WarpRows;
    static constexpr int kWarpsN    = BlockCols / WarpCols;
    static constexpr int kThreads   = kWarpsM * kWarpsN * 32;
    static constexpr int kMmaRows   = WarpRows / 16;
    static constexpr int kMmaCols   = WarpCols / 8;
    static constexpr int kSharedBytes =
        Stages * (BlockRows + BlockCols) * kBlockK * static_cast<int>(sizeof(__nv_bfloat16));

    static_assert(BlockRows % WarpRows == 0 && BlockCols % WarpCols == 0);
    static_assert(WarpRows % 16 == 0 && WarpCols % 8 == 0);
    static_assert(Stages >= 2 && Stages <= 6);
    static_assert(kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
};

// Three tiles, chosen by measurement on an RTX 5090 over the Qwen3.5-0.8B projections (T = 1 to
// 2048): one warp on 16 rows by 8 tokens for decode-width rounds, which streams the weight with
// the most CTAs; 64x64 past 32 tokens once that still yields a wave of CTAs; 32x32 otherwise.
// A 128x128 tile was slower at every measured width. Every schedule runs the same per-element
// MMA chain, so the choice follows speed alone.
using Tile16x8  = InvariantSchedule<16, 8, 16, 8, 6>;
using Tile32x32 = InvariantSchedule<32, 32, 16, 16, 4>;
using Tile64x64 = InvariantSchedule<64, 64, 32, 32, 3>;

__device__ __forceinline__ int swizzled_col(int row, int col) {
    return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
}

template <class S, bool Accumulate>
__global__ __launch_bounds__(S::kThreads) void bf16_invariant_gemm_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out, std::int32_t rows, std::int32_t k, std::int32_t tokens,
    std::int32_t ldc) {
    constexpr int BM      = S::kBlockRows;
    constexpr int BN      = S::kBlockCols;
    constexpr int BK      = kBlockK;
    constexpr int WM      = S::kWarpRows;
    constexpr int WN      = S::kWarpCols;
    constexpr int MT      = S::kMmaRows;
    constexpr int NT      = S::kMmaCols;
    constexpr int KSUB    = BK / 16;
    constexpr int STAGES  = S::kStages;
    constexpr int WARPS_N = S::kWarpsN;
    constexpr int THREADS = S::kThreads;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* As = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* Bs = As + STAGES * BM * BK;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / WARPS_N;
    const int wn   = warp - wm * WARPS_N;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Token tiles of one row tile are adjacent in the launch order, so the weight tile they share
    // is read from L2 rather than DRAM by all but the first.
    const int tiles_n = (tokens + BN - 1) / BN;
    const int tile_m  = static_cast<int>(blockIdx.x) / tiles_n;
    const int tile_n  = static_cast<int>(blockIdx.x) - tile_m * tiles_n;
    const int m0      = tile_m * BM;
    const int n0      = tile_n * BN;
    const int k_tiles = (k + BK - 1) / BK;

    float accum[MT][NT][4] = {};

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    // Out-of-range rows, tokens and the k tail are zero-filled: a zero product leaves an FP32
    // accumulator unchanged, and every schedule pads k the same way.
    auto stage_inputs = [&](int stage, int k_tile) {
        const int k0  = k_tile * BK;
        auto* a_stage = As + stage * BM * BK;
        auto* b_stage = Bs + stage * BN * BK;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 8); item += THREADS) {
            const int row    = item / (BK / 8);
            const int kk     = (item - row * (BK / 8)) * 8;
            const bool valid = m0 + row < rows && k0 + kk < k;
            const auto* src =
                valid ? &weight[static_cast<std::int64_t>(m0 + row) * k + k0 + kk] : weight;
            cp_async_zfill<16, Cache::cg>(&a_stage[row * BK + swizzled_col(row, kk)], src,
                                          valid ? 16 : 0);
        }
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int col    = item / (BK / 8);
            const int kk     = (item - col * (BK / 8)) * 8;
            const bool valid = n0 + col < tokens && k0 + kk < k;
            const auto* src  = valid ? &x[static_cast<std::int64_t>(n0 + col) * k + k0 + kk] : x;
            cp_async_zfill<16, Cache::cg>(&b_stage[col * BK + swizzled_col(col, kk)], src,
                                          valid ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }

#pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % STAGES;
        // Refill the slot consumed last iteration before waiting on this one.
        const int next = k_tile + STAGES - 1;
        if (next < k_tiles) { stage_inputs(next % STAGES, next); }
        cp_commit();
        cp_wait<STAGES - 1>();
        __syncthreads();

        const auto* a_stage = As + stage * BM * BK;
        const auto* b_stage = Bs + stage * BN * BK;
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            unsigned a_frag[MT][4];
            unsigned b_frag[NT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + a_row_offset;
                const int col = k_step * 16 + a_col_offset;
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            smem_addr(&a_stage[row * BK + swizzled_col(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = wn * WN + ni * 8 + b_inner_row;
                const int col = k_step * 16 + b_k_offset;
                ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                            smem_addr(&b_stage[row * BK + swizzled_col(row, col)]));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2], accum[mi][ni][3],
                             a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                             b_frag[ni][0], b_frag[ni][1]);
                }
            }
        }
        __syncthreads();
    }
    cp_wait<0>();

    const auto store = [&](int row, int token, float value) {
        if (row >= rows || token >= tokens) { return; }
        __nv_bfloat16* at = out + static_cast<std::int64_t>(token) * ldc + row;
        if constexpr (Accumulate) { value += __bfloat162float(*at); }
        *at = __float2bfloat16_rn(value);
    };
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row0 = m0 + wm * WM + mi * 16 + gid;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int token0 = n0 + wn * WN + ni * 8 + 2 * lid;
            store(row0, token0, accum[mi][ni][0]);
            store(row0, token0 + 1, accum[mi][ni][1]);
            store(row0 + 8, token0, accum[mi][ni][2]);
            store(row0 + 8, token0 + 1, accum[mi][ni][3]);
        }
    }
}

template <class S, bool Accumulate>
void launch(const void* weight, std::int32_t n, std::int32_t k, const void* x, std::int32_t tokens,
            void* out, std::int32_t ldc, cudaStream_t stream) {
    const std::int64_t tiles_m = (n + S::kBlockRows - 1) / S::kBlockRows;
    const std::int64_t tiles_n = (tokens + S::kBlockCols - 1) / S::kBlockCols;
    const std::int64_t blocks  = tiles_m * tiles_n;
    if (blocks > 0x7fffffff) { throw std::invalid_argument("bf16 invariant GEMM: grid too large"); }
    if constexpr (S::kSharedBytes > 48 * 1024) {
        CUDA_CHECK(set_func_attribute_per_device(bf16_invariant_gemm_kernel<S, Accumulate>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 S::kSharedBytes));
    }
    bf16_invariant_gemm_kernel<S, Accumulate>
        <<<static_cast<unsigned>(blocks), S::kThreads, S::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(weight),
            static_cast<__nv_bfloat16*>(out), n, k, tokens, ldc);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Accumulate>
void dispatch(const void* weight, std::int32_t n, std::int32_t k, const void* x,
              std::int32_t tokens, void* out, std::int32_t ldc, cudaStream_t stream) {
    // SUROGATE_SERVE_INVARIANT_TILE=1|2|3 pins one schedule (16x8, 32x32, 64x64), for
    // measurement: the bits are the same whichever runs.
    static const int kPinned = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_INVARIANT_TILE");
        return raw != nullptr ? std::atoi(raw) : 0;
    }();
    int tile = kPinned;
    if (tile < 1 || tile > 3) {
        const std::int64_t wide_tiles =
            static_cast<std::int64_t>((n + 63) / 64) * ((tokens + 63) / 64);
        tile = tokens <= 8 ? 1 : (tokens > 32 && wide_tiles >= 128 ? 3 : 2);
    }
    switch (tile) {
    case 1:
        launch<Tile16x8, Accumulate>(weight, n, k, x, tokens, out, ldc, stream);
        return;
    case 2:
        launch<Tile32x32, Accumulate>(weight, n, k, x, tokens, out, ldc, stream);
        return;
    default:
        launch<Tile64x64, Accumulate>(weight, n, k, x, tokens, out, ldc, stream);
        return;
    }
}

} // namespace

bool bf16_invariant_gemm_supports(std::int32_t n, std::int32_t k, std::int32_t ldc) noexcept {
    return n > 0 && k > 0 && (k % 8) == 0 && ldc >= n;
}

void bf16_invariant_gemm(const void* weight, std::int32_t n, std::int32_t k, const void* x,
                         std::int32_t tokens, void* out, std::int32_t ldc, bool accumulate,
                         cudaStream_t stream) {
    const auto aligned = [](const void* pointer) {
        return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & 15u) == 0;
    };
    if (!bf16_invariant_gemm_supports(n, k, ldc) || tokens <= 0 || !aligned(weight) ||
        !aligned(x) || out == nullptr) {
        throw std::invalid_argument("bf16 invariant GEMM: k must be a positive multiple of 8, "
                                    "ldc >= n, and weight/x 16-byte aligned");
    }
    if (accumulate) {
        dispatch<true>(weight, n, k, x, tokens, out, ldc, stream);
    } else {
        dispatch<false>(weight, n, k, x, tokens, out, ldc, stream);
    }
}

} // namespace sinfer::ops::detail
