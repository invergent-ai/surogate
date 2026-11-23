// Batch- and path-invariant BF16 GEMM for the inkling2 target (api/ops/inkling_mix.h, agent-EN 10:2x decision).
//
// The per-element chain is the one of ops/linear/bf16/bf16_invariant_gemm.cu (k16 MMA steps in increasing k from a
// zero accumulator, k tiles of 64 zero-padded, no split-K by width, no cross-warp reduction); this copy adds what the
// inkling2 GEMMs need and the shared kernel does not have: an FP32 store (the controller gain / forget and the LM
// head), a batch of same-shape problems in one launch (the 3 streams of a blend), and a split of k into chunks FIXED
// by (n, k) -- each chunk its own chain, the FP32 partials added in chunk order by a second kernel -- for the small-n,
// long-k shapes that one chain per element leaves on a handful of CTAs (controller `down` 64 x 4608: 4 CTAs).

#include "api/ops/inkling_mix.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/mma.cuh"
#include "ops/kernel/func_attribute.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

constexpr int kBlockK     = 64;  // every schedule; with the k16 MMA step this fixes the chain
constexpr int kSplitChunk = 256; // k per chunk of a split shape (a multiple of kBlockK)

template <int BlockRows, int BlockCols, int WarpRows, int WarpCols, int Stages>
struct Schedule {
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
    static_assert(kSharedBytes <= 99 * 1024);
};

// The shared kernel's three tiles (measured there on an RTX 5090); the choice follows speed only.
using Tile16x8  = Schedule<16, 8, 16, 8, 6>;
// Decode widths with few row tiles (wo_ud: 144 one-warp CTAs on 170 SMs walked k = 3072 at ~0.84 TB/s, lane h's nsys
// 14:49): the same tile with 7 stages in flight instead of 5. The pipeline depth moves no product: same bits.
using Tile16x8Deep = Schedule<16, 8, 16, 8, 8>; // the loop waits for STAGES - 1 groups; PTX takes at most 7
using Tile32x32 = Schedule<32, 32, 16, 16, 4>;
using Tile64x64 = Schedule<64, 64, 32, 32, 3>;

enum class Store : int { Bf16 = 0, Fp32 = 1, Partial = 2 };

constexpr int kMaxGroup = 6;

struct Args {
    const __nv_bfloat16* weight; // [batch][n][k]
    const __nv_bfloat16* x;      // [batch][tokens][k]
    void* out;                   // Bf16 / Fp32: [batch][tokens][ldc]; Partial: FP32 [split][batch][tokens][n]
    std::int32_t rows, k, tokens, ldc, chunk, batch;
    std::int64_t weight_stride, x_stride, out_stride;
    // Group mode (grouped != 0): problem b = blockIdx.y reads these instead of the strided batch (never split).
    std::int32_t grouped;
    const __nv_bfloat16* g_weight[kMaxGroup];
    const __nv_bfloat16* g_x[kMaxGroup];
    void* g_out[kMaxGroup];
    std::int32_t g_rows[kMaxGroup];
    std::int32_t g_ldc[kMaxGroup];
};

__device__ __forceinline__ int swizzled_col(int row, int col) {
    return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
}

template <class S, Store kStore>
__global__ __launch_bounds__(S::kThreads) void gemm_inv_kernel(const __grid_constant__ Args a) {
    pdl::wait_for_dependencies(); // PDL: every read / write after the producer grid completes
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    constexpr int BM = S::kBlockRows, BN = S::kBlockCols, BK = kBlockK;
    constexpr int WM = S::kWarpRows, WN = S::kWarpCols, MT = S::kMmaRows, NT = S::kMmaCols;
    constexpr int KSUB = BK / 16, STAGES = S::kStages, WARPS_N = S::kWarpsN, THREADS = S::kThreads;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* As = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* Bs = As + STAGES * BM * BK;

    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp - wm * WARPS_N, gid = lane >> 2, lid = lane & 3;

    const int b       = static_cast<int>(blockIdx.y);
    const int split   = static_cast<int>(blockIdx.z);
    const auto* weight = a.grouped ? a.g_weight[b] : a.weight + static_cast<std::int64_t>(b) * a.weight_stride;
    const auto* x      = a.grouped ? a.g_x[b] : a.x + static_cast<std::int64_t>(b) * a.x_stride;
    const int rows     = a.grouped ? a.g_rows[b] : a.rows;
    const int tiles_n = (a.tokens + BN - 1) / BN;
    const int tile_m  = static_cast<int>(blockIdx.x) / tiles_n;
    const int tile_n  = static_cast<int>(blockIdx.x) - tile_m * tiles_n;
    const int m0 = tile_m * BM, n0 = tile_n * BN;
    if (m0 >= rows) { return; } // a group's smaller problem: the whole CTA leaves before any barrier
    // This CTA's chain: k in [k_begin, k_end), from zero.
    const int k_begin = split * a.chunk;
    const int k_end   = min(a.k, k_begin + a.chunk);
    const int k_tiles = (k_end - k_begin + BK - 1) / BK;

    float accum[MT][NT][4] = {};

    const int a_matrix = lane >> 3, a_inner_row = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3), a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row = lane & 7, b_k_offset = ((lane >> 3) & 1) << 3;

    auto stage_inputs = [&](int stage, int k_tile) {
        const int k0  = k_begin + k_tile * BK;
        auto* a_stage = As + stage * BM * BK;
        auto* b_stage = Bs + stage * BN * BK;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 8); item += THREADS) {
            const int row = item / (BK / 8), kk = (item - row * (BK / 8)) * 8;
            const bool valid = m0 + row < rows && k0 + kk < k_end;
            const auto* src  = valid ? &weight[static_cast<std::int64_t>(m0 + row) * a.k + k0 + kk] : weight;
            cp_async_zfill<16, Cache::cg>(&a_stage[row * BK + swizzled_col(row, kk)], src, valid ? 16 : 0);
        }
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int col = item / (BK / 8), kk = (item - col * (BK / 8)) * 8;
            const bool valid = n0 + col < a.tokens && k0 + kk < k_end;
            const auto* src  = valid ? &x[static_cast<std::int64_t>(n0 + col) * a.k + k0 + kk] : x;
            cp_async_zfill<16, Cache::cg>(&b_stage[col * BK + swizzled_col(col, kk)], src, valid ? 16 : 0);
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
        const int next  = k_tile + STAGES - 1;
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
                const int row = wm * WM + mi * 16 + a_row_offset, col = k_step * 16 + a_col_offset;
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            smem_addr(&a_stage[row * BK + swizzled_col(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = wn * WN + ni * 8 + b_inner_row, col = k_step * 16 + b_k_offset;
                ldmatrix_x2(b_frag[ni][0], b_frag[ni][1], smem_addr(&b_stage[row * BK + swizzled_col(row, col)]));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2], accum[mi][ni][3], a_frag[mi][0],
                             a_frag[mi][1], a_frag[mi][2], a_frag[mi][3], b_frag[ni][0], b_frag[ni][1]);
                }
            }
        }
        __syncthreads();
    }
    cp_wait<0>();

    const auto store = [&](int row, int token, float value) {
        if (row >= rows || token >= a.tokens) { return; }
        if constexpr (kStore == Store::Partial) {
            float* p = static_cast<float*>(a.out);
            p[((static_cast<std::int64_t>(split) * a.batch + b) * a.tokens + token) * a.rows + row] = value;
        } else {
            void* base            = a.grouped ? a.g_out[b] : a.out;
            const std::int64_t at = (a.grouped ? 0 : static_cast<std::int64_t>(b) * a.out_stride) +
                                    static_cast<std::int64_t>(token) * (a.grouped ? a.g_ldc[b] : a.ldc) + row;
            if constexpr (kStore == Store::Fp32) {
                static_cast<float*>(base)[at] = value;
            } else {
                static_cast<__nv_bfloat16*>(base)[at] = __float2bfloat16_rn(value);
            }
        }
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

/// out = p[0] + p[1] + ... + p[splits - 1], added in that order, one thread per output element. The partials are read
/// through L2 (ld.global.cg), never the non-coherent read-only path: they are written by the split GEMM just before this
/// kernel in the PDL chain (this grid may be resident while it runs) and the visit arena reuses their address every
/// visit, which breaks ld.global.nc's read-only-for-the-kernel's-lifetime contract (lane c's FP8 decode finding 17:47).
template <typename OutT>
__global__ void __launch_bounds__(256) reduce_kernel(const float* partial, OutT* __restrict__ out,
                                                     std::int32_t rows, std::int32_t tokens, std::int32_t batch,
                                                     std::int32_t splits, std::int32_t ldc, std::int64_t out_stride) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    const std::int64_t total = static_cast<std::int64_t>(batch) * tokens * rows;
    const std::int64_t i     = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) { return; }
    const std::int64_t plane = static_cast<std::int64_t>(batch) * tokens * rows;
    float sum                = __ldcg(partial + i);
    for (int s = 1; s < splits; ++s) { sum += __ldcg(partial + s * plane + i); }
    const std::int32_t row   = static_cast<std::int32_t>(i % rows);
    const std::int64_t rest  = i / rows;
    const std::int32_t token = static_cast<std::int32_t>(rest % tokens);
    const std::int32_t b     = static_cast<std::int32_t>(rest / tokens);
    const std::int64_t at    = static_cast<std::int64_t>(b) * out_stride + static_cast<std::int64_t>(token) * ldc + row;
    if constexpr (sizeof(OutT) == 4) {
        out[at] = sum;
    } else {
        out[at] = __float2bfloat16_rn(sum);
    }
}

// ---- wo at decode widths (lanes/e1/NOTES.md 'wo BF16 KERNEL'; default OFF, SUROGATE_SERVE_INKLING_WO_NARROW=1 = on) -----
// gemm_inv_kernel<Tile16x8Deep, Bf16> for T <= 8, one problem, n % 16 == 0, k % 64 == 0: the same one-warp 16 x 8 tile, the
// same copy pipeline, the same smem placement, the same m16n8k16 chain and the same store, so the same bits. What differs
// is when the addresses are computed: each thread's 4 weight + 2 activation pieces and its 8 fragment offsets once, before
// the loop (the generic loop spends 178 of its 241 instructions per 64-k tile on them), and __syncwarp in place of a CTA
// barrier, the CTA being one warp.
struct NarrowArgs {
    const __nv_bfloat16* weight; // [n][k]
    const __nv_bfloat16* x;      // [tokens][k]
    __nv_bfloat16* out;          // [tokens][ldc]
    std::int32_t k, tokens, ldc;
};

template <int STAGES>
__global__ __launch_bounds__(32) void gemm_inv_narrow_kernel(const __grid_constant__ NarrowArgs a) {
    pdl::wait_for_dependencies(); // PDL: every read / write after the producer grid completes
    pdl::trigger_dependents(); // PDL: the next kernel may launch now (it waits for this grid before any read)
    constexpr int BM = 16, BN = 8, BK = kBlockK;
    __shared__ __align__(16) __nv_bfloat16 As[STAGES * BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[STAGES * BN * BK];

    const int lane = static_cast<int>(threadIdx.x), gid = lane >> 2, lid = lane & 3;
    const int m0      = static_cast<int>(blockIdx.x) * BM;
    const int k_tiles = a.k / BK;
    const int piece   = (lane & 7) * 8; // this thread's 16-byte piece of a row's 64-k tile

    const __nv_bfloat16* a_src[4];
    int a_dst[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int row = (lane >> 3) + 4 * j;
        a_src[j]      = a.weight + static_cast<std::int64_t>(m0 + row) * a.k + piece;
        a_dst[j]      = row * BK + swizzled_col(row, piece);
    }
    const __nv_bfloat16* b_src[2];
    int b_dst[2], b_bytes[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int col    = (lane >> 3) + 4 * j;
        const bool valid = col < a.tokens;
        // A column past T zero-fills: no byte is read (the address stays inside x's first token).
        b_src[j]   = a.x + static_cast<std::int64_t>(valid ? col : 0) * a.k + piece;
        b_dst[j]   = col * BK + swizzled_col(col, piece);
        b_bytes[j] = valid ? 16 : 0;
    }
    const int a_matrix = lane >> 3, a_row = (lane & 7) + ((a_matrix & 1) << 3), a_col = (a_matrix >> 1) << 3;
    const int b_row = lane & 7, b_col = ((lane >> 3) & 1) << 3;
    int a_frag[4], b_frag[4];
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        a_frag[step] = a_row * BK + swizzled_col(a_row, step * 16 + a_col);
        b_frag[step] = b_row * BK + swizzled_col(b_row, step * 16 + b_col);
    }

    const auto stage_inputs = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
        auto* as     = As + stage * BM * BK;
        auto* bs     = Bs + stage * BN * BK;
#pragma unroll
        for (int j = 0; j < 4; ++j) { cp_async<16, Cache::cg>(as + a_dst[j], a_src[j] + k0); }
#pragma unroll
        for (int j = 0; j < 2; ++j) { cp_async_zfill<16, Cache::cg>(bs + b_dst[j], b_src[j] + k0, b_bytes[j]); }
    };
#pragma unroll
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }
    float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
#pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % STAGES;
        const int next  = k_tile + STAGES - 1;
        if (next < k_tiles) { stage_inputs(next % STAGES, next); }
        cp_commit();
        cp_wait<STAGES - 1>();
        __syncwarp();
        const auto* as = As + stage * BM * BK;
        const auto* bs = Bs + stage * BN * BK;
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            unsigned f0, f1, f2, f3, g0, g1;
            ldmatrix_x4(f0, f1, f2, f3, smem_addr(as + a_frag[step]));
            ldmatrix_x2(g0, g1, smem_addr(bs + b_frag[step]));
            mma_bf16(c0, c1, c2, c3, f0, f1, f2, f3, g0, g1);
        }
        __syncwarp();
    }
    cp_wait<0>();

    const int row0 = m0 + gid, token0 = 2 * lid;
    const auto store = [&](int row, int token, float value) {
        if (token >= a.tokens) { return; }
        a.out[static_cast<std::int64_t>(token) * a.ldc + row] = __float2bfloat16_rn(value);
    };
    store(row0, token0, c0);
    store(row0, token0 + 1, c1);
    store(row0 + 8, token0, c2);
    store(row0 + 8, token0 + 1, c3);
}

// The switch: default OFF; SUROGATE_SERVE_INKLING_WO_NARROW=1 or inkling_gemm_inv_use_narrow(true) = the narrow kernel (same bits).
// MTP off on M11: 1x8k +3.57 %, 8x8k +2.89 %. MTP on at 8x8k on M13 (the served shape): gain -0.13 %, 95 % CI [-0.73, +0.48] %, which
// the registered line reads as not passed (lanes/e1/NOTES.md 'STEP B RESULT').
std::atomic<int> g_wo_narrow{-1};

bool use_wo_narrow() {
    int v = g_wo_narrow.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("SUROGATE_SERVE_INKLING_WO_NARROW");
        v             = e != nullptr && e[0] == '1' && e[1] == '\0' ? 1 : 0;
        g_wo_narrow.store(v, std::memory_order_relaxed);
    }
    return v != 0;
}

/// The shapes whose generic launch is gemm_inv_kernel<Tile16x8Deep, Bf16> with full row tiles and whole k tiles.
bool wo_narrow_shape(std::int32_t n, std::int32_t k, std::int32_t tokens, std::int32_t batch) {
    static const int sms = [] {
        int device = 0, count = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            return 0;
        }
        return count;
    }();
    return batch == 1 && tokens <= 8 && n % 16 == 0 && k % kBlockK == 0 && n / 16 < 2 * sms;
}

template <class S, Store kStore>
void launch(const Args& a, std::int32_t splits, cudaStream_t stream) {
    const std::int64_t tiles = static_cast<std::int64_t>((a.rows + S::kBlockRows - 1) / S::kBlockRows) *
                               ((a.tokens + S::kBlockCols - 1) / S::kBlockCols);
    if (tiles > 0x7fffffff || a.batch > 65535 || splits > 65535) {
        throw std::invalid_argument("inkling_gemm_inv: grid too large");
    }
    if constexpr (S::kSharedBytes > 48 * 1024) {
        CUDA_CHECK(set_func_attribute_per_device(gemm_inv_kernel<S, kStore>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 S::kSharedBytes));
    }
    const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(a.batch), static_cast<unsigned>(splits));
    CUDA_CHECK(pdl::launch_dependent({grid, dim3(S::kThreads), static_cast<std::size_t>(S::kSharedBytes), stream},
                                     gemm_inv_kernel<S, kStore>, a));
    CUDA_CHECK(cudaGetLastError());
}

template <Store kStore>
void dispatch(const Args& a, std::int32_t splits, cudaStream_t stream) {
    // The shared route's rule (bf16_invariant_gemm.cu): one warp per 16 x 8 at decode widths, 64 x 64 once that still
    // gives a wave of CTAs, 32 x 32 otherwise. Counted over the whole launch (batch x splits). Speed only.
    const std::int64_t wide_tiles = static_cast<std::int64_t>((a.rows + 63) / 64) * ((a.tokens + 63) / 64) * a.batch * splits;
    const int tile                = a.tokens <= 8 ? 1 : (a.tokens > 32 && wide_tiles >= 128 ? 3 : 2);
    static const int sms          = [] {
        int device = 0, count = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            return 0;
        }
        return count;
    }();
    const std::int64_t narrow_ctas = static_cast<std::int64_t>((a.rows + 15) / 16) * a.batch * splits;
    switch (tile) {
    case 1:
        if (narrow_ctas < 2LL * sms) {
            launch<Tile16x8Deep, kStore>(a, splits, stream);
        } else {
            launch<Tile16x8, kStore>(a, splits, stream);
        }
        return;
    case 2: launch<Tile32x32, kStore>(a, splits, stream); return;
    default: launch<Tile64x64, kStore>(a, splits, stream); return;
    }
}

} // namespace

std::int32_t inkling_gemm_split(std::int32_t n, std::int32_t k) noexcept {
    // Small n and long k only (the controller `down`, the dynamic taps): one chain per element would leave
    // n / 16 CTAs walking k. A function of (n, k) alone.
    if (n > 64 || k < 4 * kSplitChunk || k % kSplitChunk != 0) { return 1; }
    return k / kSplitChunk;
}

std::size_t inkling_gemm_inv_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                             std::int32_t batch) noexcept {
    const std::int32_t splits = inkling_gemm_split(n, k);
    if (splits <= 1 || tokens <= 0) { return 0; }
    return static_cast<std::size_t>(splits) * std::max(batch, 1) * tokens * n * sizeof(float);
}

void inkling_gemm_inv(const void* weight, std::int32_t n, std::int32_t k, const void* x, std::int32_t tokens,
                      void* out, DType out_dtype, std::int32_t ldc, const InklingGemmBatch& batch, void* workspace,
                      std::size_t workspace_bytes, cudaStream_t stream) {
    if (tokens <= 0 || batch.count <= 0) { return; }
    const auto aligned = [](const void* p) { return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    if (n <= 0 || k <= 0 || k % 8 != 0 || ldc < n || !aligned(weight) || !aligned(x) || out == nullptr ||
        (out_dtype != DType::BF16 && out_dtype != DType::FP32) ||
        (batch.count > 1 && (batch.weight_stride % 8 != 0 || batch.x_stride % 8 != 0))) {
        throw std::invalid_argument("inkling_gemm_inv: k must be a positive multiple of 8, ldc >= n, weight / x "
                                    "16-byte aligned (batch strides too), out BF16 or FP32");
    }
    const std::int32_t splits = inkling_gemm_split(n, k);
    Args a{static_cast<const __nv_bfloat16*>(weight), static_cast<const __nv_bfloat16*>(x), out, n, k, tokens, ldc,
           splits > 1 ? kSplitChunk : k, batch.count, batch.weight_stride, batch.x_stride, batch.out_stride};
    if (splits <= 1) {
        a.chunk = ((k + kBlockK - 1) / kBlockK) * kBlockK; // one chunk: the whole (padded) k
        if (out_dtype == DType::FP32) {
            dispatch<Store::Fp32>(a, 1, stream);
        } else if (use_wo_narrow() && wo_narrow_shape(n, k, tokens, batch.count)) {
            const NarrowArgs na{static_cast<const __nv_bfloat16*>(weight), static_cast<const __nv_bfloat16*>(x),
                                static_cast<__nv_bfloat16*>(out), k, tokens, ldc};
            CUDA_CHECK(pdl::launch_dependent({dim3(static_cast<unsigned>(n / 16)), dim3(32), 0, stream},
                                             gemm_inv_narrow_kernel<Tile16x8Deep::kStages>, na));
            CUDA_CHECK(cudaGetLastError());
        } else {
            dispatch<Store::Bf16>(a, 1, stream);
        }
        return;
    }
    const std::size_t need = inkling_gemm_inv_workspace_bytes(n, k, tokens, batch.count);
    if (!aligned(workspace) || workspace_bytes < need) {
        throw std::invalid_argument("inkling_gemm_inv: a split shape (" + std::to_string(n) + " x " + std::to_string(k) +
                                    ") needs " + std::to_string(need) + " bytes of workspace");
    }
    Args p = a;
    p.out  = workspace;
    dispatch<Store::Partial>(p, splits, stream);
    const std::int64_t total = static_cast<std::int64_t>(batch.count) * tokens * n;
    const unsigned blocks    = static_cast<unsigned>((total + 255) / 256);
    if (out_dtype == DType::FP32) {
        CUDA_CHECK(pdl::launch_dependent({dim3(blocks), dim3(256), 0, stream}, reduce_kernel<float>,
                                         static_cast<const float*>(workspace), static_cast<float*>(out), n, tokens,
                                         batch.count, splits, ldc, batch.out_stride));
    } else {
        CUDA_CHECK(pdl::launch_dependent({dim3(blocks), dim3(256), 0, stream}, reduce_kernel<__nv_bfloat16>,
                                         static_cast<const float*>(workspace), static_cast<__nv_bfloat16*>(out), n,
                                         tokens, batch.count, splits, ldc, batch.out_stride));
    }
    CUDA_CHECK(cudaGetLastError());
}

void inkling_gemm_inv_partials(const void* weight, std::int32_t n, std::int32_t k, const void* x, std::int32_t tokens,
                               const InklingGemmBatch& batch, float* partials, std::size_t partials_bytes,
                               cudaStream_t stream) {
    if (tokens <= 0 || batch.count <= 0) { return; }
    const auto aligned = [](const void* p) { return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    const std::int32_t splits = inkling_gemm_split(n, k);
    if (splits <= 1 || k % 8 != 0 || !aligned(weight) || !aligned(x) || !aligned(partials) ||
        partials_bytes < inkling_gemm_inv_workspace_bytes(n, k, tokens, batch.count) ||
        (batch.count > 1 && (batch.weight_stride % 8 != 0 || batch.x_stride % 8 != 0))) {
        throw std::invalid_argument("inkling_gemm_inv_partials: a split shape, aligned operands and the partials buffer");
    }
    Args a{static_cast<const __nv_bfloat16*>(weight), static_cast<const __nv_bfloat16*>(x), partials, n, k, tokens, n,
           kSplitChunk, batch.count, batch.weight_stride, batch.x_stride, 0};
    dispatch<Store::Partial>(a, splits, stream);
}

void inkling_gemm_inv_use_narrow(bool on) noexcept { g_wo_narrow.store(on ? 1 : 0, std::memory_order_relaxed); }

void inkling_gemm_inv_group(std::span<const InklingGemmProblem> problems, std::int32_t k, std::int32_t tokens,
                            DType out_dtype, cudaStream_t stream) {
    if (problems.empty() || tokens <= 0) { return; }
    const auto aligned = [](const void* p) { return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    if (problems.size() > static_cast<std::size_t>(kMaxGroup) || k <= 0 || k % 8 != 0 ||
        (out_dtype != DType::BF16 && out_dtype != DType::FP32)) {
        throw std::invalid_argument("inkling_gemm_inv_group: 1-6 problems, k a positive multiple of 8, out BF16 or FP32");
    }
    Args a{};
    a.k       = k;
    a.tokens  = tokens;
    a.chunk   = ((k + kBlockK - 1) / kBlockK) * kBlockK;
    a.batch   = static_cast<std::int32_t>(problems.size());
    a.grouped = 1;
    a.rows    = 0;
    for (std::size_t i = 0; i < problems.size(); ++i) {
        const InklingGemmProblem& p = problems[i];
        if (p.n <= 0 || p.ldc < p.n || !aligned(p.weight) || !aligned(p.x) || p.out == nullptr) {
            throw std::invalid_argument("inkling_gemm_inv_group: a problem needs n > 0, ldc >= n, aligned weight / x");
        }
        a.g_weight[i] = static_cast<const __nv_bfloat16*>(p.weight);
        a.g_x[i]      = static_cast<const __nv_bfloat16*>(p.x);
        a.g_out[i]    = p.out;
        a.g_rows[i]   = p.n;
        a.g_ldc[i]    = p.ldc;
        a.rows        = std::max(a.rows, p.n); // the grid covers the largest problem
    }
    if (out_dtype == DType::FP32) {
        dispatch<Store::Fp32>(a, 1, stream);
    } else {
        dispatch<Store::Bf16>(a, 1, stream);
    }
}

} // namespace sinfer::ops
