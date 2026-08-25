#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/common.cuh"

#include <cmath>

// Stage 2 (fused): build T_inv in smem, immediately consume it to produce
// W and U. T_inv never crosses HBM or occupies workspace.
//
// Math + I/O layouts: see the Gated DeltaNet chunked prepare_wy_wu_config.
// The wide route (including H_v=48) uses 8 warps and 28.75 KiB (3 CTAs/SM);
// H_v=32 uses 4 warps and 23.75 KiB (4 CTAs/SM). Both retain a 16 KiB FP32
// T_inv and 0.75 KiB of controls.

namespace ninfer::ops::detail::gated_delta_net::chunked::prepare_wy_wu {

using ninfer::ops::mma_tf32;
using ninfer::ops::Cache;
using ninfer::ops::cp_async;
using ninfer::ops::cp_commit;
using ninfer::ops::cp_wait;
using ninfer::ops::ldmatrix_x2;
using ninfer::ops::ldmatrix_x4;
using ninfer::ops::mma_bf16;
using ninfer::ops::smem_addr;

static_assert(kChunkSize == 64, "stage_prepare_wy_wu: kChunkSize must be 64 (kernel hard-codes "
                                "BT=64 = 4 * BC=16)");
static_assert(kStateDim == 128);

constexpr int N_SUB      = BT / BC;    // 4
constexpr int WY_WARPS   = N_SUB;      // 4 warps own the triangular construction
constexpr int N_K_TILES  = BT / MMA_K; // 8 (recompute_wu)
constexpr int BF16_MMA_K = 16;

static_assert(MMA_M == BC, "kernel assumes MMA m == BC");

// Phase D scratch row stride. Smallest value >=16 (per-warp data needs
// 16 cols) that preserves the 2-way-write / 0-way-read property:
//   * stride%32 = 20 -> (20*lane_g + 2*lane_t) lands all 32 lanes on 16
//     distinct even banks (strict 2-way write); cumulative shifts
//     {0,20,8,28,16,4,24,12} are all distinct mod 32.
//   * Reads (20*lane_g + lane_t) span 32 distinct banks (0-way read).
// Equivalent BC to stride=36 but saves 1024B of scratch_floats.
constexpr int SCR_STRIDE = 20;

template <int KPanelCols, int WuPanelCols>
struct kernel_dims {
    static_assert(KPanelCols == 32 || KPanelCols == 64);
    static_assert(WuPanelCols == 16 || WuPanelCols == 32);
    static_assert(KPanelCols % BF16_MMA_K == 0);
    static_assert(WuPanelCols % MMA_N == 0);

    static constexpr int N_K_PANELS        = kStateDim / KPanelCols;
    static constexpr int K_TILES_PER_PANEL = KPanelCols / BF16_MMA_K;
    static constexpr int N_WU_PANELS       = kStateDim / WuPanelCols;

    static constexpr int T_inv_floats    = BT * BT;                    // 16 KiB
    static constexpr int scratch_floats  = WY_WARPS * BC * SCR_STRIDE; // 5 KiB
    static constexpr int wu_stage_floats = BT * WuPanelCols;
    static constexpr int k_stage_bf16    = BT * KPanelCols;
    static constexpr int k_stage_floats  = k_stage_bf16 / 2;
    static constexpr int stage_floats =
        scratch_floats > wu_stage_floats ? scratch_floats : wu_stage_floats;
    static constexpr int output_stage_floats = BT * WuPanelCols / 2;
    static_assert(stage_floats >= k_stage_floats);
    static_assert(stage_floats >= scratch_floats);

    static constexpr int g_floats    = BT;
    static constexpr int beta_floats = BT;
    static constexpr int bg_floats   = BT;
    static constexpr int SMEM_FLOATS =
        T_inv_floats + stage_floats + output_stage_floats + g_floats + beta_floats + bg_floats;
};

template <int STRIDE>
struct Bf16SmemTile {
    __nv_bfloat16* __restrict__ base;
    static_assert(STRIDE == 32 || STRIDE == 64);

    __device__ __forceinline__ int swizzled_col(int row, int col) const {
        return col ^ ((row & (STRIDE / 8 - 1)) << 3);
    }

    __device__ __forceinline__ __nv_bfloat16* ptr(int row, int col) const {
        return base + row * STRIDE + swizzled_col(row, col);
    }
};

template <int Stride>
__device__ __forceinline__ int wu_output_swizzled_col(int row, int col) {
    return (((col >> 3) ^ (row & (Stride / 8 - 1))) << 3) | (col & 7);
}

// ---------------------------------------------------------------------------
// Phase D helpers. Refactored from in-kernel lambdas to namespace-scope
// __device__ functions to drop the spill-via-lambda-capture pattern that
// nvcc's register allocator could not see through.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
scatter_frag_to_scr(const float frag[8], float* __restrict__ scr_smem, int warp, int lane) {
    float* Sptr                                  = scr_smem + warp * BC * SCR_STRIDE;
    const int lane_g                             = lane >> 2;
    const int col_2t                             = (lane & 3) << 1;
    Sptr[lane_g * SCR_STRIDE + col_2t]           = frag[0];
    Sptr[lane_g * SCR_STRIDE + col_2t + 1]       = frag[1];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t]     = frag[2];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 1] = frag[3];
    Sptr[lane_g * SCR_STRIDE + col_2t + 8]       = frag[4];
    Sptr[lane_g * SCR_STRIDE + col_2t + 9]       = frag[5];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 8] = frag[6];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 9] = frag[7];
}

// 16x16x16 mma: A from raw row-major scratch (stride SCR_STRIDE),
// B from swizzled M_view at offset (M_row_off, M_col_off).
__device__ __forceinline__ void mma16_raw_x_swiz(float D[8], int lane,
                                                 const float* __restrict__ A_buf,
                                                 SmemTile<BT> M_view, int M_row_off,
                                                 int M_col_off) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0  = A_buf[lane_g * SCR_STRIDE + (k_off + lane_t)];
        const float a1  = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t)];
        const float a2  = A_buf[lane_g * SCR_STRIDE + (k_off + lane_t + 4)];
        const float a3  = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t + 4)];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0  = M_view.at(M_row_off + k_off + lane_t, M_col_off + n_off + lane_g);
            const float b1  = M_view.at(M_row_off + k_off + lane_t + 4, M_col_off + n_off + lane_g);
            mma_tf32(D[nt * 4 + 0], D[nt * 4 + 1], D[nt * 4 + 2], D[nt * 4 + 3], a0, a1, a2, a3, b0,
                     b1);
        }
    }
}

// 16x16x16 mma: A from swizzled M_view, B from raw row-major scratch.
__device__ __forceinline__ void mma16_swiz_x_raw(float D[8], int lane, SmemTile<BT> M_view,
                                                 int M_row_off, int M_col_off,
                                                 const float* __restrict__ B_buf) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0  = M_view.at(M_row_off + lane_g, M_col_off + k_off + lane_t);
        const float a1  = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t);
        const float a2  = M_view.at(M_row_off + lane_g, M_col_off + k_off + lane_t + 4);
        const float a3  = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t + 4);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0  = B_buf[(k_off + lane_t) * SCR_STRIDE + (n_off + lane_g)];
            const float b1  = B_buf[(k_off + lane_t + 4) * SCR_STRIDE + (n_off + lane_g)];
            mma_tf32(D[nt * 4 + 0], D[nt * 4 + 1], D[nt * 4 + 2], D[nt * 4 + 3], a0, a1, a2, a3, b0,
                     b1);
        }
    }
}

// Templated on (MY_W, MY_J) so `A_reg[k]` resolves to compile-time indices
// after unrolling; otherwise nvcc parks A_reg in local memory.
template <int MY_W, int MY_J>
__device__ __forceinline__ void
compute_off_diag(float out[8], int warp, int lane, const float A_reg[N_SUB][8],
                 float* __restrict__ scr_smem, SmemTile<BT> M_view) {
    static_assert(0 <= MY_J && MY_J < MY_W && MY_W <= N_SUB);

    float sum[8] = {};

#pragma unroll
    for (int k = MY_J; k < MY_W; ++k) {
        scatter_frag_to_scr(A_reg[k], scr_smem, warp, lane);
        __syncwarp();
        const float* A_buf = scr_smem + warp * BC * SCR_STRIDE;
        mma16_raw_x_swiz(sum, lane, A_buf, M_view, k * BC, MY_J * BC);
        __syncwarp();
        if (k == MY_J) { // diagonal-block correction (folded after unroll)
#pragma unroll
            for (int e = 0; e < 8; ++e) sum[e] += A_reg[k][e];
        }
    }

    scatter_frag_to_scr(sum, scr_smem, warp, lane);
    __syncwarp();
    const float* B_buf = scr_smem + warp * BC * SCR_STRIDE;

    float prod[8] = {};
    mma16_swiz_x_raw(prod, lane, M_view, MY_W * BC, MY_W * BC, B_buf);

#pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = prod[e] + sum[e];
}

__device__ __forceinline__ void store_frag_to_M(const float frag[8], int my_w, int my_j, int lane_g,
                                                int lane_t, SmemTile<BT> M_view) {
    const int row_g0                = my_w * BC + lane_g;
    const int row_g1                = row_g0 + 8;
    const int col_base              = my_j * BC + 2 * lane_t;
    M_view.at(row_g0, col_base)     = frag[0];
    M_view.at(row_g0, col_base + 1) = frag[1];
    M_view.at(row_g1, col_base)     = frag[2];
    M_view.at(row_g1, col_base + 1) = frag[3];
    M_view.at(row_g0, col_base + 8) = frag[4];
    M_view.at(row_g0, col_base + 9) = frag[5];
    M_view.at(row_g1, col_base + 8) = frag[6];
    M_view.at(row_g1, col_base + 9) = frag[7];
}

template <int DIAG_BLOCK>
__device__ __forceinline__ void solve_diag_block(int lane, SmemTile<BT> M_view) {
    constexpr int diag_off = DIAG_BLOCK * BC;
    const int wcol         = lane & 15;
    for (int i = 1; i < BC; ++i) {
        const int row_i = diag_off + i;
        const int col   = diag_off + wcol;
        float sum       = 0.0f;
#pragma unroll
        for (int j = 0; j < BC - 1; ++j) {
            if (j < i) { sum += M_view.at(row_i, diag_off + j) * M_view.at(diag_off + j, col); }
        }
        __syncwarp();
        if (lane < 16 && wcol < i) { M_view.at(row_i, col) += sum; }
        __syncwarp();
    }
}

template <int WU_PANEL_COLS, int BLOCK_THREADS>
__device__ __forceinline__ void
load_scaled_wu_panel(SmemTile<WU_PANEL_COLS> panel, const __nv_bfloat16* __restrict__ input_row0,
                     std::int64_t input_row_stride, const float* __restrict__ scale, int panel_col,
                     int tid) {
    constexpr int VEC_PER_ROW = WU_PANEL_COLS / 4;
    constexpr int N_VEC       = BT * VEC_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VEC; v += BLOCK_THREADS) {
        const int row           = v / VEC_PER_ROW;
        const int col4          = (v - row * VEC_PER_ROW) * 4;
        const Bf16x4Pack packed = load_vec<Bf16x4Pack>(
            input_row0 + (std::int64_t)row * input_row_stride + panel_col + col4);
        const float2 lo          = bf16x2_to_float2(packed.pair[0]);
        const float2 hi          = bf16x2_to_float2(packed.pair[1]);
        const float s            = scale[row];
        panel.vec4_at(row, col4) = make_float4(lo.x * s, lo.y * s, hi.x * s, hi.y * s);
    }
}

template <int WU_PANEL_COLS, int BLOCK_THREADS>
__device__ __forceinline__ void
load_scaled_wu_panel_from_smem(SmemTile<WU_PANEL_COLS> panel, Bf16SmemTile<WU_PANEL_COLS> input,
                               const float* __restrict__ scale, int tid) {
    constexpr int VEC_PER_ROW = WU_PANEL_COLS / 4;
    constexpr int N_VEC       = BT * VEC_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VEC; v += BLOCK_THREADS) {
        const int row            = v / VEC_PER_ROW;
        const int col4           = (v - row * VEC_PER_ROW) * 4;
        const Bf16x4Pack packed  = load_vec<Bf16x4Pack>(input.ptr(row, col4));
        const float2 lo          = bf16x2_to_float2(packed.pair[0]);
        const float2 hi          = bf16x2_to_float2(packed.pair[1]);
        const float s            = scale[row];
        panel.vec4_at(row, col4) = make_float4(lo.x * s, lo.y * s, hi.x * s, hi.y * s);
    }
}

template <bool InterleaveOutput, int WU_PANEL_COLS, int BLOCK_WARPS>
__device__ __forceinline__ void
compute_store_wu_panel(SmemTile<BT> T_view, SmemTile<WU_PANEL_COLS> panel,
                       __nv_bfloat16* __restrict__ output_smem,
                       __nv_bfloat16* __restrict__ output_row0, std::int64_t output_row_stride,
                       int panel_col, int warp, int lane) {
    static_assert(BLOCK_WARPS == 4 || BLOCK_WARPS == 8);
    constexpr int WARPS_PER_ROW   = BLOCK_WARPS / N_SUB;
    constexpr int WARP_PANEL_COLS = WU_PANEL_COLS / WARPS_PER_ROW;
    constexpr int WU_N_TILES      = WARP_PANEL_COLS / MMA_N;

    const int lane_g         = lane >> 2;
    const int lane_t         = lane & 3;
    const int row_tile       = warp / WARPS_PER_ROW;
    const int warp_panel_col = (warp - row_tile * WARPS_PER_ROW) * WARP_PANEL_COLS;
    const int row_g0         = row_tile * MMA_M + lane_g;
    const int row_g1         = row_g0 + 8;
    const int col_pair       = lane_t << 1;

    float D[WU_N_TILES][4] = {};
#pragma unroll
    for (int k_tile = 0; k_tile < N_K_TILES; ++k_tile) {
        const int k_off  = k_tile * MMA_K;
        const int col_t0 = k_off + lane_t;
        const int col_t1 = col_t0 + 4;
        const float a0   = T_view.at(row_g0, col_t0);
        const float a1   = T_view.at(row_g1, col_t0);
        const float a2   = T_view.at(row_g0, col_t1);
        const float a3   = T_view.at(row_g1, col_t1);

        const int row_t0 = k_off + lane_t;
        const int row_t1 = row_t0 + 4;
#pragma unroll
        for (int n = 0; n < WU_N_TILES; ++n) {
            const int col  = warp_panel_col + n * MMA_N + lane_g;
            const float b0 = panel.at(row_t0, col);
            const float b1 = panel.at(row_t1, col);
            mma_tf32(D[n][0], D[n][1], D[n][2], D[n][3], a0, a1, a2, a3, b0, b1);
        }
    }

    __nv_bfloat16* const warp_output = output_smem + warp * MMA_M * WARP_PANEL_COLS;
#pragma unroll
    for (int n = 0; n < WU_N_TILES; ++n) {
        const int col = n * MMA_N + col_pair;
        store_vec(&warp_output[lane_g * WARP_PANEL_COLS +
                               wu_output_swizzled_col<WARP_PANEL_COLS>(lane_g, col)],
                  __floats2bfloat162_rn(D[n][0], D[n][1]));
        store_vec(&warp_output[(lane_g + 8) * WARP_PANEL_COLS +
                               wu_output_swizzled_col<WARP_PANEL_COLS>(lane_g + 8, col)],
                  __floats2bfloat162_rn(D[n][2], D[n][3]));
    }
    __syncwarp();

    constexpr int STORE_ELEMS    = 8;
    constexpr int STORE_PER_ROW  = WARP_PANEL_COLS / STORE_ELEMS;
    constexpr int STORE_PER_WARP = MMA_M * STORE_PER_ROW;
#pragma unroll
    for (int v = lane; v < STORE_PER_WARP; v += ninfer::ops::kWarpSize) {
        const int row  = v / STORE_PER_ROW;
        const int col8 = (v - row * STORE_PER_ROW) * STORE_ELEMS;
        uint4 packed =
            load_vec<uint4>(&warp_output[row * WARP_PANEL_COLS +
                                         wu_output_swizzled_col<WARP_PANEL_COLS>(row, col8)]);
        if constexpr (InterleaveOutput) {
            // W is a private prepare -> state-passing workspace. Store every
            // group of eight columns as {0,4,1,5,2,6,3,7}, so the consumer's
            // native-BF16 ldmatrix.x2 directly produces its TF32 A fragment.
            packed = {
                (packed.x & 0x0000ffffU) | (packed.z << 16),
                (packed.x >> 16) | (packed.z & 0xffff0000U),
                (packed.y & 0x0000ffffU) | (packed.w << 16),
                (packed.y >> 16) | (packed.w & 0xffff0000U),
            };
        }
        store_vec(output_row0 + (std::int64_t)(row_tile * MMA_M + row) * output_row_stride +
                      panel_col + warp_panel_col + col8,
                  packed);
    }
}

template <int K_PANEL_COLS, int WU_PANEL_COLS, int BLOCK_WARPS>
__global__ void
prepare_wy_wu_kernel(const __nv_bfloat16* __restrict__ k_in, const __nv_bfloat16* __restrict__ v_in,
                     const float* __restrict__ g_in, const float* __restrict__ beta_in,
                     __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ U,
                     float* __restrict__ g_cumsum_out, head_map qk_map) {
    static_assert(BLOCK_WARPS == 4 || BLOCK_WARPS == 8);
    static_assert(BLOCK_WARPS % N_SUB == 0);
    static_assert(WU_PANEL_COLS % (BLOCK_WARPS / N_SUB) == 0);

    using dims                  = kernel_dims<K_PANEL_COLS, WU_PANEL_COLS>;
    constexpr int BLOCK_THREADS = BLOCK_WARPS * ninfer::ops::kWarpSize;
    constexpr int N_K_PANELS    = dims::N_K_PANELS;
    constexpr int N_WU_PANELS   = dims::N_WU_PANELS;

    extern __shared__ float smem[];
    float* const T_inv_smem = smem;
    float* const stage_smem = smem + dims::T_inv_floats;
    auto* const output_smem = reinterpret_cast<__nv_bfloat16*>(stage_smem + dims::stage_floats);
    float* const g_smem     = stage_smem + dims::stage_floats + dims::output_stage_floats;
    float* const beta_smem  = g_smem + BT;
    float* const bg_smem    = beta_smem + BT;

    // stage_smem aliases BF16 K (WY-B), FP32 Schur scratch (WY-D), and one
    // pre-scaled FP32 V/K panel (WU). All handovers are block-barrier guarded.
    auto* const K_smem    = reinterpret_cast<__nv_bfloat16*>(stage_smem);
    float* const scr_smem = stage_smem;

    SmemTile<BT> M_view{T_inv_smem};
    SmemTile<BT> T_view{T_inv_smem};
    Bf16SmemTile<K_PANEL_COLS> K_view{K_smem};
    SmemTile<WU_PANEL_COLS> WU_view{stage_smem};

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & (kWarpSize - 1);
    const int warp   = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const int chunk               = static_cast<int>(blockIdx.x);
    const int h_v                 = static_cast<int>(blockIdx.y);
    const std::int64_t cs         = static_cast<std::int64_t>(chunk) * BT;
    const std::int64_t H_v        = qk_map.H_v;
    const std::int64_t k_stride_t = static_cast<std::int64_t>(qk_map.H_qk) * kStateDim;
    const std::int64_t v_stride_t = H_v * kStateDim;

    // === Phase WY-A: cooperative load of beta + warp-0 in-place scan of g ===
    //
    // beta load: 64 threads (warp 0 + 1) load BT entries.
    // g scan: only warp 0 participates -- each lane handles 2 consecutive
    // tokens (a[2L], a[2L+1]) of g_in for this (chunk, h_v), runs a
    // Hillis-Steele inclusive scan via shfl, and writes both g_smem[]
    // (consumed by Phase WY-C / WU-A) AND HBM g_cumsum_out (consumed
    // by stages 3/4 unchanged). This folds the old standalone g_cumsum
    // kernel (~10 us, 1.3% of e2e) into here at zero added latency: the
    // scan is hidden behind Phase WY-B's K load + KKT mma.
    //
    // No __syncthreads here -- per-chunk K loader below issues one before
    // any read of g_smem / beta_smem can race the scan stores.
    if (tid < BT) {
        const int64_t boff = (cs + tid) * H_v + h_v;
        beta_smem[tid]     = beta_in[boff];
    }

    if (warp == 0) {
        const int64_t g_row_base = cs * H_v + h_v;
        const int t0             = 2 * lane; // 0, 2, ..., 62
        const int t1             = t0 + 1;   // 1, 3, ..., 63

        const float a  = g_in[g_row_base + (int64_t)t0 * H_v];
        const float bv = g_in[g_row_base + (int64_t)t1 * H_v];

        // Hillis-Steele inclusive scan over per-lane partials (a + bv).
        float partial = a + bv;
#pragma unroll
        for (int o = 1; o < ninfer::ops::kWarpSize; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, partial, o);
            if (lane >= o) partial += n;
        }
        // Inclusive -> exclusive shift: lane 0's prefix is 0.
        const float prev_inc  = __shfl_up_sync(0xffffffffu, partial, 1);
        const float ex_prefix = (lane == 0) ? 0.0f : prev_inc;

        const float c0 = ex_prefix + a;
        const float c1 = c0 + bv;

        g_smem[t0]                                   = c0;
        g_smem[t1]                                   = c1;
        g_cumsum_out[g_row_base + (int64_t)t0 * H_v] = c0;
        g_cumsum_out[g_row_base + (int64_t)t1 * H_v] = c1;
    }

    // === Phase WY-B: native BF16 KKT on the lower-tri 4x4 sub-block grid ===
    //
    // K is a represented BF16 Op input. Keeping it BF16 through shared memory
    // and m16n8k16 changes no operand precision relative to the former
    // BF16->FP32->TF32 path: every BF16 value is exactly representable in TF32.
    // The KKT accumulator remains FP32.
    float A_reg[N_SUB][8] = {};

    const int64_t k_base = cs * k_stride_t + static_cast<int64_t>(qk_map.qk_head(h_v)) * kStateDim;
    const int64_t v_base = cs * v_stride_t + static_cast<int64_t>(h_v) * kStateDim;
    constexpr int K_VECS_PER_ROW = K_PANEL_COLS / 8;
    constexpr int K_STAGE_VECS   = BT * K_VECS_PER_ROW;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    auto kkt_strip = [&]<int N_OWNED>() {
#pragma unroll
        for (int k_tile = 0; k_tile < dims::K_TILES_PER_PANEL; ++k_tile) {
            const int k_off = k_tile * BF16_MMA_K;
            unsigned af[4];
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(K_view.ptr(warp * BC + a_rowoff, k_off + a_coloff)));

#pragma unroll
            for (int j_sub = 0; j_sub < N_OWNED; ++j_sub) {
#pragma unroll
                for (int n_tile = 0; n_tile < 2; ++n_tile) {
                    const int row_b = j_sub * BC + n_tile * MMA_N + b_rin;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1], smem_addr(K_view.ptr(row_b, k_off + b_koff)));
                    mma_bf16(A_reg[j_sub][n_tile * 4 + 0], A_reg[j_sub][n_tile * 4 + 1],
                             A_reg[j_sub][n_tile * 4 + 2], A_reg[j_sub][n_tile * 4 + 3], af[0],
                             af[1], af[2], af[3], bf[0], bf[1]);
                }
            }
        }
    };

#pragma unroll
    for (int kp = 0; kp < N_K_PANELS; ++kp) {
        const int panel_col = kp * K_PANEL_COLS;

#pragma unroll
        for (int v = tid; v < K_STAGE_VECS; v += BLOCK_THREADS) {
            const int row            = v / K_VECS_PER_ROW;
            const int col8           = (v - row * K_VECS_PER_ROW) * 8;
            const __nv_bfloat16* src = k_in + k_base + (int64_t)row * k_stride_t + panel_col + col8;
            cp_async<16, Cache::cg>(K_view.ptr(row, col8), src);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        switch (warp) {
        case 0:
            kkt_strip.template operator()<1>();
            break;
        case 1:
            kkt_strip.template operator()<2>();
            break;
        case 2:
            kkt_strip.template operator()<3>();
            break;
        case 3:
            kkt_strip.template operator()<4>();
            break;
        }

        // The wide route's four W/U helper warps would otherwise wait for the
        // triangular KKT work. Use that interval to stage the first BF16 V
        // panel in output_smem; scaling remains FP32 after T_inv is ready.
        if constexpr (BLOCK_WARPS == 8) {
            if (kp == 0 && warp >= WY_WARPS) {
                constexpr int HELPER_THREADS = (BLOCK_WARPS - WY_WARPS) * ninfer::ops::kWarpSize;
                constexpr int VECS_PER_ROW   = WU_PANEL_COLS / 8;
                constexpr int N_VECS         = BT * VECS_PER_ROW;
                const int helper_tid         = tid - WY_WARPS * ninfer::ops::kWarpSize;
                Bf16SmemTile<WU_PANEL_COLS> preload{output_smem};
#pragma unroll
                for (int v = helper_tid; v < N_VECS; v += HELPER_THREADS) {
                    const int row  = v / VECS_PER_ROW;
                    const int col8 = (v - row * VECS_PER_ROW) * 8;
                    cp_async<16, Cache::cg>(preload.ptr(row, col8),
                                            v_in + v_base + static_cast<int64_t>(row) * v_stride_t +
                                                col8);
                }
                cp_commit();
                cp_wait<0>();
            }
        }

        if (kp + 1 < N_K_PANELS) { __syncthreads(); }
    }

    if (warp < WY_WARPS) {
        const int r_g0       = warp * BC + lane_g;
        const int r_g1       = r_g0 + 8;
        const float beta_r0  = beta_smem[r_g0];
        const float beta_r1  = beta_smem[r_g1];
        const float g_r0     = g_smem[r_g0];
        const float g_r1     = g_smem[r_g1];
        const float nbeta_r0 = -beta_r0;
        const float nbeta_r1 = -beta_r1;

#pragma unroll
        for (int j_sub = 0; j_sub < N_SUB; ++j_sub) {
            if (j_sub > warp) continue;

            const int c_base = j_sub * BC + 2 * lane_t;
            const int c0     = c_base;
            const int c1     = c_base + 1;
            const int c2     = c_base + 8;
            const int c3     = c_base + 9;

            const float g_c0 = g_smem[c0];
            const float g_c1 = g_smem[c1];
            const float g_c2 = g_smem[c2];
            const float g_c3 = g_smem[c3];

            const bool is_diag = (j_sub == warp);

            // Diagonal block: keep strict lower triangle (r > c), zero the rest.
            // Off-diagonal blocks (j_sub < warp): keep all entries.
            //
            // Footgun: g_cumsum is monotone decreasing, so on the diagonal block's
            // strict-upper triangle (r < c) g_r - g_c is large positive and expf
            // can saturate to +inf. A `* mask` (mask = 0/1 float) would then
            // produce inf * 0 = NaN. The conditional select below overwrites the
            // bad product with 0.0f instead, so no NaN escapes (same pattern as
            // stage_chunk_output.cu Phase C).
            A_reg[j_sub][0] =
                (!is_diag || r_g0 > c0) ? nbeta_r0 * A_reg[j_sub][0] * expf(g_r0 - g_c0) : 0.0f;
            A_reg[j_sub][1] =
                (!is_diag || r_g0 > c1) ? nbeta_r0 * A_reg[j_sub][1] * expf(g_r0 - g_c1) : 0.0f;
            A_reg[j_sub][2] =
                (!is_diag || r_g1 > c0) ? nbeta_r1 * A_reg[j_sub][2] * expf(g_r1 - g_c0) : 0.0f;
            A_reg[j_sub][3] =
                (!is_diag || r_g1 > c1) ? nbeta_r1 * A_reg[j_sub][3] * expf(g_r1 - g_c1) : 0.0f;
            A_reg[j_sub][4] =
                (!is_diag || r_g0 > c2) ? nbeta_r0 * A_reg[j_sub][4] * expf(g_r0 - g_c2) : 0.0f;
            A_reg[j_sub][5] =
                (!is_diag || r_g0 > c3) ? nbeta_r0 * A_reg[j_sub][5] * expf(g_r0 - g_c3) : 0.0f;
            A_reg[j_sub][6] =
                (!is_diag || r_g1 > c2) ? nbeta_r1 * A_reg[j_sub][6] * expf(g_r1 - g_c2) : 0.0f;
            A_reg[j_sub][7] =
                (!is_diag || r_g1 > c3) ? nbeta_r1 * A_reg[j_sub][7] * expf(g_r1 - g_c3) : 0.0f;
        }
    }

    __syncthreads();

    {
        constexpr int N = BT * BT;
#pragma unroll
        for (int idx = tid; idx < N; idx += BLOCK_THREADS) T_inv_smem[idx] = 0.0f;
    }
    __syncthreads();

    switch (warp) {
    case 0:
        store_frag_to_M(A_reg[0], 0, 0, lane_g, lane_t, M_view);
        break;
    case 1:
        store_frag_to_M(A_reg[1], 1, 1, lane_g, lane_t, M_view);
        break;
    case 2:
        store_frag_to_M(A_reg[2], 2, 2, lane_g, lane_t, M_view);
        break;
    case 3:
        store_frag_to_M(A_reg[3], 3, 3, lane_g, lane_t, M_view);
        break;
    }
    __syncwarp();

    switch (warp) {
    case 0:
        solve_diag_block<0>(lane, M_view);
        break;
    case 1:
        solve_diag_block<1>(lane, M_view);
        break;
    case 2:
        solve_diag_block<2>(lane, M_view);
        break;
    case 3:
        solve_diag_block<3>(lane, M_view);
        break;
    }
    __syncthreads();

    // === Phase WY-D: block-Schur off-diagonal completion (3 waves) ===
    // Switch on `warp` so (MY_W, MY_J) are compile-time per case.
    auto wave_compute_store = [&]<int MY_W, int MY_J>() {
        float out[8];
        compute_off_diag<MY_W, MY_J>(out, warp, lane, A_reg, scr_smem, M_view);
        store_frag_to_M(out, MY_W, MY_J, lane_g, lane_t, M_view);
    };

    switch (warp) {
    case 1:
        wave_compute_store.template operator()<1, 0>();
        break;
    case 2:
        wave_compute_store.template operator()<2, 1>();
        break;
    case 3:
        wave_compute_store.template operator()<3, 2>();
        break;
    }
    __syncthreads();

    switch (warp) {
    case 2:
        wave_compute_store.template operator()<2, 0>();
        break;
    case 3:
        wave_compute_store.template operator()<3, 1>();
        break;
    }
    __syncthreads();

    if (warp == 3) { wave_compute_store.template operator()<3, 0>(); }
    __syncthreads();

    // === Phase WY-E: +I on diagonal of T_inv (no HBM write; sync fused into WU-A) ===
    if (tid < BT) { M_view.at(tid, tid) += 1.0f; }

    // W/U preserve the former precision path. BF16 inputs are converted to
    // FP32, multiplied by their FP32 row scale, stored as FP32, and consumed by
    // TF32 MMA with FP32 accumulation. In particular, T_inv and scaled V/K are
    // never down-cast to BF16.
    if (tid < BT) { bg_smem[tid] = beta_smem[tid] * expf(g_smem[tid]); }

    const int64_t out_base       = cs * H_v * kStateDim + static_cast<int64_t>(h_v) * kStateDim;
    const int64_t out_row_stride = H_v * kStateDim;
    const int64_t k_wu_base =
        cs * k_stride_t + static_cast<int64_t>(qk_map.qk_head(h_v)) * kStateDim;

    // === Phase WU-A/B: U = T_inv @ (beta * V), one 64x32 FP32 panel at a time ===
#pragma unroll
    for (int panel = 0; panel < N_WU_PANELS; ++panel) {
        if (panel != 0) { __syncthreads(); }
        const int panel_col = panel * WU_PANEL_COLS;
        if constexpr (BLOCK_WARPS == 8) {
            if (panel == 0) {
                load_scaled_wu_panel_from_smem<WU_PANEL_COLS, BLOCK_THREADS>(
                    WU_view, Bf16SmemTile<WU_PANEL_COLS>{output_smem}, beta_smem, tid);
            } else {
                load_scaled_wu_panel<WU_PANEL_COLS, BLOCK_THREADS>(
                    WU_view, v_in + v_base, v_stride_t, beta_smem, panel_col, tid);
            }
        } else {
            load_scaled_wu_panel<WU_PANEL_COLS, BLOCK_THREADS>(WU_view, v_in + v_base, v_stride_t,
                                                               beta_smem, panel_col, tid);
        }
        __syncthreads();
        compute_store_wu_panel<false, WU_PANEL_COLS, BLOCK_WARPS>(
            T_view, WU_view, output_smem, U + out_base, out_row_stride, panel_col, warp, lane);
    }

    // === Phase WU-C/D: W = T_inv @ (beta * exp(g) * K), same FP32 panel path ===
    __syncthreads();
#pragma unroll
    for (int panel = 0; panel < N_WU_PANELS; ++panel) {
        if (panel != 0) { __syncthreads(); }
        const int panel_col = panel * WU_PANEL_COLS;
        load_scaled_wu_panel<WU_PANEL_COLS, BLOCK_THREADS>(WU_view, k_in + k_wu_base, k_stride_t,
                                                           bg_smem, panel_col, tid);
        __syncthreads();
        compute_store_wu_panel<true, WU_PANEL_COLS, BLOCK_WARPS>(
            T_view, WU_view, output_smem, W + out_base, out_row_stride, panel_col, warp, lane);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked::prepare_wy_wu
