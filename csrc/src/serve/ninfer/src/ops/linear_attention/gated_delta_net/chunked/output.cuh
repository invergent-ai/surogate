#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/common.cuh"

#include <cmath>

// Stage 4: chunk_output.
//
//   A[t,s] = (s <= t) ? dot(q[t], k[s]) * exp(g[t] - g[s]) : 0
//   out    = scale * (exp(g) * q @ h_chunk^T + A @ v_new)
//
// Q/K/H/V are represented BF16 inputs. Q @ K^T and Q @ H^T therefore use
// native BF16 MMA with FP32 accumulation. The decayed A matrix is FP32 and
// remains on the existing TF32-MMA/FP32-accumulation path for A @ V; no FP32
// intermediate is down-cast to BF16.
//
// Shared memory: persistent BF16 Q (16 KiB) and two 4 KiB BF16 staging
// buffers, for 24 KiB total. FP32 g reuses the K buffer that becomes dead
// while the final K panel is consumed.

namespace ninfer::ops::detail::gated_delta_net::chunked::output {

using ninfer::ops::Cache;
using ninfer::ops::cp_async;
using ninfer::ops::cp_commit;
using ninfer::ops::cp_wait;
using ninfer::ops::exp2_approx;
using ninfer::ops::ldmatrix_x2;
using ninfer::ops::ldmatrix_x4;
using ninfer::ops::mma_bf16;
using ninfer::ops::mma_tf32;
using ninfer::ops::smem_addr;

static_assert(kChunkSize == 64,
              "stage_chunk_output: kChunkSize must be 64 (kernel hard-codes BT=64)");
static_assert(kStateDim == 128);

constexpr int N_WARPS = 4;
constexpr int THREADS = N_WARPS * ninfer::ops::kWarpSize;

static_assert(BT == N_WARPS * MMA_M,
              "kernel assigns one 16-row strip per warp; BT must equal N_WARPS * MMA_M");

constexpr int BF16_MMA_K = 16;
constexpr int K_PANEL     = 32;
constexpr int D_PANEL     = 16;
constexpr int N_K_PANELS  = kStateDim / K_PANEL;
constexpr int N_D_PANELS  = kStateDim / D_PANEL;
constexpr int N_TILES_BT  = BT / MMA_N;
constexpr int K_TILES_BT  = BT / MMA_K;

static_assert(K_PANEL % BF16_MMA_K == 0);
static_assert(D_PANEL % MMA_N == 0);

struct kernel_dims {
    static constexpr int Q_BF16       = BT * kStateDim;
    static constexpr int K_PANEL_BF16 = BT * K_PANEL;
    static constexpr int H_PANEL_BF16 = D_PANEL * kStateDim;
    static constexpr int V_PANEL_BF16 = BT * D_PANEL;
    static constexpr int STAGE_BF16 =
        K_PANEL_BF16 > H_PANEL_BF16 ? K_PANEL_BF16 : H_PANEL_BF16;
    static_assert(STAGE_BF16 >= V_PANEL_BF16);

    static constexpr int BF16_SMEM_ELEMS = Q_BF16 + 2 * STAGE_BF16;
    static constexpr int SMEM_BYTES =
        BF16_SMEM_ELEMS * static_cast<int>(sizeof(__nv_bfloat16));
};

template <int STRIDE>
struct Bf16SmemTile {
    __nv_bfloat16* __restrict__ base;
    static_assert(STRIDE == 16 || STRIDE == 32 || STRIDE == 128);

    __device__ __forceinline__ int swizzled_col(int row, int col) const {
        return col ^ ((row & (STRIDE / 8 - 1)) << 3);
    }

    __device__ __forceinline__ __nv_bfloat16* ptr(int row, int col) const {
        return base + row * STRIDE + swizzled_col(row, col);
    }
};

template <int ROWS, int COLS, int BLOCK_THREADS>
__device__ __forceinline__ void
issue_cp_bf16(Bf16SmemTile<COLS> dst, const __nv_bfloat16* __restrict__ src_row0,
              std::int64_t src_row_stride, int tid) {
    static_assert(COLS % 8 == 0);
    constexpr int VECS_PER_ROW = COLS / 8;
    constexpr int N_VECS       = ROWS * VECS_PER_ROW;
#pragma unroll
    for (int v = tid; v < N_VECS; v += BLOCK_THREADS) {
        const int row  = v / VECS_PER_ROW;
        const int col8 = (v - row * VECS_PER_ROW) * 8;
        cp_async<16, Cache::cg>(dst.ptr(row, col8),
                                src_row0 + static_cast<std::int64_t>(row) * src_row_stride + col8);
    }
}

template <int N_TILES, int K_TILES, int A_STRIDE, int B_STRIDE>
__device__ __forceinline__ void
mma_bf16_panel(float (&D)[N_TILES][4], Bf16SmemTile<A_STRIDE> A,
               Bf16SmemTile<B_STRIDE> B, int a_row_base, int a_col_base, int b_col_base,
               int lane) {
    const int lane_in_8 = lane & 7;
    const int lane_q    = lane >> 3;
    const int a_row     = a_row_base + lane_in_8 + ((lane_q & 1) << 3);
    const int a_col     = a_col_base + ((lane_q >> 1) << 3);
    const int b_col     = b_col_base + ((lane_q & 1) << 3);

#pragma unroll
    for (int kt = 0; kt < K_TILES; ++kt) {
        const int k_off = kt * BF16_MMA_K;
        unsigned af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3], smem_addr(A.ptr(a_row, a_col + k_off)));

#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            unsigned bf[2];
            ldmatrix_x2(bf[0], bf[1],
                        smem_addr(B.ptr(nt * MMA_N + lane_in_8, b_col + k_off)));
            mma_bf16(D[nt][0], D[nt][1], D[nt][2], D[nt][3], af[0], af[1], af[2], af[3],
                     bf[0], bf[1]);
        }
    }
}

template <int N_TILES>
__device__ __forceinline__ void mma_av_panel(float (&D)[N_TILES][4],
                                              const float A_a[K_TILES_BT][4],
                                              Bf16SmemTile<D_PANEL> V, int lane_g, int lane_t) {
#pragma unroll
    for (int kt = 0; kt < K_TILES_BT; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0  = A_a[kt][0];
        const float a1  = A_a[kt][1];
        const float a2  = A_a[kt][2];
        const float a3  = A_a[kt][3];

#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            const int col  = nt * MMA_N + lane_g;
            const float b0 = __bfloat162float(*V.ptr(k_off + lane_t, col));
            const float b1 = __bfloat162float(*V.ptr(k_off + lane_t + 4, col));
            mma_tf32(D[nt][0], D[nt][1], D[nt][2], D[nt][3], a0, a1, a2, a3, b0, b1);
        }
    }
}

__device__ __forceinline__ void
output_job(const __nv_bfloat16* __restrict__ q_in,
           const __nv_bfloat16* __restrict__ k_in,
           const __nv_bfloat16* __restrict__ v_new_in,
           const float* __restrict__ g_cumsum_in,
           const __nv_bfloat16* __restrict__ h_chunk_in,
           __nv_bfloat16* __restrict__ attn_out, head_map qk_map, float scale, int chunk, int h_v,
           float* smem) {
    auto* const bf16_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    auto* const q_smem    = bf16_smem;
    auto* const stage0    = q_smem + kernel_dims::Q_BF16;
    auto* const stage1    = stage0 + kernel_dims::STAGE_BF16;
    float* const g_smem =
        reinterpret_cast<float*>(stage0 + kernel_dims::V_PANEL_BF16);

    Bf16SmemTile<kStateDim> q_view{q_smem};
    Bf16SmemTile<K_PANEL> k_stage0{stage0};
    Bf16SmemTile<K_PANEL> k_stage1{stage1};
    Bf16SmemTile<kStateDim> h_view{stage0};
    Bf16SmemTile<D_PANEL> v_view{stage1};

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & (kWarpSize - 1);
    const int warp   = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const std::int64_t cs          = static_cast<std::int64_t>(chunk) * BT;
    const std::int64_t H_v         = qk_map.H_v;
    const std::int64_t qk_stride_t = static_cast<std::int64_t>(qk_map.H_qk) * kStateDim;
    const std::int64_t qk_head_idx = static_cast<std::int64_t>(qk_map.qk_head(h_v)) * kStateDim;
    const std::int64_t q_base      = cs * qk_stride_t + qk_head_idx;
    const std::int64_t k_base      = cs * qk_stride_t + qk_head_idx;
    const std::int64_t vn_base =
        cs * H_v * kStateDim + static_cast<std::int64_t>(h_v) * kStateDim;
    const std::int64_t hc_base =
        (static_cast<std::int64_t>(chunk) * H_v + h_v) * kStateDim * kStateDim;

    const std::int64_t value_row_stride = H_v * kStateDim;

    // Q is permanent. K uses two 64x32 BF16 buffers and is prefetched one
    // panel ahead while the current panel feeds BF16 MMA.
    issue_cp_bf16<BT, kStateDim, THREADS>(q_view, q_in + q_base, qk_stride_t, tid);
    issue_cp_bf16<BT, K_PANEL, THREADS>(k_stage0, k_in + k_base, qk_stride_t, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    float A_strip[N_TILES_BT][4] = {};

#pragma unroll
    for (int panel = 0; panel < N_K_PANELS; ++panel) {
        Bf16SmemTile<K_PANEL> current = (panel & 1) == 0 ? k_stage0 : k_stage1;
        if (panel + 1 < N_K_PANELS) {
            Bf16SmemTile<K_PANEL> next = (panel & 1) == 0 ? k_stage1 : k_stage0;
            issue_cp_bf16<BT, K_PANEL, THREADS>(
                next, k_in + k_base + static_cast<std::int64_t>(panel + 1) * K_PANEL, qk_stride_t,
                tid);
            cp_commit();
        } else if (tid < BT) {
            // stage0 was consumed by panel 2 and is now dead. Reuse its
            // otherwise-idle tail for g while panel 3 consumes stage1.
            g_smem[tid] =
                g_cumsum_in[(cs + static_cast<std::int64_t>(tid)) * H_v + h_v];
        }

        mma_bf16_panel<N_TILES_BT, K_PANEL / BF16_MMA_K>(
            A_strip, q_view, current, warp * MMA_M, panel * K_PANEL, 0, lane);

        if (panel + 1 < N_K_PANELS) {
            cp_wait<0>();
            __syncthreads();
        }
    }
    __syncthreads();

    // Apply causal decay in FP32. Upper-triangle exp2 values may overflow, so
    // the conditional select must replace the entire product rather than
    // multiply by a zero mask.
    const int row_g0   = warp * MMA_M + lane_g;
    const int row_g1   = row_g0 + 8;
    const float g_r0   = g_smem[row_g0];
    const float g_r1   = g_smem[row_g1];
    const float gamma0 = exp2_approx(g_r0 * kLog2E);
    const float gamma1 = exp2_approx(g_r1 * kLog2E);

#pragma unroll
    for (int nt = 0; nt < N_TILES_BT; ++nt) {
        const int s0     = nt * MMA_N + 2 * lane_t;
        const int s1     = s0 + 1;
        const float g_s0 = g_smem[s0];
        const float g_s1 = g_smem[s1];

        const float dec00 = exp2_approx((g_r0 - g_s0) * kLog2E);
        const float dec01 = exp2_approx((g_r0 - g_s1) * kLog2E);
        const float dec10 = exp2_approx((g_r1 - g_s0) * kLog2E);
        const float dec11 = exp2_approx((g_r1 - g_s1) * kLog2E);

        A_strip[nt][0] = (s0 <= row_g0) ? A_strip[nt][0] * dec00 : 0.0f;
        A_strip[nt][1] = (s1 <= row_g0) ? A_strip[nt][1] * dec01 : 0.0f;
        A_strip[nt][2] = (s0 <= row_g1) ? A_strip[nt][2] * dec10 : 0.0f;
        A_strip[nt][3] = (s1 <= row_g1) ? A_strip[nt][3] * dec11 : 0.0f;
    }

    // Convert the m16n8 accumulator layout into the TF32 A-operand layout
    // needed by A @ V. A remains FP32 throughout.
    float A_a[K_TILES_BT][4];
    {
        const int src_lo        = (lane_g << 2) | (lane_t >> 1);
        const int src_hi        = src_lo + 2;
        const bool t_odd        = (lane_t & 1) != 0;
        constexpr unsigned mask = 0xFFFFFFFFu;

#pragma unroll
        for (int kt = 0; kt < K_TILES_BT; ++kt) {
            const float d0_lo = __shfl_sync(mask, A_strip[kt][0], src_lo);
            const float d1_lo = __shfl_sync(mask, A_strip[kt][1], src_lo);
            const float d2_lo = __shfl_sync(mask, A_strip[kt][2], src_lo);
            const float d3_lo = __shfl_sync(mask, A_strip[kt][3], src_lo);
            const float d0_hi = __shfl_sync(mask, A_strip[kt][0], src_hi);
            const float d1_hi = __shfl_sync(mask, A_strip[kt][1], src_hi);
            const float d2_hi = __shfl_sync(mask, A_strip[kt][2], src_hi);
            const float d3_hi = __shfl_sync(mask, A_strip[kt][3], src_hi);

            A_a[kt][0] = t_odd ? d1_lo : d0_lo;
            A_a[kt][1] = t_odd ? d3_lo : d2_lo;
            A_a[kt][2] = t_odd ? d1_hi : d0_hi;
            A_a[kt][3] = t_odd ? d3_hi : d2_hi;
        }
    }

    // All K reads must finish before stage0 becomes H panel 0.
    __syncthreads();
    issue_cp_bf16<D_PANEL, kStateDim, THREADS>(h_view, h_chunk_in + hc_base, kStateDim, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    // H and V use disjoint buffers. V[c] is fetched while Q @ H[c]^T runs;
    // H[c+1] is fetched while the FP32 A @ V[c] path runs.
#pragma unroll 1
    for (int panel = 0; panel < N_D_PANELS; ++panel) {
        const int d_off = panel * D_PANEL;

        issue_cp_bf16<BT, D_PANEL, THREADS>(
            v_view, v_new_in + vn_base + static_cast<std::int64_t>(d_off), value_row_stride, tid);
        cp_commit();

        float D_frag[D_PANEL / MMA_N][4] = {};
        mma_bf16_panel<D_PANEL / MMA_N, kStateDim / BF16_MMA_K>(
            D_frag, q_view, h_view, warp * MMA_M, 0, 0, lane);

#pragma unroll
        for (int nt = 0; nt < D_PANEL / MMA_N; ++nt) {
            D_frag[nt][0] *= gamma0;
            D_frag[nt][1] *= gamma0;
            D_frag[nt][2] *= gamma1;
            D_frag[nt][3] *= gamma1;
        }

        cp_wait<0>();
        __syncthreads();

        if (panel + 1 < N_D_PANELS) {
            issue_cp_bf16<D_PANEL, kStateDim, THREADS>(
                h_view, h_chunk_in + hc_base +
                            static_cast<std::int64_t>(panel + 1) * D_PANEL * kStateDim,
                kStateDim, tid);
            cp_commit();
        }

        mma_av_panel(D_frag, A_a, v_view, lane_g, lane_t);

#pragma unroll
        for (int nt = 0; nt < D_PANEL / MMA_N; ++nt) {
            const int d_global = d_off + nt * MMA_N + 2 * lane_t;
            const __nv_bfloat162 out0 =
                __floats2bfloat162_rn(scale * D_frag[nt][0], scale * D_frag[nt][1]);
            const __nv_bfloat162 out1 =
                __floats2bfloat162_rn(scale * D_frag[nt][2], scale * D_frag[nt][3]);
            store_vec(&attn_out[vn_base + static_cast<std::int64_t>(row_g0) * value_row_stride +
                                d_global],
                      out0);
            store_vec(&attn_out[vn_base + static_cast<std::int64_t>(row_g1) * value_row_stride +
                                d_global],
                      out1);
        }

        if (panel + 1 < N_D_PANELS) {
            cp_wait<0>();
            __syncthreads();
        }
    }
}

template <bool MULTI_JOB>
__launch_bounds__(THREADS, 4) __global__
    void output_kernel(const __nv_bfloat16* __restrict__ q_in,
                       const __nv_bfloat16* __restrict__ k_in,
                       const __nv_bfloat16* __restrict__ v_new_in,
                       const float* __restrict__ g_cumsum_in,
                       const __nv_bfloat16* __restrict__ h_chunk_in,
                       __nv_bfloat16* __restrict__ attn_out, head_map qk_map, float scale,
                       int chunks) {
    extern __shared__ float smem[];

    const int h_v = static_cast<int>(blockIdx.y);
    if constexpr (MULTI_JOB) {
        const int chunk_stride = static_cast<int>(gridDim.x);
        for (int chunk = static_cast<int>(blockIdx.x); chunk < chunks; chunk += chunk_stride) {
            output_job(q_in, k_in, v_new_in, g_cumsum_in, h_chunk_in, attn_out, qk_map, scale,
                       chunk, h_v, smem);
            if (chunk + chunk_stride < chunks) { __syncthreads(); }
        }
    } else {
        output_job(q_in, k_in, v_new_in, g_cumsum_in, h_chunk_in, attn_out, qk_map, scale,
                   static_cast<int>(blockIdx.x), h_v, smem);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked::output
