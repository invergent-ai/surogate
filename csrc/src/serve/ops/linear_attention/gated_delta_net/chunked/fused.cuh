#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/common.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh"

// One-kernel chunked Gated DeltaNet prefill. One CTA walks one value head's whole prompt, 64
// tokens a chunk, with the head's 128x128 FP32 state in registers: warp w owns value rows
// [16w, 16w + 16) as m16n8 accumulator tiles over the 128 key columns. Only Q, K, V, the gates,
// the output and the end state cross HBM. The staged path (prepare_wy_wu, state_passing,
// output) writes W, U, v_new and every chunk's state to the workspace and reads them back, which
// on a bandwidth-bound part costs several times the arithmetic.
//
// Per chunk, with gc the in-chunk cumulative log decay, beta the update gate and S the state
// entering the chunk (rows value, columns key):
//   T     = (I + strict_lower(beta_t exp(gc_t - gc_s) k_t.k_s))^-1 diag(beta)
//   delta = T (V - diag(exp(gc)) K S^T)
//   out   = scale (diag(exp(gc)) Q S^T + lower(exp(gc_t - gc_s) q_t.k_s) delta)
//   S'    = exp(gc_last) S + (diag(exp(gc_last - gc)) delta)^T K
// which is the staged path's U - W S^T, the same algebra without W and U. Every product is taken
// transposed, value rows first, so the register-resident S is always the A operand.
//
// Precision: K K^T and Q K^T are bf16 tensor-core products of the bf16 inputs; T is built in
// FP32 by the staged path's own triangular solve (prepare_wy_wu). The state is rounded to bf16
// at every chunk boundary, as the staged path rounds it, so a prompt split across prefill calls
// gives the same bits and S enters S K^T and S Q^T exactly. T X and the intra-chunk A delta are
// TF32 products of FP32 values; delta enters the state update rounded to bf16, as FLA's chunked
// kernels round it (the staged path keeps it TF32).
//
// The chunk's Q, K and V land by cp.async a phase ahead of their use: K and V share two tiles
// whose roles swap every chunk, so the next K lands in the tile this chunk's V leaves and the
// next V in the tile its K leaves.

namespace sinfer::ops::detail::gated_delta_net::chunked::fused {

using sinfer::ops::Cache;
using sinfer::ops::cp_async_zfill;
using sinfer::ops::cp_commit;
using sinfer::ops::cp_wait;
using sinfer::ops::ldmatrix_x4;
using sinfer::ops::ldmatrix_x4_t;
using sinfer::ops::mma_bf16;
using sinfer::ops::mma_tf32;
using sinfer::ops::pack_bf16x2;
using sinfer::ops::smem_addr;

static_assert(BT == 64 && kStateDim == 128);

inline constexpr int kWarps        = 8;
inline constexpr int kThreads      = kWarps * 32;
inline constexpr int kInverseWarps = prepare_wy_wu::WY_WARPS;
inline constexpr int kTileElems    = BT * kStateDim;
static_assert(kInverseWarps == 4 && kWarps * 16 == kStateDim);

// Dynamic shared memory, in bytes: the Q tile, the two K/V tiles, T and the decayed Q K^T
// (FP32 [64][64] each), the triangular solve's scratch, then the raw gates of two chunks and this
// chunk's cumulative decay and update gate.
struct smem_layout {
    static constexpr int q       = 0;
    static constexpr int kv      = q + kTileElems * 2;
    static constexpr int t       = kv + 2 * kTileElems * 2;
    static constexpr int a       = t + BT * BT * 4;
    static constexpr int scr     = a + BT * BT * 4;
    static constexpr int g_raw   = scr + kInverseWarps * BC * prepare_wy_wu::SCR_STRIDE * 4;
    static constexpr int b_raw   = g_raw + 2 * BT * 4;
    static constexpr int gc      = b_raw + 2 * BT * 4;
    static constexpr int beta    = gc + BT * 4;
    static constexpr int bytes   = beta + BT * 4;
};
static_assert(smem_layout::bytes == 88576);

// A bf16 [64][128] tile, its 16-byte chunks XOR-swizzled by row so ldmatrix reads are
// conflict-free.
__device__ __forceinline__ int tile_off(int row, int col) {
    return row * kStateDim + ((((col >> 3) ^ (row & 7)) << 3) | (col & 7));
}

__device__ __forceinline__ void load_tile(__nv_bfloat16* tile, const __nv_bfloat16* src,
                                          std::int64_t row_stride, int valid_rows, int tid) {
#pragma unroll
    for (int c = tid; c < BT * (kStateDim / 8); c += kThreads) {
        const int row    = c >> 4;
        const int col    = (c & 15) << 3;
        const bool valid = row < valid_rows;
        cp_async_zfill<16, Cache::cg>(tile + tile_off(row, col),
                                      valid ? src + row * row_stride + col : src, valid ? 16 : 0);
    }
}

// One chunk's g and beta for this head, strided by the value-head count; rows past the prompt
// read zero (no decay, no update).
__device__ __forceinline__ void load_gates(float* g_dst, float* b_dst, const float* g,
                                           const float* beta, std::int64_t base, int H_v,
                                           int valid_rows, int tid) {
    if (tid < 2 * BT) {
        const int t        = tid & (BT - 1);
        const bool valid   = t < valid_rows;
        const float* src   = tid < BT ? g : beta;
        float* dst         = tid < BT ? g_dst : b_dst;
        cp_async_zfill<4>(dst + t, valid ? src + base + static_cast<std::int64_t>(t) * H_v : src,
                          valid ? 4 : 0);
    }
}

// Inclusive cumulative sum of one chunk's log decays (warp 0), as prepare_wy_wu scans them.
__device__ __forceinline__ void scan_gates(float* gc, float* beta_out, const float* g_raw,
                                           const float* b_raw, int lane) {
    const int t0  = 2 * lane;
    const float a = g_raw[t0];
    const float b = g_raw[t0 + 1];
    float partial = a + b;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float n = __shfl_up_sync(0xffffffffu, partial, o);
        if (lane >= o) { partial += n; }
    }
    const float prev = __shfl_up_sync(0xffffffffu, partial, 1);
    const float c0   = (lane == 0 ? 0.0f : prev) + a;
    gc[t0]           = c0;
    gc[t0 + 1]       = c0 + b;
    beta_out[t0]     = b_raw[t0];
    beta_out[t0 + 1] = b_raw[t0 + 1];
}

// acc[j][nt * 4 + e] for j <= RB: the m16n8 tiles of rows [16 RB, 16 RB + 16) of `a` against
// rows [16 j, 16 j + 16) of `b`, contracted over the 128 columns -- prepare_wy_wu's KKT strip
// over whole tiles.
template <int RB>
__device__ __forceinline__ void block_gram(float acc[prepare_wy_wu::N_SUB][8],
                                           const __nv_bfloat16* a, const __nv_bfloat16* b,
                                           int lane) {
    const int a_row = 16 * RB + (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = (lane >> 4) << 3;
    const int b_row = (lane & 7) + ((lane >> 4) << 3);
    const int b_col = ((lane >> 3) & 1) << 3;
#pragma unroll
    for (int k = 0; k < kStateDim / 16; ++k) {
        unsigned a0, a1, a2, a3;
        ldmatrix_x4(a0, a1, a2, a3, smem_addr(a + tile_off(a_row, 16 * k + a_col)));
#pragma unroll
        for (int j = 0; j <= RB; ++j) {
            unsigned b0, b1, b2, b3;
            ldmatrix_x4(b0, b1, b2, b3, smem_addr(b + tile_off(16 * j + b_row, 16 * k + b_col)));
            mma_bf16(acc[j][0], acc[j][1], acc[j][2], acc[j][3], a0, a1, a2, a3, b0, b1);
            mma_bf16(acc[j][4], acc[j][5], acc[j][6], acc[j][7], a0, a1, a2, a3, b2, b3);
        }
    }
}

__device__ __forceinline__ void inverse_barrier() {
    asm volatile("bar.sync 1, %0;\n" : : "n"(kInverseWarps * 32) : "memory");
}

// Warps 0-3: T for rows [16 RB, 16 RB + 16), by prepare_wy_wu's construction (the masked,
// decayed, -beta-scaled K K^T, the diagonal solves, three waves of off-diagonal blocks, +I),
// with a named barrier for the four warps in place of the CTA barriers there.
template <int RB>
__device__ __forceinline__ void inverse_role(SmemTile<BT> T_view, float* scr,
                                             const __nv_bfloat16* k_s, const float* gc_s,
                                             const float* beta_s, int warp, int lane) {
    using namespace prepare_wy_wu;
    const int gid = lane >> 2;
    const int lid = lane & 3;
    float A_reg[N_SUB][8] = {};
    block_gram<RB>(A_reg, k_s, k_s, lane);

    const int r0    = RB * BC + gid;
    const int r1    = r0 + 8;
    const float nb0 = -beta_s[r0];
    const float nb1 = -beta_s[r1];
    const float g0  = gc_s[r0];
    const float g1  = gc_s[r1];
#pragma unroll
    for (int j = 0; j <= RB; ++j) {
        const int c0      = j * BC + 2 * lid;
        const bool diag   = j == RB;
        const int cols[4] = {c0, c0 + 1, c0 + 8, c0 + 9};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c    = cols[q];
            const float gc = gc_s[c];
            const int e    = (q >> 1) * 4 + (q & 1);
            // A select, not a 0/1 multiply: exp of the upper triangle can be +inf.
            A_reg[j][e]     = (!diag || r0 > c) ? nb0 * A_reg[j][e] * expf(g0 - gc) : 0.0f;
            A_reg[j][e + 2] = (!diag || r1 > c) ? nb1 * A_reg[j][e + 2] * expf(g1 - gc) : 0.0f;
        }
    }

    store_frag_to_M(A_reg[RB], RB, RB, gid, lid, T_view);
    __syncwarp();
    solve_diag_block<RB>(lane, T_view);
    inverse_barrier();
    float out[8];
    if constexpr (RB >= 1) {
        compute_off_diag<RB, RB - 1>(out, warp, lane, A_reg, scr, T_view);
        store_frag_to_M(out, RB, RB - 1, gid, lid, T_view);
    }
    inverse_barrier();
    if constexpr (RB >= 2) {
        compute_off_diag<RB, RB - 2>(out, warp, lane, A_reg, scr, T_view);
        store_frag_to_M(out, RB, RB - 2, gid, lid, T_view);
    }
    inverse_barrier();
    if constexpr (RB == 3) {
        compute_off_diag<3, 0>(out, warp, lane, A_reg, scr, T_view);
        store_frag_to_M(out, 3, 0, gid, lid, T_view);
    }
    inverse_barrier();
    if (lane < BC) { T_view.at(RB * BC + lane, RB * BC + lane) += 1.0f; }
}

// Warps 4-7: rows [16 RB, 16 RB + 16) of the decayed, causal Q K^T, scaled by the output scale.
// Blocks above the diagonal are never read.
template <int RB>
__device__ __forceinline__ void qk_role(SmemTile<BT> A_view, const __nv_bfloat16* q_s,
                                        const __nv_bfloat16* k_s, const float* gc_s, float scale,
                                        int lane) {
    using namespace prepare_wy_wu;
    const int gid = lane >> 2;
    const int lid = lane & 3;
    float acc[N_SUB][8] = {};
    block_gram<RB>(acc, q_s, k_s, lane);
    const int r0   = RB * BC + gid;
    const int r1   = r0 + 8;
    const float g0 = gc_s[r0];
    const float g1 = gc_s[r1];
#pragma unroll
    for (int j = 0; j <= RB; ++j) {
        const int c0      = j * BC + 2 * lid;
        const int cols[4] = {c0, c0 + 1, c0 + 8, c0 + 9};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c    = cols[q];
            const float gc = gc_s[c];
            const int e    = (q >> 1) * 4 + (q & 1);
            A_view.at(r0, c) = r0 >= c ? acc[j][e] * expf(g0 - gc) * scale : 0.0f;
            A_view.at(r1, c) = r1 >= c ? acc[j][e + 2] * expf(g1 - gc) * scale : 0.0f;
        }
    }
}

// m16n8k8 TF32 A operand from an FP32 accumulator tile, with the contraction index permuted
// (logical k = lid holds column 2 lid, k = lid + 4 holds 2 lid + 1); the B operand reads the
// same permutation.
__device__ __forceinline__ void tf32_a_from_acc(const float c[4], float a[4]) {
    a[0] = c[0];
    a[1] = c[2];
    a[2] = c[1];
    a[3] = c[3];
}

__launch_bounds__(kThreads, 1) __global__
    void fused_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                      const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                      const float* __restrict__ beta, const __nv_bfloat16* state_in,
                      __nv_bfloat16* state_out, __nv_bfloat16* __restrict__ out, head_map qk_map,
                      float scale, int chunks, int valid_tokens) {
    using L = smem_layout;
    extern __shared__ __align__(16) unsigned char smem[];
    auto* q_s     = reinterpret_cast<__nv_bfloat16*>(smem + L::q);
    auto* kv_s    = reinterpret_cast<__nv_bfloat16*>(smem + L::kv);
    float* scr    = reinterpret_cast<float*>(smem + L::scr);
    float* g_raw  = reinterpret_cast<float*>(smem + L::g_raw);
    float* b_raw  = reinterpret_cast<float*>(smem + L::b_raw);
    float* gc_s   = reinterpret_cast<float*>(smem + L::gc);
    float* beta_s = reinterpret_cast<float*>(smem + L::beta);
    const SmemTile<BT> T_view{reinterpret_cast<float*>(smem + L::t)};
    const SmemTile<BT> A_view{reinterpret_cast<float*>(smem + L::a)};

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int h    = static_cast<int>(blockIdx.x);
    const int H_v  = qk_map.H_v;

    const std::int64_t qk_row         = static_cast<std::int64_t>(qk_map.H_qk) * kStateDim;
    const std::int64_t v_row          = static_cast<std::int64_t>(H_v) * kStateDim;
    const __nv_bfloat16* q_head       = q + static_cast<std::int64_t>(qk_map.qk_head(h)) * kStateDim;
    const __nv_bfloat16* k_head       = k + static_cast<std::int64_t>(qk_map.qk_head(h)) * kStateDim;
    const __nv_bfloat16* v_head       = v + static_cast<std::int64_t>(h) * kStateDim;
    __nv_bfloat16* out_head           = out + static_cast<std::int64_t>(h) * kStateDim;
    const std::int64_t state_base     = static_cast<std::int64_t>(h) * kStateDim * kStateDim;

    // This warp's value rows are dv0 and dv0 + 8; tile n covers key columns [8n, 8n + 8).
    const int dv0 = warp * 16 + gid;
    float S[kStateDim / 8][4];
#pragma unroll
    for (int n = 0; n < kStateDim / 8; ++n) {
        const int dk    = 8 * n + 2 * lid;
        const float2 lo = bf16x2_to_float2(
            load_vec<__nv_bfloat162>(state_in + state_base + dv0 * kStateDim + dk));
        const float2 hi = bf16x2_to_float2(
            load_vec<__nv_bfloat162>(state_in + state_base + (dv0 + 8) * kStateDim + dk));
        S[n][0] = lo.x;
        S[n][1] = lo.y;
        S[n][2] = hi.x;
        S[n][3] = hi.y;
    }

    // Chunk c's K sits in tile c & 1 and its V in the other.
    const auto rows_of   = [&](int c) { return valid_tokens - c * BT; };
    const auto issue_q   = [&](int c) {
        if (c < chunks) {
            load_tile(q_s, q_head + static_cast<std::int64_t>(c) * BT * qk_row, qk_row, BT, tid);
            load_gates(g_raw + (c & 1) * BT, b_raw + (c & 1) * BT, g, beta,
                       static_cast<std::int64_t>(c) * BT * H_v + h, H_v, rows_of(c), tid);
        }
        cp_commit();
    };
    const auto issue_k = [&](int c) {
        if (c < chunks) {
            load_tile(kv_s + (c & 1) * kTileElems,
                      k_head + static_cast<std::int64_t>(c) * BT * qk_row, qk_row, BT, tid);
        }
        cp_commit();
    };
    const auto issue_v = [&](int c) {
        if (c < chunks) {
            load_tile(kv_s + ((c + 1) & 1) * kTileElems,
                      v_head + static_cast<std::int64_t>(c) * BT * v_row, v_row, rows_of(c), tid);
        }
        cp_commit();
    };

    issue_q(0);
    issue_k(0);
    issue_v(0);
    cp_wait<2>();
    __syncthreads();
    if (warp == 0) { scan_gates(gc_s, beta_s, g_raw, b_raw, lane); }

    for (int c = 0; c < chunks; ++c) {
        const __nv_bfloat16* k_s = kv_s + (c & 1) * kTileElems;
        const __nv_bfloat16* v_s = kv_s + ((c + 1) & 1) * kTileElems;
        const int cs             = c * BT;

        cp_wait<1>();
        __syncthreads(); // K(c) landed; this chunk's gates are scanned

        // ---- Phase 1: S Q^T and S K^T for every warp; T (warps 0-3) and the decayed causal
        // Q K^T (warps 4-7). The inverse warps take the state products first, so the other
        // half's tensor-core work covers their serial solve.
        float O[BT / 8][4];
        float X[BT / 8][4];
        const auto state_products = [&]() {
#pragma unroll
            for (int n = 0; n < BT / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { O[n][e] = X[n][e] = 0.0f; }
            }
            const int b_row = (lane & 7) + ((lane >> 4) << 3);
            const int b_col = ((lane >> 3) & 1) << 3;
#pragma unroll
            for (int j = 0; j < kStateDim / 16; ++j) {
                const unsigned a0 = pack_bf16x2(S[2 * j][0], S[2 * j][1]);
                const unsigned a1 = pack_bf16x2(S[2 * j][2], S[2 * j][3]);
                const unsigned a2 = pack_bf16x2(S[2 * j + 1][0], S[2 * j + 1][1]);
                const unsigned a3 = pack_bf16x2(S[2 * j + 1][2], S[2 * j + 1][3]);
#pragma unroll
                for (int t2 = 0; t2 < BT / 16; ++t2) {
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4(b0, b1, b2, b3,
                                smem_addr(q_s + tile_off(16 * t2 + b_row, 16 * j + b_col)));
                    mma_bf16(O[2 * t2][0], O[2 * t2][1], O[2 * t2][2], O[2 * t2][3], a0, a1, a2,
                             a3, b0, b1);
                    mma_bf16(O[2 * t2 + 1][0], O[2 * t2 + 1][1], O[2 * t2 + 1][2],
                             O[2 * t2 + 1][3], a0, a1, a2, a3, b2, b3);
                    ldmatrix_x4(b0, b1, b2, b3,
                                smem_addr(k_s + tile_off(16 * t2 + b_row, 16 * j + b_col)));
                    mma_bf16(X[2 * t2][0], X[2 * t2][1], X[2 * t2][2], X[2 * t2][3], a0, a1, a2,
                             a3, b0, b1);
                    mma_bf16(X[2 * t2 + 1][0], X[2 * t2 + 1][1], X[2 * t2 + 1][2],
                             X[2 * t2 + 1][3], a0, a1, a2, a3, b2, b3);
                }
            }
        };
        if (warp < kInverseWarps) {
            state_products();
            switch (warp) {
            case 0: inverse_role<0>(T_view, scr, k_s, gc_s, beta_s, warp, lane); break;
            case 1: inverse_role<1>(T_view, scr, k_s, gc_s, beta_s, warp, lane); break;
            case 2: inverse_role<2>(T_view, scr, k_s, gc_s, beta_s, warp, lane); break;
            default: inverse_role<3>(T_view, scr, k_s, gc_s, beta_s, warp, lane); break;
            }
        } else {
            switch (warp) {
            case 4: qk_role<0>(A_view, q_s, k_s, gc_s, scale, lane); break;
            case 5: qk_role<1>(A_view, q_s, k_s, gc_s, scale, lane); break;
            case 6: qk_role<2>(A_view, q_s, k_s, gc_s, scale, lane); break;
            default: qk_role<3>(A_view, q_s, k_s, gc_s, scale, lane); break;
            }
            state_products();
        }

        cp_wait<0>();
        __syncthreads(); // V(c) landed; T and Q K^T are complete; Q(c) is consumed
        issue_q(c + 1);

        // ---- Phase 2: X = beta (V - exp(gc) S K^T), delta = T X, out.
        {
            const int v_row_l = (lane & 7) + ((lane >> 4) << 3);
            const int v_col   = warp * 16 + (((lane >> 3) & 1) << 3);
#pragma unroll
            for (int j = 0; j < BT / 16; ++j) {
                unsigned va[4];
                ldmatrix_x4_t(va[0], va[1], va[2], va[3],
                              smem_addr(v_s + tile_off(16 * j + v_row_l, v_col)));
                // va[0]/va[1]: rows dv0/dv0+8 at tokens 16j + 2lid, +1; va[2]/va[3] at +8.
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int n     = 2 * j + half;
                    const int t     = 8 * n + 2 * lid;
                    const float e0  = expf(gc_s[t]);
                    const float e1  = expf(gc_s[t + 1]);
                    const float b0  = beta_s[t];
                    const float b1  = beta_s[t + 1];
                    const float2 lo = bf16x2_bits_to_float2(va[2 * half]);
                    const float2 hi = bf16x2_bits_to_float2(va[2 * half + 1]);
                    X[n][0]         = b0 * (lo.x - e0 * X[n][0]);
                    X[n][1]         = b1 * (lo.y - e1 * X[n][1]);
                    X[n][2]         = b0 * (hi.x - e0 * X[n][2]);
                    X[n][3]         = b1 * (hi.y - e1 * X[n][3]);
                    // The S Q^T part of the output, decayed and scaled.
                    O[n][0] *= scale * e0;
                    O[n][1] *= scale * e1;
                    O[n][2] *= scale * e0;
                    O[n][3] *= scale * e1;
                }
            }
        }
        // delta = T X over the lower triangle, in place: tile n needs X tiles 0..n only, so going
        // down from the last tile overwrites each X tile after its last use.
#pragma unroll
        for (int n = BT / 8 - 1; n >= 0; --n) {
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j <= n; ++j) {
                float a[4];
                tf32_a_from_acc(X[j], a);
                const float2 b = *reinterpret_cast<const float2*>(
                    &T_view.at(8 * n + gid, 8 * j + 2 * lid));
                mma_tf32(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b.x, b.y);
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) { X[n][e] = d[e]; }
        }
        // out += (decayed causal Q K^T) delta.
#pragma unroll
        for (int n = 0; n < BT / 8; ++n) {
#pragma unroll
            for (int j = 0; j <= n; ++j) {
                float a[4];
                tf32_a_from_acc(X[j], a);
                const float2 b = *reinterpret_cast<const float2*>(
                    &A_view.at(8 * n + gid, 8 * j + 2 * lid));
                mma_tf32(O[n][0], O[n][1], O[n][2], O[n][3], a[0], a[1], a[2], a[3], b.x, b.y);
            }
        }
        // Store: a lane holds two value rows at two tokens; trading one value with the lane four
        // apart gives each a pair of adjacent value rows at one token.
        {
            const bool even = (gid & 1) == 0;
#pragma unroll
            for (int n = 0; n < BT / 8; ++n) {
                const int t = cs + 8 * n + 2 * lid;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const float v0   = O[n][2 * half];
                    const float v1   = O[n][2 * half + 1];
                    const float recv = __shfl_xor_sync(0xffffffffu, even ? v1 : v0, 4);
                    const int dv     = dv0 + 8 * half - (even ? 0 : 1);
                    const int token  = even ? t : t + 1;
                    if (token < valid_tokens) {
                        store_vec(out_head + static_cast<std::int64_t>(token) * v_row + dv,
                                  pack_bf16x2(even ? v0 : recv, even ? recv : v1));
                    }
                }
            }
        }

        __syncthreads(); // V(c) is consumed
        issue_k(c + 1);

        // ---- Phase 3: S = exp(gc_last) S + (exp(gc_last - gc) delta)^T K.
        {
            const float g_last = gc_s[BT - 1];
            const float decay  = expf(g_last);
#pragma unroll
            for (int n = 0; n < kStateDim / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { S[n][e] *= decay; }
            }
            const int k_row_l = (lane & 7) + (((lane >> 3) & 1) << 3);
            const int k_col   = (lane >> 4) << 3;
#pragma unroll
            for (int j = 0; j < BT / 16; ++j) {
                const int t0      = 16 * j + 2 * lid;
                const float d00   = expf(g_last - gc_s[t0]);
                const float d01   = expf(g_last - gc_s[t0 + 1]);
                const float d10   = expf(g_last - gc_s[t0 + 8]);
                const float d11   = expf(g_last - gc_s[t0 + 9]);
                const unsigned a0 = pack_bf16x2(X[2 * j][0] * d00, X[2 * j][1] * d01);
                const unsigned a1 = pack_bf16x2(X[2 * j][2] * d00, X[2 * j][3] * d01);
                const unsigned a2 = pack_bf16x2(X[2 * j + 1][0] * d10, X[2 * j + 1][1] * d11);
                const unsigned a3 = pack_bf16x2(X[2 * j + 1][2] * d10, X[2 * j + 1][3] * d11);
#pragma unroll
                for (int n2 = 0; n2 < kStateDim / 16; ++n2) {
                    unsigned b0, b1, b2, b3;
                    ldmatrix_x4_t(b0, b1, b2, b3,
                                  smem_addr(k_s + tile_off(16 * j + k_row_l, 16 * n2 + k_col)));
                    mma_bf16(S[2 * n2][0], S[2 * n2][1], S[2 * n2][2], S[2 * n2][3], a0, a1, a2,
                             a3, b0, b1);
                    mma_bf16(S[2 * n2 + 1][0], S[2 * n2 + 1][1], S[2 * n2 + 1][2],
                             S[2 * n2 + 1][3], a0, a1, a2, a3, b2, b3);
                }
            }
#pragma unroll
            for (int n = 0; n < kStateDim / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    S[n][e] = __bfloat162float(__float2bfloat16_rn(S[n][e]));
                }
            }
        }

        cp_wait<1>();
        __syncthreads(); // K(c) is consumed; the next chunk's Q and gates landed
        if (warp == 0 && c + 1 < chunks) {
            scan_gates(gc_s, beta_s, g_raw + ((c + 1) & 1) * BT, b_raw + ((c + 1) & 1) * BT,
                       lane);
        }
        issue_v(c + 1);
    }
    cp_wait<0>();

#pragma unroll
    for (int n = 0; n < kStateDim / 8; ++n) {
        const int dk = 8 * n + 2 * lid;
        store_vec(state_out + state_base + dv0 * kStateDim + dk, pack_bf16x2(S[n][0], S[n][1]));
        store_vec(state_out + state_base + (dv0 + 8) * kStateDim + dk,
                  pack_bf16x2(S[n][2], S[n][3]));
    }
}

} // namespace sinfer::ops::detail::gated_delta_net::chunked::fused
