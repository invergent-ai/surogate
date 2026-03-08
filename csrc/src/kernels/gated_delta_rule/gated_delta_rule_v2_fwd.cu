// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_rule_v2.cuh"
#include "gdr_fwd_launchers.h"

#include <algorithm>

namespace {

// ============================================================================
// WMMA forward kernel — tensor core accelerated
// Requires K, V multiples of 16 and ≤ 64.
//
// Shared memory layout (K=V=64):
//   smem_h      [K×V] float       = 16 KB  (persistent state)
//   scratch1    [64×64] float     = 16 KB  (WMMA output 1)
//   scratch2    [64×64] float     = 16 KB  (WMMA output 2)
//   smem_k      [64×K] TQ         =  8 KB
//   smem_q      [64×K] TQ         =  8 KB
//   buf1        [64×64] TQ        =  8 KB  (M_bf16 → h_bf16 → S)
//   buf2        [64×V] TQ         =  8 KB  (v → bv → u → v_new_pre → v_scaled)
//   buf3        [64×K] TQ         =  8 KB  (bkg → w)
//   scalars     [4×64] float      =  1 KB
//   Total: ~89 KB
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_fwd_wmma(
    TQ* __restrict__ out,
    float* __restrict__ final_state,
    float* __restrict__ state_scratch,
    float* __restrict__ fwd_checkpoints,   // [B, H, num_chunks+1, K, V] or nullptr
    const TQ* __restrict__ q,
    const TQ* __restrict__ k,
    const TQ* __restrict__ v,
    const TG* __restrict__ g,
    const TB* __restrict__ beta,
    const float* __restrict__ initial_state,
    int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, float scale, bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const long bh = (long)b * H + h;
    const long kv = (long)Kdim * Vdim;
    const int Lp = kMaxC;  // always pad to 64

    // Shared memory
    extern __shared__ char smem_raw[];
    float* smem_h    = (float*)smem_raw;
    float* scratch1  = smem_h + Kdim * Vdim;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    smem_q    = smem_k + Lp * Kdim;
    TQ*    buf1      = smem_q + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * Vdim;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invq = smem_beta + Lp;
    float* smem_invk = smem_invq + Lp;

    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;

    // Init state
    for (long idx = tid; idx < kv; idx += nthr)
        smem_h[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    __syncthreads();

    // Save checkpoint[0] = initial state
    if (fwd_checkpoints) {
        float* cp_base = fwd_checkpoints + bh * (long)(num_chunks + 1) * kv;
        for (long idx = tid; idx < kv; idx += nthr)
            cp_base[idx] = smem_h[idx];
    }
    __syncthreads();

    for (int cs = 0; cs < Tlen; cs += chunk_size) {
        const int L = min(chunk_size, Tlen - cs);

        // --- Zero-fill all buffers for WMMA padding ---
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            smem_k[idx] = from_float<TQ>(0.0f);
            smem_q[idx] = from_float<TQ>(0.0f);
        }
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf3[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        if (tid < Lp) { smem_gcum[tid] = 0.0f; smem_beta[tid] = 0.0f; }
        __syncthreads();

        // --- Load k, q, v from global ---
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim, kk = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k[gi];
            smem_q[pos * Kdim + kk] = q[gi];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Vdim + vv;
            buf2[pos * Vdim + vv] = v[gi];
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g[gh]);
                smem_gcum[i] = acc;
                smem_beta[i] = to_float(beta[gh]);
            }
        }
        __syncthreads();

        // --- L2 norms + normalize ---
        for (int pos = tid; pos < L; pos += nthr) {
            if (use_qk_l2norm_in_kernel) {
                float qn2 = 0.0f, kn2 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float qv = to_float(smem_q[pos * Kdim + kk]);
                    float kv2 = to_float(smem_k[pos * Kdim + kk]);
                    qn2 += qv * qv; kn2 += kv2 * kv2;
                }
                smem_invq[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
                smem_invk[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                smem_invq[pos] = 1.0f; smem_invk[pos] = 1.0f;
            }
        }
        __syncthreads();
        if (use_qk_l2norm_in_kernel) {
            for (int idx = tid; idx < L * Kdim; idx += nthr) {
                const int pos = idx / Kdim, kk = idx % Kdim;
                smem_k[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_invk[pos]));
                smem_q[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_q[pos * Kdim + kk]) * smem_invq[pos]));
            }
            __syncthreads();
        }

        // --- Phase 2: A = beta * exp(g) * (k @ k^T), then solve M ---
        // kkt = k @ k^T → scratch1[Lp×Lp]
        wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();

        // Scale A and zero upper triangle
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] *= smem_beta[i] * expf(smem_gcum[i] - smem_gcum[j]);
            else
                scratch1[idx] = 0.0f;
        }
        __syncthreads();

        // Solve M in scratch2: M[i,i]=1, M[i,j] = -Σ_{m=j}^{i-1} A[i,m]*M[m,j]
        for (int idx = tid; idx < Lp * Lp; idx += nthr) scratch2[idx] = 0.0f;
        __syncthreads();
        for (int i = tid; i < Lp; i += nthr) scratch2[i * Lp + i] = 1.0f;
        __syncthreads();
        for (int row = 1; row < L; ++row) {
            for (int j = tid; j < row; j += nthr) {
                float s = 0.0f;
                for (int m = j; m < row; ++m)
                    s += scratch1[row * Lp + m] * scratch2[m * Lp + j];
                scratch2[row * Lp + j] = -s;
            }
            __syncthreads();
        }

        // Convert M → bf16 in buf1 with truncation
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
        __syncthreads();

        // --- Phase 3: bv = beta*v, then u = M @ bv ---
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) * smem_beta[i]));
        }
        __syncthreads();
        wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();

        // --- Phase 4: bkg = beta*k*exp(g), then w = M @ bkg ---
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * expf(smem_gcum[i]) : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();
        wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();

        // --- Phase 5: h_bf16, then v_new_pre = u - w @ h_bf16 ---
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(smem_h[idx]));
        for (int idx = Kdim * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        // wh = w[Lp×K] @ h_bf16[K×V] → scratch1[Lp×V]
        wmma_nn<TQ>(buf3, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) - scratch1[idx]));
        __syncthreads();

        // --- Phase 6: term1 = q @ h_bf16 → scratch1[Lp×V] ---
        // h_bf16 is still in buf1[K×V]
        wmma_nn<TQ>(smem_q, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();

        // --- Phase 7: S = q @ k^T → scratch2, mask+gate → buf1 ---
        wmma_nt<TQ>(smem_q, Kdim, smem_k, Kdim, scratch2, Lp, Lp, Lp, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            float val = (i < L && j <= i) ?
                bf16_trunc<TQ>(expf(smem_gcum[i] - smem_gcum[j]) * scratch2[idx]) : 0.0f;
            buf1[idx] = from_float<TQ>(val);
        }
        __syncthreads();

        // --- Phase 8: term2 = S @ v_new_pre → scratch2[Lp×V] ---
        wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch2, Vdim, Lp, Vdim, Lp);
        __syncthreads();

        // --- Phase 9: Write output ---
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim, vv = idx % Vdim;
            const long oi = (((long)b * Tlen + cs + i) * H + h) * Vdim + vv;
            out[oi] = from_float<TQ>(
                scale * (expf(smem_gcum[i]) * scratch1[i * Vdim + vv]
                        + scratch2[i * Vdim + vv]));
        }

        // --- Phase 10: State update ---
        // v_scaled[i,v] = bf16(v_new_pre[i,v] * exp(g_last - g_cum[i]))
        const float g_last = (L > 0) ? smem_gcum[L - 1] : 0.0f;
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float val = (i < L) ?
                bf16_trunc<TQ>(to_float(buf2[idx]) * expf(g_last - smem_gcum[i])) : 0.0f;
            buf2[idx] = from_float<TQ>(val);
        }
        __syncthreads();
        // K^T @ v_scaled → scratch1[K×V]
        wmma_tn<TQ>(smem_k, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();
        const float eg = expf(g_last);
        for (long idx = tid; idx < kv; idx += nthr)
            smem_h[idx] = eg * smem_h[idx] + scratch1[idx];
        __syncthreads();

        // Save checkpoint for this chunk boundary
        if (fwd_checkpoints) {
            const int chunk_idx = cs / chunk_size + 1;
            float* cp = fwd_checkpoints + bh * (long)(num_chunks + 1) * kv + (long)chunk_idx * kv;
            for (long idx = tid; idx < kv; idx += nthr)
                cp[idx] = smem_h[idx];
            __syncthreads();
        }
    }

    // Write final state
    for (long idx = tid; idx < kv; idx += nthr) {
        final_state[bh * kv + idx] = smem_h[idx];
        state_scratch[bh * kv + idx] = smem_h[idx];
    }
}


// ============================================================================
// Forward kernel — fully parallelized (scalar fallback)
// ============================================================================
//
// Shared memory layout (for K=V=64, chunk_size=64):
//   smem_state  [K×V] float    = 16 KB
//   smem_M      [C×C] float    = 16 KB  (reused as S/h_bf16 in output phase)
//   smem_k      [C×K] TQ       =  8 KB  (normalized K)
//   smem_q      [C×K] TQ       =  8 KB  (normalized Q)
//   smem_v      [C×V] TQ       =  8 KB  (raw V, reused as S in output phase)
//   smem_u      [C×V] TQ       =  8 KB  (u then v_new_pre)
//   smem_w      [C×K] TQ       =  8 KB  (w, reused as h_bf16 in output phase)
//   scalars     [5×C] float    =  1.25KB (g_cum, beta, inv_q, inv_k, A_row_cache)
// Total: ~73 KB
//
template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_fwd_v2(
    TQ* __restrict__ out,
    float* __restrict__ final_state,
    float* __restrict__ state_scratch,
    const TQ* __restrict__ q,
    const TQ* __restrict__ k,
    const TQ* __restrict__ v,
    const TG* __restrict__ g,
    const TB* __restrict__ beta,
    const float* __restrict__ initial_state,
    int Tlen,
    int H,
    int Kdim,
    int Vdim,
    int chunk_size,
    float scale,
    bool use_qk_l2norm_in_kernel) {

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const long bh = static_cast<long>(b) * H + h;
    const long kv = static_cast<long>(Kdim) * Vdim;

    // --- Shared memory pointers ---
    extern __shared__ char smem_raw[];
    float*  smem_state = reinterpret_cast<float*>(smem_raw);
    float*  smem_M     = smem_state + Kdim * Vdim;
    TQ*     smem_k     = reinterpret_cast<TQ*>(smem_M + kMaxC * kMaxC);
    TQ*     smem_q     = smem_k + kMaxC * Kdim;
    TQ*     smem_v     = smem_q + kMaxC * Kdim;
    TQ*     smem_u     = smem_v + kMaxC * Vdim;
    TQ*     smem_w     = smem_u + kMaxC * Vdim;
    float*  smem_g_cum = reinterpret_cast<float*>(smem_w + kMaxC * Kdim);
    float*  smem_beta  = smem_g_cum + kMaxC;
    float*  smem_inv_q = smem_beta  + kMaxC;
    float*  smem_inv_k = smem_inv_q + kMaxC;

    // Initialize state from initial_state or zeros
    for (long idx = tid; idx < kv; idx += nthreads) {
        smem_state[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    }
    __syncthreads();

    for (int cs = 0; cs < Tlen; cs += chunk_size) {
        const int L = min(chunk_size, Tlen - cs);

        // ================================================================
        // Phase 1: Cooperative load of k, q, v + compute scalars
        // ================================================================
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int pos = idx / Kdim;
            const int kk  = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k[gi];
            smem_q[pos * Kdim + kk] = q[gi];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int pos = idx / Vdim;
            const int vv  = idx % Vdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Vdim + vv;
            smem_v[pos * Vdim + vv] = v[gi];
        }
        // g_cum and beta — single thread (L ≤ 64, negligible)
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g[gh]);
                smem_g_cum[i] = acc;
                smem_beta[i]  = to_float(beta[gh]);
            }
        }
        __syncthreads();

        // L2 norms — one thread per position
        for (int pos = tid; pos < L; pos += nthreads) {
            if (use_qk_l2norm_in_kernel) {
                float qn2 = 0.0f, kn2 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float qv = to_float(smem_q[pos * Kdim + kk]);
                    float kv2 = to_float(smem_k[pos * Kdim + kk]);
                    qn2 += qv * qv;
                    kn2 += kv2 * kv2;
                }
                smem_inv_q[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
                smem_inv_k[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                smem_inv_q[pos] = 1.0f;
                smem_inv_k[pos] = 1.0f;
            }
        }
        __syncthreads();

        // Normalize k and q in-place (bf16 truncation)
        if (use_qk_l2norm_in_kernel) {
            for (int idx = tid; idx < L * Kdim; idx += nthreads) {
                const int pos = idx / Kdim;
                const int kk  = idx % Kdim;
                smem_k[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_inv_k[pos]));
                smem_q[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_q[pos * Kdim + kk]) * smem_inv_q[pos]));
            }
            __syncthreads();
        }

        // ================================================================
        // Phase 2: Build M matrix (triangular solve)
        //   M[i,i] = 1,  M[i,j] = bf16(-Σ_{m=j}^{i-1} A[i,m]*M[m,j])
        //   where A[i,m] = beta[i] * dot(k_norm[i], k_norm[m]) * exp(g_cum[i] - g_cum[m])
        //
        //   Process row by row (sequential across rows).
        //   Within each row, parallelize across columns j.
        // ================================================================

        // Zero M
        for (int idx = tid; idx < kMaxC * kMaxC; idx += nthreads) {
            smem_M[idx] = 0.0f;
        }
        __syncthreads();

        // Set diagonal
        for (int i = tid; i < L; i += nthreads) {
            smem_M[i * kMaxC + i] = 1.0f;
        }
        __syncthreads();

        for (int row = 1; row < L; ++row) {
            // Each thread computes M[row, j] for some j < row
            for (int j = tid; j < row; j += nthreads) {
                float s = 0.0f;
                for (int m = j; m < row; ++m) {
                    // A[row, m]
                    float dot_k = 0.0f;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        dot_k += to_float(smem_k[row * Kdim + kk])
                               * to_float(smem_k[m   * Kdim + kk]);
                    }
                    float a_rm = smem_beta[row] * dot_k
                               * expf(smem_g_cum[row] - smem_g_cum[m]);
                    s += a_rm * smem_M[m * kMaxC + j];
                }
                smem_M[row * kMaxC + j] = bf16_trunc<TQ>(-s);
            }
            __syncthreads();
        }

        // ================================================================
        // Phase 3: u = M @ (beta * V)   [L×V]
        //   u[i,v] = bf16( Σ_m M[i,m] * bf16(V[m,v] * beta[m]) )
        // ================================================================
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int i  = idx / Vdim;
            const int vv = idx % Vdim;
            float acc = 0.0f;
            for (int m = 0; m <= i; ++m) {
                float vb = bf16_trunc<TQ>(to_float(smem_v[m * Vdim + vv]) * smem_beta[m]);
                acc += smem_M[i * kMaxC + m] * vb;
            }
            smem_u[i * Vdim + vv] = from_float<TQ>(bf16_trunc<TQ>(acc));
        }
        __syncthreads();

        // ================================================================
        // Phase 4: w = M @ (beta * k_norm * exp(g_cum))   [L×K]
        //   w[i,k] = bf16( Σ_m M[i,m] * bf16(k_norm[m,k] * beta[m] * exp(g_cum[m])) )
        // ================================================================
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int i  = idx / Kdim;
            const int kk = idx % Kdim;
            float acc = 0.0f;
            for (int m = 0; m <= i; ++m) {
                float kbg = bf16_trunc<TQ>(
                    to_float(smem_k[m * Kdim + kk]) * smem_beta[m] * expf(smem_g_cum[m]));
                acc += smem_M[i * kMaxC + m] * kbg;
            }
            smem_w[i * Kdim + kk] = from_float<TQ>(bf16_trunc<TQ>(acc));
        }
        __syncthreads();

        // ================================================================
        // Phase 5: v_new_pre = bf16(u - w @ bf16(state))   [L×V]
        //   Stored in smem_u (overwrites u).
        // ================================================================
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int i  = idx / Vdim;
            const int vv = idx % Vdim;
            float wh = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float h_val = bf16_trunc<TQ>(smem_state[kk * Vdim + vv]);
                wh += to_float(smem_w[i * Kdim + kk]) * h_val;
            }
            float u_val = to_float(smem_u[i * Vdim + vv]);
            smem_u[i * Vdim + vv] = from_float<TQ>(bf16_trunc<TQ>(u_val - wh));
        }
        __syncthreads();

        // ================================================================
        // Phase 6: Output computation
        //   out[i,v] = scale * (term1 + term2)
        //   term1 = exp(g_cum[i]) * Σ_k q_norm[i,k] * bf16(state[k,v])
        //   term2 = Σ_{j≤i} bf16(exp(g_cum[i]-g_cum[j]) * dot(q[i],k[j])) * v_new_pre[j,v]
        //
        //   We reuse smem_v and smem_w as scratch (no longer needed).
        //   smem_v → S[L×L] bf16 (attention scores, lower triangular)
        //   smem_w → h_bf16[K×V] bf16
        // ================================================================
        TQ* smem_S     = smem_v;   // [L×L] bf16 — fits in smem_v space (L*V == L*L for K=V)
        TQ* smem_h_bf16 = smem_w;  // [K×V] bf16

        // Prepare bf16-truncated state
        for (int idx = tid; idx < Kdim * Vdim; idx += nthreads) {
            smem_h_bf16[idx] = from_float<TQ>(smem_state[idx]);
        }
        __syncthreads();

        // Build S matrix (lower triangular attention scores)
        for (int idx = tid; idx < L * L; idx += nthreads) {
            const int i = idx / L;
            const int j = idx % L;
            if (j <= i) {
                float dot_qk = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_qk += to_float(smem_q[i * Kdim + kk])
                            * to_float(smem_k[j * Kdim + kk]);
                }
                smem_S[i * L + j] = from_float<TQ>(
                    expf(smem_g_cum[i] - smem_g_cum[j]) * dot_qk);
            } else {
                smem_S[i * L + j] = from_float<TQ>(0.0f);
            }
        }
        __syncthreads();

        // Compute output: term1 + term2
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int i  = idx / Vdim;
            const int vv = idx % Vdim;

            // term1 = exp(g_cum[i]) * Q_norm[i,:] @ h_bf16[:,v]
            float t1 = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                t1 += to_float(smem_q[i * Kdim + kk])
                    * to_float(smem_h_bf16[kk * Vdim + vv]);
            }
            t1 *= expf(smem_g_cum[i]);

            // term2 = Σ_{j≤i} S[i,j] * v_new_pre[j,v]
            float t2 = 0.0f;
            for (int j = 0; j <= i; ++j) {
                t2 += to_float(smem_S[i * L + j])
                    * to_float(smem_u[j * Vdim + vv]);
            }

            const long oi = (((long)b * Tlen + cs + i) * H + h) * Vdim + vv;
            out[oi] = from_float<TQ>((t1 + t2) * scale);
        }
        __syncthreads();

        // ================================================================
        // Phase 7: State update (in-place)
        //   state[k,v] = exp(g_last) * state[k,v]
        //              + Σ_i k_norm[i,k] * bf16(v_new_pre[i,v] * exp(g_last - g_cum[i]))
        // ================================================================
        const float g_last = smem_g_cum[L - 1];
        const float eg_last = expf(g_last);

        for (int idx = tid; idx < Kdim * Vdim; idx += nthreads) {
            const int kk = idx / Vdim;
            const int vv = idx % Vdim;
            float acc = smem_state[kk * Vdim + vv] * eg_last;
            for (int i = 0; i < L; ++i) {
                float v_scaled = bf16_trunc<TQ>(
                    to_float(smem_u[i * Vdim + vv]) * expf(g_last - smem_g_cum[i]));
                acc += to_float(smem_k[i * Kdim + kk]) * v_scaled;
            }
            smem_state[kk * Vdim + vv] = acc;
        }
        __syncthreads();
    }

    // Write final state to global memory
    for (long idx = tid; idx < kv; idx += nthreads) {
        final_state[bh * kv + idx] = smem_state[idx];
    }
}


// ============================================================================
// Multi-kernel forward launcher (3 kernels: precompute → state → output)
// ============================================================================

template<typename TQ, typename TG, typename TB>
void launch_fwd_v2_multikernel(
    Tensor& out, Tensor& final_state, Tensor& state_scratch,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints_tensor, Tensor* fwd_workspace_tensor, cudaStream_t stream) {

    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const int Lp = kMaxC;
    int chunk_threads = 512;
    if (const char* env = std::getenv("SUROGATE_GDR_CHUNK_THREADS")) {
        const int parsed = std::atoi(env);
        if (parsed >= 64 && parsed <= 1024) {
            chunk_threads = (parsed / 32) * 32;
        }
    }
    int state_threads = 768;
    if (const char* env = std::getenv("SUROGATE_GDR_FWD_STATE_THREADS")) {
        const int parsed = std::atoi(env);
        if (parsed >= 64 && parsed <= 1024) {
            state_threads = (parsed / 32) * 32;
        }
    }
    int v_tile = Vdim >= 64 ? 64 : Vdim;
    if (const char* env_vtile = std::getenv("SUROGATE_GDR_VTILE")) {
        const int parsed = std::atoi(env_vtile);
        if (parsed > 0) {
            v_tile = std::min(Vdim, parsed);
        }
    }
    // WMMA kernels expect tile multiples of 16.
    if (v_tile >= 16) {
        v_tile = (v_tile / 16) * 16;
    }
    if (v_tile <= 0) {
        v_tile = std::min(Vdim, 16);
    }

    FwdWorkspaceLayout fwl = make_fwd_ws(Lp, Kdim, Vdim);
    const int fwd_ws_stride = fwl.total;
    const long total_ws_floats = (long)B * H * num_chunks * fwd_ws_stride;

    float* fwd_workspace = nullptr;
    bool own_workspace = false;
    if (fwd_workspace_tensor && fwd_workspace_tensor->Data) {
        if (fwd_workspace_tensor->DType != ETensorDType::FP32 ||
            fwd_workspace_tensor->nelem() < total_ws_floats) {
            throw std::logic_error(
                "gated_delta_rule_chunk_forward_v2: provided forward workspace is invalid");
        }
        fwd_workspace = fwd_workspace_tensor->get<float>();
    } else {
        own_workspace = true;
        CUDA_CHECK(cudaMallocAsync(&fwd_workspace, total_ws_floats * sizeof(float), stream));
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    auto require_smem = [&](const char* kernel_name, std::size_t requested) {
        if (requested > static_cast<std::size_t>(smem_optin)) {
            throw std::logic_error(
                std::string("gated_delta_rule_chunk_forward_v2: ")
                + kernel_name + " requires dynamic shared memory " + std::to_string(requested)
                + " bytes, exceeds device opt-in limit " + std::to_string(smem_optin) + " bytes");
        }
    };

    static bool profile = (std::getenv("SUROGATE_GDR_PROFILE") != nullptr);
    std::vector<cudaEvent_t> ev;
    if (profile) {
        ev.resize(4);
        for (auto& e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev[0], stream);
    }

    // --- Kernel 1: Precompute (chunk-parallel) ---
    // smem: scratch1[Lp×Lp]f + scratch2[Lp×Lp]f
    //     + k[Lp×K]TQ + buf1[Lp×Lp]TQ + buf2[Lp×V]TQ + buf3[Lp×K]TQ
    //     + scalars[3×Lp]f
    const std::size_t precompute_smem =
        static_cast<std::size_t>(Lp) * Lp * sizeof(float) * 2          // scratch1 + scratch2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)             // smem_k
        + static_cast<std::size_t>(Lp) * Lp * sizeof(TQ)               // buf1
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)             // buf2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)             // buf3
        + static_cast<std::size_t>(Lp) * 3 * sizeof(float);            // gcum, beta, invk

    require_smem("gdr_fwd_precompute_wmma", precompute_smem);

    launch_gdr_fwd_precompute<TQ, TG, TB>(
        fwd_workspace,
        k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        fwd_ws_stride,
        B, Tlen, H, Kdim, Vdim, num_chunks, chunk_size,
        use_qk_l2norm_in_kernel,
        chunk_threads, precompute_smem, stream);

    if (profile) cudaEventRecord(ev[1], stream);

    // --- Kernel 2: State propagation (sequential per B,H) ---
    // Tiled state propagation (no full KxV shared state).
    const std::size_t state_smem =
        static_cast<std::size_t>(Kdim) * v_tile * sizeof(float)       // scratch
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)            // buf_w
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)            // buf_k
        + static_cast<std::size_t>(Kdim) * v_tile * sizeof(TQ)        // buf_h
        + static_cast<std::size_t>(Lp) * v_tile * sizeof(TQ)          // buf_vnp
        + static_cast<std::size_t>(Lp) * sizeof(float);               // gcum

    require_smem("gdr_fwd_state_wmma", state_smem);

    launch_gdr_fwd_state<TQ>(
        final_state.get<float>(), state_scratch.get<float>(),
        fwd_checkpoints_tensor->get<float>(),
        fwd_workspace,
        initial_state ? initial_state->get<float>() : nullptr,
        fwd_ws_stride,
        B, Tlen, H, Kdim, Vdim, num_chunks, chunk_size,
        v_tile,
        state_threads, state_smem, stream);

    if (profile) cudaEventRecord(ev[2], stream);

    // --- Kernel 3: Output (chunk-parallel) ---
    // smem: scratch_v[Lp×Vtile]f + scratch_s[Lp×Lp]f
    //     + smem_q[Lp×K]TQ + smem_k[Lp×K]TQ + buf_h[K×Vtile]TQ
    //     + buf_S[Lp×Lp]TQ + buf_vnp[Lp×Vtile]TQ
    //     + gcum[Lp]f + invq[Lp]f
    const std::size_t output_smem =
        static_cast<std::size_t>(Lp) * v_tile * sizeof(float)         // scratch_v
        + static_cast<std::size_t>(Lp) * Lp * sizeof(float)           // scratch_s
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)            // smem_q
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)            // smem_k
        + static_cast<std::size_t>(Kdim) * v_tile * sizeof(TQ)        // buf_h
        + static_cast<std::size_t>(Lp) * Lp * sizeof(TQ)              // buf_S
        + static_cast<std::size_t>(Lp) * v_tile * sizeof(TQ)          // buf_vnp
        + static_cast<std::size_t>(Lp) * 2 * sizeof(float);           // gcum + invq

    require_smem("gdr_fwd_output_wmma", output_smem);
    if (std::getenv("SUROGATE_DEBUG_GDR")) {
        fprintf(stderr,
                "[GDR fwd cfg] mode=tiled_multi K=%d V=%d Vtile=%d smem_optin=%d pre=%zu state=%zu out=%zu\n",
                Kdim, Vdim, v_tile, smem_optin, precompute_smem, state_smem, output_smem);
    }

    launch_gdr_fwd_output<TQ>(
        out.get<TQ>(),
        q.get<TQ>(),
        fwd_checkpoints_tensor->get<float>(),
        fwd_workspace,
        fwd_ws_stride,
        B, Tlen, H, Kdim, Vdim, num_chunks, chunk_size, v_tile, scale,
        use_qk_l2norm_in_kernel,
        chunk_threads, output_smem, stream);

    if (profile) {
        cudaEventRecord(ev[3], stream);
        cudaEventSynchronize(ev[3]);
        float ms[3];
        for (int i = 0; i < 3; ++i) cudaEventElapsedTime(&ms[i], ev[i], ev[i+1]);
        fprintf(stderr, "[GDR fwd multi] precompute=%.3fms state=%.3fms output=%.3fms total=%.3fms\n",
                ms[0], ms[1], ms[2], ms[0]+ms[1]+ms[2]);
        for (auto& e : ev) cudaEventDestroy(e);
    }

    if (own_workspace) {
        CUDA_CHECK(cudaFreeAsync(fwd_workspace, stream));
    }
}

// ============================================================================
// Launch helpers with dtype dispatch
// ============================================================================

template<typename TQ, typename TG, typename TB>
void launch_fwd_v2(
    Tensor& out, Tensor& final_state, Tensor& state_scratch,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints_tensor, Tensor* fwd_workspace_tensor, cudaStream_t stream) {
    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);

    const int threads = 128;
    dim3 grid(B, H);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    auto require_smem = [&](const char* kernel_name, std::size_t requested) {
        if (requested > static_cast<std::size_t>(smem_optin)) {
            throw std::logic_error(
                std::string("gated_delta_rule_chunk_forward_v2: ")
                + kernel_name + " requires dynamic shared memory " + std::to_string(requested)
                + " bytes, exceeds device opt-in limit " + std::to_string(smem_optin) + " bytes");
        }
    };

    // Try multi-kernel WMMA path for bf16/fp16 when K,V are tensor-core friendly.
    const bool force_scalar_fwd = (std::getenv("SUROGATE_GDR_FORCE_SCALAR_FWD") != nullptr);
    // Supports <=64 generally, and Qwen3.5 K=V=128 via low-SMEM state-kernel aliasing.
    // Requires checkpoints (output kernel reads h_in from them).
    constexpr bool can_wmma = std::is_same_v<TQ, nv_bfloat16> || std::is_same_v<TQ, half>;
    if constexpr (can_wmma) {
        if (!force_scalar_fwd) {
        const bool can_use_multi_wmma =
            (Kdim % 16 == 0) && (Vdim % 16 == 0) &&
            ((Kdim <= 64 && Vdim <= 64) || (Kdim == 128 && Vdim == 128));
        if (can_use_multi_wmma) {
            if (fwd_checkpoints_tensor) {
                launch_fwd_v2_multikernel<TQ, TG, TB>(
                    out, final_state, state_scratch,
                    q, k, v, g, beta, initial_state,
                    scale, chunk_size, use_qk_l2norm_in_kernel,
                    fwd_checkpoints_tensor, fwd_workspace_tensor, stream);
                return;
            }

            // Single-kernel WMMA fallback (no checkpoints)
            const std::size_t wmma_smem =
                static_cast<std::size_t>(Kdim) * Vdim * sizeof(float)           // h
                + static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float) * 2   // scratch1 + scratch2
                + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ) * 2       // k, q
                + static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(TQ)          // buf1
                + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ)           // buf2
                + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)           // buf3
                + static_cast<std::size_t>(kMaxC) * 4 * sizeof(float);          // scalars

            require_smem("gdr_chunk_fwd_wmma", wmma_smem);
            if (std::getenv("SUROGATE_DEBUG_GDR")) {
                fprintf(stderr,
                        "[GDR fwd cfg] mode=single_wmma K=%d V=%d smem_optin=%d smem=%zu\n",
                        Kdim, Vdim, smem_optin, wmma_smem);
            }
            CUDA_CHECK(cudaFuncSetAttribute(
                gdr_chunk_fwd_wmma<TQ, TG, TB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(wmma_smem)));

            gdr_chunk_fwd_wmma<TQ, TG, TB><<<grid, threads, wmma_smem, stream>>>(
                out.get<TQ>(), final_state.get<float>(), state_scratch.get<float>(),
                nullptr,
                q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
                g.get<TG>(), beta.get<TB>(),
                initial_state ? initial_state->get<float>() : nullptr,
                Tlen, H, Kdim, Vdim, chunk_size, scale, use_qk_l2norm_in_kernel);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        }
    }

    // Fallback: scalar kernel
    const std::size_t smem_bytes =
        static_cast<std::size_t>(Kdim) * Vdim * sizeof(float)       // state
        + static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float)   // M
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ) * 2   // k, q
        + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ) * 2   // v, u
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)       // w
        + static_cast<std::size_t>(kMaxC) * 4 * sizeof(float);      // scalars

    require_smem("gdr_chunk_fwd_v2", smem_bytes);
    if (std::getenv("SUROGATE_DEBUG_GDR")) {
        fprintf(stderr,
                "[GDR fwd cfg] mode=scalar K=%d V=%d smem_optin=%d smem=%zu\n",
                Kdim, Vdim, smem_optin, smem_bytes);
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        gdr_chunk_fwd_v2<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    gdr_chunk_fwd_v2<TQ, TG, TB><<<grid, threads, smem_bytes, stream>>>(
        out.get<TQ>(), final_state.get<float>(), state_scratch.get<float>(),
        q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        initial_state ? initial_state->get<float>() : nullptr,
        Tlen, H, Kdim, Vdim, chunk_size, scale, use_qk_l2norm_in_kernel);
    CUDA_CHECK(cudaGetLastError());
}

// Dtype dispatch chain for forward
template<typename TQ, typename TG>
void dispatch_fwd_v2_beta(
    Tensor& out, Tensor& final_state, Tensor& state_scratch,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints_tensor, Tensor* fwd_workspace_tensor, cudaStream_t stream) {
    switch (beta.DType) {
        case ETensorDType::FP32:
            launch_fwd_v2<TQ, TG, float>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel,
                fwd_checkpoints_tensor, fwd_workspace_tensor, stream); return;
        case ETensorDType::BF16:
            launch_fwd_v2<TQ, TG, nv_bfloat16>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel,
                fwd_checkpoints_tensor, fwd_workspace_tensor, stream); return;
        case ETensorDType::FP16:
            launch_fwd_v2<TQ, TG, half>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel,
                fwd_checkpoints_tensor, fwd_workspace_tensor, stream); return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_forward_v2: unsupported beta dtype");
    }
}

template<typename TQ>
void dispatch_fwd_v2_g(
    Tensor& out, Tensor& final_state, Tensor& state_scratch,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints_tensor, Tensor* fwd_workspace_tensor, cudaStream_t stream) {
    switch (g.DType) {
        case ETensorDType::FP32:
            dispatch_fwd_v2_beta<TQ, float>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel,
                fwd_checkpoints_tensor, fwd_workspace_tensor, stream); return;
        case ETensorDType::BF16:
            dispatch_fwd_v2_beta<TQ, nv_bfloat16>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel,
                fwd_checkpoints_tensor, fwd_workspace_tensor, stream); return;
        case ETensorDType::FP16:
            dispatch_fwd_v2_beta<TQ, half>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel,
                fwd_checkpoints_tensor, fwd_workspace_tensor, stream); return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_forward_v2: unsupported g dtype");
    }
}

} // namespace

// ============================================================================
// Public API
// ============================================================================


void gated_delta_rule_chunk_forward_v2(
    Tensor& out,
    Tensor& final_state,
    Tensor& state_scratch,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints,
    Tensor* fwd_workspace,
    cudaStream_t stream) {

    switch (q.DType) {
        case ETensorDType::FP32:
            dispatch_fwd_v2_g<float>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel,
                fwd_checkpoints, fwd_workspace, stream); break;
        case ETensorDType::BF16:
            dispatch_fwd_v2_g<nv_bfloat16>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel,
                fwd_checkpoints, fwd_workspace, stream); break;
        case ETensorDType::FP16:
            dispatch_fwd_v2_g<half>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel,
                fwd_checkpoints, fwd_workspace, stream); break;
        default:
            throw std::logic_error("gated_delta_rule_chunk_forward_v2: unsupported q dtype");
    }
}
