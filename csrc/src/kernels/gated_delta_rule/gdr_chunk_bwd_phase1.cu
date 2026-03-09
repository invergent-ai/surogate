// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_rule_v2.cuh"
#include "gdr_bwd_launchers.h"

namespace {

// ============================================================================
// Multi-kernel backward — Phase 1: chunk-parallel local gradients
//
// Grid: (B*H*num_chunks, 1, 1), each block processes one (b,h,chunk).
    // Recomputes M, W, VNEW(pre-gate) from checkpoints.
// Computes all ds-independent gradients: DQ, DK (term1+term2), DU (term2),
// DG (term1+term2), DB partial, DHT1 (dh contribution from term1).
// Stores results in per-chunk workspace for Phase 2 and 3.
// ============================================================================

template<typename TQ, typename TG, typename TB, bool kHasTail>
__global__ void gdr_bwd_phase1_wmma(
    const TQ* __restrict__ d_out,
    const TQ* __restrict__ q_global,
    const TQ* __restrict__ k_global,
    const TQ* __restrict__ v_global,
    const TG* __restrict__ g_global,
    const TB* __restrict__ beta_global,
    const float* __restrict__ checkpoints,
    const float* __restrict__ d_final_state,
    float* __restrict__ chunk_workspace,  // [B*H*num_chunks, chunk_ws_stride]
    float* __restrict__ ds_accum,         // [B, H, K*V] — each chunk atomicAdds dh_term1
    int chunk_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, float scale,
    bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int block_id = blockIdx.x;
    const int bh = block_id / num_chunks;
    const int chunk = block_id % num_chunks;
    const int b = bh / H;
    const int h = bh % H;
    const long kv = (long)Kdim * Vdim;
    const int Lp = kMaxC;
    const int v_tile = (Vdim >= 64) ? 64 : Vdim;
    const int k_tile = (Kdim >= 64) ? 64 : Kdim;
    const bool small_kv = (Kdim <= Lp && Vdim <= Lp);
    const int smem_v = small_kv ? Vdim : v_tile;

    const int cs = chunk * chunk_size;
    const int L = kHasTail ? min(chunk_size, Tlen - cs) : chunk_size;

    const float* cp_base = checkpoints + (long)bh * (num_chunks + 1) * kv;
    const float* h_in = cp_base + (long)chunk * kv;

    // Per-chunk workspace
    float* cws = chunk_workspace + (long)block_id * chunk_ws_stride;
    ChunkWorkspaceLayout cwl = make_chunk_ws<TQ>(Lp, Kdim, Vdim);
    float* M    = cws + cwl.M_off;
    float* W    = cws + cwl.W_off;
    float* VNEW = cws + cwl.VNEW_off;
    float* DU   = cws + cwl.DU_off;
    float* DW   = cws + cwl.DW_off;
    float* DQ   = cws + cwl.DQ_off;
    float* DK   = cws + cwl.DK_off;
    float* DG   = cws + cwl.DG_off;
    float* DB   = cws + cwl.DB_off;
    float* DHT1 = cws + cwl.DHT1_off;
    float* K_NORM = cws + cwl.KNORM_off;
    float* INVQ_WS = cws + cwl.INVQ_off;
    float* INVK_WS = cws + cwl.INVK_off;

    // Shared memory
    extern __shared__ char smem_raw[];
    float* scratch1  = (float*)smem_raw;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    smem_q    = smem_k + Lp * Kdim;
    TQ*    buf1      = smem_q + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * smem_v;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invq = smem_beta + Lp;
    float* smem_invk = smem_invq + Lp;
    float* smem_eg   = smem_invk + Lp;
    float* smem_e_last = smem_eg + Lp;

    // Tail chunks need explicit zero-fill for padded rows/cols.
    if constexpr (kHasTail) {
        const bool full_chunk = (L == chunk_size);
        if (!full_chunk) {
            for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
                smem_k[idx] = from_float<TQ>(0.0f);
                smem_q[idx] = from_float<TQ>(0.0f);
            }
            for (int idx = tid; idx < Lp * smem_v; idx += nthr)
                buf2[idx] = from_float<TQ>(0.0f);
            for (int idx = tid; idx < Lp * Kdim; idx += nthr)
                buf3[idx] = from_float<TQ>(0.0f);
            for (int idx = tid; idx < Lp * Lp; idx += nthr)
                buf1[idx] = from_float<TQ>(0.0f);
            if (tid < Lp) {
                smem_gcum[tid] = 0.0f;
                smem_beta[tid] = 0.0f;
                smem_eg[tid] = 0.0f;
                smem_e_last[tid] = 0.0f;
            }
            __syncthreads();
        }
    }

    // Load k, q (vectorized for bf16/fp16 when possible).
    if constexpr (std::is_same_v<TQ, nv_bfloat16>) {
        if ((Kdim & 1) == 0) {
            const int k2 = Kdim >> 1;
            auto* smem_k2 = reinterpret_cast<__nv_bfloat162*>(smem_k);
            auto* smem_q2 = reinterpret_cast<__nv_bfloat162*>(smem_q);
            const auto* k2_ptr = reinterpret_cast<const __nv_bfloat162*>(k_global);
            const auto* q2_ptr = reinterpret_cast<const __nv_bfloat162*>(q_global);
            for (int idx = tid; idx < L * k2; idx += nthr) {
                const int pos = idx / k2;
                const int kk2 = idx % k2;
                const long gi2 = (((long)b * Tlen + cs + pos) * H + h) * k2 + kk2;
                smem_k2[pos * k2 + kk2] = k2_ptr[gi2];
                smem_q2[pos * k2 + kk2] = q2_ptr[gi2];
            }
        } else {
            for (int idx = tid; idx < L * Kdim; idx += nthr) {
                const int pos = idx / Kdim;
                const int kk = idx % Kdim;
                const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
                smem_k[pos * Kdim + kk] = k_global[gi];
                smem_q[pos * Kdim + kk] = q_global[gi];
            }
        }
        if (small_kv) {
            if ((Vdim & 1) == 0) {
                const int v2 = Vdim >> 1;
                auto* smem_v2 = reinterpret_cast<__nv_bfloat162*>(buf2);
                const auto* v2_ptr = reinterpret_cast<const __nv_bfloat162*>(v_global);
                for (int idx = tid; idx < L * v2; idx += nthr) {
                    const int pos = idx / v2;
                    const int vv2 = idx % v2;
                    const long gi2 = (((long)b * Tlen + cs + pos) * H + h) * v2 + vv2;
                    smem_v2[pos * v2 + vv2] = v2_ptr[gi2];
                }
            } else {
                for (int idx = tid; idx < L * Vdim; idx += nthr) {
                    const int pos = idx / Vdim;
                    const int vv = idx % Vdim;
                    buf2[pos * Vdim + vv] =
                        v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
                }
            }
        }
    } else if constexpr (std::is_same_v<TQ, half>) {
        if ((Kdim & 1) == 0) {
            const int k2 = Kdim >> 1;
            auto* smem_k2 = reinterpret_cast<__half2*>(smem_k);
            auto* smem_q2 = reinterpret_cast<__half2*>(smem_q);
            const auto* k2_ptr = reinterpret_cast<const __half2*>(k_global);
            const auto* q2_ptr = reinterpret_cast<const __half2*>(q_global);
            for (int idx = tid; idx < L * k2; idx += nthr) {
                const int pos = idx / k2;
                const int kk2 = idx % k2;
                const long gi2 = (((long)b * Tlen + cs + pos) * H + h) * k2 + kk2;
                smem_k2[pos * k2 + kk2] = k2_ptr[gi2];
                smem_q2[pos * k2 + kk2] = q2_ptr[gi2];
            }
        } else {
            for (int idx = tid; idx < L * Kdim; idx += nthr) {
                const int pos = idx / Kdim;
                const int kk = idx % Kdim;
                const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
                smem_k[pos * Kdim + kk] = k_global[gi];
                smem_q[pos * Kdim + kk] = q_global[gi];
            }
        }
        if (small_kv) {
            if ((Vdim & 1) == 0) {
                const int v2 = Vdim >> 1;
                auto* smem_v2 = reinterpret_cast<__half2*>(buf2);
                const auto* v2_ptr = reinterpret_cast<const __half2*>(v_global);
                for (int idx = tid; idx < L * v2; idx += nthr) {
                    const int pos = idx / v2;
                    const int vv2 = idx % v2;
                    const long gi2 = (((long)b * Tlen + cs + pos) * H + h) * v2 + vv2;
                    smem_v2[pos * v2 + vv2] = v2_ptr[gi2];
                }
            } else {
                for (int idx = tid; idx < L * Vdim; idx += nthr) {
                    const int pos = idx / Vdim;
                    const int vv = idx % Vdim;
                    buf2[pos * Vdim + vv] =
                        v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
                }
            }
        }
    } else {
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim;
            const int kk = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k_global[gi];
            smem_q[pos * Kdim + kk] = q_global[gi];
        }
        if (small_kv) {
            for (int idx = tid; idx < L * Vdim; idx += nthr) {
                const int pos = idx / Vdim;
                const int vv = idx % Vdim;
                buf2[pos * Vdim + vv] =
                    v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
            }
        }
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < L; ++i) {
            const long gh = ((long)b * Tlen + cs + i) * H + h;
            acc += to_float(g_global[gh]);
            smem_gcum[i] = acc;
            smem_beta[i] = to_float(beta_global[gh]);
        }
    }
    __syncthreads();

    for (int pos = tid; pos < L; pos += nthr) {
        smem_eg[pos] = expf(smem_gcum[pos]);
    }
    __syncthreads();
    const float g_last_val = (L > 0) ? smem_gcum[L - 1] : 0.0f;
    const float eg_last_val = expf(g_last_val);
    for (int pos = tid; pos < L; pos += nthr) {
        smem_e_last[pos] = expf(g_last_val - smem_gcum[pos]);
    }
    __syncthreads();

    // L2 norms
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
            const int pos = idx / Kdim;
            smem_k[pos * Kdim + idx % Kdim] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_k[idx]) * smem_invk[pos]));
            smem_q[pos * Kdim + idx % Kdim] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_q[idx]) * smem_invq[pos]));
        }
        __syncthreads();
    }

    // Cache normalized k and q/k inverse norms for Phase 3 reuse.
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        K_NORM[idx] = to_float(smem_k[idx]);
    }
    for (int idx = tid; idx < Lp; idx += nthr) {
        INVQ_WS[idx] = smem_invq[idx];
        INVK_WS[idx] = smem_invk[idx];
    }
    __syncthreads();

    // ================================================================
    // Build M: A = beta*exp(g)*(k@k^T), solve M
    // ================================================================
    wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
    __syncthreads();
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        if (i < L && j <= i) {
            const float eg_j = smem_eg[j];
            const float exp_ij =
                (eg_j > 0.0f) ? (smem_eg[i] / eg_j) : expf(smem_gcum[i] - smem_gcum[j]);
            scratch1[idx] *= smem_beta[i] * exp_ij;
        } else {
            scratch1[idx] = 0.0f;
        }
    }
    __syncthreads();
    // Solve M → scratch2
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
    // Save M
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        M[idx] = scratch2[idx];
    __syncthreads();

    // M → bf16 in buf1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    __syncthreads();

    // ================================================================
    // Recompute W and VNEW
    // ================================================================
    if (small_kv) {
        // bv = beta*v → buf2
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) * smem_beta[i]));
        }
        __syncthreads();

        // u = M @ bv → scratch1[Lp×V]
        wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();
    } else {
        // bkg = beta*k*exp(g) → buf3
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * smem_eg[i] : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // w = M @ bkg (K tiled)
        for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
            wmma_nn<TQ>(buf1, Lp, buf3 + k0, Kdim, scratch1, k_tile, Lp, k_tile, Lp);
            __syncthreads();
            for (int idx = tid; idx < Lp * k_tile; idx += nthr) {
                const int i = idx / k_tile;
                const int kk = idx % k_tile;
                const int out_idx = i * Kdim + (k0 + kk);
                float wval = bf16_trunc<TQ>(scratch1[idx]);
                buf3[out_idx] = from_float<TQ>(wval);
                W[out_idx] = wval;
            }
            __syncthreads();
        }

        // Stash W in smem_q so buf3 can be used for h tiles.
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            smem_q[idx] = buf3[idx];
        __syncthreads();

        // Recompute u/wh/VNEW(pre-gate) in V tiles.
        for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
            const int vt = min(v_tile, Vdim - v0);

            // beta * v tile -> buf2[Lp×vt]
            for (int idx = tid; idx < Lp * vt; idx += nthr) {
                const int i = idx / vt;
                const int vv = idx % vt;
                float val = 0.0f;
                if (i < L) {
                    const long vi = (((long)b * Tlen + cs + i) * H + h) * Vdim + (v0 + vv);
                    val = bf16_trunc<TQ>(to_float(v_global[vi]) * smem_beta[i]);
                }
                buf2[idx] = from_float<TQ>(val);
            }
            __syncthreads();

            // u_tile = M @ (beta*v_tile) -> scratch2
            wmma_nn<TQ>(buf1, Lp, buf2, vt, scratch2, vt, Lp, vt, Lp);
            __syncthreads();

            // h_in tile -> buf3[K×vt]
            for (int idx = tid; idx < Kdim * vt; idx += nthr) {
                const int kk = idx / vt;
                const int vv = idx % vt;
                const long h_idx = static_cast<long>(kk) * Vdim + (v0 + vv);
                buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[h_idx]));
            }
            __syncthreads();

            // wh_tile = W @ h_tile (W in smem_q)
            wmma_nn<TQ>(smem_q, Kdim, buf3, vt, scratch1, vt, Lp, vt, Kdim);
            __syncthreads();

            for (int idx = tid; idx < Lp * vt; idx += nthr) {
                const int i = idx / vt;
                const int vv = idx % vt;
                const int out_idx = i * Vdim + (v0 + vv);
                const float vnew_pre = bf16_trunc<TQ>(scratch2[idx] - scratch1[idx]);
                VNEW[out_idx] = vnew_pre;
            }
            __syncthreads();
        }

        // Restore normalized q into smem_q (it was used as temporary W storage).
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int pos = idx / Kdim;
            const int kk = idx % Kdim;
            float qv = 0.0f;
            if (pos < L) {
                const long qi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
                qv = to_float(q_global[qi]);
                if (use_qk_l2norm_in_kernel) {
                    qv = bf16_trunc<TQ>(qv * smem_invq[pos]);
                }
            }
            smem_q[idx] = from_float<TQ>(qv);
        }
        __syncthreads();
    }

    if (small_kv) {
        // bkg = beta*k*exp(g) → buf3
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * smem_eg[i] : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // w = M @ bkg → scratch1[Lp×K]
        wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            float wval = bf16_trunc<TQ>(scratch1[idx]);
            buf3[idx] = from_float<TQ>(wval);
            W[idx] = wval;
        }
        __syncthreads();

        // h_bf16 → buf1[K×V]
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[idx]));
        for (int idx = Kdim * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();

        // wh = w @ h_bf16 → scratch1[Lp×V]
        wmma_nn<TQ>(buf3, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();

        // vnew_pre = u - wh
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            float vnew_pre = bf16_trunc<TQ>(to_float(buf2[idx]) - scratch1[idx]);
            buf2[idx] = from_float<TQ>(vnew_pre);
            VNEW[idx] = vnew_pre;
        }
        __syncthreads();
    }

    // ================================================================
    // Zero gradient accumulators
    // ================================================================
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        DQ[idx] = 0.0f; DK[idx] = 0.0f; DW[idx] = 0.0f;
    }
    for (int idx = tid; idx < Lp * Vdim; idx += nthr)
        DU[idx] = 0.0f;
    for (int idx = tid; idx < Lp; idx += nthr) {
        DG[idx] = 0.0f; DB[idx] = 0.0f;
    }
    for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
        DHT1[idx] = 0.0f;
    __syncthreads();

    // ================================================================
    // term1: q @ h contribution
    // ================================================================
    if (small_kv) {
        // Load d_out → buf2 (overwriting vnew bf16, VNEW saved to workspace)
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[idx] = d_out[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        __syncthreads();

        // DQ_term1 = d_out @ h_bf16^T → scratch1[Lp×K]
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            DQ[idx] += scale * smem_eg[i] * scratch1[i * Kdim + idx % Kdim];
        }
        __syncthreads();

        // DHT1 = (scale*exp(g)*q)^T @ d_out → [K×V]
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_q[idx]) * scale * smem_eg[i] : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();
        wmma_tn<TQ>(buf3, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            DHT1[idx] = scratch1[idx];
        __syncthreads();

        // DG term1: q@h per row, dot with d_out
        wmma_nn<TQ>(smem_q, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();
        for (int i = tid; i < L; i += nthr) {
            float dot_val = 0.0f;
            for (int vv = 0; vv < Vdim; ++vv)
                dot_val += scratch1[i * Vdim + vv] * to_float(buf2[i * Vdim + vv]);
            DG[i] += scale * smem_eg[i] * dot_val;
        }
        __syncthreads();
    } else {
        // DQ_term1 = d_out @ h_bf16^T, tiled on K and V.
        for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
            const int kt = min(k_tile, Kdim - k0);
            for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
                const int vt = min(v_tile, Vdim - v0);

                for (int idx = tid; idx < Lp * vt; idx += nthr) {
                    const int pos = idx / vt;
                    const int vv = idx % vt;
                    TQ val = from_float<TQ>(0.0f);
                    if (pos < L) {
                        const long do_idx = (((long)b * Tlen + cs + pos) * H + h) * Vdim + (v0 + vv);
                        val = d_out[do_idx];
                    }
                    buf2[idx] = val;
                }

                for (int idx = tid; idx < kt * vt; idx += nthr) {
                    const int kk = idx / vt;
                    const int vv = idx % vt;
                    const long h_idx = static_cast<long>(k0 + kk) * Vdim + (v0 + vv);
                    buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[h_idx]));
                }
                __syncthreads();

                wmma_nt<TQ>(buf2, vt, buf3, vt, scratch1, kt, Lp, kt, vt);
                __syncthreads();

                for (int idx = tid; idx < L * kt; idx += nthr) {
                    const int i = idx / kt;
                    const int kk = idx % kt;
                    const int out_idx = i * Kdim + (k0 + kk);
                    DQ[out_idx] += scale * smem_eg[i] * scratch1[idx];
                }
                __syncthreads();
            }
        }

        // DHT1 = (scale*exp(g)*q)^T @ d_out, tiled on K and V.
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_q[idx]) * scale * smem_eg[i] : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
            const int kt = min(k_tile, Kdim - k0);
            for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
                const int vt = min(v_tile, Vdim - v0);
                for (int idx = tid; idx < Lp * vt; idx += nthr) {
                    const int pos = idx / vt;
                    const int vv = idx % vt;
                    TQ val = from_float<TQ>(0.0f);
                    if (pos < L) {
                        const long do_idx = (((long)b * Tlen + cs + pos) * H + h) * Vdim + (v0 + vv);
                        val = d_out[do_idx];
                    }
                    buf2[idx] = val;
                }
                __syncthreads();

                wmma_tn<TQ>(buf3 + k0, Kdim, buf2, vt, scratch1, vt, kt, vt, Lp);
                __syncthreads();

                for (int idx = tid; idx < kt * vt; idx += nthr) {
                    const int kk = idx / vt;
                    const int vv = idx % vt;
                    const long out_idx = static_cast<long>(k0 + kk) * Vdim + (v0 + vv);
                    DHT1[out_idx] = scratch1[idx];
                }
                __syncthreads();
            }
        }

        // DG term1: q@h per row, dot with d_out; tile on V.
        for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
            const int vt = min(v_tile, Vdim - v0);
            for (int idx = tid; idx < Kdim * vt; idx += nthr) {
                const int kk = idx / vt;
                const int vv = idx % vt;
                const long h_idx = static_cast<long>(kk) * Vdim + (v0 + vv);
                buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[h_idx]));
            }
            __syncthreads();

            wmma_nn<TQ>(smem_q, Kdim, buf3, vt, scratch1, vt, Lp, vt, Kdim);
            __syncthreads();

            for (int idx = tid; idx < Lp * vt; idx += nthr) {
                const int pos = idx / vt;
                const int vv = idx % vt;
                TQ val = from_float<TQ>(0.0f);
                if (pos < L) {
                    const long do_idx = (((long)b * Tlen + cs + pos) * H + h) * Vdim + (v0 + vv);
                    val = d_out[do_idx];
                }
                buf2[idx] = val;
            }
            __syncthreads();

            for (int i = tid; i < L; i += nthr) {
                float dot_val = 0.0f;
                for (int vv = 0; vv < vt; ++vv)
                    dot_val += scratch1[i * vt + vv] * to_float(buf2[i * vt + vv]);
                DG[i] += scale * smem_eg[i] * dot_val;
            }
            __syncthreads();
        }
    }

    // ================================================================
    // term2: Batched intra-chunk attention gradient
    // ================================================================
    // S = q @ k^T
    wmma_nt<TQ>(smem_q, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
    __syncthreads();
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i) {
                const float eg_j = smem_eg[j];
                const float exp_ij =
                    (eg_j > 0.0f) ? (smem_eg[i] / eg_j) : expf(smem_gcum[i] - smem_gcum[j]);
                scratch1[idx] = bf16_trunc<TQ>(exp_ij * scratch1[idx]);
            }
            else
            scratch1[idx] = 0.0f;
    }
    __syncthreads();

    if (small_kv) {
        // Scale d_out
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float val = (i < L) ? to_float(buf2[idx]) * scale : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();
    }

    // Save S to scratch2 for later
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        scratch2[idx] = scratch1[idx];
    __syncthreads();

    if (small_kv) {
        // VNEW stores vnew_pre directly.
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int j = idx / Vdim;
            float vnp = (j < L) ? VNEW[j * Vdim + idx % Vdim] : 0.0f;
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(vnp));
        }
        __syncthreads();

        // grad_S = scale_d_out @ vnew_pre^T → scratch1[Lp×Lp]
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Lp, Lp, Lp, Vdim);
        __syncthreads();

        // DG from S gradient without atomics:
        // DG[i] += sum_{j<i}(grad_S[i,j] * S[i,j]) - sum_{r>i}(grad_S[r,i] * S[r,i])
        for (int i = tid; i < L; i += nthr) {
            float dg_pos = 0.0f;
            float dg_neg = 0.0f;
            for (int j = 0; j < i; ++j) {
                dg_pos += scratch1[i * Lp + j] * scratch2[i * Lp + j];
            }
            for (int r = i + 1; r < L; ++r) {
                dg_neg += scratch1[r * Lp + i] * scratch2[r * Lp + i];
            }
            DG[i] += dg_pos - dg_neg;
        }
        __syncthreads();

        // grad_S_masked = grad_S * exp(g[i]-g[j]) → buf1
        // (scratch1 still has grad_S from the WMMA above)
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            float val = 0.0f;
            if (i < L && j <= i) {
                const float eg_j = smem_eg[j];
                const float exp_ij =
                    (eg_j > 0.0f) ? (smem_eg[i] / eg_j) : expf(smem_gcum[i] - smem_gcum[j]);
                val = scratch1[idx] * exp_ij;
            }
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // DQ_term2 = grad_S_masked @ k
        wmma_nn<TQ>(buf1, Lp, smem_k, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DQ[idx] += scratch1[idx];
        __syncthreads();

        // DK_term2 = grad_S_masked^T @ q
        wmma_tn<TQ>(buf1, Lp, smem_q, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DK[idx] += scratch1[idx];
        __syncthreads();

        // DU_term2 (w.r.t. vnew_pre) = S^T @ scale_d_out
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            float val = 0.0f;
            if (i < L && j <= i && j < L)
                val = scratch2[idx];
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();
        wmma_tn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr)
            DU[idx] += scratch1[idx];
        __syncthreads();
    } else {
        // Stream grad_S over V tiles, accumulating DG/DQ/DK per tile.
        for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
            const int vt = min(v_tile, Vdim - v0);
            for (int idx = tid; idx < Lp * vt; idx += nthr) {
                const int j = idx / vt;
                const int vv = idx % vt;
                float vnp = (j < L) ? VNEW[j * Vdim + (v0 + vv)] : 0.0f;
                buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(vnp));
            }
            __syncthreads();

            for (int idx = tid; idx < Lp * vt; idx += nthr) {
                const int pos = idx / vt;
                const int vv = idx % vt;
                float val = 0.0f;
                if (pos < L) {
                    const long do_idx = (((long)b * Tlen + cs + pos) * H + h) * Vdim + (v0 + vv);
                    val = bf16_trunc<TQ>(to_float(d_out[do_idx]) * scale);
                }
                buf2[idx] = from_float<TQ>(val);
            }
            __syncthreads();

            wmma_nt<TQ>(buf2, vt, buf1, vt, scratch1, Lp, Lp, Lp, vt);
            __syncthreads();

            for (int i = tid; i < L; i += nthr) {
                float dg_pos = 0.0f;
                float dg_neg = 0.0f;
                for (int j = 0; j < i; ++j) {
                    dg_pos += scratch1[i * Lp + j] * scratch2[i * Lp + j];
                }
                for (int r = i + 1; r < L; ++r) {
                    dg_neg += scratch1[r * Lp + i] * scratch2[r * Lp + i];
                }
                DG[i] += dg_pos - dg_neg;
            }
            __syncthreads();

            for (int idx = tid; idx < Lp * Lp; idx += nthr) {
                const int i = idx / Lp, j = idx % Lp;
                float val = 0.0f;
                if (i < L && j <= i) {
                    const float eg_j = smem_eg[j];
                    const float exp_ij =
                        (eg_j > 0.0f) ? (smem_eg[i] / eg_j) : expf(smem_gcum[i] - smem_gcum[j]);
                    val = scratch1[idx] * exp_ij;
                }
                buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
            }
            __syncthreads();

            for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
                wmma_nn<TQ>(buf1, Lp, smem_k + k0, Kdim, scratch1, k_tile, Lp, k_tile, Lp);
                __syncthreads();
                for (int idx = tid; idx < L * k_tile; idx += nthr) {
                    const int i = idx / k_tile;
                    const int kk = idx % k_tile;
                    const int out_idx = i * Kdim + (k0 + kk);
                    DQ[out_idx] += scratch1[idx];
                }
                __syncthreads();
            }

            for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
                wmma_tn<TQ>(buf1, Lp, smem_q + k0, Kdim, scratch1, k_tile, Lp, k_tile, Lp);
                __syncthreads();
                for (int idx = tid; idx < L * k_tile; idx += nthr) {
                    const int i = idx / k_tile;
                    const int kk = idx % k_tile;
                    const int out_idx = i * Kdim + (k0 + kk);
                    DK[out_idx] += scratch1[idx];
                }
                __syncthreads();
            }
        }

        // DU_term2 (w.r.t. vnew_pre) = S^T @ scale_d_out, tiled on V.
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            float val = 0.0f;
            if (i < L && j <= i && j < L)
                val = scratch2[idx];
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
            const int vt = min(v_tile, Vdim - v0);
            for (int idx = tid; idx < Lp * vt; idx += nthr) {
                const int pos = idx / vt;
                const int vv = idx % vt;
                float val = 0.0f;
                if (pos < L) {
                    const long do_idx = (((long)b * Tlen + cs + pos) * H + h) * Vdim + (v0 + vv);
                    val = bf16_trunc<TQ>(to_float(d_out[do_idx]) * scale);
                }
                buf2[idx] = from_float<TQ>(val);
            }
            __syncthreads();

            wmma_tn<TQ>(buf1, Lp, buf2, vt, scratch1, vt, Lp, vt, Lp);
            __syncthreads();
            for (int idx = tid; idx < L * vt; idx += nthr) {
                const int i = idx / vt;
                const int vv = idx % vt;
                const int out_idx = i * Vdim + (v0 + vv);
                DU[out_idx] += scratch1[idx];
            }
            __syncthreads();
        }
    }

    // ================================================================
    // Precompute correction matrix C and correction_local for Phase 2
    // This allows Phase 2 to use a single matmul: ds = ds*eg + DHT1_corrected - C@ds
    // ================================================================
    float* C_mat = cws + cwl.C_off;

    // Store eg_last for Phase 2
    if (tid == 0) {
        float* eg_ptr = cws + cwl.EG_off;
        *eg_ptr = eg_last_val;
    }

    // W_gated → buf3[Lp×K]: W[i,k] * exp(g_last - g[i])
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        const int i = idx / Kdim;
        float val = (i < L) ? W[idx] * smem_e_last[i] : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    if (small_kv) {
        // C = W_gated^T @ k → scratch1[K×K]
        // buf3^T[K×Lp] @ smem_k[Lp×K] → scratch1[K×K]
        wmma_tn<TQ>(buf3, Kdim, smem_k, Kdim, scratch1, Kdim, Kdim, Kdim, Lp);
        __syncthreads();

        // Store C in workspace as fp32; Phase 2 converts to TQ for WMMA use.
        for (int idx = tid; idx < Kdim * Kdim; idx += nthr)
            C_mat[idx] = scratch1[idx];
        __syncthreads();
    } else {
        // Tile C over K to keep C-tile output within scratch1[64x64].
        for (int k_row0 = 0; k_row0 < Kdim; k_row0 += k_tile) {
            for (int k_col0 = 0; k_col0 < Kdim; k_col0 += k_tile) {
                wmma_tn<TQ>(
                    buf3 + k_row0, Kdim,
                    smem_k + k_col0, Kdim,
                    scratch1, k_tile,
                    k_tile, k_tile, Lp);
                __syncthreads();

                for (int idx = tid; idx < k_tile * k_tile; idx += nthr) {
                    const int kr = idx / k_tile;
                    const int kc = idx % k_tile;
                    const long c_idx = static_cast<long>(k_row0 + kr) * Kdim + (k_col0 + kc);
                    C_mat[c_idx] = scratch1[idx];
                }
                __syncthreads();
            }
        }
    }

    // correction_local = W^T @ d_pre_local → scratch1[K×V]
    // W is in workspace (float). Load W → buf3 bf16 (we have it from W_gated, need ungated W)
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        float val = (idx < L * Kdim) ? W[idx] : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    if (small_kv) {
        // d_pre_local is stored directly in DU.
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const float val = (idx < L * Vdim) ? DU[idx] : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // W^T @ d_pre_local → scratch1[K×V]
        wmma_tn<TQ>(buf3, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();

        // DHT1 -= correction_local
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            DHT1[idx] -= scratch1[idx];
        __syncthreads();
    } else {
        // Tile correction_local over K and V.
        for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
            const int kt = min(k_tile, Kdim - k0);
            for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
                const int vt = min(v_tile, Vdim - v0);
                for (int idx = tid; idx < Lp * vt; idx += nthr) {
                    const int i = idx / vt;
                    const int vv = idx % vt;
                    const float val = (i < L) ? DU[i * Vdim + (v0 + vv)] : 0.0f;
                    buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
                }
                __syncthreads();

                wmma_tn<TQ>(
                    buf3 + k0, Kdim,
                    buf2, vt,
                    scratch1, vt,
                    kt, vt, Lp);
                __syncthreads();

                for (int idx = tid; idx < kt * vt; idx += nthr) {
                    const int kk = idx / vt;
                    const int vv = idx % vt;
                    const long dht_idx = static_cast<long>(k0 + kk) * Vdim + (v0 + vv);
                    DHT1[dht_idx] -= scratch1[idx];
                }
                __syncthreads();
            }
        }
    }
}

} // namespace

template<typename TQ, typename TG, typename TB>
void launch_gdr_bwd_phase1(
    const TQ* d_out,
    const TQ* q, const TQ* k, const TQ* v,
    const TG* g, const TB* beta,
    const float* checkpoints,
    const float* d_final_state,
    float* chunk_workspace,
    float* ds_accum,
    int chunk_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, float scale,
    bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream)
{
    const bool has_tail = (Tlen % chunk_size) != 0;
    auto launch_once = [&](int launch_threads) -> cudaError_t {
        if (launch_threads < 64 || launch_threads > 1024 || (launch_threads % 32) != 0) {
            return cudaErrorInvalidConfiguration;
        }
        cudaError_t err = cudaSuccess;
        if (has_tail) {
            err = cudaFuncSetAttribute(
                gdr_bwd_phase1_wmma<TQ, TG, TB, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem));
            if (err != cudaSuccess) return err;
            gdr_bwd_phase1_wmma<TQ, TG, TB, true><<<B * H * num_chunks, launch_threads, smem, stream>>>(
                d_out, q, k, v, g, beta,
                checkpoints, d_final_state,
                chunk_workspace, ds_accum,
                chunk_ws_stride,
                Tlen, H, Kdim, Vdim, num_chunks, chunk_size, scale,
                use_qk_l2norm_in_kernel);
        } else {
            err = cudaFuncSetAttribute(
                gdr_bwd_phase1_wmma<TQ, TG, TB, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem));
            if (err != cudaSuccess) return err;
            gdr_bwd_phase1_wmma<TQ, TG, TB, false><<<B * H * num_chunks, launch_threads, smem, stream>>>(
                d_out, q, k, v, g, beta,
                checkpoints, d_final_state,
                chunk_workspace, ds_accum,
                chunk_ws_stride,
                Tlen, H, Kdim, Vdim, num_chunks, chunk_size, scale,
                use_qk_l2norm_in_kernel);
        }
        return cudaGetLastError();
    };

    std::vector<int> candidates;
    candidates.reserve(12);
    auto push_unique = [&](int t) {
        if (t < 64 || t > 1024 || (t % 32) != 0) return;
        for (int x : candidates) if (x == t) return;
        candidates.push_back(t);
    };
    push_unique(threads);
    push_unique(1024);
    push_unique(896);
    push_unique(768);
    push_unique(640);
    push_unique(512);
    push_unique(384);
    push_unique(256);
    push_unique(192);
    push_unique(128);
    push_unique(96);
    push_unique(64);

    cudaError_t last_err = cudaErrorInvalidConfiguration;
    for (int t : candidates) {
        last_err = launch_once(t);
        if (last_err == cudaSuccess) {
            return;
        }
        if (last_err != cudaErrorLaunchOutOfResources &&
            last_err != cudaErrorInvalidConfiguration) {
            CUDA_CHECK(last_err);
            return;
        }
    }
    CUDA_CHECK(last_err);
}

#define INSTANTIATE_PHASE1(TQ, TG, TB) \
    template void launch_gdr_bwd_phase1<TQ, TG, TB>( \
        const TQ*, const TQ*, const TQ*, const TQ*, const TG*, const TB*, \
        const float*, const float*, float*, float*, \
        int, int, int, int, int, int, int, int, float, bool, \
        int, std::size_t, cudaStream_t);

#define INSTANTIATE_PHASE1_ALL_GB(TQ) \
    INSTANTIATE_PHASE1(TQ, float, float) \
    INSTANTIATE_PHASE1(TQ, float, nv_bfloat16) \
    INSTANTIATE_PHASE1(TQ, float, half) \
    INSTANTIATE_PHASE1(TQ, nv_bfloat16, float) \
    INSTANTIATE_PHASE1(TQ, nv_bfloat16, nv_bfloat16) \
    INSTANTIATE_PHASE1(TQ, nv_bfloat16, half) \
    INSTANTIATE_PHASE1(TQ, half, float) \
    INSTANTIATE_PHASE1(TQ, half, nv_bfloat16) \
    INSTANTIATE_PHASE1(TQ, half, half)

INSTANTIATE_PHASE1_ALL_GB(nv_bfloat16)
INSTANTIATE_PHASE1_ALL_GB(half)

#undef INSTANTIATE_PHASE1_ALL_GB
#undef INSTANTIATE_PHASE1
