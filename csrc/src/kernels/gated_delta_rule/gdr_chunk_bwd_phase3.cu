// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_rule_v2.cuh"
#include "gdr_bwd_launchers.h"

namespace {

// ============================================================================
// Multi-kernel backward — Phase 3: chunk-parallel finalization
//
// Grid: (B*H*num_chunks, 1, 1)
// Computes dM, M^T@DU, M^T@DW, dA_grad, writes d_q/d_k/d_v/d_beta/d_g.
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_bwd_phase3_wmma(
    TQ* __restrict__ d_q,
    TQ* __restrict__ d_k,
    TQ* __restrict__ d_v,
    TG* __restrict__ d_g,
    TB* __restrict__ d_beta,
    const TQ* __restrict__ d_out_global,
    const TQ* __restrict__ q_global,
    const TQ* __restrict__ k_global,
    const TQ* __restrict__ v_global,
    const TG* __restrict__ g_global,
    const TB* __restrict__ beta_global,
    const float* __restrict__ checkpoints,
    const float* __restrict__ dh_storage,   // [B*H*num_chunks, K*V] per-chunk dh from Phase 2
    float* __restrict__ chunk_workspace,
    int chunk_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size,
    float scale,
    bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int block_id = blockIdx.x;
    const int bh = block_id / num_chunks;
    const int chunk = block_id % num_chunks;
    const int b = bh / H;
    const int h = bh % H;
    const int Lp = kMaxC;
    const bool small_kv = (Kdim <= Lp && Vdim <= Lp);
    const int v_tile = (Vdim > Lp) ? Lp : Vdim;
    const int k_tile = (Kdim > Lp) ? Lp : Kdim;

    const int cs = chunk * chunk_size;
    const int L = min(chunk_size, Tlen - cs);
    const bool tail_chunk = (L < chunk_size);

    ChunkWorkspaceLayout cwl = make_chunk_ws<TQ>(Lp, Kdim, Vdim);
    float* cws = chunk_workspace + (long)block_id * chunk_ws_stride;
    float* M_ws  = cws + cwl.M_off;
    float* VNEW  = cws + cwl.VNEW_off;
    float* DU    = cws + cwl.DU_off;
    float* DW    = cws + cwl.DW_off;
    float* DQ    = cws + cwl.DQ_off;
    float* DK    = cws + cwl.DK_off;
    float* DG    = cws + cwl.DG_off;
    float* DB    = cws + cwl.DB_off;

    // Per-chunk dh from Phase 2
    const long kv = (long)Kdim * Vdim;
    const float* dh_chunk = dh_storage + (long)block_id * kv;

    // Shared memory
    extern __shared__ char smem_raw[];
    float* scratch1  = (float*)smem_raw;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    buf1      = smem_k + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * Vdim;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invq = smem_beta + Lp;
    float* smem_invk = smem_invq + Lp;
    float* smem_eg   = smem_invk + Lp;
    float* smem_e_last = smem_eg + Lp;

    // Load k, q, scalars
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        smem_k[idx] = from_float<TQ>(0.0f);
    }
    if (tid < Lp) {
        smem_gcum[tid] = 0.0f;
        smem_beta[tid] = 0.0f;
        smem_eg[tid] = 0.0f;
        smem_e_last[tid] = 0.0f;
    }
    __syncthreads();

    for (int idx = tid; idx < L * Kdim; idx += nthr) {
        const int pos = idx / Kdim, kk = idx % Kdim;
        const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
        smem_k[pos * Kdim + kk] = k_global[gi];
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
    const float g_last_p3 = (L > 0) ? smem_gcum[L - 1] : 0.0f;
    const float eg_last_p3 = (L > 0) ? smem_eg[L - 1] : 0.0f;
    for (int pos = tid; pos < L; pos += nthr) {
        smem_e_last[pos] = expf(g_last_p3 - smem_gcum[pos]);
    }
    __syncthreads();

    // L2 norms
    for (int pos = tid; pos < L; pos += nthr) {
        if (use_qk_l2norm_in_kernel) {
            float qn2 = 0.0f, kn2 = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                const long qi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
                float qv = to_float(q_global[qi]);
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
        }
        __syncthreads();
    }

    // Checkpoint access for h_in
    const float* cp_base = checkpoints + (long)bh * (num_chunks + 1) * kv;
    const float* h_in = cp_base + (long)chunk * kv;
    // Reuse per-chunk DHT1 storage as temporary global staging for phase3.
    float* tmp_area = cws + cwl.DHT1_off;

    // Tail chunks are sensitive to padded WMMA numerics; recompute DG base terms
    // (term1 + intra-chunk term2) in scalar math to match reference behavior.
    if (tail_chunk) {
        for (int i = tid; i < L; i += nthr) {
            DG[i] = 0.0f;
        }
        __syncthreads();

        // term1: scale * exp(g[i]) * dot(d_out[i], q[i] @ bf16(h_in))
        for (int i = tid; i < L; i += nthr) {
            float dg_i = 0.0f;
            const long do_i_base = (((long)b * Tlen + cs + i) * H + h) * Vdim;
            for (int vv = 0; vv < Vdim; ++vv) {
                const float do_v = to_float(d_out_global[do_i_base + vv]) * scale;
                float qh = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const long qi_idx = (((long)b * Tlen + cs + i) * H + h) * Kdim + kk;
                    float qi = to_float(q_global[qi_idx]);
                    if (use_qk_l2norm_in_kernel) {
                        qi = bf16_trunc<TQ>(qi * smem_invq[i]);
                    }
                    const float hq = bf16_trunc<TQ>(h_in[kk * Vdim + vv]);
                    qh += qi * hq;
                }
                dg_i += do_v * smem_eg[i] * qh;
            }
            DG[i] = dg_i;
        }
        __syncthreads();

        // term2: DG[i] += sum_{j<i}(grad_S[i,j] * S[i,j]) - sum_{r>i}(grad_S[r,i] * S[r,i])
        for (int i = tid; i < L; i += nthr) {
            float dg_pos = 0.0f;
            float dg_neg = 0.0f;

            for (int j = 0; j < i; ++j) {
                const float eg_j = smem_eg[j];
                const float exp_ij =
                    (eg_j > 0.0f) ? (smem_eg[i] / eg_j) : expf(smem_gcum[i] - smem_gcum[j]);
                float dot_qk = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const long qi_idx = (((long)b * Tlen + cs + i) * H + h) * Kdim + kk;
                    float qi = to_float(q_global[qi_idx]);
                    if (use_qk_l2norm_in_kernel) {
                        qi = bf16_trunc<TQ>(qi * smem_invq[i]);
                    }
                    dot_qk += qi * to_float(smem_k[j * Kdim + kk]);
                }
                const float s_ij = bf16_trunc<TQ>(exp_ij * dot_qk);

                float grad_s = 0.0f;
                const long do_i_base = (((long)b * Tlen + cs + i) * H + h) * Vdim;
                for (int vv = 0; vv < Vdim; ++vv) {
                    const float do_v = to_float(d_out_global[do_i_base + vv]) * scale;
                    const float v_new_pre = VNEW[j * Vdim + vv];
                    grad_s += do_v * v_new_pre;
                }
                dg_pos += grad_s * s_ij;
            }

            for (int r = i + 1; r < L; ++r) {
                const float eg_i = smem_eg[i];
                const float exp_ri =
                    (eg_i > 0.0f) ? (smem_eg[r] / eg_i) : expf(smem_gcum[r] - smem_gcum[i]);
                float dot_qk = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const long qr_idx = (((long)b * Tlen + cs + r) * H + h) * Kdim + kk;
                    float qr = to_float(q_global[qr_idx]);
                    if (use_qk_l2norm_in_kernel) {
                        qr = bf16_trunc<TQ>(qr * smem_invq[r]);
                    }
                    dot_qk += qr * to_float(smem_k[i * Kdim + kk]);
                }
                const float s_ri = bf16_trunc<TQ>(exp_ri * dot_qk);

                float grad_s = 0.0f;
                const long do_r_base = (((long)b * Tlen + cs + r) * H + h) * Vdim;
                for (int vv = 0; vv < Vdim; ++vv) {
                    const float do_v = to_float(d_out_global[do_r_base + vv]) * scale;
                    const float v_new_pre = VNEW[i * Vdim + vv];
                    grad_s += do_v * v_new_pre;
                }
                dg_neg += grad_s * s_ri;
            }

            DG[i] += dg_pos - dg_neg;
        }
        __syncthreads();
    }

    // ================================================================
    // Step 0-pre: k@ds → add to DU, apply gating, DG contributions
    // (moved from Phase 2 to enable single-matmul Phase 2)
    // ================================================================
    // <=64 fast path keeps the original full KxV staging in buf1.
    if (small_kv) {
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (long idx = tid; idx < kv; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(dh_chunk[idx]));
        __syncthreads();
    }

    float dg_last_local = 0.0f;
    if (!tail_chunk) {
        if (small_kv) {
            // k @ ds → scratch1[Lp×V]
            wmma_nn<TQ>(smem_k, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
            __syncthreads();

            // DU += k@ds, then apply gating: DU → d_pre
            for (int idx = tid; idx < L * Vdim; idx += nthr) {
                const int i = idx / Vdim;
                const float e_i = smem_e_last[i];
                const float kds_val = scratch1[idx];
                const float d_pre = DU[idx] + kds_val * e_i;

                DU[idx] = d_pre;  // overwrite DU with d_pre
            }
            __syncthreads();

            // DG gating contribution:
            //   DG[i]     -= Σ_v (k@ds)[i,v] * VNEW[i,v]
            //   DG[L - 1] += Σ_i,v (k@ds)[i,v] * VNEW[i,v]
            for (int i = tid; i < L; i += nthr) {
                const float e_i = smem_e_last[i];
                float row_sum = 0.0f;
                for (int vv = 0; vv < Vdim; ++vv) {
                    row_sum += scratch1[i * Vdim + vv] * (VNEW[i * Vdim + vv] * e_i);
                }
                DG[i] -= row_sum;
                dg_last_local += row_sum;
            }
        } else {
            // Tile-safe path for K=V=128: keep outputs at [Lp×64].
            for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
                const int vt = min(v_tile, Vdim - v0);
                for (int idx = tid; idx < Kdim * vt; idx += nthr) {
                    const int kk = idx / vt;
                    const int vv = idx % vt;
                    const long dh_idx = (long)kk * Vdim + (v0 + vv);
                    buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(dh_chunk[dh_idx]));
                }
                __syncthreads();

                wmma_nn<TQ>(smem_k, Kdim, buf3, vt, scratch1, vt, Lp, vt, Kdim);
                __syncthreads();

                for (int i = tid; i < L; i += nthr) {
                    const float e_i = smem_e_last[i];
                    float row_sum = 0.0f;
                    for (int vv = 0; vv < vt; ++vv) {
                        const int v_col = v0 + vv;
                        const int idx = i * Vdim + v_col;
                        const float kds_val = scratch1[i * vt + vv];
                        DU[idx] = DU[idx] + kds_val * e_i;
                        row_sum += kds_val * (VNEW[idx] * e_i);
                    }
                    DG[i] -= row_sum;
                    dg_last_local += row_sum;
                }
                __syncthreads();
            }
        }
    } else {
        // Tail chunks: compute k@ds in fp32 scalar math to match reference numerics.
        for (int i = tid; i < L; i += nthr) {
            float row_sum = 0.0f;
            const float e_i = smem_e_last[i];
            for (int vv = 0; vv < Vdim; ++vv) {
                float kds_val = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    kds_val += dh_chunk[(long)kk * Vdim + vv] * to_float(smem_k[i * Kdim + kk]);
                }
                const int idx = i * Vdim + vv;
                const float d_pre = DU[idx] + kds_val * e_i;
                DU[idx] = d_pre;
                row_sum += kds_val * (VNEW[idx] * e_i);
            }
            DG[i] -= row_sum;
            dg_last_local += row_sum;
        }
    }

    const float dg_last_warp = warp_reduce_sum_f32(dg_last_local);
    if ((tid & 31) == 0) {
        scratch1[tid / 32] = dg_last_warp;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sum = 0.0f;
        for (int w = 0; w < nthr / 32; ++w) {
            block_sum += scratch1[w];
        }
        DG[L - 1] += block_sum;
    }
    __syncthreads();

    // DG[L-1] += eg_last * Σ(ds * h_in)
    float ds_h_local = 0.0f;
    for (long idx = tid; idx < kv; idx += nthr) {
        ds_h_local += dh_chunk[idx] * h_in[idx];
    }
    const float ds_h_warp = warp_reduce_sum_f32(ds_h_local);
    if ((tid & 31) == 0) {
        scratch1[tid / 32] = ds_h_warp;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sum = 0.0f;
        for (int w = 0; w < nthr / 32; ++w) {
            block_sum += scratch1[w];
        }
        DG[L - 1] += block_sum * eg_last_p3;
    }
    __syncthreads();

    // ================================================================
    // Step 0a: (VNEW * exp(g_last-g_i)) @ dh^T → DK
    // ================================================================
    // Load VNEW → buf2[Lp×V]
    // buf1 keeps full dh_chunk only on <=64 path; tiled path streams dh through buf3.
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        const int i = idx / Vdim;
        const float e_i = (i < L) ? smem_e_last[i] : 0.0f;
        float val = (idx < L * Vdim) ? (VNEW[idx] * e_i) : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    if (small_kv) {
        // VNEW @ dh^T → scratch1[Lp×K]
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DK[idx] += scratch1[idx];
        __syncthreads();
    } else {
        for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
            const int kt = min(k_tile, Kdim - k0);
            for (int idx = tid; idx < kt * Vdim; idx += nthr) {
                const int kk = idx / Vdim;
                const int vv = idx % Vdim;
                const long dh_idx = (long)(k0 + kk) * Vdim + vv;
                buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(dh_chunk[dh_idx]));
            }
            __syncthreads();

            wmma_nt<TQ>(buf2, Vdim, buf3, Vdim, scratch1, kt, Lp, kt, Vdim);
            __syncthreads();
            for (int idx = tid; idx < L * kt; idx += nthr) {
                const int i = idx / kt;
                const int kk = idx % kt;
                DK[i * Kdim + (k0 + kk)] += scratch1[idx];
            }
            __syncthreads();
        }
    }

    // ================================================================
    // Step 0b: DW = -d_pre @ h_in^T
    // ================================================================
    // d_pre is now in DU[L×V] (computed in Step 0-pre)
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        float val = (idx < L * Vdim) ? DU[idx] : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    if (small_kv) {
        // Load h_in → buf1[K×V] (zero-pad to Lp×Lp)
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (long idx = tid; idx < kv; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[idx]));
        __syncthreads();

        // d_pre @ h_in^T → scratch1[Lp×K]
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DW[idx] = -scratch1[idx];
        __syncthreads();
    } else {
        for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
            const int kt = min(k_tile, Kdim - k0);
            for (int idx = tid; idx < kt * Vdim; idx += nthr) {
                const int kk = idx / Vdim;
                const int vv = idx % Vdim;
                const long h_idx = (long)(k0 + kk) * Vdim + vv;
                buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[h_idx]));
            }
            __syncthreads();

            wmma_nt<TQ>(buf2, Vdim, buf3, Vdim, scratch1, kt, Lp, kt, Vdim);
            __syncthreads();
            for (int idx = tid; idx < L * kt; idx += nthr) {
                const int i = idx / kt;
                const int kk = idx % kt;
                DW[i * Kdim + (k0 + kk)] = -scratch1[idx];
            }
            __syncthreads();
        }
    }

    // Load active causal block of M and zero-pad the rest.
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp;
        const int j = idx % Lp;
        scratch2[idx] = (i < L && j <= i) ? M_ws[idx] : 0.0f;
    }
    __syncthreads();

    // ================================================================
    // Step 1: M^T @ DU → d_v, DB
    // ================================================================
    // M_bf16 → buf1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    // DU_bf16 → buf2[Lp×V]
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        float val = (idx < L * Vdim) ? DU[idx] : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    if (small_kv) {
        // MT_DU → scratch1[Lp×V]
        wmma_tn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();

        // Load raw v → buf2 for DB (and later reuse for beta*V in Step 3)
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[idx] = (pos < L) ?
                v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv] :
                from_float<TQ>(0.0f);
        }
        __syncthreads();

        for (int j = tid; j < L; j += nthr) {
            const long v_j_base = (((long)b * Tlen + cs + j) * H + h) * Vdim;
            float db_acc = 0.0f;
            for (int vv = 0; vv < Vdim; ++vv) {
                float mt_du = scratch1[j * Vdim + vv];
                d_v[v_j_base + vv] = from_float<TQ>(mt_du * smem_beta[j]);
                db_acc += mt_du * to_float(buf2[j * Vdim + vv]);
            }
            DB[j] += db_acc;
        }
        __syncthreads();
    } else {
        for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
            const int vt = min(v_tile, Vdim - v0);
            wmma_tn<TQ>(buf1, Lp, buf2 + v0, Vdim, scratch1, vt, Lp, vt, Lp);
            __syncthreads();

            for (int j = tid; j < L; j += nthr) {
                const long v_j_base = (((long)b * Tlen + cs + j) * H + h) * Vdim;
                float db_acc = 0.0f;
                for (int vv = 0; vv < vt; ++vv) {
                    const int v_col = v0 + vv;
                    const float mt_du = scratch1[j * vt + vv];
                    const long v_j_idx = v_j_base + v_col;
                    d_v[v_j_idx] = from_float<TQ>(mt_du * smem_beta[j]);
                    db_acc += mt_du * to_float(v_global[v_j_idx]);
                }
                DB[j] += db_acc;
            }
            __syncthreads();
        }

        // Keep raw-v cached in buf2 for Step 3 (matches small-kv behavior).
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[idx] = (pos < L) ?
                v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv] :
                from_float<TQ>(0.0f);
        }
        __syncthreads();
    }

    // ================================================================
    // Step 2: M^T @ DW → DK, DB, DG
    // ================================================================
    // DW_bf16 → buf3
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        float val = (idx < L * Kdim) ? DW[idx] : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    if (small_kv) {
        // MT_DW → scratch1[Lp×K]
        wmma_tn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();

        for (int j = tid; j < L; j += nthr) {
            const float egj = smem_eg[j];
            float db_acc = 0.0f, dg_acc = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float mt_dw = scratch1[j * Kdim + kk];
                float kj = to_float(smem_k[j * Kdim + kk]);
                DK[j * Kdim + kk] += mt_dw * smem_beta[j] * egj;
                db_acc += mt_dw * egj * kj;
                dg_acc += mt_dw * smem_beta[j] * egj * kj;
            }
            DB[j] += db_acc;
            DG[j] += dg_acc;
        }
        __syncthreads();
    } else {
        for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
            const int kt = min(k_tile, Kdim - k0);
            wmma_tn<TQ>(buf1, Lp, buf3 + k0, Kdim, scratch1, kt, Lp, kt, Lp);
            __syncthreads();

            for (int j = tid; j < L; j += nthr) {
                const float egj = smem_eg[j];
                float db_acc = 0.0f, dg_acc = 0.0f;
                for (int kk = 0; kk < kt; ++kk) {
                    const int k_col = k0 + kk;
                    const float mt_dw = scratch1[j * kt + kk];
                    const float kj = to_float(smem_k[j * Kdim + k_col]);
                    DK[j * Kdim + k_col] += mt_dw * smem_beta[j] * egj;
                    db_acc += mt_dw * egj * kj;
                    dg_acc += mt_dw * smem_beta[j] * egj * kj;
                }
                DB[j] += db_acc;
                DG[j] += dg_acc;
            }
            __syncthreads();
        }
    }

    // ================================================================
    // Step 3: dM = DU @ (bv)^T + DW @ bkg^T
    // ================================================================
    // beta*V → buf2[Lp×V] (reuse raw-v values already in buf2)
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        const int pos = idx / Vdim;
        float val = 0.0f;
        if (pos < L) {
            float vraw = to_float(buf2[idx]);
            val = bf16_trunc<TQ>(vraw * smem_beta[pos]);
        }
        buf2[idx] = from_float<TQ>(val);
    }
    if (small_kv) {
        // DU_bf16 → buf1
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(DU[idx]));
        __syncthreads();

        // dM_part1 = DU @ (bv)^T → scratch1
        wmma_nt<TQ>(buf1, Vdim, buf2, Vdim, scratch1, Lp, Lp, Lp, Vdim);
        __syncthreads();

        // Save dM_part1
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            tmp_area[idx] = scratch1[idx];
    } else {
        // Tile V to keep DU staging in 64x64 buf1 while accumulating full dM_part1.
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            tmp_area[idx] = 0.0f;
        __syncthreads();

        for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
            const int vt = min(v_tile, Vdim - v0);
            for (int idx = tid; idx < Lp * vt; idx += nthr) {
                const int i = idx / vt;
                const int vv = idx % vt;
                const float val = (i < L) ? DU[i * Vdim + (v0 + vv)] : 0.0f;
                buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
            }
            __syncthreads();

            wmma_nt<TQ>(buf1, vt, buf2 + v0, Vdim, scratch1, Lp, Lp, Lp, vt);
            __syncthreads();

            for (int idx = tid; idx < Lp * Lp; idx += nthr)
                tmp_area[idx] += scratch1[idx];
            __syncthreads();
        }
    }

    // bkg → buf2[Lp×K]
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        const int pos = idx / Kdim;
        float val = (pos < L) ? to_float(smem_k[idx]) * smem_beta[pos] * smem_eg[pos] : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // dM_part2 = DW @ bkg^T
    // buf3 still has DW_bf16 from step 2
    wmma_nt<TQ>(buf3, Kdim, buf2, Kdim, scratch1, Lp, Lp, Lp, Kdim);
    __syncthreads();

    // dM = part1 + part2, masked
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        if (i < L && j <= i)
            scratch1[idx] = tmp_area[idx] + scratch1[idx];
        else
            scratch1[idx] = 0.0f;
    }
    __syncthreads();

    // ================================================================
    // Step 4: dA_grad = -M^T @ dM @ M^T  (WMMA)
    // ================================================================
    // dM is in scratch1[Lp×Lp], M is in scratch2[Lp×Lp]
    // Convert both to bf16
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    __syncthreads();

    // tmp1 = dM @ M^T → scratch1 (reuse since dM is now in buf1)
    wmma_nt<TQ>(buf1, Lp, buf2, Lp, scratch1, Lp, Lp, Lp, Lp);
    __syncthreads();

    // Convert tmp1 to bf16 → buf1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
    __syncthreads();

    // dA_grad = -M^T @ tmp1 → scratch1
    wmma_tn<TQ>(buf2, Lp, buf1, Lp, scratch1, Lp, Lp, Lp, Lp);
    __syncthreads();

    // Negate
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        scratch1[idx] = -scratch1[idx];
    __syncthreads();

    // dA_grad → DK, DB, DG
    // Precompute kkt = k @ k^T via WMMA → scratch2[Lp×Lp]
    // dA_grad is in scratch1[Lp×Lp]
    wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch2, Lp, Lp, Lp, Kdim);
    __syncthreads();

    // Save kkt to global tmp_area, then reuse scratch2 for ddot_sym
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        tmp_area[idx] = scratch2[idx];
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        scratch2[idx] = 0.0f;
    __syncthreads();

    // Process lower triangle:
    //   - form ddot_sym for DK matmul in scratch2
    //   - stash DB pair contributions in scratch1 lower triangle
    //   - stash DG pair contributions in tmp_area lower triangle
    for (int idx = tid; idx < L * L; idx += nthr) {
        const int i = idx / L, j = idx % L;
        if (j >= i) continue;

        const float eg_j = smem_eg[j];
        const float exp_ij =
            (eg_j > 0.0f) ? (smem_eg[i] / eg_j) : expf(smem_gcum[i] - smem_gcum[j]);
        const float a_grad = scratch1[i * Lp + j];
        const float dot_k = tmp_area[i * Lp + j];
        const float val = dot_k * exp_ij;

        const float db_pair = a_grad * val;
        const float dval = a_grad * smem_beta[i];
        const float dg_pair = dval * dot_k * exp_ij;
        const float ddot = dval * exp_ij;

        scratch1[i * Lp + j] = db_pair;
        tmp_area[i * Lp + j] = dg_pair;

        // Store symmetric ddot for DK matmul
        scratch2[i * Lp + j] = ddot;
        scratch2[j * Lp + i] = ddot;
    }
    __syncthreads();

    // Reduce pairwise contributions into DB and DG without atomics.
    for (int i = tid; i < L; i += nthr) {
        float db_acc = 0.0f;
        float dg_acc = 0.0f;
        for (int j = 0; j < i; ++j) {
            db_acc += scratch1[i * Lp + j];
            dg_acc += tmp_area[i * Lp + j];
        }
        for (int j = i + 1; j < L; ++j) {
            dg_acc -= tmp_area[j * Lp + i];
        }
        DB[i] += db_acc;
        DG[i] += dg_acc;
    }
    __syncthreads();

    // DK += ddot_sym @ k  (WMMA matmul)
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    __syncthreads();

    if (small_kv) {
        wmma_nn<TQ>(buf1, Lp, smem_k, Kdim, scratch2, Kdim, Lp, Kdim, Lp);
        __syncthreads();

        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DK[idx] += scratch2[idx];
        __syncthreads();
    } else {
        for (int k0 = 0; k0 < Kdim; k0 += k_tile) {
            const int kt = min(k_tile, Kdim - k0);
            wmma_nn<TQ>(buf1, Lp, smem_k + k0, Kdim, scratch2, kt, Lp, kt, Lp);
            __syncthreads();

            for (int idx = tid; idx < L * kt; idx += nthr) {
                const int i = idx / kt;
                const int kk = idx % kt;
                DK[i * Kdim + (k0 + kk)] += scratch2[idx];
            }
            __syncthreads();
        }
    }

    // ================================================================
    // Step 5: Write d_q, d_k (with L2norm backward), d_beta, d_g
    // ================================================================
    for (int i = tid; i < L; i += nthr) {
        const long q_i_base = (((long)b * Tlen + cs + i) * H + h) * Kdim;
        const long gh_idx = ((long)b * Tlen + cs + i) * H + h;

        if (use_qk_l2norm_in_kernel) {
            float dot_q = 0.0f, dot_k = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                const long qi_idx = (((long)b * Tlen + cs + i) * H + h) * Kdim + kk;
                float qn = to_float(q_global[qi_idx]);
                qn = bf16_trunc<TQ>(qn * smem_invq[i]);
                dot_q += DQ[i * Kdim + kk] * qn;
                dot_k += DK[i * Kdim + kk] * to_float(smem_k[i * Kdim + kk]);
            }
            for (int kk = 0; kk < Kdim; ++kk) {
                const long qi_idx = (((long)b * Tlen + cs + i) * H + h) * Kdim + kk;
                float qn = to_float(q_global[qi_idx]);
                qn = bf16_trunc<TQ>(qn * smem_invq[i]);
                const float kn = to_float(smem_k[i * Kdim + kk]);
                d_q[q_i_base + kk] = from_float<TQ>(
                    (DQ[i * Kdim + kk] - qn * dot_q) * smem_invq[i]);
                d_k[q_i_base + kk] = from_float<TQ>(
                    (DK[i * Kdim + kk] - kn * dot_k) * smem_invk[i]);
            }
        } else {
            for (int kk = 0; kk < Kdim; ++kk) {
                d_q[q_i_base + kk] = from_float<TQ>(DQ[i * Kdim + kk]);
                d_k[q_i_base + kk] = from_float<TQ>(DK[i * Kdim + kk]);
            }
        }
        d_beta[gh_idx] = from_float<TB>(DB[i]);
    }
    __syncthreads();

    // Reverse cumsum for d_g
    if (tid == 0) {
        float running = 0.0f;
        for (int i = L - 1; i >= 0; --i) {
            running += DG[i];
            const long gh_idx = ((long)b * Tlen + cs + i) * H + h;
            d_g[gh_idx] = from_float<TG>(running);
        }
    }
}


} // namespace

template<typename TQ, typename TG, typename TB>
void launch_gdr_bwd_phase3(
    TQ* d_q, TQ* d_k, TQ* d_v,
    TG* d_g, TB* d_beta,
    const TQ* d_out,
    const TQ* q, const TQ* k, const TQ* v,
    const TG* g, const TB* beta,
    const float* checkpoints,
    const float* dh_storage,
    float* chunk_workspace,
    int chunk_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, float scale,
    bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream)
{
    CUDA_CHECK(cudaFuncSetAttribute(
        gdr_bwd_phase3_wmma<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    gdr_bwd_phase3_wmma<TQ, TG, TB><<<B * H * num_chunks, threads, smem, stream>>>(
        d_q, d_k, d_v, d_g, d_beta,
        d_out, q, k, v, g, beta,
        checkpoints, dh_storage,
        chunk_workspace, chunk_ws_stride,
        Tlen, H, Kdim, Vdim, num_chunks, chunk_size, scale,
        use_qk_l2norm_in_kernel);
    CUDA_CHECK(cudaGetLastError());
}

#define INSTANTIATE_PHASE3(TQ, TG, TB) \
    template void launch_gdr_bwd_phase3<TQ, TG, TB>( \
        TQ*, TQ*, TQ*, TG*, TB*, \
        const TQ*, const TQ*, const TQ*, const TQ*, const TG*, const TB*, \
        const float*, const float*, float*, \
        int, int, int, int, int, int, int, int, float, bool, \
        int, std::size_t, cudaStream_t);

#define INSTANTIATE_PHASE3_ALL_GB(TQ) \
    INSTANTIATE_PHASE3(TQ, float, float) \
    INSTANTIATE_PHASE3(TQ, float, nv_bfloat16) \
    INSTANTIATE_PHASE3(TQ, float, half) \
    INSTANTIATE_PHASE3(TQ, nv_bfloat16, float) \
    INSTANTIATE_PHASE3(TQ, nv_bfloat16, nv_bfloat16) \
    INSTANTIATE_PHASE3(TQ, nv_bfloat16, half) \
    INSTANTIATE_PHASE3(TQ, half, float) \
    INSTANTIATE_PHASE3(TQ, half, nv_bfloat16) \
    INSTANTIATE_PHASE3(TQ, half, half)

INSTANTIATE_PHASE3_ALL_GB(nv_bfloat16)
INSTANTIATE_PHASE3_ALL_GB(half)

#undef INSTANTIATE_PHASE3_ALL_GB
#undef INSTANTIATE_PHASE3
