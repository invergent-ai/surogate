// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_rule_v2.cuh"
#include "gdr_bwd_launchers.h"

namespace {

// ============================================================================
// Backward kernel — WMMA tensor-core accelerated
//
// Uses checkpoints (from checkpoint kernel) and a global workspace.
// WMMA accelerates: A=k@k^T, M@bkg, M@bv, W@h, and the key optimization:
//   Batched attention gradient via S=q@k^T, dS=d_out@vnew^T, WMMA matmuls
//   for DQ/DK/DU from attention gradient.
// Shared memory: scratch1[C×C]f + scratch2[C×C]f
//   + k[C×K]TQ + q[C×K]TQ + buf1[C×C]TQ + buf2[C×V]TQ + buf3[C×K]TQ
//   + scalars[4×C]f
// ============================================================================
template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_bwd_wmma(
    TQ* __restrict__ d_q,
    TQ* __restrict__ d_k,
    TQ* __restrict__ d_v,
    TG* __restrict__ d_g,
    TB* __restrict__ d_beta,
    float* __restrict__ d_initial_state,
    const TQ* __restrict__ d_out,
    const float* __restrict__ d_final_state,
    const TQ* __restrict__ q_global,
    const TQ* __restrict__ k_global,
    const TQ* __restrict__ v_global,
    const TG* __restrict__ g_global,
    const TB* __restrict__ beta_global,
    const float* __restrict__ checkpoints,
    float* __restrict__ workspace,
    int workspace_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, float scale, bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = (long)Kdim * Vdim;
    const long bh = (long)b * H + h;
    const float* cp_base = checkpoints + bh * (long)(num_chunks + 1) * kv;
    float* ds = d_initial_state + bh * kv;
    const int Lp = kMaxC;

    // Shared memory layout
    extern __shared__ char smem_raw[];
    float* scratch1  = (float*)smem_raw;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    smem_q    = smem_k + Lp * Kdim;
    TQ*    buf1      = smem_q + Lp * Kdim;       // [C×C] TQ
    TQ*    buf2      = buf1 + Lp * Lp;            // [C×V] TQ
    TQ*    buf3      = buf2 + Lp * Vdim;          // [C×K] TQ
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invq = smem_beta + Lp;
    float* smem_invk = smem_invq + Lp;
    float* smem_eg   = smem_invk + Lp;
    float* smem_e_last = smem_eg + Lp;

    // Global workspace per (b,h)
    float* ws = workspace + bh * workspace_stride;
    const int ws_W    = 0;
    const int ws_VNEW = ws_W    + chunk_size * Kdim;
    const int ws_DU   = ws_VNEW + chunk_size * Vdim;
    const int ws_DW   = ws_DU   + chunk_size * Vdim;
    const int ws_DQ   = ws_DW   + chunk_size * Kdim;
    const int ws_DK   = ws_DQ   + chunk_size * Kdim;
    const int ws_DG   = ws_DK   + chunk_size * Kdim;
    const int ws_DB   = ws_DG   + chunk_size;
    float* W    = ws + ws_W;
    float* VNEW = ws + ws_VNEW;
    float* DU   = ws + ws_DU;
    float* DW   = ws + ws_DW;
    float* DQ   = ws + ws_DQ;
    float* DK   = ws + ws_DK;
    float* DG   = ws + ws_DG;
    float* DB   = ws + ws_DB;

    // Seed ds
    for (long idx = tid; idx < kv; idx += nthr)
        ds[idx] = d_final_state ? d_final_state[bh * kv + idx] : 0.0f;
    __syncthreads();

    for (int chunk = num_chunks - 1; chunk >= 0; --chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);
        const float* h_in = cp_base + (long)chunk * kv;
        float* dh_in = const_cast<float*>(cp_base + (long)(chunk + 1) * kv);

        // Zero-fill
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

        // Load k, q, v
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim, kk = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k_global[gi];
            smem_q[pos * Kdim + kk] = q_global[gi];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[pos * Vdim + vv] = v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
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

        // ================================================================
        // Build M: A = beta*exp(g)*(k@k^T), solve M
        // ================================================================
        wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] *= smem_beta[i] * expf(smem_gcum[i] - smem_gcum[j]);
            else
                scratch1[idx] = 0.0f;
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

        // M → bf16 in buf1
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
        __syncthreads();

        // ================================================================
        // Recompute W and VNEW using WMMA
        // ================================================================

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

        // bkg = beta*k*exp(g) → buf3
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * expf(smem_gcum[i]) : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // w = M @ bkg → scratch1[Lp×K]
        wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            float wval = bf16_trunc<TQ>(scratch1[idx]);
            buf3[idx] = from_float<TQ>(wval);
            if (idx < L * Kdim) W[idx] = wval;
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

        // vnew_pre = u - wh, then vnew = vnew_pre * exp(g_last - g_cum[i])
        const float g_last_val = (L > 0) ? smem_gcum[L - 1] : 0.0f;
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float vnew_pre = bf16_trunc<TQ>(to_float(buf2[idx]) - scratch1[idx]);
            float e_i = (i < L) ? expf(g_last_val - smem_gcum[i]) : 0.0f;
            float vnew = bf16_trunc<TQ>(vnew_pre * e_i);
            buf2[idx] = from_float<TQ>(vnew);
            if (idx < L * Vdim) VNEW[idx] = vnew;
        }
        __syncthreads();

        // ================================================================
        // dh_in = ds * eg_last; dg_last_extra
        // ================================================================
        const float eg_last = expf(g_last_val);

        if (tid == 0) scratch1[0] = 0.0f;
        __syncthreads();

        for (long idx = tid; idx < kv; idx += nthr) {
            dh_in[idx] = ds[idx] * eg_last;
            atomicAdd(&scratch1[0], ds[idx] * h_in[idx]);
        }
        __syncthreads();

        // Zero workspace arrays
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            DW[idx] = 0.0f; DQ[idx] = 0.0f; DK[idx] = 0.0f;
        }
        for (int idx = tid; idx < L * Vdim; idx += nthr)
            DU[idx] = 0.0f;
        for (int idx = tid; idx < L; idx += nthr) {
            DG[idx] = 0.0f; DB[idx] = 0.0f;
        }
        if (tid == 0) DG[L - 1] += scratch1[0] * eg_last;
        __syncthreads();

        // ================================================================
        // Contribution from ds via WMMA:
        //   DU += k @ ds  (k[L×K] @ ds[K×V] → [L×V])
        //   DK += VNEW @ ds^T  (VNEW[L×V] @ ds^T[V×K] → [L×K])
        // ================================================================
        // Load ds[K×V] as bf16 → buf1[K×V] (fits in Lp×Lp since K=V=Lp)
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(ds[idx]));
        __syncthreads();

        // DU_state = k @ ds_bf16 → scratch1[Lp×V]
        // Also save to ws area for reuse in v_new gating (d_vnew_state)
        float* ws_tmp = ws + ws_DB + chunk_size;
        float* ws_gs  = ws_tmp + Lp * Lp;
        wmma_nn<TQ>(smem_k, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            DU[idx] += scratch1[idx];
            ws_gs[idx] = scratch1[idx];  // save k@ds for v_new gating
        }
        __syncthreads();

        // DK_state = VNEW_bf16 @ ds_bf16^T → scratch1[Lp×K]
        // Load VNEW → buf2[Lp×V] as bf16
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            float val = (idx < L * Vdim) ? VNEW[idx] : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DK[idx] += scratch1[idx];
        __syncthreads();

        // ================================================================
        // Output gradient — BATCHED with WMMA where possible
        // ================================================================

        // --- term1: q @ h contribution ---
        // DQ[i,k] += scale * exp(g_cum[i]) * Σ_v d_out[i,v] * h_bf16[k,v]
        // dh_in[k,v] += Σ_i scale * exp(g_cum[i]) * d_out[i,v] * q[i,k]
        // DG[i] += scale * exp(g_cum[i]) * Σ_v d_out[i,v] * Σ_k q[i,k] * h_bf16[k,v]

        // Load d_out into buf2 (overwriting vnew bf16 — we have VNEW in workspace)
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[idx] = d_out[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        __syncthreads();

        // h_bf16 is still in buf1[K×V] from earlier.
        // DQ_term1 = scale * diag(exp(g)) * d_out @ h_bf16^T → need d_out[L×V] @ h_bf16^T[V×K]
        // = d_out @ h_bf16^T = WMMA_NT(d_out[Lp×V], h_bf16[K×V]) → scratch1[Lp×K]
        // But h_bf16 is [K×V] not [Lp×V], so NT won't work directly.
        // Use WMMA_NN: d_out[Lp×V] @ h_bf16_T[V×K]. We need to transpose h.
        // Actually h_bf16 is Kdim×Vdim stored row-major. For A@B^T with B=[K×V]:
        //   A[Lp×V] @ B^T [V×K] where B is row-major [K×V]
        //   This is A @ B^T = wmma_nt(A, B) only if B is [K×V] row-major.
        //   Wait, wmma_nt loads B with col_major, so B^T is [V×K].
        //   If B is stored row-major [K×V], loading with col_major gives us V×K layout.
        //   So wmma_nt(d_out[Lp×Vdim], h_bf16[K×Vdim]) → [Lp×K] if K==Lp
        //   Actually the wmma_nt computes C[M×N] = A[M×K_inner] @ B[N×K_inner]^T
        //   So: M=Lp, N=Kdim, K_inner=Vdim
        //   C[Lp×Kdim] = d_out[Lp×Vdim] @ h_bf16[Kdim×Vdim]^T
        //   This works! But Kdim must be multiple of 16 (it is).
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
        __syncthreads();

        // Apply scale * exp(g) and accumulate to DQ
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            DQ[idx] += scale * expf(smem_gcum[i]) * scratch1[i * Kdim + idx % Kdim];
        }
        __syncthreads();

        // dh_in[k,v] += Σ_i scale*exp(g[i]) * q[i,k] * d_out[i,v]
        // = (scale*exp(g)*q)^T @ d_out = q_scaled^T[K×Lp] @ d_out[Lp×V] → [K×V]
        // Prepare q_scaled in buf3 (overwriting w — we have W in workspace)
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_q[idx]) * scale * expf(smem_gcum[i]) : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // q_scaled^T @ d_out → scratch1[K×V]
        wmma_tn<TQ>(buf3, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();

        // Accumulate to dh_in
        for (long idx = tid; idx < kv; idx += nthr)
            dh_in[idx] += scratch1[idx];
        __syncthreads();

        // DG[i] from term1: scale*exp(g[i]) * dot(d_out[i,:], q[i,:]@h[:,:])
        // We already computed d_out @ h^T → scratch1 was overwritten. Recompute per-row.
        // Actually let's compute it differently: DG[i] += scale*exp(g[i]) * q[i,:] @ h @ d_out[i,:]^T
        // = scale*exp(g[i]) * Σ_k q[i,k] * Σ_v h[k,v] * d_out[i,v]
        // We need q@h per row. Let's do q@h_bf16 → scratch1[Lp×V] using WMMA.
        // h_bf16 is still in buf1[K×V].
        wmma_nn<TQ>(smem_q, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();

        // DG[i] += scale * exp(g[i]) * dot(scratch1[i,:], d_out[i,:])
        for (int i = tid; i < L; i += nthr) {
            float dot_val = 0.0f;
            for (int vv = 0; vv < Vdim; ++vv)
                dot_val += scratch1[i * Vdim + vv] * to_float(buf2[i * Vdim + vv]);
            DG[i] += scale * expf(smem_gcum[i]) * dot_val;
        }
        __syncthreads();

        // --- term2: Batched intra-chunk attention gradient ---
        // S[i,j] = bf16_trunc(exp(g[i]-g[j]) * dot(q[i,:], k[j,:]))  for j<=i, else 0
        // Already have q, k in smem. Compute S = q @ k^T via WMMA.
        wmma_nt<TQ>(smem_q, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();

        // Apply gating mask: S[i,j] = bf16_trunc(exp(g[i]-g[j]) * raw_S[i,j]) for j<=i
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] = bf16_trunc<TQ>(expf(smem_gcum[i] - smem_gcum[j]) * scratch1[idx]);
            else
                scratch1[idx] = 0.0f;
        }
        __syncthreads();

        // vnew_pre[j,v] = VNEW[j,v] / exp(g_last - g_cum[j])
        // Load vnew_pre into buf1 as bf16 (reusing buf1 — h_bf16 no longer needed)
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        // Only need Lp×Vdim portion of buf1 for vnew_pre, but buf1 is Lp×Lp.
        // We'll store vnew_pre in the first Lp×Vdim of buf1 if Vdim<=Lp (true since Vdim<=64=Lp).
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int j = idx / Vdim;
            float e_j = (j < L) ? expf(g_last_val - smem_gcum[j]) : 1.0f;
            float vnew_pre = (j < L) ? VNEW[j * Vdim + idx % Vdim] / e_j : 0.0f;
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(vnew_pre));
        }
        // Zero rest of buf1 if Vdim < Lp
        for (int idx = Lp * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();

        // grad_S[i,j] = Σ_v scale * d_out[i,v] * vnew_pre[j,v]
        // = scale * d_out[Lp×V] @ vnew_pre[Lp×V]^T → [Lp×Lp]
        // Use WMMA_NT(scale*d_out, vnew_pre)
        // Scale d_out: buf2 currently has d_out. Multiply by scale.
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float val = (i < L) ? to_float(buf2[idx]) * scale : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // Save S to workspace area 1 (ws_tmp, ws_gs already declared above)
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            ws_tmp[idx] = scratch1[idx];  // S saved
        __syncthreads();

        // Load vnew_pre into buf1[Lp×V] for grad_S computation
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int j = idx / Vdim;
            float e_j = (j < L) ? expf(g_last_val - smem_gcum[j]) : 1.0f;
            float vnp = (j < L) ? VNEW[j * Vdim + idx % Vdim] / e_j : 0.0f;
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(vnp));
        }
        __syncthreads();

        // grad_S_raw = scale_d_out @ vnew_pre^T → scratch1[Lp×Lp]
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Lp, Lp, Lp, Vdim);
        __syncthreads();

        // Save grad_S_raw to workspace area 2
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            ws_gs[idx] = scratch1[idx];
        __syncthreads();

        // DG from S gradient without atomics:
        // DG[i] += sum_{j<=i}(grad_S[i,j] * S[i,j]) - sum_{r>i}(grad_S[r,i] * S[r,i])
        for (int i = tid; i < L; i += nthr) {
            float dg_pos = 0.0f;
            float dg_neg = 0.0f;
            for (int j = 0; j <= i; ++j) {
                dg_pos += ws_gs[i * Lp + j] * ws_tmp[i * Lp + j];
            }
            for (int r = i + 1; r < L; ++r) {
                dg_neg += ws_gs[r * Lp + i] * ws_tmp[r * Lp + i];
            }
            DG[i] += dg_pos - dg_neg;
        }
        __syncthreads();

        // DU[j,v] += Σ_i S[i,j]/e_j * scale*d_out[i,v]
        // S_scaled = S/e_j → buf1 as bf16
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            float val = 0.0f;
            if (i < L && j <= i && j < L)
                val = ws_tmp[idx] / expf(g_last_val - smem_gcum[j]);
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // DU_term2 = S_scaled^T @ scale_d_out → [Lp×V]
        wmma_tn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr)
            DU[idx] += scratch1[idx];
        __syncthreads();

        // grad_S_masked = grad_S_raw * exp(g[i]-g[j]) for j<=i → buf1 as bf16
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            float val = 0.0f;
            if (i < L && j <= i)
                val = ws_gs[idx] * expf(smem_gcum[i] - smem_gcum[j]);
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // DQ_term2 = grad_S_masked @ k → [Lp×K]
        wmma_nn<TQ>(buf1, Lp, smem_k, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DQ[idx] += scratch1[idx];
        __syncthreads();

        // DK_term2 = grad_S_masked^T @ q → [Lp×K]
        wmma_tn<TQ>(buf1, Lp, smem_q, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DK[idx] += scratch1[idx];
        __syncthreads();

        // ================================================================
        // v_new gating gradients — batched with WMMA
        // ================================================================
        // Pass 1: compute d_pre[i,v] = DU[i,v] * e_i, DG contributions
        // Use precomputed k@ds from ws_gs for d_vnew_state
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim, vv = idx % Vdim;
            const float e_i = expf(g_last_val - smem_gcum[i]);
            const float v_new = VNEW[idx];
            const float pre = v_new / e_i;
            const float d_vnew = DU[idx];
            const float d_pre = d_vnew * e_i;
            const float d_vnew_state = ws_gs[idx]; // precomputed k@ds[i,v]
            const float d_e_state = d_vnew_state * pre;

            DU[idx] = d_pre;

            atomicAdd(&DG[L - 1],  d_e_state * e_i);
            atomicAdd(&DG[i],     -d_e_state * e_i);
        }
        __syncthreads();

        // Pass 2: DW = -d_pre @ h_bf16^T, dh_in += -W^T @ d_pre
        // Load d_pre (DU) → buf2 as bf16
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            float val = (idx < L * Vdim) ? DU[idx] : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        // Load h_bf16 → buf1[K×V]
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[idx]));
        for (int idx = Kdim * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        // Load W → buf3 as bf16
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            float val = (idx < L * Kdim) ? W[idx] : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // DW += -d_pre @ h_bf16^T = -(buf2[Lp×V] @ buf1[K×V]^T) → scratch1[Lp×K]
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DW[idx] += -scratch1[idx];
        __syncthreads();

        // dh_in += -W^T @ d_pre = -(buf3[Lp×K]^T @ buf2[Lp×V]) → scratch1[K×V]
        wmma_tn<TQ>(buf3, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();
        for (long idx = tid; idx < kv; idx += nthr)
            dh_in[idx] += -scratch1[idx];
        __syncthreads();

        // ================================================================
        // dM = DU @ (beta*V)^T + DW @ bkg^T via WMMA
        // ================================================================
        // Load raw v → buf2 (bf16, with beta scaling)
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            float val = 0.0f;
            if (pos < L) {
                float vraw = to_float(v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv]);
                val = bf16_trunc<TQ>(vraw * smem_beta[pos]);
            }
            buf2[idx] = from_float<TQ>(val);
        }
        // DU_bf16 → buf1 (note: buf1 is Lp×Lp, DU is L×V, Lp×V fits if V<=Lp)
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(DU[idx]));
        __syncthreads();

        // dM_part1 = DU_bf16 @ (beta*V)^T → scratch1[Lp×Lp]
        wmma_nt<TQ>(buf1, Vdim, buf2, Vdim, scratch1, Lp, Lp, Lp, Vdim);
        __syncthreads();

        // Prepare bkg → buf2[Lp×K] and DW_bf16 → buf1[Lp×K]
        // Reuse buf2 for bkg (Lp×K) — overwriting beta*V since we're done with it
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int pos = idx / Kdim;
            float val = (pos < L) ? to_float(smem_k[idx]) * smem_beta[pos] * expf(smem_gcum[pos]) : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        // Need DW in bf16 for WMMA — put in buf3
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            float val = (idx < L * Kdim) ? DW[idx] : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // dM_part2 = DW_bf16 @ bkg^T → add to scratch1 via temp
        // But we only have one scratch float buffer free (scratch1 has dM_part1).
        // Compute part2 into ws_gs (workspace), then add.
        // Actually ws_gs has grad_S_raw — we're done with it. Reuse.
        // But ws_gs is not Lp-strided. Use it as L×L or Lp×Lp.
        // Let me use a different approach: compute into scratch1 directly by doing
        // the WMMA and adding results manually.
        // Save dM_part1 to ws_gs temporarily.
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            ws_gs[idx] = scratch1[idx];
        __syncthreads();

        // dM_part2 = DW_bf16 @ bkg^T → scratch1[Lp×Lp]
        wmma_nt<TQ>(buf3, Kdim, buf2, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();

        // dM = part1 + part2, masked to lower triangle
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] = ws_gs[idx] + scratch1[idx];
            else
                scratch1[idx] = 0.0f;
        }
        __syncthreads();

        // ================================================================
        // d_v, d_beta from M^T @ DU and M^T @ DW via WMMA
        // ================================================================
        // M_bf16 → buf1[Lp×Lp]
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
        __syncthreads();

        // DU_bf16 → buf2[Lp×V]
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            float val = (idx < L * Vdim) ? DU[idx] : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // MT_DU = M^T @ DU_bf16 → ws_gs[Lp×V] (float, via WMMA then copy)
        wmma_tn<TQ>(buf1, Lp, buf2, Vdim, ws_gs, Vdim, Lp, Vdim, Lp);
        __syncthreads();

        // d_v[j,v] = beta[j] * MT_DU[j,v]
        // DB[j] += Σ_v MT_DU[j,v] * v_raw[j,v]
        // Reload raw v into buf2 for DB computation
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
                float mt_du = ws_gs[j * Vdim + vv];
                d_v[v_j_base + vv] = from_float<TQ>(mt_du * smem_beta[j]);
                db_acc += mt_du * to_float(buf2[j * Vdim + vv]);
            }
            DB[j] += db_acc;
        }
        __syncthreads();

        // DW_bf16 already in buf3. MT_DW = M^T @ DW_bf16 → ws_gs[Lp×K]
        wmma_tn<TQ>(buf1, Lp, buf3, Kdim, ws_gs, Kdim, Lp, Kdim, Lp);
        __syncthreads();

        // DK[j,k] += MT_DW[j,k] * beta[j] * exp(g[j])
        // DB[j] += Σ_k MT_DW[j,k] * exp(g[j]) * k[j,k]
        // DG[j] += Σ_k MT_DW[j,k] * beta[j] * exp(g[j]) * k[j,k]
        for (int j = tid; j < L; j += nthr) {
            const float egj = expf(smem_gcum[j]);
            float db_acc = 0.0f, dg_acc = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float mt_dw = ws_gs[j * Kdim + kk];
                float kj = to_float(smem_k[j * Kdim + kk]);
                DK[j * Kdim + kk] += mt_dw * smem_beta[j] * egj;
                db_acc += mt_dw * egj * kj;
                dg_acc += mt_dw * smem_beta[j] * egj * kj;
            }
            DB[j] += db_acc;
            DG[j] += dg_acc;
        }
        __syncthreads();

        // ================================================================
        // dA_grad = -M^T @ dM @ M^T
        // ================================================================
        // tmp1 = dM @ M^T
        for (int idx = tid; idx < L * L; idx += nthr) {
            const int i = idx / L, j = idx % L;
            float s = 0.0f;
            for (int m = 0; m < L; ++m)
                s += scratch1[i * Lp + m] * scratch2[j * Lp + m];
            ws_tmp[idx] = s;
        }
        __syncthreads();

        // dA_grad = -M^T @ tmp1
        for (int idx = tid; idx < L * L; idx += nthr) {
            const int i = idx / L, j = idx % L;
            float s = 0.0f;
            for (int m = 0; m < L; ++m)
                s += scratch2[m * Lp + i] * ws_tmp[m * L + j];
            scratch1[i * Lp + j] = -s;
        }
        __syncthreads();

        // dA_grad → DK, DB, DG
        for (int i = 0; i < L; ++i) {
            for (int j = tid; j < i; j += nthr) {
                float dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk)
                    dot_k += to_float(smem_k[i * Kdim + kk])
                           * to_float(smem_k[j * Kdim + kk]);
                const float exp_ij = expf(smem_gcum[i] - smem_gcum[j]);
                const float a_grad = scratch1[i * Lp + j];
                const float val = dot_k * exp_ij;

                atomicAdd(&DB[i], a_grad * val);
                const float dval = a_grad * smem_beta[i];
                const float ddot = dval * exp_ij;
                const float dexp = dval * dot_k;

                atomicAdd(&DG[i],  dexp * exp_ij);
                atomicAdd(&DG[j], -dexp * exp_ij);

                for (int kk = 0; kk < Kdim; ++kk) {
                    atomicAdd(&DK[i * Kdim + kk], ddot * to_float(smem_k[j * Kdim + kk]));
                    atomicAdd(&DK[j * Kdim + kk], ddot * to_float(smem_k[i * Kdim + kk]));
                }
            }
            __syncthreads();
        }

        // ================================================================
        // Write d_q, d_k, d_beta
        // ================================================================
        for (int i = tid; i < L; i += nthr) {
            const long qi_base = (((long)b * Tlen + cs + i) * H + h) * Kdim;
            const long gh_idx = ((long)b * Tlen + cs + i) * H + h;
            if (use_qk_l2norm_in_kernel) {
                float dot_q = 0.0f, dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_q += DQ[i * Kdim + kk] * to_float(smem_q[i * Kdim + kk]);
                    dot_k += DK[i * Kdim + kk] * to_float(smem_k[i * Kdim + kk]);
                }
                for (int kk = 0; kk < Kdim; ++kk) {
                    d_q[qi_base + kk] = from_float<TQ>(
                        (DQ[i * Kdim + kk] - to_float(smem_q[i * Kdim + kk]) * dot_q) * smem_invq[i]);
                    d_k[qi_base + kk] = from_float<TQ>(
                        (DK[i * Kdim + kk] - to_float(smem_k[i * Kdim + kk]) * dot_k) * smem_invk[i]);
                }
            } else {
                for (int kk = 0; kk < Kdim; ++kk) {
                    d_q[qi_base + kk] = from_float<TQ>(DQ[i * Kdim + kk]);
                    d_k[qi_base + kk] = from_float<TQ>(DK[i * Kdim + kk]);
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
                d_g[((long)b * Tlen + cs + i) * H + h] = from_float<TG>(running);
            }
        }
        __syncthreads();

        // Propagate ds backward
        for (long idx = tid; idx < kv; idx += nthr)
            ds[idx] = dh_in[idx];
        __syncthreads();
    }
}

// ============================================================================
// Backward kernel — fully parallelized (scalar fallback)
//
// Uses checkpoints (from checkpoint kernel) and a global workspace.
// All heavy loops are distributed across threads.
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_bwd_v2(
    TQ* __restrict__ d_q,
    TQ* __restrict__ d_k,
    TQ* __restrict__ d_v,
    TG* __restrict__ d_g,
    TB* __restrict__ d_beta,
    float* __restrict__ d_initial_state,
    const TQ* __restrict__ d_out,
    const float* __restrict__ d_final_state,
    const TQ* __restrict__ q_global,
    const TQ* __restrict__ k_global,
    const TQ* __restrict__ v_global,
    const TG* __restrict__ g_global,
    const TB* __restrict__ beta_global,
    const float* __restrict__ checkpoints,
    float* __restrict__ workspace,
    int workspace_stride,
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
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = static_cast<long>(Kdim) * Vdim;
    const long bh = static_cast<long>(b) * H + h;
    const float* cp_base = checkpoints + bh * static_cast<long>(num_chunks + 1) * kv;
    float* ds = d_initial_state + bh * kv;

    // --- Shared memory ---
    extern __shared__ char smem_raw[];
    float*  smem_M     = reinterpret_cast<float*>(smem_raw);                    // [C×C] float
    float*  smem_dM    = smem_M + kMaxC * kMaxC;                                // [C×C] float
    TQ*     smem_k     = reinterpret_cast<TQ*>(smem_dM + kMaxC * kMaxC);        // [C×K] TQ
    TQ*     smem_q     = smem_k + kMaxC * Kdim;                                 // [C×K] TQ
    TQ*     smem_v_loc = smem_q + kMaxC * Kdim;                                 // [C×V] TQ
    float*  smem_g_cum = reinterpret_cast<float*>(smem_v_loc + kMaxC * Vdim);   // [C] float
    float*  smem_beta  = smem_g_cum + kMaxC;                                     // [C] float
    float*  smem_inv_q = smem_beta  + kMaxC;                                     // [C] float
    float*  smem_inv_k = smem_inv_q + kMaxC;                                     // [C] float

    // --- Global workspace per (b,h) ---
    float* ws = workspace + bh * workspace_stride;
    const int ws_W    = 0;
    const int ws_VNEW = ws_W    + chunk_size * Kdim;
    const int ws_DU   = ws_VNEW + chunk_size * Vdim;
    const int ws_DW   = ws_DU   + chunk_size * Vdim;
    const int ws_DQ   = ws_DW   + chunk_size * Kdim;
    const int ws_DK   = ws_DQ   + chunk_size * Kdim;
    const int ws_DG   = ws_DK   + chunk_size * Kdim;
    const int ws_DB   = ws_DG   + chunk_size;
    float* W    = ws + ws_W;
    float* VNEW = ws + ws_VNEW;
    float* DU   = ws + ws_DU;
    float* DW   = ws + ws_DW;
    float* DQ   = ws + ws_DQ;
    float* DK   = ws + ws_DK;
    float* DG   = ws + ws_DG;
    float* DB   = ws + ws_DB;

    // Seed ds
    for (long idx = tid; idx < kv; idx += nthreads) {
        ds[idx] = d_final_state ? d_final_state[bh * kv + idx] : 0.0f;
    }
    __syncthreads();

    for (int chunk = num_chunks - 1; chunk >= 0; --chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);
        const float* h_in = cp_base + static_cast<long>(chunk) * kv;
        // dh_in aliases the NEXT checkpoint slot (will be overwritten)
        float* dh_in = const_cast<float*>(cp_base + static_cast<long>(chunk + 1) * kv);

        // ================================================================
        // Load q, k, v for this chunk
        // ================================================================
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int pos = idx / Kdim;
            const int kk  = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k_global[gi];
            smem_q[pos * Kdim + kk] = q_global[gi];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int pos = idx / Vdim;
            const int vv  = idx % Vdim;
            smem_v_loc[pos * Vdim + vv] = v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        __syncthreads();

        // Compute scalars and norms
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g_global[gh]);
                smem_g_cum[i] = acc;
                smem_beta[i]  = to_float(beta_global[gh]);
            }
        }
        __syncthreads();

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

        // Normalize k, q in-place
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
        // Build M matrix (same as forward)
        // ================================================================
        for (int idx = tid; idx < kMaxC * kMaxC; idx += nthreads) {
            smem_M[idx] = 0.0f;
        }
        __syncthreads();
        for (int i = tid; i < L; i += nthreads) {
            smem_M[i * kMaxC + i] = 1.0f;
        }
        __syncthreads();

        for (int row = 1; row < L; ++row) {
            for (int j = tid; j < row; j += nthreads) {
                float s = 0.0f;
                for (int m = j; m < row; ++m) {
                    float dot_k = 0.0f;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        dot_k += to_float(smem_k[row * Kdim + kk])
                               * to_float(smem_k[m   * Kdim + kk]);
                    }
                    s += smem_beta[row] * dot_k
                       * expf(smem_g_cum[row] - smem_g_cum[m])
                       * smem_M[m * kMaxC + j];
                }
                smem_M[row * kMaxC + j] = bf16_trunc<TQ>(-s);
            }
            __syncthreads();
        }

        // ================================================================
        // Recompute W [L×K] and VNEW [L×V] from M, k, v, beta, g_cum, h_in
        // ================================================================

        // Zero workspace arrays
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            W[idx] = 0.0f;
            DW[idx] = 0.0f;
            DQ[idx] = 0.0f;
            DK[idx] = 0.0f;
        }
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            VNEW[idx] = 0.0f;
            DU[idx] = 0.0f;
        }
        for (int idx = tid; idx < L; idx += nthreads) {
            DG[idx] = 0.0f;
            DB[idx] = 0.0f;
        }
        __syncthreads();

        // Compute W[i,k] = bf16(Σ_m M[i,m] * bf16(k_norm[m,k] * beta[m] * exp(g_cum[m])))
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int i  = idx / Kdim;
            const int kk = idx % Kdim;
            float acc = 0.0f;
            for (int m = 0; m <= i; ++m) {
                float kbg = bf16_trunc<TQ>(
                    to_float(smem_k[m * Kdim + kk]) * smem_beta[m] * expf(smem_g_cum[m]));
                acc += smem_M[i * kMaxC + m] * kbg;
            }
            W[i * Kdim + kk] = bf16_trunc<TQ>(acc);
        }
        __syncthreads();

        // Compute VNEW[i,v] = bf16((u_i - w_i @ bf16(h_in)) * exp(g_last - g_cum[i]))
        const float g_last_val = smem_g_cum[L - 1];
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int i  = idx / Vdim;
            const int vv = idx % Vdim;

            // u_i_v
            float u_val = 0.0f;
            for (int m = 0; m <= i; ++m) {
                float vb = bf16_trunc<TQ>(
                    to_float(smem_v_loc[m * Vdim + vv]) * smem_beta[m]);
                u_val += smem_M[i * kMaxC + m] * vb;
            }
            u_val = bf16_trunc<TQ>(u_val);

            // wh
            float wh = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float h_cs = bf16_trunc<TQ>(h_in[kk * Vdim + vv]);
                wh += W[i * Kdim + kk] * h_cs;
            }

            float e_i = expf(g_last_val - smem_g_cum[i]);
            VNEW[i * Vdim + vv] = bf16_trunc<TQ>((u_val - wh) * e_i);
        }
        __syncthreads();

        // ================================================================
        // dh_in = ds * eg_last; dg_last_extra
        // ================================================================
        const float eg_last = expf(g_last_val);

        // dh_in = ds * eg_last, and accumulate dg_last_extra using atomicAdd
        // We use smem_dM[0] as a temporary for dg_last_extra
        if (tid == 0) {
            smem_dM[0] = 0.0f;
        }
        __syncthreads();

        for (long idx = tid; idx < kv; idx += nthreads) {
            dh_in[idx] = ds[idx] * eg_last;
            float contrib = ds[idx] * h_in[idx];
            atomicAdd(&smem_dM[0], contrib);
        }
        __syncthreads();

        if (tid == 0) {
            DG[L - 1] += smem_dM[0] * eg_last;
        }
        __syncthreads();

        // Zero smem_dM properly now (was used as temp)
        for (int idx = tid; idx < kMaxC * kMaxC; idx += nthreads) {
            smem_dM[idx] = 0.0f;
        }
        __syncthreads();

        // ================================================================
        // Contribution from ds: DU += ds^T @ k_norm (per position)
        //                       DK += ds @ VNEW^T (per position)
        // ================================================================
        for (int i = 0; i < L; ++i) {
            // DU[i,v] += Σ_k ds[k,v] * k_norm[i,k]
            for (int vv = tid; vv < Vdim; vv += nthreads) {
                float acc = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    acc += ds[kk * Vdim + vv] * to_float(smem_k[i * Kdim + kk]);
                }
                DU[i * Vdim + vv] += acc;
            }
            // DK[i,k] += Σ_v ds[k,v] * VNEW[i,v]
            for (int kk = tid; kk < Kdim; kk += nthreads) {
                float acc = 0.0f;
                for (int vv = 0; vv < Vdim; ++vv) {
                    acc += ds[kk * Vdim + vv] * VNEW[i * Vdim + vv];
                }
                DK[i * Kdim + kk] += acc;
            }
            __syncthreads();
        }

        // ================================================================
        // Output gradient contribution: d_out → DQ, DK, DG, DU, dh_in
        // ================================================================
        for (int i = 0; i < L; ++i) {
            const long do_i_base = (((long)b * Tlen + cs + i) * H + h) * Vdim;
            const float eg_i = expf(smem_g_cum[i]);

            // term1: d_out[i] @ h_in^T contribution to DQ, dh_in, DG
            for (int vv = tid; vv < Vdim; vv += nthreads) {
                const float do_v = to_float(d_out[do_i_base + vv]) * scale;
                float qh = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qi = to_float(smem_q[i * Kdim + kk]);
                    const float hq = bf16_trunc<TQ>(h_in[kk * Vdim + vv]);
                    qh += qi * hq;
                    atomicAdd(&DQ[i * Kdim + kk], do_v * eg_i * hq);
                    atomicAdd(&dh_in[kk * Vdim + vv], do_v * eg_i * qi);
                }
                atomicAdd(&DG[i], do_v * eg_i * qh);
            }
            __syncthreads();

            // term2: intra-chunk attention gradient
            for (int j = 0; j <= i; ++j) {
                const float e_j = expf(g_last_val - smem_g_cum[j]);
                const float exp_ij = expf(smem_g_cum[i] - smem_g_cum[j]);

                // dot(q_norm[i], k_norm[j])
                float dot_qk = 0.0f;
                // Parallel reduction across threads
                for (int kk = tid; kk < Kdim; kk += nthreads) {
                    dot_qk += to_float(smem_q[i * Kdim + kk])
                            * to_float(smem_k[j * Kdim + kk]);
                }
                // Warp reduction
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dot_qk += __shfl_down_sync(0xffffffff, dot_qk, offset);
                }
                // Cross-warp reduction via shared memory
                // Use smem_dM scratch space for reduction
                if (tid % 32 == 0) {
                    smem_dM[tid / 32] = dot_qk;
                }
                __syncthreads();
                if (tid == 0) {
                    float total = 0.0f;
                    for (int w = 0; w < (nthreads + 31) / 32; ++w) {
                        total += smem_dM[w];
                    }
                    smem_dM[kMaxC * kMaxC - 1] = total;  // store result
                }
                __syncthreads();
                dot_qk = smem_dM[kMaxC * kMaxC - 1];

                const float s_ij = bf16_trunc<TQ>(exp_ij * dot_qk);

                // grad_s = Σ_v d_out[i,v] * v_new_pre[j,v] / e_j
                float grad_s_local = 0.0f;
                for (int vv = tid; vv < Vdim; vv += nthreads) {
                    const float do_v = to_float(d_out[do_i_base + vv]) * scale;
                    const float v_new_pre = VNEW[j * Vdim + vv] / e_j;
                    grad_s_local += do_v * v_new_pre;
                    atomicAdd(&DU[j * Vdim + vv], do_v * (s_ij / e_j));
                }
                // Reduce grad_s
                for (int offset = 16; offset > 0; offset >>= 1) {
                    grad_s_local += __shfl_down_sync(0xffffffff, grad_s_local, offset);
                }
                if (tid % 32 == 0) {
                    smem_dM[tid / 32] = grad_s_local;
                }
                __syncthreads();
                float grad_s = 0.0f;
                if (tid == 0) {
                    for (int w = 0; w < (nthreads + 31) / 32; ++w) {
                        grad_s += smem_dM[w];
                    }
                    smem_dM[kMaxC * kMaxC - 2] = grad_s;
                }
                __syncthreads();
                grad_s = smem_dM[kMaxC * kMaxC - 2];

                const float coeff = grad_s * exp_ij;
                for (int kk = tid; kk < Kdim; kk += nthreads) {
                    const float qi = to_float(smem_q[i * Kdim + kk]);
                    const float kj = to_float(smem_k[j * Kdim + kk]);
                    atomicAdd(&DQ[i * Kdim + kk], coeff * kj);
                    atomicAdd(&DK[j * Kdim + kk], coeff * qi);
                }
                if (tid == 0) {
                    DG[i] += grad_s * s_ij;
                    DG[j] -= grad_s * s_ij;
                }
                __syncthreads();
            }
        }

        // ================================================================
        // v_new gating gradients: merge into DU, DW, DG, dh_in
        // ================================================================
        for (int i = 0; i < L; ++i) {
            const float e_i = expf(g_last_val - smem_g_cum[i]);

            for (int vv = tid; vv < Vdim; vv += nthreads) {
                const float v_new = VNEW[i * Vdim + vv];
                const float pre = v_new / e_i;
                const float d_vnew = DU[i * Vdim + vv];
                const float d_pre = d_vnew * e_i;

                // d_vnew from state contribution
                float d_vnew_state = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    d_vnew_state += ds[kk * Vdim + vv]
                                  * to_float(smem_k[i * Kdim + kk]);
                }
                const float d_e_state = d_vnew_state * pre;

                DU[i * Vdim + vv] = d_pre;  // overwrite with corrected value

                atomicAdd(&DG[L - 1],  d_e_state * e_i);
                atomicAdd(&DG[i],     -d_e_state * e_i);

                for (int kk = 0; kk < Kdim; ++kk) {
                    float h_cs = bf16_trunc<TQ>(h_in[kk * Vdim + vv]);
                    atomicAdd(&DW[i * Kdim + kk], -d_pre * h_cs);
                    atomicAdd(&dh_in[kk * Vdim + vv], -d_pre * W[i * Kdim + kk]);
                }
            }
            __syncthreads();
        }

        // ================================================================
        // Compute dM[i,j] = DU[i,:] @ (beta*V)[j,:] + DW[i,:] @ (beta*K*exp(g))[j,:]
        // ================================================================
        for (int idx = tid; idx < L * L; idx += nthreads) {
            const int i = idx / L;
            const int j = idx % L;
            if (j <= i) {
                float s = 0.0f;
                for (int vv = 0; vv < Vdim; ++vv) {
                    float vb = bf16_trunc<TQ>(
                        to_float(smem_v_loc[j * Vdim + vv]) * smem_beta[j]);
                    s += DU[i * Vdim + vv] * vb;
                }
                for (int kk = 0; kk < Kdim; ++kk) {
                    float kbg = bf16_trunc<TQ>(
                        to_float(smem_k[j * Kdim + kk]) * smem_beta[j]
                        * expf(smem_g_cum[j]));
                    s += DW[i * Kdim + kk] * kbg;
                }
                smem_dM[i * kMaxC + j] = s;
            }
        }
        __syncthreads();

        // ================================================================
        // d_v, d_beta from M^T @ DU and M^T @ DW
        // ================================================================
        for (int j = 0; j < L; ++j) {
            const long v_j_base = (((long)b * Tlen + cs + j) * H + h) * Vdim;

            // d_v[j,v] = beta[j] * Σ_i M[i,j] * DU[i,v]   (M^T column j)
            for (int vv = tid; vv < Vdim; vv += nthreads) {
                float t = 0.0f;
                for (int i = j; i < L; ++i) {
                    t += smem_M[i * kMaxC + j] * DU[i * Vdim + vv];
                }
                d_v[v_j_base + vv] = from_float<TQ>(t * smem_beta[j]);
                atomicAdd(&DB[j], t * to_float(smem_v_loc[j * Vdim + vv]));
            }
            __syncthreads();

            // DK from M^T @ DW, and DB, DG contributions
            for (int kk = tid; kk < Kdim; kk += nthreads) {
                float t = 0.0f;
                for (int i = j; i < L; ++i) {
                    t += smem_M[i * kMaxC + j] * DW[i * Kdim + kk];
                }
                const float kj = to_float(smem_k[j * Kdim + kk]);
                const float egj = expf(smem_g_cum[j]);
                atomicAdd(&DK[j * Kdim + kk], t * smem_beta[j] * egj);
                atomicAdd(&DB[j], t * egj * kj);
                atomicAdd(&DG[j], t * smem_beta[j] * egj * kj);
            }
            __syncthreads();
        }

        // ================================================================
        // Back-propagate through A: dA_grad = -M^T @ dM @ M^T
        // tmp = dM @ M^T then result = -M^T @ tmp
        // Reuse smem_dM for tmp_mat after dM is consumed.
        // We need dM and M simultaneously, so use a 2-pass approach
        // writing tmp over dM.
        // ================================================================

        // Pass 1: tmp[i,j] = Σ_m dM[i,m] * M[j,m]  (= dM @ M^T)
        // Compute into smem_dM (overwriting dM)
        // But we need dM during this computation! Do it in-place with a
        // temporary in registers (process one row at a time).

        // Use workspace DB area as temporary (it's small and we're done with it
        // for the current row computations). Actually, let's use a different
        // approach: compute tmp into the workspace area after DB.
        // The workspace has extra space after DB.

        // Actually, we have space in the workspace after DB:
        // ws_after_DB = ws_DB + chunk_size (= DB end)
        float* ws_tmp = ws + ws_DB + chunk_size;
        // This overlaps nothing since workspace_stride >= ws_DB + chunk_size + 2*L*L
        // (caller must allocate enough). We'll check this in the launch function.

        // tmp1[i,j] = Σ_m dM[i,m] * M[j,m]
        for (int idx = tid; idx < L * L; idx += nthreads) {
            const int i = idx / L;
            const int j = idx % L;
            float s = 0.0f;
            for (int m = 0; m < L; ++m) {
                s += smem_dM[i * kMaxC + m] * smem_M[j * kMaxC + m];
            }
            ws_tmp[i * L + j] = s;
        }
        __syncthreads();

        // tmp2[i,j] = -Σ_m M[m,i] * tmp1[m,j]   (= -M^T @ tmp1)
        // Write result back to smem_dM (reuse as dA_grad)
        for (int idx = tid; idx < L * L; idx += nthreads) {
            const int i = idx / L;
            const int j = idx % L;
            float s = 0.0f;
            for (int m = 0; m < L; ++m) {
                s += smem_M[m * kMaxC + i] * ws_tmp[m * L + j];
            }
            smem_dM[i * kMaxC + j] = -s;
        }
        __syncthreads();

        // dA_grad is now in smem_dM. Use it to update DK, DB, DG.
        for (int i = 0; i < L; ++i) {
            for (int j = tid; j < i; j += nthreads) {
                // dot(k_norm[i], k_norm[j])
                float dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_k += to_float(smem_k[i * Kdim + kk])
                           * to_float(smem_k[j * Kdim + kk]);
                }
                const float exp_ij = expf(smem_g_cum[i] - smem_g_cum[j]);
                const float a_grad = smem_dM[i * kMaxC + j];
                const float val = dot_k * exp_ij;

                atomicAdd(&DB[i], a_grad * val);

                const float dval = a_grad * smem_beta[i];
                const float ddot = dval * exp_ij;
                const float dexp = dval * dot_k;

                atomicAdd(&DG[i],  dexp * exp_ij);
                atomicAdd(&DG[j], -dexp * exp_ij);

                for (int kk = 0; kk < Kdim; ++kk) {
                    const float ki = to_float(smem_k[i * Kdim + kk]);
                    const float kj = to_float(smem_k[j * Kdim + kk]);
                    atomicAdd(&DK[i * Kdim + kk], ddot * kj);
                    atomicAdd(&DK[j * Kdim + kk], ddot * ki);
                }
            }
            __syncthreads();
        }

        // ================================================================
        // Write d_q, d_k (with L2norm backward if needed), d_beta
        // ================================================================
        for (int i = tid; i < L; i += nthreads) {
            const long q_i_base = (((long)b * Tlen + cs + i) * H + h) * Kdim;
            const long k_i_base = q_i_base;
            const long gh_idx = ((long)b * Tlen + cs + i) * H + h;

            if (use_qk_l2norm_in_kernel) {
                float dot_q = 0.0f, dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_q += DQ[i * Kdim + kk] * to_float(smem_q[i * Kdim + kk]);
                    dot_k += DK[i * Kdim + kk] * to_float(smem_k[i * Kdim + kk]);
                }
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qn = to_float(smem_q[i * Kdim + kk]);
                    const float kn = to_float(smem_k[i * Kdim + kk]);
                    d_q[q_i_base + kk] = from_float<TQ>(
                        (DQ[i * Kdim + kk] - qn * dot_q) * smem_inv_q[i]);
                    d_k[k_i_base + kk] = from_float<TQ>(
                        (DK[i * Kdim + kk] - kn * dot_k) * smem_inv_k[i]);
                }
            } else {
                for (int kk = 0; kk < Kdim; ++kk) {
                    d_q[q_i_base + kk] = from_float<TQ>(DQ[i * Kdim + kk]);
                    d_k[k_i_base + kk] = from_float<TQ>(DK[i * Kdim + kk]);
                }
            }
            d_beta[gh_idx] = from_float<TB>(DB[i]);
        }
        __syncthreads();

        // ================================================================
        // Reverse cumsum for d_g
        // ================================================================
        // DG[i] contains per-position gradients. d_g output needs reverse cumsum.
        if (tid == 0) {
            float running = 0.0f;
            for (int i = L - 1; i >= 0; --i) {
                running += DG[i];
                const long gh_idx = ((long)b * Tlen + cs + i) * H + h;
                d_g[gh_idx] = from_float<TG>(running);
            }
        }
        __syncthreads();

        // ================================================================
        // Propagate ds backward: ds = dh_in
        // ================================================================
        for (long idx = tid; idx < kv; idx += nthreads) {
            ds[idx] = dh_in[idx];
        }
        __syncthreads();
    }
}


// ============================================================================
// Multi-kernel backward launch
// ============================================================================

template<typename TQ, typename TG, typename TB>
void launch_bwd_v2_multikernel(
    Tensor& d_q, Tensor& d_k, Tensor& d_v, Tensor& d_g, Tensor& d_beta,
    Tensor& d_initial_state, const Tensor& d_out, const Tensor* d_final_state,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints, Tensor& workspace, bool skip_checkpoint,
    cudaStream_t stream) {

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
    int phase2_threads = 1024;
    if (const char* env = std::getenv("SUROGATE_GDR_BWD_PHASE2_THREADS")) {
        const int parsed = std::atoi(env);
        if (parsed >= 64 && parsed <= 1024) {
            phase2_threads = (parsed / 32) * 32;
        }
    }
    const int v_tile = Vdim >= 64 ? 64 : Vdim;

    ChunkWorkspaceLayout cwl = make_chunk_ws<TQ>(Lp, Kdim, Vdim);
    const int chunk_ws_stride = cwl.total;

    // Workspace layout: [chunk_workspace | dh_storage]
    // chunk_workspace: B*H*num_chunks * chunk_ws_stride floats
    // dh_storage: B*H*num_chunks * Kdim*Vdim floats (per-chunk dh for Phase 3)
    float* chunk_ws = workspace.get<float>();
    float* dh_storage = chunk_ws + static_cast<long>(B) * H * num_chunks * chunk_ws_stride;

    const int phase1_v_tile = Vdim >= 64 ? 64 : Vdim;

    // Phase 1 shared memory
    const std::size_t phase1_smem =
        static_cast<std::size_t>(Lp) * Lp * sizeof(float) * 2    // scratch1 + scratch2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ) * 2   // k, q
        + static_cast<std::size_t>(Lp) * Lp * sizeof(TQ)         // buf1
        + static_cast<std::size_t>(Lp) * phase1_v_tile * sizeof(TQ) // buf2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)       // buf3
        + static_cast<std::size_t>(Lp) * 6 * sizeof(float);      // gcum, beta, invq, invk, eg, e_last

    // Phase 3 shared memory (q is streamed from global; only k is staged in shared).
    const std::size_t phase3_smem =
        static_cast<std::size_t>(Lp) * Lp * sizeof(float) * 2    // scratch1 + scratch2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)       // k
        + static_cast<std::size_t>(Lp) * Lp * sizeof(TQ)         // buf1
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)       // buf2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)       // buf3
        + static_cast<std::size_t>(Lp) * 6 * sizeof(float);      // gcum, beta, invq, invk, eg, e_last

    // Phase 2: ultra-lightweight, just scratch1[K×V] + buf1[K×K bf16] + buf2[K×V bf16]
    const std::size_t phase2_smem =
        static_cast<std::size_t>(Kdim) * v_tile * sizeof(float)  // scratch1[K×Vtile]
        + static_cast<std::size_t>(Kdim) * Kdim * sizeof(TQ)     // buf1[K×K] (C_bf16)
        + static_cast<std::size_t>(Kdim) * v_tile * sizeof(TQ);  // buf2[K×Vtile] (ds_bf16)

    // --- Checkpoint kernel (same as before) ---
    const std::size_t cp_wmma_smem =
        static_cast<std::size_t>(Kdim) * Vdim * sizeof(float)
        + static_cast<std::size_t>(Lp) * Lp * sizeof(float) * 2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)
        + static_cast<std::size_t>(Lp) * Lp * sizeof(TQ)
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)
        + static_cast<std::size_t>(Lp) * 3 * sizeof(float);
    const std::size_t cp_scalar_smem =
        static_cast<std::size_t>(Lp) * Lp * sizeof(float)          // M
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)         // k
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)         // v
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)         // u
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)         // w
        + static_cast<std::size_t>(Lp) * 3 * sizeof(float);        // g_cum, beta, inv_k

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    auto require_smem = [&](const char* kernel_name, std::size_t requested) {
        if (requested > static_cast<std::size_t>(smem_optin)) {
            throw std::logic_error(
                std::string("gated_delta_rule_chunk_backward_v2: ")
                + kernel_name + " requires dynamic shared memory " + std::to_string(requested)
                + " bytes, exceeds device opt-in limit " + std::to_string(smem_optin) + " bytes");
        }
    };

    require_smem("gdr_bwd_phase1_wmma", phase1_smem);
    require_smem("gdr_bwd_phase2_wmma", phase2_smem);
    require_smem("gdr_bwd_phase3_wmma", phase3_smem);
    const bool use_wmma_checkpoint = (cp_wmma_smem <= static_cast<std::size_t>(smem_optin));
    if (!use_wmma_checkpoint) {
        require_smem("gdr_checkpoint_scalar", cp_scalar_smem);
    }
    if (std::getenv("SUROGATE_DEBUG_GDR")) {
        fprintf(stderr,
                "[GDR bwd cfg] mode=tiled_multi K=%d V=%d Vtile=%d cp=%s smem_optin=%d cp_wmma=%zu cp_scalar=%zu p1=%zu p2=%zu p3=%zu\n",
                Kdim, Vdim, v_tile, use_wmma_checkpoint ? "wmma" : "scalar",
                smem_optin, cp_wmma_smem, cp_scalar_smem, phase1_smem, phase2_smem, phase3_smem);
    }

    // Optional per-phase timing via env var
    const bool profile = (std::getenv("SUROGATE_GDR_PROFILE") != nullptr);
    cudaEvent_t ev[5];
    if (profile) {
        for (auto& e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev[0], stream);
    }

    if (!skip_checkpoint) {
        if (use_wmma_checkpoint) {
            launch_gdr_checkpoint_wmma<TQ, TG, TB>(
                checkpoints.get<float>(),
                k.get<TQ>(), v.get<TQ>(), g.get<TG>(), beta.get<TB>(),
                initial_state ? initial_state->get<float>() : nullptr,
                B, Tlen, H, Kdim, Vdim, chunk_size, use_qk_l2norm_in_kernel,
                chunk_threads, cp_wmma_smem, stream);
        } else {
            launch_gdr_checkpoint_scalar<TQ, TG, TB>(
                checkpoints.get<float>(),
                k.get<TQ>(), v.get<TQ>(), g.get<TG>(), beta.get<TB>(),
                initial_state ? initial_state->get<float>() : nullptr,
                B, Tlen, H, Kdim, Vdim, chunk_size, use_qk_l2norm_in_kernel,
                chunk_threads, cp_scalar_smem, stream);
        }
    }

    if (profile) cudaEventRecord(ev[1], stream);

    // --- Phase 1: chunk-parallel ---
    launch_gdr_bwd_phase1<TQ, TG, TB>(
        d_out.get<TQ>(),
        q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        checkpoints.get<float>(),
        d_final_state ? d_final_state->get<float>() : nullptr,
        chunk_ws,
        d_initial_state.get<float>(),
        chunk_ws_stride,
        B, Tlen, H, Kdim, Vdim, num_chunks, chunk_size, scale,
        use_qk_l2norm_in_kernel,
        chunk_threads, phase1_smem, stream);

    if (profile) cudaEventRecord(ev[2], stream);

    // --- Phase 2: sequential (single matmul per chunk) ---
    launch_gdr_bwd_phase2<TQ>(
        checkpoints.get<float>(),
        d_final_state ? d_final_state->get<float>() : nullptr,
        d_initial_state.get<float>(),
        chunk_ws,
        dh_storage,
        chunk_ws_stride,
        B, Tlen, H, Kdim, Vdim, num_chunks, chunk_size,
        phase2_threads, phase2_smem, stream);

    if (profile) cudaEventRecord(ev[3], stream);

    // --- Phase 3: chunk-parallel ---
    launch_gdr_bwd_phase3<TQ, TG, TB>(
        d_q.get<TQ>(), d_k.get<TQ>(), d_v.get<TQ>(),
        d_g.get<TG>(), d_beta.get<TB>(),
        d_out.get<TQ>(),
        q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        checkpoints.get<float>(),
        dh_storage,
        chunk_ws,
        chunk_ws_stride,
        B, Tlen, H, Kdim, Vdim, num_chunks, chunk_size, scale,
        use_qk_l2norm_in_kernel,
        chunk_threads, phase3_smem, stream);

    if (profile) {
        cudaEventRecord(ev[4], stream);
        cudaEventSynchronize(ev[4]);
        float ms[4];
        for (int i = 0; i < 4; ++i) cudaEventElapsedTime(&ms[i], ev[i], ev[i+1]);
        fprintf(stderr, "[GDR bwd] cp=%.3fms p1=%.3fms p2=%.3fms p3=%.3fms total=%.3fms\n",
                ms[0], ms[1], ms[2], ms[3], ms[0]+ms[1]+ms[2]+ms[3]);
        for (auto& e : ev) cudaEventDestroy(e);
    }
}

template<typename TQ, typename TG, typename TB>
void launch_bwd_v2(
    Tensor& d_q, Tensor& d_k, Tensor& d_v, Tensor& d_g, Tensor& d_beta,
    Tensor& d_initial_state, const Tensor& d_out, const Tensor* d_final_state,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints, Tensor& workspace, bool skip_checkpoint,
    cudaStream_t stream) {
    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);

    const int threads = 128;
    dim3 grid(B, H);
    const bool force_scalar_bwd = (std::getenv("SUROGATE_GDR_FORCE_SCALAR_BWD") != nullptr);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    // Try WMMA multi-kernel path for bf16/fp16 with tensor-core friendly K,V.
    constexpr bool can_wmma = std::is_same_v<TQ, nv_bfloat16> || std::is_same_v<TQ, half>;
    if constexpr (can_wmma) {
        if (force_scalar_bwd) {
            // Debug switch to compare multikernel path against scalar fallback.
        } else
        if (Kdim <= 128 && Vdim <= 128 && Kdim % 16 == 0 && Vdim % 16 == 0) {
            const int phase1_v_tile = Vdim >= 64 ? 64 : Vdim;
            const std::size_t phase1_smem =
                static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float) * 2
                + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ) * 2
                + static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(TQ)
                + static_cast<std::size_t>(kMaxC) * phase1_v_tile * sizeof(TQ)
                + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)
                + static_cast<std::size_t>(kMaxC) * 6 * sizeof(float);
            const std::size_t phase3_smem =
                static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float) * 2
                + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)
                + static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(TQ)
                + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ)
                + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)
                + static_cast<std::size_t>(kMaxC) * 6 * sizeof(float);
            const bool fits_multikernel =
                (phase1_smem <= static_cast<std::size_t>(smem_optin))
                && (phase3_smem <= static_cast<std::size_t>(smem_optin));
            if (!fits_multikernel) {
                if (std::getenv("SUROGATE_DEBUG_GDR")) {
                    fprintf(stderr,
                            "[GDR bwd cfg] mode=scalar_fallback reason=smem_limit K=%d V=%d p1=%zu p3=%zu optin=%d\n",
                            Kdim, Vdim, phase1_smem, phase3_smem, smem_optin);
                }
            } else {
            launch_bwd_v2_multikernel<TQ, TG, TB>(
                d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state,
                scale, chunk_size, use_qk_l2norm_in_kernel,
                checkpoints, workspace, skip_checkpoint, stream);
            return;
            }
        }
    }

    // Fallback: scalar kernels
    // Checkpoint kernel shared memory
    const std::size_t cp_smem =
        static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float)    // M
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)      // k
        + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ)      // v
        + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ)      // u
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)      // w
        + static_cast<std::size_t>(kMaxC) * 3 * sizeof(float);     // g_cum, beta, inv_k

    launch_gdr_checkpoint_scalar<TQ, TG, TB>(
        checkpoints.get<float>(),
        k.get<TQ>(), v.get<TQ>(), g.get<TG>(), beta.get<TB>(),
        initial_state ? initial_state->get<float>() : nullptr,
        B, Tlen, H, Kdim, Vdim, chunk_size, use_qk_l2norm_in_kernel,
        threads, cp_smem, stream);

    // Backward kernel shared memory
    const std::size_t bwd_smem =
        static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float) * 2  // M + dM
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ) * 2    // k, q
        + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ)        // v
        + static_cast<std::size_t>(kMaxC) * 4 * sizeof(float);       // scalars

    cudaFuncSetAttribute(
        gdr_chunk_bwd_v2<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(bwd_smem));

    gdr_chunk_bwd_v2<TQ, TG, TB><<<grid, threads, bwd_smem, stream>>>(
        d_q.get<TQ>(), d_k.get<TQ>(), d_v.get<TQ>(),
        d_g.get<TG>(), d_beta.get<TB>(),
        d_initial_state.get<float>(),
        d_out.get<TQ>(),
        d_final_state ? d_final_state->get<float>() : nullptr,
        q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        checkpoints.get<float>(),
        workspace.get<float>(),
        static_cast<int>(workspace.Sizes[2]),
        Tlen, H, Kdim, Vdim, chunk_size, scale, use_qk_l2norm_in_kernel);
}

// Dtype dispatch chain for backward
template<typename TQ, typename TG>
void dispatch_bwd_v2_beta(
    Tensor& d_q, Tensor& d_k, Tensor& d_v, Tensor& d_g, Tensor& d_beta,
    Tensor& d_initial_state, const Tensor& d_out, const Tensor* d_final_state,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints, Tensor& workspace, bool skip_checkpoint,
    cudaStream_t stream) {
    switch (beta.DType) {
        case ETensorDType::FP32:
            launch_bwd_v2<TQ, TG, float>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        case ETensorDType::BF16:
            launch_bwd_v2<TQ, TG, nv_bfloat16>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        case ETensorDType::FP16:
            launch_bwd_v2<TQ, TG, half>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_backward_v2: unsupported beta dtype");
    }
}

template<typename TQ>
void dispatch_bwd_v2_g(
    Tensor& d_q, Tensor& d_k, Tensor& d_v, Tensor& d_g, Tensor& d_beta,
    Tensor& d_initial_state, const Tensor& d_out, const Tensor* d_final_state,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints, Tensor& workspace, bool skip_checkpoint,
    cudaStream_t stream) {
    switch (g.DType) {
        case ETensorDType::FP32:
            dispatch_bwd_v2_beta<TQ, float>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        case ETensorDType::BF16:
            dispatch_bwd_v2_beta<TQ, nv_bfloat16>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        case ETensorDType::FP16:
            dispatch_bwd_v2_beta<TQ, half>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_backward_v2: unsupported g dtype");
    }
}


} // namespace

// ============================================================================
// Public API
// ============================================================================

void gated_delta_rule_chunk_backward_v2(
    Tensor& d_q,
    Tensor& d_k,
    Tensor& d_v,
    Tensor& d_g,
    Tensor& d_beta,
    Tensor& d_initial_state,
    const Tensor& d_out,
    const Tensor* d_final_state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints,
    Tensor& workspace,
    bool skip_checkpoint,
    cudaStream_t stream) {

    switch (q.DType) {
        case ETensorDType::FP32:
            dispatch_bwd_v2_g<float>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); break;
        case ETensorDType::BF16:
            dispatch_bwd_v2_g<nv_bfloat16>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); break;
        case ETensorDType::FP16:
            dispatch_bwd_v2_g<half>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); break;
        default:
            throw std::logic_error("gated_delta_rule_chunk_backward_v2: unsupported q dtype");
    }
}
