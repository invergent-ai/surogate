// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_rule_v2.cuh"
#include "gdr_fwd_launchers.h"

namespace {

// ============================================================================
// Forward multi-kernel: Output computation (chunk-parallel)
// Grid: B*H*num_chunks. Computes output using precomputed S, vnew_pre
// and state checkpoints.
// ============================================================================
template<typename TQ>
__global__ void gdr_fwd_output_wmma(
    TQ* __restrict__ out,
    const TQ* __restrict__ q,
    const float* __restrict__ fwd_checkpoints,
    const float* __restrict__ fwd_workspace,
    int fwd_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, int v_tile_max, float scale,
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
    const int v_tile = (v_tile_max > 0) ? min(Vdim, v_tile_max) : min(Vdim, 64);

    const int cs = chunk * chunk_size;
    const int L = min(chunk_size, Tlen - cs);

    const float* ws = fwd_workspace + (long)block_id * fwd_ws_stride;
    FwdWorkspaceLayout fwl = make_fwd_ws(Lp, Kdim, Vdim);

    // h_in from checkpoint (state entering this chunk)
    const float* h_in = fwd_checkpoints + (long)bh * (num_chunks + 1) * kv + (long)chunk * kv;

    extern __shared__ char smem_raw[];
    float* scratch_v = (float*)smem_raw;                              // [Lp*Vtile]
    float* scratch_s = scratch_v + Lp * v_tile;                       // [Lp*Lp] then [Lp*Vtile]
    TQ*    smem_q    = (TQ*)(scratch_s + Lp * Lp);                    // [Lp*K]
    TQ*    smem_k    = smem_q + Lp * Kdim;                            // [Lp*K]
    TQ*    buf_h     = smem_k + Lp * Kdim;                            // [K*Vtile]
    TQ*    buf_S     = buf_h + Kdim * v_tile;                         // [Lp*Lp]
    TQ*    buf_vnp   = buf_S + Lp * Lp;                               // [Lp*Vtile]
    float* smem_gcum = (float*)(buf_vnp + Lp * v_tile);               // [Lp]
    float* smem_invq = smem_gcum + Lp;                                // [Lp]

    // Zero-fill q
    for (int idx = tid; idx < Lp * Kdim; idx += nthr)
        smem_q[idx] = from_float<TQ>(0.0f);
    for (int idx = tid; idx < Lp * Kdim; idx += nthr)
        smem_k[idx] = from_float<TQ>(0.0f);
    if (tid < Lp) smem_gcum[tid] = 0.0f;
    __syncthreads();

    // Load q from global
    for (int idx = tid; idx < L * Kdim; idx += nthr) {
        const int pos = idx / Kdim, kk = idx % Kdim;
        smem_q[pos * Kdim + kk] = q[(((long)b * Tlen + cs + pos) * H + h) * Kdim + kk];
    }
    // Load normalized k from workspace.
    for (int idx = tid; idx < Lp * Kdim; idx += nthr)
        smem_k[idx] = from_float<TQ>(bf16_trunc<TQ>(ws[fwl.k_off + idx]));

    // Load gcum from workspace
    for (int idx = tid; idx < Lp; idx += nthr)
        smem_gcum[idx] = ws[fwl.gcum_off + idx];
    __syncthreads();

    // L2 normalize q
    for (int pos = tid; pos < L; pos += nthr) {
        if (use_qk_l2norm_in_kernel) {
            float qn2 = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float qv = to_float(smem_q[pos * Kdim + kk]);
                qn2 += qv * qv;
            }
            smem_invq[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
        } else {
            smem_invq[pos] = 1.0f;
        }
    }
    __syncthreads();
    if (use_qk_l2norm_in_kernel) {
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim;
            smem_q[pos * Kdim + idx % Kdim] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_q[idx]) * smem_invq[pos]));
        }
        __syncthreads();
    }

    // Compute output in V tiles.
    for (int v0 = 0; v0 < Vdim; v0 += v_tile) {
        // Load h_in tile.
        for (int idx = tid; idx < Kdim * v_tile; idx += nthr) {
            const int kk = idx / v_tile;
            const int vv = idx % v_tile;
            const long h_idx = static_cast<long>(kk) * Vdim + (v0 + vv);
            buf_h[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[h_idx]));
        }
        __syncthreads();

        // term1 = q @ h_tile
        wmma_nn<TQ>(smem_q, Kdim, buf_h, v_tile, scratch_v, v_tile, Lp, v_tile, Kdim);
        __syncthreads();

        // S = bf16_trunc(exp(g_i-g_j) * (q @ k^T)) with causal mask.
        wmma_nt<TQ>(smem_q, Kdim, smem_k, Kdim, scratch_s, Lp, Lp, Lp, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp;
            const int j = idx % Lp;
            float val = 0.0f;
            if (i < L && j <= i) {
                val = bf16_trunc<TQ>(expf(smem_gcum[i] - smem_gcum[j]) * scratch_s[idx]);
            }
            buf_S[idx] = from_float<TQ>(val);
        }
        __syncthreads();

        // Load vnew_pre tile and compute term2 = S @ vnew_pre_tile.
        for (int idx = tid; idx < Lp * v_tile; idx += nthr) {
            const int i = idx / v_tile;
            const int vv = idx % v_tile;
            const long ws_idx = static_cast<long>(i) * Vdim + (v0 + vv);
            buf_vnp[idx] = from_float<TQ>(bf16_trunc<TQ>(ws[fwl.vnew_pre_off + ws_idx]));
        }
        __syncthreads();
        wmma_nn<TQ>(buf_S, Lp, buf_vnp, v_tile, scratch_s, v_tile, Lp, v_tile, Lp);
        __syncthreads();

        // Write tile: o = scale * (exp(gcum) * term1 + term2)
        for (int idx = tid; idx < L * v_tile; idx += nthr) {
            const int i = idx / v_tile;
            const int vv = idx % v_tile;
            const long oi = (((long)b * Tlen + cs + i) * H + h) * Vdim + (v0 + vv);
            out[oi] = from_float<TQ>(
                scale * (expf(smem_gcum[i]) * scratch_v[idx] + scratch_s[idx]));
        }
        __syncthreads();
    }
}

} // namespace

// Launch wrapper
template<typename TQ>
void launch_gdr_fwd_output(
    TQ* out, const TQ* q,
    const float* fwd_checkpoints, const float* fwd_workspace,
    int fwd_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, int v_tile, float scale,
    bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream)
{
    CUDA_CHECK(cudaFuncSetAttribute(
        gdr_fwd_output_wmma<TQ>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    gdr_fwd_output_wmma<TQ><<<B * H * num_chunks, threads, smem, stream>>>(
        out, q, fwd_checkpoints, fwd_workspace,
        fwd_ws_stride, Tlen, H, Kdim, Vdim,
        num_chunks, chunk_size, v_tile, scale, use_qk_l2norm_in_kernel);
    CUDA_CHECK(cudaGetLastError());
}

// Explicit template instantiations
template void launch_gdr_fwd_output<nv_bfloat16>(
    nv_bfloat16*, const nv_bfloat16*,
    const float*, const float*,
    int, int, int, int, int, int, int, int, int, float, bool,
    int, std::size_t, cudaStream_t);
template void launch_gdr_fwd_output<half>(
    half*, const half*,
    const float*, const float*,
    int, int, int, int, int, int, int, int, int, float, bool,
    int, std::size_t, cudaStream_t);
