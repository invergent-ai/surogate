// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_rule_v2.cuh"
#include "gdr_fwd_launchers.h"

namespace {

// ============================================================================
// Forward multi-kernel: State propagation (sequential)
// Grid: dim3(B, H). Propagates state h across chunks using precomputed u, w.
// Saves vnew_pre to workspace for output kernel, checkpoints for backward.
// ============================================================================
template<typename TQ>
__global__ void gdr_fwd_state_wmma(
    float* __restrict__ final_state,
    float* __restrict__ state_scratch,
    float* __restrict__ fwd_checkpoints,
    float* __restrict__ fwd_workspace,
    const float* __restrict__ initial_state,
    int fwd_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, int v_tile_max)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tile_idx = blockIdx.z;
    const int bh = b * H + h;
    const long kv = (long)Kdim * Vdim;
    const int Lp = kMaxC;
    const int v_tile = (v_tile_max > 0) ? min(Vdim, v_tile_max) : min(Vdim, 64);
    const int v0 = tile_idx * v_tile;
    if (v0 >= Vdim) {
        return;
    }
    const int vt = min(v_tile, Vdim - v0);

    FwdWorkspaceLayout fwl = make_fwd_ws(Lp, Kdim, Vdim);

    extern __shared__ char smem_raw[];
    float* scratch   = (float*)smem_raw;                            // [K*Vtile] and [Lp*Vtile]
    TQ*    buf_w     = (TQ*)(scratch + Kdim * v_tile);              // [Lp*K]
    TQ*    buf_k     = buf_w + Lp * Kdim;                           // [Lp*K]
    TQ*    buf_h     = buf_k + Lp * Kdim;                           // [K*Vtile]
    TQ*    buf_vnp   = buf_h + Kdim * v_tile;                       // [Lp*Vtile]
    float* smem_gcum = (float*)(buf_vnp + Lp * v_tile);             // [Lp]
    float* smem_e_last = smem_gcum + Lp;                            // [Lp]

    float* state_data = state_scratch + bh * kv;

    // Init only this V-tile slice of state in global scratch.
    for (int idx = tid; idx < Kdim * vt; idx += nthr) {
        const int kk = idx / vt;
        const int vv = idx % vt;
        const long s_idx = static_cast<long>(kk) * Vdim + (v0 + vv);
        state_data[s_idx] = initial_state ? initial_state[bh * kv + s_idx] : 0.0f;
    }
    __syncthreads();

    // Save checkpoint[0] for this V-tile slice.
    if (fwd_checkpoints) {
        float* cp_base = fwd_checkpoints + bh * (long)(num_chunks + 1) * kv;
        for (int idx = tid; idx < Kdim * vt; idx += nthr) {
            const int kk = idx / vt;
            const int vv = idx % vt;
            const long s_idx = static_cast<long>(kk) * Vdim + (v0 + vv);
            cp_base[s_idx] = state_data[s_idx];
        }
    }
    __syncthreads();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);
        const int block_id = bh * num_chunks + chunk;
        float* ws = fwd_workspace + (long)block_id * fwd_ws_stride;

        // Load w and gcum from workspace (k is loaded later so we can alias memory).
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf_w[idx] = from_float<TQ>(bf16_trunc<TQ>(ws[fwl.w_off + idx]));
        for (int idx = tid; idx < Lp; idx += nthr)
            smem_gcum[idx] = ws[fwl.gcum_off + idx];
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf_k[idx] = from_float<TQ>(bf16_trunc<TQ>(ws[fwl.k_off + idx]));
        __syncthreads();
        const float g_last = (L > 0) ? smem_gcum[L - 1] : 0.0f;
        const float eg = expf(g_last);
        for (int i = tid; i < Lp; i += nthr) {
            smem_e_last[i] = (i < L) ? expf(g_last - smem_gcum[i]) : 0.0f;
        }
        __syncthreads();

        // Load state tile and cast to bf16 for tensor-core matmul.
        for (int idx = tid; idx < Kdim * vt; idx += nthr) {
            const int kk = idx / vt;
            const int vv = idx % vt;
            const long s_idx = static_cast<long>(kk) * Vdim + (v0 + vv);
            buf_h[idx] = from_float<TQ>(bf16_trunc<TQ>(state_data[s_idx]));
        }
        __syncthreads();

        // wh = w @ h_tile -> scratch[Lp, vt]
        wmma_nn<TQ>(buf_w, Kdim, buf_h, vt, scratch, vt, Lp, vt, Kdim);
        __syncthreads();

        // vnew_pre tile and v_scaled tile.
        for (int idx = tid; idx < Lp * vt; idx += nthr) {
            const int i = idx / vt;
            const int vv = idx % vt;
            const long ws_idx = static_cast<long>(i) * Vdim + (v0 + vv);
            const float u_val = ws[fwl.u_off + ws_idx];
            const float vnew = bf16_trunc<TQ>(u_val - scratch[idx]);
            ws[fwl.vnew_pre_off + ws_idx] = vnew;
            buf_vnp[idx] = from_float<TQ>(bf16_trunc<TQ>(vnew * smem_e_last[i]));
        }
        __syncthreads();

        // delta_tile = k^T @ v_scaled_tile -> scratch[K, vt]
        wmma_tn<TQ>(buf_k, Kdim, buf_vnp, vt, scratch, vt, Kdim, vt, Lp);
        __syncthreads();

        // State update in FP32 global storage for this V-tile slice.
        for (int idx = tid; idx < Kdim * vt; idx += nthr) {
            const int kk = idx / vt;
            const int vv = idx % vt;
            const long s_idx = static_cast<long>(kk) * Vdim + (v0 + vv);
            const float old_h = state_data[s_idx];
            state_data[s_idx] = eg * old_h + scratch[idx];
        }
        __syncthreads();

        // Save checkpoint
        if (fwd_checkpoints) {
            const int chunk_idx = chunk + 1;
            float* cp = fwd_checkpoints + bh * (long)(num_chunks + 1) * kv + (long)chunk_idx * kv;
            for (int idx = tid; idx < Kdim * vt; idx += nthr) {
                const int kk = idx / vt;
                const int vv = idx % vt;
                const long s_idx = static_cast<long>(kk) * Vdim + (v0 + vv);
                cp[s_idx] = state_data[s_idx];
            }
            __syncthreads();
        }
    }

    // Write final state slice for this V-tile.
    for (int idx = tid; idx < Kdim * vt; idx += nthr) {
        const int kk = idx / vt;
        const int vv = idx % vt;
        const long s_idx = static_cast<long>(kk) * Vdim + (v0 + vv);
        final_state[bh * kv + s_idx] = state_data[s_idx];
    }
}

} // namespace

// Launch wrapper
template<typename TQ>
void launch_gdr_fwd_state(
    float* final_state, float* state_scratch,
    float* fwd_checkpoints, float* fwd_workspace,
    const float* initial_state,
    int fwd_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, int v_tile,
    int threads, std::size_t smem, cudaStream_t stream)
{
    const int num_v_tiles = (Vdim + v_tile - 1) / v_tile;
    CUDA_CHECK(cudaFuncSetAttribute(
        gdr_fwd_state_wmma<TQ>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    gdr_fwd_state_wmma<TQ><<<dim3(B, H, num_v_tiles), threads, smem, stream>>>(
        final_state, state_scratch, fwd_checkpoints, fwd_workspace,
        initial_state, fwd_ws_stride,
        Tlen, H, Kdim, Vdim, num_chunks, chunk_size, v_tile);
    CUDA_CHECK(cudaGetLastError());
}

// Explicit template instantiations
template void launch_gdr_fwd_state<nv_bfloat16>(
    float*, float*, float*, float*, const float*,
    int, int, int, int, int, int, int, int, int,
    int, std::size_t, cudaStream_t);
template void launch_gdr_fwd_state<half>(
    float*, float*, float*, float*, const float*,
    int, int, int, int, int, int, int, int, int,
    int, std::size_t, cudaStream_t);
