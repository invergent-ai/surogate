// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_rule_v2.cuh"
#include "gdr_bwd_launchers.h"

namespace {

constexpr int kPhase2VTile = 64;

// ============================================================================
// Multi-kernel backward — Phase 2: sequential ds propagation
//
// Grid: (B*H, 1, 1), each block processes one (b,h) sequentially over chunks.
// Iterates chunks backward. For each chunk:
//   - Reads DU, W, VNEW, DHT1 from Phase 1 workspace
//   - Computes ds-dependent contributions: k@ds → DU_state, VNEW@ds^T → DK_state
//   - Computes d_pre = DU*e_i, DW = -d_pre@h^T, dh_in updates
//   - Propagates ds backward
// ============================================================================

template<typename TQ>
__global__ void gdr_bwd_phase2_wmma(
    const float* __restrict__ checkpoints,
    const float* __restrict__ d_final_state,
    float* __restrict__ d_initial_state,
    float* __restrict__ chunk_workspace,
    float* __restrict__ dh_storage,       // [B*H*num_chunks, K*V] per-chunk dh
    int chunk_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int bh = blockIdx.x;
    const int tile_idx = blockIdx.y;
    const long kv = (long)Kdim * Vdim;
    const long kk = (long)Kdim * Kdim;
    const int Lp = kMaxC;
    ChunkWorkspaceLayout cwl = make_chunk_ws<TQ>(Lp, Kdim, Vdim);

    float* ds = d_initial_state + (long)bh * kv;

    const int v_tile_max = (Vdim > kPhase2VTile) ? kPhase2VTile : Vdim;
    const int v0 = tile_idx * v_tile_max;
    if (v0 >= Vdim) {
        return;
    }
    const int vt = min(v_tile_max, Vdim - v0);
    const long kvt_max = static_cast<long>(Kdim) * v_tile_max;

    // Shared memory: scratch1[K×Vtile] + buf1[K×K bf16] + buf2[K×Vtile bf16]
    extern __shared__ char smem_raw[];
    float* scratch1 = reinterpret_cast<float*>(smem_raw);        // [K×Vtile]
    TQ*    buf1     = reinterpret_cast<TQ*>(scratch1 + kvt_max); // [K×K] (C_bf16)
    TQ*    buf2     = buf1 + kk;                                 // [K×Vtile] (ds_bf16)

    // Seed ds from d_final_state for this V-tile slice.
    for (int idx = tid; idx < Kdim * vt; idx += nthr) {
        const int kk_row = idx / vt;
        const int vv = idx % vt;
        const long ds_idx = static_cast<long>(kk_row) * Vdim + (v0 + vv);
        ds[ds_idx] = d_final_state ? d_final_state[(long)bh * kv + ds_idx] : 0.0f;
    }
    __syncthreads();

    for (int chunk = num_chunks - 1; chunk >= 0; --chunk) {
        // Store entering ds for this chunk and V-tile slice (Phase 3 needs it).
        float* dh_store = dh_storage + (long)(bh * num_chunks + chunk) * kv;
        for (int idx = tid; idx < Kdim * vt; idx += nthr) {
            const int kk_row = idx / vt;
            const int vv = idx % vt;
            const long ds_idx = static_cast<long>(kk_row) * Vdim + (v0 + vv);
            dh_store[ds_idx] = ds[ds_idx];
        }
        __syncthreads();

        // Per-chunk workspace
        const int chunk_block_id = bh * num_chunks + chunk;
        float* cws = chunk_workspace + (long)chunk_block_id * chunk_ws_stride;
        float* DHT1 = cws + cwl.DHT1_off;   // already corrected by Phase 1
        const float* C_mat = cws + cwl.C_off;
        float  eg_last = *(cws + cwl.EG_off);

        // Load C (fp32) and convert to TQ into buf1 [K×K]
        for (long idx = tid; idx < kk; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(C_mat[idx]));
        const long kvt = static_cast<long>(Kdim) * vt;
        for (long idx = tid; idx < kvt; idx += nthr) {
            const int kk_row = static_cast<int>(idx / vt);
            const int vv = static_cast<int>(idx % vt);
            const long ds_idx = static_cast<long>(kk_row) * Vdim + (v0 + vv);
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(ds[ds_idx]));
        }
        __syncthreads();

        wmma_nn<TQ>(buf1, Kdim, buf2, vt, scratch1, vt, Kdim, vt, Kdim);
        __syncthreads();

        for (long idx = tid; idx < kvt; idx += nthr) {
            const int kk_row = static_cast<int>(idx / vt);
            const int vv = static_cast<int>(idx % vt);
            const long ds_idx = static_cast<long>(kk_row) * Vdim + (v0 + vv);
            ds[ds_idx] = ds[ds_idx] * eg_last + DHT1[ds_idx] - scratch1[idx];
        }
        __syncthreads();
    }
}

} // namespace

template<typename TQ>
void launch_gdr_bwd_phase2(
    const float* checkpoints,
    const float* d_final_state,
    float* d_initial_state,
    float* chunk_workspace,
    float* dh_storage,
    int chunk_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size,
    int threads, std::size_t smem, cudaStream_t stream)
{
    (void)smem;
    const int v_tile = (Vdim > kPhase2VTile) ? kPhase2VTile : Vdim;
    const int num_v_tiles = (Vdim + v_tile - 1) / v_tile;
    const std::size_t phase2_smem_tiled =
        static_cast<std::size_t>(Kdim) * v_tile * sizeof(float)
        + static_cast<std::size_t>(Kdim) * Kdim * sizeof(TQ)
        + static_cast<std::size_t>(Kdim) * v_tile * sizeof(TQ);

    CUDA_CHECK(cudaFuncSetAttribute(
        gdr_bwd_phase2_wmma<TQ>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(phase2_smem_tiled)));

    gdr_bwd_phase2_wmma<TQ><<<dim3(B * H, num_v_tiles), threads, phase2_smem_tiled, stream>>>(
        checkpoints, d_final_state, d_initial_state,
        chunk_workspace, dh_storage,
        chunk_ws_stride,
        Tlen, H, Kdim, Vdim, num_chunks, chunk_size);
    CUDA_CHECK(cudaGetLastError());
}

template void launch_gdr_bwd_phase2<nv_bfloat16>(
    const float*, const float*, float*, float*, float*,
    int, int, int, int, int, int, int, int,
    int, std::size_t, cudaStream_t);
template void launch_gdr_bwd_phase2<half>(
    const float*, const float*, float*, float*, float*,
    int, int, int, int, int, int, int, int,
    int, std::size_t, cudaStream_t);
