// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_KERNELS_GDR_FWD_LAUNCHERS_H
#define SUROGATE_SRC_KERNELS_GDR_FWD_LAUNCHERS_H

#include <cuda_runtime.h>
#include <cstddef>

// Launch wrappers for the three forward sub-kernels.
// Each is compiled in its own .cu translation unit for parallel compilation.

template<typename TQ, typename TG, typename TB>
void launch_gdr_fwd_precompute(
    float* fwd_workspace,
    const TQ* k, const TQ* v, const TG* g, const TB* beta,
    int fwd_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream);

template<typename TQ>
void launch_gdr_fwd_state(
    float* final_state, float* state_scratch,
    float* fwd_checkpoints, float* fwd_workspace,
    const float* initial_state,
    int fwd_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, int v_tile,
    int threads, std::size_t smem, cudaStream_t stream);

template<typename TQ>
void launch_gdr_fwd_output(
    TQ* out, const TQ* q,
    const float* fwd_checkpoints, const float* fwd_workspace,
    int fwd_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, int v_tile, float scale,
    bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream);

#endif  // SUROGATE_SRC_KERNELS_GDR_FWD_LAUNCHERS_H
