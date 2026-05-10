// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Token-row compaction utilities for the lm_head loss path.
//
// Skips the lm_head matmul + cross-entropy on rows whose target == -100
// (e.g. prompt tokens in completion-only SFT, masked tokens in GRPO).
// Inspired by TRL's chunked-NLL loss (huggingface/trl#5575) but operates on
// the device-side hidden states; the index compaction + gather/scatter run
// inline inside `dispatch_fused_lm_head_loss(_backward)`.

#include "kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "utilities/utils.h"

namespace {

// Single-block exclusive scan over targets[BT] producing valid_idx[n_valid] +
// n_valid (host int returned via cudaMemcpy after the kernel).
// Assumes BT <= 65536; for the typical packed-sequence shapes (BT <= 32k) one
// block of 1024 threads handles it iteratively.
__global__ void compact_valid_indices_kernel(const int* __restrict__ targets,
                                             int* __restrict__ valid_idx,
                                             int* __restrict__ n_valid_out,
                                             int BT) {
    int tid = static_cast<int>(threadIdx.x);
    int counter = 0;
    if (tid == 0) {
        n_valid_out[0] = 0;
    }
    __syncthreads();

    // Block-wide atomic compaction. Order of valid rows is non-deterministic
    // but irrelevant because each row's loss/grad is row-local.
    for (int i = tid; i < BT; i += blockDim.x) {
        if (targets[i] != -100) {
            int slot = atomicAdd(n_valid_out, 1);
            valid_idx[slot] = i;
            ++counter;
        }
    }
}

template <class T>
__global__ void gather_rows_kernel(const T* __restrict__ src,
                                   const int* __restrict__ valid_idx,
                                   T* __restrict__ dst,
                                   int n_valid,
                                   int C) {
    int row = static_cast<int>(blockIdx.x);
    if (row >= n_valid) return;
    int src_row = valid_idx[row];
    const T* sp = src + static_cast<int64_t>(src_row) * C;
    T* dp = dst + static_cast<int64_t>(row) * C;
    for (int c = static_cast<int>(threadIdx.x); c < C; c += static_cast<int>(blockDim.x)) {
        dp[c] = sp[c];
    }
}

__global__ void
gather_int_kernel(const int* __restrict__ src, const int* __restrict__ valid_idx, int* __restrict__ dst, int n_valid) {
    int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= n_valid) return;
    dst[idx] = src[valid_idx[idx]];
}

__global__ void gather_float_kernel(const float* __restrict__ src,
                                    const int* __restrict__ valid_idx,
                                    float* __restrict__ dst,
                                    int n_valid) {
    int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= n_valid) return;
    dst[idx] = src[valid_idx[idx]];
}

template <class T>
__global__ void scatter_rows_zerofill_kernel(const T* __restrict__ src,
                                             const int* __restrict__ valid_idx,
                                             T* __restrict__ dst,
                                             int BT,
                                             int n_valid,
                                             int C) {
    // Phase 1: zero-fill all BT rows. blockIdx.x = dst row.
    int row = static_cast<int>(blockIdx.x);
    if (row >= BT) return;
    T* dp = dst + static_cast<int64_t>(row) * C;
    for (int c = static_cast<int>(threadIdx.x); c < C; c += static_cast<int>(blockDim.x)) {
        dp[c] = (T)0;
    }
}

template <class T>
__global__ void scatter_rows_writeback_kernel(const T* __restrict__ src,
                                              const int* __restrict__ valid_idx,
                                              T* __restrict__ dst,
                                              int n_valid,
                                              int C) {
    int row = static_cast<int>(blockIdx.x);
    if (row >= n_valid) return;
    int dst_row = valid_idx[row];
    const T* sp = src + static_cast<int64_t>(row) * C;
    T* dp = dst + static_cast<int64_t>(dst_row) * C;
    for (int c = static_cast<int>(threadIdx.x); c < C; c += static_cast<int>(blockDim.x)) {
        dp[c] = sp[c];
    }
}

// Add per-valid-row contributions into the full loss[BT] buffer at the
// row's original position. -100 rows are left untouched (matching the
// existing CE kernel contract where loss[BT] accumulates across micro-steps
// and -100 rows never contribute).
__global__ void scatter_loss_add_kernel(const float* __restrict__ src,
                                        const int* __restrict__ valid_idx,
                                        float* __restrict__ dst,
                                        int n_valid) {
    int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= n_valid) return;
    atomicAdd(&dst[valid_idx[idx]], src[idx]);
}

}  // namespace

void compact_valid_indices(const int* targets, int* valid_idx, int* n_valid_dev, int BT, cudaStream_t stream) {
    int threads = 1024;
    compact_valid_indices_kernel<<<1, threads, 0, stream>>>(targets, valid_idx, n_valid_dev, BT);
    CUDA_CHECK(cudaGetLastError());
}

void gather_rows_bf16(const nv_bfloat16* src,
                      const int* valid_idx,
                      nv_bfloat16* dst,
                      int n_valid,
                      int C,
                      cudaStream_t stream) {
    if (n_valid <= 0) return;
    int threads = 256;
    gather_rows_kernel<nv_bfloat16><<<n_valid, threads, 0, stream>>>(src, valid_idx, dst, n_valid, C);
    CUDA_CHECK(cudaGetLastError());
}

void gather_rows_fp32(const float* src, const int* valid_idx, float* dst, int n_valid, int C, cudaStream_t stream) {
    if (n_valid <= 0) return;
    int threads = 256;
    gather_rows_kernel<float><<<n_valid, threads, 0, stream>>>(src, valid_idx, dst, n_valid, C);
    CUDA_CHECK(cudaGetLastError());
}

void gather_int(const int* src, const int* valid_idx, int* dst, int n_valid, cudaStream_t stream) {
    if (n_valid <= 0) return;
    int threads = 256;
    int blocks = (n_valid + threads - 1) / threads;
    gather_int_kernel<<<blocks, threads, 0, stream>>>(src, valid_idx, dst, n_valid);
    CUDA_CHECK(cudaGetLastError());
}

void gather_float(const float* src, const int* valid_idx, float* dst, int n_valid, cudaStream_t stream) {
    if (n_valid <= 0) return;
    int threads = 256;
    int blocks = (n_valid + threads - 1) / threads;
    gather_float_kernel<<<blocks, threads, 0, stream>>>(src, valid_idx, dst, n_valid);
    CUDA_CHECK(cudaGetLastError());
}

void scatter_rows_zerofill_bf16(const nv_bfloat16* src,
                                const int* valid_idx,
                                nv_bfloat16* dst,
                                int BT,
                                int n_valid,
                                int C,
                                cudaStream_t stream) {
    int threads = 256;
    scatter_rows_zerofill_kernel<nv_bfloat16><<<BT, threads, 0, stream>>>(src, valid_idx, dst, BT, n_valid, C);
    CUDA_CHECK(cudaGetLastError());
    if (n_valid > 0) {
        scatter_rows_writeback_kernel<nv_bfloat16><<<n_valid, threads, 0, stream>>>(src, valid_idx, dst, n_valid, C);
        CUDA_CHECK(cudaGetLastError());
    }
}

void scatter_rows_zerofill_fp32(const float* src,
                                const int* valid_idx,
                                float* dst,
                                int BT,
                                int n_valid,
                                int C,
                                cudaStream_t stream) {
    int threads = 256;
    scatter_rows_zerofill_kernel<float><<<BT, threads, 0, stream>>>(src, valid_idx, dst, BT, n_valid, C);
    CUDA_CHECK(cudaGetLastError());
    if (n_valid > 0) {
        scatter_rows_writeback_kernel<float><<<n_valid, threads, 0, stream>>>(src, valid_idx, dst, n_valid, C);
        CUDA_CHECK(cudaGetLastError());
    }
}

void scatter_loss_add(const float* src, const int* valid_idx, float* dst, int n_valid, cudaStream_t stream) {
    if (n_valid <= 0) return;
    int threads = 256;
    int blocks = (n_valid + threads - 1) / threads;
    scatter_loss_add_kernel<<<blocks, threads, 0, stream>>>(src, valid_idx, dst, n_valid);
    CUDA_CHECK(cudaGetLastError());
}
