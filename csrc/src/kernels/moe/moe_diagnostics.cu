// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_diagnostics.cu.

// ============================================================================
// Utility: sanitize non-finite values (NaN/Inf) in-place
// ============================================================================
template <typename T>
__global__ void sanitize_non_finite_kernel(T* data, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = static_cast<float>(data[idx]);
    if (!isfinite(v)) {
        data[idx] = static_cast<T>(0.0f);
    }
}

template <typename T>
static void sanitize_non_finite_impl(T* data, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    sanitize_non_finite_kernel<<<grid, block, 0, stream>>>(data, n);
    CUDA_CHECK(cudaGetLastError());
}

void sanitize_non_finite(nv_bfloat16* data, int n, cudaStream_t stream) {
    sanitize_non_finite_impl(data, n, stream);
}

void sanitize_non_finite(float* data, int n, cudaStream_t stream) {
    sanitize_non_finite_impl(data, n, stream);
}

// ============================================================================
// Utility: clamp absolute values in-place
// ============================================================================
template <typename T>
__global__ void clamp_abs_kernel(T* data, int n, float max_abs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = static_cast<float>(data[idx]);
    if (v > max_abs)
        v = max_abs;
    else if (v < -max_abs)
        v = -max_abs;
    data[idx] = static_cast<T>(v);
}

template <typename T>
static void clamp_abs_impl(T* data, int n, float max_abs, cudaStream_t stream) {
    if (n <= 0 || max_abs <= 0.0f) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    clamp_abs_kernel<<<grid, block, 0, stream>>>(data, n, max_abs);
    CUDA_CHECK(cudaGetLastError());
}

void clamp_abs(nv_bfloat16* data, int n, float max_abs, cudaStream_t stream) {
    clamp_abs_impl(data, n, max_abs, stream);
}

void clamp_abs(float* data, int n, float max_abs, cudaStream_t stream) {
    clamp_abs_impl(data, n, max_abs, stream);
}

// ============================================================================
// Utility: count non-finite values (NaN/Inf)
// ============================================================================
template <typename T>
__global__ void count_non_finite_kernel(int* out_count, const T* data, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = static_cast<float>(data[idx]);
    if (!isfinite(v)) {
        atomicAdd(out_count, 1);
    }
}

template <typename T>
static void count_non_finite_impl(int* out_count, const T* data, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    count_non_finite_kernel<<<grid, block, 0, stream>>>(out_count, data, n);
    CUDA_CHECK(cudaGetLastError());
}

void count_non_finite(int* out_count, const nv_bfloat16* data, int n, cudaStream_t stream) {
    count_non_finite_impl(out_count, data, n, stream);
}

void count_non_finite(int* out_count, const float* data, int n, cudaStream_t stream) {
    count_non_finite_impl(out_count, data, n, stream);
}

// ============================================================================
// Utility: count values with |x| > threshold (for diagnosing extreme grads).
// ============================================================================
template <typename T>
__global__ void count_above_threshold_kernel(int* out_count, const T* data, int n, float threshold) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = fabsf(static_cast<float>(data[idx]));
    if (isfinite(v) && v > threshold) {
        atomicAdd(out_count, 1);
    }
}

template <typename T>
static void count_above_threshold_impl(int* out_count, const T* data, int n, float threshold, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    count_above_threshold_kernel<<<grid, block, 0, stream>>>(out_count, data, n, threshold);
    CUDA_CHECK(cudaGetLastError());
}

void count_above_threshold(int* out_count, const nv_bfloat16* data, int n, float threshold, cudaStream_t stream) {
    count_above_threshold_impl(out_count, data, n, threshold, stream);
}

void count_above_threshold(int* out_count, const float* data, int n, float threshold, cudaStream_t stream) {
    count_above_threshold_impl(out_count, data, n, threshold, stream);
}

// ============================================================================
// Utility: count invalid indices
// ============================================================================
__global__ void count_invalid_indices_kernel(int* out_count, const int* indices, int n, int num_experts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int v = indices[idx];
    if (v < 0 || v >= num_experts) {
        atomicAdd(out_count, 1);
    }
}

void count_invalid_indices(int* out_count, const int* indices, int n, int num_experts, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    count_invalid_indices_kernel<<<grid, block, 0, stream>>>(out_count, indices, n, num_experts);
    CUDA_CHECK(cudaGetLastError());
}
