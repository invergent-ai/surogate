// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Simple elementwise activation kernels (ReLU^2, SiLU)

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "utilities/utils.h"

namespace {

template<typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

template<>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

template<typename T>
__device__ __forceinline__ T from_float(float v);

template<>
__device__ __forceinline__ float from_float<float>(float v) {
    return v;
}

template<>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template<typename T>
__global__ void relu2_forward_kernel(T* out, const T* inp, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float y = x > 0.0f ? x * x : 0.0f;
    out[idx] = from_float<T>(y);
}

template<typename T>
__global__ void relu2_backward_kernel(T* dinp, const T* inp, const T* dout, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float dy = to_float(dout[idx]);
    float dx = x > 0.0f ? (2.0f * x * dy) : 0.0f;
    dinp[idx] = from_float<T>(dx);
}

template<typename T>
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template<typename T>
__global__ void silu_forward_kernel(T* out, const T* inp, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float s = sigmoid<T>(x);
    out[idx] = from_float<T>(x * s);
}

template<typename T>
__global__ void silu_backward_kernel(T* dinp, const T* inp, const T* dout, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float dy = to_float(dout[idx]);
    float s = sigmoid<T>(x);
    float dx = dy * (s * (1.0f + x * (1.0f - s)));
    dinp[idx] = from_float<T>(dx);
}

template<typename T>
void launch_relu2_forward(T* out, const T* inp, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    relu2_forward_kernel<T><<<grid, block, 0, stream>>>(out, inp, n);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_relu2_backward(T* dinp, const T* inp, const T* dout, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    relu2_backward_kernel<T><<<grid, block, 0, stream>>>(dinp, inp, dout, n);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_silu_forward(T* out, const T* inp, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    silu_forward_kernel<T><<<grid, block, 0, stream>>>(out, inp, n);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_silu_backward(T* dinp, const T* inp, const T* dout, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    silu_backward_kernel<T><<<grid, block, 0, stream>>>(dinp, inp, dout, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void relu2_forward(float* out, const float* inp, long n, cudaStream_t stream) {
    launch_relu2_forward(out, inp, n, stream);
}

void relu2_forward(nv_bfloat16* out, const nv_bfloat16* inp, long n, cudaStream_t stream) {
    launch_relu2_forward(out, inp, n, stream);
}

void relu2_backward(float* dinp, const float* inp, const float* dout, long n, cudaStream_t stream) {
    launch_relu2_backward(dinp, inp, dout, n, stream);
}

void relu2_backward(nv_bfloat16* dinp, const nv_bfloat16* inp, const nv_bfloat16* dout, long n, cudaStream_t stream) {
    launch_relu2_backward(dinp, inp, dout, n, stream);
}

void silu_forward(float* out, const float* inp, long n, cudaStream_t stream) {
    launch_silu_forward(out, inp, n, stream);
}

void silu_forward(nv_bfloat16* out, const nv_bfloat16* inp, long n, cudaStream_t stream) {
    launch_silu_forward(out, inp, n, stream);
}

void silu_backward(float* dinp, const float* inp, const float* dout, long n, cudaStream_t stream) {
    launch_silu_backward(dinp, inp, dout, n, stream);
}

void silu_backward(nv_bfloat16* dinp, const nv_bfloat16* inp, const nv_bfloat16* dout, long n, cudaStream_t stream) {
    launch_silu_backward(dinp, inp, dout, n, stream);
}
