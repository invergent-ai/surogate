// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Simple elementwise activation kernels (ReLU^2, SiLU, GeLU)

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "utilities/utils.h"
#include "utilities/tensor.h"

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__global__ void relu2_forward_kernel(T* out, const T* inp, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float y = x > 0.0f ? x * x : 0.0f;
    out[idx] = from_float<T>(y);
}

template <typename T>
__global__ void relu2_backward_kernel(T* dinp, const T* inp, const T* dout, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float dy = to_float(dout[idx]);
    float dx = x > 0.0f ? (2.0f * x * dy) : 0.0f;
    dinp[idx] = from_float<T>(dx);
}

template <typename T>
__device__ __forceinline__ float sigmoid(float x) {
    // Numerically stable sigmoid that avoids overflow in expf().
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ float silu_grad(float x) {
    // d/dx [x * sigmoid(x)].
    // Handle infinities explicitly to avoid inf*0 -> NaN.
    if (!isfinite(x)) {
        return x > 0.0f ? 1.0f : 0.0f;
    }
    float s = sigmoid<float>(x);
    float ds = s * (1.0f - s);
    return fmaf(x, ds, s);  // s + x * ds
}

template <typename T>
__device__ __forceinline__ float gelu_tanh(float x) {
    // GeLU tanh approximation (HF gelu_pytorch_tanh)
    constexpr float k0 = 0.7978845608f;  // sqrt(2/pi)
    constexpr float k1 = 0.044715f;
    float x3 = x * x * x;
    float tanh_out = tanhf(k0 * (x + k1 * x3));
    return 0.5f * x * (1.0f + tanh_out);
}

template <typename T>
__device__ __forceinline__ float gelu_tanh_grad(float x) {
    // Derivative of tanh-approx GeLU.
    constexpr float k0 = 0.7978845608f;  // sqrt(2/pi)
    constexpr float k1 = 0.044715f;
    float x2 = x * x;
    float x3 = x2 * x;
    float u = k0 * (x + k1 * x3);
    float t = tanhf(u);
    float dt = (1.0f - t * t) * k0 * (1.0f + 3.0f * k1 * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * dt;
}

template <typename T>
__global__ void softplus_forward_kernel(T* out, const T* inp, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    // softplus(x) = log(1 + exp(x)); linear above the threshold like
    // torch.nn.functional.softplus (beta=1, threshold=20).
    float y = (x > 20.0f) ? x : log1pf(expf(x));
    out[idx] = from_float<T>(y);
}

template <typename T>
__global__ void softplus_backward_kernel(T* dinp, const T* inp, const T* dout, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float dy = to_float(dout[idx]);
    // d/dx softplus(x) = sigmoid(x); 1 in the linear region.
    float g = (x > 20.0f) ? 1.0f : sigmoid<T>(x);
    dinp[idx] = from_float<T>(dy * g);
}

template <typename T>
__global__ void silu_forward_kernel(T* out, const T* inp, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float s = sigmoid<T>(x);
    float y;
    if (!isfinite(x)) {
        // Asymptotic limits of SiLU.
        y = x > 0.0f ? x : 0.0f;
    } else {
        y = x * s;
    }
    out[idx] = from_float<T>(y);
}


template <typename T>
__global__ void gelu_exact_forward_kernel(T* out, const T* inp, long n) {
    const long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = to_float(inp[i]);
    out[i] = from_float<T>(0.5f * x * (1.0f + erff(x * 0.7071067811865475f)));
}

template <typename T>
__global__ void gelu_exact_backward_kernel(T* dx, const T* inp, const T* dy, long n) {
    const long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = to_float(inp[i]);
    const float derivative = 0.5f * (1.0f + erff(x * 0.7071067811865475f))
                           + x * expf(-0.5f * x * x) * 0.3989422804014327f;
    dx[i] = from_float<T>(to_float(dy[i]) * derivative);
}

template <typename T>
__global__ void gelu_forward_kernel(T* out, const T* inp, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    out[idx] = from_float<T>(gelu_tanh<T>(x));
}

template <typename T>
__global__ void silu_backward_kernel(T* dinp, const T* inp, const T* dout, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float dy = to_float(dout[idx]);
    float grad = silu_grad(x);
    float dx = dy * grad;
    dinp[idx] = from_float<T>(dx);
}

template <typename T>
__global__ void sigmoid_forward_kernel(T* out, const T* inp, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    out[idx] = from_float<T>(sigmoid<T>(x));
}

template <typename T>
__global__ void sigmoid_backward_kernel(T* dinp, const T* inp, const T* dout, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float dy = to_float(dout[idx]);
    float s = sigmoid<T>(x);
    dinp[idx] = from_float<T>(dy * s * (1.0f - s));
}

template <typename T>
__global__ void gelu_backward_kernel(T* dinp, const T* inp, const T* dout, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = to_float(inp[idx]);
    float dy = to_float(dout[idx]);
    float dx = dy * gelu_tanh_grad<T>(x);
    dinp[idx] = from_float<T>(dx);
}

template <typename T>
void launch_relu2_forward(T* out, const T* inp, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    relu2_forward_kernel<T><<<grid, block, 0, stream>>>(out, inp, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_relu2_backward(T* dinp, const T* inp, const T* dout, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    relu2_backward_kernel<T><<<grid, block, 0, stream>>>(dinp, inp, dout, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_silu_forward(T* out, const T* inp, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    silu_forward_kernel<T><<<grid, block, 0, stream>>>(out, inp, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_softplus_forward(T* out, const T* inp, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    softplus_forward_kernel<T><<<grid, block, 0, stream>>>(out, inp, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_softplus_backward(T* dinp, const T* inp, const T* dout, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    softplus_backward_kernel<T><<<grid, block, 0, stream>>>(dinp, inp, dout, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_gelu_forward(T* out, const T* inp, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    gelu_forward_kernel<T><<<grid, block, 0, stream>>>(out, inp, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_silu_backward(T* dinp, const T* inp, const T* dout, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    silu_backward_kernel<T><<<grid, block, 0, stream>>>(dinp, inp, dout, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_sigmoid_forward(T* out, const T* inp, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    sigmoid_forward_kernel<T><<<grid, block, 0, stream>>>(out, inp, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_sigmoid_backward(T* dinp, const T* inp, const T* dout, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    sigmoid_backward_kernel<T><<<grid, block, 0, stream>>>(dinp, inp, dout, n);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_gelu_backward(T* dinp, const T* inp, const T* dout, long n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    gelu_backward_kernel<T><<<grid, block, 0, stream>>>(dinp, inp, dout, n);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

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

void gelu_forward(float* out, const float* inp, long n, cudaStream_t stream) {
    launch_gelu_forward(out, inp, n, stream);
}

void gelu_forward(nv_bfloat16* out, const nv_bfloat16* inp, long n, cudaStream_t stream) {
    launch_gelu_forward(out, inp, n, stream);
}

void silu_backward(float* dinp, const float* inp, const float* dout, long n, cudaStream_t stream) {
    launch_silu_backward(dinp, inp, dout, n, stream);
}

void silu_backward(nv_bfloat16* dinp, const nv_bfloat16* inp, const nv_bfloat16* dout, long n, cudaStream_t stream) {
    launch_silu_backward(dinp, inp, dout, n, stream);
}

void sigmoid_forward(float* out, const float* inp, long n, cudaStream_t stream) {
    launch_sigmoid_forward(out, inp, n, stream);
}

void sigmoid_forward(nv_bfloat16* out, const nv_bfloat16* inp, long n, cudaStream_t stream) {
    launch_sigmoid_forward(out, inp, n, stream);
}

void sigmoid_backward(float* dinp, const float* inp, const float* dout, long n, cudaStream_t stream) {
    launch_sigmoid_backward(dinp, inp, dout, n, stream);
}

void sigmoid_backward(nv_bfloat16* dinp, const nv_bfloat16* inp, const nv_bfloat16* dout, long n, cudaStream_t stream) {
    launch_sigmoid_backward(dinp, inp, dout, n, stream);
}

void softplus_forward(float* out, const float* inp, long n, cudaStream_t stream) {
    launch_softplus_forward(out, inp, n, stream);
}

void softplus_forward(nv_bfloat16* out, const nv_bfloat16* inp, long n, cudaStream_t stream) {
    launch_softplus_forward(out, inp, n, stream);
}

void softplus_backward(float* dinp, const float* inp, const float* dout, long n, cudaStream_t stream) {
    launch_softplus_backward(dinp, inp, dout, n, stream);
}

void softplus_backward(nv_bfloat16* dinp,
                       const nv_bfloat16* inp,
                       const nv_bfloat16* dout,
                       long n,
                       cudaStream_t stream) {
    launch_softplus_backward(dinp, inp, dout, n, stream);
}

void gelu_backward(float* dinp, const float* inp, const float* dout, long n, cudaStream_t stream) {
    launch_gelu_backward(dinp, inp, dout, n, stream);
}

void gelu_backward(nv_bfloat16* dinp, const nv_bfloat16* inp, const nv_bfloat16* dout, long n, cudaStream_t stream) {
    launch_gelu_backward(dinp, inp, dout, n, stream);
}

void gelu_exact_forward(Tensor& out, const Tensor& inp, long n, cudaStream_t stream) {
    if (n == 0) return;
    if (out.DType != inp.DType) throw std::logic_error("gelu_exact_forward: dtype mismatch");
    if (inp.DType == ETensorDType::BF16) {
        gelu_exact_forward_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), n);
    } else if (inp.DType == ETensorDType::FP32) {
        gelu_exact_forward_kernel<<<(n + 255) / 256, 256, 0, stream>>>(out.get<float>(), inp.get<float>(), n);
    } else throw std::logic_error("gelu_exact_forward: unsupported dtype");
    CUDA_CHECK(cudaGetLastError());
}

void gelu_exact_backward(Tensor& dx, const Tensor& inp, const Tensor& dy, long n, cudaStream_t stream) {
    if (n == 0) return;
    if (dx.DType != inp.DType || dy.DType != inp.DType)
        throw std::logic_error("gelu_exact_backward: dtype mismatch");
    if (inp.DType == ETensorDType::BF16) {
        gelu_exact_backward_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            dx.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), dy.get<nv_bfloat16>(), n);
    } else if (inp.DType == ETensorDType::FP32) {
        gelu_exact_backward_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            dx.get<float>(), inp.get<float>(), dy.get<float>(), n);
    } else throw std::logic_error("gelu_exact_backward: unsupported dtype");
    CUDA_CHECK(cudaGetLastError());
}
