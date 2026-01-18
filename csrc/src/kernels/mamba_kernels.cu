// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba2 kernels and wrappers (conv1d + selective scan + group RMSNorm).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "kernels/kernels.h"
#include "third_party/causal_conv1d/causal_conv1d.h"
#include "third_party/mamba/selective_scan/selective_scan.h"
#include "utilities/tensor.h"
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

template<>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
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

template<>
__device__ __forceinline__ half from_float<half>(float v) {
    return __float2half(v);
}

inline int div_up(long n, int d) {
    return static_cast<int>((n + d - 1) / d);
}

template<typename T>
__global__ void mamba_copy_gate_kernel(T* gate, const T* proj, long total, int D, int proj_size) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long d = idx % D;
    long bt = idx / D;
    gate[idx] = proj[bt * proj_size + d];
}

template<typename T>
__global__ void mamba_copy_conv_in_kernel(T* conv_in, const T* proj, long total, int Tlen, int D, int conv_dim, int proj_size) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long t = idx % Tlen;
    long c = (idx / Tlen) % conv_dim;
    long b = idx / (static_cast<long>(conv_dim) * Tlen);
    long proj_base = (b * Tlen + t) * proj_size;
    conv_in[idx] = proj[proj_base + D + c];
}

template<typename T>
__global__ void mamba_expand_delta_kernel(T* delta, const T* proj, long total, int Tlen, int D, int conv_dim, int num_heads, int head_dim, int proj_size) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long t = idx % Tlen;
    long d = (idx / Tlen) % D;
    long b = idx / (static_cast<long>(D) * Tlen);
    int h = static_cast<int>(d / head_dim);
    if (h >= num_heads) return;
    long proj_base = (b * Tlen + t) * proj_size;
    delta[idx] = proj[proj_base + D + conv_dim + h];
}

template<typename T>
__global__ void mamba_split_conv_out_kernel(const T* conv_out, T* u, T* B, T* C,
                                            long total, int Tlen, int D, int groups, int dstate) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long t = idx % Tlen;
    long c = (idx / Tlen) % (D + 2 * groups * dstate);
    long b = idx / (static_cast<long>(D + 2 * groups * dstate) * Tlen);

    if (c < D) {
        u[(b * D + c) * Tlen + t] = conv_out[idx];
    } else if (c < D + groups * dstate) {
        long off = c - D;
        int g = static_cast<int>(off / dstate);
        int s = static_cast<int>(off - g * dstate);
        B[(((b * groups + g) * dstate + s) * Tlen) + t] = conv_out[idx];
    } else {
        long off = c - D - groups * dstate;
        int g = static_cast<int>(off / dstate);
        int s = static_cast<int>(off - g * dstate);
        C[(((b * groups + g) * dstate + s) * Tlen) + t] = conv_out[idx];
    }
}

template<typename T>
__global__ void mamba_pack_conv_out_kernel(T* d_conv_out, const T* d_u, const float* d_B, const float* d_C,
                                           long total, int Tlen, int D, int groups, int dstate) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long t = idx % Tlen;
    long c = (idx / Tlen) % (D + 2 * groups * dstate);
    long b = idx / (static_cast<long>(D + 2 * groups * dstate) * Tlen);

    if (c < D) {
        d_conv_out[idx] = d_u[(b * D + c) * Tlen + t];
    } else if (c < D + groups * dstate) {
        long off = c - D;
        int g = static_cast<int>(off / dstate);
        int s = static_cast<int>(off - g * dstate);
        float v = d_B[(((b * groups + g) * dstate + s) * Tlen) + t];
        d_conv_out[idx] = from_float<T>(v);
    } else {
        long off = c - D - groups * dstate;
        int g = static_cast<int>(off / dstate);
        int s = static_cast<int>(off - g * dstate);
        float v = d_C[(((b * groups + g) * dstate + s) * Tlen) + t];
        d_conv_out[idx] = from_float<T>(v);
    }
}

template<typename T>
__global__ void mamba_transpose_btd_to_bdt_kernel(T* out, const T* in, long total, int Tlen, int D) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long d = idx % D;
    long bt = idx / D;
    long t = bt % Tlen;
    long b = bt / Tlen;
    out[(b * D + d) * Tlen + t] = in[idx];
}

template<typename T>
__global__ void mamba_transpose_bdt_to_btd_kernel(T* out, const T* in, long total, int Tlen, int D) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long t = idx % Tlen;
    long d = (idx / Tlen) % D;
    long b = idx / (static_cast<long>(D) * Tlen);
    out[(b * Tlen + t) * D + d] = in[(b * D + d) * Tlen + t];
}

__global__ void mamba_expand_A_kernel(float* A, const float* A_log, long total, int dstate, int head_dim) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long d = idx / dstate;
    int h = static_cast<int>(d / head_dim);
    float val = -expf(A_log[h]);
    A[idx] = val;
}

__global__ void mamba_expand_head_param_kernel(float* out, const float* param, long total, int head_dim) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = static_cast<int>(idx / head_dim);
    out[idx] = param[h];
}

__global__ void mamba_reduce_dA_log_kernel(float* dA_log, const float* dA, const float* A, long total, int dstate, int head_dim) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long d = idx / dstate;
    int h = static_cast<int>(d / head_dim);
    atomicAdd(&dA_log[h], dA[idx] * A[idx]);
}

__global__ void mamba_reduce_head_param_kernel(float* d_param, const float* d_param_exp, long total, int head_dim) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = static_cast<int>(idx / head_dim);
    atomicAdd(&d_param[h], d_param_exp[idx]);
}

template<typename T>
__global__ void mamba_reduce_delta_kernel(T* d_dt, const T* d_delta, long total, int Tlen, int D, int num_heads, int head_dim) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long t = idx % Tlen;
    long h = (idx / Tlen) % num_heads;
    long b = idx / (static_cast<long>(num_heads) * Tlen);
    float sum = 0.f;
    long base = (b * D + h * head_dim) * Tlen + t;
    for (int i = 0; i < head_dim; ++i) {
        sum += to_float(d_delta[base + i * Tlen]);
    }
    d_dt[idx] = from_float<T>(sum);
}

template<typename T>
__global__ void mamba_pack_dproj_kernel(T* d_proj, const T* d_gate, const T* d_conv_in, const T* d_dt,
                                        long total, int Tlen, int D, int conv_dim, int num_heads) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long k = idx % (D + conv_dim + num_heads);
    long bt = idx / (D + conv_dim + num_heads);
    long t = bt % Tlen;
    long b = bt / Tlen;

    if (k < D) {
        d_proj[idx] = d_gate[(b * Tlen + t) * D + k];
    } else if (k < D + conv_dim) {
        long c = k - D;
        d_proj[idx] = d_conv_in[(b * conv_dim + c) * Tlen + t];
    } else {
        long h = k - D - conv_dim;
        d_proj[idx] = d_dt[(b * Tlen + t) * num_heads + h];
    }
}

template<typename T>
__global__ void mamba_group_rmsnorm_forward_kernel(const T* inp, const T* weight, T* out, float* rstd,
                                                   int tokens, int D, int groups, int group_size, float eps) {
    int token = blockIdx.x;
    int group = blockIdx.y;
    if (token >= tokens || group >= groups) return;
    int tid = threadIdx.x;
    int base = token * D + group * group_size;

    float sum = 0.f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        float v = to_float(inp[base + i]);
        sum += v * v;
    }
    // Shared reduction
    extern __shared__ float smem[];
    smem[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }
    float r = rsqrtf(smem[0] / static_cast<float>(group_size) + eps);
    if (tid == 0) {
        rstd[token * groups + group] = r;
    }
    __syncthreads();
    for (int i = tid; i < group_size; i += blockDim.x) {
        float v = to_float(inp[base + i]);
        float w = to_float(weight[group * group_size + i]);
        out[base + i] = from_float<T>(v * r * w);
    }
}

template<typename T>
__global__ void mamba_group_rmsnorm_backward_dx_kernel(const T* dout, const T* inp, const T* weight,
                                                       const float* rstd, T* dinp,
                                                       int tokens, int D, int groups, int group_size) {
    int token = blockIdx.x;
    int group = blockIdx.y;
    if (token >= tokens || group >= groups) return;
    int tid = threadIdx.x;
    int base = token * D + group * group_size;

    float sum = 0.f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        float x = to_float(inp[base + i]);
        float dy = to_float(dout[base + i]);
        float w = to_float(weight[group * group_size + i]);
        sum += dy * w * x;
    }
    extern __shared__ float smem[];
    smem[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }
    float r = rstd[token * groups + group];
    float r3 = r * r * r;
    float inv_n = 1.0f / static_cast<float>(group_size);
    float coeff = r3 * inv_n * smem[0];

    for (int i = tid; i < group_size; i += blockDim.x) {
        float x = to_float(inp[base + i]);
        float dy = to_float(dout[base + i]);
        float w = to_float(weight[group * group_size + i]);
        float dx = dy * w * r - x * coeff;
        dinp[base + i] = from_float<T>(dx);
    }
}

template<typename T>
__global__ void mamba_group_rmsnorm_backward_dweight_kernel(float* dweight, const T* dout, const T* inp,
                                                            const float* rstd, long total, int D, int groups, int group_size) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long token = idx / D;
    long d = idx % D;
    int group = static_cast<int>(d / group_size);
    float r = rstd[token * groups + group];
    float grad = to_float(dout[idx]) * to_float(inp[idx]) * r;
    atomicAdd(&dweight[d], grad);
}

} // namespace

// =============================================================================
// Public wrappers
// =============================================================================

void mamba_split_proj(Tensor& gate, Tensor& conv_in, Tensor& delta, const Tensor& proj,
                      int B, int Tlen, int D, int conv_dim, int num_heads, int head_dim,
                      cudaStream_t stream) {
    const long gate_total = static_cast<long>(B) * Tlen * D;
    const long conv_total = static_cast<long>(B) * conv_dim * Tlen;
    const long delta_total = static_cast<long>(B) * D * Tlen;
    const int proj_size = D + conv_dim + num_heads;

    const int threads = 256;
    if (proj.DType == ETensorDType::BF16) {
        mamba_copy_gate_kernel<<<div_up(gate_total, threads), threads, 0, stream>>>(
            gate.get<nv_bfloat16>(), proj.get<nv_bfloat16>(), gate_total, D, proj_size);
        mamba_copy_conv_in_kernel<<<div_up(conv_total, threads), threads, 0, stream>>>(
            conv_in.get<nv_bfloat16>(), proj.get<nv_bfloat16>(), conv_total, Tlen, D, conv_dim, proj_size);
        mamba_expand_delta_kernel<<<div_up(delta_total, threads), threads, 0, stream>>>(
            delta.get<nv_bfloat16>(), proj.get<nv_bfloat16>(), delta_total, Tlen, D, conv_dim, num_heads, head_dim, proj_size);
    } else if (proj.DType == ETensorDType::FP16) {
        mamba_copy_gate_kernel<<<div_up(gate_total, threads), threads, 0, stream>>>(
            gate.get<half>(), proj.get<half>(), gate_total, D, proj_size);
        mamba_copy_conv_in_kernel<<<div_up(conv_total, threads), threads, 0, stream>>>(
            conv_in.get<half>(), proj.get<half>(), conv_total, Tlen, D, conv_dim, proj_size);
        mamba_expand_delta_kernel<<<div_up(delta_total, threads), threads, 0, stream>>>(
            delta.get<half>(), proj.get<half>(), delta_total, Tlen, D, conv_dim, num_heads, head_dim, proj_size);
    } else {
        throw std::logic_error("mamba_split_proj: unsupported dtype");
    }
}

void mamba_split_conv_out(Tensor& u, Tensor& B, Tensor& C, const Tensor& conv_out,
                          int Bsz, int Tlen, int D, int groups, int dstate,
                          cudaStream_t stream) {
    const long total = static_cast<long>(Bsz) * (D + 2 * groups * dstate) * Tlen;
    const int threads = 256;
    if (conv_out.DType == ETensorDType::BF16) {
        mamba_split_conv_out_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            conv_out.get<nv_bfloat16>(), u.get<nv_bfloat16>(), B.get<nv_bfloat16>(), C.get<nv_bfloat16>(),
            total, Tlen, D, groups, dstate);
    } else if (conv_out.DType == ETensorDType::FP16) {
        mamba_split_conv_out_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            conv_out.get<half>(), u.get<half>(), B.get<half>(), C.get<half>(),
            total, Tlen, D, groups, dstate);
    } else {
        throw std::logic_error("mamba_split_conv_out: unsupported dtype");
    }
}

void mamba_pack_conv_out(Tensor& d_conv_out, const Tensor& d_u, const Tensor& d_B, const Tensor& d_C,
                         int Bsz, int Tlen, int D, int groups, int dstate,
                         cudaStream_t stream) {
    const long total = static_cast<long>(Bsz) * (D + 2 * groups * dstate) * Tlen;
    const int threads = 256;
    if (d_conv_out.DType == ETensorDType::BF16) {
        mamba_pack_conv_out_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            d_conv_out.get<nv_bfloat16>(), d_u.get<nv_bfloat16>(),
            d_B.get<float>(), d_C.get<float>(), total, Tlen, D, groups, dstate);
    } else if (d_conv_out.DType == ETensorDType::FP16) {
        mamba_pack_conv_out_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            d_conv_out.get<half>(), d_u.get<half>(),
            d_B.get<float>(), d_C.get<float>(), total, Tlen, D, groups, dstate);
    } else {
        throw std::logic_error("mamba_pack_conv_out: unsupported dtype");
    }
}

void mamba_transpose_btd_to_bdt(Tensor& out, const Tensor& inp, int B, int Tlen, int D, cudaStream_t stream) {
    const long total = static_cast<long>(B) * Tlen * D;
    const int threads = 256;
    if (inp.DType == ETensorDType::BF16) {
        mamba_transpose_btd_to_bdt_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), total, Tlen, D);
    } else if (inp.DType == ETensorDType::FP16) {
        mamba_transpose_btd_to_bdt_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            out.get<half>(), inp.get<half>(), total, Tlen, D);
    } else {
        throw std::logic_error("mamba_transpose_btd_to_bdt: unsupported dtype");
    }
}

void mamba_transpose_bdt_to_btd(Tensor& out, const Tensor& inp, int B, int Tlen, int D, cudaStream_t stream) {
    const long total = static_cast<long>(B) * Tlen * D;
    const int threads = 256;
    if (inp.DType == ETensorDType::BF16) {
        mamba_transpose_bdt_to_btd_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), total, Tlen, D);
    } else if (inp.DType == ETensorDType::FP16) {
        mamba_transpose_bdt_to_btd_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            out.get<half>(), inp.get<half>(), total, Tlen, D);
    } else {
        throw std::logic_error("mamba_transpose_bdt_to_btd: unsupported dtype");
    }
}

void mamba_expand_A(Tensor& A, const Tensor& A_log, int num_heads, int head_dim, int dstate, cudaStream_t stream) {
    (void)num_heads;
    const long total = static_cast<long>(head_dim) * num_heads * dstate;
    const int threads = 256;
    mamba_expand_A_kernel<<<div_up(total, threads), threads, 0, stream>>>(
        A.get<float>(), A_log.get<float>(), total, dstate, head_dim);
}

void mamba_expand_head_param(Tensor& out, const Tensor& param, int num_heads, int head_dim, cudaStream_t stream) {
    (void)num_heads;
    const long total = static_cast<long>(head_dim) * num_heads;
    const int threads = 256;
    mamba_expand_head_param_kernel<<<div_up(total, threads), threads, 0, stream>>>(
        out.get<float>(), param.get<float>(), total, head_dim);
}

void mamba_reduce_dA_log(Tensor& dA_log, const Tensor& dA, const Tensor& A,
                         int num_heads, int head_dim, int dstate, bool accumulate, cudaStream_t stream) {
    if (!accumulate) {
        CUDA_CHECK(cudaMemsetAsync(dA_log.Data, 0, dA_log.bytes(), stream));
    }
    const long total = static_cast<long>(head_dim) * num_heads * dstate;
    const int threads = 256;
    mamba_reduce_dA_log_kernel<<<div_up(total, threads), threads, 0, stream>>>(
        dA_log.get<float>(), dA.get<float>(), A.get<float>(), total, dstate, head_dim);
}

void mamba_reduce_head_param(Tensor& d_param, const Tensor& d_param_exp,
                             int num_heads, int head_dim, bool accumulate, cudaStream_t stream) {
    if (!accumulate) {
        CUDA_CHECK(cudaMemsetAsync(d_param.Data, 0, d_param.bytes(), stream));
    }
    const long total = static_cast<long>(head_dim) * num_heads;
    const int threads = 256;
    mamba_reduce_head_param_kernel<<<div_up(total, threads), threads, 0, stream>>>(
        d_param.get<float>(), d_param_exp.get<float>(), total, head_dim);
}

void mamba_reduce_delta_to_dt(Tensor& d_dt, const Tensor& d_delta,
                              int B, int Tlen, int num_heads, int head_dim, cudaStream_t stream) {
    const long total = static_cast<long>(B) * Tlen * num_heads;
    const int threads = 256;
    if (d_delta.DType == ETensorDType::BF16) {
        mamba_reduce_delta_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            d_dt.get<nv_bfloat16>(), d_delta.get<nv_bfloat16>(), total, Tlen,
            num_heads * head_dim, num_heads, head_dim);
    } else if (d_delta.DType == ETensorDType::FP16) {
        mamba_reduce_delta_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            d_dt.get<half>(), d_delta.get<half>(), total, Tlen,
            num_heads * head_dim, num_heads, head_dim);
    } else {
        throw std::logic_error("mamba_reduce_delta_to_dt: unsupported dtype");
    }
}

void mamba_pack_dproj(Tensor& d_proj, const Tensor& d_gate, const Tensor& d_conv_in, const Tensor& d_dt,
                      int B, int Tlen, int D, int conv_dim, int num_heads, cudaStream_t stream) {
    const long total = static_cast<long>(B) * Tlen * (D + conv_dim + num_heads);
    const int threads = 256;
    if (d_proj.DType == ETensorDType::BF16) {
        mamba_pack_dproj_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            d_proj.get<nv_bfloat16>(),
            d_gate.get<nv_bfloat16>(),
            d_conv_in.get<nv_bfloat16>(),
            d_dt.get<nv_bfloat16>(),
            total, Tlen, D, conv_dim, num_heads);
    } else if (d_proj.DType == ETensorDType::FP16) {
        mamba_pack_dproj_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            d_proj.get<half>(),
            d_gate.get<half>(),
            d_conv_in.get<half>(),
            d_dt.get<half>(),
            total, Tlen, D, conv_dim, num_heads);
    } else {
        throw std::logic_error("mamba_pack_dproj: unsupported dtype");
    }
}

void mamba_group_rmsnorm_forward(Tensor& out, Tensor& rstd, const Tensor& inp, const Tensor& weight,
                                 float epsilon, int B, int Tlen, int D, int groups, cudaStream_t stream) {
    const int tokens = B * Tlen;
    const int group_size = D / groups;
    dim3 grid(tokens, groups, 1);
    const int threads = 256;
    const size_t smem = threads * sizeof(float);
    if (inp.DType == ETensorDType::BF16) {
        mamba_group_rmsnorm_forward_kernel<<<grid, threads, smem, stream>>>(
            inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), out.get<nv_bfloat16>(),
            rstd.get<float>(), tokens, D, groups, group_size, epsilon);
    } else if (inp.DType == ETensorDType::FP16) {
        mamba_group_rmsnorm_forward_kernel<<<grid, threads, smem, stream>>>(
            inp.get<half>(), weight.get<half>(), out.get<half>(),
            rstd.get<float>(), tokens, D, groups, group_size, epsilon);
    } else {
        throw std::logic_error("mamba_group_rmsnorm_forward: unsupported dtype");
    }
}

void mamba_group_rmsnorm_backward_dx(Tensor& dinp, const Tensor& dout, const Tensor& inp, const Tensor& weight,
                                     const Tensor& rstd, int B, int Tlen, int D, int groups, cudaStream_t stream) {
    const int tokens = B * Tlen;
    const int group_size = D / groups;
    dim3 grid(tokens, groups, 1);
    const int threads = 256;
    const size_t smem = threads * sizeof(float);
    if (inp.DType == ETensorDType::BF16) {
        mamba_group_rmsnorm_backward_dx_kernel<<<grid, threads, smem, stream>>>(
            dout.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
            rstd.get<float>(), dinp.get<nv_bfloat16>(),
            tokens, D, groups, group_size);
    } else if (inp.DType == ETensorDType::FP16) {
        mamba_group_rmsnorm_backward_dx_kernel<<<grid, threads, smem, stream>>>(
            dout.get<half>(), inp.get<half>(), weight.get<half>(),
            rstd.get<float>(), dinp.get<half>(),
            tokens, D, groups, group_size);
    } else {
        throw std::logic_error("mamba_group_rmsnorm_backward_dx: unsupported dtype");
    }
}

void mamba_group_rmsnorm_backward_dweight_fp32(Tensor& dweight_fp32, const Tensor& dout, const Tensor& inp,
                                               const Tensor& rstd, int B, int Tlen, int D, int groups,
                                               cudaStream_t stream) {
    const long total = static_cast<long>(B) * Tlen * D;
    const int group_size = D / groups;
    const int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(dweight_fp32.Data, 0, dweight_fp32.bytes(), stream));
    if (inp.DType == ETensorDType::BF16) {
        mamba_group_rmsnorm_backward_dweight_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            dweight_fp32.get<float>(), dout.get<nv_bfloat16>(), inp.get<nv_bfloat16>(),
            rstd.get<float>(), total, D, groups, group_size);
    } else if (inp.DType == ETensorDType::FP16) {
        mamba_group_rmsnorm_backward_dweight_kernel<<<div_up(total, threads), threads, 0, stream>>>(
            dweight_fp32.get<float>(), dout.get<half>(), inp.get<half>(),
            rstd.get<float>(), total, D, groups, group_size);
    } else {
        throw std::logic_error("mamba_group_rmsnorm_backward_dweight_fp32: unsupported dtype");
    }
}

// -----------------------------------------------------------------------------
// Causal Conv1D wrappers
// -----------------------------------------------------------------------------

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);
template<typename input_t, typename weight_t>
void causal_conv1d_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream);

void mamba_causal_conv1d_forward(Tensor& out, const Tensor& x, const Tensor& weight, const Tensor* bias,
                                int B, int Tlen, int conv_dim, int kernel, bool silu, cudaStream_t stream) {
    ConvParamsBase params{};
    params.batch = B;
    params.dim = conv_dim;
    params.seqlen = Tlen;
    params.width = kernel;
    params.silu_activation = silu;

    params.x_batch_stride = conv_dim * Tlen;
    params.x_c_stride = Tlen;
    params.x_l_stride = 1;
    params.weight_c_stride = kernel;
    params.weight_width_stride = 1;
    params.out_batch_stride = conv_dim * Tlen;
    params.out_c_stride = Tlen;
    params.out_l_stride = 1;

    params.x_ptr = x.Data;
    params.weight_ptr = weight.Data;
    params.bias_ptr = bias ? bias->Data : nullptr;
    params.out_ptr = out.Data;

    params.conv_state_ptr = nullptr;
    params.cache_seqlens = nullptr;
    params.conv_state_indices_ptr = nullptr;
    params.seq_idx_ptr = nullptr;
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = nullptr;
    params.conv_state_len = 0;

    if (x.DType == ETensorDType::BF16 && weight.DType == ETensorDType::BF16) {
        causal_conv1d_fwd_cuda<at::BFloat16, at::BFloat16>(params, stream);
    } else if (x.DType == ETensorDType::FP16 && weight.DType == ETensorDType::FP16) {
        causal_conv1d_fwd_cuda<at::Half, at::Half>(params, stream);
    } else if (x.DType == ETensorDType::FP32 && weight.DType == ETensorDType::FP32) {
        causal_conv1d_fwd_cuda<float, float>(params, stream);
    } else {
        throw std::logic_error("mamba_causal_conv1d_forward: unsupported dtype");
    }
}

void mamba_causal_conv1d_backward(Tensor& dx, Tensor& dweight_fp32, Tensor* dbias_fp32,
                                 const Tensor& x, const Tensor& weight, const Tensor& dout,
                                 int B, int Tlen, int conv_dim, int kernel, bool silu, cudaStream_t stream) {
    ConvParamsBwd params{};
    params.batch = B;
    params.dim = conv_dim;
    params.seqlen = Tlen;
    params.width = kernel;
    params.silu_activation = silu;

    params.x_batch_stride = conv_dim * Tlen;
    params.x_c_stride = Tlen;
    params.x_l_stride = 1;
    params.weight_c_stride = kernel;
    params.weight_width_stride = 1;
    params.out_batch_stride = conv_dim * Tlen;
    params.out_c_stride = Tlen;
    params.out_l_stride = 1;

    params.dx_batch_stride = conv_dim * Tlen;
    params.dx_c_stride = Tlen;
    params.dx_l_stride = 1;
    params.dweight_c_stride = kernel;
    params.dweight_width_stride = 1;
    params.dout_batch_stride = conv_dim * Tlen;
    params.dout_c_stride = Tlen;
    params.dout_l_stride = 1;

    params.x_ptr = x.Data;
    params.weight_ptr = weight.Data;
    params.bias_ptr = nullptr;
    params.out_ptr = nullptr;

    params.dx_ptr = dx.Data;
    params.dweight_ptr = dweight_fp32.Data;
    params.dbias_ptr = dbias_fp32 ? dbias_fp32->Data : nullptr;
    params.dout_ptr = dout.Data;

    params.conv_state_ptr = nullptr;
    params.cache_seqlens = nullptr;
    params.conv_state_indices_ptr = nullptr;
    params.seq_idx_ptr = nullptr;
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = nullptr;
    params.conv_state_len = 0;

    params.dinitial_states_ptr = nullptr;
    params.dfinal_states_ptr = nullptr;
    params.dweight_workspace_ptr = nullptr;
    params.dbias_workspace_ptr = nullptr;
    params.dweight_workspace_batch_stride = 0;
    params.dweight_workspace_dim_stride = 0;
    params.dbias_workspace_batch_stride = 0;
    params.deterministic = false;

    if (x.DType == ETensorDType::BF16 && weight.DType == ETensorDType::BF16) {
        causal_conv1d_bwd_cuda<at::BFloat16, at::BFloat16>(params, stream);
    } else if (x.DType == ETensorDType::FP16 && weight.DType == ETensorDType::FP16) {
        causal_conv1d_bwd_cuda<at::Half, at::Half>(params, stream);
    } else if (x.DType == ETensorDType::FP32 && weight.DType == ETensorDType::FP32) {
        causal_conv1d_bwd_cuda<float, float>(params, stream);
    } else {
        throw std::logic_error("mamba_causal_conv1d_backward: unsupported dtype");
    }
}

// -----------------------------------------------------------------------------
// Selective scan wrappers
// -----------------------------------------------------------------------------

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);
template<typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream);

void mamba_selective_scan_forward(Tensor& out, const Tensor& u, const Tensor& delta,
                                  const Tensor& A, const Tensor& B, const Tensor& C,
                                  const Tensor& D, const Tensor& delta_bias,
                                  Tensor& x, int Bsz, int Tlen, int Ddim, int dstate,
                                  int groups, int n_chunks, cudaStream_t stream) {
    SSMParamsBase params{};
    params.batch = Bsz;
    params.dim = Ddim;
    params.seqlen = Tlen;
    params.dstate = dstate;
    params.n_groups = groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = Ddim / groups;
    params.is_variable_B = true;
    params.is_variable_C = true;
    params.delta_softplus = true;

    params.A_ptr = A.Data;
    params.B_ptr = B.Data;
    params.C_ptr = C.Data;
    params.D_ptr = D.Data;
    params.u_ptr = u.Data;
    params.delta_ptr = delta.Data;
    params.delta_bias_ptr = delta_bias.Data;
    params.out_ptr = out.Data;
    params.x_ptr = x.Data;
    params.z_ptr = nullptr;
    params.out_z_ptr = nullptr;

    params.A_d_stride = dstate;
    params.A_dstate_stride = 1;
    params.B_batch_stride = groups * dstate * Tlen;
    params.B_group_stride = dstate * Tlen;
    params.B_dstate_stride = Tlen;
    params.C_batch_stride = groups * dstate * Tlen;
    params.C_group_stride = dstate * Tlen;
    params.C_dstate_stride = Tlen;
    params.u_batch_stride = Ddim * Tlen;
    params.u_d_stride = Tlen;
    params.delta_batch_stride = Ddim * Tlen;
    params.delta_d_stride = Tlen;
    params.out_batch_stride = Ddim * Tlen;
    params.out_d_stride = Tlen;

    if (u.DType == ETensorDType::BF16) {
        selective_scan_fwd_cuda<at::BFloat16, float>(params, stream);
    } else if (u.DType == ETensorDType::FP16) {
        selective_scan_fwd_cuda<at::Half, float>(params, stream);
    } else if (u.DType == ETensorDType::FP32) {
        selective_scan_fwd_cuda<float, float>(params, stream);
    } else {
        throw std::logic_error("mamba_selective_scan_forward: unsupported dtype");
    }
}

void mamba_selective_scan_backward(Tensor& du, Tensor& ddelta, Tensor& dA, Tensor& dB, Tensor& dC,
                                   Tensor* dD, Tensor* ddelta_bias,
                                   const Tensor& u, const Tensor& delta,
                                   const Tensor& A, const Tensor& B, const Tensor& C,
                                   const Tensor& D, const Tensor& delta_bias,
                                   const Tensor& dout, Tensor& x,
                                   int Bsz, int Tlen, int Ddim, int dstate,
                                   int groups, int n_chunks, cudaStream_t stream) {
    SSMParamsBwd params{};
    params.batch = Bsz;
    params.dim = Ddim;
    params.seqlen = Tlen;
    params.dstate = dstate;
    params.n_groups = groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = Ddim / groups;
    params.is_variable_B = true;
    params.is_variable_C = true;
    params.delta_softplus = true;

    params.A_ptr = A.Data;
    params.B_ptr = B.Data;
    params.C_ptr = C.Data;
    params.D_ptr = D.Data;
    params.u_ptr = u.Data;
    params.delta_ptr = delta.Data;
    params.delta_bias_ptr = delta_bias.Data;
    params.out_ptr = dout.Data; // not used
    params.x_ptr = x.Data;
    params.z_ptr = nullptr;
    params.out_z_ptr = nullptr;

    params.A_d_stride = dstate;
    params.A_dstate_stride = 1;
    params.B_batch_stride = groups * dstate * Tlen;
    params.B_group_stride = dstate * Tlen;
    params.B_dstate_stride = Tlen;
    params.C_batch_stride = groups * dstate * Tlen;
    params.C_group_stride = dstate * Tlen;
    params.C_dstate_stride = Tlen;
    params.u_batch_stride = Ddim * Tlen;
    params.u_d_stride = Tlen;
    params.delta_batch_stride = Ddim * Tlen;
    params.delta_d_stride = Tlen;
    params.out_batch_stride = Ddim * Tlen;
    params.out_d_stride = Tlen;

    params.dout_ptr = dout.Data;
    params.dA_ptr = dA.Data;
    params.dB_ptr = dB.Data;
    params.dC_ptr = dC.Data;
    params.dD_ptr = dD ? dD->Data : nullptr;
    params.ddelta_ptr = ddelta.Data;
    params.ddelta_bias_ptr = ddelta_bias ? ddelta_bias->Data : nullptr;
    params.dz_ptr = nullptr;

    params.dout_batch_stride = Ddim * Tlen;
    params.dout_d_stride = Tlen;
    params.dA_d_stride = dstate;
    params.dA_dstate_stride = 1;
    params.dB_batch_stride = groups * dstate * Tlen;
    params.dB_group_stride = dstate * Tlen;
    params.dB_d_stride = 0;
    params.dB_dstate_stride = Tlen;
    params.dC_batch_stride = groups * dstate * Tlen;
    params.dC_group_stride = dstate * Tlen;
    params.dC_d_stride = 0;
    params.dC_dstate_stride = Tlen;
    params.du_batch_stride = Ddim * Tlen;
    params.du_d_stride = Tlen;
    params.ddelta_batch_stride = Ddim * Tlen;
    params.ddelta_d_stride = Tlen;

    if (u.DType == ETensorDType::BF16) {
        selective_scan_bwd_cuda<at::BFloat16, float>(params, stream);
    } else if (u.DType == ETensorDType::FP16) {
        selective_scan_bwd_cuda<at::Half, float>(params, stream);
    } else if (u.DType == ETensorDType::FP32) {
        selective_scan_bwd_cuda<float, float>(params, stream);
    } else {
        throw std::logic_error("mamba_selective_scan_backward: unsupported dtype");
    }
}
