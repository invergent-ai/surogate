// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Masked scatter kernels for Qwen3-VL visual embedding injection.

#include <cuda_bf16.h>
#include <cub/cub.cuh>

#include "utilities/utils.h"
#include "utilities/tensor.h"

namespace {

template<typename T>
__device__ __forceinline__ T zero_value();

template<>
__device__ __forceinline__ float zero_value<float>() {
    return 0.0f;
}

template<>
__device__ __forceinline__ nv_bfloat16 zero_value<nv_bfloat16>() {
    return __float2bfloat16(0.0f);
}

template<typename T, bool AddMode>
__global__ void mask_scatter_forward_kernel(const T* inp, const int* mask, const int* prefix,
                                            const T* src, T* out, long total, int C) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = static_cast<int>(idx / C);
    int col = static_cast<int>(idx - static_cast<long>(row) * C);
    const int m = mask[row];
    if (m != 0) {
        const int src_row = prefix[row];
        const long src_idx = static_cast<long>(src_row) * C + col;
        if constexpr (AddMode) {
            out[idx] = inp[idx] + src[src_idx];
        } else {
            out[idx] = src[src_idx];
        }
    } else {
        out[idx] = inp[idx];
    }
}

template<typename T, bool KeepInputGrad>
__global__ void mask_scatter_backward_kernel(const T* d_out, const int* mask, const int* prefix,
                                             T* d_inp, T* d_src, long total, int C,
                                             bool write_inp, bool write_src) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = static_cast<int>(idx / C);
    int col = static_cast<int>(idx - static_cast<long>(row) * C);
    const int m = mask[row];
    if (write_inp) {
        if (KeepInputGrad || m == 0) {
            d_inp[idx] = d_out[idx];
        } else {
            d_inp[idx] = zero_value<T>();
        }
    }
    if (write_src && m != 0) {
        const int src_row = prefix[row];
        const long src_idx = static_cast<long>(src_row) * C + col;
        d_src[src_idx] = d_out[idx];
    }
}

void compute_prefix(const Tensor& mask, Tensor& prefix, Tensor& temp, int n, cudaStream_t stream) {
    if (n <= 0) return;
    if (mask.DType != ETensorDType::INT32 || prefix.DType != ETensorDType::INT32) {
        throw std::logic_error("mask_scatter: mask/prefix must be int32");
    }
    std::size_t temp_bytes = temp.bytes();
    if (temp_bytes == 0) {
        return;
    }
    cub::DeviceScan::ExclusiveSum(temp.Data, temp_bytes,
                                  mask.get<int>(), prefix.get<int>(), n, stream);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, bool AddMode>
void launch_forward(Tensor& out, const Tensor& inp, const Tensor& mask, const Tensor& src,
                    Tensor& prefix, Tensor& temp, int B, int Tn, int C, cudaStream_t stream) {
    const long n = static_cast<long>(B) * static_cast<long>(Tn);
    const long total = n * static_cast<long>(C);
    if (total <= 0) return;
    compute_prefix(mask, prefix, temp, static_cast<int>(n), stream);
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    mask_scatter_forward_kernel<T, AddMode><<<grid, block, 0, stream>>>(
        inp.get<T>(), mask.get<int>(), prefix.get<int>(), src.get<T>(), out.get<T>(), total, C);
    CUDA_CHECK(cudaGetLastError());
}

template<typename T, bool KeepInputGrad>
void launch_backward(Tensor& d_inp, Tensor& d_src, const Tensor& d_out, const Tensor& mask,
                     Tensor& prefix, Tensor& temp, int B, int Tn, int C, cudaStream_t stream,
                     bool write_inp, bool write_src) {
    const long n = static_cast<long>(B) * static_cast<long>(Tn);
    const long total = n * static_cast<long>(C);
    if (total <= 0) return;
    compute_prefix(mask, prefix, temp, static_cast<int>(n), stream);
    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);
    mask_scatter_backward_kernel<T, KeepInputGrad><<<grid, block, 0, stream>>>(
        d_out.get<T>(), mask.get<int>(), prefix.get<int>(),
        write_inp ? d_inp.get<T>() : nullptr,
        write_src ? d_src.get<T>() : nullptr,
        total, C, write_inp, write_src);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t mask_scatter_temp_bytes(int n) {
    std::size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes,
                                  static_cast<const int*>(nullptr),
                                  static_cast<int*>(nullptr), n);
    return temp_bytes;
}

void mask_scatter_forward(Tensor& out, const Tensor& inp, const Tensor& mask, const Tensor& src,
                          Tensor& prefix, Tensor& temp, int B, int Tn, int C, cudaStream_t stream) {
    if (out.DType != inp.DType || out.DType != src.DType) {
        throw std::logic_error("mask_scatter_forward: dtype mismatch");
    }
    if (out.DType == ETensorDType::BF16) {
        launch_forward<nv_bfloat16, false>(out, inp, mask, src, prefix, temp, B, Tn, C, stream);
    } else if (out.DType == ETensorDType::FP32) {
        launch_forward<float, false>(out, inp, mask, src, prefix, temp, B, Tn, C, stream);
    } else {
        throw std::logic_error("mask_scatter_forward: unsupported dtype");
    }
}

void mask_scatter_backward(Tensor& d_inp, Tensor& d_src, const Tensor& d_out, const Tensor& mask,
                           Tensor& prefix, Tensor& temp, int B, int Tn, int C, cudaStream_t stream,
                           bool write_inp, bool write_src) {
    if (write_inp && d_out.DType != d_inp.DType) {
        throw std::logic_error("mask_scatter_backward: dtype mismatch (d_inp)");
    }
    if (write_src && d_out.DType != d_src.DType) {
        throw std::logic_error("mask_scatter_backward: dtype mismatch (d_src)");
    }
    if (d_out.DType == ETensorDType::BF16) {
        launch_backward<nv_bfloat16, false>(d_inp, d_src, d_out, mask, prefix, temp, B, Tn, C, stream, write_inp, write_src);
    } else if (d_out.DType == ETensorDType::FP32) {
        launch_backward<float, false>(d_inp, d_src, d_out, mask, prefix, temp, B, Tn, C, stream, write_inp, write_src);
    } else {
        throw std::logic_error("mask_scatter_backward: unsupported dtype");
    }
}

void deepstack_inject_forward(Tensor& out, const Tensor& inp, const Tensor& mask, const Tensor& src,
                              Tensor& prefix, Tensor& temp, int B, int Tn, int C, cudaStream_t stream) {
    if (out.DType != inp.DType || out.DType != src.DType) {
        throw std::logic_error("deepstack_inject_forward: dtype mismatch");
    }
    if (out.DType == ETensorDType::BF16) {
        launch_forward<nv_bfloat16, true>(out, inp, mask, src, prefix, temp, B, Tn, C, stream);
    } else if (out.DType == ETensorDType::FP32) {
        launch_forward<float, true>(out, inp, mask, src, prefix, temp, B, Tn, C, stream);
    } else {
        throw std::logic_error("deepstack_inject_forward: unsupported dtype");
    }
}

void deepstack_inject_backward(Tensor& d_inp, Tensor& d_src, const Tensor& d_out, const Tensor& mask,
                               Tensor& prefix, Tensor& temp, int B, int Tn, int C, cudaStream_t stream,
                               bool write_inp, bool write_src) {
    if (write_inp && d_out.DType != d_inp.DType) {
        throw std::logic_error("deepstack_inject_backward: dtype mismatch (d_inp)");
    }
    if (write_src && d_out.DType != d_src.DType) {
        throw std::logic_error("deepstack_inject_backward: dtype mismatch (d_src)");
    }
    if (d_out.DType == ETensorDType::BF16) {
        launch_backward<nv_bfloat16, true>(d_inp, d_src, d_out, mask, prefix, temp, B, Tn, C, stream, write_inp, write_src);
    } else if (d_out.DType == ETensorDType::FP32) {
        launch_backward<float, true>(d_inp, d_src, d_out, mask, prefix, temp, B, Tn, C, stream, write_inp, write_src);
    } else {
        throw std::logic_error("deepstack_inject_backward: unsupported dtype");
    }
}
