// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Multimodal RoPE (MRoPE) kernel for Qwen3-VL style 3D position IDs.

#include <cuda_bf16.h>

#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"
#include "utilities/tensor.h"

namespace {

template <bool Backward, typename floatX>
__global__ void mrope_kernel(floatX* out,
                             const floatX* inp,
                             const floatX* freqs_cis,
                             const int* position_ids,
                             int pos_planes,
                             int section_t,
                             int section_h,
                             int section_w,
                             float* abs_max_ptr,
                             int B,
                             int T,
                             int Nq,
                             int Nkv,
                             int head_dim,
                             int rotary_dim,
                             std::bool_constant<Backward> bw = {}) {
    __shared__ float block_abs_max;
    if (abs_max_ptr) {
        if (threadIdx.x == 0) block_abs_max = 0.f;
        __syncthreads();
    }
    float thread_abs_max = 0.f;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = head_dim / 2;
    const int heads = Nq + 2 * Nkv;
    if (idx < B * T * heads * half) {
        const int d = idx % half;
        const int h = (idx / half) % heads;
        const int token = idx / (half * heads);
        const int b = token / T;
        const int t = token % T;
        const int base = (token * heads + h) * head_dim;
        const int rotary_half = rotary_dim / 2;
        // One thread owns both members of a rotary pair, including in-place
        // execution. Partial rotary pairs within [0, rotary_dim), and the
        // rest of the head (and all value heads) passes through unchanged.
        const bool rotate = h < Nq + Nkv && d < rotary_half;
        const int real_idx = base + (rotate ? d : 2 * d);
        const int imag_idx = rotate ? real_idx + rotary_half : real_idx + 1;
        float real = static_cast<float>(inp[real_idx]);
        float imag = static_cast<float>(inp[imag_idx]);
        if (rotate) {
            int pos = t;
            if (position_ids) {
                int plane = 0;
                if (pos_planes > 1) {
                    // Interleaving is per frequency, not per vector load.
                    if (d % 3 == 1 && d < section_h * 3) plane = 1;
                    else if (d % 3 == 2 && d < section_w * 3) plane = 2;
                }
                pos = position_ids[(plane * B + b) * T + t];
            }
            const int freq = pos * rotary_dim + 2 * d;
            const float cosine = static_cast<float>(freqs_cis[freq]);
            const float sine = static_cast<float>(freqs_cis[freq + 1]) * (Backward ? -1.f : 1.f);
            const float rotated_real = real * cosine - imag * sine;
            imag = real * sine + imag * cosine;
            real = rotated_real;
        }
        out[real_idx] = static_cast<floatX>(real);
        out[imag_idx] = static_cast<floatX>(imag);
        thread_abs_max = fmaxf(fabsf(real), fabsf(imag));
    }

    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);
}

// Wrapper for kernel launch
template <bool Backward, typename floatX>
void mrope_imp(floatX* out,
               const floatX* in,
               const floatX* freqs_cis,
               const int* position_ids,
               int pos_planes,
               int section_t,
               int section_h,
               int section_w,
               float* abs_max_ptr,
               int B,
               int T,
               int Nq,
               int Nkv,
               int head_dim,
               int rotary_dim,
               cudaStream_t stream,
               std::bool_constant<Backward> bw = {}) {
    int head_dim_half = head_dim / 2;
    int N = Nq + 2 * Nkv;
    int total_threads = B * T * N * head_dim_half;
    if (total_threads <= 0) return;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    mrope_kernel<Backward><<<num_blocks, block_size, 0, stream>>>(out,
                                                                  in,
                                                                  freqs_cis,
                                                                  position_ids,
                                                                  pos_planes,
                                                                  section_t,
                                                                  section_h,
                                                                  section_w,
                                                                  abs_max_ptr,
                                                                  B,
                                                                  T,
                                                                  Nq,
                                                                  Nkv,
                                                                  head_dim,
                                                                  rotary_dim,
                                                                  bw);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void mrope_forward(Tensor& out,
                   const Tensor& inp,
                   const Tensor& freqs_cis,
                   const int* position_ids,
                   int pos_planes,
                   int section_t,
                   int section_h,
                   int section_w,
                   float* abs_max_ptr,
                   int B,
                   int T,
                   int Nq,
                   int Nkv,
                   int head_dim,
                   int rotary_dim,
                   cudaStream_t stream) {
    if (out.DType == ETensorDType::BF16) {
        mrope_imp(out.get<nv_bfloat16>(),
                  inp.get<nv_bfloat16>(),
                  freqs_cis.get<nv_bfloat16>(),
                  position_ids,
                  pos_planes,
                  section_t,
                  section_h,
                  section_w,
                  abs_max_ptr,
                  B,
                  T,
                  Nq,
                  Nkv,
                  head_dim,
                  rotary_dim,
                  stream,
                  std::bool_constant<false>());
    } else if (out.DType == ETensorDType::FP32) {
        mrope_imp(out.get<float>(),
                  inp.get<float>(),
                  freqs_cis.get<float>(),
                  position_ids,
                  pos_planes,
                  section_t,
                  section_h,
                  section_w,
                  abs_max_ptr,
                  B,
                  T,
                  Nq,
                  Nkv,
                  head_dim,
                  rotary_dim,
                  stream,
                  std::bool_constant<false>());
    } else {
        throw std::logic_error("mrope_forward: unsupported dtype");
    }
}

void mrope_backward(Tensor& dinp,
                    const Tensor& dout,
                    const Tensor& freqs_cis,
                    const int* position_ids,
                    int pos_planes,
                    int section_t,
                    int section_h,
                    int section_w,
                    float* abs_max_ptr,
                    int B,
                    int T,
                    int Nq,
                    int Nkv,
                    int head_dim,
                    int rotary_dim,
                    cudaStream_t stream) {
    if (dinp.DType == ETensorDType::BF16) {
        mrope_imp(dinp.get<nv_bfloat16>(),
                  dout.get<nv_bfloat16>(),
                  freqs_cis.get<nv_bfloat16>(),
                  position_ids,
                  pos_planes,
                  section_t,
                  section_h,
                  section_w,
                  abs_max_ptr,
                  B,
                  T,
                  Nq,
                  Nkv,
                  head_dim,
                  rotary_dim,
                  stream,
                  std::bool_constant<true>());
    } else if (dinp.DType == ETensorDType::FP32) {
        mrope_imp(dinp.get<float>(),
                  dout.get<float>(),
                  freqs_cis.get<float>(),
                  position_ids,
                  pos_planes,
                  section_t,
                  section_h,
                  section_w,
                  abs_max_ptr,
                  B,
                  T,
                  Nq,
                  Nkv,
                  head_dim,
                  rotary_dim,
                  stream,
                  std::bool_constant<true>());
    } else {
        throw std::logic_error("mrope_backward: unsupported dtype");
    }
}
