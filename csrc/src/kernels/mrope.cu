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

template<bool Backward, typename floatX>
__global__ void mrope_kernel(floatX *out, const floatX *inp, const floatX *freqs_cis, const int* position_ids, int pos_planes,
                             int section_t, int section_h, int section_w, float* abs_max_ptr,
                             int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim,
                             std::bool_constant<Backward> bw = {}) {
    using x64 = GenericVector<floatX, 8/sizeof(floatX)>;
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    __shared__ float block_abs_max;
    if (abs_max_ptr) {
        if(threadIdx.x == 0)
            block_abs_max = 0.f;
        __syncthreads();
    }
    float thread_abs_max = 0.f;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x64::size;
    int head_dim_half = head_dim / 2;
    int N = Nq + 2*Nkv;
    if (idx >= B * T * N * head_dim_half) return;

    int h = (idx / head_dim_half) % N;
    int qkv = 2;
    if(h < Nq) {
        qkv = 0;
    } else if (h < Nq + Nkv) {
        qkv = 1;
        h -= Nq;
    }

    if (qkv == 2) {
        if(abs_max_ptr) {
            x128 val = x128::load_cs(inp + 2 * idx);
            for(int k = 0; k < x128::size; k++) {
                thread_abs_max = fmaxf(thread_abs_max, fabsf(val[k]));
            }
            if (out != inp) {
                val.store(out + 2 * idx);
            }
        } else {
            if (out != inp) {
                x128::load_cs(inp + 2 * idx).store(out + 2 * idx);
            }
            return;
        }
    } else {
        int b = idx / (T * N * head_dim_half);
        int t = (idx / (N * head_dim_half)) % T;
        int d = idx % head_dim_half;

        int t_pos = t;
        if (position_ids) {
            if (pos_planes <= 1) {
                t_pos = position_ids[b * T + t];
            } else {
                int dim = 0;
                int mod = d % 3;
                int len_h = section_h * 3;
                int len_w = section_w * 3;
                if (mod == 1 && d < len_h) {
                    dim = 1;
                } else if (mod == 2 && d < len_w) {
                    dim = 2;
                }
                t_pos = position_ids[(dim * B + b) * T + t];
            }
        }

        int idx_bt = b * (T * N * head_dim) + t * (N * head_dim);
        int idx_bth = idx_bt + qkv * (Nq * head_dim) + h * head_dim;
        int idxi = idx_bth + d;

        int rotary_dim_half = rotary_dim / 2;
        x64 v_real = x64::load(inp + idxi);
        x64 v_imag = x64::load(inp + idxi + head_dim_half);
        x64 o_real;
        x64 o_imag;

        if (d + x64::size <= rotary_dim_half) {
            x128 freqs_vec = x128::load_ldg(freqs_cis + t_pos * rotary_dim + 2 * d);
            #pragma unroll
            for (int k = 0; k < x64::size; ++k) {
                float cos = (float)freqs_vec[2 * k];
                float sin = (float)freqs_vec[2 * k + 1];
                if constexpr (Backward) {
                    sin = -sin;
                }
                float real = (float)v_real[k];
                float imag = (float)v_imag[k];
                o_real[k] = real * cos - imag * sin;
                o_imag[k] = real * sin + imag * cos;
            }
        } else {
            o_real = v_real;
            o_imag = v_imag;
        }

        if(abs_max_ptr) {
            for(int i = 0; i < x64::size; i++) {
                thread_abs_max = fmaxf(thread_abs_max, fabsf(o_real[i]));
                thread_abs_max = fmaxf(thread_abs_max, fabsf(o_imag[i]));
            }
        }

        o_real.store(out + idxi);
        o_imag.store(out + idxi + head_dim_half);
    }

    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);
}


// Wrapper for kernel launch
 template<bool Backward, typename floatX>
 void mrope_imp(floatX* out, const floatX* in, const floatX *freqs_cis, const int* position_ids, int pos_planes,
                int section_t, int section_h, int section_w, float* abs_max_ptr,
                int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream,
                std::bool_constant<Backward> bw = {}) {
    int head_dim_half = head_dim / 2;
    int N = Nq + 2*Nkv;
    int total_threads = (B * T * N * head_dim_half) / (8/sizeof(floatX));
    if (total_threads <= 0) return;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    mrope_kernel<Backward><<<num_blocks, block_size, 0, stream>>>(
        out, in, freqs_cis, position_ids, pos_planes,
        section_t, section_h, section_w, abs_max_ptr,
        B, T, Nq, Nkv, head_dim, rotary_dim, bw);
    CUDA_CHECK(cudaGetLastError());
 }

} // namespace

void mrope_forward(Tensor& out, const Tensor& inp, const Tensor& freqs_cis, const int* position_ids, int pos_planes,
                   int section_t, int section_h, int section_w, float* abs_max_ptr,
                   int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream) {
    if (out.DType == ETensorDType::BF16) {
        mrope_imp(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), freqs_cis.get<nv_bfloat16>(), position_ids, pos_planes,
                  section_t, section_h, section_w, abs_max_ptr,
                  B, T, Nq, Nkv, head_dim, rotary_dim, stream, std::bool_constant<false>());
    } else if (out.DType == ETensorDType::FP32) {
        mrope_imp(out.get<float>(), inp.get<float>(), freqs_cis.get<float>(), position_ids, pos_planes,
                  section_t, section_h, section_w, abs_max_ptr,
                  B, T, Nq, Nkv, head_dim, rotary_dim, stream, std::bool_constant<false>());
    } else {
        throw std::logic_error("mrope_forward: unsupported dtype");
    }
}

void mrope_backward(Tensor& dinp, const Tensor& dout, const Tensor& freqs_cis, const int* position_ids, int pos_planes,
                    int section_t, int section_h, int section_w, float* abs_max_ptr,
                    int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream) {
    if (dinp.DType == ETensorDType::BF16) {
        mrope_imp(dinp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), freqs_cis.get<nv_bfloat16>(), position_ids, pos_planes,
                  section_t, section_h, section_w, abs_max_ptr,
                  B, T, Nq, Nkv, head_dim, rotary_dim, stream, std::bool_constant<true>());
    } else if (dinp.DType == ETensorDType::FP32) {
        mrope_imp(dinp.get<float>(), dout.get<float>(), freqs_cis.get<float>(), position_ids, pos_planes,
                  section_t, section_h, section_w, abs_max_ptr,
                  B, T, Nq, Nkv, head_dim, rotary_dim, stream, std::bool_constant<true>());
    } else {
        throw std::logic_error("mrope_backward: unsupported dtype");
    }
}
