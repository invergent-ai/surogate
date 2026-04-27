// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_gpt_oss_activation.cu.

// ============================================================================
// GPT-OSS MoE Activation (interleaved gate/up)
// ============================================================================
template <typename T>
__global__ void gpt_oss_moe_act_forward_kernel(T* __restrict__ out,        // (N, D)
                                               const T* __restrict__ inp,  // (N, 2*D) interleaved [gate, up]
                                               int total_elements,         // N * D
                                               int D,
                                               float alpha,
                                               float limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int d = idx % D;
    int n = idx / D;
    int base = n * (2 * D) + 2 * d;
    float gate = static_cast<float>(inp[base]);
    float up = static_cast<float>(inp[base + 1]);
    if (gate > limit) gate = limit;
    if (up > limit) up = limit;
    if (up < -limit) up = -limit;
    float sig = 1.0f / (1.0f + expf(-alpha * gate));
    float glu = gate * sig;
    float out_val = (up + 1.0f) * glu;
    out[idx] = static_cast<T>(out_val);
}

template <typename T>
__global__ void gpt_oss_moe_act_backward_kernel(T* __restrict__ d_inp,        // (N, 2*D) interleaved [gate, up]
                                                const T* __restrict__ d_out,  // (N, D)
                                                const T* __restrict__ inp,    // (N, 2*D) interleaved [gate, up]
                                                int total_elements,           // N * D
                                                int D,
                                                float alpha,
                                                float limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int d = idx % D;
    int n = idx / D;
    int base = n * (2 * D) + 2 * d;

    float gate_in = static_cast<float>(inp[base]);
    float up_in = static_cast<float>(inp[base + 1]);
    float gate = gate_in;
    if (gate > limit) gate = limit;
    float up = up_in;
    if (up > limit) up = limit;
    if (up < -limit) up = -limit;

    float sig = 1.0f / (1.0f + expf(-alpha * gate));
    float glu = gate * sig;

    float d_out_val = static_cast<float>(d_out[idx]);

    float d_up = d_out_val * glu;
    if (up_in > limit || up_in < -limit) {
        d_up = 0.0f;
    }

    float sig_deriv = sig * (1.0f - sig);
    float fprime = sig + alpha * gate * sig_deriv;
    float d_gate = d_out_val * (up + 1.0f) * fprime;
    if (gate_in > limit) {
        d_gate = 0.0f;
    }

    d_inp[base] = static_cast<T>(d_gate);
    d_inp[base + 1] = static_cast<T>(d_up);
}

void gpt_oss_moe_act_forward(nv_bfloat16* out,
                             const nv_bfloat16* inp,
                             int N,
                             int D,
                             float alpha,
                             float limit,
                             cudaStream_t stream) {
    const int total = N * D;
    if (total == 0) return;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gpt_oss_moe_act_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, total, D, alpha, limit);
    CUDA_CHECK(cudaGetLastError());
}

void gpt_oss_moe_act_forward(float* out,
                             const float* inp,
                             int N,
                             int D,
                             float alpha,
                             float limit,
                             cudaStream_t stream) {
    const int total = N * D;
    if (total == 0) return;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gpt_oss_moe_act_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, total, D, alpha, limit);
    CUDA_CHECK(cudaGetLastError());
}

void gpt_oss_moe_act_backward(nv_bfloat16* d_inp,
                              const nv_bfloat16* d_out,
                              const nv_bfloat16* inp,
                              int N,
                              int D,
                              float alpha,
                              float limit,
                              cudaStream_t stream) {
    const int total = N * D;
    if (total == 0) return;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gpt_oss_moe_act_backward_kernel<<<grid_size, block_size, 0, stream>>>(d_inp, d_out, inp, total, D, alpha, limit);
    CUDA_CHECK(cudaGetLastError());
}

void gpt_oss_moe_act_backward(float* d_inp,
                              const float* d_out,
                              const float* inp,
                              int N,
                              int D,
                              float alpha,
                              float limit,
                              cudaStream_t stream) {
    const int total = N * D;
    if (total == 0) return;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gpt_oss_moe_act_backward_kernel<<<grid_size, block_size, 0, stream>>>(d_inp, d_out, inp, total, D, alpha, limit);
    CUDA_CHECK(cudaGetLastError());
}
