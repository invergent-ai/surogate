// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_expert_bias.cu.

// ============================================================================
// MoE per-expert bias add
// ============================================================================
template <typename T>
__global__ void moe_expert_bias_add_forward_kernel(T* __restrict__ out,
                                                   const T* __restrict__ inp,
                                                   const T* __restrict__ bias,
                                                   const int* __restrict__ expert_offsets,
                                                   int num_experts,
                                                   int hidden_size) {
    int e = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts || d >= hidden_size) return;
    int start = expert_offsets[e];
    int end = expert_offsets[e + 1];
    float b = static_cast<float>(bias[e * hidden_size + d]);
    for (int t = start; t < end; ++t) {
        int idx = t * hidden_size + d;
        float v = static_cast<float>(inp[idx]) + b;
        out[idx] = static_cast<T>(v);
    }
}

template <typename T>
__global__ void moe_expert_bias_add_backward_kernel(T* __restrict__ d_inp,
                                                    float* __restrict__ d_bias,
                                                    const T* __restrict__ d_out,
                                                    const int* __restrict__ expert_offsets,
                                                    int num_experts,
                                                    int hidden_size) {
    int e = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts || d >= hidden_size) return;
    int start = expert_offsets[e];
    int end = expert_offsets[e + 1];
    float sum = 0.0f;
    for (int t = start; t < end; ++t) {
        int idx = t * hidden_size + d;
        float v = static_cast<float>(d_out[idx]);
        if (d_inp) {
            d_inp[idx] = static_cast<T>(v);
        }
        sum += v;
    }
    d_bias[e * hidden_size + d] = sum;
}

void moe_expert_bias_add_forward(float* out,
                                 const float* inp,
                                 const float* bias,
                                 const int* expert_offsets,
                                 int num_experts,
                                 int hidden_size,
                                 int total_tokens,
                                 cudaStream_t stream) {
    (void)total_tokens;
    if (num_experts <= 0 || hidden_size <= 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((hidden_size + block.x - 1) / block.x, num_experts, 1);
    moe_expert_bias_add_forward_kernel<<<grid, block, 0, stream>>>(out,
                                                                   inp,
                                                                   bias,
                                                                   expert_offsets,
                                                                   num_experts,
                                                                   hidden_size);
}

void moe_expert_bias_add_forward(nv_bfloat16* out,
                                 const nv_bfloat16* inp,
                                 const nv_bfloat16* bias,
                                 const int* expert_offsets,
                                 int num_experts,
                                 int hidden_size,
                                 int total_tokens,
                                 cudaStream_t stream) {
    (void)total_tokens;
    if (num_experts <= 0 || hidden_size <= 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((hidden_size + block.x - 1) / block.x, num_experts, 1);
    moe_expert_bias_add_forward_kernel<<<grid, block, 0, stream>>>(out,
                                                                   inp,
                                                                   bias,
                                                                   expert_offsets,
                                                                   num_experts,
                                                                   hidden_size);
}

void moe_expert_bias_add_backward(float* d_inp,
                                  float* d_bias,
                                  const float* d_out,
                                  const int* expert_offsets,
                                  int num_experts,
                                  int hidden_size,
                                  int total_tokens,
                                  cudaStream_t stream) {
    (void)total_tokens;
    if (num_experts <= 0 || hidden_size <= 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((hidden_size + block.x - 1) / block.x, num_experts, 1);
    moe_expert_bias_add_backward_kernel<<<grid, block, 0, stream>>>(d_inp,
                                                                    d_bias,
                                                                    d_out,
                                                                    expert_offsets,
                                                                    num_experts,
                                                                    hidden_size);
}

void moe_expert_bias_add_backward(nv_bfloat16* d_inp,
                                  float* d_bias,
                                  const nv_bfloat16* d_out,
                                  const int* expert_offsets,
                                  int num_experts,
                                  int hidden_size,
                                  int total_tokens,
                                  cudaStream_t stream) {
    (void)total_tokens;
    if (num_experts <= 0 || hidden_size <= 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((hidden_size + block.x - 1) / block.x, num_experts, 1);
    moe_expert_bias_add_backward_kernel<<<grid, block, 0, stream>>>(d_inp,
                                                                    d_bias,
                                                                    d_out,
                                                                    expert_offsets,
                                                                    num_experts,
                                                                    hidden_size);
}
