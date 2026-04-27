// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_routing_activation.cu.

// ============================================================================
// Softmax Kernel for MoE Routing
// ============================================================================
// Computes row-wise softmax over routing logits: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
// Each row corresponds to one token, each column to one expert.
// Optimized for small num_experts (typical: 8-256 experts).

template <typename T, int BLOCK_SIZE = 256>
__global__ void moe_softmax_forward_kernel(T* __restrict__ out,        // (num_tokens, num_experts)
                                           const T* __restrict__ inp,  // (num_tokens, num_experts)
                                           int num_tokens,
                                           int num_experts) {
    // One block per token (row)
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const T* row_in = inp + token_idx * num_experts;
    T* row_out = out + token_idx * num_experts;

    // Step 1: Find max for numerical stability
    float thread_max = -FLT_MAX;
    for (int i = threadIdx.x; i < num_experts; i += BLOCK_SIZE) {
        float val = static_cast<float>(row_in[i]);
        thread_max = fmaxf(thread_max, val);
    }

    // Warp-level reduction for max
    float row_max = warpReduceMax(thread_max);

    // Block-level reduction using shared memory
    __shared__ float smem[32];  // One per warp
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        smem[warp_id] = row_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? smem[lane_id] : -FLT_MAX;
        row_max = warpReduceMax(val);
        if (lane_id == 0) smem[0] = row_max;
    }
    __syncthreads();
    row_max = smem[0];

    // Step 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < num_experts; i += BLOCK_SIZE) {
        float val = static_cast<float>(row_in[i]);
        float exp_val = expf(val - row_max);
        thread_sum += exp_val;
    }

    // Warp-level reduction for sum
    float row_sum = warpReduceSum(thread_sum);

    if (lane_id == 0) {
        smem[warp_id] = row_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? smem[lane_id] : 0.0f;
        row_sum = warpReduceSum(val);
        if (lane_id == 0) smem[0] = row_sum;
    }
    __syncthreads();
    row_sum = smem[0];

    // Step 3: Normalize
    float inv_sum = 1.0f / (row_sum + 1e-9f);
    for (int i = threadIdx.x; i < num_experts; i += BLOCK_SIZE) {
        float val = static_cast<float>(row_in[i]);
        float softmax_val = expf(val - row_max) * inv_sum;
        row_out[i] = static_cast<T>(softmax_val);
    }
}

// Sigmoid activation for DeepSeek-style routing
template <typename T>
__global__ void moe_sigmoid_forward_kernel(T* __restrict__ out, const T* __restrict__ inp, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = static_cast<float>(inp[idx]);
    float sigmoid_val = 1.0f / (1.0f + expf(-val));
    out[idx] = static_cast<T>(sigmoid_val);
}

template <typename T>
__global__ void moe_sigmoid_backward_kernel(T* __restrict__ d_inp,
                                            const T* __restrict__ grad,
                                            const T* __restrict__ out,
                                            int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float g = static_cast<float>(grad[idx]);
    float y = static_cast<float>(out[idx]);
    float dy = g * y * (1.0f - y);
    d_inp[idx] = static_cast<T>(dy);
}

template <typename T>
__global__ void
moe_scale_forward_kernel(T* __restrict__ out, const T* __restrict__ inp, float scale, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float val = static_cast<float>(inp[idx]);
    out[idx] = static_cast<T>(val * scale);
}

// ============================================================================

void moe_softmax_forward(nv_bfloat16* out,
                         const nv_bfloat16* inp,
                         int num_tokens,
                         int num_experts,
                         cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_softmax_forward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(out, inp, num_tokens, num_experts);
}

void moe_softmax_forward(float* out, const float* inp, int num_tokens, int num_experts, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_softmax_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(out, inp, num_tokens, num_experts);
}

void moe_sigmoid_forward(nv_bfloat16* out, const nv_bfloat16* inp, int num_elements, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_sigmoid_forward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(out, inp, num_elements);
}

void moe_sigmoid_forward(float* out, const float* inp, int num_elements, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_sigmoid_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(out, inp, num_elements);
}

void moe_sigmoid_backward(nv_bfloat16* d_inp,
                          const nv_bfloat16* grad,
                          const nv_bfloat16* out,
                          int num_elements,
                          cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_sigmoid_backward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(d_inp, grad, out, num_elements);
}

void moe_sigmoid_backward(float* d_inp, const float* grad, const float* out, int num_elements, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_sigmoid_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(d_inp, grad, out, num_elements);
}

void moe_scale_forward(nv_bfloat16* out, const nv_bfloat16* inp, float scale, int num_elements, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_scale_forward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(out, inp, scale, num_elements);
}

void moe_scale_forward(float* out, const float* inp, float scale, int num_elements, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_scale_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(out, inp, scale, num_elements);
}

// ============================================================================
// Backward Kernels
// ============================================================================

// Softmax backward kernel
// d_logits = softmax_probs * (d_output - sum_j(d_output_j * softmax_probs_j))
template <typename T, int BLOCK_SIZE = 256>
__global__ void
moe_softmax_backward_kernel(T* __restrict__ d_logits,             // (num_tokens, num_experts)
                            const T* __restrict__ d_probs,        // (num_tokens, num_experts) - upstream gradient
                            const T* __restrict__ softmax_probs,  // (num_tokens, num_experts)
                            int num_tokens,
                            int num_experts) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const T* d_prob_row = d_probs + token_idx * num_experts;
    const T* prob_row = softmax_probs + token_idx * num_experts;
    T* d_logit_row = d_logits + token_idx * num_experts;

    // Compute sum(d_output * softmax_probs) for this token
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < num_experts; i += BLOCK_SIZE) {
        float d_p = static_cast<float>(d_prob_row[i]);
        float p = static_cast<float>(prob_row[i]);
        thread_sum += d_p * p;
    }

    // Warp-level reduction
    float row_sum = warpReduceSum(thread_sum);

    // Block-level reduction
    __shared__ float smem[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) smem[warp_id] = row_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? smem[lane_id] : 0.0f;
        row_sum = warpReduceSum(val);
        if (lane_id == 0) smem[0] = row_sum;
    }
    __syncthreads();
    row_sum = smem[0];

    // Compute gradient: d_logits = softmax_probs * (d_probs - row_sum)
    for (int i = threadIdx.x; i < num_experts; i += BLOCK_SIZE) {
        float d_p = static_cast<float>(d_prob_row[i]);
        float p = static_cast<float>(prob_row[i]);
        float grad = p * (d_p - row_sum);
        d_logit_row[i] = static_cast<T>(grad);
    }
}

// Backward Host Wrapper Functions
// ============================================================================

void moe_softmax_backward(nv_bfloat16* d_logits,
                          const nv_bfloat16* d_probs,
                          const nv_bfloat16* softmax_probs,
                          int num_tokens,
                          int num_experts,
                          cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_softmax_backward_kernel<nv_bfloat16>
        <<<grid_size, block_size, 0, stream>>>(d_logits, d_probs, softmax_probs, num_tokens, num_experts);
}

void moe_softmax_backward(float* d_logits,
                          const float* d_probs,
                          const float* softmax_probs,
                          int num_tokens,
                          int num_experts,
                          cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_softmax_backward_kernel<float>
        <<<grid_size, block_size, 0, stream>>>(d_logits, d_probs, softmax_probs, num_tokens, num_experts);
}
