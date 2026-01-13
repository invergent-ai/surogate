// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mixture of Experts (MoE) CUDA Kernels
//
// This file implements high-performance kernels for MoE routing and expert computation:
// - Softmax/Sigmoid routing activation
// - Top-K expert selection
// - Token permutation (dispatch tokens to expert order)
// - Token unpermutation (gather outputs back to token order)
// - Auxiliary load-balancing loss computation
//
// Design philosophy (inspired by Unsloth grouped GEMM optimizations):
// - Fuse permutation into GEMM prologue/epilogue when possible
// - Use persistent kernel patterns for expert iteration
// - Support both softmax (standard MoE) and sigmoid (DeepSeek-style) routing

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cfloat>

#include "kernels/kernels.h"
#include "kernel_utils.cuh"
#include "utilities/utils.h"

// ============================================================================
// Softmax Kernel for MoE Routing
// ============================================================================
// Computes row-wise softmax over routing logits: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
// Each row corresponds to one token, each column to one expert.
// Optimized for small num_experts (typical: 8-256 experts).

template<typename T, int BLOCK_SIZE = 256>
__global__ void moe_softmax_forward_kernel(
    T* __restrict__ out,              // (num_tokens, num_experts)
    const T* __restrict__ inp,         // (num_tokens, num_experts)
    int num_tokens,
    int num_experts
) {
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
template<typename T>
__global__ void moe_sigmoid_forward_kernel(
    T* __restrict__ out,
    const T* __restrict__ inp,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = static_cast<float>(inp[idx]);
    float sigmoid_val = 1.0f / (1.0f + expf(-val));
    out[idx] = static_cast<T>(sigmoid_val);
}

// ============================================================================
// Top-K Selection Kernel
// ============================================================================
// Selects top-K experts per token based on routing scores.
// Outputs: expert indices (int32) and routing weights (float/bf16).

// Simple single-threaded top-K selection kernel (best for small num_experts < 64)
// For larger num_experts, consider using bitonic sort or radix select
template<typename T, int MAX_K = 8>
__global__ void moe_topk_forward_kernel(
    int* __restrict__ expert_indices,      // (num_tokens, top_k)
    T* __restrict__ routing_weights,       // (num_tokens, top_k)
    const T* __restrict__ scores,          // (num_tokens, num_experts)
    int num_tokens,
    int num_experts,
    int top_k,
    bool normalize_weights
) {
    // One thread per token for simplicity and correctness
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    const T* token_scores = scores + token_idx * num_experts;
    int* token_indices = expert_indices + token_idx * top_k;
    T* token_weights = routing_weights + token_idx * top_k;

    // Thread-local top-K tracking using insertion sort
    float topk_vals[MAX_K];
    int topk_idx[MAX_K];

    // Initialize with -inf
    #pragma unroll
    for (int k = 0; k < MAX_K; k++) {
        topk_vals[k] = -FLT_MAX;
        topk_idx[k] = -1;
    }

    // Scan through all experts and maintain top-K
    for (int e = 0; e < num_experts; e++) {
        float val = static_cast<float>(token_scores[e]);

        // Check if this value should be inserted into top-K
        if (val > topk_vals[top_k - 1]) {
            // Find insertion position (values are sorted descending)
            int insert_pos = top_k - 1;
            for (int k = 0; k < top_k - 1; k++) {
                if (val > topk_vals[k]) {
                    insert_pos = k;
                    break;
                }
            }

            // Shift elements down to make room
            for (int k = top_k - 1; k > insert_pos; k--) {
                topk_vals[k] = topk_vals[k - 1];
                topk_idx[k] = topk_idx[k - 1];
            }

            // Insert new element
            topk_vals[insert_pos] = val;
            topk_idx[insert_pos] = e;
        }
    }

    // Optionally normalize weights to sum to 1
    float sum = 0.0f;
    if (normalize_weights) {
        for (int k = 0; k < top_k; k++) {
            sum += topk_vals[k];
        }
        sum = fmaxf(sum, 1e-9f);
    }

    // Write output
    for (int k = 0; k < top_k; k++) {
        token_indices[k] = topk_idx[k];
        float weight = normalize_weights ? (topk_vals[k] / sum) : topk_vals[k];
        token_weights[k] = static_cast<T>(weight);
    }
}

// ============================================================================
// Top-K Backward Kernel
// ============================================================================
// Backward through top-k selection (treating indices as constants).
//
// Forward (when normalize_weights=true):
//   p = softmax_probs[token, :]
//   p_k = p[idx_k]
//   S = sum_k p_k
//   w_k = p_k / S
//
// Given d_w_k, compute sparse gradients for the selected probs:
//   d_p_k = (d_w_k * S - sum_j d_w_j * p_j) / (S * S)
// Non-selected experts receive zero gradient.
//
// For normalize_weights=false, w_k = p_k and d_p_k = d_w_k.
__global__ void moe_topk_backward_kernel(
    float* __restrict__ d_probs,              // (num_tokens, num_experts)
    const float* __restrict__ d_routing_w,    // (num_tokens, top_k)
    const float* __restrict__ probs,          // (num_tokens, num_experts)
    const int* __restrict__ expert_indices,   // (num_tokens, top_k)
    int num_tokens,
    int num_experts,
    int top_k,
    bool normalize_weights
) {
    constexpr int MAX_K = 8;
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;
    if (top_k <= 0 || top_k > MAX_K) return;

    float* d_row = d_probs + token_idx * num_experts;
    const float* p_row = probs + token_idx * num_experts;
    const float* d_w_row = d_routing_w + token_idx * top_k;
    const int* idx_row = expert_indices + token_idx * top_k;

    if (!normalize_weights) {
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            if (k >= top_k) break;
            int e = idx_row[k];
            if (e >= 0 && e < num_experts) {
                d_row[e] = d_w_row[k];
            }
        }
        return;
    }

    float p_k[MAX_K];
    float sum_p = 0.0f;
    float dot = 0.0f;

    #pragma unroll
    for (int k = 0; k < MAX_K; ++k) {
        if (k >= top_k) break;
        int e = idx_row[k];
        float p = (e >= 0 && e < num_experts) ? p_row[e] : 0.0f;
        p_k[k] = p;
        sum_p += p;
        dot += d_w_row[k] * p;
    }

    // S should be > 0 (sum of selected probs), but clamp for safety.
    sum_p = fmaxf(sum_p, 1e-20f);
    float inv_s2 = 1.0f / (sum_p * sum_p);

    #pragma unroll
    for (int k = 0; k < MAX_K; ++k) {
        if (k >= top_k) break;
        int e = idx_row[k];
        if (e >= 0 && e < num_experts) {
            float d_p = (d_w_row[k] * sum_p - dot) * inv_s2;
            d_row[e] = d_p;
        }
    }
}

// ============================================================================
// Token Permutation / Dispatch Kernels
// ============================================================================
// Reorders tokens from natural order to expert-grouped order for efficient GEMM.
// Also computes histograms of tokens per expert.

// Compute histogram of tokens per expert
__global__ void moe_compute_expert_counts_kernel(
    int* __restrict__ expert_counts,       // (num_experts,) output
    const int* __restrict__ expert_indices, // (num_tokens, top_k)
    int num_tokens,
    int top_k,
    int num_experts
) {
    // Use atomics for simplicity; for high perf, use CUB histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_assignments = num_tokens * top_k;

    if (idx >= total_assignments) return;

    int expert_id = expert_indices[idx];
    if (expert_id >= 0 && expert_id < num_experts) {
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// Compute gather indices that reorder tokens to expert-grouped order
// This is the key data structure for fused permute operations
__global__ void moe_compute_gather_indices_kernel(
    int* __restrict__ gather_indices,      // (total_tokens,) output: index of token in original order
    int* __restrict__ scatter_indices,     // (total_tokens,) output: inverse mapping
    const int* __restrict__ expert_indices, // (num_tokens, top_k)
    const int* __restrict__ expert_offsets, // (num_experts + 1,) cumsum of expert_counts
    int* __restrict__ expert_positions,    // (num_experts,) current write position per expert
    int num_tokens,
    int top_k,
    int num_experts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_assignments = num_tokens * top_k;

    if (idx >= total_assignments) return;

    int expert_id = expert_indices[idx];
    if (expert_id < 0 || expert_id >= num_experts) return;

    // Atomically claim a slot in the expert's region
    int slot = atomicAdd(&expert_positions[expert_id], 1);
    int dest_idx = expert_offsets[expert_id] + slot;

    gather_indices[dest_idx] = idx;  // Token assignment idx -> goes to position dest_idx
    scatter_indices[idx] = dest_idx; // Inverse mapping
}

// Permute hidden states from token order to expert-grouped order
template<typename T>
__global__ void moe_permute_tokens_kernel(
    T* __restrict__ out,                   // (total_tokens, hidden_size)
    const T* __restrict__ inp,             // (num_tokens, hidden_size)
    const int* __restrict__ gather_indices, // (total_tokens,)
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k
) {
    int out_idx = blockIdx.x;
    if (out_idx >= total_tokens) return;

    // Which token (original) to read from
    int token_assignment_idx = gather_indices[out_idx];
    int token_idx = token_assignment_idx / top_k;  // Original token index

    // Copy hidden state
    const T* src = inp + token_idx * hidden_size;
    T* dst = out + out_idx * hidden_size;

    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        dst[d] = src[d];
    }
}

// Unpermute and weight-combine expert outputs back to token order
template<typename T>
__global__ void moe_unpermute_and_combine_kernel(
    T* __restrict__ out,                    // (num_tokens, hidden_size)
    const T* __restrict__ expert_out,       // (total_tokens, hidden_size)
    const T* __restrict__ routing_weights,  // (num_tokens, top_k)
    const int* __restrict__ scatter_indices, // (total_tokens,)
    int num_tokens,
    int total_tokens,
    int hidden_size,
    int top_k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    T* dst = out + token_idx * hidden_size;
    const T* weights = routing_weights + token_idx * top_k;

    // Zero initialize output
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        float acc = 0.0f;

        // Accumulate weighted expert outputs
        for (int k = 0; k < top_k; k++) {
            int assignment_idx = token_idx * top_k + k;
            int expert_pos = scatter_indices[assignment_idx];
            float weight = static_cast<float>(weights[k]);
            float val = static_cast<float>(expert_out[expert_pos * hidden_size + d]);
            acc += weight * val;
        }

        dst[d] = static_cast<T>(acc);
    }
}

// ============================================================================
// Auxiliary Loss Computation
// ============================================================================
// Load-balancing loss to encourage uniform expert utilization.
// aux_loss = alpha * num_experts * sum_e(f_e * P_e)
// where f_e = fraction of tokens routed to expert e
//       P_e = average routing probability to expert e

template<typename T>
__global__ void moe_aux_loss_kernel(
    float* __restrict__ aux_loss,          // scalar output
    float* __restrict__ router_z_loss,     // scalar output (optional)
    const T* __restrict__ routing_probs,   // (num_tokens, num_experts) - post softmax
    const int* __restrict__ expert_indices, // (num_tokens, top_k)
    int num_tokens,
    int num_experts,
    int top_k,
    float aux_loss_coef,
    float z_loss_coef
) {
    // This kernel computes both load-balancing loss and router z-loss
    // For simplicity, use atomics; production should use proper reductions

    extern __shared__ float smem[];
    float* expert_fractions = smem;                    // num_experts
    float* expert_probs = smem + num_experts;          // num_experts

    // Initialize shared memory
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        expert_fractions[e] = 0.0f;
        expert_probs[e] = 0.0f;
    }
    __syncthreads();

    // Compute expert fractions (tokens assigned / total assignments)
    int total_assignments = num_tokens * top_k;
    for (int i = threadIdx.x; i < total_assignments; i += blockDim.x) {
        int expert_id = expert_indices[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&expert_fractions[expert_id], 1.0f / total_assignments);
        }
    }

    // Compute average routing probability per expert
    for (int t = threadIdx.x; t < num_tokens; t += blockDim.x) {
        for (int e = 0; e < num_experts; e++) {
            float prob = static_cast<float>(routing_probs[t * num_experts + e]);
            atomicAdd(&expert_probs[e], prob / num_tokens);
        }
    }
    __syncthreads();

    // Compute load-balancing loss
    if (threadIdx.x == 0) {
        float load_balance_loss = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            load_balance_loss += expert_fractions[e] * expert_probs[e];
        }
        load_balance_loss *= num_experts * aux_loss_coef;
        atomicAdd(aux_loss, load_balance_loss);
    }

    // Compute router z-loss (optional): encourages smaller logits
    // z_loss = (1/num_tokens) * sum_t(log(sum_e(exp(logits))))^2
    // This requires the pre-softmax logits, so we skip it here
    // A separate kernel or the softmax kernel should compute this
}

// ============================================================================
// Router Z-Loss Kernel
// ============================================================================
// Z-loss encourages smaller router logits to prevent instability.
// z_loss = coef * (1/num_tokens) * sum_t(logsumexp(logits_t))^2
//
// The logsumexp is computed as: max + log(sum(exp(x - max)))
// This is numerically stable and avoids overflow.

template<typename T, int BLOCK_SIZE = 256>
__global__ void moe_router_z_loss_kernel(
    float* __restrict__ z_loss,           // scalar output (accumulated via atomicAdd)
    const T* __restrict__ router_logits,  // (num_tokens, num_experts) - pre-softmax
    int num_tokens,
    int num_experts,
    float z_loss_coef
) {
    // Each block processes one token (row)
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const T* row = router_logits + token_idx * num_experts;

    // Step 1: Find max for numerical stability (logsumexp trick)
    float thread_max = -FLT_MAX;
    for (int e = threadIdx.x; e < num_experts; e += BLOCK_SIZE) {
        float val = static_cast<float>(row[e]);
        thread_max = fmaxf(thread_max, val);
    }

    // Warp-level reduction for max
    float row_max = warpReduceMax(thread_max);

    // Block-level reduction using shared memory
    __shared__ float smem[32];
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

    // Step 2: Compute sum(exp(x - max))
    float thread_sum = 0.0f;
    for (int e = threadIdx.x; e < num_experts; e += BLOCK_SIZE) {
        float val = static_cast<float>(row[e]);
        thread_sum += expf(val - row_max);
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

    // Step 3: Compute logsumexp = max + log(sum)
    // Then square it for z-loss contribution
    if (threadIdx.x == 0) {
        float logsumexp = row_max + logf(row_sum + 1e-9f);
        float z_contribution = logsumexp * logsumexp;
        // Scale by coefficient and normalize by num_tokens
        atomicAdd(z_loss, z_loss_coef * z_contribution / num_tokens);
    }
}

// Z-loss backward kernel
// d_logits = coef * (2 * logsumexp / num_tokens) * softmax(logits)
template<typename T, int BLOCK_SIZE = 256>
__global__ void moe_router_z_loss_backward_kernel(
    T* __restrict__ d_logits,             // (num_tokens, num_experts) - gradient output
    const T* __restrict__ router_logits,  // (num_tokens, num_experts) - pre-softmax
    int num_tokens,
    int num_experts,
    float z_loss_coef
) {
    // Each block processes one token (row)
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const T* row_in = router_logits + token_idx * num_experts;
    T* row_out = d_logits + token_idx * num_experts;

    // Step 1: Find max for numerical stability
    float thread_max = -FLT_MAX;
    for (int e = threadIdx.x; e < num_experts; e += BLOCK_SIZE) {
        float val = static_cast<float>(row_in[e]);
        thread_max = fmaxf(thread_max, val);
    }

    float row_max = warpReduceMax(thread_max);

    __shared__ float smem[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) smem[warp_id] = row_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? smem[lane_id] : -FLT_MAX;
        row_max = warpReduceMax(val);
        if (lane_id == 0) smem[0] = row_max;
    }
    __syncthreads();
    row_max = smem[0];

    // Step 2: Compute sum(exp(x - max))
    float thread_sum = 0.0f;
    for (int e = threadIdx.x; e < num_experts; e += BLOCK_SIZE) {
        float val = static_cast<float>(row_in[e]);
        thread_sum += expf(val - row_max);
    }

    float row_sum = warpReduceSum(thread_sum);

    if (lane_id == 0) smem[warp_id] = row_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? smem[lane_id] : 0.0f;
        row_sum = warpReduceSum(val);
        if (lane_id == 0) smem[0] = row_sum;
    }
    __syncthreads();
    row_sum = smem[0];

    // Step 3: Compute gradient scale factor
    // d_z_loss/d_logits = coef * (2 * logsumexp / num_tokens) * softmax(logits)
    float logsumexp = row_max + logf(row_sum + 1e-9f);
    float scale = z_loss_coef * 2.0f * logsumexp / num_tokens;
    float inv_sum = 1.0f / (row_sum + 1e-9f);

    // Step 4: Write gradients
    for (int e = threadIdx.x; e < num_experts; e += BLOCK_SIZE) {
        float val = static_cast<float>(row_in[e]);
        float softmax_val = expf(val - row_max) * inv_sum;
        row_out[e] = static_cast<T>(scale * softmax_val);
    }
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

void moe_softmax_forward(
    nv_bfloat16* out,
    const nv_bfloat16* inp,
    int num_tokens,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_softmax_forward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        out, inp, num_tokens, num_experts
    );
}

void moe_softmax_forward(
    float* out,
    const float* inp,
    int num_tokens,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_softmax_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(
        out, inp, num_tokens, num_experts
    );
}

void moe_sigmoid_forward(
    nv_bfloat16* out,
    const nv_bfloat16* inp,
    int num_elements,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_sigmoid_forward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        out, inp, num_elements
    );
}

void moe_sigmoid_forward(
    float* out,
    const float* inp,
    int num_elements,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_sigmoid_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(
        out, inp, num_elements
    );
}

void moe_topk_forward(
    int* expert_indices,
    nv_bfloat16* routing_weights,
    const nv_bfloat16* scores,
    int num_tokens,
    int num_experts,
    int top_k,
    bool normalize_weights,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;
    moe_topk_forward_kernel<nv_bfloat16, 8><<<grid_size, block_size, 0, stream>>>(
        expert_indices, routing_weights, scores,
        num_tokens, num_experts, top_k, normalize_weights
    );
}

void moe_topk_forward(
    int* expert_indices,
    float* routing_weights,
    const float* scores,
    int num_tokens,
    int num_experts,
    int top_k,
    bool normalize_weights,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;
    moe_topk_forward_kernel<float, 8><<<grid_size, block_size, 0, stream>>>(
        expert_indices, routing_weights, scores,
        num_tokens, num_experts, top_k, normalize_weights
    );
}

void moe_topk_backward(
    float* d_probs,
    const float* d_routing_weights,
    const float* probs,
    const int* expert_indices,
    int num_tokens,
    int num_experts,
    int top_k,
    bool normalize_weights,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;
    moe_topk_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_probs, d_routing_weights, probs, expert_indices,
        num_tokens, num_experts, top_k, normalize_weights
    );
}

void moe_compute_expert_counts(
    int* expert_counts,
    const int* expert_indices,
    int num_tokens,
    int top_k,
    int num_experts,
    cudaStream_t stream
) {
    // Zero the output first
    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);

    int block_size = 256;
    int total = num_tokens * top_k;
    int grid_size = (total + block_size - 1) / block_size;
    moe_compute_expert_counts_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_counts, expert_indices, num_tokens, top_k, num_experts
    );
}

// Compute exclusive prefix sum of expert_counts into expert_offsets (length num_experts + 1).
// num_experts is small (typically <= 128), so a single-thread kernel is sufficient and avoids
// alignment/temporary-storage pitfalls of generic scan implementations.
__global__ void moe_compute_expert_offsets_kernel(
    int* __restrict__ expert_offsets,
    const int* __restrict__ expert_counts,
    int num_experts
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int sum = 0;
    expert_offsets[0] = 0;
    for (int e = 0; e < num_experts; ++e) {
        sum += expert_counts[e];
        expert_offsets[e + 1] = sum;
    }
}

void moe_compute_expert_offsets(
    int* expert_offsets,
    const int* expert_counts,
    int num_experts,
    cudaStream_t stream
) {
    // Compute exclusive prefix sum of expert_counts.
    // We intentionally use a tiny single-thread kernel here:
    // - num_experts is small (typically <= 128)
    // - avoids CUB alignment pitfalls (expert_offsets + 1 may be misaligned)
    // - avoids per-call cudaMallocAsync/cudaFreeAsync overhead
    moe_compute_expert_offsets_kernel<<<1, 1, 0, stream>>>(expert_offsets, expert_counts, num_experts);
    CUDA_CHECK(cudaGetLastError());
}

void moe_build_indices(
    int* gather_indices,
    int* scatter_indices,
    const int* expert_indices,
    const int* expert_offsets,
    int* expert_positions,
    int num_tokens,
    int top_k,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int total = num_tokens * top_k;
    int grid_size = (total + block_size - 1) / block_size;

    moe_compute_gather_indices_kernel<<<grid_size, block_size, 0, stream>>>(
        gather_indices, scatter_indices, expert_indices, expert_offsets,
        expert_positions, num_tokens, top_k, num_experts
    );
}

// ============================================================================
// Expert Index Remapping Kernel for Selective Dequantization
// ============================================================================

__global__ void moe_remap_expert_indices_kernel(
    int* __restrict__ remapped_indices,
    const int* __restrict__ expert_indices,
    const int* __restrict__ expert_to_compact,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int global_expert = expert_indices[idx];
    int compact_expert = expert_to_compact[global_expert];
    remapped_indices[idx] = compact_expert;
}

void moe_remap_expert_indices(
    int* remapped_indices,
    const int* expert_indices,
    const int* expert_to_compact,
    int num_tokens,
    int top_k,
    cudaStream_t stream
) {
    int total = num_tokens * top_k;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    moe_remap_expert_indices_kernel<<<grid_size, block_size, 0, stream>>>(
        remapped_indices, expert_indices, expert_to_compact, total
    );
}

void moe_permute_tokens(
    nv_bfloat16* out,
    const nv_bfloat16* inp,
    const int* gather_indices,
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = total_tokens;
    moe_permute_tokens_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        out, inp, gather_indices, total_tokens, num_tokens, hidden_size, top_k
    );
}

void moe_permute_tokens(
    float* out,
    const float* inp,
    const int* gather_indices,
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = total_tokens;
    moe_permute_tokens_kernel<float><<<grid_size, block_size, 0, stream>>>(
        out, inp, gather_indices, total_tokens, num_tokens, hidden_size, top_k
    );
}

void moe_unpermute_and_combine(
    nv_bfloat16* out,
    const nv_bfloat16* expert_out,
    const nv_bfloat16* routing_weights,
    const int* scatter_indices,
    int num_tokens,
    int total_tokens,
    int hidden_size,
    int top_k,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_unpermute_and_combine_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        out, expert_out, routing_weights, scatter_indices,
        num_tokens, total_tokens, hidden_size, top_k
    );
}

void moe_unpermute_and_combine(
    float* out,
    const float* expert_out,
    const float* routing_weights,
    const int* scatter_indices,
    int num_tokens,
    int total_tokens,
    int hidden_size,
    int top_k,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_unpermute_and_combine_kernel<float><<<grid_size, block_size, 0, stream>>>(
        out, expert_out, routing_weights, scatter_indices,
        num_tokens, total_tokens, hidden_size, top_k
    );
}

void moe_compute_aux_loss(
    float* aux_loss,
    const nv_bfloat16* routing_probs,
    const int* expert_indices,
    int num_tokens,
    int num_experts,
    int top_k,
    float aux_loss_coef,
    cudaStream_t stream
) {
    // Initialize output
    cudaMemsetAsync(aux_loss, 0, sizeof(float), stream);

    int block_size = 256;
    int shared_mem = 2 * num_experts * sizeof(float);
    moe_aux_loss_kernel<nv_bfloat16><<<1, block_size, shared_mem, stream>>>(
        aux_loss, nullptr, routing_probs, expert_indices,
        num_tokens, num_experts, top_k, aux_loss_coef, 0.0f
    );
}

void moe_compute_aux_loss(
    float* aux_loss,
    const float* routing_probs,
    const int* expert_indices,
    int num_tokens,
    int num_experts,
    int top_k,
    float aux_loss_coef,
    cudaStream_t stream
) {
    cudaMemsetAsync(aux_loss, 0, sizeof(float), stream);

    int block_size = 256;
    int shared_mem = 2 * num_experts * sizeof(float);
    moe_aux_loss_kernel<float><<<1, block_size, shared_mem, stream>>>(
        aux_loss, nullptr, routing_probs, expert_indices,
        num_tokens, num_experts, top_k, aux_loss_coef, 0.0f
    );
}

void moe_router_z_loss_forward(
    float* z_loss,
    const nv_bfloat16* router_logits,
    int num_tokens,
    int num_experts,
    float z_loss_coef,
    cudaStream_t stream
) {
    // Initialize output to zero (will be accumulated via atomicAdd)
    cudaMemsetAsync(z_loss, 0, sizeof(float), stream);

    int block_size = 256;
    int grid_size = num_tokens;
    moe_router_z_loss_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        z_loss, router_logits, num_tokens, num_experts, z_loss_coef
    );
}

void moe_router_z_loss_forward(
    float* z_loss,
    const float* router_logits,
    int num_tokens,
    int num_experts,
    float z_loss_coef,
    cudaStream_t stream
) {
    cudaMemsetAsync(z_loss, 0, sizeof(float), stream);

    int block_size = 256;
    int grid_size = num_tokens;
    moe_router_z_loss_kernel<float><<<grid_size, block_size, 0, stream>>>(
        z_loss, router_logits, num_tokens, num_experts, z_loss_coef
    );
}

void moe_router_z_loss_backward(
    nv_bfloat16* d_logits,
    const nv_bfloat16* router_logits,
    int num_tokens,
    int num_experts,
    float z_loss_coef,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_router_z_loss_backward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        d_logits, router_logits, num_tokens, num_experts, z_loss_coef
    );
}

void moe_router_z_loss_backward(
    float* d_logits,
    const float* router_logits,
    int num_tokens,
    int num_experts,
    float z_loss_coef,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_router_z_loss_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_logits, router_logits, num_tokens, num_experts, z_loss_coef
    );
}

// ============================================================================
// Grouped GEMM for MoE Expert Computation
// ============================================================================
// Uses cuBLAS grouped batched GEMM to run all experts in parallel.
// This reduces kernel launch overhead from O(num_experts) to O(1).
//
// The expert weights are stored in a batched layout:
//   gate_up_proj: (num_experts, 2*D, C)
//   down_proj:    (num_experts, C, D)
//
// Input tokens are permuted to expert-grouped order, with expert_offsets[e]
// pointing to where expert e's tokens start.

// Helper to get cuBLAS data type from C++ type
template<typename T>
constexpr cudaDataType_t cublas_dtype() {
    if constexpr (std::is_same_v<T, float>) return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, nv_bfloat16>) return CUDA_R_16BF;
    else if constexpr (std::is_same_v<T, half>) return CUDA_R_16F;
    else static_assert(!sizeof(T), "Unsupported type for cuBLAS");
}

// Kernel to build pointer arrays on device (avoids host-device sync)
template<typename T>
__global__ void build_gemm_pointers_gate_up_kernel(
    const T** A_ptrs,           // output: input pointers
    const T** B_ptrs,           // output: weight pointers
    T** C_ptrs,                 // output: output pointers
    int* lda_arr,
    int* ldb_arr,
    int* ldc_arr,
    int* m_arr,
    int* n_arr,
    int* k_arr,
    const T* input,
    const T* weights,
    T* output,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,            // C
    int intermediate_size       // D
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts) return;

    int start = expert_offsets[e];
    int end = expert_offsets[e + 1];
    int tokens_e = end - start;

    // Input: (tokens_e, C) at offset start * C
    A_ptrs[e] = input + start * hidden_size;
    // Weight: (2*D, C) for expert e
    B_ptrs[e] = weights + e * (2 * intermediate_size) * hidden_size;
    // Output: (tokens_e, 2*D) at offset start * 2*D
    C_ptrs[e] = output + start * (2 * intermediate_size);

    // Row-major: output(tokens, 2*D) = input(tokens, C) @ weight^T(C, 2*D)
    // In column-major (treating row-major as col-major):
    // - input becomes (C, tokens) col-major
    // - weight becomes (C, 2*D) col-major
    // - output becomes (2*D, tokens) col-major
    //
    // So: output(2*D, tokens) = weight^T(2*D, C) @ input(C, tokens)
    // cuBLAS: C = op(A) @ op(B)
    // A = weight with CUBLAS_OP_T: op(A) = (2*D, C)
    // B = input with CUBLAS_OP_N: op(B) = (C, tokens)
    // M = 2*D, N = tokens, K = C

    m_arr[e] = 2 * intermediate_size;  // M = 2*D
    n_arr[e] = tokens_e;               // N = tokens
    k_arr[e] = hidden_size;            // K = C
    lda_arr[e] = hidden_size;          // lda = C (leading dim of weight in col-major)
    ldb_arr[e] = hidden_size;          // ldb = C (leading dim of input in col-major)
    ldc_arr[e] = 2 * intermediate_size; // ldc = 2*D (leading dim of output in col-major)
}

template<typename T>
__global__ void build_gemm_pointers_down_kernel(
    const T** A_ptrs,
    const T** B_ptrs,
    T** C_ptrs,
    int* lda_arr,
    int* ldb_arr,
    int* ldc_arr,
    int* m_arr,
    int* n_arr,
    int* k_arr,
    const T* input,
    const T* weights,
    T* output,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,            // C
    int intermediate_size       // D
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts) return;

    int start = expert_offsets[e];
    int end = expert_offsets[e + 1];
    int tokens_e = end - start;

    // Input: (tokens_e, D) at offset start * D
    A_ptrs[e] = input + start * intermediate_size;
    // Weight: (C, D) for expert e
    B_ptrs[e] = weights + e * hidden_size * intermediate_size;
    // Output: (tokens_e, C) at offset start * C
    C_ptrs[e] = output + start * hidden_size;

    // Row-major: output(tokens, C) = input(tokens, D) @ weight^T(D, C)
    // Col-major: output(C, tokens) = weight^T(C, D) @ input(D, tokens)
    // A = weight with CUBLAS_OP_T: op(A) = (C, D)
    // B = input with CUBLAS_OP_N: op(B) = (D, tokens)
    // M = C, N = tokens, K = D

    m_arr[e] = hidden_size;       // M = C
    n_arr[e] = tokens_e;          // N = tokens
    k_arr[e] = intermediate_size; // K = D
    lda_arr[e] = intermediate_size; // lda = D
    ldb_arr[e] = intermediate_size; // ldb = D
    ldc_arr[e] = hidden_size;       // ldc = C
}

template<typename T>
void moe_grouped_gemm_impl(
    T* output,
    const T* input,
    const T* weights,
    const int* expert_offsets,
    int num_experts,
    int M,
    int K,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    float alpha,
    float beta,
    EMMTranspose mode,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    std::vector<int> local_offsets;
    const int* h_offsets;

    if (host_offsets) {
        h_offsets = host_offsets;
    } else {
        local_offsets.resize(num_experts + 1);
        CUDA_CHECK(cudaMemcpyAsync(local_offsets.data(), expert_offsets,
                                   (num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_offsets = local_offsets.data();
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    cublasOperation_t transa = (mode == EMMTranspose::TN || mode == EMMTranspose::TT) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (mode == EMMTranspose::NT || mode == EMMTranspose::TT) ? CUBLAS_OP_T : CUBLAS_OP_N;

    std::vector<int> m_vec, n_vec, k_vec;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<const T*> A_vec, B_vec;
    std::vector<T*> C_vec;

    m_vec.reserve(n_active);
    n_vec.reserve(n_active);
    k_vec.reserve(n_active);
    lda_vec.reserve(n_active);
    ldb_vec.reserve(n_active);
    ldc_vec.reserve(n_active);
    A_vec.reserve(n_active);
    B_vec.reserve(n_active);
    C_vec.reserve(n_active);

    for (int e = 0; e < n_active; ++e) {
        int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
        if (tokens_e == 0) continue;

        m_vec.push_back(M);
        n_vec.push_back(tokens_e);
        k_vec.push_back(K);

        // Row-major A(M, K) @ B(K, N) = C(M, N)
        // In column-major: C(M, N) = A(M, K) @ B(K, N)
        // transa on A, transb on B
        lda_vec.push_back((transa == CUBLAS_OP_N) ? M : K);
        ldb_vec.push_back((transb == CUBLAS_OP_N) ? K : tokens_e);
        ldc_vec.push_back(M);

        A_vec.push_back(weights + (weight_is_compact ? e : global_idx) * M * K);
        B_vec.push_back(input + h_offsets[global_idx] * K);
        C_vec.push_back(output + h_offsets[global_idx] * M);
    }

    if (m_vec.empty()) return;

    const int gemm_count = static_cast<int>(m_vec.size());

    // cublasGemmGroupedBatchedEx requires pointer arrays to be in device memory
    const T** d_A_array = nullptr;
    const T** d_B_array = nullptr;
    T** d_C_array = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_A_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_B_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_C_array), sizeof(T*) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, A_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, B_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, C_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));

    std::vector<cublasOperation_t> transa_vec(gemm_count, transa);
    std::vector<cublasOperation_t> transb_vec(gemm_count, transb);
    std::vector<int> group_size_vec(gemm_count, 1);
    std::vector<float> alpha_vec(gemm_count, alpha);
    std::vector<float> beta_vec(gemm_count, beta);

    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
        cublas_handle,
        transa_vec.data(), transb_vec.data(),
        m_vec.data(), n_vec.data(), k_vec.data(),
        alpha_vec.data(),
        reinterpret_cast<const void**>(d_A_array), cublas_dtype<T>(), lda_vec.data(),
        reinterpret_cast<const void**>(d_B_array), cublas_dtype<T>(), ldb_vec.data(),
        beta_vec.data(),
        reinterpret_cast<void**>(d_C_array), cublas_dtype<T>(), ldc_vec.data(),
        gemm_count,
        group_size_vec.data(),
        CUBLAS_COMPUTE_32F
    ));

    // Free device pointer arrays
    CUDA_CHECK(cudaFreeAsync(d_A_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_array, stream));
}

void moe_grouped_gemm(float* output, const float* input, const float* weights,
                      const int* expert_offsets, int num_experts,
                      int M, int K,
                      cublasHandle_t cublas_handle, cudaStream_t stream,
                      const int* host_offsets,
                      float alpha, float beta, EMMTranspose mode,
                      const int* active_expert_indices,
                      bool weight_is_compact,
                      int num_active_experts) {
    moe_grouped_gemm_impl(output, input, weights, expert_offsets, num_experts, M, K, cublas_handle, stream, host_offsets, alpha, beta, mode, active_expert_indices, weight_is_compact, num_active_experts);
}

void moe_grouped_gemm(nv_bfloat16* output, const nv_bfloat16* input, const nv_bfloat16* weights,
                      const int* expert_offsets, int num_experts,
                      int M, int K,
                      cublasHandle_t cublas_handle, cudaStream_t stream,
                      const int* host_offsets,
                      float alpha, float beta, EMMTranspose mode,
                      const int* active_expert_indices,
                      bool weight_is_compact,
                      int num_active_experts) {
    moe_grouped_gemm_impl(output, input, weights, expert_offsets, num_experts, M, K, cublas_handle, stream, host_offsets, alpha, beta, mode, active_expert_indices, weight_is_compact, num_active_experts);
}

template<typename T>
void moe_grouped_gemm_weight_grad_impl(
    T* d_weight,
    const T* grad_output,
    const T* input,
    const int* expert_offsets,
    int num_experts,
    int M,
    int N,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    float alpha,
    float beta,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    std::vector<int> local_offsets;
    const int* h_offsets;

    if (host_offsets) {
        h_offsets = host_offsets;
    } else {
        local_offsets.resize(num_experts + 1);
        CUDA_CHECK(cudaMemcpyAsync(local_offsets.data(), expert_offsets,
                                   (num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_offsets = local_offsets.data();
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    // dW(M, N) = grad_output^T(M, K) @ input(K, N)  where K = tokens_e
    // In column-major: dW(M, N) = A @ B
    // A is grad_output treated as (K, M) col-major => A^T is (M, K)
    // B is input treated as (K, N) col-major => B is (K, N)

    std::vector<int> m_vec, n_vec, k_vec;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<const T*> A_vec, B_vec;
    std::vector<T*> C_vec;

    m_vec.reserve(n_active);
    n_vec.reserve(n_active);
    k_vec.reserve(n_active);
    lda_vec.reserve(n_active);
    ldb_vec.reserve(n_active);
    ldc_vec.reserve(n_active);
    A_vec.reserve(n_active);
    B_vec.reserve(n_active);
    C_vec.reserve(n_active);

    for (int e = 0; e < n_active; ++e) {
        int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
        if (tokens_e == 0) continue;

        m_vec.push_back(M);
        n_vec.push_back(N);
        k_vec.push_back(tokens_e);

        lda_vec.push_back(M);
        ldb_vec.push_back(N);
        ldc_vec.push_back(M);

        // Row-major grad_output is (tokens, M). Treated as col-major it's (M, tokens).
        // Transpose A (CUBLAS_OP_T) gives (M, tokens)? NO.
        // If row-major (tokens, M) is treated as col-major (M, tokens),
        // we want result (M, N).
        // C(M, N) = A(M, K) @ B(K, N)
        // A is grad_output(M, K) col-major. OP_N.
        // B is input(K, N) col-major. OP_T?
        // Row-major input is (K, N). Treated as col-major it's (N, K).
        // OP_T on B gives (K, N).
        // So: C(M, N) = A(M, K) @ B^T(K, N)

        A_vec.push_back(grad_output + h_offsets[global_idx] * M);
        B_vec.push_back(input + h_offsets[global_idx] * N);
        C_vec.push_back(d_weight + (weight_is_compact ? e : global_idx) * M * N);
    }

    if (m_vec.empty()) return;

    const int gemm_count = static_cast<int>(m_vec.size());

    // cublasGemmGroupedBatchedEx requires pointer arrays to be in device memory
    const T** d_A_array = nullptr;
    const T** d_B_array = nullptr;
    T** d_C_array = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_A_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_B_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_C_array), sizeof(T*) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, A_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, B_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, C_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_N);
    std::vector<cublasOperation_t> transb_vec(gemm_count, CUBLAS_OP_T);
    std::vector<int> group_size_vec(gemm_count, 1);
    std::vector<float> alpha_vec(gemm_count, alpha);
    std::vector<float> beta_vec(gemm_count, beta);

    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
        cublas_handle,
        transa_vec.data(), transb_vec.data(),
        m_vec.data(), n_vec.data(), k_vec.data(),
        alpha_vec.data(),
        reinterpret_cast<const void**>(d_A_array), cublas_dtype<T>(), lda_vec.data(),
        reinterpret_cast<const void**>(d_B_array), cublas_dtype<T>(), ldb_vec.data(),
        beta_vec.data(),
        reinterpret_cast<void**>(d_C_array), cublas_dtype<T>(), ldc_vec.data(),
        gemm_count,
        group_size_vec.data(),
        CUBLAS_COMPUTE_32F
    ));

    // Free device pointer arrays
    CUDA_CHECK(cudaFreeAsync(d_A_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_array, stream));
}

void moe_grouped_gemm_weight_grad(float* d_weight, const float* grad_output, const float* input,
                                  const int* expert_offsets, int num_experts,
                                  int M, int N,
                                  cublasHandle_t cublas_handle, cudaStream_t stream,
                                  const int* host_offsets,
                                  float alpha, float beta,
                                  const int* active_expert_indices,
                                  bool weight_is_compact,
                                  int num_active_experts) {
    moe_grouped_gemm_weight_grad_impl(d_weight, grad_output, input, expert_offsets, num_experts, M, N, cublas_handle, stream, host_offsets, alpha, beta, active_expert_indices, weight_is_compact, num_active_experts);
}

void moe_grouped_gemm_weight_grad(nv_bfloat16* d_weight, const nv_bfloat16* grad_output, const nv_bfloat16* input,
                                  const int* expert_offsets, int num_experts,
                                  int M, int N,
                                  cublasHandle_t cublas_handle, cudaStream_t stream,
                                  const int* host_offsets,
                                  float alpha, float beta,
                                  const int* active_expert_indices,
                                  bool weight_is_compact,
                                  int num_active_experts) {
    moe_grouped_gemm_weight_grad_impl(d_weight, grad_output, input, expert_offsets, num_experts, M, N, cublas_handle, stream, host_offsets, alpha, beta, active_expert_indices, weight_is_compact, num_active_experts);
}

template<typename T>
void moe_grouped_gemm_gate_up_impl(
    T* output,                        // (total_tokens, 2*D) - gate+up output
    const T* input,                   // (total_tokens, C) - permuted tokens
    const T* weights,                 // (num_experts, 2*D, C) - batched weights
    const int* expert_offsets,        // (num_experts + 1) - token offsets per expert (device)
    int num_experts,
    int hidden_size,                  // C
    int intermediate_size,            // D (output is 2*D for gate+up)
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,          // Optional: pre-cached host offsets to avoid D2H sync
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    // Get host-side offsets - either use cached or copy from device
    std::vector<int> local_offsets;
    const int* h_offsets;

    if (host_offsets) {
        // Use pre-cached host offsets (no sync needed)
        h_offsets = host_offsets;
    } else {
        // Copy from device (requires sync - slower path)
        local_offsets.resize(num_experts + 1);
        CUDA_CHECK(cudaMemcpyAsync(local_offsets.data(), expert_offsets,
                                   (num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_offsets = local_offsets.data();
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int out_dim = 2 * intermediate_size;

    // Use Grouped GEMM to submit all expert computations in a single call.
    // This significantly reduces CPU overhead and kernel launch latency compared to a loop.
    std::vector<int> m_vec, n_vec, k_vec;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<const T*> A_vec, B_vec;
    std::vector<T*> C_vec;

    m_vec.reserve(n_active);
    n_vec.reserve(n_active);
    k_vec.reserve(n_active);
    lda_vec.reserve(n_active);
    ldb_vec.reserve(n_active);
    ldc_vec.reserve(n_active);
    A_vec.reserve(n_active);
    B_vec.reserve(n_active);
    C_vec.reserve(n_active);

    for (int e = 0; e < n_active; ++e) {
        int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
        if (tokens_e == 0) continue;

        const T* A_ptr = weights + (weight_is_compact ? e : global_idx) * out_dim * hidden_size;
        const T* B_ptr = input + h_offsets[global_idx] * hidden_size;
        T* C_ptr = output + h_offsets[global_idx] * out_dim;

        m_vec.push_back(out_dim);
        n_vec.push_back(tokens_e);
        k_vec.push_back(hidden_size);

        lda_vec.push_back(hidden_size);
        ldb_vec.push_back(hidden_size);
        ldc_vec.push_back(out_dim);

        A_vec.push_back(A_ptr);
        B_vec.push_back(B_ptr);
        C_vec.push_back(C_ptr);
    }

    if (m_vec.empty()) return;

    const int gemm_count = static_cast<int>(m_vec.size());

    // cublasGemmGroupedBatchedEx requires pointer arrays to be in device memory
    const T** d_A_array = nullptr;
    const T** d_B_array = nullptr;
    T** d_C_array = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_A_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_B_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_C_array), sizeof(T*) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, A_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, B_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, C_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_T);
    std::vector<cublasOperation_t> transb_vec(gemm_count, CUBLAS_OP_N);
    std::vector<int> group_size_vec(gemm_count, 1);
    std::vector<float> alpha_vec(gemm_count, alpha);
    std::vector<float> beta_vec(gemm_count, beta);

    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
        cublas_handle,
        transa_vec.data(), transb_vec.data(),
        m_vec.data(), n_vec.data(), k_vec.data(),
        alpha_vec.data(),
        reinterpret_cast<const void**>(d_A_array), cublas_dtype<T>(), lda_vec.data(),
        reinterpret_cast<const void**>(d_B_array), cublas_dtype<T>(), ldb_vec.data(),
        beta_vec.data(),
        reinterpret_cast<void**>(d_C_array), cublas_dtype<T>(), ldc_vec.data(),
        gemm_count,
        group_size_vec.data(),
        CUBLAS_COMPUTE_32F
    ));

    // Free device pointer arrays
    CUDA_CHECK(cudaFreeAsync(d_A_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_array, stream));
}

template<typename T>
void moe_grouped_gemm_down_impl(
    T* output,                        // (total_tokens, C) - down proj output
    const T* input,                   // (total_tokens, D) - SwiGLU output
    const T* weights,                 // (num_experts, C, D) - batched weights
    const int* expert_offsets,        // (num_experts + 1) - token offsets per expert (device)
    int num_experts,
    int hidden_size,                  // C
    int intermediate_size,            // D
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,          // Optional: pre-cached host offsets to avoid D2H sync
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    // Get host-side offsets - either use cached or copy from device
    std::vector<int> local_offsets;
    const int* h_offsets;

    if (host_offsets) {
        h_offsets = host_offsets;
    } else {
        local_offsets.resize(num_experts + 1);
        CUDA_CHECK(cudaMemcpyAsync(local_offsets.data(), expert_offsets,
                                   (num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_offsets = local_offsets.data();
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use Grouped GEMM to submit all expert computations in a single call.
    std::vector<int> m_vec, n_vec, k_vec;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<const T*> A_vec, B_vec;
    std::vector<T*> C_vec;

    m_vec.reserve(n_active);
    n_vec.reserve(n_active);
    k_vec.reserve(n_active);
    lda_vec.reserve(n_active);
    ldb_vec.reserve(n_active);
    ldc_vec.reserve(n_active);
    A_vec.reserve(n_active);
    B_vec.reserve(n_active);
    C_vec.reserve(n_active);

    for (int e = 0; e < n_active; ++e) {
        int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
        if (tokens_e == 0) continue;

        m_vec.push_back(hidden_size);
        n_vec.push_back(tokens_e);
        k_vec.push_back(intermediate_size);

        lda_vec.push_back(intermediate_size);
        ldb_vec.push_back(intermediate_size);
        ldc_vec.push_back(hidden_size);

        A_vec.push_back(weights + (weight_is_compact ? e : global_idx) * hidden_size * intermediate_size);
        B_vec.push_back(input + h_offsets[global_idx] * intermediate_size);
        C_vec.push_back(output + h_offsets[global_idx] * hidden_size);
    }

    if (m_vec.empty()) return;

    const int gemm_count = static_cast<int>(m_vec.size());

    // cublasGemmGroupedBatchedEx requires pointer arrays to be in device memory
    const T** d_A_array = nullptr;
    const T** d_B_array = nullptr;
    T** d_C_array = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_A_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_B_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_C_array), sizeof(T*) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, A_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, B_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, C_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_T);
    std::vector<cublasOperation_t> transb_vec(gemm_count, CUBLAS_OP_N);
    std::vector<int> group_size_vec(gemm_count, 1);
    std::vector<float> alpha_vec(gemm_count, alpha);
    std::vector<float> beta_vec(gemm_count, beta);

    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
        cublas_handle,
        transa_vec.data(), transb_vec.data(),
        m_vec.data(), n_vec.data(), k_vec.data(),
        alpha_vec.data(),
        reinterpret_cast<const void**>(d_A_array), cublas_dtype<T>(), lda_vec.data(),
        reinterpret_cast<const void**>(d_B_array), cublas_dtype<T>(), ldb_vec.data(),
        beta_vec.data(),
        reinterpret_cast<void**>(d_C_array), cublas_dtype<T>(), ldc_vec.data(),
        gemm_count,
        group_size_vec.data(),
        CUBLAS_COMPUTE_32F
    ));

    // Free device pointer arrays
    CUDA_CHECK(cudaFreeAsync(d_A_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_array, stream));
}

void moe_grouped_gemm_gate_up(
    nv_bfloat16* output,
    const nv_bfloat16* input,
    const nv_bfloat16* weights,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    moe_grouped_gemm_gate_up_impl(output, input, weights, expert_offsets,
                                   num_experts, hidden_size, intermediate_size,
                                   cublas_handle, stream, host_offsets, active_expert_indices, weight_is_compact, num_active_experts);
}

void moe_grouped_gemm_gate_up(
    float* output,
    const float* input,
    const float* weights,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    moe_grouped_gemm_gate_up_impl(output, input, weights, expert_offsets,
                                   num_experts, hidden_size, intermediate_size,
                                   cublas_handle, stream, host_offsets, active_expert_indices, weight_is_compact, num_active_experts);
}

void moe_grouped_gemm_down(
    nv_bfloat16* output,
    const nv_bfloat16* input,
    const nv_bfloat16* weights,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    moe_grouped_gemm_down_impl(output, input, weights, expert_offsets,
                                num_experts, hidden_size, intermediate_size,
                                cublas_handle, stream, host_offsets, active_expert_indices, weight_is_compact, num_active_experts);
}

void moe_grouped_gemm_down(
    float* output,
    const float* input,
    const float* weights,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    moe_grouped_gemm_down_impl(output, input, weights, expert_offsets,
                                num_experts, hidden_size, intermediate_size,
                                cublas_handle, stream, host_offsets, active_expert_indices, weight_is_compact, num_active_experts);
}

// ============================================================================
// Grouped GEMM Backward for MoE Expert Computation
// ============================================================================
// These compute the backward pass through expert projections:
// - down_backward: d_swiglu = d_output @ down_proj (no transpose on weight)
// - gate_up_backward: d_input = d_gate_up @ gate_up_proj (no transpose on weight)

// Kernel to build pointer arrays for down backward on device
template<typename T>
__global__ void build_gemm_pointers_down_backward_kernel(
    const T** A_ptrs,           // output: d_output pointers
    const T** B_ptrs,           // output: weight pointers
    T** C_ptrs,                 // output: d_input pointers
    int* lda_arr,
    int* ldb_arr,
    int* ldc_arr,
    int* m_arr,
    int* n_arr,
    int* k_arr,
    const T* d_output,
    const T* weights,
    T* d_input,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,            // C
    int intermediate_size       // D
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts) return;

    int start = expert_offsets[e];
    int end = expert_offsets[e + 1];
    int tokens_e = end - start;

    // d_output: (tokens_e, C) at offset start * C
    A_ptrs[e] = d_output + start * hidden_size;
    // Weight: (C, D) for expert e
    B_ptrs[e] = weights + e * hidden_size * intermediate_size;
    // d_input: (tokens_e, D) at offset start * D
    C_ptrs[e] = d_input + start * intermediate_size;

    // For backward: d_input = d_output @ W (no transpose on W)
    // Row-major: d_input[t][d] = sum_c d_output[t][c] * W[c][d]
    //
    // In column-major:
    // - d_output is (C, tokens) col-major
    // - W is (D, C) col-major (because row-major (C, D))
    // - d_input is (D, tokens) col-major
    //
    // So: d_input(D, tokens) = W(D, C) @ d_output(C, tokens)
    // With CUBLAS_OP_N on both: M = D, N = tokens, K = C

    m_arr[e] = intermediate_size;   // M = D
    n_arr[e] = tokens_e;            // N = tokens
    k_arr[e] = hidden_size;         // K = C
    lda_arr[e] = intermediate_size; // lda = D (leading dim of W in col-major)
    ldb_arr[e] = hidden_size;       // ldb = C (leading dim of d_output in col-major)
    ldc_arr[e] = intermediate_size; // ldc = D (leading dim of d_input in col-major)
}

// Kernel to build pointer arrays for gate_up backward on device
template<typename T>
__global__ void build_gemm_pointers_gate_up_backward_kernel(
    const T** A_ptrs,           // output: d_gate_up pointers
    const T** B_ptrs,           // output: weight pointers
    T** C_ptrs,                 // output: d_input pointers
    int* lda_arr,
    int* ldb_arr,
    int* ldc_arr,
    int* m_arr,
    int* n_arr,
    int* k_arr,
    const T* d_gate_up,
    const T* weights,
    T* d_input,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,            // C
    int intermediate_size       // D
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts) return;

    int start = expert_offsets[e];
    int end = expert_offsets[e + 1];
    int tokens_e = end - start;

    // d_gate_up: (tokens_e, 2*D) at offset start * 2*D
    A_ptrs[e] = d_gate_up + start * (2 * intermediate_size);
    // Weight: (2*D, C) for expert e
    B_ptrs[e] = weights + e * (2 * intermediate_size) * hidden_size;
    // d_input: (tokens_e, C) at offset start * C
    C_ptrs[e] = d_input + start * hidden_size;

    // For backward: d_input = d_gate_up @ W (no transpose on W)
    // Row-major: d_input[t][c] = sum_d d_gate_up[t][d] * W[d][c]
    //
    // In column-major:
    // - d_gate_up is (2*D, tokens) col-major
    // - W is (C, 2*D) col-major (because row-major (2*D, C))
    // - d_input is (C, tokens) col-major
    //
    // So: d_input(C, tokens) = W(C, 2*D) @ d_gate_up(2*D, tokens)
    // With CUBLAS_OP_N on both: M = C, N = tokens, K = 2*D

    m_arr[e] = hidden_size;           // M = C
    n_arr[e] = tokens_e;              // N = tokens
    k_arr[e] = 2 * intermediate_size; // K = 2*D
    lda_arr[e] = hidden_size;         // lda = C (leading dim of W in col-major)
    ldb_arr[e] = 2 * intermediate_size; // ldb = 2*D (leading dim of d_gate_up in col-major)
    ldc_arr[e] = hidden_size;         // ldc = C (leading dim of d_input in col-major)
}

template<typename T>
void moe_grouped_gemm_down_backward_impl(
    T* d_input,                       // (total_tokens, D) - gradient w.r.t. SwiGLU output
    const T* d_output,                // (total_tokens, C) - gradient from downstream
    const T* weights,                 // (num_experts, C, D) - down_proj weights
    const int* expert_offsets,        // (num_experts + 1) - token offsets per expert (device)
    int num_experts,
    int hidden_size,                  // C
    int intermediate_size,            // D
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,          // Optional: pre-cached host offsets to avoid D2H sync
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    std::vector<int> local_offsets;
    const int* h_offsets;

    if (host_offsets) {
        h_offsets = host_offsets;
    } else {
        local_offsets.resize(num_experts + 1);
        CUDA_CHECK(cudaMemcpyAsync(local_offsets.data(), expert_offsets,
                                   (num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_offsets = local_offsets.data();
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use Grouped GEMM to submit all expert computations in a single call.
    std::vector<int> m_vec, n_vec, k_vec;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<const T*> A_vec, B_vec;
    std::vector<T*> C_vec;

    m_vec.reserve(n_active);
    n_vec.reserve(n_active);
    k_vec.reserve(n_active);
    lda_vec.reserve(n_active);
    ldb_vec.reserve(n_active);
    ldc_vec.reserve(n_active);
    A_vec.reserve(n_active);
    B_vec.reserve(n_active);
    C_vec.reserve(n_active);

    for (int e = 0; e < n_active; ++e) {
        int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
        if (tokens_e == 0) continue;

        m_vec.push_back(intermediate_size);
        n_vec.push_back(tokens_e);
        k_vec.push_back(hidden_size);

        lda_vec.push_back(intermediate_size);
        ldb_vec.push_back(hidden_size);
        ldc_vec.push_back(intermediate_size);

        A_vec.push_back(weights + (weight_is_compact ? e : global_idx) * hidden_size * intermediate_size);
        B_vec.push_back(d_output + h_offsets[global_idx] * hidden_size);
        C_vec.push_back(d_input + h_offsets[global_idx] * intermediate_size);
    }

    if (m_vec.empty()) return;

    const int gemm_count = static_cast<int>(m_vec.size());

    // cublasGemmGroupedBatchedEx requires pointer arrays to be in device memory
    const T** d_A_array = nullptr;
    const T** d_B_array = nullptr;
    T** d_C_array = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_A_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_B_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_C_array), sizeof(T*) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, A_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, B_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, C_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_N);
    std::vector<cublasOperation_t> transb_vec(gemm_count, CUBLAS_OP_N);
    std::vector<int> group_size_vec(gemm_count, 1);
    std::vector<float> alpha_vec(gemm_count, alpha);
    std::vector<float> beta_vec(gemm_count, beta);

    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
        cublas_handle,
        transa_vec.data(), transb_vec.data(),
        m_vec.data(), n_vec.data(), k_vec.data(),
        alpha_vec.data(),
        reinterpret_cast<const void**>(d_A_array), cublas_dtype<T>(), lda_vec.data(),
        reinterpret_cast<const void**>(d_B_array), cublas_dtype<T>(), ldb_vec.data(),
        beta_vec.data(),
        reinterpret_cast<void**>(d_C_array), cublas_dtype<T>(), ldc_vec.data(),
        gemm_count,
        group_size_vec.data(),
        CUBLAS_COMPUTE_32F
    ));

    // Free device pointer arrays
    CUDA_CHECK(cudaFreeAsync(d_A_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_array, stream));
}

template<typename T>
void moe_grouped_gemm_gate_up_backward_impl(
    T* d_input,                       // (total_tokens, C) - gradient w.r.t. input
    const T* d_gate_up,               // (total_tokens, 2*D) - gradient from SwiGLU backward
    const T* weights,                 // (num_experts, 2*D, C) - gate_up_proj weights
    const int* expert_offsets,        // (num_experts + 1) - token offsets per expert (device)
    int num_experts,
    int hidden_size,                  // C
    int intermediate_size,            // D (d_gate_up is 2*D)
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,          // Optional: pre-cached host offsets to avoid D2H sync
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    std::vector<int> local_offsets;
    const int* h_offsets;

    if (host_offsets) {
        h_offsets = host_offsets;
    } else {
        local_offsets.resize(num_experts + 1);
        CUDA_CHECK(cudaMemcpyAsync(local_offsets.data(), expert_offsets,
                                   (num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_offsets = local_offsets.data();
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int gate_up_dim = 2 * intermediate_size;

    // Use Grouped GEMM to submit all expert computations in a single call.
    std::vector<int> m_vec, n_vec, k_vec;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<const T*> A_vec, B_vec;
    std::vector<T*> C_vec;

    m_vec.reserve(n_active);
    n_vec.reserve(n_active);
    k_vec.reserve(n_active);
    lda_vec.reserve(n_active);
    ldb_vec.reserve(n_active);
    ldc_vec.reserve(n_active);
    A_vec.reserve(n_active);
    B_vec.reserve(n_active);
    C_vec.reserve(n_active);

    for (int e = 0; e < n_active; ++e) {
        int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
        if (tokens_e == 0) continue;

        m_vec.push_back(hidden_size);
        n_vec.push_back(tokens_e);
        k_vec.push_back(gate_up_dim);

        lda_vec.push_back(hidden_size);
        ldb_vec.push_back(gate_up_dim);
        ldc_vec.push_back(hidden_size);

        A_vec.push_back(weights + (weight_is_compact ? e : global_idx) * gate_up_dim * hidden_size);
        B_vec.push_back(d_gate_up + h_offsets[global_idx] * gate_up_dim);
        C_vec.push_back(d_input + h_offsets[global_idx] * hidden_size);
    }

    if (m_vec.empty()) return;

    const int gemm_count = static_cast<int>(m_vec.size());

    // cublasGemmGroupedBatchedEx requires pointer arrays to be in device memory
    const T** d_A_array = nullptr;
    const T** d_B_array = nullptr;
    T** d_C_array = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_A_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_B_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_C_array), sizeof(T*) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, A_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, B_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, C_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_N);
    std::vector<cublasOperation_t> transb_vec(gemm_count, CUBLAS_OP_N);
    std::vector<int> group_size_vec(gemm_count, 1);
    std::vector<float> alpha_vec(gemm_count, alpha);
    std::vector<float> beta_vec(gemm_count, beta);

    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
        cublas_handle,
        transa_vec.data(), transb_vec.data(),
        m_vec.data(), n_vec.data(), k_vec.data(),
        alpha_vec.data(),
        reinterpret_cast<const void**>(d_A_array), cublas_dtype<T>(), lda_vec.data(),
        reinterpret_cast<const void**>(d_B_array), cublas_dtype<T>(), ldb_vec.data(),
        beta_vec.data(),
        reinterpret_cast<void**>(d_C_array), cublas_dtype<T>(), ldc_vec.data(),
        gemm_count,
        group_size_vec.data(),
        CUBLAS_COMPUTE_32F
    ));

    // Free device pointer arrays
    CUDA_CHECK(cudaFreeAsync(d_A_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_array, stream));
}

// Host wrappers for grouped GEMM backward
void moe_grouped_gemm_down_backward(
    nv_bfloat16* d_input,
    const nv_bfloat16* d_output,
    const nv_bfloat16* weights,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    moe_grouped_gemm_down_backward_impl(d_input, d_output, weights, expert_offsets,
                                         num_experts, hidden_size, intermediate_size,
                                         cublas_handle, stream, host_offsets, active_expert_indices, weight_is_compact, num_active_experts);
}

void moe_grouped_gemm_down_backward(
    float* d_input,
    const float* d_output,
    const float* weights,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    moe_grouped_gemm_down_backward_impl(d_input, d_output, weights, expert_offsets,
                                         num_experts, hidden_size, intermediate_size,
                                         cublas_handle, stream, host_offsets, active_expert_indices, weight_is_compact, num_active_experts);
}

void moe_grouped_gemm_gate_up_backward(
    nv_bfloat16* d_input,
    const nv_bfloat16* d_gate_up,
    const nv_bfloat16* weights,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    moe_grouped_gemm_gate_up_backward_impl(d_input, d_gate_up, weights, expert_offsets,
                                            num_experts, hidden_size, intermediate_size,
                                            cublas_handle, stream, host_offsets, active_expert_indices, weight_is_compact, num_active_experts);
}

void moe_grouped_gemm_gate_up_backward(
    float* d_input,
    const float* d_gate_up,
    const float* weights,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int intermediate_size,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts
) {
    moe_grouped_gemm_gate_up_backward_impl(d_input, d_gate_up, weights, expert_offsets,
                                            num_experts, hidden_size, intermediate_size,
                                            cublas_handle, stream, host_offsets, active_expert_indices, weight_is_compact, num_active_experts);
}

// ============================================================================
// Backward Kernels
// ============================================================================

// Softmax backward kernel
// d_logits = softmax_probs * (d_output - sum_j(d_output_j * softmax_probs_j))
template<typename T, int BLOCK_SIZE = 256>
__global__ void moe_softmax_backward_kernel(
    T* __restrict__ d_logits,             // (num_tokens, num_experts)
    const T* __restrict__ d_probs,        // (num_tokens, num_experts) - upstream gradient
    const T* __restrict__ softmax_probs,  // (num_tokens, num_experts)
    int num_tokens,
    int num_experts
) {
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

// Backward through unpermute+combine: scatter gradient to expert outputs
// d_expert_outputs[permuted_idx] = routing_weights[token, k] * d_output[token]
template<typename T>
__global__ void moe_combine_backward_kernel(
    T* __restrict__ d_expert_out,         // (total_tokens, hidden_size)
    T* __restrict__ d_routing_weights,    // (num_tokens, top_k) - optional, can be NULL
    const T* __restrict__ d_output,       // (num_tokens, hidden_size)
    const T* __restrict__ expert_out,     // (total_tokens, hidden_size) - for weight gradient
    const T* __restrict__ routing_weights,// (num_tokens, top_k)
    const int* __restrict__ scatter_indices, // (total_tokens,)
    int num_tokens,
    int total_tokens,
    int hidden_size,
    int top_k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const T* d_out = d_output + token_idx * hidden_size;
    const T* weights = routing_weights + token_idx * top_k;

    // For each expert assigned to this token
    for (int k = 0; k < top_k; k++) {
        int assignment_idx = token_idx * top_k + k;
        int expert_pos = scatter_indices[assignment_idx];

        T* d_exp_out = d_expert_out + expert_pos * hidden_size;
        float weight = static_cast<float>(weights[k]);

        // d_expert_out[expert_pos] = weight * d_output[token]
        for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
            float grad = weight * static_cast<float>(d_out[d]);
            d_exp_out[d] = static_cast<T>(grad);
        }

        // Compute gradient w.r.t. routing weights (if needed)
        // d_routing_weights[token, k] = dot(expert_out[expert_pos], d_output[token])
        if (d_routing_weights != nullptr && threadIdx.x == 0) {
            const T* exp_out = expert_out + expert_pos * hidden_size;
            float dot = 0.0f;
            for (int d = 0; d < hidden_size; d++) {
                dot += static_cast<float>(exp_out[d]) * static_cast<float>(d_out[d]);
            }
            d_routing_weights[assignment_idx] = static_cast<T>(dot);
        }
    }
}

// Backward through permute: gather gradient back to original token order
// d_input[token] += d_permuted[permuted_idx] for each assignment
// FP32 version - uses native atomicAdd
__global__ void moe_permute_backward_kernel_fp32(
    float* __restrict__ d_input,              // (num_tokens, hidden_size)
    const float* __restrict__ d_permuted,     // (total_tokens, hidden_size)
    const int* __restrict__ gather_indices,   // (total_tokens,)
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k
) {
    int out_idx = blockIdx.x;
    if (out_idx >= total_tokens) return;

    // Which token this permuted position corresponds to
    int token_assignment_idx = gather_indices[out_idx];
    int token_idx = token_assignment_idx / top_k;

    const float* d_perm = d_permuted + out_idx * hidden_size;
    float* d_in = d_input + token_idx * hidden_size;

    // Atomically add gradient back to original token
    // (multiple assignments may map to the same token)
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        atomicAdd(d_in + d, d_perm[d]);
    }
}

// BF16 version - uses atomicAdd for __nv_bfloat16 (requires SM80+)
__global__ void moe_permute_backward_kernel_bf16(
    nv_bfloat16* __restrict__ d_input,              // (num_tokens, hidden_size)
    const nv_bfloat16* __restrict__ d_permuted,     // (total_tokens, hidden_size)
    const int* __restrict__ gather_indices,         // (total_tokens,)
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k
) {
    int out_idx = blockIdx.x;
    if (out_idx >= total_tokens) return;

    // Which token this permuted position corresponds to
    int token_assignment_idx = gather_indices[out_idx];
    int token_idx = token_assignment_idx / top_k;

    const nv_bfloat16* d_perm = d_permuted + out_idx * hidden_size;
    nv_bfloat16* d_in = d_input + token_idx * hidden_size;

    // Atomically add gradient back to original token
    // (multiple assignments may map to the same token)
    // Use native BF16 atomicAdd (SM80+)
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        atomicAdd(d_in + d, d_perm[d]);
    }
}

// ============================================================================
// Backward Host Wrapper Functions
// ============================================================================

void moe_softmax_backward(
    nv_bfloat16* d_logits,
    const nv_bfloat16* d_probs,
    const nv_bfloat16* softmax_probs,
    int num_tokens,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_softmax_backward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        d_logits, d_probs, softmax_probs, num_tokens, num_experts
    );
}

void moe_softmax_backward(
    float* d_logits,
    const float* d_probs,
    const float* softmax_probs,
    int num_tokens,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_softmax_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_logits, d_probs, softmax_probs, num_tokens, num_experts
    );
}

void moe_combine_backward(
    nv_bfloat16* d_expert_out,
    nv_bfloat16* d_routing_weights,
    const nv_bfloat16* d_output,
    const nv_bfloat16* expert_out,
    const nv_bfloat16* routing_weights,
    const int* scatter_indices,
    int num_tokens,
    int total_tokens,
    int hidden_size,
    int top_k,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_combine_backward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        d_expert_out, d_routing_weights, d_output, expert_out,
        routing_weights, scatter_indices, num_tokens, total_tokens, hidden_size, top_k
    );
}

void moe_combine_backward(
    float* d_expert_out,
    float* d_routing_weights,
    const float* d_output,
    const float* expert_out,
    const float* routing_weights,
    const int* scatter_indices,
    int num_tokens,
    int total_tokens,
    int hidden_size,
    int top_k,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_combine_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_expert_out, d_routing_weights, d_output, expert_out,
        routing_weights, scatter_indices, num_tokens, total_tokens, hidden_size, top_k
    );
}

void moe_permute_backward(
    nv_bfloat16* d_input,
    const nv_bfloat16* d_permuted,
    const int* gather_indices,
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k,
    cudaStream_t stream
) {
    // Zero the output first (since we're accumulating)
    cudaMemsetAsync(d_input, 0, num_tokens * hidden_size * sizeof(nv_bfloat16), stream);

    int block_size = 256;
    int grid_size = total_tokens;
    moe_permute_backward_kernel_bf16<<<grid_size, block_size, 0, stream>>>(
        d_input, d_permuted, gather_indices, total_tokens, num_tokens, hidden_size, top_k
    );
}

void moe_permute_backward(
    float* d_input,
    const float* d_permuted,
    const int* gather_indices,
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k,
    cudaStream_t stream
) {
    cudaMemsetAsync(d_input, 0, num_tokens * hidden_size * sizeof(float), stream);

    int block_size = 256;
    int grid_size = total_tokens;
    moe_permute_backward_kernel_fp32<<<grid_size, block_size, 0, stream>>>(
        d_input, d_permuted, gather_indices, total_tokens, num_tokens, hidden_size, top_k
    );
}
