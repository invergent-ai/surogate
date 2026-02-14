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
// Design philosophy:
// - Fuse permutation into GEMM prologue/epilogue when possible
// - Use persistent kernel patterns for expert iteration
// - Support both softmax (standard MoE) and sigmoid (DeepSeek-style) routing

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cfloat>
#include <climits>

#include "kernels/kernels.h"
#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

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

template<typename T>
__global__ void moe_sigmoid_backward_kernel(
    T* __restrict__ d_inp,
    const T* __restrict__ grad,
    const T* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float g = static_cast<float>(grad[idx]);
    float y = static_cast<float>(out[idx]);
    float dy = g * y * (1.0f - y);
    d_inp[idx] = static_cast<T>(dy);
}

template<typename T>
__global__ void moe_scale_forward_kernel(
    T* __restrict__ out,
    const T* __restrict__ inp,
    float scale,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float val = static_cast<float>(inp[idx]);
    out[idx] = static_cast<T>(val * scale);
}

// ============================================================================
// Top-K Selection Kernel — Warp-Level Tournament
// ============================================================================
// Selects top-K experts per token based on routing scores.
// Uses warp-level parallelism: one warp (32 threads) cooperatively selects the
// top-K experts for a single token. Each lane maintains a local top-K list from
// its portion of experts, then warp shuffles merge all per-lane lists in
// O(K * log(WARP_SIZE)) steps without shared memory synchronization.
//
// Outputs: expert indices (int32) and routing weights (float/bf16).

constexpr int MOE_TOPK_MAX_K = 8;
constexpr int MOE_WARP_SIZE = 32;

// Insert a (val, idx) pair into a descending-sorted register array of size K.
// Replaces the last element and bubbles it up to the correct position.
template<int K>
__device__ __forceinline__ void topk_insert(float* vals, int* idxs, float val, int idx) {
    vals[K - 1] = val;
    idxs[K - 1] = idx;
    #pragma unroll
    for (int j = K - 2; j >= 0; j--) {
        if ((vals[j + 1] > vals[j]) ||
            (vals[j + 1] == vals[j] && idxs[j + 1] < idxs[j])) {
            float tv = vals[j]; vals[j] = vals[j + 1]; vals[j + 1] = tv;
            int ti = idxs[j]; idxs[j] = idxs[j + 1]; idxs[j + 1] = ti;
        }
    }
}

// Warp-level tournament top-K selection.
// Each lane scans its stripe of experts, maintains a local sorted top-K,
// then 5 rounds of shuffle-based merging produce the global top-K on all lanes.
// When correction_bias is non-null, selection is based on (score + bias) but
// the values stored in out_vals are the biased scores (caller must re-read
// original scores after if needed).
template<int K, typename T>
__device__ __forceinline__ void warp_topk(
    const T* __restrict__ token_scores,
    int num_experts,
    float* out_vals,
    int* out_idxs,
    const float* __restrict__ correction_bias = nullptr,
    float rounding_scale = 0.0f
) {
    const int lane = threadIdx.x & (MOE_WARP_SIZE - 1);

    // Per-lane local top-K
    float my_vals[MOE_TOPK_MAX_K];
    int my_idxs[MOE_TOPK_MAX_K];

    #pragma unroll
    for (int i = 0; i < K; i++) {
        my_vals[i] = -FLT_MAX;
        my_idxs[i] = -1;
    }

    // Each lane processes experts at stride MOE_WARP_SIZE
    for (int e = lane; e < num_experts; e += MOE_WARP_SIZE) {
        float val = static_cast<float>(token_scores[e]);
        // Use biased score for selection if correction_bias is provided
        float selection_val = correction_bias ? (val + correction_bias[e]) : val;
        if (rounding_scale > 0.0f) {
            selection_val = nearbyintf(selection_val * rounding_scale) / rounding_scale;
        }
        if (selection_val > my_vals[K - 1] ||
            (selection_val == my_vals[K - 1] && (my_idxs[K - 1] < 0 || e < my_idxs[K - 1]))) {
            topk_insert<K>(my_vals, my_idxs, selection_val, e);
        }
    }

    // Warp-level merge: 5 rounds (log2(32)) of shuffle-based tournament.
    // Each round, exchange top-K lists with a partner lane and merge the two
    // sorted K-lists into a single K-list (keep top K from 2K candidates).
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        // Get partner's top-K via warp shuffle
        float partner_vals[MOE_TOPK_MAX_K];
        int partner_idxs[MOE_TOPK_MAX_K];

        #pragma unroll
        for (int i = 0; i < K; i++) {
            partner_vals[i] = __shfl_xor_sync(0xFFFFFFFFu, my_vals[i], offset);
            partner_idxs[i] = __shfl_xor_sync(0xFFFFFFFFu, my_idxs[i], offset);
        }

        // Two-pointer merge of two sorted-descending K-lists → keep top K
        float merged_vals[MOE_TOPK_MAX_K];
        int merged_idxs[MOE_TOPK_MAX_K];
        int a = 0, b = 0;

        #pragma unroll
        for (int m = 0; m < K; m++) {
            // Pick the larger head element
            bool take_partner = (a >= K) || (b < K && partner_vals[b] > my_vals[a]);
            if (take_partner) {
                merged_vals[m] = partner_vals[b];
                merged_idxs[m] = partner_idxs[b];
                b++;
            } else {
                merged_vals[m] = my_vals[a];
                merged_idxs[m] = my_idxs[a];
                a++;
            }
        }

        #pragma unroll
        for (int i = 0; i < K; i++) {
            my_vals[i] = merged_vals[i];
            my_idxs[i] = merged_idxs[i];
        }
    }

    // After merging, all lanes hold the same global top-K (from lane 0).
    // Broadcast from lane 0 to ensure consistency.
    #pragma unroll
    for (int i = 0; i < K; i++) {
        out_vals[i] = __shfl_sync(0xFFFFFFFFu, my_vals[i], 0);
        out_idxs[i] = __shfl_sync(0xFFFFFFFFu, my_idxs[i], 0);
    }
}

// Warp-per-token top-K kernel. Packs multiple warps per block.
// When correction_bias is non-null, expert selection uses (score + bias) but
// routing weights are computed from the original unbiased scores.
template<typename T, int K>
__global__ void moe_topk_forward_kernel(
    int* __restrict__ expert_indices,      // (num_tokens, top_k)
    T* __restrict__ routing_weights,       // (num_tokens, top_k)
    const T* __restrict__ scores,          // (num_tokens, num_experts)
    const float* __restrict__ correction_bias,  // (num_experts) or nullptr
    int num_tokens,
    int num_experts,
    int top_k,
    bool normalize_weights,
    bool softmax_weights,
    bool sort_by_index,
    float rounding_scale
) {
    const int warps_per_block = blockDim.x / MOE_WARP_SIZE;
    const int warp_id = threadIdx.x / MOE_WARP_SIZE;
    const int lane = threadIdx.x & (MOE_WARP_SIZE - 1);
    const int token_idx = blockIdx.x * warps_per_block + warp_id;
    if (token_idx >= num_tokens) return;

    const T* token_scores = scores + token_idx * num_experts;

    // Warp-cooperative top-K selection (uses biased scores if bias present)
    float topk_vals[MOE_TOPK_MAX_K];
    int topk_idxs[MOE_TOPK_MAX_K];
    warp_topk<K>(token_scores, num_experts, topk_vals, topk_idxs, correction_bias, rounding_scale);

    // If correction_bias was used, topk_vals contain biased scores.
    // Re-read original unbiased scores for the selected experts (used for weights).
    if (correction_bias || rounding_scale > 0.0f) {
        for (int k = 0; k < top_k; k++) {
            int idx = topk_idxs[k];
            if (idx >= 0 && idx < num_experts) {
                topk_vals[k] = static_cast<float>(token_scores[idx]);
            }
        }
    }

    // Optionally normalize weights (sum of selected scores → 1).
    // Must sum over top_k (runtime), not K (template), since K may be larger
    // when top_k doesn't match a specialized template (e.g. top_k=3, K=8).
    if (softmax_weights) {
        float maxv = topk_vals[0];
        for (int k = 1; k < top_k; ++k) {
            maxv = fmaxf(maxv, topk_vals[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            topk_vals[k] = expf(topk_vals[k] - maxv);
            sum += topk_vals[k];
        }
        sum = fmaxf(sum, 1e-9f);
        for (int k = 0; k < top_k; ++k) {
            topk_vals[k] /= sum;
        }
    } else if (normalize_weights) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            sum += topk_vals[k];
        }
        sum = fmaxf(sum, 1e-9f);
        for (int k = 0; k < top_k; k++) {
            topk_vals[k] /= sum;
        }
    }

    if (sort_by_index && top_k > 1) {
        for (int i = 0; i < top_k - 1; ++i) {
            int min_idx = i;
            int min_val = topk_idxs[i];
            if (min_val < 0) {
                min_val = INT_MAX;
            }
            for (int j = i + 1; j < top_k; ++j) {
                int idx = topk_idxs[j];
                int idx_val = idx < 0 ? INT_MAX : idx;
                if (idx_val < min_val) {
                    min_val = idx_val;
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                int tmp_idx = topk_idxs[i];
                float tmp_val = topk_vals[i];
                topk_idxs[i] = topk_idxs[min_idx];
                topk_vals[i] = topk_vals[min_idx];
                topk_idxs[min_idx] = tmp_idx;
                topk_vals[min_idx] = tmp_val;
            }
        }
    }

    // Parallel write: lanes < top_k each write one result
    if (lane < top_k) {
        int idx = topk_idxs[lane];
        float val = topk_vals[lane];
        // Sanitize invalid selections (can happen if logits are non-finite).
        if (idx < 0 || idx >= num_experts || !isfinite(val)) {
            idx = 0;
            val = 0.0f;
        }
        expert_indices[token_idx * top_k + lane] = idx;
        routing_weights[token_idx * top_k + lane] = static_cast<T>(val);
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
    bool normalize_weights,
    bool softmax_weights
) {
    constexpr int MAX_K = 8;
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;
    if (top_k <= 0 || top_k > MAX_K) return;

    float* d_row = d_probs + token_idx * num_experts;
    const float* p_row = probs + token_idx * num_experts;
    const float* d_w_row = d_routing_w + token_idx * top_k;
    const int* idx_row = expert_indices + token_idx * top_k;

    if (softmax_weights) {
        float maxv = -INFINITY;
        float z_vals[MAX_K];
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            if (k >= top_k) break;
            int e = idx_row[k];
            float z = (e >= 0 && e < num_experts) ? p_row[e] : -INFINITY;
            z_vals[k] = z;
            maxv = fmaxf(maxv, z);
        }

        float sum = 0.0f;
        float w_vals[MAX_K];
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            if (k >= top_k) break;
            float w = expf(z_vals[k] - maxv);
            w_vals[k] = w;
            sum += w;
        }
        sum = fmaxf(sum, 1e-20f);
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            if (k >= top_k) break;
            w_vals[k] /= sum;
            dot += d_w_row[k] * w_vals[k];
        }

        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            if (k >= top_k) break;
            int e = idx_row[k];
            if (e >= 0 && e < num_experts) {
                float d_z = w_vals[k] * (d_w_row[k] - dot);
                d_row[e] = d_z;
            }
        }
        return;
    }

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

    float sum_p = 0.0f;
    float dot = 0.0f;

    #pragma unroll
    for (int k = 0; k < MAX_K; ++k) {
        if (k >= top_k) break;
        int e = idx_row[k];
        float p = (e >= 0 && e < num_experts) ? p_row[e] : 0.0f;
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
// GPT-OSS MoE Activation (interleaved gate/up)
// ============================================================================
template<typename T>
__global__ void gpt_oss_moe_act_forward_kernel(
    T* __restrict__ out,           // (N, D)
    const T* __restrict__ inp,     // (N, 2*D) interleaved [gate, up]
    int total_elements,            // N * D
    int D,
    float alpha,
    float limit
) {
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

template<typename T>
__global__ void gpt_oss_moe_act_backward_kernel(
    T* __restrict__ d_inp,          // (N, 2*D) interleaved [gate, up]
    const T* __restrict__ d_out,    // (N, D)
    const T* __restrict__ inp,      // (N, 2*D) interleaved [gate, up]
    int total_elements,             // N * D
    int D,
    float alpha,
    float limit
) {
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

void gpt_oss_moe_act_forward(
    nv_bfloat16* out,
    const nv_bfloat16* inp,
    int N,
    int D,
    float alpha,
    float limit,
    cudaStream_t stream
) {
    const int total = N * D;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gpt_oss_moe_act_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        out, inp, total, D, alpha, limit
    );
}

void gpt_oss_moe_act_forward(
    float* out,
    const float* inp,
    int N,
    int D,
    float alpha,
    float limit,
    cudaStream_t stream
) {
    const int total = N * D;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gpt_oss_moe_act_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        out, inp, total, D, alpha, limit
    );
}

void gpt_oss_moe_act_backward(
    nv_bfloat16* d_inp,
    const nv_bfloat16* d_out,
    const nv_bfloat16* inp,
    int N,
    int D,
    float alpha,
    float limit,
    cudaStream_t stream
) {
    const int total = N * D;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gpt_oss_moe_act_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_inp, d_out, inp, total, D, alpha, limit
    );
}

void gpt_oss_moe_act_backward(
    float* d_inp,
    const float* d_out,
    const float* inp,
    int N,
    int D,
    float alpha,
    float limit,
    cudaStream_t stream
) {
    const int total = N * D;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gpt_oss_moe_act_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_inp, d_out, inp, total, D, alpha, limit
    );
}

// ============================================================================
// Utility: sanitize non-finite values (NaN/Inf) in-place
// ============================================================================
template<typename T>
__global__ void sanitize_non_finite_kernel(T* data, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = static_cast<float>(data[idx]);
    if (!isfinite(v)) {
        data[idx] = static_cast<T>(0.0f);
    }
}

template<typename T>
static void sanitize_non_finite_impl(T* data, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    sanitize_non_finite_kernel<<<grid, block, 0, stream>>>(data, n);
    CUDA_CHECK(cudaGetLastError());
}

void sanitize_non_finite(nv_bfloat16* data, int n, cudaStream_t stream) {
    sanitize_non_finite_impl(data, n, stream);
}

void sanitize_non_finite(float* data, int n, cudaStream_t stream) {
    sanitize_non_finite_impl(data, n, stream);
}

// ============================================================================
// Utility: clamp absolute values in-place
// ============================================================================
template<typename T>
__global__ void clamp_abs_kernel(T* data, int n, float max_abs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = static_cast<float>(data[idx]);
    if (v > max_abs) v = max_abs;
    else if (v < -max_abs) v = -max_abs;
    data[idx] = static_cast<T>(v);
}

template<typename T>
static void clamp_abs_impl(T* data, int n, float max_abs, cudaStream_t stream) {
    if (n <= 0 || max_abs <= 0.0f) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    clamp_abs_kernel<<<grid, block, 0, stream>>>(data, n, max_abs);
    CUDA_CHECK(cudaGetLastError());
}

void clamp_abs(nv_bfloat16* data, int n, float max_abs, cudaStream_t stream) {
    clamp_abs_impl(data, n, max_abs, stream);
}

void clamp_abs(float* data, int n, float max_abs, cudaStream_t stream) {
    clamp_abs_impl(data, n, max_abs, stream);
}

// ============================================================================
// Utility: count non-finite values (NaN/Inf)
// ============================================================================
template<typename T>
__global__ void count_non_finite_kernel(int* out_count, const T* data, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = static_cast<float>(data[idx]);
    if (!isfinite(v)) {
        atomicAdd(out_count, 1);
    }
}

template<typename T>
static void count_non_finite_impl(int* out_count, const T* data, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    count_non_finite_kernel<<<grid, block, 0, stream>>>(out_count, data, n);
    CUDA_CHECK(cudaGetLastError());
}

void count_non_finite(int* out_count, const nv_bfloat16* data, int n, cudaStream_t stream) {
    count_non_finite_impl(out_count, data, n, stream);
}

void count_non_finite(int* out_count, const float* data, int n, cudaStream_t stream) {
    count_non_finite_impl(out_count, data, n, stream);
}

// ============================================================================
// Utility: count invalid indices
// ============================================================================
__global__ void count_invalid_indices_kernel(int* out_count, const int* indices, int n, int num_experts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int v = indices[idx];
    if (v < 0 || v >= num_experts) {
        atomicAdd(out_count, 1);
    }
}

void count_invalid_indices(int* out_count, const int* indices, int n, int num_experts, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    count_invalid_indices_kernel<<<grid, block, 0, stream>>>(out_count, indices, n, num_experts);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// MoE per-expert bias add
// ============================================================================
template<typename T>
__global__ void moe_expert_bias_add_forward_kernel(
    T* __restrict__ out,
    const T* __restrict__ inp,
    const T* __restrict__ bias,
    const int* __restrict__ expert_offsets,
    int num_experts,
    int hidden_size
) {
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

template<typename T>
__global__ void moe_expert_bias_add_backward_kernel(
    T* __restrict__ d_inp,
    float* __restrict__ d_bias,
    const T* __restrict__ d_out,
    const int* __restrict__ expert_offsets,
    int num_experts,
    int hidden_size
) {
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

void moe_expert_bias_add_forward(
    float* out,
    const float* inp,
    const float* bias,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int total_tokens,
    cudaStream_t stream
) {
    (void)total_tokens;
    if (num_experts <= 0 || hidden_size <= 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((hidden_size + block.x - 1) / block.x, num_experts, 1);
    moe_expert_bias_add_forward_kernel<<<grid, block, 0, stream>>>(
        out, inp, bias, expert_offsets, num_experts, hidden_size
    );
}

void moe_expert_bias_add_forward(
    nv_bfloat16* out,
    const nv_bfloat16* inp,
    const nv_bfloat16* bias,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int total_tokens,
    cudaStream_t stream
) {
    (void)total_tokens;
    if (num_experts <= 0 || hidden_size <= 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((hidden_size + block.x - 1) / block.x, num_experts, 1);
    moe_expert_bias_add_forward_kernel<<<grid, block, 0, stream>>>(
        out, inp, bias, expert_offsets, num_experts, hidden_size
    );
}

void moe_expert_bias_add_backward(
    float* d_inp,
    float* d_bias,
    const float* d_out,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int total_tokens,
    cudaStream_t stream
) {
    (void)total_tokens;
    if (num_experts <= 0 || hidden_size <= 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((hidden_size + block.x - 1) / block.x, num_experts, 1);
    moe_expert_bias_add_backward_kernel<<<grid, block, 0, stream>>>(
        d_inp, d_bias, d_out, expert_offsets, num_experts, hidden_size
    );
}

void moe_expert_bias_add_backward(
    nv_bfloat16* d_inp,
    float* d_bias,
    const nv_bfloat16* d_out,
    const int* expert_offsets,
    int num_experts,
    int hidden_size,
    int total_tokens,
    cudaStream_t stream
) {
    (void)total_tokens;
    if (num_experts <= 0 || hidden_size <= 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((hidden_size + block.x - 1) / block.x, num_experts, 1);
    moe_expert_bias_add_backward_kernel<<<grid, block, 0, stream>>>(
        d_inp, d_bias, d_out, expert_offsets, num_experts, hidden_size
    );
}

// ============================================================================
// Token Permutation / Dispatch Kernels
// ============================================================================
// Reorders tokens from natural order to expert-grouped order for efficient GEMM.
// Also computes histograms of tokens per expert.

// Compute histogram of tokens per expert.
// Uses shared-memory block-local histogram to reduce global atomic contention:
// each block accumulates into a private shared histogram, then a single pass
// flushes the per-block counts to global memory with one atomicAdd per expert.
// This reduces global atomics from O(num_tokens * top_k) to O(num_experts * num_blocks).
__global__ void moe_compute_expert_counts_kernel(
    int* __restrict__ expert_counts,       // (num_experts,) output
    const int* __restrict__ expert_indices, // (num_tokens, top_k)
    int num_tokens,
    int top_k,
    int num_experts
) {
    extern __shared__ int shared_hist[];

    // Zero shared histogram cooperatively
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        shared_hist[e] = 0;
    }
    __syncthreads();

    // Accumulate into shared histogram (contention limited to threads in this block)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_assignments = num_tokens * top_k;
    if (idx < total_assignments) {
        int expert_id = expert_indices[idx];
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&shared_hist[expert_id], 1);
        }
    }
    __syncthreads();

    // Flush shared histogram to global — one atomicAdd per expert per block
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        if (shared_hist[e] > 0) {
            atomicAdd(&expert_counts[e], shared_hist[e]);
        }
    }
}

// Compute gather indices that reorder tokens to expert-grouped order.
// This is the key data structure for fused permute operations.
//
// Uses warp-level ballot aggregation to reduce atomic contention:
// threads in the same warp targeting the same expert batch their atomicAdd
// into a single warp-aggregated add, then each thread computes its slot
// from the warp-local prefix count.
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

    const int lane = threadIdx.x & 31;

    // Warp-aggregated atomic: find all lanes in this warp targeting the same expert
    // and batch them into a single atomicAdd.
    // We iterate over the set of unique expert_ids in this warp using a peer mask.
    int remaining_mask = __ballot_sync(0xFFFFFFFFu, true); // mask of all active lanes
    int slot = -1;

    while (remaining_mask) {
        // Pick the expert_id from the lowest active lane as the "leader" for this round
        int leader = __ffs(remaining_mask) - 1;
        int leader_expert = __shfl_sync(0xFFFFFFFFu, expert_id, leader);

        // Find all lanes in this warp with the same expert_id
        unsigned int peer_mask = __ballot_sync(0xFFFFFFFFu, expert_id == leader_expert);
        int peer_count = __popc(peer_mask);

        // Only process if this lane matches the current leader's expert
        if (expert_id == leader_expert) {
            // Warp-local prefix count: how many matching lanes come before me?
            unsigned int lanes_before_me = peer_mask & ((1u << lane) - 1u);
            int my_offset = __popc(lanes_before_me);

            // Leader lane does one atomic for the entire group
            int base_slot = 0;
            if (my_offset == 0) {
                base_slot = atomicAdd(&expert_positions[leader_expert], peer_count);
            }
            // Broadcast base_slot from the leader of this peer group
            int first_peer = __ffs(peer_mask) - 1;
            base_slot = __shfl_sync(peer_mask, base_slot, first_peer);

            slot = base_slot + my_offset;
        }

        // Remove processed lanes from the remaining mask
        remaining_mask &= ~peer_mask;
    }

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
    using x128 = GenericVector<T, 16/sizeof(T)>;

    int out_idx = blockIdx.x;
    if (out_idx >= total_tokens) return;

    // Which token (original) to read from
    int token_assignment_idx = gather_indices[out_idx];
    int token_idx = token_assignment_idx / top_k;  // Original token index

    // Copy hidden state with 128-bit vectorized loads/stores
    const T* src = inp + token_idx * hidden_size;
    T* dst = out + out_idx * hidden_size;

    int d = threadIdx.x * x128::size;
    for (; d + x128::size <= hidden_size; d += blockDim.x * x128::size) {
        x128::load(src + d).store(dst + d);
    }
    // Scalar remainder
    for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
        dst[r] = src[r];
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
    using x128 = GenericVector<T, 16/sizeof(T)>;

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    T* dst = out + token_idx * hidden_size;
    const T* weights_ptr = routing_weights + token_idx * top_k;

    // Pre-load routing weights and expert positions into registers
    constexpr int MAX_K = 8;
    float w[MAX_K];
    int expert_pos[MAX_K];
    for (int k = 0; k < top_k && k < MAX_K; k++) {
        w[k] = static_cast<float>(weights_ptr[k]);
        int assignment_idx = token_idx * top_k + k;
        expert_pos[k] = scatter_indices[assignment_idx];
    }

    // Vectorized accumulation with 128-bit loads/stores
    int d = threadIdx.x * x128::size;
    for (; d + x128::size <= hidden_size; d += blockDim.x * x128::size) {
        x128 acc_vec;
        for (int i = 0; i < x128::size; i++) {
            acc_vec[i] = static_cast<T>(0);
        }

        for (int k = 0; k < top_k && k < MAX_K; k++) {
            if (expert_pos[k] < 0 || expert_pos[k] >= total_tokens) continue;
            x128 val = x128::load(expert_out + expert_pos[k] * hidden_size + d);
            for (int i = 0; i < x128::size; i++) {
                acc_vec[i] = static_cast<T>(static_cast<float>(acc_vec[i]) + w[k] * static_cast<float>(val[i]));
            }
        }

        acc_vec.store(dst + d);
    }

    // Scalar remainder
    for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < top_k && k < MAX_K; k++) {
            if (expert_pos[k] < 0 || expert_pos[k] >= total_tokens) continue;
            acc += w[k] * static_cast<float>(expert_out[expert_pos[k] * hidden_size + r]);
        }
        dst[r] = static_cast<T>(acc);
    }
}

// Unpermute and weight-combine expert outputs back to token order (FP32 routing weights)
template<typename T>
__global__ void moe_unpermute_and_combine_kernel_mixed(
    T* __restrict__ out,                    // (num_tokens, hidden_size)
    const T* __restrict__ expert_out,       // (total_tokens, hidden_size)
    const float* __restrict__ routing_weights,  // (num_tokens, top_k) in FP32
    const int* __restrict__ scatter_indices, // (total_tokens,)
    int num_tokens,
    int total_tokens,
    int hidden_size,
    int top_k
) {
    using x128 = GenericVector<T, 16/sizeof(T)>;

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    T* dst = out + token_idx * hidden_size;
    const float* weights_ptr = routing_weights + token_idx * top_k;

    // Pre-load routing weights and expert positions into registers
    constexpr int MAX_K = 8;
    float w[MAX_K];
    int expert_pos[MAX_K];
    for (int k = 0; k < top_k && k < MAX_K; k++) {
        w[k] = weights_ptr[k];
        int assignment_idx = token_idx * top_k + k;
        expert_pos[k] = scatter_indices[assignment_idx];
    }

    // Vectorized accumulation with 128-bit loads/stores
    int d = threadIdx.x * x128::size;
    for (; d + x128::size <= hidden_size; d += blockDim.x * x128::size) {
        x128 acc_vec;
        for (int i = 0; i < x128::size; i++) {
            acc_vec[i] = static_cast<T>(0);
        }

        for (int k = 0; k < top_k && k < MAX_K; k++) {
            if (expert_pos[k] < 0 || expert_pos[k] >= total_tokens) continue;
            x128 val = x128::load(expert_out + expert_pos[k] * hidden_size + d);
            for (int i = 0; i < x128::size; i++) {
                acc_vec[i] = static_cast<T>(static_cast<float>(acc_vec[i]) + w[k] * static_cast<float>(val[i]));
            }
        }

        acc_vec.store(dst + d);
    }

    // Scalar remainder
    for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < top_k && k < MAX_K; k++) {
            if (expert_pos[k] < 0 || expert_pos[k] >= total_tokens) continue;
            acc += w[k] * static_cast<float>(expert_out[expert_pos[k] * hidden_size + r]);
        }
        dst[r] = static_cast<T>(acc);
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

void moe_sigmoid_backward(
    nv_bfloat16* d_inp,
    const nv_bfloat16* grad,
    const nv_bfloat16* out,
    int num_elements,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_sigmoid_backward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        d_inp, grad, out, num_elements
    );
}

void moe_sigmoid_backward(
    float* d_inp,
    const float* grad,
    const float* out,
    int num_elements,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_sigmoid_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(
        d_inp, grad, out, num_elements
    );
}

void moe_scale_forward(
    nv_bfloat16* out,
    const nv_bfloat16* inp,
    float scale,
    int num_elements,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_scale_forward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
        out, inp, scale, num_elements
    );
}

void moe_scale_forward(
    float* out,
    const float* inp,
    float scale,
    int num_elements,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    moe_scale_forward_kernel<float><<<grid_size, block_size, 0, stream>>>(
        out, inp, scale, num_elements
    );
}

// Dispatch helper: select template K and launch warp-per-token kernel
template<typename T>
static void moe_topk_forward_dispatch(
    int* expert_indices,
    T* routing_weights,
    const T* scores,
    const float* correction_bias,
    int num_tokens,
    int num_experts,
    int top_k,
    bool normalize_weights,
    bool softmax_weights,
    bool sort_by_index,
    float rounding_scale,
    cudaStream_t stream
) {
    // 8 warps per block = 256 threads, each warp handles one token
    constexpr int warps_per_block = 8;
    constexpr int block_size = warps_per_block * MOE_WARP_SIZE;
    int grid_size = (num_tokens + warps_per_block - 1) / warps_per_block;

    // Template-specialize for common K values to enable full unrolling
    switch (top_k) {
    case 1:
        moe_topk_forward_kernel<T, 1><<<grid_size, block_size, 0, stream>>>(
            expert_indices, routing_weights, scores, correction_bias, num_tokens, num_experts, top_k,
            normalize_weights, softmax_weights, sort_by_index, rounding_scale);
        break;
    case 2:
        moe_topk_forward_kernel<T, 2><<<grid_size, block_size, 0, stream>>>(
            expert_indices, routing_weights, scores, correction_bias, num_tokens, num_experts, top_k,
            normalize_weights, softmax_weights, sort_by_index, rounding_scale);
        break;
    case 4:
        moe_topk_forward_kernel<T, 4><<<grid_size, block_size, 0, stream>>>(
            expert_indices, routing_weights, scores, correction_bias, num_tokens, num_experts, top_k,
            normalize_weights, softmax_weights, sort_by_index, rounding_scale);
        break;
    default:
        // K=8 covers top_k 3,5,6,7,8 — the merge always produces K entries,
        // but we only write top_k of them via the lane < top_k guard.
        moe_topk_forward_kernel<T, 8><<<grid_size, block_size, 0, stream>>>(
            expert_indices, routing_weights, scores, correction_bias, num_tokens, num_experts, top_k,
            normalize_weights, softmax_weights, sort_by_index, rounding_scale);
        break;
    }
}

void moe_topk_forward(
    int* expert_indices,
    nv_bfloat16* routing_weights,
    const nv_bfloat16* scores,
    const float* correction_bias,
    int num_tokens,
    int num_experts,
    int top_k,
    bool normalize_weights,
    bool softmax_weights,
    bool sort_by_index,
    float rounding_scale,
    cudaStream_t stream
) {
    moe_topk_forward_dispatch(expert_indices, routing_weights, scores, correction_bias,
                              num_tokens, num_experts, top_k, normalize_weights,
                              softmax_weights, sort_by_index, rounding_scale, stream);
}

void moe_topk_forward(
    int* expert_indices,
    float* routing_weights,
    const float* scores,
    const float* correction_bias,
    int num_tokens,
    int num_experts,
    int top_k,
    bool normalize_weights,
    bool softmax_weights,
    bool sort_by_index,
    float rounding_scale,
    cudaStream_t stream
) {
    moe_topk_forward_dispatch(expert_indices, routing_weights, scores, correction_bias,
                              num_tokens, num_experts, top_k, normalize_weights,
                              softmax_weights, sort_by_index, rounding_scale, stream);
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
    bool softmax_weights,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;
    moe_topk_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_probs, d_routing_weights, probs, expert_indices,
        num_tokens, num_experts, top_k, normalize_weights, softmax_weights
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
    size_t smem = num_experts * sizeof(int);
    moe_compute_expert_counts_kernel<<<grid_size, block_size, smem, stream>>>(
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
    nv_bfloat16* out,
    const nv_bfloat16* expert_out,
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
    moe_unpermute_and_combine_kernel_mixed<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
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

    // Optional debug: force per-expert GEMM loop to bypass grouped GEMM.
    static int force_loop = -1;
    if (force_loop < 0) {
        force_loop = (std::getenv("SUROGATE_MOE_GEMM_LOOP") != nullptr) ? 1 : 0;
    }
    if (force_loop) {
        static int force_default_algo = -1;
        if (force_default_algo < 0) {
            force_default_algo = (std::getenv("SUROGATE_MOE_GEMM_DEFAULT") != nullptr) ? 1 : 0;
        }
        const cublasGemmAlgo_t algo = force_default_algo ? CUBLAS_GEMM_DEFAULT
                                                         : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        for (int e = 0; e < n_active; ++e) {
            int global_idx = active_expert_indices ? active_expert_indices[e] : e;
            int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
            if (tokens_e == 0) continue;
            const int weight_idx = weight_is_compact ? e : global_idx;
            const T* A_ptr = weights + weight_idx * out_dim * hidden_size;
            const T* B_ptr = input + h_offsets[global_idx] * hidden_size;
            T* C_ptr = output + h_offsets[global_idx] * out_dim;

            CUBLAS_CHECK(cublasGemmEx(
                cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_dim, tokens_e, hidden_size,
                &alpha,
                A_ptr, cublas_dtype<T>(), hidden_size,
                B_ptr, cublas_dtype<T>(), hidden_size,
                &beta,
                C_ptr, cublas_dtype<T>(), out_dim,
                CUBLAS_COMPUTE_32F, algo));
        }
        return;
    }

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

    // Optional debug: force per-expert GEMM loop to bypass grouped GEMM.
    static int force_loop = -1;
    if (force_loop < 0) {
        force_loop = (std::getenv("SUROGATE_MOE_GEMM_LOOP") != nullptr) ? 1 : 0;
    }
    if (force_loop) {
        static int force_default_algo = -1;
        if (force_default_algo < 0) {
            force_default_algo = (std::getenv("SUROGATE_MOE_GEMM_DEFAULT") != nullptr) ? 1 : 0;
        }
        const cublasGemmAlgo_t algo = force_default_algo ? CUBLAS_GEMM_DEFAULT
                                                         : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        for (int e = 0; e < n_active; ++e) {
            int global_idx = active_expert_indices ? active_expert_indices[e] : e;
            int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
            if (tokens_e == 0) continue;
            const int weight_idx = weight_is_compact ? e : global_idx;
            const T* A_ptr = weights + weight_idx * hidden_size * intermediate_size;
            const T* B_ptr = input + h_offsets[global_idx] * intermediate_size;
            T* C_ptr = output + h_offsets[global_idx] * hidden_size;

            CUBLAS_CHECK(cublasGemmEx(
                cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                hidden_size, tokens_e, intermediate_size,
                &alpha,
                A_ptr, cublas_dtype<T>(), intermediate_size,
                B_ptr, cublas_dtype<T>(), intermediate_size,
                &beta,
                C_ptr, cublas_dtype<T>(), hidden_size,
                CUBLAS_COMPUTE_32F, algo));
        }
        return;
    }

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

template<typename T>
void moe_grouped_gemm_up_backward_impl(
    T* d_input,                       // (total_tokens, C) - gradient w.r.t. input
    const T* d_up,                    // (total_tokens, D) - gradient from activation backward
    const T* weights,                 // (num_experts, D, C) - up projection weights
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
    const int up_dim = intermediate_size;

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
        k_vec.push_back(up_dim);

        lda_vec.push_back(hidden_size);
        ldb_vec.push_back(up_dim);
        ldc_vec.push_back(hidden_size);

        A_vec.push_back(weights + (weight_is_compact ? e : global_idx) * up_dim * hidden_size);
        B_vec.push_back(d_up + h_offsets[global_idx] * up_dim);
        C_vec.push_back(d_input + h_offsets[global_idx] * hidden_size);
    }

    if (m_vec.empty()) return;

    const int gemm_count = static_cast<int>(m_vec.size());

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

void moe_grouped_gemm_up_backward(
    nv_bfloat16* d_input,
    const nv_bfloat16* d_up,
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
    moe_grouped_gemm_up_backward_impl(d_input, d_up, weights, expert_offsets,
                                      num_experts, hidden_size, intermediate_size,
                                      cublas_handle, stream, host_offsets, active_expert_indices, weight_is_compact, num_active_experts);
}

void moe_grouped_gemm_up_backward(
    float* d_input,
    const float* d_up,
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
    moe_grouped_gemm_up_backward_impl(d_input, d_up, weights, expert_offsets,
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
    using x128 = GenericVector<T, 16/sizeof(T)>;

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const T* d_out = d_output + token_idx * hidden_size;
    const T* weights = routing_weights + token_idx * top_k;

    // For each expert assigned to this token
    for (int k = 0; k < top_k; k++) {
        int assignment_idx = token_idx * top_k + k;
        int expert_pos = scatter_indices[assignment_idx];
        if (expert_pos < 0 || expert_pos >= total_tokens) {
            if (d_routing_weights != nullptr && threadIdx.x == 0) {
                d_routing_weights[assignment_idx] = static_cast<T>(0);
            }
            continue;
        }

        T* d_exp_out = d_expert_out + expert_pos * hidden_size;
        float weight = static_cast<float>(weights[k]);

        // d_expert_out[expert_pos] = weight * d_output[token]
        // Vectorized 128-bit loads/stores
        int d = threadIdx.x * x128::size;
        for (; d + x128::size <= hidden_size; d += blockDim.x * x128::size) {
            x128 grad_in = x128::load(d_out + d);
            x128 grad_out;
            for (int i = 0; i < x128::size; i++) {
                grad_out[i] = static_cast<T>(weight * static_cast<float>(grad_in[i]));
            }
            grad_out.store(d_exp_out + d);
        }
        // Scalar remainder
        for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
            d_exp_out[r] = static_cast<T>(weight * static_cast<float>(d_out[r]));
        }

        // Compute gradient w.r.t. routing weights (if needed)
        // d_routing_weights[token, k] = dot(expert_out[expert_pos], d_output[token])
        if (d_routing_weights != nullptr) {
            const T* exp_out = expert_out + expert_pos * hidden_size;
            float thread_dot = 0.0f;
            int dv = threadIdx.x * x128::size;
            for (; dv + x128::size <= hidden_size; dv += blockDim.x * x128::size) {
                x128 ev = x128::load(exp_out + dv);
                x128 dv_out = x128::load(d_out + dv);
                for (int i = 0; i < x128::size; i++) {
                    thread_dot += static_cast<float>(ev[i]) * static_cast<float>(dv_out[i]);
                }
            }
            for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
                thread_dot += static_cast<float>(exp_out[r]) * static_cast<float>(d_out[r]);
            }
            // Warp-level reduction
            thread_dot = warpReduceSum(thread_dot);
            // Block-level reduction
            __shared__ float smem_dot[32];
            int warp_id = threadIdx.x / 32;
            int lane_id = threadIdx.x % 32;
            if (lane_id == 0) smem_dot[warp_id] = thread_dot;
            __syncthreads();
            if (warp_id == 0) {
                float val = (lane_id < (blockDim.x / 32)) ? smem_dot[lane_id] : 0.0f;
                val = warpReduceSum(val);
                if (lane_id == 0) {
                    d_routing_weights[assignment_idx] = static_cast<T>(val);
                }
            }
            __syncthreads();
        }
    }
}

// Backward through unpermute+combine with FP32 routing weights and BF16 expert outputs.
// d_expert_outputs[permuted_idx] = routing_weights[token, k] * d_output[token]
template<typename T>
__global__ void moe_combine_backward_kernel_mixed(
    T* __restrict__ d_expert_out,          // (total_tokens, hidden_size)
    float* __restrict__ d_routing_weights, // (num_tokens, top_k) - optional, can be NULL
    const T* __restrict__ d_output,        // (num_tokens, hidden_size)
    const T* __restrict__ expert_out,      // (total_tokens, hidden_size) - for weight gradient
    const float* __restrict__ routing_weights, // (num_tokens, top_k) in FP32
    const int* __restrict__ scatter_indices,  // (total_tokens,)
    int num_tokens,
    int total_tokens,
    int hidden_size,
    int top_k
) {
    using x128 = GenericVector<T, 16/sizeof(T)>;

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const T* d_out = d_output + token_idx * hidden_size;
    const float* weights = routing_weights + token_idx * top_k;

    // For each expert assigned to this token
    for (int k = 0; k < top_k; k++) {
        int assignment_idx = token_idx * top_k + k;
        int expert_pos = scatter_indices[assignment_idx];
        if (expert_pos < 0 || expert_pos >= total_tokens) {
            if (d_routing_weights != nullptr && threadIdx.x == 0) {
                d_routing_weights[assignment_idx] = 0.0f;
            }
            continue;
        }

        T* d_exp_out = d_expert_out + expert_pos * hidden_size;
        float weight = weights[k];

        // d_expert_out[expert_pos] = weight * d_output[token]
        // Vectorized 128-bit loads/stores
        int d = threadIdx.x * x128::size;
        for (; d + x128::size <= hidden_size; d += blockDim.x * x128::size) {
            x128 grad_in = x128::load(d_out + d);
            x128 grad_out;
            for (int i = 0; i < x128::size; i++) {
                grad_out[i] = static_cast<T>(weight * static_cast<float>(grad_in[i]));
            }
            grad_out.store(d_exp_out + d);
        }
        // Scalar remainder
        for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
            d_exp_out[r] = static_cast<T>(weight * static_cast<float>(d_out[r]));
        }

        // Compute gradient w.r.t. routing weights (if needed)
        // d_routing_weights[token, k] = dot(expert_out[expert_pos], d_output[token])
        if (d_routing_weights != nullptr) {
            const T* exp_out = expert_out + expert_pos * hidden_size;
            float thread_dot = 0.0f;
            int dv = threadIdx.x * x128::size;
            for (; dv + x128::size <= hidden_size; dv += blockDim.x * x128::size) {
                x128 ev = x128::load(exp_out + dv);
                x128 dv_out = x128::load(d_out + dv);
                for (int i = 0; i < x128::size; i++) {
                    thread_dot += static_cast<float>(ev[i]) * static_cast<float>(dv_out[i]);
                }
            }
            for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
                thread_dot += static_cast<float>(exp_out[r]) * static_cast<float>(d_out[r]);
            }
            // Warp-level reduction
            thread_dot = warpReduceSum(thread_dot);
            // Block-level reduction
            __shared__ float smem_dot[32];
            int warp_id = threadIdx.x / 32;
            int lane_id = threadIdx.x % 32;
            if (lane_id == 0) smem_dot[warp_id] = thread_dot;
            __syncthreads();
            if (warp_id == 0) {
                float val = (lane_id < (blockDim.x / 32)) ? smem_dot[lane_id] : 0.0f;
                val = warpReduceSum(val);
                if (lane_id == 0) {
                    d_routing_weights[assignment_idx] = val;
                }
            }
            __syncthreads();
        }
    }
}

// Backward through permute: gather gradient back to original token order
// d_input[token] += d_permuted[permuted_idx] for each assignment
// FP32 version - uses native atomicAdd
// Note: atomicAdd is per-element (no vectorized atomic), but we vectorize the load
__global__ void moe_permute_backward_kernel_fp32(
    float* __restrict__ d_input,              // (num_tokens, hidden_size)
    const float* __restrict__ d_permuted,     // (total_tokens, hidden_size)
    const int* __restrict__ gather_indices,   // (total_tokens,)
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k
) {
    using x128 = GenericVector<float, 16/sizeof(float)>;

    int out_idx = blockIdx.x;
    if (out_idx >= total_tokens) return;

    // Which token this permuted position corresponds to
    int token_assignment_idx = gather_indices[out_idx];
    int token_idx = token_assignment_idx / top_k;

    const float* d_perm = d_permuted + out_idx * hidden_size;
    float* d_in = d_input + token_idx * hidden_size;

    // Vectorized load, scalar atomicAdd (no vectorized atomic exists)
    int d = threadIdx.x * x128::size;
    for (; d + x128::size <= hidden_size; d += blockDim.x * x128::size) {
        x128 val = x128::load(d_perm + d);
        for (int i = 0; i < x128::size; i++) {
            atomicAdd(d_in + d + i, (float)val[i]);
        }
    }
    for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
        atomicAdd(d_in + r, d_perm[r]);
    }
}

// BF16 version - uses atomicAdd for __nv_bfloat16 (requires SM80+)
// Vectorized load, scalar atomicAdd
__global__ void moe_permute_backward_kernel_bf16(
    nv_bfloat16* __restrict__ d_input,              // (num_tokens, hidden_size)
    const nv_bfloat16* __restrict__ d_permuted,     // (total_tokens, hidden_size)
    const int* __restrict__ gather_indices,         // (total_tokens,)
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k
) {
    using x128 = GenericVector<nv_bfloat16, 16/sizeof(nv_bfloat16)>;

    int out_idx = blockIdx.x;
    if (out_idx >= total_tokens) return;

    // Which token this permuted position corresponds to
    int token_assignment_idx = gather_indices[out_idx];
    int token_idx = token_assignment_idx / top_k;

    const nv_bfloat16* d_perm = d_permuted + out_idx * hidden_size;
    nv_bfloat16* d_in = d_input + token_idx * hidden_size;

    // Vectorized load, scalar atomicAdd (SM80+ for BF16 atomicAdd)
    int d = threadIdx.x * x128::size;
    for (; d + x128::size <= hidden_size; d += blockDim.x * x128::size) {
        x128 val = x128::load(d_perm + d);
        for (int i = 0; i < x128::size; i++) {
            atomicAdd(d_in + d + i, val[i]);
        }
    }
    for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
        atomicAdd(d_in + r, d_perm[r]);
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
    nv_bfloat16* d_expert_out,
    float* d_routing_weights,
    const nv_bfloat16* d_output,
    const nv_bfloat16* expert_out,
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
    moe_combine_backward_kernel_mixed<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
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

// ============================================================================
// FP8 MoE Grouped GEMM Implementations
// ============================================================================
//
// These implementations dispatch FP8 E4M3 × E4M3 (forward) or E4M3 × E5M2
// (backward) grouped GEMMs for MoE layers using cuBLASLt.
//
// The strategy is to loop over each expert and dispatch individual FP8 matmuls
// to cuBLASLt, similar to how the BF16 versions work. This is efficient because:
// 1. Each expert's matmul is independent
// 2. cuBLASLt handles FP8 scaling and accumulation efficiently
// 3. Host overhead is amortized by large per-expert matmuls

/// @brief FP8 MoE grouped GEMM: E4M3 input × E4M3 weights → BF16 output
void moe_grouped_gemm(nv_bfloat16* output,
                      const __nv_fp8_e4m3* input,
                      const __nv_fp8_e4m3* weights,
                      const float* scale_input,
                      const float* scale_weights,
                      const int* expert_offsets, int num_experts,
                      int M, int K,
                      cublasLtHandle_t cublas_handle, cudaStream_t stream,
                      const int* host_offsets,
                      float alpha, float beta,
                      EMMTranspose mode,
                      const int* active_expert_indices,
                      bool weight_is_compact,
                      int num_active_experts) {
    // Loop over each expert and dispatch FP8 matmul
    const int num_active = (num_active_experts > 0) ? num_active_experts : num_experts;

    for (int idx = 0; idx < num_active; ++idx) {
        const int expert_id = active_expert_indices ? active_expert_indices[idx] : idx;
        const int weight_idx = weight_is_compact ? idx : expert_id;

        // Get token range for this expert
        const int start = host_offsets[expert_id];
        const int end = host_offsets[expert_id + 1];
        const int num_tokens = end - start;

        if (num_tokens <= 0) continue;

        // Compute pointers for this expert's slice
        const __nv_fp8_e4m3* input_slice = input + static_cast<long>(start) * K;
        const __nv_fp8_e4m3* weight_slice = weights + static_cast<long>(weight_idx) * M * K;
        nv_bfloat16* output_slice = output + static_cast<long>(start) * M;

        // Dispatch FP8 matmul: output = input @ weight.T
        // input: (num_tokens, K) E4M3
        // weight: (M, K) E4M3
        // output: (num_tokens, M) BF16
        matmul(output_slice, weight_slice, input_slice,
               static_cast<nv_bfloat16*>(nullptr),  // no bias
               scale_weights, scale_input,
               cublas_handle, nullptr, 0,  // no workspace needed
               M, num_tokens, K,
               mode,  // typically TN
               /*accumulate=*/false,
               stream);
    }
}

/// @brief FP8 MoE grouped GEMM backward: E4M3 weights × E5M2 gradients → BF16 dinp
void moe_grouped_gemm_up_backward(nv_bfloat16* d_input,
                                  const __nv_fp8_e5m2* d_output,
                                  const __nv_fp8_e4m3* weights,
                                  const float* scale_dout,
                                  const float* scale_weights,
                                  const int* expert_offsets, int num_experts,
                                  int hidden_size, int intermediate_size,
                                  cublasLtHandle_t cublas_handle, cudaStream_t stream,
                                  const int* host_offsets,
                                  const int* active_expert_indices,
                                  bool weight_is_compact,
                                  int num_active_experts) {
    // Compute dinp = weights^T @ dout
    // weights: (num_experts, M, K) E4M3
    // dout: (total_tokens, M) E5M2
    // dinp: (total_tokens, K) BF16

    const int M = intermediate_size;
    const int K = hidden_size;
    const int num_active = (num_active_experts > 0) ? num_active_experts : num_experts;

    for (int idx = 0; idx < num_active; ++idx) {
        const int expert_id = active_expert_indices ? active_expert_indices[idx] : idx;
        const int weight_idx = weight_is_compact ? idx : expert_id;

        const int start = host_offsets[expert_id];
        const int end = host_offsets[expert_id + 1];
        const int num_tokens = end - start;

        if (num_tokens <= 0) continue;

        const __nv_fp8_e5m2* dout_slice = d_output + static_cast<long>(start) * M;
        const __nv_fp8_e4m3* weight_slice = weights + static_cast<long>(weight_idx) * M * K;
        nv_bfloat16* dinp_slice = d_input + static_cast<long>(start) * K;

        // dinp = W^T @ dout => (K, M) @ (num_tokens, M)^T = (K, num_tokens)^T = (num_tokens, K)
        // Using NN mode: dinp = weight @ dout where weight is (M, K) -> need (K, M)
        matmul(dinp_slice, weight_slice, dout_slice,
               static_cast<nv_bfloat16*>(nullptr),  // no bias
               scale_weights, scale_dout,
               cublas_handle, nullptr, 0,
               K, num_tokens, M,
               EMMTranspose::NN,  // weight needs to be transposed
               /*accumulate=*/false,
               stream);
    }
}
