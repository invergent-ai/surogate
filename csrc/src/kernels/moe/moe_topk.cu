// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_topk.cu.

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
template <int K>
__device__ __forceinline__ void topk_insert(float* vals, int* idxs, float val, int idx) {
    vals[K - 1] = val;
    idxs[K - 1] = idx;
#pragma unroll
    for (int j = K - 2; j >= 0; j--) {
        if ((vals[j + 1] > vals[j]) || (vals[j + 1] == vals[j] && idxs[j + 1] < idxs[j])) {
            float tv = vals[j];
            vals[j] = vals[j + 1];
            vals[j + 1] = tv;
            int ti = idxs[j];
            idxs[j] = idxs[j + 1];
            idxs[j + 1] = ti;
        }
    }
}

// Warp-level tournament top-K selection.
// Each lane scans its stripe of experts, maintains a local sorted top-K,
// then 5 rounds of shuffle-based merging produce the global top-K on all lanes.
// When correction_bias is non-null, selection is based on (score + bias) but
// the values stored in out_vals are the biased scores (caller must re-read
// original scores after if needed).
template <int K, typename T>
__device__ __forceinline__ void warp_topk(const T* __restrict__ token_scores,
                                          int num_experts,
                                          float* out_vals,
                                          int* out_idxs,
                                          const float* __restrict__ correction_bias = nullptr,
                                          float rounding_scale = 0.0f) {
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
template <typename T, int K>
__global__ void moe_topk_forward_kernel(int* __restrict__ expert_indices,           // (num_tokens, top_k)
                                        T* __restrict__ routing_weights,            // (num_tokens, top_k)
                                        const T* __restrict__ scores,               // (num_tokens, num_experts)
                                        const float* __restrict__ correction_bias,  // (num_experts) or nullptr
                                        int num_tokens,
                                        int num_experts,
                                        int top_k,
                                        bool normalize_weights,
                                        bool softmax_weights,
                                        bool sort_by_index,
                                        float rounding_scale) {
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
        float maxv = -INFINITY;
        const int limit = normalize_weights ? top_k : num_experts;
        for (int k = 0; k < limit; ++k) {
            const float v = normalize_weights ? topk_vals[k] : static_cast<float>(token_scores[k]);
            maxv = fmaxf(maxv, v);
        }
        float sum = 0.0f;
        for (int k = 0; k < limit; ++k) {
            const float v = normalize_weights ? topk_vals[k] : static_cast<float>(token_scores[k]);
            sum += expf(v - maxv);
        }
        sum = fmaxf(sum, 1e-9f);
        for (int k = 0; k < top_k; ++k) {
            topk_vals[k] = expf(topk_vals[k] - maxv) / sum;
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
__global__ void moe_topk_backward_kernel(float* __restrict__ d_probs,             // (num_tokens, num_experts)
                                         const float* __restrict__ d_routing_w,   // (num_tokens, top_k)
                                         const float* __restrict__ probs,         // (num_tokens, num_experts)
                                         const int* __restrict__ expert_indices,  // (num_tokens, top_k)
                                         int num_tokens,
                                         int num_experts,
                                         int top_k,
                                         bool normalize_weights,
                                         bool softmax_weights) {
    constexpr int MAX_K = 8;
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;
    if (top_k <= 0 || top_k > MAX_K) return;

    float* d_row = d_probs + token_idx * num_experts;
    const float* p_row = probs + token_idx * num_experts;
    const float* d_w_row = d_routing_w + token_idx * top_k;
    const int* idx_row = expert_indices + token_idx * top_k;

    if (softmax_weights) {
        if (normalize_weights) {
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
        } else {
            float maxv = -INFINITY;
            for (int e = 0; e < num_experts; ++e) {
                maxv = fmaxf(maxv, p_row[e]);
            }
            float sum = 0.0f;
            for (int e = 0; e < num_experts; ++e) {
                sum += expf(p_row[e] - maxv);
            }
            sum = fmaxf(sum, 1e-20f);

            float dot = 0.0f;
#pragma unroll
            for (int k = 0; k < MAX_K; ++k) {
                if (k >= top_k) break;
                int e = idx_row[k];
                if (e >= 0 && e < num_experts) {
                    dot += d_w_row[k] * (expf(p_row[e] - maxv) / sum);
                }
            }
            for (int e = 0; e < num_experts; ++e) {
                float prob = expf(p_row[e] - maxv) / sum;
                float grad = 0.0f;
#pragma unroll
                for (int k = 0; k < MAX_K; ++k) {
                    if (k >= top_k) break;
                    if (idx_row[k] == e) {
                        grad = d_w_row[k];
                    }
                }
                d_row[e] = prob * (grad - dot);
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

// Dispatch helper: select template K and launch warp-per-token kernel
template <typename T>
static void moe_topk_forward_dispatch(int* expert_indices,
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
                                      cudaStream_t stream) {
    // 8 warps per block = 256 threads, each warp handles one token
    constexpr int warps_per_block = 8;
    constexpr int block_size = warps_per_block * MOE_WARP_SIZE;
    int grid_size = (num_tokens + warps_per_block - 1) / warps_per_block;

    // Template-specialize for common K values to enable full unrolling
    switch (top_k) {
        case 1:
            moe_topk_forward_kernel<T, 1><<<grid_size, block_size, 0, stream>>>(expert_indices,
                                                                                routing_weights,
                                                                                scores,
                                                                                correction_bias,
                                                                                num_tokens,
                                                                                num_experts,
                                                                                top_k,
                                                                                normalize_weights,
                                                                                softmax_weights,
                                                                                sort_by_index,
                                                                                rounding_scale);
            break;
        case 2:
            moe_topk_forward_kernel<T, 2><<<grid_size, block_size, 0, stream>>>(expert_indices,
                                                                                routing_weights,
                                                                                scores,
                                                                                correction_bias,
                                                                                num_tokens,
                                                                                num_experts,
                                                                                top_k,
                                                                                normalize_weights,
                                                                                softmax_weights,
                                                                                sort_by_index,
                                                                                rounding_scale);
            break;
        case 4:
            moe_topk_forward_kernel<T, 4><<<grid_size, block_size, 0, stream>>>(expert_indices,
                                                                                routing_weights,
                                                                                scores,
                                                                                correction_bias,
                                                                                num_tokens,
                                                                                num_experts,
                                                                                top_k,
                                                                                normalize_weights,
                                                                                softmax_weights,
                                                                                sort_by_index,
                                                                                rounding_scale);
            break;
        default:
            // K=8 covers top_k 3,5,6,7,8 — the merge always produces K entries,
            // but we only write top_k of them via the lane < top_k guard.
            moe_topk_forward_kernel<T, 8><<<grid_size, block_size, 0, stream>>>(expert_indices,
                                                                                routing_weights,
                                                                                scores,
                                                                                correction_bias,
                                                                                num_tokens,
                                                                                num_experts,
                                                                                top_k,
                                                                                normalize_weights,
                                                                                softmax_weights,
                                                                                sort_by_index,
                                                                                rounding_scale);
            break;
    }
}

void moe_topk_forward(int* expert_indices,
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
                      cudaStream_t stream) {
    moe_topk_forward_dispatch(expert_indices,
                              routing_weights,
                              scores,
                              correction_bias,
                              num_tokens,
                              num_experts,
                              top_k,
                              normalize_weights,
                              softmax_weights,
                              sort_by_index,
                              rounding_scale,
                              stream);
}

void moe_topk_forward(int* expert_indices,
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
                      cudaStream_t stream) {
    moe_topk_forward_dispatch(expert_indices,
                              routing_weights,
                              scores,
                              correction_bias,
                              num_tokens,
                              num_experts,
                              top_k,
                              normalize_weights,
                              softmax_weights,
                              sort_by_index,
                              rounding_scale,
                              stream);
}

void moe_topk_backward(float* d_probs,
                       const float* d_routing_weights,
                       const float* probs,
                       const int* expert_indices,
                       int num_tokens,
                       int num_experts,
                       int top_k,
                       bool normalize_weights,
                       bool softmax_weights,
                       cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;
    moe_topk_backward_kernel<<<grid_size, block_size, 0, stream>>>(d_probs,
                                                                   d_routing_weights,
                                                                   probs,
                                                                   expert_indices,
                                                                   num_tokens,
                                                                   num_experts,
                                                                   top_k,
                                                                   normalize_weights,
                                                                   softmax_weights);
}
