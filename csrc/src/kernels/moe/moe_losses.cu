// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_losses.cu.

// ============================================================================
// Auxiliary Loss Computation
// ============================================================================
// Load-balancing loss to encourage uniform expert utilization.
// aux_loss = alpha * num_experts * sum_e(f_e * P_e)
// where f_e = fraction of tokens routed to expert e
//       P_e = average routing probability to expert e

template <typename T>
__global__ void moe_aux_loss_kernel(float* __restrict__ aux_loss,            // scalar output
                                    float* __restrict__ router_z_loss,       // scalar output (optional)
                                    const T* __restrict__ routing_probs,     // (num_tokens, num_experts) - post softmax
                                    const int* __restrict__ expert_indices,  // (num_tokens, top_k)
                                    int num_tokens,
                                    int num_experts,
                                    int top_k,
                                    float aux_loss_coef,
                                    float z_loss_coef) {
    // This kernel computes both load-balancing loss and router z-loss
    // For simplicity, use atomics; production should use proper reductions

    extern __shared__ float smem[];
    float* expert_fractions = smem;            // num_experts
    float* expert_probs = smem + num_experts;  // num_experts

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

template <typename T, int BLOCK_SIZE = 256>
__global__ void
moe_router_z_loss_kernel(float* __restrict__ z_loss,           // scalar output (accumulated via atomicAdd)
                         const T* __restrict__ router_logits,  // (num_tokens, num_experts) - pre-softmax
                         int num_tokens,
                         int num_experts,
                         float z_loss_coef) {
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
template <typename T, int BLOCK_SIZE = 256>
__global__ void
moe_router_z_loss_backward_kernel(T* __restrict__ d_logits,             // (num_tokens, num_experts) - gradient output
                                  const T* __restrict__ router_logits,  // (num_tokens, num_experts) - pre-softmax
                                  int num_tokens,
                                  int num_experts,
                                  float z_loss_coef) {
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

void moe_compute_aux_loss(float* aux_loss,
                          const nv_bfloat16* routing_probs,
                          const int* expert_indices,
                          int num_tokens,
                          int num_experts,
                          int top_k,
                          float aux_loss_coef,
                          cudaStream_t stream) {
    // Initialize output
    cudaMemsetAsync(aux_loss, 0, sizeof(float), stream);

    int block_size = 256;
    int shared_mem = 2 * num_experts * sizeof(float);
    moe_aux_loss_kernel<nv_bfloat16><<<1, block_size, shared_mem, stream>>>(aux_loss,
                                                                            nullptr,
                                                                            routing_probs,
                                                                            expert_indices,
                                                                            num_tokens,
                                                                            num_experts,
                                                                            top_k,
                                                                            aux_loss_coef,
                                                                            0.0f);
}

void moe_compute_aux_loss(float* aux_loss,
                          const float* routing_probs,
                          const int* expert_indices,
                          int num_tokens,
                          int num_experts,
                          int top_k,
                          float aux_loss_coef,
                          cudaStream_t stream) {
    cudaMemsetAsync(aux_loss, 0, sizeof(float), stream);

    int block_size = 256;
    int shared_mem = 2 * num_experts * sizeof(float);
    moe_aux_loss_kernel<float><<<1, block_size, shared_mem, stream>>>(aux_loss,
                                                                      nullptr,
                                                                      routing_probs,
                                                                      expert_indices,
                                                                      num_tokens,
                                                                      num_experts,
                                                                      top_k,
                                                                      aux_loss_coef,
                                                                      0.0f);
}

// ============================================================================
// Routing Statistics Kernel (for monitoring — not on gradient path)
// ============================================================================
// Computes aux_loss, expert_utilization, load_imbalance, and router/load diagnostics in a single pass
// and accumulates into a persistent stats buffer via atomicAdd.
// stats layout:
// [aux_loss_sum, z_loss_sum, utilization_sum, load_imbalance_sum, layer_count,
//  active_experts_sum, max_expert_fraction_sum, min_active_expert_fraction_sum,
//  load_cv_sum, router_entropy_sum, router_confidence_sum]

template <typename T>
__global__ void
moe_routing_stats_kernel(float* __restrict__ stats,               // accumulated stats
                         const T* __restrict__ routing_probs,     // (num_tokens, num_experts) post-softmax/sigmoid
                         const int* __restrict__ expert_indices,  // (num_tokens, top_k)
                         int num_tokens,
                         int num_experts,
                         int top_k,
                         float aux_loss_coef) {
    extern __shared__ float smem[];
    float* expert_counts = smem;               // num_experts
    float* expert_probs = smem + num_experts;  // num_experts
    float* router_entropy = smem + 2 * num_experts;
    float* router_confidence = smem + 2 * num_experts + 1;

    // Initialize shared memory
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        expert_counts[e] = 0.0f;
        expert_probs[e] = 0.0f;
    }
    if (threadIdx.x == 0) {
        *router_entropy = 0.0f;
        *router_confidence = 0.0f;
    }
    __syncthreads();

    // Count tokens per expert
    int total_assignments = num_tokens * top_k;
    for (int i = threadIdx.x; i < total_assignments; i += blockDim.x) {
        int expert_id = expert_indices[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&expert_counts[expert_id], 1.0f);
        }
    }

    // Compute average routing probability per expert
    for (int t = threadIdx.x; t < num_tokens; t += blockDim.x) {
        const T* row = routing_probs + t * num_experts;
        float row_sum = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            row_sum += fmaxf(static_cast<float>(row[e]), 0.0f);
        }
        row_sum = fmaxf(row_sum, 1e-20f);
        float entropy = 0.0f;
        float max_prob = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            float prob = fmaxf(static_cast<float>(row[e]), 0.0f);
            const float norm_prob = prob / row_sum;
            if (norm_prob > 0.0f) {
                entropy -= norm_prob * logf(norm_prob);
            }
            max_prob = fmaxf(max_prob, norm_prob);
            atomicAdd(&expert_probs[e], prob / num_tokens);
        }
        const float denom = (num_experts > 1) ? logf(static_cast<float>(num_experts)) : 1.0f;
        atomicAdd(router_entropy, entropy / denom);
        atomicAdd(router_confidence, max_prob);
    }
    __syncthreads();

    // Single thread computes final stats
    if (threadIdx.x == 0) {
        // Aux loss: coef * num_experts * sum(f_e * P_e)
        float aux_loss = 0.0f;
        float max_count = 0.0f;
        float min_active_count = static_cast<float>(total_assignments + 1);
        float count_sq_sum = 0.0f;
        float mean_count = static_cast<float>(total_assignments) / num_experts;
        int active_experts = 0;

        for (int e = 0; e < num_experts; e++) {
            float fraction = expert_counts[e] / total_assignments;
            aux_loss += fraction * expert_probs[e];
            if (expert_counts[e] > max_count) max_count = expert_counts[e];
            count_sq_sum += expert_counts[e] * expert_counts[e];
            if (expert_counts[e] > 0.0f) {
                min_active_count = fminf(min_active_count, expert_counts[e]);
                active_experts++;
            }
        }
        aux_loss *= num_experts * aux_loss_coef;

        float utilization = static_cast<float>(active_experts) / num_experts;
        float load_imbalance = (mean_count > 0.0f) ? (max_count / mean_count) : 0.0f;
        const float count_sq_mean = count_sq_sum / num_experts;
        const float variance = fmaxf(count_sq_mean - mean_count * mean_count, 0.0f);
        const float load_cv = (mean_count > 0.0f) ? (sqrtf(variance) / mean_count) : 0.0f;
        const float total = static_cast<float>(total_assignments);
        const float max_expert_fraction = (total > 0.0f) ? (max_count / total) : 0.0f;
        const float min_active_expert_fraction =
            (active_experts > 0 && total > 0.0f) ? (min_active_count / total) : 0.0f;
        const float avg_router_entropy = (num_tokens > 0) ? (*router_entropy / num_tokens) : 0.0f;
        const float avg_router_confidence = (num_tokens > 0) ? (*router_confidence / num_tokens) : 0.0f;

        atomicAdd(&stats[0], aux_loss);
        // stats[1] (z_loss) not computed here — needs pre-softmax logits
        atomicAdd(&stats[2], utilization);
        atomicAdd(&stats[3], load_imbalance);
        atomicAdd(&stats[4], 1.0f);  // layer count
        atomicAdd(&stats[5], static_cast<float>(active_experts));
        atomicAdd(&stats[6], max_expert_fraction);
        atomicAdd(&stats[7], min_active_expert_fraction);
        atomicAdd(&stats[8], load_cv);
        atomicAdd(&stats[9], avg_router_entropy);
        atomicAdd(&stats[10], avg_router_confidence);
    }
}

void moe_compute_routing_stats(float* stats,
                               const nv_bfloat16* routing_probs,
                               const int* expert_indices,
                               int num_tokens,
                               int num_experts,
                               int top_k,
                               float aux_loss_coef,
                               cudaStream_t stream) {
    int block_size = 256;
    int shared_mem = (2 * num_experts + 2) * sizeof(float);
    moe_routing_stats_kernel<nv_bfloat16><<<1, block_size, shared_mem, stream>>>(stats,
                                                                                 routing_probs,
                                                                                 expert_indices,
                                                                                 num_tokens,
                                                                                 num_experts,
                                                                                 top_k,
                                                                                 aux_loss_coef);
}

void moe_compute_routing_stats(float* stats,
                               const float* routing_probs,
                               const int* expert_indices,
                               int num_tokens,
                               int num_experts,
                               int top_k,
                               float aux_loss_coef,
                               cudaStream_t stream) {
    int block_size = 256;
    int shared_mem = (2 * num_experts + 2) * sizeof(float);
    moe_routing_stats_kernel<float><<<1, block_size, shared_mem, stream>>>(stats,
                                                                           routing_probs,
                                                                           expert_indices,
                                                                           num_tokens,
                                                                           num_experts,
                                                                           top_k,
                                                                           aux_loss_coef);
}

template <typename T>
__global__ void moe_routing_stats_from_logits_kernel(float* __restrict__ stats,
                                                     const T* __restrict__ routing_logits,
                                                     const int* __restrict__ expert_indices,
                                                     int num_tokens,
                                                     int num_experts,
                                                     int top_k,
                                                     float aux_loss_coef,
                                                     float z_loss_coef) {
    extern __shared__ float smem[];
    float* expert_counts = smem;
    float* expert_probs = smem + num_experts;
    float* router_entropy = smem + 2 * num_experts;
    float* router_confidence = smem + 2 * num_experts + 1;
    float* router_z_loss = smem + 2 * num_experts + 2;

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        expert_counts[e] = 0.0f;
        expert_probs[e] = 0.0f;
    }
    if (threadIdx.x == 0) {
        *router_entropy = 0.0f;
        *router_confidence = 0.0f;
        *router_z_loss = 0.0f;
    }
    __syncthreads();

    const int total_assignments = num_tokens * top_k;
    for (int i = threadIdx.x; i < total_assignments; i += blockDim.x) {
        int expert_id = expert_indices[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&expert_counts[expert_id], 1.0f);
        }
    }

    for (int t = threadIdx.x; t < num_tokens; t += blockDim.x) {
        const T* row = routing_logits + t * num_experts;
        float max_logit = -INFINITY;
        for (int e = 0; e < num_experts; ++e) {
            max_logit = fmaxf(max_logit, static_cast<float>(row[e]));
        }
        float denom = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            denom += expf(static_cast<float>(row[e]) - max_logit);
        }
        denom = fmaxf(denom, 1e-20f);
        const float logsumexp = max_logit + logf(denom);
        float entropy = 0.0f;
        float max_prob = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            const float prob = expf(static_cast<float>(row[e]) - max_logit) / denom;
            if (prob > 0.0f) {
                entropy -= prob * logf(prob);
            }
            max_prob = fmaxf(max_prob, prob);
            atomicAdd(&expert_probs[e], prob / num_tokens);
        }
        const float entropy_denom = (num_experts > 1) ? logf(static_cast<float>(num_experts)) : 1.0f;
        atomicAdd(router_entropy, entropy / entropy_denom);
        atomicAdd(router_confidence, max_prob);
        atomicAdd(router_z_loss, z_loss_coef * logsumexp * logsumexp / fmaxf(static_cast<float>(num_tokens), 1.0f));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float aux_loss = 0.0f;
        float max_count = 0.0f;
        float min_active_count = static_cast<float>(total_assignments + 1);
        float count_sq_sum = 0.0f;
        const float mean_count = static_cast<float>(total_assignments) / num_experts;
        int active_experts = 0;

        for (int e = 0; e < num_experts; e++) {
            const float fraction = expert_counts[e] / total_assignments;
            aux_loss += fraction * expert_probs[e];
            if (expert_counts[e] > max_count) max_count = expert_counts[e];
            count_sq_sum += expert_counts[e] * expert_counts[e];
            if (expert_counts[e] > 0.0f) {
                min_active_count = fminf(min_active_count, expert_counts[e]);
                active_experts++;
            }
        }
        aux_loss *= num_experts * aux_loss_coef;

        const float utilization = static_cast<float>(active_experts) / num_experts;
        const float load_imbalance = (mean_count > 0.0f) ? (max_count / mean_count) : 0.0f;
        const float count_sq_mean = count_sq_sum / num_experts;
        const float variance = fmaxf(count_sq_mean - mean_count * mean_count, 0.0f);
        const float load_cv = (mean_count > 0.0f) ? (sqrtf(variance) / mean_count) : 0.0f;
        const float total = static_cast<float>(total_assignments);
        const float max_expert_fraction = (total > 0.0f) ? (max_count / total) : 0.0f;
        const float min_active_expert_fraction =
            (active_experts > 0 && total > 0.0f) ? (min_active_count / total) : 0.0f;
        const float avg_router_entropy = (num_tokens > 0) ? (*router_entropy / num_tokens) : 0.0f;
        const float avg_router_confidence = (num_tokens > 0) ? (*router_confidence / num_tokens) : 0.0f;

        atomicAdd(&stats[0], aux_loss);
        atomicAdd(&stats[1], *router_z_loss);
        atomicAdd(&stats[2], utilization);
        atomicAdd(&stats[3], load_imbalance);
        atomicAdd(&stats[4], 1.0f);
        atomicAdd(&stats[5], static_cast<float>(active_experts));
        atomicAdd(&stats[6], max_expert_fraction);
        atomicAdd(&stats[7], min_active_expert_fraction);
        atomicAdd(&stats[8], load_cv);
        atomicAdd(&stats[9], avg_router_entropy);
        atomicAdd(&stats[10], avg_router_confidence);
    }
}

void moe_compute_routing_stats_from_logits(float* stats,
                                           const nv_bfloat16* routing_logits,
                                           const int* expert_indices,
                                           int num_tokens,
                                           int num_experts,
                                           int top_k,
                                           float aux_loss_coef,
                                           float z_loss_coef,
                                           cudaStream_t stream) {
    int block_size = 256;
    int shared_mem = (2 * num_experts + 3) * sizeof(float);
    moe_routing_stats_from_logits_kernel<nv_bfloat16><<<1, block_size, shared_mem, stream>>>(stats,
                                                                                             routing_logits,
                                                                                             expert_indices,
                                                                                             num_tokens,
                                                                                             num_experts,
                                                                                             top_k,
                                                                                             aux_loss_coef,
                                                                                             z_loss_coef);
}

void moe_compute_routing_stats_from_logits(float* stats,
                                           const float* routing_logits,
                                           const int* expert_indices,
                                           int num_tokens,
                                           int num_experts,
                                           int top_k,
                                           float aux_loss_coef,
                                           float z_loss_coef,
                                           cudaStream_t stream) {
    int block_size = 256;
    int shared_mem = (2 * num_experts + 3) * sizeof(float);
    moe_routing_stats_from_logits_kernel<float><<<1, block_size, shared_mem, stream>>>(stats,
                                                                                       routing_logits,
                                                                                       expert_indices,
                                                                                       num_tokens,
                                                                                       num_experts,
                                                                                       top_k,
                                                                                       aux_loss_coef,
                                                                                       z_loss_coef);
}

void moe_router_z_loss_forward(float* z_loss,
                               const nv_bfloat16* router_logits,
                               int num_tokens,
                               int num_experts,
                               float z_loss_coef,
                               cudaStream_t stream) {
    // Initialize output to zero (will be accumulated via atomicAdd)
    cudaMemsetAsync(z_loss, 0, sizeof(float), stream);

    int block_size = 256;
    int grid_size = num_tokens;
    moe_router_z_loss_kernel<nv_bfloat16>
        <<<grid_size, block_size, 0, stream>>>(z_loss, router_logits, num_tokens, num_experts, z_loss_coef);
}

void moe_router_z_loss_forward(float* z_loss,
                               const float* router_logits,
                               int num_tokens,
                               int num_experts,
                               float z_loss_coef,
                               cudaStream_t stream) {
    cudaMemsetAsync(z_loss, 0, sizeof(float), stream);

    int block_size = 256;
    int grid_size = num_tokens;
    moe_router_z_loss_kernel<float>
        <<<grid_size, block_size, 0, stream>>>(z_loss, router_logits, num_tokens, num_experts, z_loss_coef);
}

void moe_router_z_loss_backward(nv_bfloat16* d_logits,
                                const nv_bfloat16* router_logits,
                                int num_tokens,
                                int num_experts,
                                float z_loss_coef,
                                cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_router_z_loss_backward_kernel<nv_bfloat16>
        <<<grid_size, block_size, 0, stream>>>(d_logits, router_logits, num_tokens, num_experts, z_loss_coef);
}

void moe_router_z_loss_backward(float* d_logits,
                                const float* router_logits,
                                int num_tokens,
                                int num_experts,
                                float z_loss_coef,
                                cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_router_z_loss_backward_kernel<float>
        <<<grid_size, block_size, 0, stream>>>(d_logits, router_logits, num_tokens, num_experts, z_loss_coef);
}
