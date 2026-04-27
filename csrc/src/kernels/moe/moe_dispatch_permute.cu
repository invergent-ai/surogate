// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_dispatch_permute.cu.

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
__global__ void moe_compute_expert_counts_kernel(int* __restrict__ expert_counts,         // (num_experts,) output
                                                 const int* __restrict__ expert_indices,  // (num_tokens, top_k)
                                                 int num_tokens,
                                                 int top_k,
                                                 int num_experts) {
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
    int* __restrict__ gather_indices,        // (total_tokens,) output: index of token in original order
    int* __restrict__ scatter_indices,       // (total_tokens,) output: inverse mapping
    const int* __restrict__ expert_indices,  // (num_tokens, top_k)
    const int* __restrict__ expert_offsets,  // (num_experts + 1,) cumsum of expert_counts
    int* __restrict__ expert_positions,      // (num_experts,) current write position per expert
    int num_tokens,
    int top_k,
    int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_assignments = num_tokens * top_k;

    if (idx >= total_assignments) return;

    int expert_id = expert_indices[idx];
    if (expert_id < 0 || expert_id >= num_experts) return;

    const int lane = threadIdx.x & 31;

    // Warp-aggregated atomic: find all lanes in this warp targeting the same expert
    // and batch them into a single atomicAdd.
    // We iterate over the set of unique expert_ids in this warp using a peer mask.
    int remaining_mask = __ballot_sync(0xFFFFFFFFu, true);  // mask of all active lanes
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
    gather_indices[dest_idx] = idx;   // Token assignment idx -> goes to position dest_idx
    scatter_indices[idx] = dest_idx;  // Inverse mapping
}

// Deterministic gather/scatter index construction.
// Assignments are processed in strictly increasing `idx` order, so each expert's
// local ordering is stable and identical across replicas/devices.
// This avoids EP rank divergence caused by nondeterministic atomic scheduling.
__global__ void
moe_compute_gather_indices_deterministic_kernel(int* __restrict__ gather_indices,        // (total_tokens,) output
                                                int* __restrict__ scatter_indices,       // (total_tokens,) output
                                                const int* __restrict__ expert_indices,  // (num_tokens, top_k)
                                                const int* __restrict__ expert_offsets,  // (num_experts + 1)
                                                int* __restrict__ expert_positions,  // (num_experts,) running positions
                                                int total_assignments,
                                                int num_experts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int idx = 0; idx < total_assignments; ++idx) {
        const int expert_id = expert_indices[idx];
        if (expert_id < 0 || expert_id >= num_experts) {
            continue;
        }
        const int slot = expert_positions[expert_id]++;
        const int dest_idx = expert_offsets[expert_id] + slot;
        gather_indices[dest_idx] = idx;
        scatter_indices[idx] = dest_idx;
    }
}

// Permute hidden states from token order to expert-grouped order
template <typename T>
__global__ void moe_permute_tokens_kernel(T* __restrict__ out,                     // (total_tokens, hidden_size)
                                          const T* __restrict__ inp,               // (num_tokens, hidden_size)
                                          const int* __restrict__ gather_indices,  // (total_tokens,)
                                          int total_tokens,
                                          int num_tokens,
                                          int hidden_size,
                                          int top_k) {
    using x128 = GenericVector<T, 16 / sizeof(T)>;

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
template <typename T>
__global__ void moe_unpermute_and_combine_kernel(T* __restrict__ out,                    // (num_tokens, hidden_size)
                                                 const T* __restrict__ expert_out,       // (total_tokens, hidden_size)
                                                 const T* __restrict__ routing_weights,  // (num_tokens, top_k)
                                                 const int* __restrict__ scatter_indices,  // (total_tokens,)
                                                 int num_tokens,
                                                 int total_tokens,
                                                 int hidden_size,
                                                 int top_k) {
    using x128 = GenericVector<T, 16 / sizeof(T)>;

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
template <typename T>
__global__ void
moe_unpermute_and_combine_kernel_mixed(T* __restrict__ out,                        // (num_tokens, hidden_size)
                                       const T* __restrict__ expert_out,           // (total_tokens, hidden_size)
                                       const float* __restrict__ routing_weights,  // (num_tokens, top_k) in FP32
                                       const int* __restrict__ scatter_indices,    // (total_tokens,)
                                       int num_tokens,
                                       int total_tokens,
                                       int hidden_size,
                                       int top_k) {
    using x128 = GenericVector<T, 16 / sizeof(T)>;

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

void moe_compute_expert_counts(int* expert_counts,
                               const int* expert_indices,
                               int num_tokens,
                               int top_k,
                               int num_experts,
                               cudaStream_t stream) {
    // Zero the output first
    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);

    int total = num_tokens * top_k;
    if (total == 0) return;  // No tokens to count

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    size_t smem = num_experts * sizeof(int);
    moe_compute_expert_counts_kernel<<<grid_size, block_size, smem, stream>>>(expert_counts,
                                                                              expert_indices,
                                                                              num_tokens,
                                                                              top_k,
                                                                              num_experts);
}

// Compute exclusive prefix sum of expert_counts into expert_offsets (length num_experts + 1).
// num_experts is small (typically <= 128), so a single-thread kernel is sufficient and avoids
// alignment/temporary-storage pitfalls of generic scan implementations.
__global__ void moe_compute_expert_offsets_kernel(int* __restrict__ expert_offsets,
                                                  const int* __restrict__ expert_counts,
                                                  int num_experts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int sum = 0;
    expert_offsets[0] = 0;
    for (int e = 0; e < num_experts; ++e) {
        sum += expert_counts[e];
        expert_offsets[e + 1] = sum;
    }
}

void moe_compute_expert_offsets(int* expert_offsets, const int* expert_counts, int num_experts, cudaStream_t stream) {
    // Compute exclusive prefix sum of expert_counts.
    // We intentionally use a tiny single-thread kernel here:
    // - num_experts is small (typically <= 128)
    // - avoids CUB alignment pitfalls (expert_offsets + 1 may be misaligned)
    // - avoids per-call cudaMallocAsync/cudaFreeAsync overhead
    moe_compute_expert_offsets_kernel<<<1, 1, 0, stream>>>(expert_offsets, expert_counts, num_experts);
    CUDA_CHECK(cudaGetLastError());
}

void moe_build_indices(int* gather_indices,
                       int* scatter_indices,
                       const int* expert_indices,
                       const int* expert_offsets,
                       int* expert_positions,
                       int num_tokens,
                       int top_k,
                       int num_experts,
                       cudaStream_t stream) {
    int total = num_tokens * top_k;
    if (total == 0) return;  // No tokens to index

    // Use deterministic index construction so EP replicas observe identical
    // per-expert token ordering across devices/ranks.
    moe_compute_gather_indices_deterministic_kernel<<<1, 1, 0, stream>>>(gather_indices,
                                                                         scatter_indices,
                                                                         expert_indices,
                                                                         expert_offsets,
                                                                         expert_positions,
                                                                         total,
                                                                         num_experts);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Expert Index Remapping Kernel for Selective Dequantization
// ============================================================================

__global__ void moe_remap_expert_indices_kernel(int* __restrict__ remapped_indices,
                                                const int* __restrict__ expert_indices,
                                                const int* __restrict__ expert_to_compact,
                                                int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int global_expert = expert_indices[idx];
    int compact_expert = expert_to_compact[global_expert];
    remapped_indices[idx] = compact_expert;
}

void moe_remap_expert_indices(int* remapped_indices,
                              const int* expert_indices,
                              const int* expert_to_compact,
                              int num_tokens,
                              int top_k,
                              cudaStream_t stream) {
    int total = num_tokens * top_k;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    moe_remap_expert_indices_kernel<<<grid_size, block_size, 0, stream>>>(remapped_indices,
                                                                          expert_indices,
                                                                          expert_to_compact,
                                                                          total);
}

void moe_permute_tokens(nv_bfloat16* out,
                        const nv_bfloat16* inp,
                        const int* gather_indices,
                        int total_tokens,
                        int num_tokens,
                        int hidden_size,
                        int top_k,
                        cudaStream_t stream) {
    if (total_tokens == 0) return;
    int block_size = 256;
    int grid_size = total_tokens;
    moe_permute_tokens_kernel<nv_bfloat16>
        <<<grid_size, block_size, 0, stream>>>(out, inp, gather_indices, total_tokens, num_tokens, hidden_size, top_k);
}

void moe_permute_tokens(float* out,
                        const float* inp,
                        const int* gather_indices,
                        int total_tokens,
                        int num_tokens,
                        int hidden_size,
                        int top_k,
                        cudaStream_t stream) {
    if (total_tokens == 0) return;
    int block_size = 256;
    int grid_size = total_tokens;
    moe_permute_tokens_kernel<float>
        <<<grid_size, block_size, 0, stream>>>(out, inp, gather_indices, total_tokens, num_tokens, hidden_size, top_k);
}

void moe_unpermute_and_combine(nv_bfloat16* out,
                               const nv_bfloat16* expert_out,
                               const nv_bfloat16* routing_weights,
                               const int* scatter_indices,
                               int num_tokens,
                               int total_tokens,
                               int hidden_size,
                               int top_k,
                               cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_unpermute_and_combine_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(out,
                                                                                        expert_out,
                                                                                        routing_weights,
                                                                                        scatter_indices,
                                                                                        num_tokens,
                                                                                        total_tokens,
                                                                                        hidden_size,
                                                                                        top_k);
}

void moe_unpermute_and_combine(nv_bfloat16* out,
                               const nv_bfloat16* expert_out,
                               const float* routing_weights,
                               const int* scatter_indices,
                               int num_tokens,
                               int total_tokens,
                               int hidden_size,
                               int top_k,
                               cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_unpermute_and_combine_kernel_mixed<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(out,
                                                                                              expert_out,
                                                                                              routing_weights,
                                                                                              scatter_indices,
                                                                                              num_tokens,
                                                                                              total_tokens,
                                                                                              hidden_size,
                                                                                              top_k);
}

void moe_unpermute_and_combine(float* out,
                               const float* expert_out,
                               const float* routing_weights,
                               const int* scatter_indices,
                               int num_tokens,
                               int total_tokens,
                               int hidden_size,
                               int top_k,
                               cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    moe_unpermute_and_combine_kernel<float><<<grid_size, block_size, 0, stream>>>(out,
                                                                                  expert_out,
                                                                                  routing_weights,
                                                                                  scatter_indices,
                                                                                  num_tokens,
                                                                                  total_tokens,
                                                                                  hidden_size,
                                                                                  top_k);
}
