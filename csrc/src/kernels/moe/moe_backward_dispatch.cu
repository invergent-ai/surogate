// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_backward_dispatch.cu.

// Backward through unpermute+combine: scatter gradient to expert outputs
// d_expert_outputs[permuted_idx] = routing_weights[token, k] * d_output[token]
template <typename T>
__global__ void
moe_combine_backward_kernel(T* __restrict__ d_expert_out,           // (total_tokens, hidden_size)
                            T* __restrict__ d_routing_weights,      // (num_tokens, top_k) - optional, can be NULL
                            const T* __restrict__ d_output,         // (num_tokens, hidden_size)
                            const T* __restrict__ expert_out,       // (total_tokens, hidden_size) - for weight gradient
                            const T* __restrict__ routing_weights,  // (num_tokens, top_k)
                            const int* __restrict__ scatter_indices,  // (total_tokens,)
                            int num_tokens,
                            int total_tokens,
                            int hidden_size,
                            int top_k) {
    using x128 = GenericVector<T, 16 / sizeof(T)>;

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
template <typename T>
__global__ void moe_combine_backward_kernel_mixed(
    T* __restrict__ d_expert_out,               // (total_tokens, hidden_size)
    float* __restrict__ d_routing_weights,      // (num_tokens, top_k) - optional, can be NULL
    const T* __restrict__ d_output,             // (num_tokens, hidden_size)
    const T* __restrict__ expert_out,           // (total_tokens, hidden_size) - for weight gradient
    const float* __restrict__ routing_weights,  // (num_tokens, top_k) in FP32
    const int* __restrict__ scatter_indices,    // (total_tokens,)
    int num_tokens,
    int total_tokens,
    int hidden_size,
    int top_k) {
    using x128 = GenericVector<T, 16 / sizeof(T)>;

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
__global__ void moe_permute_backward_kernel_fp32(float* __restrict__ d_input,             // (num_tokens, hidden_size)
                                                 const float* __restrict__ d_permuted,    // (total_tokens, hidden_size)
                                                 const int* __restrict__ gather_indices,  // (total_tokens,)
                                                 int total_tokens,
                                                 int num_tokens,
                                                 int hidden_size,
                                                 int top_k) {
    using x128 = GenericVector<float, 16 / sizeof(float)>;

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
__global__ void
moe_permute_backward_kernel_bf16(nv_bfloat16* __restrict__ d_input,           // (num_tokens, hidden_size)
                                 const nv_bfloat16* __restrict__ d_permuted,  // (total_tokens, hidden_size)
                                 const int* __restrict__ gather_indices,      // (total_tokens,)
                                 int total_tokens,
                                 int num_tokens,
                                 int hidden_size,
                                 int top_k) {
    using x128 = GenericVector<nv_bfloat16, 16 / sizeof(nv_bfloat16)>;

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

template <typename T, typename AccT>
__global__ void moe_permute_backward_from_scatter_kernel(
    T* __restrict__ d_input,                  // (num_tokens, hidden_size)
    const T* __restrict__ d_permuted,         // (total_tokens, hidden_size)
    const int* __restrict__ scatter_indices,  // (total_tokens,) assignment_idx -> permuted pos
    int total_tokens,
    int num_tokens,
    int hidden_size,
    int top_k) {
    using x128 = GenericVector<T, 16 / sizeof(T)>;

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    T* d_in = d_input + token_idx * hidden_size;
    const int assignment_base = token_idx * top_k;

    int d = threadIdx.x * x128::size;
    for (; d + x128::size <= hidden_size; d += blockDim.x * x128::size) {
        AccT acc[x128::size];
#pragma unroll
        for (int i = 0; i < x128::size; ++i) {
            acc[i] = AccT(0);
        }
        for (int k = 0; k < top_k; ++k) {
            const int assignment_idx = assignment_base + k;
            const int expert_pos = scatter_indices[assignment_idx];
            if (expert_pos < 0 || expert_pos >= total_tokens) continue;
            x128 val = x128::load(d_permuted + expert_pos * hidden_size + d);
#pragma unroll
            for (int i = 0; i < x128::size; ++i) {
                acc[i] += static_cast<AccT>(val[i]);
            }
        }
        x128 out;
#pragma unroll
        for (int i = 0; i < x128::size; ++i) {
            out[i] = static_cast<T>(acc[i]);
        }
        out.store(d_in + d);
    }

    for (int r = (hidden_size / x128::size) * x128::size + threadIdx.x; r < hidden_size; r += blockDim.x) {
        AccT acc = AccT(0);
        for (int k = 0; k < top_k; ++k) {
            const int assignment_idx = assignment_base + k;
            const int expert_pos = scatter_indices[assignment_idx];
            if (expert_pos < 0 || expert_pos >= total_tokens) continue;
            acc += static_cast<AccT>(d_permuted[expert_pos * hidden_size + r]);
        }
        d_in[r] = static_cast<T>(acc);
    }
}

// ============================================================================
// Backward Host Wrapper Functions

void moe_combine_backward(nv_bfloat16* d_expert_out,
                          nv_bfloat16* d_routing_weights,
                          const nv_bfloat16* d_output,
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
    moe_combine_backward_kernel<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(d_expert_out,
                                                                                   d_routing_weights,
                                                                                   d_output,
                                                                                   expert_out,
                                                                                   routing_weights,
                                                                                   scatter_indices,
                                                                                   num_tokens,
                                                                                   total_tokens,
                                                                                   hidden_size,
                                                                                   top_k);
}

void moe_combine_backward(nv_bfloat16* d_expert_out,
                          float* d_routing_weights,
                          const nv_bfloat16* d_output,
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
    moe_combine_backward_kernel_mixed<nv_bfloat16><<<grid_size, block_size, 0, stream>>>(d_expert_out,
                                                                                         d_routing_weights,
                                                                                         d_output,
                                                                                         expert_out,
                                                                                         routing_weights,
                                                                                         scatter_indices,
                                                                                         num_tokens,
                                                                                         total_tokens,
                                                                                         hidden_size,
                                                                                         top_k);
}

void moe_combine_backward(float* d_expert_out,
                          float* d_routing_weights,
                          const float* d_output,
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
    moe_combine_backward_kernel<float><<<grid_size, block_size, 0, stream>>>(d_expert_out,
                                                                             d_routing_weights,
                                                                             d_output,
                                                                             expert_out,
                                                                             routing_weights,
                                                                             scatter_indices,
                                                                             num_tokens,
                                                                             total_tokens,
                                                                             hidden_size,
                                                                             top_k);
}

void moe_permute_backward(nv_bfloat16* d_input,
                          const nv_bfloat16* d_permuted,
                          const int* gather_indices,
                          int total_tokens,
                          int num_tokens,
                          int hidden_size,
                          int top_k,
                          cudaStream_t stream) {
    // Zero the output first (since we're accumulating)
    cudaMemsetAsync(d_input, 0, num_tokens * hidden_size * sizeof(nv_bfloat16), stream);

    int block_size = 256;
    int grid_size = total_tokens;
    moe_permute_backward_kernel_bf16<<<grid_size, block_size, 0, stream>>>(d_input,
                                                                           d_permuted,
                                                                           gather_indices,
                                                                           total_tokens,
                                                                           num_tokens,
                                                                           hidden_size,
                                                                           top_k);
}

void moe_permute_backward(float* d_input,
                          const float* d_permuted,
                          const int* gather_indices,
                          int total_tokens,
                          int num_tokens,
                          int hidden_size,
                          int top_k,
                          cudaStream_t stream) {
    cudaMemsetAsync(d_input, 0, num_tokens * hidden_size * sizeof(float), stream);

    int block_size = 256;
    int grid_size = total_tokens;
    moe_permute_backward_kernel_fp32<<<grid_size, block_size, 0, stream>>>(d_input,
                                                                           d_permuted,
                                                                           gather_indices,
                                                                           total_tokens,
                                                                           num_tokens,
                                                                           hidden_size,
                                                                           top_k);
}

void moe_permute_backward_from_scatter(nv_bfloat16* d_input,
                                       const nv_bfloat16* d_permuted,
                                       const int* scatter_indices,
                                       int total_tokens,
                                       int num_tokens,
                                       int hidden_size,
                                       int top_k,
                                       cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = num_tokens;
    moe_permute_backward_from_scatter_kernel<nv_bfloat16, float><<<grid_size, block_size, 0, stream>>>(d_input,
                                                                                                       d_permuted,
                                                                                                       scatter_indices,
                                                                                                       total_tokens,
                                                                                                       num_tokens,
                                                                                                       hidden_size,
                                                                                                       top_k);
}

void moe_permute_backward_from_scatter(float* d_input,
                                       const float* d_permuted,
                                       const int* scatter_indices,
                                       int total_tokens,
                                       int num_tokens,
                                       int hidden_size,
                                       int top_k,
                                       cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = num_tokens;
    moe_permute_backward_from_scatter_kernel<float, float><<<grid_size, block_size, 0, stream>>>(d_input,
                                                                                                 d_permuted,
                                                                                                 scatter_indices,
                                                                                                 total_tokens,
                                                                                                 num_tokens,
                                                                                                 hidden_size,
                                                                                                 top_k);
}
