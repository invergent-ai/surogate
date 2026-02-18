// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

/**
 * @file fused_classifier.cu
 * @brief Fused softmax, cross-entropy loss, and backward pass kernel.
 *
 * Implements a highly optimized kernel that fuses:
 * 1. Softmax computation over vocabulary dimension
 * 2. Cross-entropy loss calculation
 * 3. Gradient computation (dlogits = prob - indicator)
 *
 * This fusion reduces memory bandwidth by avoiding materialization of
 * intermediate probabilities and reading logits only twice (softmax prep + gradient).
 */

#include <cassert>

#include "kernel_utils.cuh"
#include "kernels.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

/// @brief Function pointer type for warp reduction operations.
using reduction_func_t = float (*) (float);

/**
 * @brief Block-wide reduction using warp shuffles and shared memory.
 *
 * Performs a two-level hierarchical reduction:
 * 1. Warp-level reduction using shuffle instructions
 * 2. Cross-warp reduction via shared memory
 * 3. Final warp-level reduction of partial results
 *
 * @note Requires all 32 threads in each warp to be active.
 * @note Uses non-dynamic shared memory (128 bytes per call).
 * @note If called inside a loop, set final_sync=true to avoid shared memory races.
 *
 * @tparam warp_reduction Warp reduction function (e.g., warpReduceMax, warpReduceSum).
 * @param val Thread's input value to reduce.
 * @param final_sync If true, adds __syncthreads() at end (needed in loops).
 * @param out_of_bounds Value for inactive lanes in final reduction (e.g., -INFINITY for max, 0 for sum).
 * @return The reduced value (same on all threads).
 */
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ float shared_val[32];
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}


/**
 * @brief Parameters for numerically stable softmax computation.
 *
 * Stores the normalization factor (1/sum) and offset (max) for computing
 * softmax as: prob[i] = exp(logit[i] - Offset) * Scale
 */
struct SoftmaxParams {
    float Scale;  ///< Reciprocal of sum of exponentials (1 / sum(exp(x - max)))
    float Offset; ///< Maximum logit value for numerical stability
};

/**
 * @brief Computes softmax parameters (max and sum) for one row using block-wide reduction.
 *
 * Uses online softmax algorithm to compute max and sum in a single pass:
 * - Maintains running max and sum, rescaling sum when max changes
 * - Vectorized loads (128-bit) for memory efficiency
 * - Two-pass block reduction: first for max, then for sum
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param idx Row index in the input tensor.
 * @param[in] inp Input logits tensor of shape (BT, P).
 * @param V Actual vocabulary size (may be < P due to padding).
 * @param P Padded vocabulary size (for memory alignment).
 * @return SoftmaxParams containing Scale (1/sum) and Offset (max).
 */
template<class floatX>
__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*static_cast<int>(x128::size) > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        x128 packed_x = x128::load(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

/**
 * @brief Fused kernel for softmax, cross-entropy loss, and gradient computation.
 *
 * Performs three operations in a single kernel launch:
 * 1. Softmax: Computes probabilities from logits (numerically stable)
 * 2. Loss: Computes cross-entropy loss = -log(prob[target])
 * 3. Gradient: Computes dlogits = (prob - indicator) * dloss
 *
 * Key optimizations:
 * - Reads logits twice: once for softmax prep, once for gradient (likely cached)
 * - Reverse block order for better cache hits on matmul output
 * - Vectorized 128-bit loads/stores
 * - Supports masked tokens (target == -100)
 *
 * @note Will _update_ logits to logit gradients in-place.
 * @note Uses template to decide whether to write logits and probs.
 * @note Split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @tparam WriteDLogits If true, writes gradients back to logits tensor.
 * @tparam WriteProbs If true, writes probabilities to probs tensor.
 * @param[in,out] logits Logits tensor of shape (BT, P), overwritten with gradients if WriteDLogits.
 * @param[in,out] losses Loss tensor of shape (BT,), accumulated (not overwritten).
 * @param[out] probs Probabilities tensor of shape (BT, P), written if WriteProbs.
 * @param dloss Upstream gradient scalar (typically 1.0 / num_tokens).
 * @param[in] targets Target token indices of shape (BT,), -100 for masked positions.
 * @param BT Batch size * sequence length.
 * @param V Actual vocabulary size.
 * @param P Padded vocabulary size.
 */
template <class floatX, bool WriteDLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, 1)
    fused_classifier_kernel5(floatX* logits, float* losses, floatX* probs,
                             const float dloss, const int* targets, int* valid_token_count,
                             int* correct_count,
                             int BT, int V, int P, std::bool_constant<WriteDLogits>) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx = gridDim.x - (blockIdx.x+1); // reverse order for cache hits on matmul data
    int ix = targets[idx];
    if(ix == -100) {
        if (WriteDLogits){
            x128 zero = x128::zeros();
            for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
                zero.store(logits + idx * P + i * x128::size);
            }
        }
        return;     // mask
    }
    assert(0 <= ix && ix < V);

    // Count this as a valid token (one thread per block to avoid races)
    if(threadIdx.x == 0 && valid_token_count != nullptr) {
        atomicAdd(valid_token_count, 1);
    }

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // Find argmax for accuracy computation (block-wide reduction)
    __shared__ int shared_max_idx[32];
    __shared__ float shared_max_val[32];

    const floatX* logits_vec = logits + idx * P;
    float thread_max_val = -INFINITY;
    int thread_max_idx = 0;

    // Each thread finds its local maximum
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float val = (float)logits_vec[i];
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_max_idx = i;
        }
    }

    // Warp-level reduction
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
        if (other_val > thread_max_val) {
            thread_max_val = other_val;
            thread_max_idx = other_idx;
        }
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_max_val[warp_id] = thread_max_val;
        shared_max_idx[warp_id] = thread_max_idx;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        thread_max_val = (lane_id < num_warps) ? shared_max_val[lane_id] : -INFINITY;
        thread_max_idx = (lane_id < num_warps) ? shared_max_idx[lane_id] : 0;

        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
            if (other_val > thread_max_val) {
                thread_max_val = other_val;
                thread_max_idx = other_idx;
            }
        }

        // Thread 0 has the final argmax, check if it matches target
        if (threadIdx.x == 0 && correct_count != nullptr) {
            if (thread_max_idx == ix) {
                atomicAdd(correct_count, 1);
            }
        }
    }
    __syncthreads();

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float logit_val = (float)logits[idx * P + ix];
        float prob = expf(logit_val - sp.Offset) * sp.Scale;
        float loss_contrib = -logf(prob);
        losses[idx] -= logf(prob);
    }

    // without this synchronization point we have a race condition:
    // the logits used above to compute the loss are concurrently (race) modified to carry backward pass grads.
    __syncthreads();

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    // Note: logits_vec already declared above for argmax computation
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = x128::load(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteDLogits){
            // reduce cache persistence for the overwritten logits
            // to maximise the probability that logits remain in cache between prepare_softmax and here
            packed_logits_vec.store_cs(logits + idx * P + i * x128::size);
        }
        if (WriteProbs) {
            packed_probs.store_cs(probs + idx * P + i * x128::size);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = threadIdx.x + unaligned_start; i < V; i += blockDim.x) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteDLogits){
            __stcs(logits + idx * P + i, (floatX)dlogit);
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

/**
 * @brief Template implementation for fused classifier kernel launch.
 *
 * Launches the fused_classifier_kernel5 with 1024 threads per block,
 * one block per sequence position. Dispatches based on write_dlogits flag.
 *
 * @note Replaces logits with logit gradients when write_dlogits is true.
 *
 * @tparam Type Data type (float or nv_bfloat16).
 * @param[in,out] logits Logits tensor, overwritten with gradients if write_dlogits.
 * @param[in,out] losses Loss tensor, accumulated.
 * @param dloss Upstream gradient scalar.
 * @param[in] targets Target token indices.
 * @param BT Batch size * sequence length.
 * @param V Actual vocabulary size.
 * @param P Padded vocabulary size.
 * @param write_dlogits If true, writes gradients to logits tensor.
 * @param stream CUDA stream for asynchronous execution.
 */
template <typename Type>
void fused_classifier_imp(Type* logits, float* losses,
                      const float dloss, const int* targets, int* valid_token_count,
                      int* correct_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    const int block_size = 1024;
    const int grid_size = BT;
    if(write_dlogits) {
        fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, (Type*) NULL, dloss, targets, valid_token_count,
                                                                       correct_count, BT, V, P, std::bool_constant<true>());
    } else {
        fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, (Type*) NULL, dloss, targets, valid_token_count,
                                                                       correct_count, BT, V, P, std::bool_constant<false>());
    }
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fused softmax + cross-entropy + backward for FP32 tensors.
 *
 * Computes cross-entropy loss and optionally the gradient in a single kernel.
 *
 * @param[in,out] logits Logits tensor of shape (BT, P) in FP32, overwritten with gradients.
 * @param[in,out] losses Loss tensor of shape (BT,) in FP32, accumulated.
 * @param dloss Upstream gradient scalar (typically 1.0 / num_tokens).
 * @param[in] targets Target token indices of shape (BT,), -100 for masked.
 * @param[out] valid_token_count Optional GPU buffer to accumulate count of non-masked tokens (nullptr to skip).
 * @param BT Batch size * sequence length.
 * @param V Actual vocabulary size.
 * @param P Padded vocabulary size.
 * @param write_dlogits If true, writes gradients to logits.
 * @param stream CUDA stream.
 */
void fused_classifier(float* logits, float* losses,
                      const float dloss, const int* targets, int* valid_token_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    fused_classifier_imp(logits, losses, dloss, targets, valid_token_count, nullptr, BT, V, P, write_dlogits, stream);
}

void fused_classifier(float* logits, float* losses,
                      const float dloss, const int* targets, int* valid_token_count,
                      int* correct_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    fused_classifier_imp(logits, losses, dloss, targets, valid_token_count, correct_count, BT, V, P, write_dlogits, stream);
}

/**
 * @brief Fused softmax + cross-entropy + backward for BF16 tensors.
 *
 * Computes cross-entropy loss and optionally the gradient in a single kernel.
 *
 * @param[in,out] logits Logits tensor of shape (BT, P) in BF16, overwritten with gradients.
 * @param[in,out] losses Loss tensor of shape (BT,) in FP32, accumulated.
 * @param dloss Upstream gradient scalar (typically 1.0 / num_tokens).
 * @param[in] targets Target token indices of shape (BT,), -100 for masked.
 * @param[out] valid_token_count Optional GPU buffer to accumulate count of non-masked tokens (nullptr to skip).
 * @param BT Batch size * sequence length.
 * @param V Actual vocabulary size.
 * @param P Padded vocabulary size.
 * @param write_dlogits If true, writes gradients to logits.
 * @param stream CUDA stream.
 */
void fused_classifier(nv_bfloat16* logits, float* losses,
                      const float dloss, const int* targets, int* valid_token_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    fused_classifier_imp(logits, losses, dloss, targets, valid_token_count, nullptr, BT, V, P, write_dlogits, stream);
}

void fused_classifier(nv_bfloat16* logits, float* losses,
                      const float dloss, const int* targets, int* valid_token_count,
                      int* correct_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    fused_classifier_imp(logits, losses, dloss, targets, valid_token_count, correct_count, BT, V, P, write_dlogits, stream);
}

// ----------------------------------------------------------------------------
// Cross-entropy forward/backward (non-fused with matmul) helpers

template <class floatX>
__global__ void cross_entropy_forward_kernel(const floatX* logits, float* losses, float* logsumexp,
                                             const int* targets, int* valid_token_count,
                                             int* correct_count, int BT, int V, int P) {
    int idx = static_cast<int>(blockIdx.x);
    if (idx >= BT) {
        return;
    }
    int ix = targets[idx];
    if (ix == -100) {
        // For padding tokens, don't touch the loss buffer to preserve accumulated values
        // from previous micro-steps. Only zero logsumexp for backward correctness.
        if (threadIdx.x == 0 && logsumexp) {
            logsumexp[idx] = 0.0f;
        }
        return;
    }

    if (threadIdx.x == 0 && valid_token_count) {
        atomicAdd(valid_token_count, 1);
    }

    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);
    float lse = sp.Offset + logf(1.0f / sp.Scale);

    // Optional accuracy computation (argmax)
    if (correct_count) {
        __shared__ int shared_max_idx[32];
        __shared__ float shared_max_val[32];

        const floatX* logits_vec = logits + static_cast<int64_t>(idx) * P;
        float thread_max_val = -INFINITY;
        int thread_max_idx = 0;

        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            float val = (float)logits_vec[i];
            if (val > thread_max_val) {
                thread_max_val = val;
                thread_max_idx = i;
            }
        }

        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int num_warps = blockDim.x / 32;

        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
            if (other_val > thread_max_val) {
                thread_max_val = other_val;
                thread_max_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            shared_max_val[warp_id] = thread_max_val;
            shared_max_idx[warp_id] = thread_max_idx;
        }
        __syncthreads();

        if (warp_id == 0) {
            thread_max_val = (lane_id < num_warps) ? shared_max_val[lane_id] : -INFINITY;
            thread_max_idx = (lane_id < num_warps) ? shared_max_idx[lane_id] : 0;

            for (int offset = 16; offset > 0; offset /= 2) {
                float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);
                int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
                if (other_val > thread_max_val) {
                    thread_max_val = other_val;
                    thread_max_idx = other_idx;
                }
            }

            if (threadIdx.x == 0) {
                if (thread_max_idx == ix) {
                    atomicAdd(correct_count, 1);
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float logit_val = (float)logits[idx * P + ix];
        // Accumulate loss (not overwrite) to support gradient accumulation
        losses[idx] += lse - logit_val;
        if (logsumexp) {
            logsumexp[idx] = lse;
        }
    }
}

template <class floatX>
__global__ void cross_entropy_backward_kernel(floatX* dlogits, const floatX* logits, const float* logsumexp,
                                              const float* dloss, const int* targets,
                                              int BT, int V, int P) {
    // HuggingFace-style normalization: dloss is already scaled by 1/accumulated_valid_tokens
    // at the caller level (GraphExecutor/CompiledExecutor). No per-batch token scaling here.
    int idx = static_cast<int>(blockIdx.x);
    if (idx >= BT) {
        return;
    }
    int ix = targets[idx];
    if (ix == -100) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            dlogits[idx * P + i] = (floatX)0.0f;
        }
        return;
    }

    float lse = 0.0f;
    if (logsumexp) {
        lse = logsumexp[idx];
    } else {
        SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);
        lse = sp.Offset + logf(1.0f / sp.Scale);
    }

    float dloss_val = dloss ? dloss[idx] : 1.0f;
    const floatX* logits_vec = logits + static_cast<int64_t>(idx) * P;

    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float prob = expf((float)logits_vec[i] - lse);
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss_val;
        dlogits[idx * P + i] = (floatX)dlogit;
    }
}

template <class floatX>
__global__ void chunked_cross_entropy_forward_kernel(const floatX* logits, float* losses, float* chunk_logsumexp,
                                                     const int* targets, int* valid_token_count,
                                                     int BT, int V, int P, int n_chunks, int chunk_size) {
    int row_idx = static_cast<int>(blockIdx.x);
    int chunk_idx = static_cast<int>(blockIdx.y);
    int start = chunk_idx * chunk_size;
    if (row_idx >= BT || start >= V) {
        return;
    }
    int end = (start + chunk_size < V) ? (start + chunk_size) : V;

    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = (float)logits[row_idx * P + i];
        float old_max = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= expf(old_max - thread_maxval);
        thread_sumval += expf(v - thread_maxval);
    }

    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    if (threadIdx.x == 0) {
        chunk_logsumexp[row_idx * n_chunks + chunk_idx] = block_maxval + logf(block_sumval);
        if (chunk_idx == 0) {
            int ix = targets[row_idx];
            if (ix != -100) {
                if (valid_token_count) {
                    atomicAdd(valid_token_count, 1);
                }
                float logit_val = (float)logits[row_idx * P + ix];
                // Accumulate loss (not overwrite) to support gradient accumulation
                losses[row_idx] -= logit_val;
            }
            // For padding tokens (ix == -100), don't touch the loss buffer
            // to preserve accumulated values from previous micro-steps
        }
    }
}

__global__ void logsumexp_reduce_kernel(float* logsumexp_out,
                                        const float* chunk_logsumexp,
                                        int BT, int n_chunks) {
    int row_idx = static_cast<int>(blockIdx.x);
    if (row_idx >= BT) {
        return;
    }

    float thread_maxval = -INFINITY;
    for (int i = threadIdx.x; i < n_chunks; i += blockDim.x) {
        float v = chunk_logsumexp[row_idx * n_chunks + i];
        thread_maxval = fmaxf(thread_maxval, v);
    }
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);

    float thread_sumval = 0.0f;
    for (int i = threadIdx.x; i < n_chunks; i += blockDim.x) {
        float v = chunk_logsumexp[row_idx * n_chunks + i];
        thread_sumval += expf(v - block_maxval);
    }
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    if (threadIdx.x == 0) {
        logsumexp_out[row_idx] = block_maxval + logf(block_sumval);
    }
}

__global__ void finalize_loss_kernel(float* losses, const float* logsumexp,
                                     const int* targets, int BT) {
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= BT) {
        return;
    }
    if (targets[idx] != -100) {
        losses[idx] += logsumexp[idx];
    }
}

template <class floatX>
__global__ void chunked_cross_entropy_backward_kernel(floatX* dlogits, const floatX* logits, const float* logsumexp,
                                                      const float* dloss, const int* targets,
                                                      int BT, int V, int P, int chunk_size) {
    // HuggingFace-style normalization: dloss is already scaled by 1/accumulated_valid_tokens
    // at the caller level (GraphExecutor/CompiledExecutor). No per-batch token scaling here.
    int row_idx = static_cast<int>(blockIdx.x);
    int block_idx = static_cast<int>(blockIdx.y);
    int start = block_idx * chunk_size;
    if (row_idx >= BT || start >= V) {
        return;
    }
    int end = (start + chunk_size < V) ? (start + chunk_size) : V;

    int ix = targets[row_idx];
    if (ix == -100) {
        for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
            dlogits[row_idx * P + i] = (floatX)0.0f;
        }
        return;
    }

    float lse = logsumexp ? logsumexp[row_idx] : 0.0f;
    float dloss_val = dloss ? dloss[row_idx] : 1.0f;
    const floatX* logits_vec = logits + static_cast<int64_t>(row_idx) * P;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float prob = expf((float)logits_vec[i] - lse);
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss_val;
        dlogits[row_idx * P + i] = (floatX)dlogit;
    }
}

template <class floatX>
__global__ void argmax_correct_kernel(const floatX* logits, const int* targets,
                                      int* correct_count, int BT, int V, int P) {
    int idx = static_cast<int>(blockIdx.x);
    if (idx >= BT) {
        return;
    }
    int ix = targets[idx];
    if (ix == -100 || correct_count == nullptr) {
        return;
    }

    __shared__ int shared_max_idx[32];
    __shared__ float shared_max_val[32];

    const floatX* logits_vec = logits + static_cast<int64_t>(idx) * P;
    float thread_max_val = -INFINITY;
    int thread_max_idx = 0;

    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float val = (float)logits_vec[i];
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_max_idx = i;
        }
    }

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
        if (other_val > thread_max_val) {
            thread_max_val = other_val;
            thread_max_idx = other_idx;
        }
    }

    if (lane_id == 0) {
        shared_max_val[warp_id] = thread_max_val;
        shared_max_idx[warp_id] = thread_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        thread_max_val = (lane_id < num_warps) ? shared_max_val[lane_id] : -INFINITY;
        thread_max_idx = (lane_id < num_warps) ? shared_max_idx[lane_id] : 0;

        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
            if (other_val > thread_max_val) {
                thread_max_val = other_val;
                thread_max_idx = other_idx;
            }
        }

        if (threadIdx.x == 0 && thread_max_idx == ix) {
            atomicAdd(correct_count, 1);
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel launchers

void fused_cross_entropy_forward(float* logits, float* losses, float* logsumexp,
                                 const int* targets, int* valid_token_count,
                                 int* correct_count,
                                 int BT, int V, int P, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = BT;
    cross_entropy_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        logits, losses, logsumexp, targets, valid_token_count, correct_count, BT, V, P);
    CUDA_CHECK(cudaGetLastError());
}

void fused_cross_entropy_forward(nv_bfloat16* logits, float* losses, float* logsumexp,
                                 const int* targets, int* valid_token_count,
                                 int* correct_count,
                                 int BT, int V, int P, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = BT;
    cross_entropy_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        logits, losses, logsumexp, targets, valid_token_count, correct_count, BT, V, P);
    CUDA_CHECK(cudaGetLastError());
}

void fused_cross_entropy_backward(float* dlogits, const float* logits, const float* logsumexp,
                                  const float* dloss, const int* targets,
                                  int BT, int V, int P, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = BT;
    cross_entropy_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        dlogits, logits, logsumexp, dloss, targets, BT, V, P);
    CUDA_CHECK(cudaGetLastError());
}

void fused_cross_entropy_backward(nv_bfloat16* dlogits, const nv_bfloat16* logits, const float* logsumexp,
                                  const float* dloss, const int* targets,
                                  int BT, int V, int P, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = BT;
    cross_entropy_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        dlogits, logits, logsumexp, dloss, targets, BT, V, P);
    CUDA_CHECK(cudaGetLastError());
}

void chunked_cross_entropy_forward(float* logits, float* losses, float* logsumexp,
                                   float* chunk_logsumexp, const int* targets,
                                   int* valid_token_count, int* correct_count,
                                   int BT, int V, int P, int n_chunks, cudaStream_t stream) {
    const int block_size = 256;
    dim3 grid(BT, n_chunks);
    chunked_cross_entropy_forward_kernel<<<grid, block_size, 0, stream>>>(
        logits, losses, chunk_logsumexp, targets, valid_token_count, BT, V, P, n_chunks, CROSS_ENTROPY_MAX_FUSED_SIZE);
    CUDA_CHECK(cudaGetLastError());

    logsumexp_reduce_kernel<<<BT, block_size, 0, stream>>>(logsumexp, chunk_logsumexp, BT, n_chunks);
    CUDA_CHECK(cudaGetLastError());

    const int threads = 256;
    const int blocks = (BT + threads - 1) / threads;
    finalize_loss_kernel<<<blocks, threads, 0, stream>>>(losses, logsumexp, targets, BT);
    CUDA_CHECK(cudaGetLastError());

    if (correct_count) {
        argmax_correct_kernel<<<BT, block_size, 0, stream>>>(
            logits, targets, correct_count, BT, V, P);
        CUDA_CHECK(cudaGetLastError());
    }
}

void chunked_cross_entropy_forward(nv_bfloat16* logits, float* losses, float* logsumexp,
                                   float* chunk_logsumexp, const int* targets,
                                   int* valid_token_count, int* correct_count,
                                   int BT, int V, int P, int n_chunks, cudaStream_t stream) {
    const int block_size = 256;
    dim3 grid(BT, n_chunks);
    chunked_cross_entropy_forward_kernel<<<grid, block_size, 0, stream>>>(
        logits, losses, chunk_logsumexp, targets, valid_token_count, BT, V, P, n_chunks, CROSS_ENTROPY_MAX_FUSED_SIZE);
    CUDA_CHECK(cudaGetLastError());

    logsumexp_reduce_kernel<<<BT, block_size, 0, stream>>>(logsumexp, chunk_logsumexp, BT, n_chunks);
    CUDA_CHECK(cudaGetLastError());

    const int threads = 256;
    const int blocks = (BT + threads - 1) / threads;
    finalize_loss_kernel<<<blocks, threads, 0, stream>>>(losses, logsumexp, targets, BT);
    CUDA_CHECK(cudaGetLastError());

    if (correct_count) {
        argmax_correct_kernel<<<BT, block_size, 0, stream>>>(
            logits, targets, correct_count, BT, V, P);
        CUDA_CHECK(cudaGetLastError());
    }
}

void chunked_cross_entropy_backward(float* dlogits, const float* logits, const float* logsumexp,
                                    const float* dloss, const int* targets,
                                    int BT, int V, int P, cudaStream_t stream) {
    const int block_size = 256;
    const int n_blocks = (V + CROSS_ENTROPY_BACKWARD_CHUNK_SIZE - 1) / CROSS_ENTROPY_BACKWARD_CHUNK_SIZE;
    dim3 grid(BT, n_blocks);
    chunked_cross_entropy_backward_kernel<<<grid, block_size, 0, stream>>>(
        dlogits, logits, logsumexp, dloss, targets,
        BT, V, P, CROSS_ENTROPY_BACKWARD_CHUNK_SIZE);
    CUDA_CHECK(cudaGetLastError());
}

void chunked_cross_entropy_backward(nv_bfloat16* dlogits, const nv_bfloat16* logits, const float* logsumexp,
                                    const float* dloss, const int* targets,
                                    int BT, int V, int P, cudaStream_t stream) {
    const int block_size = 256;
    const int n_blocks = (V + CROSS_ENTROPY_BACKWARD_CHUNK_SIZE - 1) / CROSS_ENTROPY_BACKWARD_CHUNK_SIZE;
    dim3 grid(BT, n_blocks);
    chunked_cross_entropy_backward_kernel<<<grid, block_size, 0, stream>>>(
        dlogits, logits, logsumexp, dloss, targets,
        BT, V, P, CROSS_ENTROPY_BACKWARD_CHUNK_SIZE);
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Per-token log-probability extraction

/**
 * @brief Kernel: compute log P(target | context) for each token position.
 *
 * For each row idx in [0, BT):
 *   if target == -100: logprobs[idx] = 0  (masked)
 *   else: logprobs[idx] = logit[target] - logsumexp(logits[idx, :])
 *
 * Uses prepare_softmax_blockwide3 for a numerically stable logsumexp.
 * Only thread 0 writes the output (no race).
 *
 * @tparam floatX  Data type (float or nv_bfloat16).
 * @param logits   Logits tensor of shape (BT, P), read-only.
 * @param logprobs Output per-token log-probabilities, shape (BT,), FP32.
 * @param targets  Target token indices, shape (BT,); -100 = masked.
 * @param BT       Batch * sequence length.
 * @param V        Actual vocabulary size (logits dimension).
 * @param P        Padded vocabulary size (stride).
 */
template <class floatX>
__global__ void extract_logprobs_kernel(const floatX* logits, float* logprobs,
                                         const int* targets, int BT, int V, int P) {
    int64_t idx = static_cast<int64_t>(blockIdx.x);
    if (idx >= static_cast<int64_t>(BT)) { return; }

    int ix = targets[idx];
    if (ix == -100) {
        if (threadIdx.x == 0) { logprobs[idx] = 0.0f; }
        return;
    }

    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);
    float lse = sp.Offset + logf(1.0f / sp.Scale);

    if (threadIdx.x == 0) {
        logprobs[idx] = (float)logits[idx * static_cast<int64_t>(P) + ix] - lse;
    }
}

void extract_logprobs(const float* logits, float* logprobs, const int* targets,
                      int BT, int V, int P, cudaStream_t stream) {
    const int block_size = 256;
    extract_logprobs_kernel<<<BT, block_size, 0, stream>>>(logits, logprobs, targets, BT, V, P);
    CUDA_CHECK(cudaGetLastError());
}

void extract_logprobs(const nv_bfloat16* logits, float* logprobs, const int* targets,
                      int BT, int V, int P, cudaStream_t stream) {
    const int block_size = 256;
    extract_logprobs_kernel<<<BT, block_size, 0, stream>>>(logits, logprobs, targets, BT, V, P);
    CUDA_CHECK(cudaGetLastError());
}
