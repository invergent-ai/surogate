// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

/**
 * @file encoder.cu
 * @brief Token and positional embedding kernels for transformer models.
 *
 * Implements the GPT-2 style encoder that combines token and positional embeddings.
 * - Forward pass: Adds token embeddings (wte) and positional embeddings (wpe)
 * - Backward pass: Computes gradients for token embeddings using deterministic bucketing
 *
 * The backward pass uses a bucketing strategy for deterministic gradient accumulation,
 * sorting token positions by vocabulary index on the GPU to enable parallel reduction without
 * race conditions.
 *
 * Based on llm.c https://github.com/karpathy/llm.c
 */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <cub/cub.cuh>

#include "utilities/utils.h"
#include "utilities/vec.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

/**
 * @brief CUDA kernel for encoder forward pass with positional embeddings.
 *
 * Combines token embeddings (wte) and positional embeddings (wpe) by addition.
 * Uses vectorized 128-bit loads/stores for efficient memory access.
 * Each thread processes x128::size elements (e.g., 8 for BF16).
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output tensor of shape (B, T, C).
 * @param[in] inp Input token indices of shape (B, T).
 * @param[in] wte Token embedding weights of shape (V, C).
 * @param[in] wpe Positional embedding weights of shape (T, C).
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension (hidden size).
 */
template <typename floatX>
__global__ void
encoder_forward_kernel3(floatX* out, const int* inp, const floatX* wte, const floatX* wpe, int B, int T, int C) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    long long idx = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    long long N = (long long)B * T * C;
    if (idx >= N) {
        return;
    }

    long long bt = idx / C;
    int b = (int)(bt / T);
    int t = (int)(bt % T);
    int c = (int)(idx % C);

    int ix = inp[b * T + t];

    floatX* out_btc = out + (long long)b * T * C + (long long)t * C + c;
    const floatX* wte_ix = wte + (long long)ix * C + c;
    const floatX* wpe_tc = wpe + (long long)t * C + c;

    x128 packed_out;
    x128 wte128 = load128cs(wte_ix);
    x128 wpe128 = load128cs(wpe_tc);
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
    }
    store128(out_btc, packed_out);
}

/**
 * @brief CUDA kernel for encoder forward pass without positional embeddings.
 *
 * Copies token embeddings directly to output without adding positional embeddings.
 * Used for models like LLaMA that use rotary position embeddings (RoPE) instead
 * of learned positional embeddings.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output tensor of shape (B, T, C).
 * @param[in] inp Input token indices of shape (B, T).
 * @param[in] wte Token embedding weights of shape (V, C).
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension (hidden size).
 * @param V Vocabulary size (for bounds checking).
 */
template <typename floatX>
__global__ void
encoder_forward_kernel3_nowpe(floatX* out, const int* inp, const floatX* wte, int B, int T, int C, int V) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    // Use 64-bit arithmetic for element addressing: vocab_size * C can exceed
    // INT32_MAX for large embeddings (e.g., Gemma4 PLI has V=262144, C=8960
    // → V*C ≈ 2.35e9 > 2^31). INT32 overflow would make wte + ix*C + c point
    // to garbage memory for high token ids, producing NaN output.
    long long idx = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    long long N = (long long)B * T * C;
    if (idx >= N) {
        return;
    }
    long long bt = idx / C;
    int b = (int)(bt / T);
    int t = (int)(bt % T);
    int c = (int)(idx % C);
    int ix = inp[b * T + t];
    // Guardrail against upstream tokenizer / dataloader bugs feeding
    // out-of-range token ids. An unchecked `ix` produces a wild load off
    // `wte + ix*C`, surfacing asynchronously as a cudaErrorIllegalAddress
    // in whichever kernel runs next — historically the crash appeared in
    // ops far from the actual cause. __trap stops the offending thread
    // cleanly; the per-thread printf tells us the bad id and its position
    // so the upstream bug is obvious. Release builds keep the check; the
    // cost is one integer comparison per vector lane.
    if (!(0 <= ix && ix < V)) {
        printf("[encoder_forward] out-of-range token id: ix=%d (vocab=%d) at b=%d t=%d\n", ix, V, b, t);
        __trap();
    }
    x128 wte128 = x128::load(wte + (long long)ix * C + c);
    wte128.store(out + (long long)b * T * C + (long long)t * C + c);
}

// ----------------------------------------------------------------------------
// kernel launchers

/**
 * @brief Template implementation for encoder forward pass.
 *
 * Dispatches to the appropriate kernel based on whether positional embeddings
 * are provided. Uses vectorized memory access with 256 threads per block.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output tensor of shape (B, T, C).
 * @param[in] inp Input token indices of shape (B, T).
 * @param[in] wte Token embedding weights of shape (V, C).
 * @param[in] wpe Positional embedding weights of shape (T, C), or nullptr for no positional encoding.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 * @param V Vocabulary size.
 * @param stream CUDA stream for asynchronous execution.
 */
template <class floatX>
__global__ void encoder_forward_scalar(floatX* out, const int* inp, const floatX* weight,
                                        long count, int columns, int vocab) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int token = inp[i / columns];
    if (token < 0 || token >= vocab) __trap();
    out[i] = weight[static_cast<long>(token) * columns + i % columns];
}

template <class floatX>
void encoder_forward_imp(floatX* out,
                         const int* inp,
                         const floatX* wte,
                         const floatX* wpe,
                         int B,
                         int T,
                         int C,
                         int V,
                         cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    constexpr int block_size = 256;
    const long long N = (long long)B * T * C;
    if (wpe == nullptr && (C % x128::size || reinterpret_cast<std::uintptr_t>(out) % 16 ||
                           reinterpret_cast<std::uintptr_t>(wte) % 16)) {
        encoder_forward_scalar<<<static_cast<unsigned>((N + 255) / 256), 256, 0, stream>>>(out, inp, wte, N, C, V);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const long long grid_ll = (N + (long long)block_size * x128::size - 1) / ((long long)block_size * x128::size);
    const int grid_size = (int)grid_ll;
    if (wpe == nullptr) {
        // Llama 3 does not use positional encoder
        encoder_forward_kernel3_nowpe<<<grid_size, block_size, 0, stream>>>(out, inp, wte, B, T, C, V);
    } else {
        // GPT-2 does, so we use the full encoder kernel
        // encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    }
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Encoder forward pass for FP32 tensors.
 *
 * @param[out] out Output embeddings of shape (B, T, C) in FP32.
 * @param[in] inp Input token indices of shape (B, T).
 * @param[in] wte Token embedding weights of shape (V, C) in FP32.
 * @param[in] wpe Positional embedding weights of shape (T, C) in FP32, or nullptr.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 * @param V Vocabulary size.
 * @param stream CUDA stream.
 */
void encoder_forward(float* out,
                     const int* inp,
                     const float* wte,
                     const float* wpe,
                     int B,
                     int T,
                     int C,
                     int V,
                     cudaStream_t stream) {
    encoder_forward_imp(out, inp, wte, wpe, B, T, C, V, stream);
}

/**
 * @brief Encoder forward pass for BF16 tensors.
 *
 * @param[out] out Output embeddings of shape (B, T, C) in BF16.
 * @param[in] inp Input token indices of shape (B, T).
 * @param[in] wte Token embedding weights of shape (V, C) in BF16.
 * @param[in] wpe Positional embedding weights of shape (T, C) in BF16, or nullptr.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 * @param V Vocabulary size.
 * @param stream CUDA stream.
 */
void encoder_forward(nv_bfloat16* out,
                     const int* inp,
                     const nv_bfloat16* wte,
                     const nv_bfloat16* wpe,
                     int B,
                     int T,
                     int C,
                     int V,
                     cudaStream_t stream) {
    encoder_forward_imp(out, inp, wte, wpe, B, T, C, V, stream);
}

/**
 * @brief Fills out[i] = i.
 */
__global__ void encoder_backward_iota_kernel(int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}

/**
 * @brief CUDA kernel for deterministic token embedding gradient computation.
 *
 * Computes gradients for token embeddings (dwte) using a bucketing strategy for
 * determinism. Token positions are grouped into buckets by vocabulary index, allowing
 * parallel reduction without race conditions or non-deterministic atomics.
 *
 * Algorithm:
 * - The positions arrive sorted by token, in order within each token (a stable sort)
 * - The block at a token's first sorted position handles its bucket; the others exit
 * - Each block handles (WARP_SIZE * x128::size) channels (blockIdx.y picks the group)
 * - Each thread handles x128::size channels (e.g., 8 for BF16)
 * - Each block processes (BLOCK_SIZE / WARP_SIZE) bucket elements in parallel
 * - Uses shared memory for intra-block reduction, then read-modify-write to dwte
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @tparam BLOCK_SIZE Number of threads per block (default 256).
 * @param[in,out] dwte Token embedding gradients of shape (V, C), accumulated in-place.
 * @param[in] sorted_tokens The (B*T) token ids in ascending order.
 * @param[in] sorted_positions The (B*T) positions of those tokens.
 * @param[in] dout Upstream gradients of shape (B, T, C).
 * @param seed Random seed for stochastic rounding (currently disabled).
 * @param N Number of positions (B*T).
 * @param C Embedding dimension.
 */
template <typename floatX, int BLOCK_SIZE = 256>
__global__ void wte_backward_kernel(floatX* dwte,
                                    const unsigned int* sorted_tokens,
                                    const int* sorted_positions,
                                    const floatX* dout,
                                    unsigned int seed,
                                    int N,
                                    int C) {
    // Each bucket corresponds to (WARP_SIZE * x128::size) channels for a single vocabulary token
    // Each thread handles x128::size channels, e.g. 256 per warp for BF16
    // Each block handles (BLOCK_SIZE / WARP_SIZE) elements in a single bucket in parallel
    // If a bucket has less than 8 elements, some warps will return immediately
    // If a bucket has more than 8 elements, we will loop over all of them
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    const int bucket_start_idx = blockIdx.x;
    const unsigned int bucket_ix = sorted_tokens[bucket_start_idx];
    // Only the block at the token's first position works on it
    if (bucket_start_idx > 0 && sorted_tokens[bucket_start_idx - 1] == bucket_ix) {
        return;
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int c_per_warp = 32 * x128::size;
    int c = blockIdx.y * c_per_warp + (lane_id * x128::size);

    // Each thread handles "x128::size" channels, so at fp8, each warp would handle 512 channels
    // If C is not a multiple of this (e.g. 768), some buckets/c_groups cannot use the entire warp
    if (c >= C) {
        return;
    }
    // The bucket ends at the first sorted position that holds a larger token
    int bucket_end_idx = bucket_start_idx + 1;
    for (int hi = N; bucket_end_idx < hi;) {
        const int mid = bucket_end_idx + (hi - bucket_end_idx) / 2;
        if (sorted_tokens[mid] == bucket_ix) {
            bucket_end_idx = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int bucket_size = bucket_end_idx - bucket_start_idx;
    // Exit early if this is a small bucket and this warp doesn't have any items to process
    if (warp_id >= bucket_size) {
        return;
    }

    float accum[x128::size] = {0.0f};
    __shared__ float accum_shared[x128::size * BLOCK_SIZE];

    for (int item = warp_id; item < bucket_size; item += BLOCK_SIZE / 32) {
        const long bt = sorted_positions[bucket_start_idx + item];

        const floatX* dout_btc = dout + bt * C + c;
        x128 packed_inp1 = x128::load_cs(dout_btc);
        for (int k = 0; k < packed_inp1.size; k++) {
            accum[k] += (float)packed_inp1[k];
        }
    }

    if (warp_id != 0) {
        // we accumulate into warp 0, so only the other warps need to write to shared memory
        for (int k = 0; k < x128::size; k++) {
            accum_shared[threadIdx.x + k * BLOCK_SIZE] = accum[k];
        }
        return;  // only warp 0 is needed after writing to shared memory
    }

    // Read dwte for warp 0 even if other warps are not finished yet to maximise latency tolerance
    floatX* dwte_ix = dwte + static_cast<long>(bucket_ix) * C + c;
    x128 packed_in_out = x128::load(dwte_ix);

    // note: threads which have returned are considered synchronised by CUDA so no risk of deadlock
    __syncthreads();

    // Accumulate into warp 0's registers by reading the values of the other warps in shared memory
    for (int i = threadIdx.x + 32; i < min(BLOCK_SIZE, bucket_size * 32); i += 32) {
        for (int k = 0; k < x128::size; k++) {
            accum[k] += accum_shared[i + k * BLOCK_SIZE];
        }
    }

    // Add the result to dwte and write back to global memory (read-modify-write)
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        // TODO  re-enable  this
        // stochastic_rounding(accum[k] + (float)packed_in_out[k], &packed_in_out[k], seed + bucket * 32 + threadIdx.x + k);
        packed_in_out[k] = accum[k] + (float)packed_in_out[k];
    }
    packed_in_out.store(dwte_ix);
}

namespace {

constexpr std::size_t kEncoderScratchAlignment = 256;

std::size_t encoder_scratch_align(std::size_t bytes) {
    return div_ceil(bytes, kEncoderScratchAlignment) * kEncoderScratchAlignment;
}

/// Where encoder_backward keeps its arrays in the scratch buffer: the radix sort's temporary
/// storage, the sorted tokens, the positions 0..N-1 and the positions in token order.
struct EncoderBackwardScratch {
    std::size_t sort_bytes = 0;
    std::size_t tokens = 0;
    std::size_t positions_in = 0;
    std::size_t positions = 0;
    std::size_t total = 0;
};

EncoderBackwardScratch encoder_backward_scratch_layout(int n) {
    EncoderBackwardScratch layout;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,
                                               layout.sort_bytes,
                                               static_cast<const unsigned int*>(nullptr),
                                               static_cast<unsigned int*>(nullptr),
                                               static_cast<const int*>(nullptr),
                                               static_cast<int*>(nullptr),
                                               n,
                                               0,
                                               32,
                                               cudaStream_t{}));
    const std::size_t array_bytes = encoder_scratch_align(static_cast<std::size_t>(n) * sizeof(int));
    layout.tokens = encoder_scratch_align(layout.sort_bytes);
    layout.positions_in = layout.tokens + array_bytes;
    layout.positions = layout.positions_in + array_bytes;
    layout.total = layout.positions + array_bytes;
    return layout;
}

}  // namespace

std::size_t encoder_backward_scratch_bytes(long tokens) {
    return tokens > 0 ? encoder_backward_scratch_layout(static_cast<int>(tokens)).total : 0;
}

/**
 * @brief Template implementation for deterministic encoder backward pass.
 *
 * Computes token embedding gradients in two GPU steps:
 * 1. Sort the token positions by token (a stable radix sort, so in order within a token)
 * 2. Sum each token's positions in one thread block per (token, channel group)
 *
 * This avoids non-deterministic atomics by ensuring each output location is written by exactly one
 * thread block. Nothing runs on the host, so a CUDA graph that captured the call reads the tokens
 * in @p inp when it replays.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[in,out] dwte Token embedding gradients of shape (V, C), accumulated in-place.
 * @param scratch GPU scratch buffer of at least encoder_backward_scratch_bytes(B * T) bytes, 256-byte aligned.
 * @param scratch_bytes Size of @p scratch.
 * @param[in] dout Upstream gradients of shape (B, T, C).
 * @param[in] inp Input token indices on GPU of shape (B, T).
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 * @param seed Random seed for stochastic rounding.
 * @param stream CUDA stream.
 */
template <class floatX>
void encoder_backward_imp(floatX* dwte,
                          std::byte* scratch,
                          std::size_t scratch_bytes,
                          const floatX* dout,
                          const int* inp,
                          int B,
                          int T,
                          int C,
                          unsigned int seed,
                          cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    const int n = B * T;
    if (n <= 0) {
        return;
    }
    const EncoderBackwardScratch layout = encoder_backward_scratch_layout(n);
    if (scratch_bytes < layout.total) {
        throw std::runtime_error("encoder_backward: scratch holds " + std::to_string(scratch_bytes) + " bytes, " +
                                 std::to_string(layout.total) + " needed for " + std::to_string(n) + " tokens");
    }
    auto* sorted_tokens = reinterpret_cast<unsigned int*>(scratch + layout.tokens);
    auto* positions_in = reinterpret_cast<int*>(scratch + layout.positions_in);
    auto* sorted_positions = reinterpret_cast<int*>(scratch + layout.positions);

    // Step 1: Sort the positions by token. Token ids are non-negative, so they sort as unsigned.
    encoder_backward_iota_kernel<<<div_ceil(n, 256), 256, 0, stream>>>(positions_in, n);
    CUDA_CHECK(cudaGetLastError());
    std::size_t sort_bytes = layout.sort_bytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch,
                                               sort_bytes,
                                               reinterpret_cast<const unsigned int*>(inp),
                                               sorted_tokens,
                                               positions_in,
                                               sorted_positions,
                                               n,
                                               0,
                                               32,
                                               stream));

    // Step 2: one block per sorted position and channel group
    const int num_c_groups = div_ceil(C, static_cast<int>(x128::size * 32));
    wte_backward_kernel<floatX, 256>
        <<<dim3(n, num_c_groups), 256, 0, stream>>>(dwte, sorted_tokens, sorted_positions, dout, seed, n, C);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Encoder backward pass for FP32 tensors.
 *
 * Computes deterministic token embedding gradients.
 *
 * @param[in,out] dwte Token embedding gradients of shape (V, C) in FP32.
 * @param scratch GPU scratch buffer of at least encoder_backward_scratch_bytes(B * T) bytes.
 * @param scratch_bytes Size of @p scratch.
 * @param[in] dout Upstream gradients of shape (B, T, C) in FP32.
 * @param[in] inp Input token indices on GPU.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 * @param seed Random seed for stochastic rounding.
 * @param stream CUDA stream.
 */
void encoder_backward(float* dwte,
                      std::byte* scratch,
                      std::size_t scratch_bytes,
                      const float* dout,
                      const int* inp,
                      int B,
                      int T,
                      int C,
                      unsigned int seed,
                      cudaStream_t stream) {
    encoder_backward_imp(dwte, scratch, scratch_bytes, dout, inp, B, T, C, seed, stream);
}

/**
 * @brief Encoder backward pass for BF16 tensors.
 *
 * Computes deterministic token embedding gradients.
 *
 * @param[in,out] dwte Token embedding gradients of shape (V, C) in BF16.
 * @param scratch GPU scratch buffer of at least encoder_backward_scratch_bytes(B * T) bytes.
 * @param scratch_bytes Size of @p scratch.
 * @param[in] dout Upstream gradients of shape (B, T, C) in BF16.
 * @param[in] inp Input token indices on GPU.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 * @param seed Random seed for stochastic rounding.
 * @param stream CUDA stream.
 */
void encoder_backward(nv_bfloat16* dwte,
                      std::byte* scratch,
                      std::size_t scratch_bytes,
                      const nv_bfloat16* dout,
                      const int* inp,
                      int B,
                      int T,
                      int C,
                      unsigned int seed,
                      cudaStream_t stream) {
    encoder_backward_imp(dwte, scratch, scratch_bytes, dout, inp, B, T, C, seed, stream);
}

template <typename InT>
__global__ void embedding_backward_atomic_kernel(float* dwte, const InT* dout, const int* inp, int B, int T, int C) {
    const long total = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
    const long stride = static_cast<long>(blockDim.x) * static_cast<long>(gridDim.x);
    for (long idx = static_cast<long>(blockIdx.x) * static_cast<long>(blockDim.x) + static_cast<long>(threadIdx.x);
         idx < total;
         idx += stride) {
        const int c = static_cast<int>(idx % C);
        const long bt = idx / C;
        const int token = inp[bt];
        atomicAdd(&dwte[static_cast<long>(token) * C + c], static_cast<float>(dout[idx]));
    }
}

void encoder_backward_atomic(float* dwte,
                             const nv_bfloat16* dout,
                             const int* inp,
                             int B,
                             int T,
                             int C,
                             cudaStream_t stream) {
    const long total = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
    if (total <= 0) return;
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<long>(65535, (total + threads - 1) / threads));
    embedding_backward_atomic_kernel<<<blocks, threads, 0, stream>>>(dwte, dout, inp, B, T, C);
    CUDA_CHECK(cudaGetLastError());
}

void encoder_backward_atomic(float* dwte, const half* dout, const int* inp, int B, int T, int C, cudaStream_t stream) {
    const long total = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
    if (total <= 0) return;
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<long>(65535, (total + threads - 1) / threads));
    embedding_backward_atomic_kernel<<<blocks, threads, 0, stream>>>(dwte, dout, inp, B, T, C);
    CUDA_CHECK(cudaGetLastError());
}
