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
 * sorting tokens by vocabulary index to enable parallel reduction without race conditions.
 *
 * Based on llm.c https://github.com/karpathy/llm.c
 */
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <vector>

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
template<typename floatX>
__global__ void encoder_forward_kernel3(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx >= N) { return; }

    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;

    int ix = inp[b * T + t];

    floatX* out_btc = out + b * T * C + t * C + c;
    const floatX* wte_ix = wte + ix * C + c;
    const floatX* wpe_tc = wpe + t * C + c;

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
template<typename floatX>
__global__ void encoder_forward_kernel3_nowpe(floatX* out,
                               const int* inp, const floatX* wte,
                               int B, int T, int C, int V) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx >= N) { return; }
    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;
    int ix = inp[b * T + t];
    assert(0 <= ix && ix < V);
    x128 wte128 = x128::load(wte + ix * C + c);
    wte128.store(out + b * T * C + t * C + c);
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
template<class floatX>
void encoder_forward_imp(floatX* out,
                         const int* inp, const floatX* wte, const floatX* wpe,
                         int B, int T, int C, int V, cudaStream_t stream) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    constexpr int block_size = 256;
    const int N = B * T * C;
    const int grid_size = div_ceil(N, (int)(block_size * x128::size));
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
void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C, int V, cudaStream_t stream) {
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
void encoder_forward(nv_bfloat16* out, const int* inp, const nv_bfloat16* wte, const nv_bfloat16* wpe, int B, int T, int C, int V, cudaStream_t stream) {
    encoder_forward_imp(out, inp, wte, wpe, B, T, C, V, stream);
}


/**
 * @brief CUDA kernel for deterministic token embedding gradient computation.
 *
 * Computes gradients for token embeddings (dwte) using a bucketing strategy for
 * determinism. Tokens are grouped into buckets by vocabulary index, allowing
 * parallel reduction without race conditions or non-deterministic atomics.
 *
 * Algorithm:
 * - Each bucket corresponds to (WARP_SIZE * x128::size) channels for a single vocab token
 * - Each thread handles x128::size channels (e.g., 8 for BF16)
 * - Each block processes (BLOCK_SIZE / WARP_SIZE) bucket elements in parallel
 * - Buckets are sorted by size (largest first) on CPU for load balancing
 * - Uses shared memory for intra-block reduction, then read-modify-write to dwte
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @tparam BLOCK_SIZE Number of threads per block (default 256).
 * @param[in,out] dwte Token embedding gradients of shape (V, C), accumulated in-place.
 * @param[in] bucket_info Bucket metadata: (start_idx, size, vocab_idx, channel_group).
 * @param[in] workload_indices Flattened list of (batch*T) indices per bucket.
 * @param[in] dout Upstream gradients of shape (B, T, C).
 * @param[in] inp Input token indices of shape (B, T).
 * @param seed Random seed for stochastic rounding (currently disabled).
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 */
template <typename floatX, int BLOCK_SIZE=256>
__global__ void wte_backward_kernel(floatX* dwte,
                                    const int4* bucket_info, const int* workload_indices, const floatX* dout, const int* inp,
                                    unsigned int seed, int B, int T, int C) {
    // In order to be deterministic, we preprocess the inputs on the cpu into "buckets"
    // Each bucket corresponds to (WARP_SIZE * x128::size) channels for a single vocabulary token
    // Each thread handles x128::size channels, e.g. 256 per warp for BF16
    // Each block handles (BLOCK_SIZE / WARP_SIZE) elements in a single bucket in parallel
    // If a bucket has less than 8 elements, some warps will return immediately
    // If a bucket has more than 8 elements, we will loop over all of them
    // The buckets are sorted on the CPU so the largest buckets start 1st
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    int bucket = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int c_per_warp = 32 * x128::size;

    int bucket_start_idx = bucket_info[bucket].x;
    int bucket_size = bucket_info[bucket].y;
    int bucket_ix = bucket_info[bucket].z;
    int c = bucket_info[bucket].w * c_per_warp + (lane_id * x128::size);

    // Each thread handles "x128::size" channels, so at fp8, each warp would handle 512 channels
    // If C is not a multiple of this (e.g. 768), some buckets/c_groups cannot use the entire warp
    if (c >= C) { return; }
    // Exit early if this is a small bucket and this warp doesn't have any items to process
    if (warp_id >= bucket_size) { return; }

    float accum[x128::size] = {0.0f};
    __shared__ float accum_shared[x128::size * BLOCK_SIZE];

    for(int item = warp_id; item < bucket_size; item += BLOCK_SIZE/32) {
        int bt = workload_indices[bucket_start_idx + item];

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
        return; // only warp 0 is needed after writing to shared memory
    }

    // Read dwte for warp 0 even if other warps are not finished yet to maximise latency tolerance
    floatX* dwte_ix = dwte + bucket_ix * C + c;
    x128 packed_in_out = x128::load(dwte_ix);

    // note: threads which have returned are considered synchronised by CUDA so no risk of deadlock
    __syncthreads();

    // Accumulate into warp 0's registers by reading the values of the other warps in shared memory
    for (int i = threadIdx.x+32; i < min(BLOCK_SIZE, bucket_size*32); i += 32) {
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

/**
 * @brief Template implementation for deterministic encoder backward pass.
 *
 * Computes token embedding gradients using a three-step bucketing algorithm:
 * 1. CPU: Sort input tokens into buckets by (vocab_index, channel_group)
 * 2. CPU: Sort buckets by size (largest first) for GPU load balancing
 * 3. GPU: Process buckets in parallel with deterministic reduction
 *
 * This approach avoids non-deterministic atomics by ensuring each output
 * location is written by exactly one thread block.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[in,out] dwte Token embedding gradients of shape (V, C), accumulated in-place.
 * @param scratch GPU scratch buffer for bucket info and workload indices.
 * @param workload_indices CPU buffer for flattened bucket indices.
 * @param bucket_info CPU buffer for bucket metadata.
 * @param[in] dout Upstream gradients of shape (B, T, C).
 * @param[in] inp Input token indices on GPU of shape (B, T).
 * @param[in] inputs_cpu Input token indices on CPU of shape (B, T).
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 * @param seed Random seed for stochastic rounding.
 * @param stream Main CUDA stream for kernel execution.
 * @param sync_event Event for synchronizing with copy stream.
 * @param copy_stream Separate stream for async host-to-device copies.
 */
template<class floatX>
void encoder_backward_imp(floatX* dwte, int* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const floatX* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    // CUDA graph capture: avoid cross-stream sync by copying on the main stream.
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool capturing = (cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess &&
                            capture_status != cudaStreamCaptureStatusNone);
    if (capturing) {
        copy_stream = stream;
        sync_event = nullptr;
    }

    int num_c_groups = div_ceil((size_t)C, x128::size * 32);
    assert(B*T*num_c_groups * (sizeof(int4)+sizeof(int)) <= B*T*3*C * sizeof(floatX));

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // todo - passing c_group/inputs_cpu[bt] in data to avoid a second hash lookup is a bit hacky
            uint64_t data = bt + (c_group<<32ULL) + ((uint64_t)inputs_cpu[bt]<<42ULL);
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    // Step 2: Sort buckets by size in descending order
    // this is so the largest buckets are processed first by the GPU
    // otherwise, if they started late, they would still be running with the rest of the GPU idle
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(), // ugly because we don't have a typedef for the std::pair
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int num_buckets = buckets.size();
    int bucket_index = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index].x = workload_index; // bucket start
        bucket_info[bucket_index].y = bucket.second.size(); // bucket size
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL<<20ULL)-1); // bucket ix
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL<<10ULL)-1); // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL<<31ULL)-1ULL));
        }
        bucket_index++;
    }

    // Step 3: Copy data from host to device (async on a different stream unless capturing)
    int4* d_bucket_info = (int4*)scratch;
    int*  d_workload_indices = (int*)(scratch + B*T*num_c_groups * 4);
    CUDA_CHECK(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, copy_stream));
    if (sync_event) {
        CUDA_CHECK(cudaEventRecord(sync_event, copy_stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, sync_event, 0));
    }

    // Launch wte kernel
    // todo - profile block sizes on more content (depends on number of buckets and on GPU?)
    wte_backward_kernel<floatX, 256><<<num_buckets, 256, 0, stream>>>(dwte, d_bucket_info, d_workload_indices, dout, inp, seed, B, T, C);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Encoder backward pass for FP32 tensors.
 *
 * Computes deterministic token embedding gradients.
 *
 * @param[in,out] dwte Token embedding gradients of shape (V, C) in FP32.
 * @param scratch GPU scratch buffer.
 * @param workload_indices CPU scratch buffer for bucket workloads.
 * @param bucket_info CPU scratch buffer for bucket metadata.
 * @param[in] dout Upstream gradients of shape (B, T, C) in FP32.
 * @param[in] inp Input token indices on GPU.
 * @param[in] inputs_cpu Input token indices on CPU.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 * @param seed Random seed for stochastic rounding.
 * @param stream Main CUDA stream.
 * @param sync_event Synchronization event.
 * @param copy_stream Stream for async copies.
 */
void encoder_backward(float* dwte, int* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const float* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream) {
    encoder_backward_imp(dwte, scratch, workload_indices, bucket_info, dout, inp, inputs_cpu, B, T, C, seed, stream, sync_event, copy_stream);
}

/**
 * @brief Encoder backward pass for BF16 tensors.
 *
 * Computes deterministic token embedding gradients.
 *
 * @param[in,out] dwte Token embedding gradients of shape (V, C) in BF16.
 * @param scratch GPU scratch buffer.
 * @param workload_indices CPU scratch buffer for bucket workloads.
 * @param bucket_info CPU scratch buffer for bucket metadata.
 * @param[in] dout Upstream gradients of shape (B, T, C) in BF16.
 * @param[in] inp Input token indices on GPU.
 * @param[in] inputs_cpu Input token indices on CPU.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Embedding dimension.
 * @param seed Random seed for stochastic rounding.
 * @param stream Main CUDA stream.
 * @param sync_event Synchronization event.
 * @param copy_stream Stream for async copies.
 */
void encoder_backward(nv_bfloat16* dwte, int* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const nv_bfloat16* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream) {
    encoder_backward_imp(dwte, scratch, workload_indices, bucket_info, dout, inp, inputs_cpu, B, T, C, seed, stream, sync_event, copy_stream);
}
