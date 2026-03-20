// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Thin C++ wrappers around FlashInfer sampling templates.
// Instantiates templates for float + int32_t (our generation dtypes).

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <cfloat>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

// FlashInfer sampling headers (fetched via CMake FetchContent)
#include <flashinfer/sampling.cuh>

#include "kernels/sampling.h"
#include "utilities/utils.h"

namespace {

bool softmax_enable_pdl() {
    static int cached = -1;
    if (cached >= 0) {
        return cached != 0;
    }
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        cached = 0;
        return false;
    }
    // Match mini-sgl behavior: enable PDL on Hopper+.
    cached = (prop.major >= 9) ? 1 : 0;
    return cached != 0;
}

}  // namespace

// ============================================================================
// Softmax: logits → probs with optional temperature
// ============================================================================

void sampling_softmax(
        float* logits,
        float* probs,
        const float* temperature,
        int batch_size,
        int vocab_size,
        void* workspace,
        size_t workspace_bytes,
        cudaStream_t stream) {

    auto err = flashinfer::sampling::OnlineSoftmax<float>(
        logits,                             // in-place logits
        probs,                              // output probs
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(vocab_size),
        const_cast<float*>(temperature),    // per-batch temperatures (nullptr = use scalar)
        1.0f,                               // scalar temperature when array is null
        workspace,
        workspace_bytes,
        /*enable_pdl=*/softmax_enable_pdl(),
        stream);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_softmax failed: ") + cudaGetErrorString(err));
    }
}

namespace {

__global__ void sanitize_logits_kernel(
        float* __restrict__ logits,
        int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const float x = logits[idx];
    if (isnan(x)) {
        logits[idx] = 0.0f;
    } else if (isinf(x)) {
        logits[idx] = x > 0.0f ? FLT_MAX : -FLT_MAX;
    }
}

__global__ void sanitize_token_ids_kernel(
        int32_t* __restrict__ token_ids,
        int batch_size,
        int vocab_size,
        int32_t fallback_id,
        int32_t* invalid_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const int32_t tok = token_ids[idx];
    if (tok < 0 || tok >= vocab_size) {
        token_ids[idx] = fallback_id;
        if (invalid_count) {
            atomicAdd(invalid_count, 1);
        }
    }
}

}  // namespace

void sampling_sanitize_logits(
        float* logits,
        int batch_size,
        int vocab_size,
        cudaStream_t stream) {
    if (batch_size <= 0 || vocab_size <= 0) return;
    const int total = batch_size * vocab_size;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    sanitize_logits_kernel<<<blocks, threads, 0, stream>>>(logits, total);
    CUDA_CHECK(cudaGetLastError());
}

void sampling_sanitize_token_ids(
        int32_t* token_ids,
        int batch_size,
        int vocab_size,
        int32_t fallback_id,
        int32_t* invalid_count,
        cudaStream_t stream) {
    if (batch_size <= 0 || vocab_size <= 0) return;
    if (fallback_id < 0 || fallback_id >= vocab_size) {
        fallback_id = 0;
    }
    constexpr int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    sanitize_token_ids_kernel<<<blocks, threads, 0, stream>>>(
        token_ids, batch_size, vocab_size, fallback_id, invalid_count);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Categorical sampling from probability distribution
// ============================================================================

void sampling_from_probs(
        float* probs,
        int32_t* output,
        int batch_size,
        int vocab_size,
        bool deterministic,
        uint64_t seed,
        uint64_t offset,
        cudaStream_t stream) {

    int32_t* indices_null = nullptr;
    uint64_t* seed_arr_null = nullptr;
    uint64_t* offset_arr_null = nullptr;
    auto err = flashinfer::sampling::SamplingFromProb<float, int32_t>(
        probs,
        output,
        indices_null,
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(vocab_size),
        deterministic,
        seed_arr_null,
        seed,
        offset_arr_null,
        offset,
        stream);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_from_probs failed: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Gumbel-Max sampling directly from logits (no explicit softmax)
// ============================================================================

void sampling_from_logits(
        float* logits,
        int32_t* output,
        int batch_size,
        int vocab_size,
        bool deterministic,
        uint64_t seed,
        uint64_t offset,
        cudaStream_t stream) {

    int32_t* indices_null = nullptr;
    uint64_t* seed_arr_null = nullptr;
    uint64_t* offset_arr_null = nullptr;
    auto err = flashinfer::sampling::SamplingFromLogits<float, int32_t>(
        logits,
        output,
        indices_null,
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(vocab_size),
        deterministic,
        seed_arr_null,
        seed,
        offset_arr_null,
        offset,
        stream);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_from_logits failed: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Top-K sampling
// ============================================================================

void sampling_top_k(
        float* probs,
        int32_t* output,
        int top_k,
        int batch_size,
        int vocab_size,
        bool deterministic,
        uint64_t seed,
        uint64_t offset,
        cudaStream_t stream) {

    int32_t* indices_null = nullptr;
    float* top_k_arr_null = nullptr;
    uint64_t* seed_arr_null = nullptr;
    uint64_t* offset_arr_null = nullptr;
    auto err = flashinfer::sampling::TopKSamplingFromProb<float, int32_t>(
        probs, output, indices_null, top_k_arr_null,
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(top_k),
        static_cast<uint32_t>(vocab_size),
        deterministic,
        seed_arr_null, seed, offset_arr_null, offset,
        stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_top_k failed: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Top-P (nucleus) sampling
// ============================================================================

void sampling_top_p(
        float* probs,
        int32_t* output,
        float top_p,
        int batch_size,
        int vocab_size,
        bool deterministic,
        uint64_t seed,
        uint64_t offset,
        cudaStream_t stream) {

    int32_t* indices_null = nullptr;
    float* top_p_arr_null = nullptr;
    uint64_t* seed_arr_null = nullptr;
    uint64_t* offset_arr_null = nullptr;
    auto err = flashinfer::sampling::TopPSamplingFromProb<float, int32_t>(
        probs, output, indices_null, top_p_arr_null,
        static_cast<uint32_t>(batch_size),
        top_p,
        static_cast<uint32_t>(vocab_size),
        deterministic,
        seed_arr_null, seed, offset_arr_null, offset,
        stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_top_p failed: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Combined Top-K + Top-P sampling
// ============================================================================

void sampling_top_k_top_p(
        float* probs,
        int32_t* output,
        int top_k,
        float top_p,
        int batch_size,
        int vocab_size,
        bool deterministic,
        uint64_t seed,
        uint64_t offset,
        cudaStream_t stream) {

    int32_t* indices_null = nullptr;
    int32_t* top_k_arr_null = nullptr;
    float* top_p_arr_null = nullptr;
    uint64_t* seed_arr_null = nullptr;
    uint64_t* offset_arr_null = nullptr;
    auto err = flashinfer::sampling::TopKTopPSamplingFromProb<float, int32_t>(
        probs, top_k_arr_null, top_p_arr_null, output, indices_null,
        static_cast<uint32_t>(batch_size),
        static_cast<int32_t>(top_k),
        top_p,
        static_cast<uint32_t>(vocab_size),
        deterministic,
        seed_arr_null, seed, offset_arr_null, offset,
        stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_top_k_top_p failed: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Min-P sampling
// ============================================================================

void sampling_min_p(
        float* probs,
        int32_t* output,
        float min_p,
        int batch_size,
        int vocab_size,
        bool deterministic,
        uint64_t seed,
        uint64_t offset,
        cudaStream_t stream) {

    int32_t* indices_null = nullptr;
    float* min_p_arr_null = nullptr;
    uint64_t* seed_arr_null = nullptr;
    uint64_t* offset_arr_null = nullptr;
    auto err = flashinfer::sampling::MinPSamplingFromProb<float, int32_t>(
        probs, min_p_arr_null, output, indices_null,
        static_cast<uint32_t>(batch_size),
        min_p,
        static_cast<uint32_t>(vocab_size),
        deterministic,
        seed_arr_null, seed, offset_arr_null, offset,
        stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_min_p failed: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Per-sequence Top-K sampling
// ============================================================================

void sampling_top_k_per_seq(
        float* probs,
        int32_t* output,
        const int32_t* top_k_arr,
        int batch_size,
        int vocab_size,
        bool deterministic,
        uint64_t seed,
        uint64_t offset,
        cudaStream_t stream) {

    int32_t* indices_null = nullptr;
    uint64_t* seed_arr_null = nullptr;
    uint64_t* offset_arr_null = nullptr;
    // When top_k_arr is provided, FlashInfer reads per-row K values from it.
    // We cast away const — FlashInfer doesn't modify it but the API isn't const-correct.
    auto err = flashinfer::sampling::TopKSamplingFromProb<float, int32_t>(
        probs, output, indices_null,
        const_cast<float*>(reinterpret_cast<const float*>(top_k_arr)),  // top_k_arr as float* (API quirk)
        static_cast<uint32_t>(batch_size),
        0,  // top_k_val unused when array is provided
        static_cast<uint32_t>(vocab_size),
        deterministic,
        seed_arr_null, seed, offset_arr_null, offset,
        stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_top_k_per_seq failed: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Per-sequence Top-P sampling
// ============================================================================

void sampling_top_p_per_seq(
        float* probs,
        int32_t* output,
        const float* top_p_arr,
        int batch_size,
        int vocab_size,
        bool deterministic,
        uint64_t seed,
        uint64_t offset,
        cudaStream_t stream) {

    int32_t* indices_null = nullptr;
    uint64_t* seed_arr_null = nullptr;
    uint64_t* offset_arr_null = nullptr;
    auto err = flashinfer::sampling::TopPSamplingFromProb<float, int32_t>(
        probs, output, indices_null,
        const_cast<float*>(top_p_arr),
        static_cast<uint32_t>(batch_size),
        1.0f,  // top_p_val unused when array is provided
        static_cast<uint32_t>(vocab_size),
        deterministic,
        seed_arr_null, seed, offset_arr_null, offset,
        stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_top_p_per_seq failed: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Per-sequence Top-K + Top-P combined sampling
// ============================================================================

void sampling_top_k_top_p_per_seq(
        float* probs,
        int32_t* output,
        const int32_t* top_k_arr,
        const float* top_p_arr,
        int batch_size,
        int vocab_size,
        bool deterministic,
        uint64_t seed,
        uint64_t offset,
        cudaStream_t stream) {

    int32_t* indices_null = nullptr;
    uint64_t* seed_arr_null = nullptr;
    uint64_t* offset_arr_null = nullptr;
    auto err = flashinfer::sampling::TopKTopPSamplingFromProb<float, int32_t>(
        probs,
        const_cast<int32_t*>(top_k_arr),
        const_cast<float*>(top_p_arr),
        output, indices_null,
        static_cast<uint32_t>(batch_size),
        0,    // top_k_val unused
        1.0f, // top_p_val unused
        static_cast<uint32_t>(vocab_size),
        deterministic,
        seed_arr_null, seed, offset_arr_null, offset,
        stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_top_k_top_p_per_seq failed: ") + cudaGetErrorString(err));
    }
}

// ============================================================================
// Repetition penalty
// ============================================================================

namespace {

// For each batch item, scan its token history and penalize the corresponding logit.
// Grid: (batch_size)  Block: (256)
__global__ void repetition_penalty_kernel(
        float* __restrict__ logits,
        const int32_t* __restrict__ token_ids,
        const int* __restrict__ seq_lens,
        float penalty,
        int vocab_size, int max_seq_len) {
    const int batch_idx = blockIdx.x;
    const int seq_len = seq_lens[batch_idx];
    float* row = logits + static_cast<long>(batch_idx) * vocab_size;
    const int32_t* tokens = token_ids + static_cast<long>(batch_idx) * max_seq_len;

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        const int tok = tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            float val = row[tok];
            // Penalize: divide positive logits, multiply negative logits
            row[tok] = (val > 0.0f) ? val / penalty : val * penalty;
        }
    }
}

}  // anonymous namespace

void sampling_repetition_penalty(
        float* logits,
        const int32_t* token_ids,
        const int* seq_lens,
        float penalty,
        int batch_size,
        int vocab_size,
        int max_seq_len,
        cudaStream_t stream) {

    if (penalty == 1.0f) return;  // no-op
    repetition_penalty_kernel<<<batch_size, 256, 0, stream>>>(
        logits, token_ids, seq_lens, penalty, vocab_size, max_seq_len);
}

// ============================================================================
// Log-probability extraction: logprob[i] = log(probs[i, token_ids[i]])
// ============================================================================

namespace {

__global__ void extract_logprob_kernel(
        float* __restrict__ logprobs,
        const float* __restrict__ probs,
        const int32_t* __restrict__ token_ids,
        int batch_size, int vocab_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const int token = token_ids[idx];
    if (token < 0 || token >= vocab_size) {
        logprobs[idx] = logf(1e-10f);
        return;
    }
    const float prob = probs[static_cast<long>(idx) * vocab_size + token];
    logprobs[idx] = logf(fmaxf(prob, 1e-10f));  // clamp to avoid log(0)
}

}  // anonymous namespace

void sampling_extract_logprob(
        const float* probs,
        const int32_t* token_ids,
        float* logprobs,
        int batch_size,
        int vocab_size,
        cudaStream_t stream) {

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    extract_logprob_kernel<<<blocks, threads, 0, stream>>>(
        logprobs, probs, token_ids, batch_size, vocab_size);
}

namespace {

template <int Threads>
__global__ void extract_logprob_from_logits_kernel(
        const float* __restrict__ logits,
        const int32_t* __restrict__ token_ids,
        const float* __restrict__ temperature,
        float* __restrict__ logprobs,
        int vocab_size) {
    const int row_idx = blockIdx.x;
    const float* row = logits + static_cast<long>(row_idx) * vocab_size;

    float inv_t = 1.0f;
    if (temperature) {
        const float t = fmaxf(temperature[row_idx], 1e-6f);
        inv_t = 1.0f / t;
    }

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab_size; i += Threads) {
        const float v = row[i] * inv_t;
        local_max = fmaxf(local_max, v);
    }
    using MaxReduce = cub::BlockReduce<float, Threads>;
    __shared__ typename MaxReduce::TempStorage max_storage;
    const float row_max = MaxReduce(max_storage).Reduce(local_max, cub::Max());

    __shared__ float shared_max;
    __shared__ float shared_token_logit;
    if (threadIdx.x == 0) {
        shared_max = row_max;
        const int tok = token_ids[row_idx];
        if (tok >= 0 && tok < vocab_size) {
            shared_token_logit = row[tok] * inv_t;
        } else {
            shared_token_logit = -INFINITY;
        }
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += Threads) {
        local_sum += expf(row[i] * inv_t - shared_max);
    }
    using SumReduce = cub::BlockReduce<float, Threads>;
    __shared__ typename SumReduce::TempStorage sum_storage;
    const float row_sum = SumReduce(sum_storage).Sum(local_sum);

    if (threadIdx.x == 0) {
        const int tok = token_ids[row_idx];
        if (tok < 0 || tok >= vocab_size) {
            logprobs[row_idx] = logf(1e-10f);
            return;
        }
        const float log_z = logf(fmaxf(row_sum, 1e-10f)) + shared_max;
        logprobs[row_idx] = shared_token_logit - log_z;
    }
}

}  // anonymous namespace

void sampling_extract_logprob_from_logits(
        const float* logits,
        const int32_t* token_ids,
        float* logprobs,
        const float* temperature,
        int batch_size,
        int vocab_size,
        cudaStream_t stream) {

    if (batch_size <= 0 || vocab_size <= 0) {
        return;
    }
    constexpr int kThreads = 256;
    extract_logprob_from_logits_kernel<kThreads><<<batch_size, kThreads, 0, stream>>>(
        logits, token_ids, temperature, logprobs, vocab_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Greedy (argmax) sampling
// ============================================================================

namespace {

__global__ void argmax_kernel(
        int32_t* __restrict__ output,
        const float* __restrict__ logits,
        int vocab_size) {
    const int batch_idx = blockIdx.x;
    const float* row = logits + static_cast<long>(batch_idx) * vocab_size;

    float best_val = -INFINITY;
    int best_idx = 0;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = row[i];
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    typedef cub::BlockReduce<cub::KeyValuePair<int, float>, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    cub::KeyValuePair<int, float> thread_data(best_idx, best_val);
    auto result = BlockReduce(temp_storage).Reduce(
        thread_data,
        [](const cub::KeyValuePair<int, float>& a, const cub::KeyValuePair<int, float>& b) {
            return a.value > b.value ? a : b;
        });

    if (threadIdx.x == 0) {
        output[batch_idx] = result.key;
    }
}

}  // anonymous namespace

void sampling_argmax(
        const float* logits,
        int32_t* output,
        int batch_size,
        int vocab_size,
        cudaStream_t stream) {

    argmax_kernel<<<batch_size, 256, 0, stream>>>(
        output, logits, vocab_size);
}
