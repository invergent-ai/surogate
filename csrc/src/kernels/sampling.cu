// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Thin C++ wrappers around FlashInfer sampling templates.
// Instantiates templates for float + int32_t (our generation dtypes).

#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

// FlashInfer sampling headers (fetched via CMake FetchContent)
#include <flashinfer/sampling.cuh>

#include "kernels/sampling.h"
#include "utilities/utils.h"

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
        /*enable_pdl=*/false,
        stream);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sampling_softmax failed: ") + cudaGetErrorString(err));
    }
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
