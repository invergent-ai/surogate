// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Sampling kernels for autoregressive generation.
// Thin C++ wrappers around FlashInfer's header-only CUDA sampling templates.
//
// Phase 2: OnlineSoftmax + SamplingFromProbs (temperature + categorical)
// Phase 6: TopK/TopP/MinP (full sampling suite)

#ifndef SUROGATE_SRC_KERNELS_SAMPLING_H
#define SUROGATE_SRC_KERNELS_SAMPLING_H

#include <cstdint>
#include <cuda_runtime.h>

/// Apply temperature-scaled softmax: logits → probs.
///
/// @param logits       [batch_size, vocab_size] input logits (FP32, in-place or separate output)
/// @param probs        [batch_size, vocab_size] output probabilities (FP32)
///                     May alias logits for in-place operation.
/// @param temperature  [batch_size] per-sequence temperatures (FP32), or nullptr for temperature=1.0
/// @param batch_size   Number of sequences
/// @param vocab_size   Vocabulary size
/// @param workspace    Temporary workspace (needed for large-vocab path, may be nullptr for small vocab)
/// @param workspace_bytes  Size of workspace in bytes
/// @param stream       CUDA stream
void sampling_softmax(
    float* logits,
    float* probs,
    const float* temperature,
    int batch_size,
    int vocab_size,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

/// Sample token IDs from probability distribution (categorical sampling).
///
/// @param probs        [batch_size, vocab_size] normalized probabilities (FP32)
/// @param output       [batch_size] sampled token IDs (INT32)
/// @param batch_size   Number of sequences
/// @param vocab_size   Vocabulary size
/// @param deterministic If true, use deterministic algorithm (slightly slower)
/// @param seed         RNG seed
/// @param offset       RNG offset (incremented per call for different random draws)
/// @param stream       CUDA stream
void sampling_from_probs(
    float* probs,
    int32_t* output,
    int batch_size,
    int vocab_size,
    bool deterministic,
    uint64_t seed,
    uint64_t offset,
    cudaStream_t stream);

/// Combined softmax + sample: logits → sampled token ID (Gumbel-Max trick).
/// Avoids materializing the full probability distribution.
///
/// @param logits       [batch_size, vocab_size] input logits (FP32)
/// @param output       [batch_size] sampled token IDs (INT32)
/// @param batch_size   Number of sequences
/// @param vocab_size   Vocabulary size
/// @param deterministic If true, use deterministic algorithm
/// @param seed         RNG seed
/// @param offset       RNG offset
/// @param stream       CUDA stream
void sampling_from_logits(
    float* logits,
    int32_t* output,
    int batch_size,
    int vocab_size,
    bool deterministic,
    uint64_t seed,
    uint64_t offset,
    cudaStream_t stream);

/// Extract log-probability of the sampled token from probs.
/// logprob[i] = log(probs[i, token_ids[i]])
///
/// @param probs        [batch_size, vocab_size] probabilities (FP32)
/// @param token_ids    [batch_size] sampled token IDs (INT32)
/// @param logprobs     [batch_size] output log-probabilities (FP32)
/// @param batch_size   Number of sequences
/// @param vocab_size   Vocabulary size
/// @param stream       CUDA stream
void sampling_extract_logprob(
    const float* probs,
    const int32_t* token_ids,
    float* logprobs,
    int batch_size,
    int vocab_size,
    cudaStream_t stream);

/// Greedy sampling: argmax over logits.
/// Equivalent to temperature=0 sampling.
///
/// @param logits       [batch_size, vocab_size] input logits (FP32)
/// @param output       [batch_size] token IDs with max logit (INT32)
/// @param batch_size   Number of sequences
/// @param vocab_size   Vocabulary size
/// @param stream       CUDA stream
void sampling_argmax(
    const float* logits,
    int32_t* output,
    int batch_size,
    int vocab_size,
    cudaStream_t stream);

#endif  // SUROGATE_SRC_KERNELS_SAMPLING_H
