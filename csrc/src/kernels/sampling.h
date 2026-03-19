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

/// Top-K sampling from probability distribution.
///
/// @param probs        [batch_size, vocab_size] normalized probabilities (FP32, modified in-place)
/// @param output       [batch_size] sampled token IDs (INT32)
/// @param top_k        Top-K value (same for all sequences)
/// @param batch_size, vocab_size, deterministic, seed, offset, stream
void sampling_top_k(
    float* probs,
    int32_t* output,
    int top_k,
    int batch_size,
    int vocab_size,
    bool deterministic,
    uint64_t seed,
    uint64_t offset,
    cudaStream_t stream);

/// Top-P (nucleus) sampling from probability distribution.
///
/// @param probs        [batch_size, vocab_size] normalized probabilities (FP32, modified in-place)
/// @param output       [batch_size] sampled token IDs (INT32)
/// @param top_p        Top-P threshold (same for all sequences)
/// @param batch_size, vocab_size, deterministic, seed, offset, stream
void sampling_top_p(
    float* probs,
    int32_t* output,
    float top_p,
    int batch_size,
    int vocab_size,
    bool deterministic,
    uint64_t seed,
    uint64_t offset,
    cudaStream_t stream);

/// Combined Top-K + Top-P sampling.
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
    cudaStream_t stream);

/// Min-P sampling from probability distribution.
///
/// @param probs        [batch_size, vocab_size] normalized probabilities (FP32)
/// @param output       [batch_size] sampled token IDs (INT32)
/// @param min_p        Min-P threshold (tokens with prob < min_p * max_prob are masked)
/// @param batch_size, vocab_size, deterministic, seed, offset, stream
void sampling_min_p(
    float* probs,
    int32_t* output,
    float min_p,
    int batch_size,
    int vocab_size,
    bool deterministic,
    uint64_t seed,
    uint64_t offset,
    cudaStream_t stream);

/// Apply repetition penalty to logits based on tokens already generated.
///
/// For each token that appears in the context, its logit is divided by the
/// penalty factor (if positive) or multiplied (if negative).
/// penalty = 1.0 means no change.
///
/// @param logits       [batch_size, vocab_size] logits to modify in-place (FP32)
/// @param token_ids    [batch_size, seq_len] token IDs in the context (INT32)
/// @param seq_lens     [batch_size] actual length per sequence (INT32)
/// @param penalty      Repetition penalty factor (> 1.0 penalizes, < 1.0 encourages)
/// @param batch_size, vocab_size, max_seq_len, stream
void sampling_repetition_penalty(
    float* logits,
    const int32_t* token_ids,
    const int* seq_lens,
    float penalty,
    int batch_size,
    int vocab_size,
    int max_seq_len,
    cudaStream_t stream);

/// Per-sequence Top-K sampling (different K per batch item).
///
/// @param probs        [batch_size, vocab_size] normalized probabilities (FP32, modified in-place)
/// @param output       [batch_size] sampled token IDs (INT32)
/// @param top_k_arr    [batch_size] per-sequence Top-K values on GPU (INT32)
/// @param batch_size, vocab_size, deterministic, seed, offset, stream
void sampling_top_k_per_seq(
    float* probs,
    int32_t* output,
    const int32_t* top_k_arr,
    int batch_size,
    int vocab_size,
    bool deterministic,
    uint64_t seed,
    uint64_t offset,
    cudaStream_t stream);

/// Per-sequence Top-P sampling (different P per batch item).
void sampling_top_p_per_seq(
    float* probs,
    int32_t* output,
    const float* top_p_arr,
    int batch_size,
    int vocab_size,
    bool deterministic,
    uint64_t seed,
    uint64_t offset,
    cudaStream_t stream);

/// Per-sequence Top-K + Top-P combined sampling.
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
