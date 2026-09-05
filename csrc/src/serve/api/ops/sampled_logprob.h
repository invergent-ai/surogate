#pragma once

#include "api/ops/sampling.h"
#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace sinfer::ops {

/**
 * The log-probability of one chosen token per column, under that column's
 * temperature-scaled distribution over the whole vocabulary.
 *
 *   out[c] = logits[tokens[c],c]/T_c - log(sum_v exp(logits[v,c]/T_c))
 *
 * evaluated in FP32 from the represented BF16 logits with a max-subtracted sum,
 * where T_c is `configs[c].temperature` and a non-positive temperature -- a
 * greedy row -- reads as 1.
 *
 * This is the quantity a reinforcement-learning trainer divides by: it computes
 * the same log-softmax from its own forward pass, so the two agree exactly when
 * the sampled sequence is on-policy, and their difference is the importance
 * ratio when it is not. It is deliberately over the *whole* vocabulary rather
 * than over the sampler's candidate set: the trainer has no candidate set, so a
 * renormalised-over-20 number would not be comparable with anything it computes.
 *
 * Only rows v in [0, token_domain) participate, as in `sample`: a head padded
 * past the vocabulary carries untrained rows that must not enter the sum. A
 * token outside the vocabulary yields NaN for its column rather than throwing.
 *
 * `logits` is BF16 [vocab, n], `tokens` is I32 [n], `out` is FP32 [n], and
 * `configs` is a device-resident SamplingConfig[n]. The Op writes all of out and
 * uses no workspace or persistent state.
 */
void sampled_logprob(const Tensor& logits, const Tensor& tokens, Tensor& out,
                     std::int32_t token_domain, const SamplingConfig* configs,
                     cudaStream_t stream);

} // namespace sinfer::ops
