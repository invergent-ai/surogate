#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops {

/**
 * Negative log-likelihood of one target token per column of a logits matrix.
 *
 * `logits` is BF16 [vocab, n] with the vocabulary contiguous, `targets` is I32 [n] naming the
 * token each column is scored against, and `out` is FP32 [n]:
 *
 *   out[c] = log(sum_v exp(logits[v,c])) - logits[targets[c], c]
 *
 * evaluated in FP32 from the represented BF16 logits with a max-subtracted sum. That is the
 * cross-entropy a perplexity measurement accumulates; the serving path never needs it, so this
 * op exists for measurement rather than for any request. A target outside the vocabulary
 * yields NaN for its column rather than an exception, so a probe can flag the position.
 * `argmax`, when given, is I32 [n] and receives each column's highest-scoring token. As in
 * `sample`, only rows v in [0, token_domain) participate: a head padded past the vocabulary
 * carries untrained rows that must not enter the sum.
 */
void next_token_nll(const Tensor& logits, const Tensor& targets, Tensor& out, Tensor* argmax,
                    std::int32_t token_domain, cudaStream_t stream);

} // namespace sinfer::ops
