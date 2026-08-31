#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops

{

/**
 * Mean over the token axis of a hidden-state matrix:
 *
 *   ideal[h] = (1/count) * sum over t in [0, count) of x[h, t]
 *
 * `x` is a contiguous BF16 tensor shaped [hidden, tokens] with hidden fastest --
 * the layout the text stack already materialises. `out` is a contiguous BF16
 * tensor of [hidden]. `count` is how many columns participate and may be fewer
 * than `x.ne[1]`, which is what lets one padded batch slot pool only its own
 * tokens.
 *
 * `accumulate` adds this chunk's *sum* into out instead of writing the mean, so
 * a prompt longer than one prefill chunk can be pooled across several calls; the
 * caller divides once at the end. When false the Op writes the mean directly.
 *
 * Summation is FP32 regardless of storage: a 2048-token mean of BF16 values
 * accumulated in BF16 loses roughly a decimal digit, and the whole output of an
 * embedding model is this one vector.
 *
 * The oracle evaluates `ideal` in FP64 from the represented input. The BF16
 * output is promoted and compared directly with that result; output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. Private
 * kernel arithmetic is implementation-defined. The Op writes all of out and uses
 * no workspace or persistent state.
 */
void mean_pool(const Tensor& x, int count, bool accumulate, Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops
