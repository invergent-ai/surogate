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
 * the layout the text stack already materialises. `count` is how many columns
 * participate and may be fewer than `x.ne[1]`, which is what lets one padded
 * batch slot pool only its own tokens.
 *
 * `accumulate` adds this chunk's *sum* into out instead of writing the mean, so
 * a prompt longer than one prefill chunk can be pooled across several calls; the
 * caller divides once at the end. When false the Op writes the mean directly.
 *
 * `out` is a contiguous **FP32** tensor of [hidden], and that is not a
 * convenience. Accumulating a running sum through BF16 storage rounds it once
 * per chunk: over a 512-token sequence split in two that alone costs 0.36%,
 * which is an order of magnitude worse than the quantised weights. This one
 * vector is the entire output of an embedding model, so it stays FP32 until the
 * caller casts it for the head.
 *
 * The oracle evaluates `ideal` in FP64 from the represented input. The FP32
 * output is promoted and compared directly with that result; output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. Private
 * kernel arithmetic is implementation-defined. The Op writes all of out and uses
 * no workspace or persistent state.
 */
void mean_pool(const Tensor& x, int count, bool accumulate, Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops
