#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

/**
 * Scales a tensor by a constant, in place:
 *
 *   ideal[i] = x[i] * factor.
 *
 * `x` is an arbitrary-rank contiguous BF16 tensor and `factor` is finite. Gemma
 * multiplies its embedding table output by sqrt(hidden), and that factor cannot
 * be folded anywhere cheaper: the norms divide it out again, but the residual
 * stream carries the scaled value forward, so it sets the size of the residual
 * relative to every block's output.
 *
 * It could instead be folded into a quantised table's scale plane at conversion,
 * which costs nothing at runtime. It deliberately is not: that would make the
 * artifact's tensors no longer equal the checkpoint's, and decoding an artifact
 * and diffing it against the checkpoint is the first thing worth doing when
 * output looks wrong.
 *
 * The oracle evaluates `ideal` in FP64 from the represented input. The updated
 * BF16 x is promoted and compared directly with that result; output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. The Op
 * mutates only x and uses no workspace or persistent state.
 */
void scale(Tensor& x, float factor, cudaStream_t stream);

} // namespace sinfer::ops
