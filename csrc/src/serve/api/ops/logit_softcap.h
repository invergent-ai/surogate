#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

/**
 * Squashes logits into (-cap, +cap), elementwise and in place:
 *
 *   ideal[i] = tanh(x[i] / cap) * cap.
 *
 * `x` is an arbitrary-rank contiguous BF16 tensor and `cap` is positive and finite.
 *
 * Gemma is the family that needs this: `final_logit_softcapping` is 30 on every published
 * Gemma 4, and the reference applies it to the head's output before anything reads it
 * (`logits = tanh(logits / c) * c`). It is not a sampling detail that a temperature could
 * stand in for -- it is part of the model, it changes the *ratios* between logits rather
 * than scaling them, and the model was trained with it.
 *
 * Applied in place because the logits plane is the largest transient in a round -- 262,144
 * rows on this family -- and a second copy of it buys nothing.
 *
 * The oracle evaluates `ideal` in FP64 from the represented input. The updated BF16 x is
 * promoted and compared directly with that result; output storage rounding belongs to the
 * Op's numerical criterion, not the oracle. The Op mutates only x and uses no workspace or
 * persistent state.
 */
void logit_softcap(Tensor& x, float cap, cudaStream_t stream);

} // namespace sinfer::ops
