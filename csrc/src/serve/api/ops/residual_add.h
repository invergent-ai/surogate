#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise residual update:
 *
 *   ideal[i] = x[i] + y[i].
 *
 * `y` and `x` are non-overlapping, same-shaped contiguous BF16 tensors. The Op updates all of x
 * in place and leaves y unchanged. The oracle evaluates `ideal` in FP64 from the represented
 * inputs. The updated BF16 x is promoted and compared directly with that result; output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. Private kernel arithmetic is
 * implementation-defined. The Op uses no workspace or other persistent state.
 */
void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
