#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Adds one bias value to every element whose fastest-dimension index is the same:
 *
 *   ideal[d,r] = x[d,r] + bias[d],  0 <= d < D.
 *
 * Here r flattens all dimensions above ne[0]. `bias` is contiguous BF16 [D] and `x` is a
 * contiguous BF16 tensor with ne[0]=D. The oracle evaluates `ideal` in FP64 from the represented
 * inputs. The updated BF16 x is promoted and compared directly with that result; output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. Private kernel arithmetic is
 * implementation-defined. `bias` must not overlap `x`. There is no workspace or other state side
 * effect.
 */
void add_bias(const Tensor& bias, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
