#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise sigmoid gate:
 *
 *   ideal[i] = x[i] * (1 / (1 + exp(-gate[i]))).
 *
 * `gate` and `x` are non-overlapping, same-shaped contiguous BF16 tensors. The oracle evaluates
 * `ideal` in FP64 from the represented inputs. The updated BF16 x is promoted and compared directly
 * with that result; output storage rounding belongs to the Op's numerical criterion, not the
 * oracle. Private kernel arithmetic is implementation-defined. The Op uses no workspace or other
 * persistent state.
 */
void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
