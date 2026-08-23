#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

enum class GeluMode {
    Exact,
    Tanh,
};

/**
 * Applies GELU elementwise in place. For z=x[i]:
 *
 *   Exact: ideal[i] = 0.5*z*(1 + erf(z/sqrt(2)))
 *   Tanh:  ideal[i] = 0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3))).
 *
 * `x` is an arbitrary-rank contiguous BF16 tensor. The oracle evaluates `ideal` in FP64 from the
 * represented input. The updated BF16 x is promoted and compared directly with that result; output
 * storage rounding belongs to the Op's numerical criterion, not the oracle. Private kernel
 * arithmetic is implementation-defined. The Op mutates only x and uses no workspace or persistent
 * state.
 */
void gelu(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace ninfer::ops
