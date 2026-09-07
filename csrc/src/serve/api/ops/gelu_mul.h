#pragma once

#include "core/tensor.h"
#include "api/ops/gelu.h" // GeluMode

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

/**
 * Elementwise gated-GELU activation, the GELU twin of `silu_mul`:
 *
 *   Exact: ideal[i] = 0.5*g*(1 + erf(g/sqrt(2))) * up[i]
 *   Tanh:  ideal[i] = 0.5*g*(1 + tanh(sqrt(2/pi)*(g + 0.044715*g^3))) * up[i]
 *
 * for g = gate[i]. Gemma's MLP is gated GELU with the tanh formulation
 * (`hidden_activation: gelu_pytorch_tanh`), which is why the mode is a parameter
 * rather than a constant: the two differ by more than rounding.
 *
 * `gate`, `up`, and `out` are same-shaped contiguous BF16 tensors. out must not
 * overlap either input (the two read-only inputs may overlap one another). The
 * oracle evaluates `ideal` in FP64 from the represented inputs. The BF16 output
 * is promoted and compared directly with that result; output storage rounding
 * belongs to the Op's numerical criterion, not the oracle. Private kernel
 * arithmetic is implementation-defined. The Op writes all of out and uses no
 * workspace or persistent state.
 */
void gelu_mul(const Tensor& gate, const Tensor& up, GeluMode mode, Tensor& out,
              cudaStream_t stream);

} // namespace sinfer::ops
