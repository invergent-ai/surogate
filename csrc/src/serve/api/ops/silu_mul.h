#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

/**
 * Elementwise SwiGLU activation:
 *
 *   ideal[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
 *
 * `gate`, `up`, and `out` are same-shaped BF16 tensors. out is contiguous; gate and up may use
 * arbitrary valid Tensor strides. out must not overlap either input (the two read-only inputs may
 * overlap one another). The oracle evaluates `ideal` in FP64 from the represented inputs. The BF16
 * output is promoted and compared directly with that result; output storage rounding belongs to
 * the Op's numerical criterion, not the oracle. Private kernel arithmetic is
 * implementation-defined. The Op writes all of out and uses no workspace or persistent state.
 */
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

/**
 * The same with the clamp a checkpoint may have been trained under:
 *
 *   ideal[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, limit).
 *
 * GLM-5.3 bounds both halves this way at every SwiGLU it has -- its dense layers, its routed
 * experts and its always-on expert -- so a value that saturated in training saturates here too.
 * `limit <= 0` is the unclamped product above, which is what the four-argument form passes.
 */
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, float limit,
              cudaStream_t stream);

} // namespace sinfer::ops
