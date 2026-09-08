#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

/**
 * Applies RMSNorm over the fastest dimension D=ne[0]. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   gain[d]  = unit_offset ? 1 + weight[d] : weight[d]
 *   ideal[d,r] = x[d,r] * inv_r * gain[d].
 *
 * `x` is BF16 or FP32 and `out` is BF16 with the same contiguous shape; weight is BF16 [D], and eps
 * is positive and finite. Input, weight, and output must not overlap. The oracle evaluates `ideal`
 * naively in FP64 from the represented inputs. The BF16 output is promoted and compared directly
 * with that result; output storage rounding belongs to the Op's numerical criterion, not the
 * oracle. Kernel reduction, staging, and accumulator precision are implementation choices. There
 * is no workspace or persistent state side effect.
 */
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream);

/**
 * The same normalisation, accumulated into `out` rather than written over it:
 *
 *   ideal[d,r] = out[d,r] + bf16(x[d,r] * inv_r * gain[d])
 *
 * where bf16() is round-to-nearest-even. The normalised term is rounded before
 * the addition, so this is bit-identical to `rmsnorm` into a scratch plane
 * followed by `residual_add` -- which is the pair it exists to replace, and
 * which the reference implementation also performs in that order.
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors and must not overlap
 * each other; `out` is read and written. Shape, weight and eps rules are
 * `rmsnorm`'s. There is no workspace or persistent state side effect.
 */
void rmsnorm_add(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
                 cudaStream_t stream);

/**
 * The same normalisation with no gain at all:
 *
 *   inv_r      = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   ideal[d,r] = x[d,r] * inv_r.
 *
 * There is no weight, rather than a weight of ones: Gemma 4 normalises its attention value
 * with `RMSNorm(dim, with_scale=False)`, which owns no parameter, and a caller would
 * otherwise have to keep a ones vector somewhere on the device for a kernel to multiply by.
 *
 * This is *not* `l2norm`, which is the other weightless normalisation in this directory:
 * that one divides by `sqrt(sum_d x^2 + eps)` where this divides by `sqrt(mean_d x^2 + eps)`,
 * so they differ by `sqrt(D)` and place their eps differently.
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors and eps is positive and finite. Input
 * and output must not overlap. The oracle evaluates `ideal` naively in FP64 from the
 * represented input; output storage rounding belongs to the Op's numerical criterion. There
 * is no workspace or persistent state side effect.
 */
void rmsnorm_unweighted(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops
