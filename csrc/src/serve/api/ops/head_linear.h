#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops {

/**
 * A projection applied per head, each head with its own matrix:
 *
 *   out[h*n + i, t] = out_scale * sum_j FP32Dequant(w)[h*n + i, j] * FP32(x[h*k + j, t])
 *
 * `x` is contiguous BF16 `[heads*k, T]` -- head h's input is rows `[h*k, (h+1)*k)`; `w` is a
 * W8G32_F16S row-split weight of logical shape `[heads*n, k]`, head h's matrix its rows
 * `[h*n, (h+1)*n)`; `out` is contiguous BF16 `[heads*n, T]`. Dimension zero is stored fastest.
 *
 * This is what a latent attention needs on both sides of its scores: the query folded through
 * one head-specific matrix into the latent on the way in, and the attended latent unfolded
 * through another on the way out. `linear` applies one matrix to every column and cannot say
 * it; a loop over `linear` could, at sixty-four launches a site.
 *
 * `out_scale` multiplies every output. It exists because the attention kernels implement one
 * softmax scale, `1/sqrt(head_dim)`, and a latent attention scores over a head wider than the
 * one its scale was trained for; the ratio is folded into the query here rather than into the
 * weights, which are the checkpoint's and read where they lie.
 *
 * Numerical contract: the oracle materialises the dequantised weight in FP32, takes the FP32
 * values of the BF16 activations, and accumulates every dot product in FP64; the BF16 output
 * is compared against that under the criterion the test names. W8's dequantisation -- an int8
 * code times an FP16 scale -- is exact in FP32, so the only private effects are the FP32
 * accumulation order and the output rounding.
 */
void head_linear(const Tensor& x, const Weight& w, std::int32_t heads, float out_scale,
                 Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops
