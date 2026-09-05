#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace sinfer::ops {

/**
 * Op: short_conv
 *
 * The LFM2 short-convolution mixer, from the projection it reads to the value it
 * hands the output projection.
 *
 * Math / indexing. `bcx` is one projection holding three equal parts stacked --
 * B, C and x, each `channels` rows -- so with `u[c,t] = B[c,t] * x[c,t]` and a
 * state carrying the `K-1` columns that preceded column 0:
 *
 *   out[c,t] = C[c,t] * sum_{j=0..K-1} taps[j,c] * u[c, t-(K-1)+j]
 *
 * where `u[c,t']` for `t' < 0` is `state[c, t' + K - 1]`.
 *
 * The gate is applied after the convolution and there is no activation between
 * them, which is what separates this from `causal_conv1d_silu`: that one fuses a
 * SiLU at a fixed width of four and is the linear-attention family's, not this.
 * The three parts stay in one tensor because the projection that produced them is
 * one matmul; slicing them here costs nothing and reading them apart would cost a
 * copy.
 *
 * Shapes. `bcx` is contiguous BF16 [3*channels, T]; `taps` contiguous BF16 [K,
 * channels], tap-major so the inner sum reads consecutive channels; `state` is
 * contiguous BF16 [channels, K-1], oldest column first; `out` is contiguous BF16
 * [channels, T]. K is taken from `taps.ne[0]` and must be in [2,4]. T may be any
 * positive value.
 *
 * Effects. Writes all of `out`. `state` is read as the initial window and replaced
 * with the trailing `K-1` columns of `u`, so a decode round carries its own
 * history; pass the same storage for both roles. No workspace, no other state.
 */
void short_conv(const Tensor& bcx, const Tensor& taps, Tensor& state, Tensor& out,
                std::int32_t channels, cudaStream_t stream);

} // namespace sinfer::ops
