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
 * Shapes, in this engine's layout, where the first dimension is the fastest and a
 * column's channels are therefore contiguous. `bcx` is contiguous BF16
 * [3*channels, T], so one column holds its B, C and x parts one after another;
 * `taps` is contiguous BF16 [channels, K], one tap plane after another, exactly
 * as the linear-attention convolution stores its own; `state` is contiguous BF16
 * [channels, K-1], oldest column first; `out` is contiguous BF16 [channels, T].
 * K is taken from `taps.ne[1]` and must be in [2,4]. T may be any positive value.
 *
 * Effects. Writes all of `out`. `state` is read as the initial window and replaced
 * with the trailing `K-1` columns of `u`, so a decode round carries its own
 * history; pass the same storage for both roles. No workspace, no other state.
 */
void short_conv(const Tensor& bcx, const Tensor& taps, Tensor& state, Tensor& out,
                std::int32_t channels, cudaStream_t stream);

/**
 * Snapshot form, for B independent sequences that do not share a history.
 *
 * A decode round carries one column for each of B lanes, and each lane's convolution must
 * continue that lane's own K-1 columns -- so the state is a pool of slots and every row says
 * which one it starts from and where its new windows go.
 *
 * Shapes. `bcx` is contiguous BF16 [3*channels, W, B] and `out` contiguous BF16 [channels, W, B].
 * `conv_states` is contiguous BF16 [channels, K-1, slots], so a slot's window is contiguous. `initial_state_slots` and
 * `snapshot_base_slots` are contiguous I32 [B]. `valid_columns` is contiguous I32 [B] with every
 * value in [1,W], or an empty Tensor meaning every row is W columns wide.
 *
 * Effects. Row b reads the window in `initial_state_slots[b]` and, after each valid column j,
 * writes the window that follows it to `snapshot_base_slots[b] + j` -- one checkpoint per
 * column, which is what lets a round be rolled back to any column it accepted. Output columns
 * past a row's valid count are exact BF16 zero and change no state. The caller reserves the
 * whole [base, base+W) interval for every row, all reservations disjoint; a row's own initial
 * slot may lie inside its own reservation, which is the ordinary decode case where a lane
 * overwrites the window it just read.
 */
void short_conv_snapshot(const Tensor& bcx, const Tensor& taps, Tensor& conv_states,
                         const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                         const Tensor& valid_columns, Tensor& out, std::int32_t channels,
                         cudaStream_t stream);

} // namespace sinfer::ops
