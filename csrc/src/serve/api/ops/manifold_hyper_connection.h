#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops {

/**
 * Manifold-constrained hyper-connections (mHC), GLM-5.3's residual topology.
 *
 * Read against `hyper_connection.h`, which is Flash-Next's. Both carry `streams` copies of the
 * model width side by side -- column t of a [streams*hidden, T] tensor holds stream s at rows
 * [s*hidden, (s+1)*hidden) -- and both collapse them to one block input and scatter the block's
 * output back. The difference is the scatter: Flash-Next *adds* the block output to each stream
 * and leaves the streams otherwise untouched, so a stream only ever accumulates. Here the
 * streams are also *mixed with each other*, by a stream x stream matrix projected onto the
 * doubly-stochastic manifold. That projection is why nothing about this can be spelled as the
 * other one with different weights.
 *
 * One projection produces all three of the per-token mixings, from the residual normalised
 * without a learned scale -- `mix` has `(2 + streams) * streams` rows:
 *
 *   n[:,t]      = residual[:,t] * rsqrt(mean over all streams*hidden of residual^2 + rms_eps)
 *   l[:,t]      = mix * n[:,t]                                    [(2+streams)*streams]
 *   pre[s]      = sigmoid(l[s]*scale[0] + base[s]) + hc_eps
 *   post[s]     = 2 * sigmoid(l[streams+s]*scale[1] + base[streams+s])
 *   c[i,j]      = l[2*streams + i*streams + j]*scale[2] + base[2*streams + i*streams + j]
 *   comb        = Sinkhorn(softmax_j(c) + hc_eps)
 *   collapsed[d,t] = sum_s pre[s,t] * residual[s*hidden+d, t]
 *
 * Sinkhorn-Knopp, exactly as the reference iterates it: normalise by the column sums once, then
 * `iterations - 1` further rounds of rows-then-columns, every division guarded by `hc_eps`. The
 * result is the doubly-stochastic matrix the name refers to; a stream neither gains nor loses
 * total weight through the mix, which is what keeps a 45-layer stack of these from drifting.
 *
 * The `pre` weights are applied to the *raw* residual, not to `n`: the normalisation exists to
 * make the projection scale-free, not to rescale the stream that the block then reads.
 *
 * Shapes. `residual` contiguous BF16 [streams*hidden, T]; `mix` contiguous BF16_CTRL
 * [(2+streams)*streams, streams*hidden]; `base` FP32 [(2+streams)*streams]; `scale` FP32 [3];
 * `collapsed` contiguous BF16 [hidden, T]; `post` contiguous FP32 [streams, T]; `comb`
 * contiguous FP32 [streams, streams, T] with the second index fastest, so column t holds the
 * matrix row-major. `streams` is 1..4 and `streams*hidden` is a multiple of 8 no wider than
 * 32768 -- the whole residual column is staged in shared memory, which is what makes one pass
 * over it answer both the norm and every projection row.
 */
struct ManifoldHyperConnectionWeights {
    Weight mix;
    Tensor base;
    Tensor scale;
};

/// Transient capacity `manifold_hyper_connection_mix` needs for every T in [min_tokens,
/// max_tokens]: the projection is split across the width into (token, slice) blocks whose
/// partial sums land in an FP32 workspace -- one sum of squares and one dot per mixing row,
/// plus the `pre` gate, per token -- which the reduce then finishes. Sized for `max_tokens`,
/// rounded the way the arena rounds an allocation.
[[nodiscard]] std::size_t manifold_hyper_connection_mix_workspace_capacity_bytes(
    std::int32_t streams, std::int32_t hidden, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Collapses the residual streams into one block input and publishes the `post` and `comb`
 * mixings the matching combine consumes. `post` and `comb` may not alias `residual`.
 */
void manifold_hyper_connection_mix(const Tensor& residual,
                                   const ManifoldHyperConnectionWeights& weights,
                                   std::int32_t streams, float rms_eps, float hc_eps,
                                   std::int32_t sinkhorn_iterations, Tensor& collapsed,
                                   Tensor& post, Tensor& comb, WorkspaceArena& workspace,
                                   cudaStream_t stream);

/**
 * Scatters a block output back into the residual streams, mixing them as it goes:
 *
 *   residual[s*hidden+d, t] = post[s,t] * block_output[d,t]
 *                             + sum_{s'} comb[s',s,t] * residual[s'*hidden+d, t]
 *
 * The sum reads the incoming residual, so this is not an accumulate: every stream is rewritten
 * from all of them. In place, which is the only form the family's hook can call.
 */
void manifold_hyper_connection_combine(const Tensor& block_output, const Tensor& post,
                                       const Tensor& comb, std::int32_t streams, Tensor& residual,
                                       cudaStream_t stream);

/// mean[d,t] = (1/streams) * sum_s residual[s*hidden+d, t]; the unweighted collapse that ends
/// the stack, before the final norm. `residual` BF16 [streams*hidden, T], `mean` BF16 [hidden,T].
void collapse_streams_mean(const Tensor& residual, std::int32_t streams, Tensor& mean,
                           cudaStream_t stream);

} // namespace sinfer::ops
