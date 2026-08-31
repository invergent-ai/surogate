#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops {

/**
 * Hyper-connection (multi-stream residual) weights. The residual is `streams` copies of the
 * model width side by side: column t of a [streams*hidden, T] tensor holds stream s at rows
 * [s*hidden, (s+1)*hidden).
 *
 *   norm    FP32 [streams*hidden]           per-stream RMSNorm gamma, folded (1 + w)
 *   down    BF16_CTRL [low_rank, streams*hidden]
 *   up      BF16_CTRL [streams*hidden, low_rank]
 *   inject  BF16_CTRL [streams, streams*hidden]; n == 0 for a mixer without a combine step
 */
struct HyperConnectionWeights {
    Tensor norm;
    Weight down;
    Weight up;
    Weight inject;
};

/**
 * Returns the transient capacity `hyper_connection_mix` needs for every T in
 * [min_tokens, max_tokens].
 */
[[nodiscard]] std::size_t hyper_connection_mix_workspace_capacity_bytes(std::int32_t streams,
                                                                        std::int32_t hidden,
                                                                        std::int32_t low_rank,
                                                                        std::int32_t min_tokens,
                                                                        std::int32_t max_tokens);

/**
 * Mixes the residual streams into one block input.
 *
 *   n[i,t]      = residual[i,t] * rsqrt(mean_{d in stream(i)} residual^2 + eps) * norm[i]
 *   lo          = silu(down · n / streams)                       [low_rank, T]
 *   gate        = sigmoid(up · lo)                                [streams*hidden, T]
 *   mixed[d,t]  = mean_s gate[s*hidden+d,t] * n[s*hidden+d,t]    [hidden, T]
 *   inject[s,t] = 2 * sigmoid((inject_w[s,:] · n[:,t]) / streams) [streams, T] (FP32)
 *
 * `residual` is contiguous BF16 [streams*hidden, T]; `mixed` contiguous BF16 [hidden, T];
 * `inject` contiguous FP32 [streams, T] or nullptr when the weights carry no inject rows. The
 * two projections run through cuBLASLt (prewarm before stream capture). Workspace is
 * caller-owned transient storage; nothing persists between calls.
 */
void hyper_connection_mix(const Tensor& residual, const HyperConnectionWeights& weights,
                          std::int32_t streams, float eps, Tensor& mixed, Tensor* inject,
                          WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Scatters a block output back into the residual streams:
 * residual[s*hidden+d, t] += inject[s,t] * block_output[d,t].
 */
void hyper_connection_combine(const Tensor& block_output, const Tensor& inject, Tensor& residual,
                              cudaStream_t stream);
/// As above with an FP32 [hidden, T] term added to `block_output` before the scatter (a partial
/// computed elsewhere, e.g. on the host, read through a device-mapped pointer); null = none.
void hyper_connection_combine(const Tensor& block_output, const float* extra, const Tensor& inject,
                              Tensor& residual, cudaStream_t stream);

/// residual[s*hidden+d, t] = source[d, t] for every stream s (the embedding entry point).
void broadcast_streams(const Tensor& source, std::int32_t streams, Tensor& residual,
                       cudaStream_t stream);

} // namespace sinfer::ops
