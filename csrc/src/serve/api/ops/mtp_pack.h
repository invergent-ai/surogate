#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops {

/**
 * Op: mtp_pack_fc_input
 *
 * Math / indexing:
 *   out[0:D, t] = embedding_norm[:, t]
 *   out[D:2D, t] = hidden_norm[:, t]
 *
 * Logical shapes:
 *   BF16 embedding_norm and hidden_norm [D,T], out [2D,T], contiguous. The registered domains are
 *   D=5120 for Qwen3.6-27B and D=2048 for Qwen3.6-35B-A3B.
 *
 * Numeric:
 *   Exact BF16 element copies; no arithmetic or conversion.
 *
 * Effects:
 *   Writes the full output. Inputs and output must not alias.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
                       cudaStream_t stream);

/**
 * Op: mtp_pack_fc_input_streams
 *
 * The same pack for a residual carried as `streams` copies of the model width (a
 * hyper-connection stack). The embedding is shared across the streams and broadcast; each
 * stream takes its own slice of the hidden:
 *
 *   out[0:D,   t*streams + s] = embedding_norm[:, t]
 *   out[D:2D,  t*streams + s] = hidden_norm[s*D : (s+1)*D, t]
 *
 * Logical shapes: embedding_norm [D,T], hidden_norm [streams*D, T], out [2D, streams*T], all
 * contiguous BF16. The column order puts the stream fastest, which is what lets one matmul
 * over the result write a stream-major residual straight back.
 *
 * Numeric: exact BF16 element copies. Inputs and output must not alias.
 */
void mtp_pack_fc_input_streams(const Tensor& embedding_norm, const Tensor& hidden_norm,
                               std::int32_t streams, Tensor& out, cudaStream_t stream);

/**
 * Op: mtp_split_attn_in
 *
 * Math / indexing:
 *   For each token, rows [0,6144), [6144,7168), [7168,13312), and [13312,14336) are copied to
 *   flattened Q[6144], K[1024], Gate[6144], and V[1024], respectively.
 *
 * Logical shapes:
 *   attn_in [14336,T]; q/gate [256,24,T]; k/v [256,4,T], all contiguous BF16.
 *
 * Numeric:
 *   Exact BF16 element copies with only an index remap.
 *
 * Effects:
 *   Writes every output element. Outputs and input must be pairwise non-aliasing.
 *
 * Workspace:
 *   None. The Op has no persistent state side effect.
 */
void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
                       cudaStream_t stream);

} // namespace sinfer::ops
