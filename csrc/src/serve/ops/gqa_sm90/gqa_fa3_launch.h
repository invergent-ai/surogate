#pragma once
// sinfer::ops - the FlashAttention-3 forward instantiations behind gqa_fa3::run (see gqa_fa3.h).
// gqa_fa3.cu picks one by head dim, cache dtype, mask and split; gqa_fa3_launch_impl.cuh defines
// them, and each gqa_fa3_hdim<D>_<dtype>[_local][_split].cu instantiates one so the eighteen
// compile in parallel (split ones over BF16 only: see gqa_fa3.h). gqa_fa3_combine.cu holds the
// split-KV combine.

#include "flash.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail::gqa_fa3 {

/// FA3's sm90 forward for one head dim: causal, or a causal sliding window when `Local` (FA3's
/// local mask with no keys to the right), varlen, paged without TMA (a 64-slot page is not a
/// multiple of every key tile, the condition under which FA3 itself takes this path), the query
/// group packed into the M tile. Over e4m3 queries, keys and values when `Fp8`, else BF16; the
/// output is BF16 either way. With `Split` each segment's keys may be spread over up to
/// `num_splits` CTAs, as many as the scheduler's prepare kernel gives it on the device; a segment
/// it splits writes FP32 partials for `combine`, the rest write the output directly.
template <int HeadDim, bool Fp8, bool Local, bool Split>
void launch(Flash_fwd_params& params, cudaStream_t stream);

/// FA3's split-KV combine: merges the partials of every segment the forward split into the BF16
/// output, by their log-sum-exps. `num_splits` must be at most 64.
void combine(Flash_fwd_params& params, cudaStream_t stream);

} // namespace sinfer::ops::detail::gqa_fa3
