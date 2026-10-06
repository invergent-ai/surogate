#pragma once
// sinfer::ops - the FlashAttention-3 forward instantiations behind gqa_fa3::run (see gqa_fa3.h).
// gqa_fa3.cu picks one by head dim, cache dtype and mask; gqa_fa3_launch_impl.cuh defines them, and
// each gqa_fa3_hdim<D>_<dtype>[_local].cu instantiates one so the twelve compile in parallel.

#include "flash.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail::gqa_fa3 {

/// FA3's sm90 forward for one head dim: causal, or a causal sliding window when `Local` (FA3's
/// local mask with no keys to the right), varlen, paged without TMA (a 64-slot page is not a
/// multiple of every key tile, the condition under which FA3 itself takes this path), the query
/// group packed into the M tile, no split. Over e4m3 queries, keys and values when `Fp8`, else
/// BF16; the output is BF16 either way.
template <int HeadDim, bool Fp8, bool Local>
void launch(Flash_fwd_params& params, cudaStream_t stream);

} // namespace sinfer::ops::detail::gqa_fa3
