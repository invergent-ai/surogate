// sinfer::ops - FlashAttention-3's forward at head dim 128 over an e4m3 paged cache,
// causal sliding window (see gqa_fa3.h and gqa_fa3_launch.h).
#include "ops/gqa_sm90/gqa_fa3_launch_impl.cuh"

namespace sinfer::ops::detail::gqa_fa3 {

template void launch<128, true, true, false>(Flash_fwd_params& params, cudaStream_t stream);

} // namespace sinfer::ops::detail::gqa_fa3
