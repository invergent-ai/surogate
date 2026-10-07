// sinfer::ops - FlashAttention-3's forward at head dim 128 over a BF16 paged cache,
// causal mask, on one MMA warpgroup (64-row M tile), split over the keys, for launches
// whose segments pack at most 64 query rows (see gqa_fa3.h and gqa_fa3_launch.h).
#include "ops/gqa_sm90/gqa_fa3_launch_impl.cuh"

namespace sinfer::ops::detail::gqa_fa3 {

template void launch<128, false, false, true, true>(Flash_fwd_params& params, cudaStream_t stream);

} // namespace sinfer::ops::detail::gqa_fa3
