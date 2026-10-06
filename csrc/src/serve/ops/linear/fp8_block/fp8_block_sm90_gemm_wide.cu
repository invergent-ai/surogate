// The Hopper block-scaled FP8 GEMM's cooperative 128- and 256-token tiles over a 1 x 2 cluster
// (fp8_block_sm90_gemm.cuh): a translation unit of their own so they compile in parallel with
// the other tiles.

#include "ops/linear/fp8_block/fp8_block_sm90_gemm.cuh"

namespace sinfer::ops::detail::fp8_block::sm90 {

bool launch_wide(const Operands& o, bool residual, int tile_tokens) {
    if (tile_tokens == 128) { return residual ? launch<Wide<true, 128>>(o) : launch<Wide<false, 128>>(o); }
    if (tile_tokens == 256) { return residual ? launch<Wide<true, 256>>(o) : launch<Wide<false, 256>>(o); }
    return false;
}

} // namespace sinfer::ops::detail::fp8_block::sm90
