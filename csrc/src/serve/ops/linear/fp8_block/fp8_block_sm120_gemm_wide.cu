// The sm_12x FP8 GEMM's 128-token cooperative tile, over block-scaled and per-row weights
// (fp8_block_sm120_gemm.cuh): a translation unit of its own so it compiles in parallel with the
// other tiles.

#include "ops/linear/fp8_block/fp8_block_sm120_gemm.cuh"

namespace sinfer::ops::detail::fp8_block::sm120 {

bool launch_wide(const Operands& o, bool residual, bool per_row) {
    if (per_row) { return residual ? launch_blockwise<Wide<true, true>>(o) : launch_blockwise<Wide<false, true>>(o); }
    return residual ? launch_blockwise<Wide<true, false>>(o) : launch_blockwise<Wide<false, false>>(o);
}

} // namespace sinfer::ops::detail::fp8_block::sm120
