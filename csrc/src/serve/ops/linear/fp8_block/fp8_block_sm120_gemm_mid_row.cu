// The sm_12x FP8 GEMM's swapped cooperative tiles over per-row weights, 128 weight rows by 32 or
// 64 tokens (fp8_block_sm120_gemm.cuh): a translation unit of their own so they compile in
// parallel with the other tiles.

#include "ops/linear/fp8_block/fp8_block_sm120_gemm.cuh"

namespace sinfer::ops::detail::fp8_block::sm120 {

bool launch_mid_row(const Operands& o, bool residual, int tile_tokens) {
    if (tile_tokens == 32) { return residual ? launch_blockwise<Mid<true, true, 32>>(o) : launch_blockwise<Mid<false, true, 32>>(o); }
    if (tile_tokens == 64) { return residual ? launch_blockwise<Mid<true, true, 64>>(o) : launch_blockwise<Mid<false, true, 64>>(o); }
    return false;
}

} // namespace sinfer::ops::detail::fp8_block::sm120
