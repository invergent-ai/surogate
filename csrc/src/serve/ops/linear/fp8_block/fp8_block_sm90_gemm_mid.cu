// The Hopper block-scaled FP8 GEMM's swapped cooperative tiles, 128 weight rows by 32 or 64
// tokens (fp8_block_sm90_gemm.cuh): a translation unit of their own so they compile in parallel
// with the other tiles.

#include "ops/linear/fp8_block/fp8_block_sm90_gemm.cuh"

namespace sinfer::ops::detail::fp8_block::sm90 {

bool launch_mid(const Operands& o, bool residual, int tile_tokens) {
    if (tile_tokens == 32) { return residual ? launch<Mid<true, 32>>(o) : launch<Mid<false, 32>>(o); }
    if (tile_tokens == 64) { return residual ? launch<Mid<true, 64>>(o) : launch<Mid<false, 64>>(o); }
    return false;
}

} // namespace sinfer::ops::detail::fp8_block::sm90
