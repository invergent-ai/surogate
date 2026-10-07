// The build's architecture set has no sm_12x target, so it carries no sm_120 FP8 kernel: every
// call declines and the block-scaled FP8 routes run the engine's own tile (fp8_block_sm120_gemm.h).

#include "ops/linear/fp8_block/fp8_block_sm120_gemm.h"

namespace sinfer::ops::detail::fp8_block {

bool sm120_gemm_available() noexcept { return false; }

bool sm120_gemm(const std::uint8_t*, const float*, const std::uint8_t*, const float*, bool, void*,
                bool, std::int32_t, std::int32_t, std::int32_t, cudaStream_t) {
    return false;
}

} // namespace sinfer::ops::detail::fp8_block
