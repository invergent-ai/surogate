// The build's architecture set has no 90a, so it carries no Hopper FP8 kernel: every call
// declines and the block-scaled FP8 routes run the engine's own tile (fp8_block_sm90_gemm.h).

#include "ops/linear/fp8_block/fp8_block_sm90_deepgemm.h"
#include "ops/linear/fp8_block/fp8_block_sm90_gemm.h"

namespace sinfer::ops::detail::fp8_block {

bool sm90_gemm_available() noexcept { return false; }

bool sm90_gemm(const std::uint8_t*, const float*, const std::uint8_t*, const float*, void*, bool,
               std::int32_t, std::int32_t, std::int32_t, cudaStream_t) {
    return false;
}

bool sm90::deepgemm(const sm90::DeepGemmOperands&) { return false; }

} // namespace sinfer::ops::detail::fp8_block
