// The build's architecture set has no 90a, so it carries no Hopper grouped GEMM: block-FP8 routed
// experts have no kernel here, and every entry point but `available()` says so.

#include "ops/sparse_moe/fp8_sm90/fp8_moe_sm90.h"

#include <stdexcept>

namespace sinfer::ops::detail::fp8_moe_sm90 {
namespace {

[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "fp8_moe_sm90: FP8 routed experts need the sm_90a architecture in SUROGATE_SERVE_CUDA_ARCHS "
        "and an H100/H200");
}

} // namespace

bool available() noexcept { return false; }

bool supports(const Geometry&) noexcept { return false; }

Tile resolved_tile(const Geometry&, std::int32_t) { unavailable(); }

std::size_t workspace_bytes(const Geometry&, std::int32_t) { unavailable(); }

void run(const Geometry&, const __nv_bfloat16*, std::int32_t, const std::int32_t*,
         const std::int32_t*, const Fp8RoutedExperts&, void*, std::size_t, __nv_bfloat16*,
         __nv_bfloat16*, cudaStream_t, Tile) {
    unavailable();
}

} // namespace sinfer::ops::detail::fp8_moe_sm90
