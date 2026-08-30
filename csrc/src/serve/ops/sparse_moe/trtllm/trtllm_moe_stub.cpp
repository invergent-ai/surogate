#include "ops/sparse_moe/trtllm/trtllm_moe.h"

#include <stdexcept>
#include <string>

// The vendored runner's grouped block-scaled GEMMs are compiled for sm_120a only. On a build
// without that architecture the routed-NVFP4 profile has no kernel at all, so every entry point
// but `available()` says so rather than returning a wrong answer.

namespace ninfer::ops::detail::trtllm_moe {
namespace {

[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "trtllm_moe: the TRT-LLM cutlass MoE runner is not in this build (it needs the sm_120a "
        "architecture in SUROGATE_SERVE_CUDA_ARCHS)");
}

} // namespace

bool available() noexcept { return false; }

std::int32_t bucket_of(std::int32_t) { unavailable(); }

std::size_t workspace_bytes(const Geometry&, std::int32_t) { unavailable(); }

void prepare(const Geometry&, const Nvfp4RoutedExperts&, std::int32_t, cudaStream_t) {
    unavailable();
}

void run(const Geometry&, const __nv_bfloat16*, std::int32_t, const std::int32_t*, const float*,
         const Nvfp4RoutedExperts&, void*, std::size_t, float*, cudaStream_t) {
    unavailable();
}

} // namespace ninfer::ops::detail::trtllm_moe
