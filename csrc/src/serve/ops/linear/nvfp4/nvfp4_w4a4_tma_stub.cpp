// SPDX-License-Identifier: Apache-2.0
// surogate vendor patch (csrc/src/serve/PATCHES.md): W4A4 TMA stub.
//
// The NVFP4 W4A4 prefill kernels use TMA (cp.async.bulk.tensor), mbarrier,
// cluster launch and block-scaled FP4 MMA — sm_90+/sm_120 hardware that does
// not exist on sm_89 (Ada has no FP4 tensor cores). When the build targets an
// arch set without 120a, this stub replaces the two TMA translation units so
// the rest of the engine links and runs; reaching a W4A4 route on such a
// build is a dispatch/admission bug and fails loudly.

#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <cstdio>
#include <cstdlib>

namespace ninfer::ops::detail {

namespace {
[[noreturn]] void w4a4_unavailable(const char* which) {
    std::fprintf(stderr,
                 "ninfer: %s requires NVFP4 W4A4 tensor-core hardware (sm_120); "
                 "this build targets an architecture without it. NVFP4 W4A4 "
                 "artifacts are not servable on this GPU.\n",
                 which);
    std::abort();
}
} // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    w4a4_unavailable("launch_nvfp4_w4a4_tma_linear");
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                                     float, cudaStream_t) {
    w4a4_unavailable("launch_nvfp4_w4a4_tma_attention");
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) {
    w4a4_unavailable("launch_nvfp4_w4a4_tma_gdn");
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    w4a4_unavailable("launch_nvfp4_w4a4_tma_linear_add");
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    w4a4_unavailable("launch_nvfp4_linear_swiglu_w4a4_tma");
}

} // namespace ninfer::ops::detail
