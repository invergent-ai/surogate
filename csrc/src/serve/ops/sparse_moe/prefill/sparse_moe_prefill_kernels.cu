#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/ggml/ggml_i8_tile.cuh"
#include "ops/linear/ggml/ggml_prefill_codec.cuh"
#include "ops/linear/ggml/ggml_q8_1.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "ops/sparse_moe/marlin/marlin_moe_gemm.h"

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <stdexcept>

namespace sinfer::ops::detail {

// One block per registered mixture; every constant is read off the geometry rather than
// restated, so a mixture without an always-on expert is described by its own numbers.
#define SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(Registered)                                           \
    constexpr SparseMoeGeometry kGeometry = (Registered);                                          \
    constexpr int kHidden                 = kGeometry.hidden;                                      \
    constexpr int kExperts                = kGeometry.experts;                                     \
    constexpr int kRouterRows             = kGeometry.router_rows();                               \
    constexpr int kTopK                   = kGeometry.experts_per_token;                           \
    constexpr bool kHasShared             = kGeometry.has_shared();                                \
    constexpr int kIntermediate           = kGeometry.intermediate;             \
    constexpr int kPaths                  = kGeometry.paths();                         \
    constexpr SparseMoeGating kGating     = kGeometry.gating;                          \
    constexpr float kRoutedScale          = kGeometry.routed_scale;                    \
    constexpr bool kSharedGated           = kGeometry.shared_gated;                    \
    constexpr float kSwigluLimit          = kGeometry.swiglu_limit;                    \
    constexpr GatedActivation kActivation = kGeometry.activation;                      \
    constexpr bool kPerExpertScaled       = kGeometry.per_expert_scaled;

namespace geometry_qwen36 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeQwen36Geometry)
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_qwen36

namespace geometry_flash_next {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeFlashNextGeometry)
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_flash_next

namespace geometry_qwen3_moe {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeQwen3MoeGeometry)
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_qwen3_moe

namespace geometry_qwen3_moe_235b {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeQwen3Moe235BGeometry)
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_qwen3_moe_235b

namespace geometry_glm53 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeGlm53Geometry)
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_glm53

namespace geometry_gemma4 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeGemma4Geometry)
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_gemma4

namespace geometry_lfm2_moe32 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeLfm2Moe32Geometry)
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_lfm2_moe32

namespace geometry_lfm2_moe64 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeLfm2Moe64Geometry)
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_lfm2_moe64

void sparse_moe_prefill_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                               const Tensor& router_x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, cudaStream_t stream,
                               const SparseMoeRoundHook* hook) {
    if (geometry == kSparseMoeQwen36Geometry) {
        geometry_qwen36::prefill_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeFlashNextGeometry) {
        geometry_flash_next::prefill_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeQwen3MoeGeometry) {
        geometry_qwen3_moe::prefill_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeQwen3Moe235BGeometry) {
        geometry_qwen3_moe_235b::prefill_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeGlm53Geometry) {
        geometry_glm53::prefill_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeGemma4Geometry) {
        geometry_gemma4::prefill_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe32Geometry) {
        geometry_lfm2_moe32::prefill_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe64Geometry) {
        geometry_lfm2_moe64::prefill_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry has no compiled prefill kernels");
}

} // namespace sinfer::ops::detail
