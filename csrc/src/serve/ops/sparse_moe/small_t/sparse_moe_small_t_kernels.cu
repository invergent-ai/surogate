#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/sparse_moe_route.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
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
    constexpr int kPaths                  = kGeometry.paths();                         \
    constexpr SparseMoeGating kGating     = kGeometry.gating;                          \
    constexpr float kRoutedScale          = kGeometry.routed_scale;                    \
    constexpr bool kSharedGated           = kGeometry.shared_gated;                    \
    constexpr float kSwigluLimit          = kGeometry.swiglu_limit;                    \
    constexpr GatedActivation kActivation = kGeometry.activation;                      \
    constexpr bool kPerExpertScaled       = kGeometry.per_expert_scaled;

namespace geometry_qwen36 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeQwen36Geometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_qwen36

namespace geometry_flash_next {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeFlashNextGeometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_flash_next

namespace geometry_qwen3_moe {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeQwen3MoeGeometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_qwen3_moe

namespace geometry_glm53 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeGlm53Geometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_glm53

namespace geometry_gemma4 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeGemma4Geometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_gemma4

namespace geometry_lfm2_moe32 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeLfm2Moe32Geometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_lfm2_moe32

namespace geometry_lfm2_moe64 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeLfm2Moe64Geometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_lfm2_moe64

void sparse_moe_small_t_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                               const Tensor& router_x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoeSmallTPlan& plan,
                               const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream,
                               const SparseMoeRoundHook* hook) {
    if (geometry == kSparseMoeQwen36Geometry) {
        geometry_qwen36::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeFlashNextGeometry) {
        geometry_flash_next::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeQwen3MoeGeometry) {
        geometry_qwen3_moe::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeGlm53Geometry) {
        geometry_glm53::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeGemma4Geometry) {
        geometry_gemma4::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe32Geometry) {
        geometry_lfm2_moe32::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe64Geometry) {
        geometry_lfm2_moe64::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry has no compiled small-T kernels");
}

} // namespace sinfer::ops::detail
