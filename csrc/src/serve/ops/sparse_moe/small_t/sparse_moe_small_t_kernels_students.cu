// The small-T kernels of the Qwen3.8-Flash-Next student mixtures (192, 160 and 128 routed experts;
// api/ops/sparse_moe.h, kSparseMoeFlashNext*Geometry), in a translation unit of their own so they
// build beside sparse_moe_small_t_kernels.cu, whose dispatcher forwards those geometries here.

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

// The same constants sparse_moe_small_t_kernels.cu reads off each registered mixture.
#define SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(Registered)                                           \
    constexpr SparseMoeGeometry kGeometry = (Registered);                                          \
    constexpr int kHidden                 = kGeometry.hidden;                                      \
    constexpr int kExperts                = kGeometry.experts;                                     \
    constexpr int kRouterRows             = kGeometry.router_rows();                               \
    constexpr int kTopK                   = kGeometry.experts_per_token;                           \
    constexpr bool kHasShared             = kGeometry.has_shared();                                \
    constexpr int kPaths                  = kGeometry.paths();                                     \
    constexpr SparseMoeGating kGating     = kGeometry.gating;                                      \
    constexpr float kRoutedScale          = kGeometry.routed_scale;                                \
    constexpr bool kSharedGated           = kGeometry.shared_gated;                                \
    constexpr float kSwigluLimit          = kGeometry.swiglu_limit;                                \
    constexpr GatedActivation kActivation = kGeometry.activation;                                  \
    constexpr bool kPerExpertScaled       = kGeometry.per_expert_scaled;

namespace geometry_flash_next_192 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeFlashNext192Geometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_flash_next_192

namespace geometry_flash_next_160 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeFlashNext160Geometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_flash_next_160

namespace geometry_flash_next_128 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeFlashNext128Geometry)
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_flash_next_128

void sparse_moe_small_t_students_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                                        const Tensor& router_x,
                                        const PreparedSparseMoeWeights& weights,
                                        Tensor& destination, const SparseMoeSmallTPlan& plan,
                                        const SparseMoeSmallTWorkspace& workspace,
                                        cudaStream_t stream, const SparseMoeRoundHook* hook) {
    if (geometry == kSparseMoeFlashNext192Geometry) {
        geometry_flash_next_192::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeFlashNext160Geometry) {
        geometry_flash_next_160::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeFlashNext128Geometry) {
        geometry_flash_next_128::small_t_launch(x, router_x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry is not a Flash-Next student mixture");
}

} // namespace sinfer::ops::detail
