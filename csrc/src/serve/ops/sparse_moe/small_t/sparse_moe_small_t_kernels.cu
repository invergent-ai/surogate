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

namespace ninfer::ops::detail {

namespace geometry_qwen36 {
constexpr SparseMoeGeometry kGeometry = kSparseMoeQwen36Geometry;
constexpr int kHidden                 = kGeometry.hidden;
constexpr int kExperts                = kGeometry.experts;
constexpr int kRouterRows             = kExperts + 1;
constexpr int kTopK                   = kGeometry.experts_per_token;
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_qwen36

namespace geometry_flash_next {
constexpr SparseMoeGeometry kGeometry = kSparseMoeFlashNextGeometry;
constexpr int kHidden                 = kGeometry.hidden;
constexpr int kExperts                = kGeometry.experts;
constexpr int kRouterRows             = kExperts + 1;
constexpr int kTopK                   = kGeometry.experts_per_token;
#include "ops/sparse_moe/small_t/sparse_moe_small_t_body.inc"
} // namespace geometry_flash_next

void sparse_moe_small_t_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                               const SparseMoeWeights& weights, Tensor& destination,
                               const SparseMoeSmallTPlan& plan,
                               const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream,
                               const SparseMoeRoundHook* hook) {
    if (geometry == kSparseMoeQwen36Geometry) {
        geometry_qwen36::small_t_launch(x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeFlashNextGeometry) {
        geometry_flash_next::small_t_launch(x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry has no compiled small-T kernels");
}

} // namespace ninfer::ops::detail
