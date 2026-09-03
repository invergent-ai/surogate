#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/linear/ggml/ggml_prefill_codec.cuh"
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
#include "ops/sparse_moe/marlin/marlin_moe_gemm.h"

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <stdexcept>

namespace sinfer::ops::detail {

namespace geometry_qwen36 {
constexpr SparseMoeGeometry kGeometry = kSparseMoeQwen36Geometry;
constexpr int kHidden                 = kGeometry.hidden;
constexpr int kExperts                = kGeometry.experts;
constexpr int kRouterRows             = kExperts + 1;
constexpr int kTopK                   = kGeometry.experts_per_token;
constexpr int kIntermediate           = kGeometry.intermediate;
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_qwen36

namespace geometry_flash_next {
constexpr SparseMoeGeometry kGeometry = kSparseMoeFlashNextGeometry;
constexpr int kHidden                 = kGeometry.hidden;
constexpr int kExperts                = kGeometry.experts;
constexpr int kRouterRows             = kExperts + 1;
constexpr int kTopK                   = kGeometry.experts_per_token;
constexpr int kIntermediate           = kGeometry.intermediate;
#include "ops/sparse_moe/prefill/sparse_moe_prefill_body.inc"
} // namespace geometry_flash_next

void sparse_moe_prefill_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                               const SparseMoeWeights& weights, Tensor& destination,
                               const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, cudaStream_t stream,
                               const SparseMoeRoundHook* hook) {
    if (geometry == kSparseMoeQwen36Geometry) {
        geometry_qwen36::prefill_launch(x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeFlashNextGeometry) {
        geometry_flash_next::prefill_launch(x, weights, destination, plan, workspace, stream, hook);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry has no compiled prefill kernels");
}

} // namespace sinfer::ops::detail
