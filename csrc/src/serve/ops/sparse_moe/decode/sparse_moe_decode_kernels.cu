#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace geometry_qwen36 {
constexpr int kHidden       = 2048;
constexpr int kExperts      = 256;
constexpr int kRouterRows   = kExperts + 1;
constexpr int kTopK         = 8;
constexpr int kIntermediate = 512;
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_qwen36

namespace geometry_flash_next {
constexpr int kHidden       = 2560;
constexpr int kExperts      = 512;
constexpr int kRouterRows   = kExperts + 1;
constexpr int kTopK         = 10;
constexpr int kIntermediate = 640;
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_flash_next

void sparse_moe_decode_launch_d3_small_t(const SparseMoeGeometry& geometry, const Tensor& x,
                                         const SparseMoeWeights& weights, const int* token_ids,
                                         float* token_activations, std::int32_t tokens,
                                         SparseMoeSmallTD3Schedule schedule, cudaStream_t stream,
                                         const int* adaptive_route_jobs) {
    if (geometry == kSparseMoeQwen36Geometry) {
        geometry_qwen36::decode_launch_d3_small_t(x, weights, token_ids, token_activations,
                                                  tokens, schedule, stream, adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeFlashNextGeometry) {
        geometry_flash_next::decode_launch_d3_small_t(x, weights, token_ids, token_activations,
                                                      tokens, schedule, stream,
                                                      adaptive_route_jobs);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry has no compiled decode kernels");
}

void sparse_moe_decode_launch_d4_small_t(const SparseMoeGeometry& geometry,
                                         const SparseMoeWeights& weights, Tensor& destination,
                                         const int* token_ids, const float* token_alpha,
                                         const float* shared_scale, const float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                                         cudaStream_t stream, const int* adaptive_route_jobs) {
    if (geometry == kSparseMoeQwen36Geometry) {
        geometry_qwen36::decode_launch_d4_small_t(weights, destination, token_ids, token_alpha,
                                                  shared_scale, token_activations, tokens,
                                                  schedule, stream, adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeFlashNextGeometry) {
        geometry_flash_next::decode_launch_d4_small_t(weights, destination, token_ids,
                                                      token_alpha, shared_scale,
                                                      token_activations, tokens, schedule,
                                                      stream, adaptive_route_jobs);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry has no compiled decode kernels");
}

void sparse_moe_decode_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                              const SparseMoeWeights& weights, Tensor& destination,
                              const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    if (geometry == kSparseMoeQwen36Geometry) {
        geometry_qwen36::decode_launch(x, weights, destination, workspace, stream);
        return;
    }
    if (geometry == kSparseMoeFlashNextGeometry) {
        geometry_flash_next::decode_launch(x, weights, destination, workspace, stream);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry has no compiled decode kernels");
}

} // namespace ninfer::ops::detail
