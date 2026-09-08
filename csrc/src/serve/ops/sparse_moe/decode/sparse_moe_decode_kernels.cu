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
#include "ops/linear/ggml/ggml_moe_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace sinfer::ops::detail {

// One block per registered mixture. Every constant is read off the geometry rather than
// restated, so a mixture without an always-on expert is described by its own numbers -- one
// fewer router row, one fewer path per token -- and not by a comment saying so.
#define SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(Registered)                                           \
    constexpr SparseMoeGeometry kGeometry = (Registered);                                          \
    constexpr int kHidden                 = kGeometry.hidden;                                      \
    constexpr int kExperts                = kGeometry.experts;                                     \
    constexpr int kRouterRows             = kGeometry.router_rows();                               \
    constexpr int kTopK                   = kGeometry.experts_per_token;                           \
    constexpr int kIntermediate           = kGeometry.intermediate;                                \
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
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_qwen36

namespace geometry_flash_next {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeFlashNextGeometry)
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_flash_next

namespace geometry_qwen3_moe {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeQwen3MoeGeometry)
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_qwen3_moe

namespace geometry_glm53 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeGlm53Geometry)
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_glm53

namespace geometry_gemma4 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeGemma4Geometry)
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_gemma4

namespace geometry_lfm2_moe32 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeLfm2Moe32Geometry)
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_lfm2_moe32

namespace geometry_lfm2_moe64 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeLfm2Moe64Geometry)
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_lfm2_moe64

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
    if (geometry == kSparseMoeQwen3MoeGeometry) {
        geometry_qwen3_moe::decode_launch_d3_small_t(x, weights, token_ids, token_activations,
                                                     tokens, schedule, stream,
                                                     adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeGlm53Geometry) {
        geometry_glm53::decode_launch_d3_small_t(x, weights, token_ids, token_activations,
                                                     tokens, schedule, stream,
                                                     adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeGemma4Geometry) {
        geometry_gemma4::decode_launch_d3_small_t(x, weights, token_ids, token_activations,
                                                     tokens, schedule, stream,
                                                     adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe32Geometry) {
        geometry_lfm2_moe32::decode_launch_d3_small_t(x, weights, token_ids, token_activations,
                                                     tokens, schedule, stream,
                                                     adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe64Geometry) {
        geometry_lfm2_moe64::decode_launch_d3_small_t(x, weights, token_ids, token_activations,
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
    if (geometry == kSparseMoeQwen3MoeGeometry) {
        geometry_qwen3_moe::decode_launch_d4_small_t(weights, destination, token_ids, token_alpha,
                                                     shared_scale, token_activations, tokens,
                                                     schedule, stream, adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeGlm53Geometry) {
        geometry_glm53::decode_launch_d4_small_t(weights, destination, token_ids, token_alpha,
                                                     shared_scale, token_activations, tokens,
                                                     schedule, stream, adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeGemma4Geometry) {
        geometry_gemma4::decode_launch_d4_small_t(weights, destination, token_ids, token_alpha,
                                                     shared_scale, token_activations, tokens,
                                                     schedule, stream, adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe32Geometry) {
        geometry_lfm2_moe32::decode_launch_d4_small_t(weights, destination, token_ids, token_alpha,
                                                     shared_scale, token_activations, tokens,
                                                     schedule, stream, adaptive_route_jobs);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe64Geometry) {
        geometry_lfm2_moe64::decode_launch_d4_small_t(weights, destination, token_ids, token_alpha,
                                                     shared_scale, token_activations, tokens,
                                                     schedule, stream, adaptive_route_jobs);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry has no compiled decode kernels");
}

void sparse_moe_decode_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                              const Tensor& router_x, const SparseMoeWeights& weights,
                              Tensor& destination, const SparseMoeDecodeWorkspace& workspace,
                              cudaStream_t stream, const SparseMoeRoundHook* hook) {
    if (geometry == kSparseMoeQwen36Geometry) {
        geometry_qwen36::decode_launch(x, router_x, weights, destination, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeFlashNextGeometry) {
        geometry_flash_next::decode_launch(x, router_x, weights, destination, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeQwen3MoeGeometry) {
        geometry_qwen3_moe::decode_launch(x, router_x, weights, destination, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeGlm53Geometry) {
        geometry_glm53::decode_launch(x, router_x, weights, destination, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeGemma4Geometry) {
        geometry_gemma4::decode_launch(x, router_x, weights, destination, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe32Geometry) {
        geometry_lfm2_moe32::decode_launch(x, router_x, weights, destination, workspace, stream, hook);
        return;
    }
    if (geometry == kSparseMoeLfm2Moe64Geometry) {
        geometry_lfm2_moe64::decode_launch(x, router_x, weights, destination, workspace, stream, hook);
        return;
    }
    throw std::invalid_argument("sparse_moe: geometry has no compiled decode kernels");
}

} // namespace sinfer::ops::detail
