// The decode kernels of the Qwen3.8-Flash-Next student mixtures (192, 160 and 128 routed experts;
// api/ops/sparse_moe.h, kSparseMoeFlashNext*Geometry), compiled in a translation unit of their own
// so they build beside sparse_moe_decode_kernels.cu rather than after it. The dispatchers there
// forward every student geometry here.

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
#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/ggml/ggml_moe_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

namespace sinfer::ops::detail {

// The same constants sparse_moe_decode_kernels.cu reads off each registered mixture.
#define SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(Registered)                                           \
    constexpr SparseMoeGeometry kGeometry = (Registered);                                          \
    constexpr int kHidden                 = kGeometry.hidden;                                      \
    constexpr int kExperts                = kGeometry.experts;                                     \
    constexpr int kRouterRows             = kGeometry.router_rows();                               \
    constexpr int kTopK                   = kGeometry.experts_per_token;                           \
    constexpr int kIntermediate           = kGeometry.intermediate;                                \
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
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_flash_next_192

namespace geometry_flash_next_160 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeFlashNext160Geometry)
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_flash_next_160

namespace geometry_flash_next_128 {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(kSparseMoeFlashNext128Geometry)
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"
} // namespace geometry_flash_next_128

namespace {
[[noreturn]] void not_a_student() {
    throw std::invalid_argument("sparse_moe: geometry is not a Flash-Next student mixture");
}
} // namespace

bool sparse_moe_decode_students_launch_expert_major(
    const SparseMoeGeometry& geometry, const Tensor& x, const PreparedSparseMoeWeights& weights,
    Tensor& destination, const int* token_ids, const float* token_alpha, const float* shared_scale,
    float* token_activations, int* groups, std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
    cudaStream_t stream) {
#define SINFER_STUDENT_CALL(ns)                                                                    \
    return ns::decode_launch_expert_major(x, weights, destination, token_ids, token_alpha,         \
                                          shared_scale, token_activations, groups, tokens,         \
                                          schedule, stream)
    if (geometry == kSparseMoeFlashNext192Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_192); }
    if (geometry == kSparseMoeFlashNext160Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_160); }
    if (geometry == kSparseMoeFlashNext128Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_128); }
#undef SINFER_STUDENT_CALL
    not_a_student();
}

void sparse_moe_decode_students_launch_d3_small_t(
    const SparseMoeGeometry& geometry, const Tensor& x, const PreparedSparseMoeWeights& weights,
    const int* token_ids, float* token_activations, std::int32_t tokens,
    SparseMoeSmallTD3Schedule schedule, cudaStream_t stream, const int* adaptive_route_jobs) {
#define SINFER_STUDENT_CALL(ns)                                                                    \
    ns::decode_launch_d3_small_t(x, weights, token_ids, token_activations, tokens, schedule,       \
                                 stream, adaptive_route_jobs);                                     \
    return
    if (geometry == kSparseMoeFlashNext192Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_192); }
    if (geometry == kSparseMoeFlashNext160Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_160); }
    if (geometry == kSparseMoeFlashNext128Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_128); }
#undef SINFER_STUDENT_CALL
    not_a_student();
}

void sparse_moe_decode_students_launch_d4_small_t(
    const SparseMoeGeometry& geometry, const PreparedSparseMoeWeights& weights, Tensor& destination,
    const int* token_ids, const float* token_alpha, const float* shared_scale,
    const float* token_activations, std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
    cudaStream_t stream, const int* adaptive_route_jobs) {
#define SINFER_STUDENT_CALL(ns)                                                                    \
    ns::decode_launch_d4_small_t(weights, destination, token_ids, token_alpha, shared_scale,       \
                                 token_activations, tokens, schedule, stream,                      \
                                 adaptive_route_jobs);                                             \
    return
    if (geometry == kSparseMoeFlashNext192Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_192); }
    if (geometry == kSparseMoeFlashNext160Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_160); }
    if (geometry == kSparseMoeFlashNext128Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_128); }
#undef SINFER_STUDENT_CALL
    not_a_student();
}

void sparse_moe_decode_students_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                                       const Tensor& router_x,
                                       const PreparedSparseMoeWeights& weights,
                                       Tensor& destination,
                                       const SparseMoeDecodeWorkspace& workspace,
                                       cudaStream_t stream, const SparseMoeRoundHook* hook,
                                       const LoraBank* adapters,
                                       const std::int32_t* adapter_slot) {
#define SINFER_STUDENT_CALL(ns)                                                                    \
    ns::decode_launch(x, router_x, weights, destination, workspace, stream, hook, adapters,        \
                      adapter_slot);                                                               \
    return
    if (geometry == kSparseMoeFlashNext192Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_192); }
    if (geometry == kSparseMoeFlashNext160Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_160); }
    if (geometry == kSparseMoeFlashNext128Geometry) { SINFER_STUDENT_CALL(geometry_flash_next_128); }
#undef SINFER_STUDENT_CALL
    not_a_student();
}

} // namespace sinfer::ops::detail
