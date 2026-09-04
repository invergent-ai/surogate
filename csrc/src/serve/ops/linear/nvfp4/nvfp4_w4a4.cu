#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_split.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace sinfer::ops::detail {
namespace {

using M32N64                      = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128                     = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128                     = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined           = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident            = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Nvfp4ContiguousOutput output,
                 Nvfp4W4a4MaterializedActivation activation, std::int32_t tokens,
                 cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                           cudaStream_t stream, Nvfp4ScaleLayout layout) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * ActivationGeometry::kGroupsPerRow;
    const dim3 grid((tasks + kThreads - 1) / kThreads);
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    if (layout == Nvfp4ScaleLayout::Tiled) {
        nvfp4_w4a4_quantize_kernel<ActivationGeometry, kThreads, true><<<grid, kThreads, 0, stream>>>(
            input, workspace.codes, workspace.scales, tokens, weight.input_scale_divisor);
    } else {
        nvfp4_w4a4_quantize_kernel<ActivationGeometry, kThreads, false><<<grid, kThreads, 0, stream>>>(
            input, workspace.codes, workspace.scales, tokens, weight.input_scale_divisor);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_mma(const Weight& weight, Nvfp4ContiguousOutput output,
                Nvfp4W4a4MaterializedActivation activation, std::int32_t tokens,
                cudaStream_t stream) {
    constexpr bool kResidualGeometry = std::is_same_v<Geometry, Nvfp4Residual6144Geometry> ||
                                       std::is_same_v<Geometry, Nvfp4Residual17408Geometry>;
    if (tokens <= 64) {
        launch_gemm<Geometry, M32N64>(weight, output, activation, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<Geometry, M32N128>(weight, output, activation, tokens, stream);
    } else if (tokens <= 128) {
        if constexpr (kResidualGeometry) {
            launch_gemm<Geometry, M32N128>(weight, output, activation, tokens, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, output, activation, tokens, stream);
        }
    } else if (tokens <= 192) {
        launch_gemm<Geometry, M64N128>(weight, output, activation, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<Geometry, M128N128Resident>(weight, output, activation, tokens, stream);
    } else if (tokens <= 512) {
        if constexpr (Geometry::kOutputRows == Nvfp4GdnInputGeometry::kOutputRows) {
            launch_gemm<Geometry, M128N128Resident>(weight, output, activation, tokens, stream);
        } else {
            launch_gemm<Geometry, M128N128Pipelined>(weight, output, activation, tokens, stream);
        }
    } else {
        launch_gemm<Geometry, M128N128Resident>(weight, output, activation, tokens, stream);
    }
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    const Nvfp4W4a4TmaSplit split = nvfp4_w4a4_tma_split(tokens);
    if (split.tma_tokens > 0) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_linear(
            resolve_nvfp4_problem(Geometry::kOutputRows, Geometry::kInputRows), workspace.codes,
            workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            split.tma_tokens, alpha, stream);
    }
    if (split.tail_tokens > 0) {
        const Nvfp4W4a4Workspace tail =
            nvfp4_w4a4_workspace_at(workspace, Geometry::kInputRows, split.tma_tokens);
        launch_mma<Geometry>(
            weight,
            Nvfp4ContiguousOutput{
                nvfp4_w4a4_column(out.data, Geometry::kOutputRows, split.tma_tokens),
                Geometry::kOutputRows},
            Nvfp4W4a4MaterializedActivation{tail.codes, tail.scales}, split.tail_tokens, stream);
    }
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream, Nvfp4ScaleLayout layout) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 requires caller workspace");
    }
    switch (weight.k) {
#define SINFER_NVFP4_K_CASE(K)                                                                    \
    case K:                                                                                       \
        launch_quantize_exact<Nvfp4ActivationGeometry<K>>(x, weight, workspace, stream, layout);  \
        return;
        SINFER_NVFP4_FOR_EACH_ACTIVATION_K(SINFER_NVFP4_K_CASE)
#undef SINFER_NVFP4_K_CASE
    default:
        throw std::invalid_argument("nvfp4 W4A4 quantize: unsupported K");
    }
}

void launch_nvfp4_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                       Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    // Shapes outside the registered geometries have no in-house ladder - the mma and TMA
    // schedules are templated on Geometry - so they run on cuBLASLt at every width (#82).
    if (is_nvfp4_generic_problem(weight.n, weight.k)) {
        launch_nvfp4_w4a4_quantize(x, weight, workspace, stream, Nvfp4ScaleLayout::Tiled);
        nvfp4_cublaslt_gemm(weight, 0, weight.n, workspace.codes, workspace.scales,
                            static_cast<__nv_bfloat16*>(out.data), weight.n, tokens, 0.0F, stream);
        return;
    }
    if (nvfp4_cublaslt_route(tokens) || is_nvfp4_generic_problem(weight.n, weight.k)) {
        launch_nvfp4_w4a4_quantize(x, weight, workspace, stream, Nvfp4ScaleLayout::Tiled);
        nvfp4_cublaslt_gemm(weight, 0, weight.n, workspace.codes, workspace.scales,
                            static_cast<__nv_bfloat16*>(out.data), weight.n, tokens, 0.0F, stream);
        return;
    }
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::AttnInput:
        launch_problem<Nvfp4AttnInputGeometry>(weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_problem<Nvfp4GdnInputGeometry>(weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_problem<Nvfp4MlpGateUpGeometry>(weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launch_problem<Nvfp4Residual6144Geometry>(weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_problem<Nvfp4Residual17408Geometry>(weight, out, workspace, tokens, stream);
        return;
    }
}

} // namespace sinfer::ops::detail
