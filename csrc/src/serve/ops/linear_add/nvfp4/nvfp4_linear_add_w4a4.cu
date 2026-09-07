#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_split.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

using M32N64            = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, __nv_bfloat16* output,
                 Nvfp4W4a4MaterializedActivation activation, std::int32_t tokens,
                 cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha,
        Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_mma(const Weight& weight, __nv_bfloat16* output,
                Nvfp4W4a4MaterializedActivation activation, std::int32_t tokens,
                cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<Geometry, M32N64>(weight, output, activation, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<Geometry, M32N128>(weight, output, activation, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<Geometry, M64N128>(weight, output, activation, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<Geometry, M128N128Resident>(weight, output, activation, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<Geometry, M128N128Pipelined>(weight, output, activation, tokens, stream);
    } else {
        launch_gemm<Geometry, M128N128Resident>(weight, output, activation, tokens, stream);
    }
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Nvfp4W4a4Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    const Nvfp4W4a4TmaSplit split = nvfp4_w4a4_tma_split(tokens);
    if (split.tma_tokens > 0) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_linear_add(
            resolve_nvfp4_problem(Geometry::kOutputRows, Geometry::kInputRows), workspace.codes,
            workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(residual.data), split.tma_tokens, alpha, stream);
    }
    if (split.tail_tokens > 0) {
        const Nvfp4W4a4Workspace tail =
            nvfp4_w4a4_workspace_at(workspace, Geometry::kInputRows, split.tma_tokens);
        launch_mma<Geometry>(
            weight, nvfp4_w4a4_column(residual.data, Geometry::kOutputRows, split.tma_tokens),
            Nvfp4W4a4MaterializedActivation{tail.codes, tail.scales}, split.tail_tokens, stream);
    }
}

} // namespace

bool nvfp4_linear_add_w4a4_wide(const Weight& weight, std::int32_t tokens) {
    return (nvfp4_cublaslt_route(tokens) || is_nvfp4_generic_problem(weight.n, weight.k)) &&
           is_nvfp4_linear_problem(weight.n, weight.k);
}

void nvfp4_linear_add_w4a4_wide_gemm(const Weight& weight, Nvfp4W4a4Workspace workspace,
                                     Tensor& residual, std::int32_t tokens, cudaStream_t stream) {
    if (!nvfp4_linear_add_w4a4_wide(weight, tokens)) {
        throw std::invalid_argument("nvfp4 linear_add: the wide GEMM does not serve this width");
    }
    nvfp4_cublaslt_gemm(weight, 0, weight.n, workspace.codes, workspace.scales,
                        static_cast<__nv_bfloat16*>(residual.data), weight.n, tokens, 1.0F, stream);
}

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (nvfp4_linear_add_w4a4_wide(weight, tokens)) {
        launch_nvfp4_w4a4_quantize(x, weight, workspace, stream, Nvfp4ScaleLayout::Tiled);
        nvfp4_linear_add_w4a4_wide_gemm(weight, workspace, residual, tokens, stream);
        return;
    }
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::Residual6144:
        launch_problem<Nvfp4Residual6144Geometry>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_problem<Nvfp4Residual17408Geometry>(weight, residual, workspace, tokens, stream);
        return;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace sinfer::ops::detail
