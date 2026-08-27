#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_split.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4GdnInputGeometry;

using M32N64            = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

template <class Schedule>
void launch_gemm(const Weight& weight, Nvfp4GdnInputOutput output,
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

void launch_mma(const Weight& weight, Nvfp4GdnInputOutput output,
                Nvfp4W4a4MaterializedActivation activation, std::int32_t tokens,
                cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<M32N64>(weight, output, activation, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<M32N128>(weight, output, activation, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<M128N128Pipelined>(weight, output, activation, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<M64N128>(weight, output, activation, tokens, stream);
    } else {
        launch_gemm<M128N128Resident>(weight, output, activation, tokens, stream);
    }
}

} // namespace

void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (nvfp4_cublaslt_route(tokens)) {
        launch_nvfp4_w4a4_quantize(x, weight, workspace, stream, Nvfp4ScaleLayout::Tiled);
        nvfp4_cublaslt_gemm(weight, 0, Nvfp4GdnInputOutput::kQkvRows, workspace.codes,
                            workspace.scales, static_cast<__nv_bfloat16*>(qkv.data),
                            Nvfp4GdnInputOutput::kQkvRows, tokens, 0.0F, stream);
        nvfp4_cublaslt_gemm(weight, Nvfp4GdnInputOutput::kQkvRows, Nvfp4GdnInputOutput::kZRows,
                            workspace.codes, workspace.scales, static_cast<__nv_bfloat16*>(z.data),
                            Nvfp4GdnInputOutput::kZRows, tokens, 0.0F, stream);
        return;
    }
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    const Nvfp4W4a4TmaSplit split = nvfp4_w4a4_tma_split(tokens);
    if (split.tma_tokens > 0) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_gdn(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(qkv.data),
            static_cast<__nv_bfloat16*>(z.data), split.tma_tokens, alpha, stream);
    }
    if (split.tail_tokens > 0) {
        const Nvfp4W4a4Workspace tail =
            nvfp4_w4a4_workspace_at(workspace, Geometry::kInputRows, split.tma_tokens);
        launch_mma(weight,
                   Nvfp4GdnInputOutput{
                       nvfp4_w4a4_column(qkv.data, Nvfp4GdnInputOutput::kQkvRows, split.tma_tokens),
                       nvfp4_w4a4_column(z.data, Nvfp4GdnInputOutput::kZRows, split.tma_tokens)},
                   Nvfp4W4a4MaterializedActivation{tail.codes, tail.scales}, split.tail_tokens,
                   stream);
    }
}

} // namespace ninfer::ops::detail
