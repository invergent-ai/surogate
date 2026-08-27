#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_cublaslt.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8GdnInputGeometry;
using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;

static_assert((Fp8GdnInputOutput::kQkvRows % Schedule::kBlockRows) == 0);
static_assert((Fp8GdnInputOutput::kZRows % Schedule::kBlockRows) == 0);

template <class Sched, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& qkv, Tensor& z, Fp8A8Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = Geometry::kOutputRows / Sched::kBlockRows;
    const int token_tiles   = (tokens + Sched::kBlockTokens - 1) / Sched::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};

    if constexpr (Sched::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_mma_kernel<Geometry, Sched, FullTokens, Fp8IdentityEpilogue, Fp8GdnInputOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Sched::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_mma_kernel<Geometry, Sched, FullTokens>
        <<<blocks, Sched::kThreads, Sched::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    if (workspace.staging != nullptr && fp8_cublaslt_route(x.ne[1])) {
        const std::int32_t tokens = x.ne[1];
        const auto* row_scales    = static_cast<const __nv_bfloat16*>(weight.scales);
        fp8_cublaslt_gemm(weight, 0, Fp8GdnInputOutput::kQkvRows, workspace.codes,
                          workspace.staging, tokens, stream);
        fp8_cublaslt_finish(workspace.staging, row_scales, 0, Fp8GdnInputOutput::kQkvRows,
                            workspace.scales, tokens, static_cast<__nv_bfloat16*>(qkv.data),
                            Fp8GdnInputOutput::kQkvRows, false, stream);
        fp8_cublaslt_gemm(weight, Fp8GdnInputOutput::kQkvRows, Fp8GdnInputOutput::kZRows,
                          workspace.codes, workspace.staging, tokens, stream);
        fp8_cublaslt_finish(workspace.staging, row_scales, Fp8GdnInputOutput::kQkvRows,
                            Fp8GdnInputOutput::kZRows, workspace.scales, tokens,
                            static_cast<__nv_bfloat16*>(z.data), Fp8GdnInputOutput::kZRows, false,
                            stream);
        return;
    }
    if (x.ne[1] <= 32) {
        if ((x.ne[1] % Fp8LinearA8BatchSchedule::kBlockTokens) == 0) {
            launch_mma<Fp8LinearA8BatchSchedule, true>(weight, qkv, z, workspace, x.ne[1], stream);
        } else {
            launch_mma<Fp8LinearA8BatchSchedule, false>(weight, qkv, z, workspace, x.ne[1], stream);
        }
    } else if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<Schedule, true>(weight, qkv, z, workspace, x.ne[1], stream);
    } else {
        launch_mma<Schedule, false>(weight, qkv, z, workspace, x.ne[1], stream);
    }
}

} // namespace ninfer::ops::detail
