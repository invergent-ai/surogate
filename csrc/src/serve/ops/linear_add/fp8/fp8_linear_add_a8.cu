#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/kernel/func_attribute.cuh"
#include "ops/linear/fp8/fp8_cublaslt.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

template <class Geometry, class Schedule, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    auto* output            = static_cast<__nv_bfloat16*>(residual.data);
    const Fp8AddResidualEpilogue epilogue{output, Geometry::kOutputRows};
    const Fp8ContiguousOutput destination{output, Geometry::kOutputRows};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        CUDA_CHECK(::sinfer::ops::set_func_attribute_per_device(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8AddResidualEpilogue,
                           Fp8ContiguousOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes));
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, epilogue, destination);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem(const Weight& weight, Tensor& residual, Fp8A8Workspace workspace,
                    std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 32) {
        using Batch = Fp8LinearA8BatchSchedule;
        if ((tokens % Batch::kBlockTokens) == 0) {
            launch_mma<Geometry, Batch, true>(weight, residual, workspace, tokens, stream);
        } else {
            launch_mma<Geometry, Batch, false>(weight, residual, workspace, tokens, stream);
        }
        return;
    }
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    if ((tokens % Schedule::kBlockTokens) == 0) {
        launch_mma<Geometry, Schedule, true>(weight, residual, workspace, tokens, stream);
    } else {
        launch_mma<Geometry, Schedule, false>(weight, residual, workspace, tokens, stream);
    }
}

} // namespace

void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(
        workspace, x.ne[1], weight.k, fp8_cublaslt_route(x.ne[1]) ? weight.n : 0);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    if (scratch.staging != nullptr && fp8_cublaslt_route(x.ne[1])) {
        fp8_cublaslt_gemm(weight, 0, weight.n, scratch.codes, scratch.staging, x.ne[1], stream);
        fp8_cublaslt_finish(scratch.staging, static_cast<const __nv_bfloat16*>(weight.scales), 0,
                            weight.n, scratch.scales, x.ne[1],
                            static_cast<__nv_bfloat16*>(residual.data), weight.n, true, stream);
        return;
    }
    if (!is_fp8_registered_problem(weight.n, weight.k)) {
        // The router sends an unregistered shape here only at the widths the cuBLASLt GEMM
        // above takes; there is no in-house A8 kernel for it.
        throw std::invalid_argument(
            "fp8 linear_add A8: an unregistered shape has no A8 kernel without the cuBLASLt route");
    }
    switch (resolve_fp8_problem(weight.n, weight.k)) {
    case Fp8Problem::Residual6144:
        launch_problem<Fp8Residual6144Geometry>(weight, residual, scratch, x.ne[1], stream);
        return;
    case Fp8Problem::Residual17408:
        launch_problem<Fp8Residual17408Geometry>(weight, residual, scratch, x.ne[1], stream);
        return;
    case Fp8Problem::AttnInput:
    case Fp8Problem::GdnInput:
    case Fp8Problem::MlpGateUp:
    case Fp8Problem::Vocabulary:
        break;
    }
    throw std::invalid_argument("fp8 linear_add: unsupported problem");
}

} // namespace sinfer::ops::detail
