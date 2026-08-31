#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/kernel/func_attribute.cuh"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops::detail {
namespace {

using Geometry = Fp8MlpGateUpGeometry;
using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;

constexpr int kIntermediate = Geometry::kOutputRows / 2;
using Rows                  = Fp8SwiGluRows<Schedule::kBlockRows / 2, kIntermediate>;
static_assert((Schedule::kBlockRows % 2) == 0);

template <class Sched, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& out, Fp8A8Workspace workspace, std::int32_t tokens,
                cudaStream_t stream) {
    constexpr int kRowTiles = Geometry::kOutputRows / Sched::kBlockRows;
    const int token_tiles   = (tokens + Sched::kBlockTokens - 1) / Sched::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Rows rows{};
    const Fp8SwiGluOutput output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};

    if constexpr (Sched::kSharedBytes > 48 * 1024) {
        CUDA_CHECK(::sinfer::ops::set_func_attribute_per_device(
            fp8_mma_kernel<Geometry, Sched, FullTokens, Fp8IdentityEpilogue, Fp8SwiGluOutput,
                           Rows, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Sched::kSharedBytes));
    }
    fp8_mma_kernel<Geometry, Sched, FullTokens, Fp8IdentityEpilogue, Fp8SwiGluOutput, Rows, true>
        <<<blocks, Sched::kThreads, Sched::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{}, output,
            rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    const Fp8A8Workspace scratch =
        allocate_fp8_a8_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    if (x.ne[1] <= 32) {
        if ((x.ne[1] % Fp8LinearA8BatchSchedule::kBlockTokens) == 0) {
            launch_mma<Fp8LinearA8BatchSchedule, true>(weight, out, scratch, x.ne[1], stream);
        } else {
            launch_mma<Fp8LinearA8BatchSchedule, false>(weight, out, scratch, x.ne[1], stream);
        }
    } else if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<Schedule, true>(weight, out, scratch, x.ne[1], stream);
    } else {
        launch_mma<Schedule, false>(weight, out, scratch, x.ne[1], stream);
    }
}

} // namespace sinfer::ops::detail
