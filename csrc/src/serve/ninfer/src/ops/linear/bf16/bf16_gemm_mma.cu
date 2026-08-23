#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Schedule, bool FullTokens>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);

    constexpr int tiles_m = Geometry::kOutputRows / Schedule::kBlockRows;
    const int tiles_n     = div_up(x.ne[1], Schedule::kBlockCols);
    const int blocks      = tiles_m * tiles_n;
    const Bf16MmaContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                         Geometry::kOutputRows};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attr = cudaFuncSetAttribute(
            bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens, Bf16MmaContiguousOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attr);
    }
    bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Bf16MmaProductionSchedule<Geometry>;
    if ((x.ne[1] % Schedule::kBlockCols) == 0) {
        launch_variant<Geometry, Schedule, true>(x, weight, out, stream);
    } else {
        launch_variant<Geometry, Schedule, false>(x, weight, out, stream);
    }
}

} // namespace

void launch_bf16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == 14336 && weight.k == 5120) {
        launch_geometry<Bf16GemvGeometry<14336, 5120>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 6144) {
        launch_geometry<Bf16GemvGeometry<5120, 6144>>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("bf16 linear MMA: unsupported exact problem");
}

} // namespace ninfer::ops::detail
