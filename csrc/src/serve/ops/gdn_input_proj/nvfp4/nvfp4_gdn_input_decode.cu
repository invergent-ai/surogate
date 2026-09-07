#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

namespace sinfer::ops::detail {

namespace {

/// The 4B's [qkv | z] layout: 16 key heads and 32 value heads of 128, so qkv is 2 x 2,048 +
/// 4,096 and z is 4,096. Same interface as Nvfp4GdnInputOutput, which stays the 27B's.
struct Nvfp4GdnInputOutput2560 {
    static constexpr std::int32_t kQkvRows = 8192;
    static constexpr std::int32_t kZRows   = 4096;
    __nv_bfloat16* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        __nv_bfloat16* destination =
            parent_row < kQkvRows
                ? qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row
                : z + static_cast<std::int64_t>(token) * kZRows + (parent_row - kQkvRows);
        *destination = __float2bfloat16_rn(value);
    }
};
static_assert(Nvfp4GdnInputOutput2560::kQkvRows + Nvfp4GdnInputOutput2560::kZRows ==
              Nvfp4GdnInput2560Geometry::kOutputRows);

template <class Geometry, class Output>
void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
        Output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    if (resolve_nvfp4_gemv_only_problem(weight.n, weight.k) ==
        Nvfp4GemvOnlyProblem::GdnInput2560) {
        launch<Nvfp4GdnInput2560Geometry, Nvfp4GdnInputOutput2560>(x, weight, qkv, z, stream);
        return;
    }
    launch<Nvfp4GdnInputGeometry, Nvfp4GdnInputOutput>(x, weight, qkv, z, stream);
}

} // namespace sinfer::ops::detail
