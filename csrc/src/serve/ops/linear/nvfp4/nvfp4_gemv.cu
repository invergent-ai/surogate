#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

template <class Geometry>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    if (x.ne[0] != Geometry::kInputRows || x.ne[1] != 1 || out.ne[0] != Geometry::kOutputRows ||
        out.ne[1] != 1 || weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("nvfp4 linear decode: invalid exact problem");
    }

    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

/// A generic shape at one token: N comes from the weight, K from the instantiation.
template <std::int32_t K>
void launch_generic(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Nvfp4GemvKGeometry<K>;
    using Schedule = typename Nvfp4GenericDecodeSchedule<K>::Type;
    static_assert((K % (32 * Schedule::kValuesPerLane)) == 0);
    if (x.ne[0] != K || x.ne[1] != 1 || out.ne[0] != weight.n || out.ne[1] != 1 || weight.k != K ||
        weight.n <= 0 || (weight.n % 128) != 0) {
        throw std::invalid_argument("nvfp4 linear decode: invalid generic problem");
    }
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    const int blocks                   = weight.n / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<blocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

void launch_generic_by_k(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (weight.k) {
#define SINFER_NVFP4_K_CASE(K)                                                                    \
    case K:                                                                                       \
        if constexpr (((K) % 256) == 0) {                                                         \
            launch_generic<K>(x, weight, out, stream);                                            \
            return;                                                                               \
        }                                                                                         \
        break;
        SINFER_NVFP4_FOR_EACH_ACTIVATION_K(SINFER_NVFP4_K_CASE)
#undef SINFER_NVFP4_K_CASE
    default:
        break;
    }
    throw std::invalid_argument("nvfp4 linear decode: no GEMV for this K");
}

} // namespace

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (resolve_nvfp4_gemv_only_problem(weight.n, weight.k)) {
    case Nvfp4GemvOnlyProblem::AttnInput2560:
        launch_exact<Nvfp4AttnInput2560Geometry>(x, weight, out, stream);
        return;
    case Nvfp4GemvOnlyProblem::GdnInput2560:
        launch_exact<Nvfp4GdnInput2560Geometry>(x, weight, out, stream);
        return;
    case Nvfp4GemvOnlyProblem::MlpGateUp2560:
        launch_exact<Nvfp4MlpGateUp2560Geometry>(x, weight, out, stream);
        return;
    case Nvfp4GemvOnlyProblem::Residual4096:
        launch_exact<Nvfp4Residual4096Geometry>(x, weight, out, stream);
        return;
    case Nvfp4GemvOnlyProblem::Residual9216:
        launch_exact<Nvfp4Residual9216Geometry>(x, weight, out, stream);
        return;
    case Nvfp4GemvOnlyProblem::None:
        break;
    }
    if (is_nvfp4_generic_problem(weight.n, weight.k)) {
        launch_generic_by_k(x, weight, out, stream);
        return;
    }
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::AttnInput:
        launch_exact<Nvfp4AttnInputGeometry>(x, weight, out, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_exact<Nvfp4GdnInputGeometry>(x, weight, out, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_exact<Nvfp4MlpGateUpGeometry>(x, weight, out, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launch_exact<Nvfp4Residual6144Geometry>(x, weight, out, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_exact<Nvfp4Residual17408Geometry>(x, weight, out, stream);
        return;
    }
}

} // namespace sinfer::ops::detail
