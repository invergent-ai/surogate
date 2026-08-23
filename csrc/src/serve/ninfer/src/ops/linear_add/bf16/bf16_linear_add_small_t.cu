#include "ops/linear_add/bf16/bf16_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

struct Bf16LinearAddSmallTOutput {
    __nv_bfloat16* residual;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                          float accumulator) const {
        __nv_bfloat16* destination = residual + static_cast<std::int64_t>(token) * rows + row;
        const float residual_value = __bfloat162float(*destination);
        *destination               = __float2bfloat16_rn(accumulator + residual_value);
    }
};

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Geometry = Bf16GemvGeometry<5120, 6144>;
    using Schedule = typename Bf16LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    const Bf16LinearAddSmallTOutput output{static_cast<__nv_bfloat16*>(residual.data),
                                           Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_small_t_inner_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kBf16LinearAddSmallTMinTokens + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(
    std::make_index_sequence<kBf16LinearAddSmallTMaxTokens - kBf16LinearAddSmallTMinTokens + 1>{});

} // namespace

void bf16_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream) {
    kLaunchers[static_cast<std::size_t>(x.ne[1] - kBf16LinearAddSmallTMinTokens)](x, weight,
                                                                                  residual, stream);
}

} // namespace ninfer::ops::detail
