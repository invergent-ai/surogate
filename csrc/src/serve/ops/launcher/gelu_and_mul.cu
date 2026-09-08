// sinfer::ops — gelu_mul launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/op-development.md §2.
#include "ops/launcher/gelu_and_mul.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kernel/gelu_and_mul.cuh"

#include <algorithm>
#include <cstdint>

namespace sinfer::ops::detail {

void gelu_and_mul_launch(const Tensor& gate, const Tensor& up, bool tanh_approx, Tensor& out,
                         cudaStream_t stream, bool round_gate) {
    const std::int64_t n = out.numel();
    constexpr int kBlock = 256;
    const std::int64_t pairs = n / 2;
    const auto grid = static_cast<unsigned int>(std::clamp<std::int64_t>(
        div_up(pairs, static_cast<std::int64_t>(kBlock) * kGeluAndMulPairsPerThread), 1, 65535));

    const auto* g = static_cast<const __nv_bfloat16*>(gate.data);
    const auto* u = static_cast<const __nv_bfloat16*>(up.data);
    auto* o       = static_cast<__nv_bfloat16*>(out.data);
    if (round_gate && tanh_approx) {
        gelu_and_mul_kernel<true, true><<<grid, kBlock, 0, stream>>>(g, u, o, n);
    } else if (round_gate) {
        gelu_and_mul_kernel<false, true><<<grid, kBlock, 0, stream>>>(g, u, o, n);
    } else if (tanh_approx) {
        gelu_and_mul_kernel<true><<<grid, kBlock, 0, stream>>>(g, u, o, n);
    } else {
        gelu_and_mul_kernel<false><<<grid, kBlock, 0, stream>>>(g, u, o, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
