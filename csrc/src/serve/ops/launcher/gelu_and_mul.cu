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
    const auto aligned16 = [](const void* p) {
        return (reinterpret_cast<std::uintptr_t>(p) & (alignof(Bf16x8Pack) - 1)) == 0;
    };
    if (n % 8 == 0 && aligned16(gate.data) && aligned16(up.data) && aligned16(out.data)) {
        // 16-byte packs, one load per operand per eight values, where the four-pair stream
        // below issues a 4-byte load per pair (149 us a call on EmbeddingGemma's [1152, 7040]
        // plane on a DGX Spark).
        const std::int64_t packs = n / 8;
        const auto grid = static_cast<unsigned int>(
            std::clamp<std::int64_t>(div_up(packs, static_cast<std::int64_t>(kBlock)), 1, 65535));
        const auto* g = static_cast<const Bf16x8Pack*>(gate.data);
        const auto* u = static_cast<const Bf16x8Pack*>(up.data);
        auto* o       = static_cast<Bf16x8Pack*>(out.data);
        if (round_gate && tanh_approx) {
            gelu_and_mul_bf16x8_kernel<true, true><<<grid, kBlock, 0, stream>>>(g, u, o, packs);
        } else if (round_gate) {
            gelu_and_mul_bf16x8_kernel<false, true><<<grid, kBlock, 0, stream>>>(g, u, o, packs);
        } else if (tanh_approx) {
            gelu_and_mul_bf16x8_kernel<true><<<grid, kBlock, 0, stream>>>(g, u, o, packs);
        } else {
            gelu_and_mul_bf16x8_kernel<false><<<grid, kBlock, 0, stream>>>(g, u, o, packs);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
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

void gelu_and_mul_fused_launch(const Tensor& gate_up, bool tanh_approx, Tensor& out,
                               cudaStream_t stream) {
    const std::int64_t k_pairs = out.ne[0] / 2;
    const std::int64_t columns = out.ne[1];
    constexpr int kBlock       = 256;
    const auto grid = static_cast<unsigned int>(
        std::clamp<std::int64_t>(div_up(k_pairs * columns, static_cast<std::int64_t>(kBlock)), 1, 65535));
    const auto* in = static_cast<const __nv_bfloat16*>(gate_up.data);
    auto* o        = static_cast<__nv_bfloat16*>(out.data);
    if (tanh_approx) {
        gelu_and_mul_fused_kernel<true><<<grid, kBlock, 0, stream>>>(in, o, k_pairs, columns);
    } else {
        gelu_and_mul_fused_kernel<false><<<grid, kBlock, 0, stream>>>(in, o, k_pairs, columns);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
