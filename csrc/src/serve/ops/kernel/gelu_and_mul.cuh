#pragma once

// sinfer::ops — gelu_mul kernel: out = gelu(gate) * up, elementwise.
// Shares gelu.cuh's `gelu_one`, so the two ops cannot drift on the formula.
// Included only by its launcher. See docs/op-development.md §6.

#include "ops/kernel/gelu.cuh" // gelu_one<TanhApprox>

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops {

inline constexpr int kGeluAndMulPairsPerThread = 4;

template <bool TanhApprox, bool RoundGate = false>
__device__ __forceinline__ __nv_bfloat162 gelu_mul_pair(__nv_bfloat162 g, __nv_bfloat162 u) {
    float a = gelu_one<TanhApprox>(__low2float(g));
    float b = gelu_one<TanhApprox>(__high2float(g));
    if constexpr (RoundGate) {
        a = __bfloat162float(__float2bfloat16_rn(a));
        b = __bfloat162float(__float2bfloat16_rn(b));
    }
    const float r0 = a * __low2float(u);
    const float r1 = b * __high2float(u);
    return __floats2bfloat162_rn(r0, r1);
}

template <bool TanhApprox, bool RoundGate = false>
__launch_bounds__(256) __global__
    void gelu_and_mul_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                             __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kGeluAndMulPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up);
    auto* out2        = reinterpret_cast<__nv_bfloat162*>(out);
    for (std::int64_t j = tid * kGeluAndMulPairsPerThread; j < n2; j += stride) {
#pragma unroll
        for (int item = 0; item < kGeluAndMulPairsPerThread; ++item) {
            const std::int64_t p = j + item;
            if (p < n2) { out2[p] = gelu_mul_pair<TanhApprox, RoundGate>(gate2[p], up2[p]); }
        }
    }

    // Odd tail: one scalar element no pair covers.
    if (tid == 0 && (n & 1) != 0) {
        const std::int64_t i = n - 1;
        float a = gelu_one<TanhApprox>(__bfloat162float(gate[i]));
        if constexpr (RoundGate) { a = __bfloat162float(__float2bfloat16_rn(a)); }
        out[i] = __float2bfloat16_rn(a * __bfloat162float(up[i]));
    }
}

/// The same product over one fused `[gate; up]` plane: column t of `gate_up` holds the K gate
/// values then the K up values (a `[2K, T]` matmul output), and column t of `out` is their
/// product. Same `gelu_mul_pair`, so a fused and a split projection of the same numbers give the
/// same bits.
template <bool TanhApprox>
__launch_bounds__(256) __global__
    void gelu_and_mul_fused_kernel(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                                   std::int64_t k_pairs, std::int64_t columns) {
    const std::int64_t total  = k_pairs * columns;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto* in2 = reinterpret_cast<const __nv_bfloat162*>(gate_up);
    auto* out2      = reinterpret_cast<__nv_bfloat162*>(out);
    for (std::int64_t p = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; p < total;
         p += stride) {
        const std::int64_t column = p / k_pairs;
        const std::int64_t pair   = p - column * k_pairs;
        const std::int64_t base   = column * 2 * k_pairs;
        out2[p] = gelu_mul_pair<TanhApprox>(in2[base + pair], in2[base + k_pairs + pair]);
    }
}

} // namespace sinfer::ops
