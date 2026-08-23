// Implements: include/ninfer/ops/residual_add.h
// Finite dispatch: aligned BF16x8 production route, BF16x2 fallback, then
// scalar fallback for two-byte-aligned sliced storage.
#include "ops/launcher/residual_add.h"

#include "ops/common/math.h"
#include "ops/kernel/residual_add.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void residual_add_launch(const Tensor& y, Tensor& x, cudaStream_t stream) {
    const std::int64_t n   = x.numel();
    constexpr int kBlock   = 256;
    constexpr int kMaxGrid = 4096;
    const auto y_addr      = reinterpret_cast<std::uintptr_t>(y.data);
    const auto x_addr      = reinterpret_cast<std::uintptr_t>(x.data);
    if (((y_addr | x_addr) & (alignof(Bf16x8Pack) - 1)) == 0 && (n % 8) == 0) {
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            kMaxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(kBlock)))));
        residual_add_bf16x8_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(y.data), static_cast<Bf16x8Pack*>(x.data), packs);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (((y_addr | x_addr) & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int scalar_grid = static_cast<int>(
            std::min<std::int64_t>(kMaxGrid, div_up(n, static_cast<std::int64_t>(kBlock))));
        residual_add_scalar_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(y.data), static_cast<__nv_bfloat16*>(x.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2                 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kResidualAddPairsPerThread;
    const int grid                        = static_cast<int>(
        std::min<std::int64_t>(kMaxGrid, std::max<std::int64_t>(1, div_up(n2, kPairsPerBlock))));

    residual_add_bf16x2_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(y.data), static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
