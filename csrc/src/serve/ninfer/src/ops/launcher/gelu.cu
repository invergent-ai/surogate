#include "ops/launcher/gelu.h"

#include "ops/common/math.h"
#include "ops/kernel/gelu.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream) {
    constexpr int block   = 256;
    constexpr int maxGrid = 16384;
    const std::int64_t n  = x.numel();
    const auto address    = reinterpret_cast<std::uintptr_t>(x.data);
    if ((address & (alignof(Bf16x8Pack) - 1)) == 0 && (n % 8) == 0 &&
        n <= kBf16x8CacheSizedMaxElements) {
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            maxGrid, std::max<std::int64_t>(1, div_up(packs, static_cast<std::int64_t>(block)))));
        if (mode == GeluMode::Tanh) {
            gelu_bf16x8_kernel<true, block>
                <<<grid, block, 0, stream>>>(static_cast<Bf16x8Pack*>(x.data), packs);
        } else {
            gelu_bf16x8_kernel<false, block>
                <<<grid, block, 0, stream>>>(static_cast<Bf16x8Pack*>(x.data), packs);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool paired = (address & 0x3u) == 0;
    if (paired && n >= 2) {
        const std::int64_t pairs = n / 2;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            div_up(pairs, static_cast<std::int64_t>(block * kGeluPairsPerThread)),
            std::numeric_limits<int>::max()));
        auto* x2                 = static_cast<__nv_bfloat162*>(x.data);
        auto* tail               = static_cast<__nv_bfloat16*>(x.data) + pairs * 2;
        if (mode == GeluMode::Tanh) {
            gelu_bf16x2_kernel<true, block>
                <<<grid, block, 0, stream>>>(x2, pairs, tail, (n & 1) != 0);
        } else {
            gelu_bf16x2_kernel<false, block>
                <<<grid, block, 0, stream>>>(x2, pairs, tail, (n & 1) != 0);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    if (mode == GeluMode::Tanh) {
        gelu_kernel<true><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    } else {
        gelu_kernel<false><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
