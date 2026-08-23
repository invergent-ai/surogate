#include "ops/launcher/add_bias.h"

#include "ops/common/math.h"
#include "ops/kernel/add_bias.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

void add_bias_launch(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    constexpr int block        = 256;
    constexpr int rowsPerBlock = 4;
    constexpr std::int64_t maxVectorRows =
        rowsPerBlock * std::numeric_limits<unsigned short>::max();
    const std::int64_t n    = x.numel();
    const std::int64_t rows = n / x.ne[0];
    const auto addresses =
        reinterpret_cast<std::uintptr_t>(bias.data) | reinterpret_cast<std::uintptr_t>(x.data);
    if ((x.ne[0] % 8) == 0 && (addresses & (alignof(Bf16x8Pack) - 1)) == 0 &&
        n <= kBf16x8CacheSizedMaxElements && rows <= maxVectorRows) {
        const std::int32_t packs = x.ne[0] / 8;
        const unsigned grid_x    = static_cast<unsigned>(div_up(packs, block));
        if (rows >= 1024) {
            const unsigned grid_y =
                static_cast<unsigned>(div_up(rows, static_cast<std::int64_t>(rowsPerBlock)));
            add_bias_bf16x8_kernel<block, rowsPerBlock>
                <<<dim3(grid_x, grid_y, 1u), block, 0, stream>>>(
                    static_cast<const Bf16x8Pack*>(bias.data), static_cast<Bf16x8Pack*>(x.data),
                    packs, static_cast<std::int32_t>(rows));
        } else {
            add_bias_bf16x8_kernel<block, 1>
                <<<dim3(grid_x, static_cast<unsigned>(rows), 1u), block, 0, stream>>>(
                    static_cast<const Bf16x8Pack*>(bias.data), static_cast<Bf16x8Pack*>(x.data),
                    packs, static_cast<std::int32_t>(rows));
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool paired = (x.ne[0] % 2) == 0 && (addresses & 0x3u) == 0;
    if (paired && rows <= std::numeric_limits<std::int32_t>::max()) {
        const std::int32_t pairs = x.ne[0] / 2;
        const unsigned grid_x =
            static_cast<unsigned>(div_up(pairs, block * kAddBiasPairsPerThread));
        const unsigned grid_y = static_cast<unsigned>(
            std::min<std::int64_t>(rows, std::numeric_limits<unsigned short>::max()));
        add_bias_bf16x2_kernel<block><<<dim3(grid_x, grid_y, 1u), block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(bias.data), static_cast<__nv_bfloat162*>(x.data),
            pairs, static_cast<std::int32_t>(rows));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    add_bias_kernel<<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(bias.data),
                                                static_cast<__nv_bfloat16*>(x.data), x.ne[0], n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
