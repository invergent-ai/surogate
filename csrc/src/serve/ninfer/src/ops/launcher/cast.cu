#include "ops/launcher/cast.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/cast.cuh"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 256;
constexpr int kGridCap = 4096;

int cast_grid(std::int64_t work_items) {
    return static_cast<int>(std::max<std::int64_t>(
        1,
        std::min<std::int64_t>(div_up(work_items, static_cast<std::int64_t>(kBlock)), kGridCap)));
}

} // namespace

void cast_fp32_to_bf16_launch(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    const std::int64_t count = source.numel();
    const auto source_addr   = reinterpret_cast<std::uintptr_t>(source.data);
    const auto dest_addr     = reinterpret_cast<std::uintptr_t>(destination.data);
    if ((count % 4) == 0 && (source_addr & 0xfu) == 0 && (dest_addr & 0x7u) == 0) {
        const std::int64_t vectors = count / 4;
        cast_fp32_to_bf16_x4_kernel<<<cast_grid(vectors), kBlock, 0, stream>>>(
            static_cast<const float4*>(source.data), static_cast<Bf16x4*>(destination.data),
            vectors);
    } else if ((count % 2) == 0 && (source_addr & 0x7u) == 0 && (dest_addr & 0x3u) == 0) {
        const std::int64_t pairs = count / 2;
        cast_fp32_to_bf16_x2_kernel<<<cast_grid(pairs), kBlock, 0, stream>>>(
            static_cast<const float2*>(source.data), static_cast<__nv_bfloat162*>(destination.data),
            pairs);
    } else {
        cast_fp32_to_bf16_scalar_kernel<<<cast_grid(count), kBlock, 0, stream>>>(
            static_cast<const float*>(source.data), static_cast<__nv_bfloat16*>(destination.data),
            count);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
