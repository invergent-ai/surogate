#include "ops/launcher/prepare_ragged_prefix.h"

#include "core/device.h"
#include "ops/kernel/prepare_ragged_prefix.cuh"

namespace ninfer::ops::detail {

void prepare_ragged_prefix_launch(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                                  const Tensor& ends, Tensor& destination, Tensor& positions,
                                  Tensor& counts, cudaStream_t stream) {
    constexpr int block = 128;
    const dim3 grid(source.ne[1], destination.ne[2], 1);
    prepare_ragged_prefix_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(source.data), static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(starts.data), static_cast<const std::int32_t*>(ends.data),
        static_cast<uint4*>(destination.data), static_cast<std::int32_t*>(positions.data),
        static_cast<std::int32_t*>(counts.data), source.ne[0] / 8, source.ne[1],
        source.nb[1] / static_cast<std::int64_t>(sizeof(uint4)),
        source.nb[2] / static_cast<std::int64_t>(sizeof(uint4)));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
