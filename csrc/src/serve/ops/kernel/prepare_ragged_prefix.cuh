#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void prepare_ragged_prefix_kernel(
    const uint4* __restrict__ source, const std::int32_t* __restrict__ lanes,
    const std::int32_t* __restrict__ starts, const std::int32_t* __restrict__ ends,
    uint4* __restrict__ destination, std::int32_t* __restrict__ positions,
    std::int32_t* __restrict__ counts, std::int32_t vectors_per_column, std::int32_t width,
    std::int64_t source_column_stride, std::int64_t source_lane_stride) {
    const std::int32_t column      = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t batch       = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t count       = ends[batch] - starts[batch];
    const bool live                = column < count;
    const std::int64_t source_base = static_cast<std::int64_t>(lanes[batch]) * source_lane_stride +
                                     static_cast<std::int64_t>(column) * source_column_stride;
    const std::int64_t destination_base =
        (static_cast<std::int64_t>(batch) * width + column) * vectors_per_column;
    const uint4 zero{};
    for (std::int32_t vector = static_cast<std::int32_t>(threadIdx.x); vector < vectors_per_column;
         vector += static_cast<std::int32_t>(blockDim.x)) {
        destination[destination_base + vector] = live ? source[source_base + vector] : zero;
    }
    if (threadIdx.x == 0) {
        positions[column + width * batch] = starts[batch] + (live ? column : max(count - 1, 0));
        if (column == 0) { counts[batch] = count; }
    }
}

} // namespace ninfer::ops
