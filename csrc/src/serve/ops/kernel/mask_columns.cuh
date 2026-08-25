#pragma once

// ninfer::ops - mask_columns kernel: zero-fill trailing columns of a [C,T]
// matrix past a device-resident valid-column count (PATCHES.md #27).

#include <cstdint>

namespace ninfer::ops {

template <class Element>
__global__ void mask_columns_zero_kernel(Element* matrix, const std::int32_t* valid_columns,
                                         std::int32_t rows, std::int32_t columns) {
    const std::int32_t valid = *valid_columns < 0 ? 0 : *valid_columns;
    if (valid >= columns) { return; }
    const std::int64_t begin = static_cast<std::int64_t>(valid) * rows;
    const std::int64_t total = static_cast<std::int64_t>(columns) * rows;
    const std::int64_t start = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t step  = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = begin + start; i < total; i += step) { matrix[i] = Element(0); }
}

} // namespace ninfer::ops
