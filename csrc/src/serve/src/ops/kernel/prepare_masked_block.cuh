#pragma once

#include <cstdint>

namespace ninfer::ops {

__global__ void prepare_masked_block_kernel(const std::int32_t* anchors,
                                            const std::int32_t* lengths,
                                            const std::int32_t* valid_columns, std::int32_t mask_id,
                                            std::int32_t* ids, std::int32_t* positions,
                                            std::int32_t block_size) {
    const int i = static_cast<int>(threadIdx.x);
    const int b = static_cast<int>(blockIdx.x);
    if (i >= block_size) return;
    const int offset  = i + block_size * b;
    const int valid   = valid_columns[b];
    ids[offset]       = i == 0 ? anchors[b] : mask_id;
    positions[offset] = lengths[b] + min(i, valid - 1);
}

} // namespace ninfer::ops
