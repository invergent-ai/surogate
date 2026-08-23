#include "ops/launcher/prepare_masked_block.h"

#include "core/device.h"
#include "ops/kernel/prepare_masked_block.cuh"

namespace ninfer::ops::detail {

void prepare_masked_block_launch(const Tensor& anchors, const Tensor& lengths,
                                 const Tensor& valid_columns, std::int32_t mask_id, Tensor& ids,
                                 Tensor& positions, std::int32_t block_size, cudaStream_t stream) {
    prepare_masked_block_kernel<<<ids.ne[1], 32, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(lengths.data),
        static_cast<const std::int32_t*>(valid_columns.data), mask_id,
        static_cast<std::int32_t*>(ids.data), static_cast<std::int32_t*>(positions.data),
        block_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
