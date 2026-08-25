// ninfer::ops - mask_columns launcher: grid/block/stream configuration.
#include "ops/launcher/mask_columns.h"

#include "core/device.h"
#include <algorithm>
#include "ops/kernel/mask_columns.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void mask_columns_zero_launch(Tensor& matrix, const Tensor& valid_columns, cudaStream_t stream) {
    constexpr int kBlock     = 256;
    const std::int32_t rows  = matrix.ne[0];
    const std::int32_t cols  = matrix.ne[1];
    const std::int64_t total = static_cast<std::int64_t>(rows) * cols;
    const int grid = static_cast<int>(std::min<std::int64_t>((total + kBlock - 1) / kBlock, 4096));
    const auto* valid = static_cast<const std::int32_t*>(valid_columns.data);
    if (matrix.dtype == DType::FP32) {
        mask_columns_zero_kernel<float>
            <<<grid, kBlock, 0, stream>>>(static_cast<float*>(matrix.data), valid, rows, cols);
    } else {
        mask_columns_zero_kernel<__nv_bfloat16><<<grid, kBlock, 0, stream>>>(
            static_cast<__nv_bfloat16*>(matrix.data), valid, rows, cols);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
