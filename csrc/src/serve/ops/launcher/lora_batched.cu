#include "ops/launcher/lora_batched.h"

#include "core/device.h"
#include "ops/kernel/lora_batched.cuh"

namespace sinfer::ops::detail {

void lora_batched_shrink_launch(const Tensor& x, const void* a_bank, const Tensor& ids, Tensor& low,
                                std::int32_t k, std::int32_t rank, std::int64_t a_stride,
                                cudaStream_t stream) {
    // One warp per rank row, four rows per block: rank is at most 64, so the grid
    // is a handful of blocks per token and the geometry depends only on (rank,
    // tokens) -- both fixed for a captured graph.
    constexpr int kRowsPerBlock = 4;
    const dim3 block(32, kRowsPerBlock, 1);
    const dim3 grid(static_cast<unsigned>((rank + kRowsPerBlock - 1) / kRowsPerBlock),
                    static_cast<unsigned>(x.ne[1]), 1);
    lora_batched_shrink_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(a_bank),
        static_cast<const std::int32_t*>(ids.data), static_cast<__nv_bfloat16*>(low.data), k, rank,
        static_cast<std::int32_t>(x.ne[1]), a_stride);
    CUDA_CHECK(cudaGetLastError());
}

void lora_batched_expand_launch(const Tensor& low, const void* b_bank, const Tensor& ids,
                                Tensor& out, std::int32_t n, std::int32_t rank,
                                std::int64_t b_stride, cudaStream_t stream) {
    constexpr int kThreads = 128;
    const dim3 block(kThreads, 1, 1);
    const dim3 grid(static_cast<unsigned>((n + kThreads - 1) / kThreads),
                    static_cast<unsigned>(out.ne[1]), 1);
    lora_batched_expand_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(low.data), static_cast<const __nv_bfloat16*>(b_bank),
        static_cast<const std::int32_t*>(ids.data), static_cast<__nv_bfloat16*>(out.data), n, rank,
        static_cast<std::int32_t>(out.ne[1]), b_stride);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
