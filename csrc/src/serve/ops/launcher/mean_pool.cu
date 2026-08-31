// sinfer::ops — mean_pool launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/op-development.md §2.
#include "ops/launcher/mean_pool.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/mean_pool.cuh"

#include <cstdint>

namespace sinfer::ops::detail {

void mean_pool_launch(const Tensor& x, int count, bool accumulate, Tensor& out,
                      cudaStream_t stream) {
    const auto hidden = static_cast<std::int32_t>(x.ne[0]);
    // Accumulating callers want the running sum; the divide happens once, when
    // the last chunk has landed.
    const float scale = accumulate ? 1.0f : 1.0f / static_cast<float>(count);
    mean_pool_kernel<<<static_cast<unsigned int>(hidden), kMeanPoolBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), hidden,
        static_cast<std::int32_t>(count), scale, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
