// sinfer::ops — scale launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/op-development.md §2.
#include "ops/launcher/scale.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kernel/scale.cuh"

#include <algorithm>
#include <cstdint>

namespace sinfer::ops::detail {

void scale_launch(Tensor& x, float factor, cudaStream_t stream) {
    const std::int64_t n = x.numel();
    const auto grid      = static_cast<unsigned int>(
        std::clamp<std::int64_t>(div_up<std::int64_t>(n / 2, kScaleBlock), 1, 65535));
    scale_kernel<<<grid, kScaleBlock, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n, factor);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
