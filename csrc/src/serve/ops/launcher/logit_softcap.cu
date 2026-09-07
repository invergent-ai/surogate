// sinfer::ops — logit softcap launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/op-development.md §2.
#include "ops/launcher/logit_softcap.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kernel/logit_softcap.cuh"

#include <algorithm>
#include <cstdint>

namespace sinfer::ops::detail {

void logit_softcap_launch(Tensor& x, float cap, cudaStream_t stream) {
    const std::int64_t n = x.numel();
    const auto grid      = static_cast<unsigned int>(
        std::clamp<std::int64_t>(div_up<std::int64_t>(n / 2, kLogitSoftcapBlock), 1, 65535));
    logit_softcap_kernel<<<grid, kLogitSoftcapBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x.data), n, 1.0F / cap, cap);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
