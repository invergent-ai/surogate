// sinfer::ops — mean_pool wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "api/ops/mean_pool.h"

#include "ops/launcher/mean_pool.h" // detail::mean_pool_launch

#include <stdexcept>

namespace sinfer::ops {

void mean_pool(const Tensor& x, int count, bool accumulate, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("mean_pool: x/out must be BF16");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("mean_pool: x/out must be contiguous");
    }
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("mean_pool: x must be a 2-D [hidden, tokens] matrix");
    }
    if (out.ne[0] != x.ne[0] || out.numel() != x.ne[0]) {
        throw std::invalid_argument("mean_pool: out must be [hidden]");
    }
    if (count <= 0 || count > x.ne[1]) {
        throw std::invalid_argument("mean_pool: count must be in (0, tokens]");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("mean_pool: x/out data must be non-null");
    }

    detail::mean_pool_launch(x, count, accumulate, out, stream);
}

} // namespace sinfer::ops
