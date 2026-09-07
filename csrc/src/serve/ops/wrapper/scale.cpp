// sinfer::ops — scale wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "api/ops/scale.h"

#include "ops/launcher/scale.h" // detail::scale_launch

#include <cmath>
#include <stdexcept>

namespace sinfer::ops {

void scale(Tensor& x, float factor, cudaStream_t stream) {
    if (x.dtype != DType::BF16) { throw std::invalid_argument("scale: x must be BF16"); }
    if (!x.is_contiguous()) { throw std::invalid_argument("scale: x must be contiguous"); }
    if (!std::isfinite(factor)) { throw std::invalid_argument("scale: factor must be finite"); }
    if (x.numel() == 0) { return; }
    if (x.data == nullptr) { throw std::invalid_argument("scale: x data must be non-null"); }

    detail::scale_launch(x, factor, stream);
}

} // namespace sinfer::ops
