// sinfer::ops — logit softcap wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "api/ops/logit_softcap.h"

#include "ops/launcher/logit_softcap.h" // detail::logit_softcap_launch

#include <cmath>
#include <stdexcept>

namespace sinfer::ops {

void logit_softcap(Tensor& x, float cap, cudaStream_t stream) {
    if (x.dtype != DType::BF16) { throw std::invalid_argument("logit_softcap: x must be BF16"); }
    if (!x.is_contiguous()) {
        throw std::invalid_argument("logit_softcap: x must be contiguous");
    }
    // A non-positive cap is not "no cap": `tanh(x/0)` is a sign function and a negative cap
    // would invert every logit. A caller that means "uncapped" does not call this at all.
    if (!(cap > 0.0F) || !std::isfinite(cap)) {
        throw std::invalid_argument("logit_softcap: cap must be positive and finite");
    }
    if (x.numel() == 0) { return; }
    if (x.data == nullptr) {
        throw std::invalid_argument("logit_softcap: x data must be non-null");
    }

    detail::logit_softcap_launch(x, cap, stream);
}

} // namespace sinfer::ops
