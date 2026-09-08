// sinfer::ops — gelu_mul wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "api/ops/gelu_mul.h"

#include "ops/launcher/gelu_and_mul.h" // detail::gelu_and_mul_launch

#include <stdexcept>

namespace sinfer::ops {

void gelu_mul(const Tensor& gate, const Tensor& up, GeluMode mode, Tensor& out,
              cudaStream_t stream, bool round_gate) {
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("gelu_mul: gate/up/out must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument("gelu_mul: gate/up/out shapes must match");
        }
    }
    if (!gate.is_contiguous() || !up.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("gelu_mul: gate/up/out must be contiguous");
    }
    if (out.numel() == 0) { return; }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("gelu_mul: gate/up/out data must be non-null");
    }

    detail::gelu_and_mul_launch(gate, up, mode == GeluMode::Tanh, out, stream, round_gate);
}

} // namespace sinfer::ops
