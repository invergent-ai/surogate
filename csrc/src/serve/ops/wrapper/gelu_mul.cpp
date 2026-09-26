// sinfer::ops — gelu_mul wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "api/ops/gelu_mul.h"

#include "ops/launcher/gelu_and_mul.h" // detail::gelu_and_mul_launch

#include <cstdint>
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

void gelu_mul_fused(const Tensor& gate_up, GeluMode mode, Tensor& out, cudaStream_t stream) {
    if (gate_up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("gelu_mul_fused: gate_up/out must be BF16");
    }
    if (gate_up.ne[0] != 2 * out.ne[0] || gate_up.ne[1] != out.ne[1] || gate_up.ne[2] != 1 ||
        gate_up.ne[3] != 1 || out.ne[2] != 1 || out.ne[3] != 1 || (out.ne[0] % 2) != 0) {
        throw std::invalid_argument("gelu_mul_fused: expected gate_up [2K,T] and out [K,T], K even");
    }
    if (!gate_up.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("gelu_mul_fused: gate_up/out must be contiguous");
    }
    if (out.numel() == 0) { return; }
    if (gate_up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("gelu_mul_fused: gate_up/out data must be non-null");
    }
    const auto a = reinterpret_cast<std::uintptr_t>(gate_up.data);
    const auto b = reinterpret_cast<std::uintptr_t>(out.data);
    if (a < b + out.bytes() && b < a + gate_up.bytes()) {
        throw std::invalid_argument("gelu_mul_fused: out must not overlap gate_up");
    }
    detail::gelu_and_mul_fused_launch(gate_up, mode == GeluMode::Tanh, out, stream);
}

} // namespace sinfer::ops
