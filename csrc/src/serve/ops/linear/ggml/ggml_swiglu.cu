#include "ops/linear/ggml/ggml_swiglu.h"

#include "core/device.h"
#include "ops/linear/ggml/ggml_q8_1.h"
#include "ops/linear/ggml/ggml_swiglu_decode.cuh"

#include <stdexcept>

namespace sinfer::ops::detail::ggml {

bool swiglu_decode_admits(GgmlType gate, GgmlType up, std::int32_t rows, std::int32_t k,
                          std::int32_t tokens) noexcept {
    const auto supported = [](GgmlType type) {
        return type == GgmlType::Q8_0 || type == GgmlType::IQ4_NL;
    };
    if (rows <= 0 || k <= 0 || k % 32 != 0 || tokens <= 0 || tokens > 8 || !supported(gate) ||
        !supported(up)) {
        return false;
    }
    // Keep the independent projections for large Q8-containing MMA workloads:
    // their wider row tiles sustain better bandwidth. Scalar decode and IQ4-only
    // pairs benefit from fusion at these sizes as well.
    const std::uint64_t weight_bytes =
        std::uint64_t(rows) * (k / 32) * (block_bytes(gate) + block_bytes(up));
    return tokens == 1 || (gate == GgmlType::IQ4_NL && up == GgmlType::IQ4_NL) ||
           weight_bytes <= 64ull * 1024 * 1024;
}

void swiglu_decode_launch(GgmlType gate_type, const void* gate, GgmlType up_type, const void* up,
                          std::int32_t rows, std::int32_t k, const __nv_bfloat16* x,
                          std::int32_t tokens, __nv_bfloat16* out, void* scratch,
                          std::size_t scratch_bytes, cudaStream_t stream) {
    if (!swiglu_decode_admits(gate_type, up_type, rows, k, tokens) || gate == nullptr ||
        up == nullptr || x == nullptr || out == nullptr) {
        throw std::invalid_argument("ggml swiglu decode: invalid types, dimensions or pointers");
    }
    if (scratch == nullptr || scratch_bytes < linear_workspace_bytes(rows, k, tokens) ||
        (reinterpret_cast<std::uintptr_t>(scratch) & 15u) != 0) {
        throw std::invalid_argument("ggml swiglu decode: scratch too small or misaligned");
    }
    auto* codes = static_cast<std::int8_t*>(scratch);
    auto* ds    = reinterpret_cast<__half2*>(codes + std::size_t(tokens) * k);
    quantize_q8_1_planes_launch(x, k, tokens, codes, ds, stream);
    const auto launch = [&]<GgmlType Gate, GgmlType Up>() {
        const auto* g = static_cast<const std::uint8_t*>(gate);
        const auto* u = static_cast<const std::uint8_t*>(up);
        if (tokens == 1) {
            swiglu_decode_kernel<Gate, Up><<<dim3((rows + 1) / 2, tokens), 128, 0, stream>>>(
                g, u, codes, ds, rows, k, tokens, out);
        } else {
            swiglu_decode_mma_kernel<Gate, Up>
                <<<(rows + 7) / 8, 256, 0, stream>>>(g, u, codes, ds, rows, k, tokens, out);
        }
    };
    if (gate_type == GgmlType::Q8_0) {
        if (up_type == GgmlType::Q8_0) {
            launch.template operator()<GgmlType::Q8_0, GgmlType::Q8_0>();
        } else {
            launch.template operator()<GgmlType::Q8_0, GgmlType::IQ4_NL>();
        }
    } else {
        if (up_type == GgmlType::Q8_0) {
            launch.template operator()<GgmlType::IQ4_NL, GgmlType::Q8_0>();
        } else {
            launch.template operator()<GgmlType::IQ4_NL, GgmlType::IQ4_NL>();
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail::ggml
