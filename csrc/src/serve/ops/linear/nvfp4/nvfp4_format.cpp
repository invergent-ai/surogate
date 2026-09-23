#include "ops/linear/nvfp4/nvfp4_format.h"

// For the device capability. Same library (sinfer_ops), and the accessor is
// about the device rather than the w8 path that happens to host it. The CUDA
// header comes first because w8fp8_plane.h declares stream-taking functions
// and this translation unit is plain C++ that had no reason to know
// cudaStream_t until now.
#include <cuda_runtime.h>

#include "ops/linear/w8a8/w8fp8_plane.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(std::string(operation) + ": NVFP4 geometry overflows");
    }
    return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, const char* operation) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(operation) + ": NVFP4 geometry overflows");
    }
    return left + right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* operation) {
    return checked_mul(checked_add(value, alignment - 1, operation) / alignment, alignment,
                       operation);
}

} // namespace

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation) {
    // The hardware question, asked of the weight rather than of a label.
    //
    // The planner refuses NVFP4 checkpoints on sm_89 by weights profile, but a
    // profile is metadata: `resolve_weights` reads `identity.weights_id` and
    // nothing else, while the W4A4 route is chosen per tensor from the
    // tensor's own qtype (`text_policy(const Weight&)`), and the loader
    // deliberately does not cross-check the two -- see the `(void)
    // weights_profile` note in each target's load/bindings.cpp. An artifact
    // labelled groupwise-int that declares an NVFP4 tensor therefore passes
    // the planner and lands here.
    //
    // It must not get further. `nvfp4_w4a4.cu` builds into sinfer_ops at the
    // whole arch set, so sm_89 gets a cubin whose body is __trap(); reaching
    // it aborts the device and every later CUDA call fails.
    //
    // Refusing every NVFP4 weight, not only the W4A4-bound ones, is
    // deliberate. The A16 route here is the in-house decode/small-T pair and
    // its codec has a real sm_89 software E2M1 path, so some A16 calls would
    // work. But qwen3_5 hands NVFP4 `AllowA4` (`variant.cpp`), so any
    // prefill-width call takes W4A4 and traps; refusing up front beats proving
    // a given call stays on A16.
    //
    // `!= 120`, not `< 120`: the FP4 archives are pinned `120a`, which loads
    // on exactly sm_120. On sm_121 or sm_100 the driver JITs the fatbin's only
    // other PTX, compute_89, whose W4A4 body is __trap(). cuobjdump on the
    // built library shows the arch set is exactly {sm_89, sm_120a}.
    //
    // This is the chokepoint: every NVFP4 wrapper (linear, linear_swiglu,
    // attn_input_proj, linear_add, gdn_input_proj) calls this before any NVFP4
    // kernel runs. `w4fp4_plane_for` next door guards the same way, though it
    // still spells the test `< 120` and inherits the sm_121 hole.
    if (weight.qtype == QType::NVFP4 && w8_device_compute_capability() != 120) {
        throw std::invalid_argument(
            std::string(operation) +
            ": NVFP4 weights need compute capability 12.0 or newer; this device cannot run them");
    }
    if (weight.n <= 0 || weight.k <= 0 || (weight.n % 128) != 0 || (weight.k % 64) != 0) {
        throw std::invalid_argument(std::string(operation) + ": NVFP4 requires N%128=0 and K%64=0");
    }

    const std::uint64_t elements = checked_mul(static_cast<std::uint64_t>(weight.n),
                                               static_cast<std::uint64_t>(weight.k), operation);
    Nvfp4WeightGeometry geometry{};
    geometry.code_plane_bytes   = elements / 2;
    geometry.scale_plane_offset = align_up(geometry.code_plane_bytes, 256, operation);
    geometry.scale_plane_bytes  = elements / 16;
    geometry.required_payload_bytes =
        checked_add(checked_add(geometry.scale_plane_offset, geometry.scale_plane_bytes, operation),
                    sizeof(float), operation);

    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.scale_dtype != DType::FP8_E4M3FN || weight.group_size != 16 || weight.group != 16 ||
        weight.ndim != 2 || weight.shape[0] != weight.n || weight.shape[1] != weight.k ||
        weight.padded_shape[0] != weight.n || weight.padded_shape[1] != weight.k ||
        weight.payload == nullptr || weight.qdata == nullptr || weight.scales == nullptr ||
        weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < geometry.required_payload_bytes ||
        !std::isfinite(weight.weight_scale_divisor) || weight.weight_scale_divisor <= 0.0F ||
        !std::isfinite(weight.input_scale_divisor) || weight.input_scale_divisor <= 0.0F ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 weight");
    }

    const auto* payload = static_cast<const std::byte*>(weight.payload);
    if (weight.qdata != payload || weight.scales != payload + geometry.scale_plane_offset) {
        throw std::invalid_argument(std::string(operation) + ": invalid NVFP4 plane geometry");
    }
    return geometry;
}

} // namespace sinfer::ops::detail
