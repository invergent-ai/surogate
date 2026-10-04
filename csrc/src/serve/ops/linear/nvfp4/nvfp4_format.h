#pragma once

#include "core/tensor.h"

#include <cstdint>

namespace sinfer::ops::detail {

struct Nvfp4WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation);

/// Whether this build carries FP4 kernels for a device of compute capability `sm`.
/// They are compiled for the architecture-specific targets in SINFER_FP4_ARCHS (`120a` and/or
/// `121a`), and each cubin loads on exactly its own architecture: sm_120 (RTX 50 / RTX PRO) or
/// sm_121 (GB10 / DGX Spark). On any other device -- sm_100, or an sm_121 device under a
/// 120a-only build -- the driver JITs the fatbin's compute_89 PTX, whose FP4 body is __trap().
bool fp4_kernels_built_for(int sm) noexcept;

} // namespace sinfer::ops::detail
