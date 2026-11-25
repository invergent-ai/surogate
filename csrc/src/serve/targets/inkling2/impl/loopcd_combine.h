#pragma once

// The LoopCD combine kernel's declaration (loopcd.h), in a header of its own: the kernel's .cu cannot include the target
// config (host-only family headers).

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::targets::inkling2::detail {

/// logits[:, t] (ld `ldo`) from the softcapped head columns `pairs` FP32 [rows, 2 n] (column 2t = z, 2t + 1 = z_ref):
/// z + w (z - z_ref) over the first `domain` rows, z beyond (padded rows). w = `strength` (fixed) or
/// strength * (1 - (p1 - p2)) with p = softmax(z) over the domain (adaptive).
void loopcd_combine(const float* pairs, std::int32_t rows, std::int32_t domain, std::int32_t n, bool adaptive,
                    float strength, float* logits, std::int32_t ldo, cudaStream_t stream);

} // namespace sinfer::targets::inkling2::detail
