#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <array>
#include <cstdint>

namespace sinfer::ops {

/// One layer's per-head query/key RMSNorm and the RoPE after it (qk_norm_rope).
struct QkNormRope {
    /// BF16 [head_dim, heads, T], contiguous, and the planes the result goes to (the same shape;
    /// not overlapping the inputs). The gains are BF16 [head_dim].
    const Tensor* q      = nullptr;
    const Tensor* q_norm = nullptr;
    Tensor* q_out        = nullptr;
    /// Null for a layer that reuses an earlier layer's keys: only the queries are normalised and
    /// rotated. `k_heads` is the key head count either way, the one the separate rope is given.
    const Tensor* k      = nullptr;
    const Tensor* k_norm = nullptr;
    Tensor* k_out        = nullptr;
    std::int32_t k_heads = 0;
    float eps            = 0.0F;
    bool unit_offset     = false;
    /// I32 [T] or [T,3], as ops::rope takes them.
    const Tensor* positions = nullptr;
    int rotary_dim          = 0;
    int active_pairs        = 0;
    float theta             = 0.0F;
    /// Nonzero: the rotation is rope_interleaved's with these temporal/height/width sections.
    std::array<int, 3> sections{};
    /// Optional FP32 [2, rotary_dim / 2, positions] rope_table built for this rotation: the
    /// coefficients are read from it rather than derived per token, and the result is the bits
    /// ops::rope_from_table leaves after the two norms. [T] positions within the table, no
    /// sections; `theta` and `active_pairs` are still validated, and are the table's.
    const Tensor* table = nullptr;
};

/**
 * `q_out = rope(rmsnorm(q, q_norm))` and the same for the keys, in one launch: the bits
 * ops::rmsnorm(q, q_norm, eps, unit_offset, q_out) and ops::rmsnorm(k, ...) followed by
 * ops::rope(positions, rotary_dim, active_pairs, theta, q_out, k_out) -- or ops::rope_interleaved
 * with `sections`, or ops::rope_from_table with `table` -- leave in the output planes, exactly.
 * Two norm launches and a rope launch per attention layer become one (vLLM fuses the same pair).
 *
 * Returns false, launching nothing, outside its domain, and the caller runs the separate ops:
 * head dims 64, 128, 192 and 256 on the rmsnorm kernels' aligned paths, a rotation whose partner
 * channels (rotary_dim / 2 apart) sit in the same lane or within the first 64 channels (rotary_dim
 * a multiple of 128, or a power of two from 4 to 64), [T] or [T,3] positions.
 * SUROGATE_SERVE_QK_NORM_ROPE=0 declines every call.
 */
[[nodiscard]] bool qk_norm_rope(const QkNormRope& args, cudaStream_t stream);

} // namespace sinfer::ops
