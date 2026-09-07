#pragma once

// surogate vendor patch (PATCHES.md #20): derived FP8-e4m3 prefill plane for
// W8G32 weights.
//
// Motivation (measured, see PATCHES #19/#20): the W8A8-int IMMA kernel is
// bound by its per-group software scale tail, not by MMA throughput — an
// e4m3/F16-acc mma swap alone changes nothing, but FOLDING the group scales
// into per-row-renormalized e4m3 weights removes the tail entirely and wins
// 12-16% at every prefill shape.
//
// The plane is DERIVED, not a format: the artifact stays W8G32 (decode keeps
// int8-exact dequant), and at first large-T use of a weight the engine
// materializes:
//   fp8_codes[row, k]  e4m3, value = w[row, i] / rowmax(row)  (|w~| <= 1.0)
//   row_scales[row]    f32,  rowmax(row) = max_i |code * group_scale|
// Activations quantize per token to e4m3 at x~ = x / xmax * 448 with
// x_scale = xmax / 448, so an f16 accumulator chained across one 64-wide
// k-tile is bounded by 2 * 32 * 1.0 * 448 = 28672 < 65504 (no satfinite
// clamping by construction). Quality class: FP8 per-row (vLLM-FP8-like),
// prefill only.
//
// The registry is process-global and keyed by the device codes pointer.
// Derivation is opt-in: the engine enables it (w8fp8_plane_set_enabled) so
// op tests keep int8-exact numerics by default. A VRAM guard skips
// derivation (falling back to IMMA) when free memory is below twice the
// plane size.

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

struct W8Fp8Plane {
    const std::uint8_t* codes;  // [n, k] e4m3, row-major, 16B-aligned
    const float* row_scales;    // [n]
};

// Master switch (default off). The engine turns it on at construction;
// SUROGATE_SERVE_FP8_PREFILL=0 in the environment vetoes it.
void w8fp8_plane_set_enabled(bool enabled) noexcept;
bool w8fp8_plane_enabled() noexcept;

// Total device bytes held by derived planes (FP8 + FP4 registries). The
// engine's graph-preparation accounting subtracts this: planes carry their
// own VRAM guard and are not graph memory.
std::size_t w8_derived_plane_bytes() noexcept;

// Compute capability of the active device (major*10+minor), cached.
int w8_device_compute_capability() noexcept;

// Returns the derived plane for `weight` (deriving it on first call), or
// nullptr codes when the plane is disabled, the weight is not an admitted
// W8G32 row-split parent, or the VRAM guard declined. Derivation runs on
// `stream` and is synchronized before publication; concurrent callers are
// serialized by an internal mutex.
W8Fp8Plane w8fp8_plane_for(const Weight& weight, cudaStream_t stream);

// Per-token e4m3 activation quantization (x~ = x / xmax * 448, scale =
// xmax / 448). Same workspace layout and byte count as w8a8_act_quant
// (codes are 1 byte either way), so callers reuse w8a8_act_quant_bytes.
struct W8Fp8QuantizedActivations {
    const std::uint8_t* codes;  // [tokens, hidden] e4m3
    const float* scales;        // [tokens] xmax / 448
};

W8Fp8QuantizedActivations w8fp8_act_quant(const Tensor& x, void* workspace, cudaStream_t stream);

} // namespace sinfer::ops::detail
