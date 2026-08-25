#pragma once

// surogate vendor patch (PATCHES.md #21): derived NVFP4 prefill plane
// (native 4-bit profile).
//
// Same architecture as the FP8 plane (w8fp8_plane.h) one tier down: at
// first large-T use of a W8G32 parent the engine derives
//   fp4_codes[row, k/2]   e2m1 nibbles (low nibble = even k)
//   sf[row, k/16]         ue4m3 per-16 block scales
//   row_scales[row]       f32, rowmax / (448 * 6)
// so that w ~= e2m1(w / (row_scale * sf)) * row_scale * sf. Activations
// quantize per token the same way (token_scale = tokmax / (448 * 6),
// per-16 ue4m3, e2m1 nibbles). The GEMM runs sm_120a's
// mma.kind::mxf4nvf4.block_scale (hardware scale application, f32 in-core
// accumulation): measured 274-309 TF/s vs the FP8 folded kernel's 213-241.
//
// Quality class: NVFP4 W4A4 PTQ (the vLLM-NVFP4 checkpoint class). This is
// a PROFILE, not a default: the engine selects it only when the prefill
// quant mode is Fp4; decode always stays int8-exact W8.

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class PrefillQuantMode : std::uint8_t {
    Fp8,  // derived e4m3 plane (PATCHES #20) — default
    Fp4,  // derived NVFP4 plane (this file)
};

void w8_prefill_quant_set_mode(PrefillQuantMode mode) noexcept;
PrefillQuantMode w8_prefill_quant_mode() noexcept;

struct W4Fp4Plane {
    const std::uint8_t* codes;       // [n, k/2] e2m1 pairs
    const std::uint8_t* sf;          // [n, k/16] ue4m3
    const float* row_scales;         // [n]
};

// Derive-on-first-use registry, mirroring w8fp8_plane_for (enabled flag is
// shared via w8fp8_plane_set_enabled; the mode above picks the plane).
W4Fp4Plane w4fp4_plane_for(const Weight& weight, cudaStream_t stream);

struct W4Fp4QuantizedActivations {
    const std::uint8_t* codes;  // [tokens, hidden/2]
    const std::uint8_t* sf;     // [tokens, hidden/16]
    const float* scales;        // [tokens]
};

// Workspace: hidden/2 codes + hidden/16 sf + one f32 per token, 16B-aligned
// sections. Always <= w8a8_act_quant_bytes(hidden, tokens), so A8-sized
// workspaces are reused as-is.
W4Fp4QuantizedActivations w4fp4_act_quant(const Tensor& x, void* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
