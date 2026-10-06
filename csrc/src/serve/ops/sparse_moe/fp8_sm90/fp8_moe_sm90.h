#pragma once
// Routed experts stored as block-scaled FP8 (E4M3 codes, one FP32 scale per 128 x 128 block --
// the layout of Qwen's and DeepSeek's FP8 exports), run on Hopper as two CUTLASS 3.x ptr-array
// grouped GEMMs with blockwise scaling (wgmma + TMA, CUTLASS example 68's kernel). The
// activations are quantised to E4M3 per row per 128 values, the recipe vLLM's FP8 MoE uses, so
// both GEMMs are W8A8 with per-block weight scales and per-(row, 128) activation scales.
// Rounds of a few tokens skip the grouped GEMM, whose fixed cost dominates there: a GEMV per
// packed column reads the codes, dequantises them in registers and keeps the activations BF16.
//
// The round's routing is the prefill family's: the packed columns are the assignments sorted by
// expert, `expert_offsets[e] .. expert_offsets[e + 1]` are expert e's, and `column_token[c]` is
// the token column c came from. The GEMMs read the expert boundaries from device memory, so a
// round is capture-safe and its grid does not depend on the routing.
//
// Everything this module runs is sm_90a-specific; builds without 90a link a stub where
// `available()` is false and every other entry point throws.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::fp8_moe_sm90 {

/// The gate between the two GEMMs: `silu(gate) * up` or `gelu_tanh(gate) * up`.
enum class Activation : std::uint8_t {
    Swiglu,
    GegluTanh,
};

struct Geometry {
    std::int32_t hidden            = 0;
    std::int32_t experts           = 0;
    std::int32_t experts_per_token = 0;
    std::int32_t intermediate      = 0;
    Activation activation          = Activation::Swiglu;
};

/// One layer's routed experts. Every pointer is device memory.
struct Fp8RoutedExperts {
    /// [experts][2 * intermediate][hidden] E4M3 codes, an expert's rows ordered [gate; up].
    const std::uint8_t* gate_up_codes = nullptr;
    /// [experts][2 * intermediate / 128][hidden / 128] FP32, row-major per expert.
    const float* gate_up_scales = nullptr;
    /// [experts][hidden][intermediate] E4M3 codes.
    const std::uint8_t* down_codes = nullptr;
    /// [experts][hidden / 128][intermediate / 128] FP32.
    const float* down_scales = nullptr;
};

/// The tile family a round runs. `Auto` picks by the round's width; the others exist for the op
/// test's sweep and for SUROGATE_SERVE_MOE_FP8_TILE.
enum class Tile : std::uint8_t {
    Auto,
    Swap16,  ///< weights on M (128-row tiles), assignments on N in 16-column tiles
    Swap32,
    Swap64,
    Wide128, ///< assignments on M in 128-row tiles, weights on N in 128-column tiles
    Gemv,    ///< no grouped GEMM: a GEMV per column over BF16 activations, for narrow rounds
};

/// The widest round, in packed columns (tokens * experts_per_token), the GEMV path takes.
inline constexpr std::int64_t kGemvMaxAssignments = 256;

/// True when this build carries the sm_90a kernels and the current device is an sm_90 part.
[[nodiscard]] bool available() noexcept;

/// Whether the geometry has a kernel here: hidden and intermediate whole multiples of 128.
[[nodiscard]] bool supports(const Geometry& geometry) noexcept;

/// The tile `Auto` resolves to for a round of `tokens` tokens (SUROGATE_SERVE_MOE_FP8_TILE
/// included); never `Auto`.
[[nodiscard]] Tile resolved_tile(const Geometry& geometry, std::int32_t tokens);

/// Scratch `run` needs for rounds of up to `max_tokens` tokens, 256-byte aligned: the quantised
/// expert inputs and SwiGLU outputs with their scales, the GEMV path's BF16 SwiGLU outputs, the
/// GEMMs' per-expert argument arrays and CUTLASS's per-SM TMA descriptors.
[[nodiscard]] std::size_t workspace_bytes(const Geometry& geometry, std::int32_t max_tokens);

/// The BF16 scratch `run` writes the gate/up product into: [assignments][2 * intermediate].
/// Callers that own a large enough buffer (the prefill family's gathered-input block) pass it as
/// `gate_up_scratch`; it may alias `out`.
[[nodiscard]] constexpr std::size_t gate_up_scratch_bytes(const Geometry& geometry,
                                                          std::int32_t assignments) noexcept {
    return static_cast<std::size_t>(assignments) * 2 * geometry.intermediate * 2;
}

/// out[c][:] = expert_e(x[column_token[c]]) for every packed column c of expert e, BF16
/// [assignments][hidden] row-major, unweighted (the caller's reduce applies the router weights).
/// `x` is BF16 [tokens][hidden]. `assignments` = tokens * experts_per_token.
void run(const Geometry& geometry, const __nv_bfloat16* x, std::int32_t tokens,
         const std::int32_t* column_token, const std::int32_t* expert_offsets,
         const Fp8RoutedExperts& experts, void* workspace, std::size_t workspace_capacity,
         __nv_bfloat16* gate_up_scratch, __nv_bfloat16* out, cudaStream_t stream,
         Tile tile = Tile::Auto);

} // namespace sinfer::ops::detail::fp8_moe_sm90
