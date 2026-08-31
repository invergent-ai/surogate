#pragma once

// surogate vendor patch (PATCHES.md #25): cutlass sm_120a blockscaled NVFP4
// prefill GEMM (the vLLM/flashinfer kernel class; measured 679-813 TF/s at
// the 4B shapes vs 332-391 for the hand-rolled kt4 kernel — recipe from
// flashinfer's fp4_gemm_template_sm120.h, Apache-2.0; cutlass BSD-3).
//
// Operands: A = per-token e2m1 activations [tokens, k/2] row-major with
// ue4m3 block scales in the cutlass Sm1xx SF atom layout (outer scales
// FOLDED in — alpha is 1); B = the derived weight plane's e2m1 codes
// [n, k/2] plus its atom-layout folded SF. D is BF16 [tokens, n] row-major,
// which is exactly the engine's token-major store layout; the residual
// variant computes D = acc + C with C = D = the residual tensor.

#include <cstdint>
#include <cuda_runtime.h>

namespace sinfer::ops::detail {

// Padded atom-layout SF byte count for an [n, k] operand (Blk_MN 128, Blk_SF 4).
[[nodiscard]] constexpr std::size_t w4fp4_sf_atom_bytes(std::int64_t rows,
                                                        std::int64_t k) noexcept {
    const std::int64_t r = (rows + 127) / 128 * 128;
    const std::int64_t g = (k / 16 + 3) / 4 * 4;
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(g);
}

// Returns false when the launch could not be configured (caller falls back).
bool w4fp4_cutlass_gemm_store(const std::uint8_t* act_codes, const std::uint8_t* act_sf_atom,
                              const std::uint8_t* w_codes, const std::uint8_t* w_sf_atom,
                              const float* alpha_one, void* out_bf16, std::int32_t tokens,
                              std::int32_t n, std::int32_t k, cudaStream_t stream);

bool w4fp4_cutlass_gemm_residual(const std::uint8_t* act_codes, const std::uint8_t* act_sf_atom,
                                 const std::uint8_t* w_codes, const std::uint8_t* w_sf_atom,
                                 const float* alpha_one, void* residual_bf16, std::int32_t tokens,
                                 std::int32_t n, std::int32_t k, cudaStream_t stream);

} // namespace sinfer::ops::detail
