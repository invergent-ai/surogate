// Hopper's block-scaled FP8 GEMM: the CUTLASS 3.x warp-specialized wgmma kernel vLLM runs for
// fine-grained FP8 checkpoints on sm_90 (csrc/quantization/w8a8/cutlass/c3x/
// scaled_mm_blockwise_sm90_fp8_dispatch.cuh, vLLM, Apache-2.0), with vLLM's two tile choices.
//
// Operands, in the engine's own layouts:
//   activations  E4M3 codes [tokens, k] row-major, one FP32 scale per token per 128 of k stored
//                k-block-major: scale(t, kb) = act_scales[kb * tokens + t];
//   weight       E4M3 codes [n, k] row-major (the artifact's [n, k]), one FP32 scale per
//                128 x 128 block, [n/128, k/128] row-major;
//   out          BF16 [tokens, n] row-major -- the engine's [n, T] token-major store.
// With `residual` the kernel computes out = acc + residual in place (out == residual).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail::fp8_block {

/// True when this build carries the sm_90a kernel and the current device is an sm_90 part
/// (H100, H200, GH200). The kernel is architecture-specific: it loads on exactly sm_90.
[[nodiscard]] bool sm90_gemm_available() noexcept;

/// Returns false, having launched nothing, when the problem cannot be configured (alignment,
/// a workspace the kernel would need); the caller then runs its own tile.
bool sm90_gemm(const std::uint8_t* act_codes, const float* act_scales, const std::uint8_t* w_codes,
               const float* w_scales, void* out_bf16, bool residual, std::int32_t tokens,
               std::int32_t n, std::int32_t k, cudaStream_t stream);

} // namespace sinfer::ops::detail::fp8_block
