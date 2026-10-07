// RTX Blackwell's and GB10's block-scaled FP8 GEMM: CUTLASS's sm_120 blockwise-scaling kernel
// (TMA loads, FP8 mma.sync, the block scales applied once per 128 of k), the kernel vLLM runs for
// fine-grained FP8 checkpoints on sm_120 (csrc/quantization/w8a8/cutlass/c3x/
// scaled_mm_blockwise_sm120_fp8_dispatch.cuh, vLLM, Apache-2.0), with swapped tiles for rounds
// narrower than 128 tokens (fp8_block_sm120_gemm.cu picks one per call).
//
// Operands are Hopper's (fp8_block_sm90_gemm.h): activation codes [tokens, k] row-major with one
// FP32 scale per token per 128 of k, k-block-major in rows of sm90_scale_stride(tokens); weight
// codes [n, k] with one FP32 scale per 128 x 128 block, [n/128, k/128] row-major, or one per row
// (the compressed-tensors per-channel kind); a BF16 [tokens, n] output, written, or accumulated in
// place with `residual`.
#pragma once

#include "ops/linear/fp8_block/fp8_block_sm90_gemm.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail::fp8_block {

/// True when this build carries the sm_12x kernel for the current device (an `120a`, `121a` or
/// family cubin it loads: RTX 50, RTX PRO 6000, GB10) and SUROGATE_SERVE_FP8_BLOCK_SM120 is not 0.
[[nodiscard]] bool sm120_gemm_available() noexcept;

/// Returns false, having launched nothing, when the problem cannot be configured (alignment, a
/// workspace the kernel would need); the caller then runs its own tile. `per_row`: the weight
/// has one FP32 scale per row, [n], for every k block, instead of one per 128 x 128 block.
bool sm120_gemm(const std::uint8_t* act_codes, const float* act_scales, const std::uint8_t* w_codes,
                const float* w_scales, bool per_row, void* out_bf16, bool residual,
                std::int32_t tokens, std::int32_t n, std::int32_t k, cudaStream_t stream);

} // namespace sinfer::ops::detail::fp8_block
