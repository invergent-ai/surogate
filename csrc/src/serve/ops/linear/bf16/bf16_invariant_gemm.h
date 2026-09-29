#pragma once

// sinfer::ops::detail - batch-invariant BF16 GEMM (serve --batch-invariant).
//
// out[0:n, 0:tokens] = W[n,k] · x[k,tokens] (+ out when `accumulate`), BF16 operands, FP32
// accumulation, one BF16 rounding at the end. W is [n,k] with k contiguous, x is [k,tokens] with
// k contiguous, out has `ldc` elements per token column.
//
// Every output element is one FP32 accumulator that runs the m16n8k16 BF16 MMA over k in
// increasing 16-wide steps, starting from zero, with k zero-padded to a multiple of 64. No
// split-K, no stream-K, no cross-warp reduction. The tile shape is chosen from `tokens` and `n`
// for speed, but no tile changes that chain, so an element's bits depend on its own weight row
// and token column only -- not on how many other tokens share the launch, nor where in the
// launch its column sits. cuBLASLt picks its algorithm (tile, split-K, reduction scheme) per
// problem width, which is exactly what this route exists to avoid.

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail {

/// Shapes this route runs: k a positive multiple of 8 (16-byte rows), n positive, ldc >= n.
[[nodiscard]] bool bf16_invariant_gemm_supports(std::int32_t n, std::int32_t k,
                                                std::int32_t ldc) noexcept;

/// Pointers must be 16-byte aligned (as the cuBLASLt route already requires).
void bf16_invariant_gemm(const void* weight, std::int32_t n, std::int32_t k, const void* x,
                         std::int32_t tokens, void* out, std::int32_t ldc, bool accumulate,
                         cudaStream_t stream);

} // namespace sinfer::ops::detail
