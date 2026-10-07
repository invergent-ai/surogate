// Hopper block-scaled FP8 GEMM on DeepGEMM's sm90 1D2D kernel (src/third_party/deep_gemm, MIT;
// the kernel vLLM runs for fine-grained FP8 checkpoints on H100/H200), compiled ahead of time for a
// fixed set of tiles and chosen per call by DeepGEMM's own cost model. Operands are the engine's
// layouts that fp8_block_sm90_gemm.h documents, which are DeepGEMM's: activation codes [tokens, k]
// row-major, their scales k-block-major with rows padded to four tokens, weight codes [n, k] with
// one scale per 128 x 128 block, [n/128, k/128] row-major, and a BF16 [tokens, n] output.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail::fp8_block::sm90 {

struct DeepGemmOperands {
    const std::uint8_t* act_codes;
    const float* act_scales;
    const std::uint8_t* w_codes;
    const float* w_scales;
    void* out_bf16;
    bool residual; // out = W . x + out, summed in FP32 and rounded once
    std::int32_t tokens;
    std::int32_t n;
    std::int32_t k;
    cudaStream_t stream;
    // Set by sm90_gemm: launch nothing (so it runs its CUTLASS tiles) where DeepGEMM's own pick ran
    // slower than those on an H100 across 16 dense-model shapes and 33 to 2,048 tokens: a 64-row
    // tile past 64 tokens (1.0 to 1.5x their time; its one math warpgroup waits on the MMAs before
    // promoting each K block), and a 256-row tile with a residual (each thread seeds twice the
    // accumulators from the output before its first MMA; up to 1.2x).
    bool where_faster = false;
};

/// Runs the GEMM and returns true, or returns false having launched nothing when no compiled tile
/// takes the problem (alignment, a k past the tiles' staged scale capacity) or, with
/// `where_faster`, the pick is one of the tiles it names.
bool deepgemm(const DeepGemmOperands& o);

} // namespace sinfer::ops::detail::fp8_block::sm90
