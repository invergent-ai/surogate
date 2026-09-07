#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail {

// A W8G32 scale row is `k / 32` binary16 values, so `k / 16` bytes, and every
// MMA tile family (the runtime-shaped row-split GEMM and the exact-T split-K
// bakes) stages it with a 16-byte `cp_async`. That is only aligned when
// `k % 256 == 0`; at `k = 1152` every odd row starts eight bytes short, which
// reads a neighbour's scales and eventually faults outright.
//
// Every shape registered before EmbeddingGemma happened to satisfy this -- the
// Qwen k values are all multiples of 256 -- so the requirement was never stated
// and nothing checked it. Each MMA launcher checks it rather than documents it,
// because the failure is silent: wrong numbers at small T, a misaligned-address
// fault only once a row crosses a page. A shape whose k violates it must stay
// on the SIMT routes.
inline constexpr std::int32_t kW8MmaScaleRowAlignmentK = 256;

// The MMA routes are given the same row-alignment floor the Marlin band needs, and the reason
// is what the shape that found it went on to prove.
//
// Gemma 4's 26B-A4B feed-forward (n = 2,112, k = 2,816) returned orthogonal results above
// sixteen tokens. The cause was the *Marlin* band, which admitted `n % 64 == 0` while its
// kernel tiles rows in 128s -- see `kMarlinRowTile` in `marlin/marlin_plane.cu`, where it is
// fixed. That interception also meant the untuned MMA route was never exercised for this shape
// at those widths: below the band it takes SIMT, inside the band it never ran.
//
// So this is a floor on an *untested* combination, not a diagnosis: a row count the MMA path
// has no measurement behind takes the SIMT route, exactly as one that misses the K constraint
// does. It costs a tuned kernel for shapes nobody has measured and buys the guarantee that the
// next unaligned width is slow rather than wrong.
inline constexpr std::int32_t kW8MmaRowAlignmentN = 128;

using W8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_decode_r4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_t_splitk(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_t_composite(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_dflash_medium(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_medium_splitk_c144(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_simt_r8_c4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_simt_r8_c8(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_mma_r32_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c112(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c112(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r128_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64x16_c48_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_exact_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r48_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r48_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace sinfer::ops::detail
