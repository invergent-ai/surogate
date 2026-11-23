#pragma once

// sinfer::ops::detail - MXFP8 operands and cuBLASLt's block-scaled GEMM over them.
//
// MXFP8 is the OCP Microscaling format: e4m3 values, each run of 32 along K sharing one
// power-of-two UE8M0 scale. Blackwell (compute capability 10.0 and up) multiplies it on the
// tensor cores at twice the BF16 rate with the scales applied in the MMA, so a BF16 projection
// that tolerates FP8 rounding (the vision tower's, under --vision-fp8) runs at FP8 rate with no
// separate scaling pass -- which a per-row or per-token FP8 scale would need on sm_12x, where
// cuBLASLt takes only scalar FP8 scales (fp8_cublaslt.h).
//
// A quantized matrix is [rows, k] with K contiguous: a weight [N, K], or an activation [K, T]
// read as T rows of K. K is zero-padded to a multiple of 32 so every scale block is whole, and
// the scales sit in the 128x4 tiled layout of cuBLASLt's VEC32_UE8M0 mode -- the layout
// nvfp4_tiled_scale_offset describes, at 32 values per group instead of 16.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

struct Mxfp8Matrix {
    void* codes          = nullptr; ///< rows x mxfp8_padded_k(k) e4m3, row-major
    void* scales         = nullptr; ///< mxfp8_scales_bytes(rows, k) tiled UE8M0
    std::int32_t rows    = 0;
    std::int32_t k       = 0;       ///< logical K; the code rows are padded to a multiple of 32
};

[[nodiscard]] constexpr std::int32_t mxfp8_padded_k(std::int32_t k) noexcept {
    return (k + 31) / 32 * 32;
}
[[nodiscard]] std::size_t mxfp8_codes_bytes(std::int32_t rows, std::int32_t k) noexcept;
[[nodiscard]] std::size_t mxfp8_scales_bytes(std::int32_t rows, std::int32_t k) noexcept;

/// Whether the current device multiplies block-scaled FP8 (compute capability 10.0 and up).
[[nodiscard]] bool mxfp8_device_supported();

enum class Mxfp8Transform : std::uint8_t { None, GeluTanh };

/// Quantizes `out.rows` rows of `out.k` BF16 values -- row r at `source + r * row_stride`
/// elements, 16-byte aligned -- into `out`'s codes and scales, after `transform` (tanh GELU:
/// an fc1 output becoming the fc2 input; tanh.approx, ~2^-11 relative). Each 32-value block takes the scale
/// 2^ceil(log2(amax / 448)), so no value saturates; K's padding is written as zeros.
void mxfp8_quantize(const void* source, std::int64_t row_stride, Mxfp8Transform transform,
                    Mxfp8Matrix& out, cudaStream_t stream);

/// Creates the current device's cuBLASLt handle and workspace; call before any capture.
void mxfp8_cublaslt_prewarm();

/// Whether cuBLASLt has an MXFP8 algorithm for out[rows, tokens] = W[rows, k] x X[tokens, k]^T
/// with (`bias`) a bias epilogue or (`accumulate`) beta = 1. Builds and caches the plan.
[[nodiscard]] bool mxfp8_gemm_available(std::int32_t rows, std::int32_t k, std::int32_t tokens,
                                        bool bias, bool accumulate);

/// out[weight.rows, x.rows] (BF16, contiguous, leading dimension weight.rows) =
///   weight * x^T (+ bias[weight.rows] BF16), or with `accumulate` out += weight * x^T (+ bias):
/// a residual projection in one GEMM. Unlike cuBLASLt's BF16 epilogues on sm_12x (legacy
/// kernels, 6-8x slower with beta = 1 at k = 4304), the block-scaled kernels take bias and
/// beta = 1 together at the plain GEMM's cost. `weight.k` and `x.k` must agree. Returns false,
/// having enqueued nothing, when mxfp8_gemm_available would.
[[nodiscard]] bool mxfp8_gemm(const Mxfp8Matrix& weight, const Mxfp8Matrix& x, const void* bias,
                              bool accumulate, void* out, cudaStream_t stream);

} // namespace sinfer::ops::detail
