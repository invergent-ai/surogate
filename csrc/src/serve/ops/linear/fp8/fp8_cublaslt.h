#pragma once

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// cuBLASLt FP8 route for prefill-width W8A8 GEMMs on FP8_E4M3FN_ROW_BF16S weights. The
// weight codes (row-major, K contiguous) and the activation codes from the A8 quantizer are
// consumed as they are; cuBLASLt on this device only accepts scalar FP8 scales, so the GEMM
// accumulates raw codes into an fp32 staging and an epilogue applies the weight's per-row
// bf16 scale and the activation's per-token scale exactly once - the in-house kernel's
// arithmetic with cuBLASLt's accumulation. Below kFp8CublasLtDefaultMinTokens the in-house
// small-T kernels stay (they run at the weight-bandwidth limit). SUROGATE_SERVE_FP8_CUBLASLT=0
// disables the route; SUROGATE_SERVE_FP8_CUBLASLT_MIN_TOKENS moves the threshold.
// One above the concurrency cap: a decode round (at most 64 columns) never takes the route,
// so its workspaces stay as planned; prefill tails from 65 tokens up do.
constexpr std::int32_t kFp8CublasLtDefaultMinTokens = 65;

bool fp8_cublaslt_route(std::int32_t tokens);

// See nvfp4_cublaslt_prewarm: build the device state before any capture can reach it (#85).
void fp8_cublaslt_prewarm();

// Bytes of fp32 staging the route needs for `rows` output rows and `tokens` columns.
inline std::size_t fp8_cublaslt_staging_bytes(std::int32_t rows, std::int32_t tokens) {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(tokens) * sizeof(float);
}

// staging[rows x tokens] (token-major, leading dimension rows) = W[row_begin, row_begin + rows) * X
// over raw e4m3 codes; the caller applies the scales with fp8_cublaslt_finish.
void fp8_cublaslt_gemm(const Weight& weight, std::int32_t row_begin, std::int32_t rows,
                       const std::uint8_t* activation_codes, float* staging, std::int32_t tokens,
                       cudaStream_t stream);

// out[r, t] = staging[r, t] * weight_row_scale[row_begin + r] * activation_scale[t]
//             (+ out[r, t] when accumulate), written as bf16 with leading dimension out_ld.
void fp8_cublaslt_finish(const float* staging, const __nv_bfloat16* weight_row_scales,
                         std::int32_t row_begin, std::int32_t rows, const float* activation_scales,
                         std::int32_t tokens, __nv_bfloat16* out, std::int32_t out_ld,
                         bool accumulate, cudaStream_t stream);

} // namespace ninfer::ops::detail
