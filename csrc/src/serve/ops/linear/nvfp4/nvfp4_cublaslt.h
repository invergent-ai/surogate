#pragma once

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// cuBLASLt block-scaled FP4 route for prefill-width W4A4 GEMMs. The stored weight codes
// (row-major, K-contiguous e2m1 pairs) and the 128x4-tiled UE4M3 weight scales are consumed
// in place; the activation must be quantized with Nvfp4ScaleLayout::Tiled. Below
// kNvfp4CublasLtDefaultMinTokens the in-house small-T kernels are faster (they run at the
// weight-bandwidth limit); SUROGATE_SERVE_NVFP4_CUBLASLT=0 disables the route and
// SUROGATE_SERVE_NVFP4_CUBLASLT_MIN_TOKENS moves the threshold.
constexpr std::int32_t kNvfp4CublasLtDefaultMinTokens = 64;

bool nvfp4_cublaslt_route(std::int32_t tokens);

// Create this device's handle and workspace now. The state is otherwise built on first use,
// and a first use inside a CUDA graph capture cannot cudaMalloc: capture fails with
// cudaErrorStreamCaptureUnsupported instead of the route quietly initialising (#85).
void nvfp4_cublaslt_prewarm();

// out[rows x tokens] (token-major, leading dimension out_ld) =
//     alpha * W[row_begin, row_begin + rows) * X + beta * out
// row_begin and rows must be multiples of 128 so the scale tiles slice cleanly.
void nvfp4_cublaslt_gemm(const Weight& weight, std::int32_t row_begin, std::int32_t rows,
                         const std::uint8_t* activation_codes,
                         const std::uint8_t* activation_tiled_scales, __nv_bfloat16* out,
                         std::int32_t out_ld, std::int32_t tokens, float beta,
                         cudaStream_t stream);

} // namespace ninfer::ops::detail
