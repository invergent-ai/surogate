#pragma once

// sinfer::ops::detail - cuBLASLt route for BF16_CTRL problems outside the hand-tuned registry.
//
// out[N,T] = w[N,K] · x[K,T] with BF16 operands and FP32 accumulation. Plans are cached per
// device and problem, so after `bf16_cublaslt_prewarm` (and one call per problem shape) the
// route is graph-capturable: no allocation or host synchronisation on the capture path.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail {

/// Creates the current device's handle and workspace; call before stream capture.
void bf16_cublaslt_prewarm();

/// Caches the plan for one problem shape; call before stream capture for every shape used.
void bf16_cublaslt_prepare(std::int32_t rows, std::int32_t k, std::int32_t tokens);

/// `weight` is contiguous BF16_CTRL [rows,k] (k fastest), `x` contiguous BF16 [k,T], `out`
/// contiguous BF16 [rows,T]. T is any positive count.
void bf16_cublaslt_gemm(const Weight& weight, const Tensor& x, Tensor& out, cudaStream_t stream);

/// out += weight · x, same plan: beta is a per-call scalar, not part of the descriptors.
void bf16_cublaslt_gemm_accumulate(const Weight& weight, const Tensor& x, Tensor& out,
                                   cudaStream_t stream);

enum class Bf16Epilogue : std::int32_t { Bias = 1, GeluTanhBias = 2 };

/// out = epilogue(weight · x + bias) in one matmul: the bias and the tanh GELU ride the GEMM's
/// output instead of each taking a pass over memory. `bias` is contiguous BF16 [rows], 16-byte
/// aligned. Returns false, having enqueued nothing, when this route cannot serve the call --
/// --batch-invariant, an unaligned bias, or no cuBLASLt algorithm for the fused problem -- and
/// the caller runs the unfused ops.
///
/// There is deliberately no accumulating form: on sm_121 cuBLASLt serves an epilogue only from
/// its legacy kernels, and with beta = 1 at k = 4304 (a vision tower's fc2) those ran 6-8x
/// slower than the plain GEMM. A residual projection adds its bias to the residual and then
/// accumulates with bf16_cublaslt_gemm_accumulate, which keeps the fast kernels.
[[nodiscard]] bool bf16_cublaslt_gemm_epilogue(const Weight& weight, const Tensor& x,
                                               const Tensor& bias, Bf16Epilogue epilogue,
                                               Tensor& out, cudaStream_t stream);

/// out[0:n, 0:tokens] = alpha·(W·x) + beta·out for raw BF16 operands, with an explicit output
/// leading dimension so a caller may write a row range of a taller matrix. W is [n, k] with k
/// contiguous, x is [k, tokens] with k contiguous, out has `ldc` elements per column.
/// n, k and ldc must be multiples of 8 and every pointer 16-byte aligned.
void bf16_cublaslt_gemm_raw(const void* weight, std::int32_t n, std::int32_t k, const void* x,
                            std::int32_t tokens, void* out, std::int32_t ldc, float beta,
                            cudaStream_t stream);

} // namespace sinfer::ops::detail
