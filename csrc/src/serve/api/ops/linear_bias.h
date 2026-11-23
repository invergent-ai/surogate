#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime.h>

namespace sinfer::ops {
// out[N,T] = weight[N,K] * x[K,T] + bias[N]. x, bias and out are contiguous BF16,
// with the same positive dimensions/alignment as linear. Inputs and out must not
// overlap. BF16 weights accumulate the dot product and bias in FP32 before the
// output rounds to BF16. Other supported A16 weight formats round the projection
// before adding the bias. Uses the engine's existing GEMM workspace.
void linear_bias(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                 cudaStream_t stream);

enum class LinearEpilogue {
    Bias,         // out = weight * x + bias
    GeluTanhBias, // out = gelu_tanh(weight * x + bias)
};

// The projection and what follows it in one GEMM, the bias and the tanh GELU riding its output
// instead of each taking a pass over memory: out = epilogue(weight * x + bias), rounded to BF16
// once. Returns false, having enqueued nothing, when no fused route serves the call -- a weight
// that is not contiguous BF16, an active adapter (which applies to the projection alone),
// --batch-invariant, or a problem cuBLASLt has no fused algorithm for -- and the caller runs
// linear and the separate ops.
[[nodiscard]] bool try_linear_epilogue(const Tensor& x, const Weight& weight, const Tensor& bias,
                                       LinearEpilogue epilogue, Tensor& out, cudaStream_t stream);

// A residual projection: out += weight * x + bias, with no intermediate [N,T] buffer -- the bias
// goes into `out` first, then the GEMM accumulates into it (beta = 1). Same refusals as
// try_linear_epilogue.
[[nodiscard]] bool try_linear_bias_accumulate(const Tensor& x, const Weight& weight,
                                              const Tensor& bias, Tensor& out,
                                              cudaStream_t stream);
} // namespace sinfer::ops
