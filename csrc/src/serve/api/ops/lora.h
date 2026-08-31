#pragma once

// Low-rank adapter deltas applied to an already-computed projection.
//
// A LoRA adapter adds `scale * B @ (A @ x)` to a projection's output, where A is
// `[r, K]` and B is `[N, r]` for a rank r that is two orders of magnitude below
// N and K. That shape is why this is an addition rather than a change to the
// projection itself: the base weight stays quantised, fused and untouched, and
// the delta rides on two small BF16 GEMMs afterwards. An engine whose every
// projection is a geometry-specialised kernel could not have taken the other
// route without rewriting all of them.
//
// The intermediate `A @ x` is `[r, T]` -- a few thousand values -- so it is
// scratch the caller owns, sized by `lora_workspace_elements`.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace sinfer::ops {

/// One adapter's weights for one projection, resident on the device.
///
/// `a` is BF16 [rank, k] and `b` is BF16 [n, rank], both contiguous with the
/// second extent fastest, matching the layout `bf16_cublaslt_gemm` consumes.
/// PEFT's `lora_alpha / r` is folded into `a` at load, so no scale rides along
/// here: a scale applied per call would be a multiply over the whole delta every
/// projection of every layer, to save one multiply over A once.
struct LoraWeights {
    Weight a;
    Weight b;
    std::int32_t rank = 0;
};

/// BF16 scratch elements needed to apply a rank-`rank` adapter producing `n` rows
/// over `tokens` columns.
[[nodiscard]] std::size_t lora_workspace_elements(std::int32_t rank, std::int32_t n,
                                                  std::int32_t tokens) noexcept;

/// out[n, T] += scale * b[n, r] @ (a[r, k] @ x[k, T]).
///
/// `x` is the same activation the base projection consumed and `out` the tensor
/// it wrote, so the call sits directly after it. `scratch` is BF16 and at least
/// `lora_workspace_elements(rank, n, T)` long. Shapes are checked against the
/// weights: a delta silently applied to the wrong projection would read as a
/// quality regression, not as an error.
void lora_delta(const Tensor& x, const LoraWeights& lora, Tensor& out, Tensor& scratch,
                cudaStream_t stream);

/// Caches the cuBLASLt plans both GEMMs need, for a rank and token count, before
/// stream capture. Applying an adapter inside a captured graph without this
/// would allocate on the capture path.
void lora_prepare(std::int32_t n, std::int32_t k, std::int32_t rank, std::int32_t tokens);

} // namespace sinfer::ops
