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

/// A stacked bank of adapters for one projection: every slot padded to the same
/// (max_rank, k) and (n, max_rank), so a token's slot index is the only thing
/// that varies and the launch geometry is constant.
///
/// The padding is what makes many adapters cheaper than many code paths. It is
/// also what makes the delta capturable: a graph records one launch whose shape
/// does not depend on which adapters happen to be in the batch.
struct LoraBank {
    const void* a         = nullptr; ///< BF16 [slots, max_rank, k], alpha/r folded in
    const void* b         = nullptr; ///< BF16 [slots, n, max_rank]
    std::int64_t a_stride = 0;       ///< elements between slots in `a`
    std::int64_t b_stride = 0;       ///< elements between slots in `b`
    std::int32_t rank     = 0;       ///< max_rank; a shorter adapter is zero-padded
    std::int32_t n        = 0;
    std::int32_t k        = 0;
};

/// out[n, T] += B[ids[t]] · (A[ids[t]] · x[:, t]), per token.
///
/// `ids` is device I32 [T]; a negative entry is a base-model token and
/// contributes nothing, so a round mixing adapted and unadapted requests takes
/// one launch rather than two. `scratch` is BF16 and at least rank * T long, and
/// must be storage that outlives a graph replay -- a buffer allocated per call
/// inside a captured region is baked in by address, which is how this path first
/// went wrong.
/// `ids` selects per token; pass an empty tensor and a non-negative
/// `uniform_slot` when the whole round belongs to one adapter, which is what a
/// prefill chunk is.
void lora_delta_batched(const Tensor& x, const LoraBank& bank, const Tensor& ids,
                        const std::int32_t* uniform_slot, Tensor& out, Tensor& scratch,
                        cudaStream_t stream);

/// BF16 scratch elements `lora_delta_batched` needs for a round of `tokens`.
[[nodiscard]] std::size_t lora_batched_workspace_elements(std::int32_t rank,
                                                          std::int32_t tokens) noexcept;

/// Caches the cuBLASLt plans both GEMMs need, for a rank and token count, before
/// stream capture. Applying an adapter inside a captured graph without this
/// would allocate on the capture path.
void lora_prepare(std::int32_t n, std::int32_t k, std::int32_t rank, std::int32_t tokens);

} // namespace sinfer::ops
