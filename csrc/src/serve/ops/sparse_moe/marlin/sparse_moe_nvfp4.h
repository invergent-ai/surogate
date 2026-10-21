#pragma once

// The weight-only NVFP4 mixture: routed banks and shared expert NVFP4_A16 in Marlin tiles.
//
// This is vLLM's fused_marlin_moe for a ModelOpt W4A16_NVFP4 checkpoint, with the router and
// the always-on expert around it: an FP32 router over the BF16 hidden state, the softmax top-k
// every route of this op uses, moe_align_block_size, the gate/up GEMM read through the routing
// table, SiLU-and-multiply, the down GEMM with the router weight in its epilogue, and the shared
// expert's two dense Marlin GEMMs under its sigmoid gate; one kernel then sums the paths into
// the residual. Every width runs this, from one decode token to a prefill round in slices, so
// there is no other route for these weights to fall back to and none is needed.

#include "api/ops/sparse_moe.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

/// The widest slice of a round this route computes at once; wider rounds run in slices, so the
/// workspace holds one slice's routing and intermediates.
inline constexpr std::int32_t kSparseMoeNvfp4Slice = 1024;

/// Whether the routed banks are the weight-only NVFP4 profile this route serves.
[[nodiscard]] constexpr bool sparse_moe_nvfp4_profile(QType gate_up, QType down) noexcept {
    return gate_up == QType::NVFP4_A16 && down == QType::NVFP4_A16;
}

/// Whether this route serves `geometry`: the softmax top-8 of 256 routed experts and a gated
/// SiLU shared expert (or none) -- the Qwen3.5/3.6 mixtures, at any width.
[[nodiscard]] bool sparse_moe_nvfp4_geometry(const SparseMoeGeometry& geometry) noexcept;

/// Workspace one call of up to `max_tokens` tokens needs.
[[nodiscard]] std::size_t sparse_moe_nvfp4_workspace_bytes(const SparseMoeGeometry& geometry,
                                                           std::int32_t max_tokens);

/// The round, `destination += sparse_moe(x)`. The weights are validated by the wrapper; `hook`
/// may capture the routing (its `resolve` is refused there).
void sparse_moe_nvfp4(const SparseMoeGeometry& geometry, const Tensor& x, const Tensor& router_x,
                      const SparseMoeWeights& weights, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream,
                      const SparseMoeRoundHook* hook);

} // namespace sinfer::ops::detail
