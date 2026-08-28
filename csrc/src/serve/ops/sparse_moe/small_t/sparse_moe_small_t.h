#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "api/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kSparseMoeSmallTMin = 2;
// The fixed kernel domain covers the largest codec-specific small-T frontier.
inline constexpr std::int32_t kSparseMoeSmallTMax = 46;

// D3 CTA shape: one path per CTA, three paths per CTA (Qwen3.6's nine paths only), or every
// path of a token in one CTA.
enum class SparseMoeSmallTD3Schedule : std::uint8_t {
    Paths1,
    Paths3,
    PathsAll,
};

enum class SparseMoeSmallTD4Schedule : std::uint8_t {
    Rows1,
    Rows2,
    Rows4,
};

struct SparseMoeSmallTPlan {
    std::int32_t tokens                   = 0;
    std::size_t workspace_bytes           = 0;
    SparseMoeSmallTD3Schedule d3_schedule = SparseMoeSmallTD3Schedule::Paths3;
    SparseMoeSmallTD4Schedule d4_schedule = SparseMoeSmallTD4Schedule::Rows1;
};

struct SparseMoeSmallTWorkspace {
    Tensor token_ids;
    Tensor token_alpha;
    Tensor shared_scale;
    Tensor scratch;
};

inline constexpr std::int32_t kSparseMoeRouterPartitions = 4;

template <class Arena>
SparseMoeSmallTWorkspace allocate_sparse_moe_small_t_workspace(Arena& arena,
                                                               const SparseMoeGeometry& geometry,
                                                               std::int32_t tokens) {
    SparseMoeSmallTWorkspace out;
    const std::int32_t assignments = geometry.experts_per_token * tokens;
    out.token_ids                  = arena.alloc(DType::I32, {assignments}, 16);
    out.token_alpha                = arena.alloc(DType::FP32, {assignments}, 16);
    out.shared_scale               = arena.alloc(DType::FP32, {tokens}, 16);
    // S1 uses [T,router_rows,4] partial router scores. After S2, each token reuses its
    // [paths,intermediate] region for the routed and shared SwiGLU activations.
    const std::int64_t partials = static_cast<std::int64_t>(geometry.router_rows()) *
                                  kSparseMoeRouterPartitions;
    const std::int64_t activations =
        static_cast<std::int64_t>(geometry.paths()) * geometry.intermediate;
    const std::int32_t per_token = static_cast<std::int32_t>(std::max(partials, activations));
    out.scratch                  = arena.alloc(DType::FP32, {per_token, tokens}, 256);
    return out;
}

[[nodiscard]] bool sparse_moe_uses_small_t(std::int32_t tokens) noexcept;
[[nodiscard]] std::size_t sparse_moe_small_t_workspace_bytes(const SparseMoeGeometry& geometry,
                                                             std::int32_t tokens);
[[nodiscard]] SparseMoeSmallTPlan resolve_sparse_moe_small_t_plan(const SparseMoeGeometry& geometry,
                                                                  std::int32_t tokens,
                                                                  QType routed_gate_up,
                                                                  QType routed_down);

void sparse_moe_small_t_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                               const SparseMoeWeights& weights, Tensor& destination,
                               const SparseMoeSmallTPlan& plan,
                               const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
