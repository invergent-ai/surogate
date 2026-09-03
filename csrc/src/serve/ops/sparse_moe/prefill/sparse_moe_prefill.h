#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "api/ops/sparse_moe.h"
#include "ops/sparse_moe/trtllm/trtllm_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace sinfer::ops::detail {

// RTX 5090 codec frontiers balance trace-like and independent expert distributions. The public
// workspace query starts at the earliest codec-specific prefill route.
inline constexpr std::int32_t kSparseMoePrefillQ4Q5Min      = 47;
inline constexpr std::int32_t kSparseMoePrefillQ4Q6Min      = 47;
inline constexpr std::int32_t kSparseMoePrefillW8W8Min      = 20;
/// GGML K-quants share the Q4/Q5 tiling, so they cross over where those do; measured, not
/// assumed -- below this the small-T slices win.
inline constexpr std::int32_t kSparseMoePrefillGgmlKMin     = 47;
inline constexpr std::int32_t kSparseMoePrefillWideMin      = 768;
inline constexpr std::int32_t kSparseMoePrefillSliceMax     = 4096;
inline constexpr std::int32_t kSparseMoeRouteTileTokens     = 8;
// Router logits padded to a 16-byte-aligned per-token stride (257 -> 260, 513 -> 516).
[[nodiscard]] constexpr std::int32_t
sparse_moe_router_score_rows(const SparseMoeGeometry& geometry) noexcept {
    return (geometry.router_rows() + 3) / 4 * 4;
}

struct SparseMoePrefillPlan {
    std::int32_t tokens         = 0;
    std::int32_t slice_tokens   = 0;
    std::size_t workspace_bytes = 0;
    /// The routed experts run through the vendored TRT-LLM runner rather than this family's own
    /// gather/gate-up/down/reduce chain. Set for the NVFP4 routed profile, whose weights are
    /// stored in that runner's [up; gate] row order and have no kernel of ours.
    bool routed_trtllm = false;
    /// The routed K-quant experts run on the int8 tensor-core route: activations quantised to
    /// int8 per 32 values with a (scale, sum) pair each, the way llama.cpp's MMQ feeds them,
    /// and the weights' own codes as the other MMA operand. Set for the Q4_K/Q5_K/Q6_K routed
    /// profile unless `SUROGATE_SERVE_MOE_INT8=0` asks for the BF16-activation kernels.
    bool routed_int8 = false;
};

/// Whether a routed codec pair takes the int8 tensor-core route.
[[nodiscard]] bool sparse_moe_routed_int8_profile(QType routed_gate_up, QType routed_down) noexcept;

struct SparseMoePrefillWorkspace {
    Tensor token_ids;
    Tensor token_alpha;
    // Selection first writes a rank local to one routing tile. Gather replaces
    // it in place with the inverse map from token/route slot to packed column.
    Tensor packed_index;
    Tensor shared_scale;
    Tensor tile_counts;
    Tensor tile_bases;
    Tensor expert_offsets;
    Tensor route_job_experts;
    Tensor route_job_columns;
    // A negative count selects the token-oriented adaptive route; its magnitude is the unused
    // grouped-route job count. Nonnegative values select the normal grouped route.
    Tensor route_job_count;

    // The three large allocations are lifetime unions:
    //   router scores FP32 <-> shared SwiGLU BF16
    //   gathered X BF16    <-> routed down output BF16
    //   routed SwiGLU BF16 <-> routed FP32 token reduction
    Tensor score_storage;
    Tensor shared_activation;
    Tensor grouped_io;
    Tensor routed_storage;
    Tensor routed_sum;
    /// Only for a `routed_int8` plan. `column_token` names the token each packed column came
    /// from, so the gate/up input is quantised once per token and gathered by the kernel; the
    /// down input is per assignment because the SwiGLU output is. Each `*_ds` plane holds one
    /// half2 of (scale, sum) per 32 values, stored as FP16 pairs.
    Tensor column_token;
    Tensor act_codes;
    Tensor act_ds;
    Tensor mid_codes;
    Tensor mid_ds;
    /// Only allocated for a `routed_trtllm` plan: the runner's own scratch, its BF16 output block
    /// and its permutation map, laid out by `trtllm_moe::workspace_bytes`.
    DeviceSpan trtllm_workspace;
};

template <class Arena>
SparseMoePrefillWorkspace allocate_sparse_moe_prefill_workspace(Arena& arena,
                                                                const SparseMoeGeometry& geometry,
                                                                std::int32_t capacity_tokens,
                                                                bool routed_trtllm = false,
                                                                bool routed_int8   = false) {
    SparseMoePrefillWorkspace out;
    const std::int32_t assignments = geometry.experts_per_token * capacity_tokens;
    const std::int32_t experts     = geometry.experts;
    const std::int32_t hidden      = geometry.hidden;
    const std::int32_t inter       = geometry.intermediate;
    const std::int32_t route_tiles =
        (capacity_tokens + kSparseMoeRouteTileTokens - 1) / kSparseMoeRouteTileTokens;

    out.token_ids      = arena.alloc(DType::I32, {assignments}, 256);
    out.token_alpha    = arena.alloc(DType::FP32, {assignments}, 256);
    out.packed_index   = arena.alloc(DType::I32, {assignments}, 256);
    out.shared_scale   = arena.alloc(DType::FP32, {capacity_tokens}, 256);
    out.tile_counts    = arena.alloc(DType::I32, {experts, route_tiles}, 256);
    out.tile_bases     = arena.alloc(DType::I32, {experts, route_tiles}, 256);
    out.expert_offsets = arena.alloc(DType::I32, {experts + 1}, 256);
    // A route job is one nonempty expert column tile. The bound is for the
    // narrowest prefill tile (32 assignments) and includes every expert tail.
    const std::int32_t max_route_jobs = assignments / 32 + experts;
    out.route_job_experts             = arena.alloc(DType::I32, {max_route_jobs}, 256);
    out.route_job_columns             = arena.alloc(DType::I32, {max_route_jobs}, 256);
    out.route_job_count               = arena.alloc(DType::I32, {1}, 256);

    // Lifetime unions must hold both tenants; the per-token byte counts are checked here so a
    // new geometry cannot silently overrun them.
    const std::int32_t score_rows = sparse_moe_router_score_rows(geometry);
    if (static_cast<std::int64_t>(score_rows) * 4 < static_cast<std::int64_t>(inter) * 2 ||
        static_cast<std::int64_t>(inter) * geometry.experts_per_token * 2 <
            static_cast<std::int64_t>(hidden) * 4 ||
        static_cast<std::int64_t>(hidden) * geometry.experts_per_token * 2 <
            static_cast<std::int64_t>(geometry.paths()) * inter * 4) {
        throw std::invalid_argument("sparse_moe prefill: geometry breaks a workspace union");
    }
    out.score_storage     = arena.alloc(DType::FP32, {score_rows, capacity_tokens}, 256);
    out.shared_activation = Tensor(out.score_storage.data, DType::BF16, {inter, capacity_tokens});

    out.grouped_io = arena.alloc(DType::BF16, {hidden, assignments}, 256);

    out.routed_storage = arena.alloc(DType::BF16, {inter, assignments}, 256);
    out.routed_sum     = Tensor(out.routed_storage.data, DType::FP32, {hidden, capacity_tokens});
    if (routed_int8) {
        out.column_token = arena.alloc(DType::I32, {assignments}, 256);
        out.act_codes    = arena.alloc(DType::I8, {hidden, capacity_tokens}, 256);
        out.act_ds       = arena.alloc(DType::FP16, {2 * (hidden / 32), capacity_tokens}, 256);
        out.mid_codes    = arena.alloc(DType::I8, {inter, assignments}, 256);
        out.mid_ds       = arena.alloc(DType::FP16, {2 * (inter / 32), assignments}, 256);
    }
    if (routed_trtllm) {
        out.trtllm_workspace =
            arena.alloc_bytes(trtllm_moe::workspace_bytes(
                                  trtllm_moe::Geometry{geometry.hidden, geometry.experts,
                                                       geometry.experts_per_token,
                                                       geometry.intermediate},
                                  capacity_tokens),
                              256);
    }
    return out;
}

[[nodiscard]] bool sparse_moe_uses_prefill(std::int32_t tokens, QType routed_gate_up,
                                           QType routed_down) noexcept;
[[nodiscard]] std::size_t sparse_moe_prefill_workspace_bytes(const SparseMoeGeometry& geometry,
                                                             std::int32_t max_tokens,
                                                             bool routed_trtllm,
                                                             bool routed_int8 = false);
[[nodiscard]] SparseMoePrefillPlan resolve_sparse_moe_prefill_plan(const SparseMoeGeometry& geometry,
                                                                   std::int32_t tokens,
                                                                   QType routed_gate_up,
                                                                   QType routed_down);

void sparse_moe_prefill_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                               const SparseMoeWeights& weights, Tensor& destination,
                               const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, cudaStream_t stream,
                               const SparseMoeRoundHook* hook = nullptr);

} // namespace sinfer::ops::detail
