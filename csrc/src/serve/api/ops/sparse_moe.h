#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The closed geometry of one sparse-MoE family instance. The kernels are compiled per
 * registered geometry; the wrapper derives the geometry from the weights and refuses any
 * combination that is not registered.
 */
struct SparseMoeGeometry {
    std::int32_t hidden            = 0;
    std::int32_t experts           = 0;
    std::int32_t experts_per_token = 0;
    std::int32_t intermediate      = 0;

    [[nodiscard]] constexpr std::int32_t router_rows() const noexcept { return experts + 1; }
    [[nodiscard]] constexpr std::int32_t expert_rows() const noexcept { return 2 * intermediate; }
    [[nodiscard]] constexpr std::int32_t routed_gate_rows() const noexcept {
        return experts * expert_rows();
    }
    [[nodiscard]] constexpr std::int32_t routed_down_rows() const noexcept {
        return experts * hidden;
    }
    [[nodiscard]] constexpr std::int32_t paths() const noexcept { return experts_per_token + 1; }

    friend constexpr bool operator==(const SparseMoeGeometry&, const SparseMoeGeometry&) = default;
};

/// Qwen3.5/3.6 MoE (35B-A3B and the 4B/2B MTP heads): 256 experts, top-8, FFN 512, hidden 2048.
inline constexpr SparseMoeGeometry kSparseMoeQwen36Geometry{2048, 256, 8, 512};
/// Qwen3.8-Flash-Next: 512 experts, top-10, FFN 640, hidden 2560.
inline constexpr SparseMoeGeometry kSparseMoeFlashNextGeometry{2560, 512, 10, 640};

struct SparseMoeWeights {
    Weight router_shared_gate;
    Weight routed_gate_up;
    Weight routed_down;
    Weight shared_gate_up;
    Weight shared_down;
    /// Routed experts selected per token; with the weight shapes this fixes the geometry.
    std::int32_t experts_per_token = 0;
};

/// The geometry implied by the weights (router rows, hidden, shared-down width, top-k).
[[nodiscard]] SparseMoeGeometry sparse_moe_geometry(const SparseMoeWeights& weights);

enum class SparseMoeEpilogue : std::uint8_t {
    AddResidual,
};

/**
 * Returns the transient capacity required by SparseMoe for every T in the inclusive
 * [min_tokens,max_tokens] interval. The routed QTypes are the fixed implementation profile.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(const SparseMoeGeometry& geometry,
                                                              QType routed_gate_up,
                                                              QType routed_down,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Closed sparse-MoE Op over the registered geometries (kSparseMoeQwen36Geometry,
 * kSparseMoeFlashNextGeometry); the text below names the Qwen3.6 instance.
 *
 * For contiguous BF16 x [2048,T] and destination [2048,T] with T>0, 256 routed experts, top-8
 * selection, and one always-on shared expert, this Op owns router projection and selection,
 * selected routed and shared SwiGLU projections, down projections, their merge, and the
 * AddResidual epilogue independently for every token column. At an exact top-8 boundary tie the
 * lower expert id wins. destination is the only observable mutation: its incoming value is the
 * residual and its outgoing value is the BF16 sparse-MoE result plus that residual.
 *
 * The complete mathematical oracle starts from represented BF16 inputs, exact stored-weight
 * decode, and evaluates the logical formula naively in FP32/FP64. Scores, route weights, expert
 * activations, workspace representation, reduction association, and scale placement are private
 * execution choices rather than semantic rounding boundaries.
 *
 * The five weights have the exact registered shapes: BF16 router/shared gate [257,2048], routed
 * gate/up [256*1024,2048], routed down [256*2048,512], shared gate/up [1024,2048], and shared down
 * [2048,512]. Admitted codec profiles are Q4+Q5, Q4+Q6, and W8+W8 for the two routed banks; both
 * shared banks are W8. Expert e directly selects its stored row spans; no selected-weight gather
 * or repack occurs.
 *
 * Every positive T is supported.
 *
 * x, destination, all weight planes, and live workspace must be pairwise non-overlapping.
 * Execution is enqueued on stream without host synchronization. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
