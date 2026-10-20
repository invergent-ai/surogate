#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "api/ops/sparse_moe.h"
#include "ops/sparse_moe/shared.h"
#include "api/ops/lora.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <algorithm>
#include <cstdint>

namespace sinfer::ops::detail {

inline constexpr int kMoeLoraMaxRank = 256;

/// Adapter rounds at least this wide take the expert-major kernels (sparse_moe_decode_body.inc,
/// "Adapter rounds, expert-major") where the routed codec has one: each distinct expert's rows read
/// once for all of its tokens instead of once per (token, path), with every token's sums in the
/// per-path kernels' own order. Narrower rounds -- decode -- keep the per-path kernels.
inline constexpr std::int32_t kSparseMoeAdapterGroupedMin = 64;
/// The widest adapter round one set of launches carries; a wider call runs in slices, which bounds
/// the per-(token, path) output partials the expert-major down projection keeps for its combine.
inline constexpr std::int32_t kSparseMoeAdapterSlice = 1024;
/// Entries (token, path) one expert-major tile carries at most: a hot expert's tokens spread
/// over several blocks rather than one block walking all of them.
inline constexpr std::int32_t kSparseMoeAdapterTile = 64;

/// The routed tiles an expert-major adapter round of `paths` (token, path) entries can need:
/// every used expert's last tile, plus one per whole tile of entries.
[[nodiscard]] constexpr std::int32_t sparse_moe_adapter_routed_tiles(std::int32_t paths,
                                                                     std::int32_t experts) noexcept {
    return std::min(paths, experts) + paths / kSparseMoeAdapterTile;
}

enum class SparseMoeSmallTD3Schedule : std::uint8_t;
enum class SparseMoeSmallTD4Schedule : std::uint8_t;

struct SparseMoeDecodePlan {
    std::size_t workspace_bytes = 0;
};

struct SparseMoeDecodeWorkspace {
    Tensor ids;
    Tensor alpha;
    Tensor shared_scale;
    Tensor scratch;
    Tensor lora_low;
    std::int32_t slot_stride = 0;
    /// Rank rows the adapter kernels launch a block for. Every bank of a store is padded to
    /// the store's own max rank, so that is all there is to compute -- kMoeLoraMaxRank is the
    /// stride of `lora_low`, the most a store may ask for, and launching over it put
    /// (256 - rank) blocks per path and token on the device to read a rank and exit: with a
    /// rank-16 store, fifteen blocks in sixteen of the two widest launches of every layer.
    /// A caller that does not know the store's rank leaves the stride and loses nothing but time.
    std::int32_t lora_rank = kMoeLoraMaxRank;
    /// An expert-major adapter round's tile list and sorted entries, and its down projection's
    /// per-(token, path) partials, which a combine adds to the residual in the per-path order.
    /// Empty unless the round was allocated `grouped`.
    Tensor adapter_groups;
    Tensor adapter_partials;
};

template <class Arena>
SparseMoeDecodeWorkspace allocate_sparse_moe_decode_workspace(Arena& arena,
                                                              const SparseMoeGeometry& geometry,
                                                              std::int32_t tokens = 1,
                                                              bool grouped = false) {
    SparseMoeDecodeWorkspace out;
    out.ids          = arena.alloc(DType::I32, {geometry.experts_per_token, tokens}, 16);
    out.alpha        = arena.alloc(DType::FP32, {geometry.experts_per_token, tokens}, 16);
    out.shared_scale = arena.alloc(DType::FP32, {tokens}, 4);
    // D1 uses the first router_rows values as scores. D3 then reuses the same lifetime for
    // [paths, intermediate] natural FP32 SwiGLU results consumed by D4.
    const std::int32_t scratch_rows = std::max(geometry.paths(), (geometry.router_rows() +
                                                                    geometry.intermediate - 1) /
                                                                       geometry.intermediate);
    out.scratch = arena.alloc(DType::FP32, {scratch_rows * geometry.intermediate, tokens}, 256);
    out.lora_low = arena.alloc(DType::FP32, {2 * geometry.paths() * kMoeLoraMaxRank, tokens}, 256);
    if (grouped) {
        const std::int32_t paths = geometry.experts_per_token * tokens;
        out.adapter_groups = arena.alloc(
            DType::I32, {1 + 3 * sparse_moe_adapter_routed_tiles(paths, geometry.experts) + paths}, 256);
        out.adapter_partials = arena.alloc(DType::FP32, {geometry.paths() * geometry.hidden, tokens}, 256);
    }
    return out;
}

[[nodiscard]] std::size_t sparse_moe_decode_workspace_bytes(const SparseMoeGeometry& geometry,
                                                           std::int32_t tokens = 1,
                                                           bool grouped = false);
[[nodiscard]] SparseMoeDecodePlan resolve_sparse_moe_decode_plan(const SparseMoeGeometry& geometry,
                                                                 QType routed_gate_up,
                                                                 QType routed_down);

/// The expert-major small-T round (sparse_moe_decode_body.inc): routed gate/up rows read and
/// decoded once per round for every token they serve, the same bits as the per-path kernels. False when the round
/// does not qualify (a non-GGML pair, an expert slot pool, a shared expert not yet computed).
bool sparse_moe_decode_launch_expert_major(const SparseMoeGeometry& geometry, const Tensor& x,
                                           const PreparedSparseMoeWeights& weights,
                                           Tensor& destination, const int* token_ids,
                                           const float* token_alpha, const float* shared_scale,
                                           float* token_activations, int* groups,
                                           std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                                           cudaStream_t stream);
void sparse_moe_decode_launch_d3_small_t(const SparseMoeGeometry& geometry, const Tensor& x,
                                         const PreparedSparseMoeWeights& weights, const int* token_ids,
                                         float* token_activations, std::int32_t tokens,
                                         SparseMoeSmallTD3Schedule schedule, cudaStream_t stream,
                                         const int* adaptive_route_jobs = nullptr);
void sparse_moe_decode_launch_d4_small_t(const SparseMoeGeometry& geometry,
                                         const PreparedSparseMoeWeights& weights, Tensor& destination,
                                         const int* token_ids, const float* token_alpha,
                                         const float* shared_scale, const float* token_activations,
                                         std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                                         cudaStream_t stream,
                                         const int* adaptive_route_jobs = nullptr);
void sparse_moe_decode_launch(const SparseMoeGeometry& geometry, const Tensor& x,
                              const Tensor& router_x, const PreparedSparseMoeWeights& weights,
                              Tensor& destination, const SparseMoeDecodeWorkspace& workspace,
                              cudaStream_t stream, const SparseMoeRoundHook* hook = nullptr,
                              const LoraBank* adapters = nullptr, const std::int32_t* adapter_slot = nullptr);

} // namespace sinfer::ops::detail
