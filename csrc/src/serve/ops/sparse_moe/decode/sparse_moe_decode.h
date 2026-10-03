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
};

template <class Arena>
SparseMoeDecodeWorkspace allocate_sparse_moe_decode_workspace(Arena& arena,
                                                              const SparseMoeGeometry& geometry,
                                                              std::int32_t tokens = 1) {
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
    return out;
}

[[nodiscard]] std::size_t sparse_moe_decode_workspace_bytes(const SparseMoeGeometry& geometry,
                                                           std::int32_t tokens = 1);
[[nodiscard]] SparseMoeDecodePlan resolve_sparse_moe_decode_plan(const SparseMoeGeometry& geometry,
                                                                 QType routed_gate_up,
                                                                 QType routed_down);

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
