#pragma once

// Expert slot cache: a device pool of MoE expert slots fed from a pinned host bank, with the
// routing → slot resolution and the miss gather running on the device so a decode round
// captures into a CUDA graph (design/INFERENCE.md, phase 2; design/serve-engine-flash-next.md
// D3). Model-agnostic: keyed by SparseMoeGeometry and the W8 row-split layout the MoE kernels
// already read; a slot is addressed exactly like an expert (`row_base = slot * rows`).
//
// Status (2026-08-28): API draft, unbuilt. The MoE kernels gain an optional per-layer
// `slot_of_expert` table (null = resident experts, identity); resolve/gather sit between the
// routing and the expert kernels of every schedule (decode d2→d3, small-T s2→s3, prefill
// select_count→scan).

#include "api/ops/sparse_moe.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

/// Host-side bank of one layer's routed experts: the two W8 row-split matrices as the MoE
/// kernels read them, in pinned, device-mapped host memory. Byte offsets are per plane
/// (codes, scales) so an expert is one contiguous slice per plane.
struct ExpertHostBank {
    const std::byte* gate_up_codes  = nullptr; // [experts * 2 * intermediate rows] codes plane
    const std::byte* gate_up_scales = nullptr; // scales plane, same row order
    const std::byte* down_codes     = nullptr; // [experts * hidden rows]
    const std::byte* down_scales    = nullptr;
    std::uint64_t gate_up_codes_bytes_per_expert  = 0;
    std::uint64_t gate_up_scales_bytes_per_expert = 0;
    std::uint64_t down_codes_bytes_per_expert     = 0;
    std::uint64_t down_scales_bytes_per_expert    = 0;
};

/// Device pool of `slots` experts in the same plane layout; `routed_gate_up` / `routed_down`
/// are Weights over the pool that the MoE kernels consume unchanged (n = slots * rows).
struct ExpertSlotPool {
    std::int32_t slots = 0;
    Weight routed_gate_up; // codes/scales planes over the pool
    Weight routed_down;
    std::byte* gate_up_codes  = nullptr;
    std::byte* gate_up_scales = nullptr;
    std::byte* down_codes     = nullptr;
    std::byte* down_scales    = nullptr;
};

/// Directory shared by all layers: flat expert id = layer * experts + expert.
struct ExpertSlotDirectory {
    Tensor slot_of_expert;  // I32 [layers * experts]: slot or -1
    Tensor expert_of_slot;  // I32 [slots]: flat expert id or -1
    Tensor last_used;       // I32-typed unsigned [slots]: round stamp of the last touch (LRU)
    Tensor active_round;    // I32-typed unsigned [slots]: stamp of the current touch (eviction guard)
    Tensor round;           // I32-typed unsigned [1]: current round stamp, bumped by resolve
    Tensor hand;            // I32 [1]: clock hand of the victim search
    Tensor seen;            // I32-typed unsigned [experts]: per-round dedupe stamps
    Tensor cpu_round;       // I32-typed unsigned [experts]: stamp of the round that sent the expert to the CPU
    std::int32_t layers  = 0;
    std::int32_t experts = 0;
};

/// Per-round miss list written by resolve and consumed by gather; `count` is a device word so
/// the gather's row count never becomes a host argument (graph capture).
struct ExpertMissList {
    Tensor slots;    // I32 [capacity]
    Tensor experts;  // I32 [capacity]  (expert index within the layer)
    Tensor count;    // I64 [1]
    std::int32_t capacity = 0;
};

/// Device bytes of a pool of `slots` W8 experts (both matrices, codes + scales planes).
[[nodiscard]] std::size_t expert_slot_pool_bytes(const SparseMoeGeometry& geometry,
                                                  std::int32_t slots);
/// Device bytes of the directory for `layers × experts` ids over `slots`, and of a miss list
/// able to hold `capacity` entries.
[[nodiscard]] std::size_t expert_slot_directory_bytes(std::int32_t layers, std::int32_t experts,
                                                       std::int32_t slots);
[[nodiscard]] std::size_t expert_miss_list_bytes(std::int32_t capacity);

/// Carve the pool / directory / miss list out of device memory the caller owns (sizes from the
/// functions above; 256-byte aligned). The directory starts empty (every id unmapped, stamps
/// at zero); `expert_slot_directory_reset` returns it to that state.
[[nodiscard]] ExpertSlotPool create_expert_slot_pool(const SparseMoeGeometry& geometry,
                                                      std::int32_t slots, void* device_bytes);
[[nodiscard]] ExpertSlotDirectory create_expert_slot_directory(std::int32_t layers,
                                                                std::int32_t experts,
                                                                std::int32_t slots,
                                                                void* device_bytes,
                                                                cudaStream_t stream);
[[nodiscard]] ExpertMissList create_expert_miss_list(std::int32_t capacity, void* device_bytes);
void expert_slot_directory_reset(ExpertSlotDirectory& directory, cudaStream_t stream);

/// Describes one layer's routed experts in the pinned host bank from the two W8 Weights the
/// bank already exposes (their planes are row-major, so expert `e` is a contiguous slice).
[[nodiscard]] ExpertHostBank expert_host_bank(const SparseMoeGeometry& geometry,
                                              const Weight& routed_gate_up,
                                              const Weight& routed_down);

/// Jobs for the host: the (token, expert, router weight) paths of a round whose expert the
/// split sent to the CPU instead of the pool; `count` is a device word.
struct ExpertCpuJobList {
    Tensor tokens;   // I32 [capacity]
    Tensor experts;  // I32 [capacity]
    Tensor weights;  // FP32 [capacity]
    Tensor count;    // I64 [1]
    std::int32_t capacity = 0;
};
[[nodiscard]] std::size_t expert_cpu_job_list_bytes(std::int32_t capacity);
[[nodiscard]] ExpertCpuJobList create_expert_cpu_job_list(std::int32_t capacity, void* device_bytes);

/// Resolves the routing of one layer for one round: for every expert id in `ids` (I32
/// [experts_per_token, tokens]) touch its slot or assign the least recently used slot that is
/// not active in this round, appending misses to `misses`. Writes `slot_of_expert` for the
/// layer and leaves `ids` untouched (the kernels look slots up through the table).
///
/// With a CPU split (`cpu_share_q16` > 0, `alpha` and `cpu_jobs` given) a `cpu_share_q16/65536`
/// fraction of the *missing* experts is not fetched: their table entries stay -1 (the kernels
/// contribute nothing for those paths) and every (token, path) that routes to them becomes a
/// job in `cpu_jobs` with its router weight. Hits are never split.
void expert_slot_resolve(const Tensor& ids, std::int32_t layer, ExpertSlotDirectory& directory,
                         ExpertMissList& misses, cudaStream_t stream);
void expert_slot_resolve(const Tensor& ids, const Tensor& alpha, std::int32_t layer,
                         ExpertSlotDirectory& directory, ExpertMissList& misses,
                         ExpertCpuJobList* cpu_jobs, std::uint32_t cpu_share_q16,
                         cudaStream_t stream);

/// Copies the missing experts of `layer` from the host bank into their slots: one launch,
/// four banks (gate_up codes/scales, down codes/scales), 16-byte units, row count read from
/// `misses.count`. Streaming loads (`ld.global.L1::no_allocate`) so the copy does not evict
/// the resident working set.
void expert_slot_gather(const ExpertHostBank& bank, const ExpertMissList& misses,
                        ExpertSlotPool& pool, cudaStream_t stream);

/// Adds a host-computed FP32 partial ([hidden, tokens], read through its device-mapped
/// pointer) into a BF16 destination of the same shape.
void expert_cpu_partial_add(const void* partial_device_ptr, Tensor& destination, cudaStream_t stream);

/// The MoE kernels' view of this layer: the pool's weights plus the layer's slot table.
[[nodiscard]] SparseMoeWeights expert_slot_weights(const ExpertSlotPool& pool,
                                                   const ExpertSlotDirectory& directory,
                                                   std::int32_t layer,
                                                   const SparseMoeWeights& resident_parts);

} // namespace ninfer::ops
