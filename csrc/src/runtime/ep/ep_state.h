// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Per-layer Expert Parallelism state structures.
//
// Extracted from CompiledExecutor to enable composable EP strategies.
// All state types live in the `ep::` namespace and are reused by every
// strategy implementation.

#ifndef SUROGATE_SRC_RUNTIME_EP_EP_STATE_H
#define SUROGATE_SRC_RUNTIME_EP_EP_STATE_H

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/ep/lpt_planner.h"
#include "runtime/lora/lora_types.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"

namespace ep {

/// Per-layer EP state: token splits, reorder mappings, persistent GPU buffers.
/// One instance per (layer_idx, replay_slot) key; see EPStrategy::ep_state_key().
struct EpLayerState {
    // Forward-pass dispatch plan cached for recompute replay. Replay must NOT
    // re-run the imbalance all-reduce / splits exchanges: those collectives
    // interleave with backward's combine a2a's in the shared launch queue and
    // intermittently deadlock the issue order. Routing is deterministic, so the
    // forward plan is bit-identical — replay restores it instead.
    bool plan_cached = false;
    bool cached_use_llep = false;
    int cached_total_send = 0;
    int cached_total_recv = 0;
    std::vector<int> cached_send_splits;
    std::vector<int> cached_recv_splits;
    std::vector<int> cached_recv_all_counts;
    std::vector<int> cached_expert_to_gpu;
    std::vector<int> cached_merged_experts;
    std::vector<int> cached_global_to_merged;
    LPTPlan cached_plan;

    std::vector<int> send_splits;      // tokens sent to each EP peer
    std::vector<int> recv_splits;      // tokens received from each EP peer
    int total_send = 0;                // sum of send_splits
    int total_recv = 0;                // sum of recv_splits
    void* send_order_gpu = nullptr;    // GPU buffer: reorder indices for recv re-sort gather
    void* recv_reorder_gpu = nullptr;  // GPU buffer: reorder indices for recv → local expert order
    size_t send_order_bytes = 0;
    size_t recv_reorder_bytes = 0;
    // LLEP send reorder: when LLEP is active, tokens are reordered before A2A.
    // llep_send_reorder_gpu[new_pos] = old_pos in expert-sorted input.
    // Null when standard EP (no reorder needed).
    void* llep_send_reorder_gpu = nullptr;
    size_t llep_send_reorder_bytes = 0;
    void* local_scatter_gpu = nullptr;  // local_scatter [total_recv] indices output
    size_t local_scatter_bytes = 0;
    // Forward EP outputs must remain valid until backward of this layer.
    // Shared cross-layer buffers can be overwritten by subsequent layers/recompute.
    void* sorted_recv_gpu = nullptr;  // ep_dispatch output [total_recv, hidden]
    size_t sorted_recv_bytes = 0;
    void* combined_gpu = nullptr;  // ep_combine output [total_send, hidden]
    size_t combined_bytes = 0;
    void* llep_combined_gpu = nullptr;  // ep_combine LLEP reorder output [total_send, hidden]
    size_t llep_combined_bytes = 0;
    // Backward EP outputs must also remain valid until their consumer runs.
    // Shared cross-layer buffers can be overwritten by later EP ops.
    void* dispatch_bwd_send_gpu = nullptr;  // ep_dispatch_backward reverse A2A [total_send, hidden]
    size_t dispatch_bwd_send_bytes = 0;
    void* dispatch_bwd_out_gpu = nullptr;  // ep_dispatch_backward LLEP reorder output [total_send, hidden]
    size_t dispatch_bwd_out_bytes = 0;
    void* combine_bwd_sorted_gpu = nullptr;  // ep_combine_backward output [total_recv, hidden]
    size_t combine_bwd_sorted_bytes = 0;

    /// Free all cudaMalloc'd GPU buffers owned by this state.
    void free_gpu();

    /// Build the two element-count arrays for a reverse all-to-all
    /// (combine / dispatch_backward): the A2A swaps the roles of
    /// `send_splits` and `recv_splits` and scales by `stride`.
    void build_reverse_a2a_elem_splits(int stride,
                                       std::vector<int>& reverse_send_elem,
                                       std::vector<int>& reverse_recv_elem) const;

    /// Build the two element-count arrays for a forward all-to-all
    /// (combine_backward): send_splits / recv_splits each scaled by `stride`.
    void build_forward_a2a_elem_splits(int stride, std::vector<int>& send_elem, std::vector<int>& recv_elem) const;
};

/// LLEP (Least-Loaded EP) per-layer state for dynamic load balancing.
/// When active, merged weight tensors contain native + foreign expert weights,
/// and the GEMM ops use these instead of the QLoRA-resolved weights.
struct LLEPLayerState {
    bool active = false;         // Whether LLEP rebalancing is active this step
    int num_merged_experts = 0;  // Total experts on this GPU (native + foreign)

    // Per-expert weight pointers (indexed by merged expert index 0..num_merged-1).
    // Each pointer points to one expert's weight slice in either:
    //  - native dequant buffer (for native experts), or
    //  - foreign weight receive buffer (for received foreign experts).
    // No contiguous merged buffer needed — saves ~465 MB GPU memory.
    std::vector<const void*> gate_up_weight_ptrs;  // [num_merged]: ptr to gate_up [2*D, C]
    std::vector<const void*> down_weight_ptrs;     // [num_merged]: ptr to down [C, D]
    ETensorDType weight_dtype = ETensorDType::BF16;

    std::vector<int> merged_offsets_host;  // Host expert offsets for merged set
    void* merged_offsets_gpu = nullptr;    // GPU expert offsets
    size_t merged_offsets_gpu_bytes = 0;
    // Map from merged expert index → global expert ID
    std::vector<int> merged_to_global;
    // Map from global expert ID → merged expert index (-1 if not on this GPU)
    std::vector<int> global_to_merged;

    // Merged LoRA weights [num_merged, ...] for use in GEMM dispatch.
    // Built from native expert LoRA + transferred foreign expert LoRA.
    // Null/inactive when LoRA is not enabled.
    // Per-layer owned GPU memory (NOT shared across layers).
    modules::LoRAGroupedExpertWeights<Tensor> merged_lora;
    bool has_merged_lora = false;
    // Per-layer owned GPU pointers for merged LoRA (freed on layer state clear)
    std::vector<void*> owned_lora_ptrs;

    // Foreign weight P2P receive buffers — owned by this state.
    // Must stay alive as long as weight pointers reference them.
    std::vector<void*> owned_foreign_ptrs;

    void free_lora_gpu() {
        for (void* p : owned_lora_ptrs) {
            if (p) cudaFree(p);
        }
        owned_lora_ptrs.clear();
        has_merged_lora = false;
    }
    void free_foreign_gpu() {
        for (void* p : owned_foreign_ptrs) {
            if (p) cudaFree(p);
        }
        owned_foreign_ptrs.clear();
        gate_up_weight_ptrs.clear();
        down_weight_ptrs.clear();
    }
};

/// Lightweight per-layer EP metadata — survives LLEP state clearing.
/// Backward uses this to reconstruct native-only weight pointers when
/// the full LLEP state has been freed to save GPU memory.
struct EPLayerMeta {
    int num_merged = 0;                 // total experts on this GPU (native + foreign)
    int native_start = 0;               // first native expert's global ID
    int num_local = 0;                  // number of native experts
    std::vector<int> merged_to_global;  // merged_idx → global expert ID
    // Full expert→primary-GPU map for this layer's plan. Needed by the
    // backward LoRA-wgrad exchange: every rank derives the same symmetric
    // helper→owner send/recv schedule from it.
    std::vector<int> expert_to_gpu;
};

}  // namespace ep

#endif  // SUROGATE_SRC_RUNTIME_EP_EP_STATE_H
