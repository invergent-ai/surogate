// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Phase-tree debug / introspection surface (design/buffer-runtime-v4.md Phase 4).
//
// Structured collectors that read the compiled forward / backward graphs and
// phase arenas after a `MultiGPUPyTrainer` has been constructed (which drives
// the compile). Used by the `surogate debug tensor-*` subcommands to emit
// JSONL records instead of setting the grab-bag of `SUROGATE_DEBUG_*` env
// vars and grepping stderr.
//
// All collectors operate on `const DslModel&` and do not allocate / mutate
// runtime state. They return plain-data structs that nanobind marshals to
// Python dicts/lists.

#ifndef SUROGATE_SRC_DSL_DSL_DEBUG_H
#define SUROGATE_SRC_DSL_DSL_DEBUG_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dsl {

class DslModel;

enum class DebugGraphKind : std::uint8_t {
    Forward = 0,
    Backward = 1,
};

const char* debug_graph_kind_name(DebugGraphKind k);

//! Per-tid layout entry — one row per (graph, tid) pair. This is the
//! structured form of the `SUROGATE_DEBUG_TID_TABLE` dump plus the region /
//! offset / bytes info from `SUROGATE_DEBUG_LAYOUT` and
//! `SUROGATE_DEBUG_REGIONS`.
struct DebugTensorEntry {
    DebugGraphKind graph = DebugGraphKind::Forward;
    int tid = -1;
    std::string name;
    std::string kind;    //!< TensorKind stringified ("ForwardParam", ...)
    std::string region;  //!< RegionKind stringified ("FwdStack", ...)
    int block_layer_idx = -1;
    std::uint64_t offset = 0;  //!< SIZE_MAX means unassigned
    std::uint64_t bytes = 0;
    bool offset_assigned = false;  //!< offset != SIZE_MAX
    bool retain_through_forward = false;
    int base_param_tid = -1;
    int base_producer_tid = -1;
    int base_grad_tid = -1;
    bool is_blocks = false;
    bool is_d_blocks = false;
    bool is_cross_layer = false;
    bool is_moe_offsets = false;
    bool is_moe_gather = false;
};

//! Phase-tree node — mirrors `dsl::PhaseNode` with kind stringified.
struct DebugPhaseNode {
    std::string kind;  //!< PhaseKind stringified
    std::string label;
    std::uint64_t op_start = 0;
    std::uint64_t op_end = 0;
    int block_index = -1;
    std::vector<DebugPhaseNode> children;
};

//! Instruction-stream entry — mirrors `dsl::Instruction` with kind stringified.
struct DebugInstruction {
    std::string kind;        //!< InstKind stringified
    std::string phase_kind;  //!< For PhaseEnter/PhaseExit; empty otherwise
    int block_index = -1;
    std::uint64_t op_start = 0;
    std::uint64_t op_end = 0;
    bool eager = false;
};

//! Full per-graph phase tree + flattened instruction stream.
struct DebugPhaseTree {
    DebugGraphKind graph = DebugGraphKind::Forward;
    bool present = false;  //!< false if CompiledGraph::phase_tree is empty
    DebugPhaseNode root;
    std::vector<DebugInstruction> instruction_stream;
};

//! Per-region coverage slice. One entry per RegionKind, per graph.
struct DebugRegionCount {
    std::string region;
    std::uint64_t tid_count = 0;
    std::uint64_t tid_bytes = 0;  //!< Sum of bytes over tids in this region
};

struct DebugGraphArena {
    DebugGraphKind graph = DebugGraphKind::Forward;
    std::string name;
    std::uint64_t num_tensors = 0;
    std::uint64_t num_ops = 0;
    //! CompiledGraph::{persistent,accumulator,fwd_stack,bwd_stack,save_for_bwd}_bytes.
    //! Note: these live on both fwd and bwd CompiledGraph; typically only one graph
    //! populates a given field (e.g., fwd_stack_peak only meaningful on forward).
    std::uint64_t persistent_bytes = 0;
    std::uint64_t accumulator_bytes = 0;
    std::uint64_t fwd_stack_peak = 0;
    std::uint64_t bwd_stack_peak = 0;
    std::uint64_t save_for_bwd_bytes = 0;
    std::vector<std::uint64_t> save_for_bwd_block_bytes;
    std::uint64_t layout_hash = 0;
    //! validate_arena_coverage() result.
    std::uint64_t tids_covered = 0;
    std::uint64_t tids_total = 0;
    std::uint64_t tids_size_exceeded = 0;
    //! validate_op_operand_coverage() result.
    std::uint64_t covered_inputs = 0;
    std::uint64_t covered_outputs = 0;
    std::uint64_t total_inputs = 0;
    std::uint64_t total_outputs = 0;
    //! Per-region tid count + byte total.
    std::vector<DebugRegionCount> regions;
};

//! Top-level arena summary. Covers both graphs + the `PhaseArenas` allocator.
struct DebugArenaSummary {
    //! PhaseArenas sizes (what `allocate_phase_arenas` actually cudaMalloc'd).
    std::uint64_t arena_persistent_bytes = 0;
    std::uint64_t arena_persistent_activation_bytes = 0;
    std::uint64_t arena_model_scope_persistent_bytes = 0;
    std::uint64_t arena_accumulator_bytes = 0;
    std::uint64_t arena_fwd_stack_bytes = 0;
    std::uint64_t arena_bwd_stack_bytes = 0;
    std::uint64_t arena_save_for_bwd_bytes = 0;
    std::uint64_t arena_unified_stack_bytes = 0;
    std::uint64_t arena_bwd_cross_layer_bytes = 0;
    std::uint64_t arena_moe_saved_bytes = 0;
    std::vector<std::uint64_t> arena_save_for_bwd_block_bases;
    bool arenas_allocated = false;
    DebugGraphArena forward;
    DebugGraphArena backward;
};

//! Descriptor/capability summary for one compiled graph. This is intentionally
//! count-based so regression artifacts can track descriptor coverage without
//! depending on enum dumps or graph-specific op ordering.
struct DebugGraphDescriptorSummary {
    DebugGraphKind graph = DebugGraphKind::Forward;
    std::string name;
    std::uint64_t num_tensors = 0;
    std::uint64_t num_ops = 0;
    std::uint64_t no_comm_ops = 0;
    std::uint64_t all_reduce_after_ops = 0;
    std::uint64_t reduce_scatter_after_ops = 0;
    std::uint64_t all_to_all_in_ops = 0;
    std::uint64_t all_to_all_out_ops = 0;
    std::uint64_t expert_parallel_routed_ops = 0;
    std::uint64_t grouped_ops = 0;
    std::uint64_t dense_matmul_ops = 0;
    std::uint64_t grouped_matmul_ops = 0;
    std::uint64_t moe_routed_ops = 0;
    std::uint64_t fp8_eligible_ops = 0;
    std::uint64_t fp4_eligible_ops = 0;
    std::uint64_t matmul_fp8_forward_eligible_ops = 0;
    std::uint64_t matmul_fp8_backward_eligible_ops = 0;
    std::uint64_t matmul_fp4_forward_eligible_ops = 0;
    std::uint64_t matmul_fp4_backward_eligible_ops = 0;
    std::uint64_t moe_fp8_grouped_eligible_ops = 0;
    std::uint64_t moe_fp4_grouped_eligible_ops = 0;
    std::uint64_t moe_fp8_backward_implemented_ops = 0;
    std::uint64_t moe_nvfp4_no_fallback_ops = 0;
    std::uint64_t lora_compatible_ops = 0;
    std::uint64_t weight_cache_eligible_ops = 0;
    std::uint64_t activation_epilogue_ops = 0;
    std::uint64_t cpu_pinned_stream_ops = 0;
    std::uint64_t fusion_candidate_starts = 0;
    std::uint64_t fp8_pending_tensors = 0;
    std::uint64_t fp8_ready_tensors = 0;
    std::uint64_t fp4_ready_tensors = 0;
    std::uint64_t lora_slices = 0;
    std::uint64_t lora_schema_slot_slices = 0;
    std::uint64_t grouped_lora_schema_slot_slices = 0;
};

struct DebugDescriptorSummary {
    DebugGraphDescriptorSummary forward;
    DebugGraphDescriptorSummary backward;
};

//! BufferPlan summary for schema-driven allocation migration. Count and byte
//! fields mirror BufferPlan's Phase 4b dual-path diagnostics.
struct DebugBufferPlanSummary {
    std::uint64_t schema_record_count = 0;
    std::uint64_t schema_routing_layers = 0;
    std::uint64_t schema_ep_layers = 0;
    std::uint64_t schema_dense_layers = 0;
    std::uint64_t schema_moe_layers = 0;
    std::uint64_t schema_mamba_layers = 0;
    std::uint64_t schema_linear_mixer_layers = 0;
    std::uint64_t schema_slot_count = 0;
    std::uint64_t schema_param_slots = 0;
    std::uint64_t schema_activation_slots = 0;
    std::uint64_t schema_op_lifetime_slots = 0;
    std::uint64_t schema_layer_lifetime_slots = 0;
    std::uint64_t schema_block_lifetime_slots = 0;
    std::uint64_t schema_model_lifetime_slots = 0;
    std::uint64_t schema_persistent_lifetime_slots = 0;
    std::uint64_t schema_registry_registered_activation_slots = 0;
    std::uint64_t schema_registry_missing_activation_slots = 0;
    std::uint64_t schema_registry_save_for_backward_activation_slots = 0;
    std::uint64_t schema_registry_save_for_backward_mismatch_slots = 0;
    std::uint64_t schema_resolved_activation_shape_slots = 0;
    std::uint64_t schema_unresolved_activation_shape_slots = 0;
    std::uint64_t schema_dynamic_activation_shape_slots = 0;
    std::uint64_t schema_resolved_activation_shape_bytes = 0;
    std::uint64_t schema_save_for_backward_activation_slots = 0;
    std::uint64_t schema_frame_activation_slots = 0;
    std::uint64_t schema_save_for_backward_activation_bytes = 0;
    std::uint64_t schema_frame_activation_bytes = 0;
    std::uint64_t schema_max_layer_activation_shape_bytes = 0;
    std::uint64_t schema_legacy_max_activation_shape_bytes = 0;
    std::uint64_t schema_activation_shape_savings_bytes = 0;
    std::uint64_t schema_resolved_param_shape_slots = 0;
    std::uint64_t schema_unresolved_param_shape_slots = 0;
    std::uint64_t schema_expert_parallel_param_slots = 0;
    std::uint64_t schema_resolved_param_shape_bytes = 0;
    std::uint64_t schema_resolved_param_shape_local_bytes = 0;
    std::uint64_t schema_expert_parallel_param_shape_bytes = 0;
    std::uint64_t schema_expert_parallel_param_shape_local_bytes = 0;
    std::uint64_t schema_expert_parallel_param_shape_savings_bytes = 0;
    std::uint64_t hook_after_produce_targets = 0;
    std::uint64_t hook_before_consume_targets = 0;
    std::uint64_t hook_after_all_to_all_targets = 0;
    std::uint64_t hook_after_reduce_scatter_targets = 0;
    std::uint64_t hook_registry_registrations = 0;
    std::uint64_t hook_registry_distribution_aware_registrations = 0;
};

//! One pair of overlapping `(region, block_layer_idx, offset, bytes)` ranges
//! in the same CompiledGraph. Under correct compilation, the only overlaps
//! are intentional aliases validated at compile time. Unexpected overlaps
//! flag a planner bug — the bug class buffer-runtime-v4 claims is
//! structurally impossible.
struct DebugAliasingPair {
    DebugGraphKind graph = DebugGraphKind::Forward;
    std::string region;
    int block_layer_idx = -1;
    int tid_a = -1;
    std::string name_a;
    std::uint64_t offset_a = 0;
    std::uint64_t bytes_a = 0;
    int tid_b = -1;
    std::string name_b;
    std::uint64_t offset_b = 0;
    std::uint64_t bytes_b = 0;
    std::uint64_t overlap_bytes = 0;  //!< byte count in [max(start), min(end))
};

//! Single-tensor provenance — what `tensor-resolve` emits.
struct DebugTensorResolution {
    bool found = false;
    DebugGraphKind graph = DebugGraphKind::Forward;
    DebugTensorEntry entry;
    //! `describe_tensor_id()` output from `CompiledGraph`.
    std::string description;
    //! The earliest op that writes this tid as an output (or SIZE_MAX if none).
    std::uint64_t first_write_op = static_cast<std::uint64_t>(-1);
    //! The latest op that reads/writes this tid (or SIZE_MAX if none).
    std::uint64_t last_use_op = static_cast<std::uint64_t>(-1);
    //! Phase path from root to the node whose op range contains first_write_op
    //! (deepest enclosing phase). Empty if no phase tree.
    std::vector<std::string> phase_path;
};

// ============================================================================
// Collectors
// ============================================================================

//! Flatten tensor metadata across both compiled graphs into one list.
//! Order: forward graph tids in ascending tid order, then backward graph tids
//! in ascending tid order.
std::vector<DebugTensorEntry> collect_tensor_layout(const DslModel& model);

//! Aggregate arena sizes + coverage across both graphs and the PhaseArenas.
DebugArenaSummary collect_arena_summary(const DslModel& model);

//! Aggregate descriptor/capability counts across both compiled graphs.
DebugDescriptorSummary collect_descriptor_summary(const DslModel& model);

//! BufferPlan schema/allocation diagnostics for regression artifacts.
DebugBufferPlanSummary collect_buffer_plan_summary(const DslModel& model);

//! Collect the phase tree + instruction stream for one graph.
//! `DebugPhaseTree::present == false` if the graph has no phase tree
//! (happens for non-block-stacked compiles).
DebugPhaseTree collect_phase_tree(const DslModel& model, bool is_backward);

//! Enumerate pairs of distinct tids whose `(region, block_layer_idx)` bucket
//! matches and whose `[offset, offset + bytes)` byte ranges overlap. Only
//! scans arena-backed regions (Persistent / Accumulator / FwdStack / BwdStack
//! / SaveForBwd); bucketing ensures pairs can only form within a single
//! coloring frame. Covers both graphs.
std::vector<DebugAliasingPair> collect_static_aliasing(const DslModel& model);

//! Resolve a tensor by name or by tid in a specified graph (forward or
//! backward). If `name` is non-empty, `tid` is ignored and the lookup goes
//! through `CompiledGraph::find_tensor_id`. Result `.found == false` when
//! the name / tid does not exist in the specified graph.
DebugTensorResolution resolve_tensor(const DslModel& model, const std::string& name, int tid, bool is_backward);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_DSL_DEBUG_H
