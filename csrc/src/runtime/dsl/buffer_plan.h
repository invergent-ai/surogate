// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compile-time buffer plan for activations, gradients, and scratch.
//
// Historically, `DslRunState::allocate_simplified_{activations,gradients}`
// computed per-slot sharing/sizing decisions inline at runtime-init time
// (re-reading `RuntimeOptions`, querying the slot registry's share policy,
// resolving hybrid dims, etc.). That made the decisions hard to reason about
// in isolation and mixed policy with mechanism.
//
// `BufferPlan` captures those decisions in a single POD, built once from the
// compile-time inputs (config, runtime config, options, slot registry, B, T).
// The allocate_* functions then become mechanical walks over the plan.

#ifndef SUROGATE_SRC_DSL_BUFFER_PLAN_H
#define SUROGATE_SRC_DSL_BUFFER_PLAN_H

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "config/pretrained_config.h"
#include "runtime/dsl/dsl_runtime_config.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "runtime/training/runtime_options.h"
#include "utilities/utils.h"

namespace dsl {

struct CompiledGraph;  // fwd — defined in graph_compiler.h; used by stack-sizing helpers.
struct Graph;          // fwd — defined in ir.h; used by schema metadata helpers.

/// Phase 4 dual-path record derived from graph.metadata.block_schemas.
/// The legacy enum BufferPlan remains authoritative until parity checks are
/// wired, but this typed summary is the C++ handoff for schema-driven planning.
enum class BlockSchemaFamilyKind {
    Unknown,
    Dense,
    MoE,
    Mamba,
    LinearMixer,
};

struct BlockSchemaSlotSummary {
    std::string name;
    std::string kind;
    std::string dtype;
    std::string lifetime;
    std::string residency;
    std::string distribution_kind;
    int shape_rank = 0;
    std::vector<std::string> shape_dims;
    bool shape_resolved = false;
    bool shape_dynamic = false;
    std::vector<long> resolved_shape;
    long resolved_numel = 0;
    long resolved_bytes = 0;
    long resolved_local_bytes = 0;
    bool grouped = false;
    bool save_for_backward = false;
    int streaming_prefetch_distance = -1;
};

struct BlockSchemaPlanRecord {
    int layer = -1;
    int block_index = -1;
    std::string block_type;
    std::string blocks_param;
    std::string block_name;
    std::string block_family;
    BlockSchemaFamilyKind family_kind = BlockSchemaFamilyKind::Unknown;
    int slot_count = 0;
    int param_slots = 0;
    int activation_slots = 0;
    int op_lifetime_slots = 0;
    int layer_lifetime_slots = 0;
    int block_lifetime_slots = 0;
    int model_lifetime_slots = 0;
    int persistent_lifetime_slots = 0;
    int replicated_slots = 0;
    int sharded_dim_slots = 0;
    int router_replicated_slots = 0;
    int expert_parallel_slots = 0;
    int streaming_slots = 0;
    int gpu_resident_slots = 0;
    int auto_resident_slots = 0;
    int cpu_pinned_stream_slots = 0;
    int cpu_pageable_slots = 0;
    int nvme_offload_slots = 0;
    std::vector<BlockSchemaSlotSummary> slots;
    std::string routing_kind;
    int routing_topk = -1;
    std::string routing_topk_param;
    bool routing_norm_topk_prob = false;
    std::string routing_norm_topk_prob_param;
    bool routing_scoring_bias = false;
    int routing_shared_experts = 0;
    std::string routing_shared_experts_param;
    std::string ep_size_param;
    bool ep_weight_transfer_eligible = false;
    bool has_routing = false;
    bool has_ep_topology = false;
};

struct BlockSchemaCoverageValidation {
    bool ok = true;
    std::string message;
};

struct BlockSchemaLayerSummary {
    int layer = -1;
    bool has_schema = false;
    std::string block_family;
    BlockSchemaFamilyKind family_kind = BlockSchemaFamilyKind::Unknown;
    int slot_count = 0;
    int param_slots = 0;
    int activation_slots = 0;
    int op_lifetime_slots = 0;
    int layer_lifetime_slots = 0;
    int block_lifetime_slots = 0;
    int model_lifetime_slots = 0;
    int persistent_lifetime_slots = 0;
    int replicated_slots = 0;
    int sharded_dim_slots = 0;
    int router_replicated_slots = 0;
    int expert_parallel_slots = 0;
    int streaming_slots = 0;
    int gpu_resident_slots = 0;
    int auto_resident_slots = 0;
    int cpu_pinned_stream_slots = 0;
    int cpu_pageable_slots = 0;
    int nvme_offload_slots = 0;
    int registry_registered_activation_slots = 0;
    int registry_missing_activation_slots = 0;
    int registry_save_for_backward_activation_slots = 0;
    int registry_save_for_backward_mismatch_slots = 0;
    int resolved_activation_shape_slots = 0;
    int unresolved_activation_shape_slots = 0;
    int dynamic_activation_shape_slots = 0;
    long resolved_activation_shape_bytes = 0;
    int save_for_backward_activation_slots = 0;
    int frame_activation_slots = 0;
    long save_for_backward_activation_bytes = 0;
    long frame_activation_bytes = 0;
    int resolved_param_shape_slots = 0;
    int unresolved_param_shape_slots = 0;
    int expert_parallel_param_slots = 0;
    long resolved_param_shape_bytes = 0;
    long resolved_param_shape_local_bytes = 0;
    std::vector<BlockSchemaSlotSummary> slots;
    std::string routing_kind;
    int routing_topk = -1;
    std::string routing_topk_param;
    bool routing_norm_topk_prob = false;
    std::string routing_norm_topk_prob_param;
    bool routing_scoring_bias = false;
    int routing_shared_experts = 0;
    std::string routing_shared_experts_param;
    std::string ep_size_param;
    bool ep_weight_transfer_eligible = false;
    bool has_routing = false;
    bool has_ep_topology = false;
};

/// Extract per-layer block schema records from a compiled graph's metadata.
/// Malformed records are skipped; absence of metadata returns an empty vector.
[[nodiscard]] std::vector<BlockSchemaPlanRecord> collect_block_schema_plan_records(const Graph& graph);

/// Validate that schema records describe exactly one block for every model
/// layer. Empty record sets are valid here; callers can decide whether absence
/// of schema metadata should be an error.
[[nodiscard]] BlockSchemaCoverageValidation
validate_block_schema_plan_coverage(const std::vector<BlockSchemaPlanRecord>& records, int num_layers);

// Model-config helpers shared by the plan builder and runtime allocators.
// Both handle the case where the passed PretrainedConfig is actually a
// `modules::ModelConfig` and fall back to safe defaults otherwise.

/// Up-projection factor for the MLP (2 for gated activations like SwiGLU, 1 otherwise).
[[nodiscard]] int resolve_mlp_up_factor(const PretrainedConfig& cfg);

/// True iff the model uses the Hybrid architecture variant
/// (per-layer block types, e.g. Mamba + Attention + MLP).
[[nodiscard]] bool is_hybrid_architecture(const PretrainedConfig& cfg);

/// Immutable, data-only description of how activation/gradient buffers should
/// be sized and shared. Built once per (B, T, options) via `BufferPlan::build`,
/// then consumed by `DslRunState` during allocation.
struct BufferPlan {
    // ---------------- Stack-based temps ----------------
    bool ffn_temps_on_stack = false;
    bool can_recompute_ffn_temps = false;
    bool large_bwd_temps_on_stack = false;

    // ---------------- QKV / qkv_rope ----------------
    // allocate_shared_qkv_rope removed (was part of forward-activation
    // sharing; the per-layer qkv_rope buffer is now always allocated
    // when need_separate_qkv_rope is true).
    bool need_separate_qkv_rope = false;  ///< recompute && use_qk_norm

    // ---------------- Slot availability ----------------
    bool has_mlp_up_slot = false;
    bool has_swiglu_slot = false;
    bool has_dsl_layout = false;

    // ---------------- Derived modes ----------------
    bool recompute_enabled = false;
    bool lora_only = false;
    bool use_qk_norm = false;
    bool is_hybrid = false;

    // ---------------- Schema-driven dual path ----------------
    // Populated from graph metadata when available. These counters are
    // diagnostics/parity inputs only until schema-driven allocation replaces
    // the legacy TensorSlot enum path.
    int schema_record_count = 0;
    int schema_routing_layers = 0;
    int schema_ep_layers = 0;
    int schema_dense_layers = 0;
    int schema_moe_layers = 0;
    int schema_mamba_layers = 0;
    int schema_linear_mixer_layers = 0;
    int schema_slot_count = 0;
    int schema_param_slots = 0;
    int schema_activation_slots = 0;
    int schema_op_lifetime_slots = 0;
    int schema_layer_lifetime_slots = 0;
    int schema_block_lifetime_slots = 0;
    int schema_model_lifetime_slots = 0;
    int schema_persistent_lifetime_slots = 0;
    int schema_replicated_slots = 0;
    int schema_sharded_dim_slots = 0;
    int schema_router_replicated_slots = 0;
    int schema_expert_parallel_slots = 0;
    int schema_streaming_slots = 0;
    int schema_gpu_resident_slots = 0;
    int schema_auto_resident_slots = 0;
    int schema_cpu_pinned_stream_slots = 0;
    int schema_cpu_pageable_slots = 0;
    int schema_nvme_offload_slots = 0;
    int schema_registry_registered_activation_slots = 0;
    int schema_registry_missing_activation_slots = 0;
    int schema_registry_save_for_backward_activation_slots = 0;
    int schema_registry_save_for_backward_mismatch_slots = 0;
    int schema_resolved_activation_shape_slots = 0;
    int schema_unresolved_activation_shape_slots = 0;
    int schema_dynamic_activation_shape_slots = 0;
    long schema_resolved_activation_shape_bytes = 0;
    int schema_save_for_backward_activation_slots = 0;
    int schema_frame_activation_slots = 0;
    long schema_save_for_backward_activation_bytes = 0;
    long schema_frame_activation_bytes = 0;
    long schema_max_layer_activation_shape_bytes = 0;
    long schema_legacy_max_activation_shape_bytes = 0;
    long schema_activation_shape_savings_bytes = 0;
    int schema_resolved_param_shape_slots = 0;
    int schema_unresolved_param_shape_slots = 0;
    int schema_expert_parallel_param_slots = 0;
    long schema_resolved_param_shape_bytes = 0;
    long schema_resolved_param_shape_local_bytes = 0;
    long schema_expert_parallel_param_shape_bytes = 0;
    long schema_expert_parallel_param_shape_local_bytes = 0;
    long schema_expert_parallel_param_shape_savings_bytes = 0;
    int schema_scoring_bias_routing_layers = 0;
    int schema_shared_expert_routing_layers = 0;
    int schema_weight_transfer_layers = 0;
    std::vector<BlockSchemaLayerSummary> schema_layers;

    // ---------------- Dimensions ----------------
    long B = 0;
    long T = 0;
    long C = 0;    ///< HiddenSize
    long Hq = 0;   ///< NumQueryHeads
    long Hkv = 0;  ///< NumKeyValHeads

    /// Maxed-across-hybrid-layers dims for *shared* buffers. Per-layer dims are
    /// looked up via `layer_*` accessors.
    long AttnDim = 0;
    long QKV = 0;
    long M = 0;
    long MUp = 0;

    long MoeM = 0;
    long MoeMUp = 0;
    long NumExperts = 0;
    long TopK = 0;
    int EPSize = 1;
    long LinearConvK = 0;
    long LinearKeyHeads = 0;
    long LinearValueHeads = 0;
    long LinearKeyDim = 0;
    long LinearValueDim = 0;
    long PerLayerInputDim = 0;
    long MambaHeads = 0;
    long MambaHeadDim = 0;
    long MambaStateSize = 0;
    long MambaGroups = 0;
    long MambaIntermediate = 0;
    long MambaConvDim = 0;
    long MambaProjSize = 0;

    int NumLayers = 0;

    /// Empty for homogeneous models.
    std::vector<BlockTypeDims> per_layer_dims;

    ETensorDType act_dtype = ETensorDType::BF16;
    ETensorDType grad_dtype = ETensorDType::BF16;

    // ---------------- Accessors ----------------
    [[nodiscard]] bool has_per_layer_dims() const {
        return !per_layer_dims.empty();
    }
    [[nodiscard]] bool has_moe() const {
        return NumExperts > 0;
    }
    [[nodiscard]] const BlockSchemaLayerSummary* schema_layer(int i) const;
    [[nodiscard]] const BlockSchemaSlotSummary* schema_slot(int i, std::string_view name) const;
    [[nodiscard]] bool schema_layer_has_slot(int i, std::string_view name) const {
        return schema_slot(i, name) != nullptr;
    }
    [[nodiscard]] std::vector<std::string>
    schema_activation_slots_missing_from_registry(const TensorSlotRegistry& slot_registry) const;
    [[nodiscard]] std::vector<std::string>
    schema_save_for_backward_slots_not_saved_in_registry(const TensorSlotRegistry& slot_registry) const;

    [[nodiscard]] long layer_qkv(int i) const {
        return (has_per_layer_dims() && i < static_cast<int>(per_layer_dims.size())) ? per_layer_dims[i].qkv_channels
                                                                                     : QKV;
    }
    [[nodiscard]] long layer_attn_dim(int i) const {
        return (has_per_layer_dims() && i < static_cast<int>(per_layer_dims.size())) ? per_layer_dims[i].attn_dim
                                                                                     : AttnDim;
    }
    [[nodiscard]] long layer_mlp_up(int i) const {
        return (has_per_layer_dims() && i < static_cast<int>(per_layer_dims.size())) ? per_layer_dims[i].mlp_up : MUp;
    }
    [[nodiscard]] long layer_intermediate(int i) const {
        return (has_per_layer_dims() && i < static_cast<int>(per_layer_dims.size())) ? per_layer_dims[i].intermediate
                                                                                     : M;
    }

    // ---------------- Stack sizing ----------------

    /// Peak bytes of DSL stack memory required by the plan-level stack temps:
    ///   - forward FFN temps (`mlp_up`, `swiglu`) when `ffn_temps_on_stack`
    ///   - backward FFN+QKV temps (`d_qkv`, `d_mlp_up`, `d_swiglu`, `d_up`)
    ///     when `large_bwd_temps_on_stack`
    /// Each contribution is rounded up to the 4 KiB stack alignment. This
    /// replaces the "simulate into a dummy stack and read `max_utilization`"
    /// measurement pass that used to live in `DslRunState` init.
    ///
    /// Does NOT include op-internal temps (flash-attention workspace, Mamba
    /// scan buffers, ChunkGatedDeltaRule recompute, etc.) — those require
    /// walking the compiled backward graph; see `graph_backward_stack_peak`.
    [[nodiscard]] long plan_stack_peak_bytes() const;

    // ---------------- Builder ----------------
    static BufferPlan build(const PretrainedConfig& cfg,
                            const DslRuntimeConfig& runtime_config,
                            const RuntimeOptions& options,
                            const TensorSlotRegistry& slot_registry,
                            bool lora_only_mode,
                            long B,
                            long T,
                            ETensorDType act_dtype,
                            ETensorDType grad_dtype,
                            const std::vector<BlockSchemaPlanRecord>* schema_records = nullptr);
};

// ============================================================================
// Stack sizing helpers
// ============================================================================
//
// `plan_stack_peak_bytes()` above covers DSL-managed stack allocations.
// Dispatch functions (flash-attention backward, Mamba scan, ChunkGatedDelta-
// Rule backward, MoE expert backward, ...) also push sizeable temps onto the
// same stack — those are not in the BufferPlan, so we walk the compiled
// backward graph to find them.

/// Peak stack bytes implied by a compiled *backward* graph. Sums the bytes
/// of stack-resident outputs (Temporary/Mapped slots plus d_qkv / d_mlp_up /
/// d_swiglu and mlp_up / swiglu outputs that are stack-backed per the plan),
/// plus op-internal temps for known-heavy ops. Stack resets at layer_end
/// boundaries so the peak is computed per-layer and max'd across layers.
///
/// Returns 0 if `bwd_graph` is null or has no operations.
[[nodiscard]] long graph_backward_stack_peak(const CompiledGraph* bwd_graph, const BufferPlan& plan);

/// Total DSL stack size required for training, combining plan-level and
/// graph-level peaks with the safety / MoE / CUDA-graph / architecture
/// slack margins inherited from the legacy heuristic sizing.
///
/// `bwd_graph == nullptr` is allowed: returns a provisional size driven by
/// the plan only (used to allocate the stack *before* the backward graph is
/// compiled). Call again with the real backward graph after compile to get
/// the final size, and resize if larger.
[[nodiscard]] long required_stack_bytes(const BufferPlan& plan,
                                        const CompiledGraph* bwd_graph,
                                        const PretrainedConfig& cfg,
                                        const RuntimeOptions& options);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_BUFFER_PLAN_H
