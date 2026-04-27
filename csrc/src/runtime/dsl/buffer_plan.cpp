// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/buffer_plan.h"

#include <algorithm>
#include <cstdlib>
#include <utility>

#include "runtime/core/model_config.h"
#include "runtime/dsl/graph_compiler.h"
#include "runtime/dsl/ir.h"
#include "runtime/executor/op_registry.h"
#include "utilities/stack.h"

namespace dsl {

namespace {

[[nodiscard]] const AttrMap* attr_map(const AttrValue& value) {
    if (const auto* ptr = std::get_if<AttrValue::MapPtr>(&value.value)) {
        return ptr->get();
    }
    return nullptr;
}

[[nodiscard]] const AttrList* attr_list(const AttrValue& value) {
    if (const auto* ptr = std::get_if<AttrValue::ListPtr>(&value.value)) {
        return ptr->get();
    }
    return nullptr;
}

[[nodiscard]] const AttrValue* find_attr(const AttrMap& map, const std::string& key) {
    const auto it = map.find(key);
    return (it == map.end()) ? nullptr : &it->second;
}

[[nodiscard]] std::string attr_string(const AttrMap& map, const std::string& key) {
    const auto* value = find_attr(map, key);
    if (!value) return {};
    if (const auto* text = std::get_if<std::string>(&value->value)) {
        return *text;
    }
    return {};
}

[[nodiscard]] int attr_int(const AttrMap& map, const std::string& key) {
    const auto* value = find_attr(map, key);
    if (!value) return -1;
    if (const auto* v = std::get_if<std::int64_t>(&value->value)) {
        return static_cast<int>(*v);
    }
    return -1;
}

}  // namespace

std::vector<BlockSchemaPlanRecord> collect_block_schema_plan_records(const Graph& graph) {
    std::vector<BlockSchemaPlanRecord> records;
    const auto meta_it = graph.metadata.find("block_schemas");
    if (meta_it == graph.metadata.end()) return records;

    const AttrList* raw_records = attr_list(meta_it->second);
    if (!raw_records) return records;

    records.reserve(raw_records->size());
    for (const AttrValue& raw_record : *raw_records) {
        const AttrMap* record = attr_map(raw_record);
        if (!record) continue;

        BlockSchemaPlanRecord out;
        out.layer = attr_int(*record, "layer");
        out.block_index = attr_int(*record, "block_index");
        out.block_type = attr_string(*record, "block_type");
        out.blocks_param = attr_string(*record, "blocks_param");
        out.block_name = attr_string(*record, "block_name");

        const AttrValue* schema_value = find_attr(*record, "schema");
        const AttrMap* schema = schema_value ? attr_map(*schema_value) : nullptr;
        if (schema) {
            const AttrValue* routing_value = find_attr(*schema, "routing");
            out.has_routing = routing_value && attr_map(*routing_value) != nullptr;
            const AttrValue* ep_value = find_attr(*schema, "ep_topology");
            out.has_ep_topology = ep_value && attr_map(*ep_value) != nullptr;

            const AttrValue* attrs_value = find_attr(*schema, "attrs");
            if (const AttrMap* attrs = attrs_value ? attr_map(*attrs_value) : nullptr) {
                out.block_family = attr_string(*attrs, "block_family");
            }

            const AttrValue* slots_value = find_attr(*schema, "slots");
            if (const AttrList* slots = slots_value ? attr_list(*slots_value) : nullptr) {
                out.slot_count = static_cast<int>(slots->size());
                for (const AttrValue& raw_slot : *slots) {
                    const AttrMap* slot = attr_map(raw_slot);
                    if (!slot) continue;
                    const std::string kind = attr_string(*slot, "kind");
                    if (kind == "param") {
                        ++out.param_slots;
                    } else {
                        ++out.activation_slots;
                    }
                    if (const AttrValue* dist_value = find_attr(*slot, "distribution")) {
                        if (const AttrMap* dist = attr_map(*dist_value);
                            dist && attr_string(*dist, "kind") == "expert_parallel") {
                            ++out.expert_parallel_slots;
                        }
                    }
                    if (const AttrValue* streaming_value = find_attr(*slot, "streaming_hint")) {
                        if (attr_map(*streaming_value)) {
                            ++out.streaming_slots;
                        }
                    }
                }
            }
        }

        if (out.layer < 0 || out.block_type.empty()) continue;
        records.push_back(std::move(out));
    }

    return records;
}

BlockSchemaCoverageValidation validate_block_schema_plan_coverage(const std::vector<BlockSchemaPlanRecord>& records,
                                                                  int num_layers) {
    BlockSchemaCoverageValidation result;
    if (records.empty()) {
        return result;
    }
    if (num_layers < 0) {
        result.ok = false;
        result.message = "NumLayers is negative";
        return result;
    }
    if (records.size() != static_cast<std::size_t>(num_layers)) {
        result.ok = false;
        result.message = "block schema plan record count " + std::to_string(records.size()) + " != NumLayers " +
                         std::to_string(num_layers);
        return result;
    }

    std::vector<bool> seen(static_cast<std::size_t>(num_layers), false);
    for (const auto& record : records) {
        if (record.layer < 0 || record.layer >= num_layers) {
            result.ok = false;
            result.message = "block schema layer " + std::to_string(record.layer) + " outside [0, " +
                             std::to_string(num_layers) + ")";
            return result;
        }
        if (seen[static_cast<std::size_t>(record.layer)]) {
            result.ok = false;
            result.message = "duplicate block schema layer " + std::to_string(record.layer);
            return result;
        }
        seen[static_cast<std::size_t>(record.layer)] = true;
    }
    for (int layer = 0; layer < num_layers; ++layer) {
        if (!seen[static_cast<std::size_t>(layer)]) {
            result.ok = false;
            result.message = "missing block schema layer " + std::to_string(layer);
            return result;
        }
    }
    return result;
}

int resolve_mlp_up_factor(const PretrainedConfig& cfg) {
    if (auto* mc = dynamic_cast<const modules::ModelConfig*>(&cfg)) {
        return mc->mlp_up_factor();
    }
    return 2;  // Default for non-ModelConfig callers; legacy gated-MLP assumption.
}

bool is_hybrid_architecture(const PretrainedConfig& cfg) {
    if (auto* mc = dynamic_cast<const modules::ModelConfig*>(&cfg)) {
        return mc->architecture == modules::ArchitectureType::Hybrid;
    }
    return false;
}

BufferPlan BufferPlan::build(const PretrainedConfig& cfg,
                             const DslRuntimeConfig& runtime_config,
                             const RuntimeOptions& options,
                             const TensorSlotRegistry& slot_registry,
                             bool lora_only_mode,
                             long B,
                             long T,
                             ETensorDType act_dtype,
                             ETensorDType grad_dtype,
                             const std::vector<BlockSchemaPlanRecord>* schema_records) {
    BufferPlan p;

    // ---------------- Dimensions ----------------
    p.B = B;
    p.T = T;
    p.C = cfg.HiddenSize;
    p.Hq = cfg.NumQueryHeads;
    p.Hkv = cfg.NumKeyValHeads;
    const long head_size = cfg.head_size();
    p.AttnDim = p.Hq * head_size;
    p.QKV = head_size * (p.Hq + 2 * p.Hkv);
    p.M = cfg.IntermediateSize;
    p.MUp = static_cast<long>(resolve_mlp_up_factor(cfg)) * p.M;
    p.NumLayers = cfg.NumLayers;

    // Hybrid dims: max dims across all layers drive max-buffer sizing.
    p.per_layer_dims = runtime_config.per_layer_dims;
    if (!p.per_layer_dims.empty()) {
        for (const auto& pld : p.per_layer_dims) {
            p.QKV = std::max(p.QKV, pld.qkv_channels);
            p.AttnDim = std::max(p.AttnDim, pld.attn_dim);
            p.M = std::max(p.M, pld.intermediate);
            p.MUp = std::max(p.MUp, pld.mlp_up);
        }
    }

    p.NumExperts = runtime_config.num_experts;
    p.TopK = (runtime_config.num_experts_per_tok > 0) ? runtime_config.num_experts_per_tok : 1;
    p.MoeM = (runtime_config.moe_intermediate_size > 0) ? runtime_config.moe_intermediate_size : cfg.IntermediateSize;
    p.MoeMUp = static_cast<long>(resolve_mlp_up_factor(cfg)) * p.MoeM;
    p.use_qk_norm = runtime_config.use_qk_norm;

    p.act_dtype = act_dtype;
    p.grad_dtype = grad_dtype;

    // ---------------- Derived modes ----------------
    p.recompute_enabled = options.Recompute >= RecomputeLevel::Enabled;
    p.lora_only = lora_only_mode;
    p.is_hybrid = is_hybrid_architecture(cfg);
    p.has_dsl_layout = slot_registry.has_dsl_layout();

    // ---------------- Slot availability ----------------
    p.has_mlp_up_slot = p.has_dsl_layout && slot_registry.lookup(builtin_slot_name(TensorSlot::BlockMLPUp)).has_value();
    p.has_swiglu_slot =
        p.has_dsl_layout && slot_registry.lookup(builtin_slot_name(TensorSlot::BlockSwiGLU)).has_value();

    // ---------------- Schema-driven dual path ----------------
    if (schema_records) {
        p.schema_record_count = static_cast<int>(schema_records->size());
        for (const auto& record : *schema_records) {
            if (record.has_routing) {
                ++p.schema_routing_layers;
            }
            if (record.has_ep_topology) {
                ++p.schema_ep_layers;
            }
            p.schema_slot_count += record.slot_count;
            p.schema_param_slots += record.param_slots;
            p.schema_activation_slots += record.activation_slots;
            p.schema_expert_parallel_slots += record.expert_parallel_slots;
            p.schema_streaming_slots += record.streaming_slots;
        }
    }

    // ---------------- FFN temps on stack ----------------
    // Only safe when both mlp_up and swiglu are recomputable — otherwise the
    // backward pass can't reconstruct them and we'd have to fall back to
    // persistent saves (severe memory pressure).
    p.can_recompute_ffn_temps = slot_registry.will_recompute(builtin_slot_name(TensorSlot::BlockMLPUp), p.lora_only) &&
                                slot_registry.will_recompute(builtin_slot_name(TensorSlot::BlockSwiGLU), p.lora_only);
    p.ffn_temps_on_stack = p.recompute_enabled && p.lora_only && p.can_recompute_ffn_temps;

    // ---------------- qkv_rope ----------------
    // Separate per-layer qkv_rope whenever recompute is on and QK-norm
    // is used.
    p.need_separate_qkv_rope = p.recompute_enabled && p.use_qk_norm;

    // ---------------- Stack-resident backward temps ----------------
    p.large_bwd_temps_on_stack = p.recompute_enabled;

    return p;
}

// ============================================================================
// Stack sizing
// ============================================================================

namespace {

[[nodiscard]] long bytes_of(long count, ETensorDType dtype) {
    return count * static_cast<long>(get_dtype_size(dtype));
}

[[nodiscard]] long tensor_stack_bytes(ETensorDType dtype, const std::vector<long>& shape) {
    if (shape.empty()) return 0;
    long total = static_cast<long>(get_dtype_size(dtype));
    for (long d : shape) {
        total *= d;
    }
    return align_stack_bytes(total);
}

}  // namespace

long BufferPlan::plan_stack_peak_bytes() const {
    // Each block below mirrors a simulation in `allocate_simplified_*` —
    // same allocations, same order, same dtypes. Peak is the max across
    // blocks since each block fully releases before the next runs (the sim
    // `free()`s in reverse at the end of each block).

    // Forward FFN temps: mlp_up + swiglu (both live simultaneously).
    long ffn_peak = 0;
    if (ffn_temps_on_stack) {
        ffn_peak += align_stack_bytes(bytes_of(B * T * MUp, act_dtype));  // mlp_up
        ffn_peak += align_stack_bytes(bytes_of(B * T * M, act_dtype));    // swiglu
    }

    // Backward temps: d_qkv, d_mlp_up, d_swiglu, d_up (all live simultaneously).
    // d_up is gated on has_mlp_up_slot (matches the gradient-slot allowlist).
    long bwd_peak = 0;
    if (large_bwd_temps_on_stack) {
        bwd_peak += align_stack_bytes(bytes_of(B * T * QKV, grad_dtype));  // d_qkv
        if (has_mlp_up_slot) {
            bwd_peak += align_stack_bytes(bytes_of(B * T * MUp, grad_dtype));  // d_mlp_up
        }
        if (has_swiglu_slot) {
            bwd_peak += align_stack_bytes(bytes_of(B * T * M, grad_dtype));  // d_swiglu
        }
        if (has_mlp_up_slot) {
            bwd_peak += align_stack_bytes(bytes_of(B * T * MUp, grad_dtype));  // d_up
        }
    }

    // Scratch sim in allocate_scratch_buffers: just d_qkv, already covered
    // by bwd_peak's first term since large_bwd_temps_on_stack == recompute_enabled.

    return std::max(ffn_peak, bwd_peak);
}

// ----------------------------------------------------------------------------
// Graph-level peak: op-internal + stack-resident outputs in compiled backward
// ----------------------------------------------------------------------------

long graph_backward_stack_peak(const CompiledGraph* bwd_graph, const BufferPlan& plan) {
    if (bwd_graph == nullptr || bwd_graph->ops.empty()) {
        return 0;
    }

    const bool bwd_on_stack = plan.large_bwd_temps_on_stack;
    const bool ffn_on_stack = plan.ffn_temps_on_stack;

    long peak = 0;
    long current = 0;

    for (const auto& op : bwd_graph->ops) {
        // (1) Graph-level outputs that land on the stack.
        for (const auto& ref : op.outputs) {
            if (ref.shape.empty()) continue;
            bool on_stack = false;
            switch (ref.slot) {
                case TensorSlot::Mapped: on_stack = true; break;
                case TensorSlot::BlockDQKV:
                case TensorSlot::BlockDMLPUp:
                case TensorSlot::BlockDSwiGLU: on_stack = bwd_on_stack; break;
                case TensorSlot::BlockMLPUp:
                case TensorSlot::BlockSwiGLU: on_stack = ffn_on_stack; break;
                default: break;
            }
            if (on_stack) {
                current += tensor_stack_bytes(ref.dtype, ref.shape);
            }
        }

        // (2) Op-internal temps allocated by dispatch functions — workspaces,
        //     recompute scratch, fused-kernel temps — that are NOT visible
        //     as graph-output TensorRefs. Ops opt in by registering a
        //     `StackBoundFn` alongside their dispatch (see op_registry.h's
        //     `REGISTER_STACK_BOUND`). Ops without a bound contribute 0; the
        //     outer safety margin in `required_stack_bytes` covers them.
        if (const auto* desc = OpRegistry::instance().find(op.type); desc && desc->stack_bound_fn) {
            current += desc->stack_bound_fn(op, plan);
        }

        peak = std::max(peak, current);

        // At layer_end the runtime restores the stack to initial_checkpoint;
        // op-internal temps within a layer are freed by then.
        if (op.layer_end >= 0) {
            current = 0;
        }
    }

    return peak;
}

// ----------------------------------------------------------------------------
// Unified stack-size estimator
// ----------------------------------------------------------------------------

namespace {

/// Qwen3.5 hybrid blocks (Mamba + Attention + MLP) under LoRA hit a backward
/// peak around SwiGLU + LoRA hooks that is not captured by either the plan
/// or the graph walk. Gate the extra slacks with this predicate.
[[nodiscard]] bool is_qwen3_hybrid_lora(const BufferPlan& plan, const PretrainedConfig& cfg) {
    if (!plan.lora_only || !plan.is_hybrid) return false;
    return cfg.Architecture == PretrainedConfig::QWEN3;
}

/// Extra bytes to add to the heuristic when `is_qwen3_hybrid_lora` holds,
/// accounting for the unmodeled SwiGLU-backward + LoRA-hook transient peak.
[[nodiscard]] long qwen3_hybrid_lora_heuristic_slack(const BufferPlan& plan, const RuntimeOptions& options) {
    const long dtype_bytes = static_cast<long>(get_dtype_size(plan.act_dtype));
    const long swiglu_peak = plan.B * plan.T * plan.MUp * dtype_bytes;
    long slack = std::max(128L * 1024 * 1024, swiglu_peak + 64L * 1024 * 1024);
    if (options.UseCudaGraphs) {
        // CUDA graph capture retains additional transient tensors during
        // replay (mamba_gated_rmsnorm / swiglu backward).
        slack += 512L * 1024 * 1024;
    }
    return slack;
}

/// Minimum stack floor for Qwen3.5 hybrid LoRA — higher with CUDA graphs
/// because capture pins more state. Returns 0 when the predicate doesn't hold.
[[nodiscard]] long
qwen3_hybrid_lora_floor(const BufferPlan& plan, const PretrainedConfig& cfg, const RuntimeOptions& options) {
    if (!is_qwen3_hybrid_lora(plan, cfg)) return 0;
    return options.UseCudaGraphs ? (1536L * 1024 * 1024) : (1024L * 1024 * 1024);
}

/// MoE backward temps are op-internal (`gpt_oss_moe_act_backward` allocates
/// in intermediate dim, not hidden dim) and not yet modeled in
/// `graph_backward_stack_peak`. This reproduces the legacy slack.
[[nodiscard]] long moe_op_internal_estimate(const BufferPlan& plan) {
    if (!plan.has_moe()) return 0;
    // Matches `moe_extra` in the pre-refactor heuristic, but in plan units.
    constexpr long kBytesBF16 = 2;
    const long expert_gate_up = plan.NumExperts * plan.MoeMUp * plan.C * kBytesBF16;
    const long expert_down = plan.NumExperts * plan.MoeM * plan.C * kBytesBF16;
    const long permuted_tokens = 2L * plan.B * plan.T * plan.TopK * plan.C * kBytesBF16;
    const long moe_bwd_act = 2L * plan.B * plan.T * plan.TopK * plan.MoeMUp * kBytesBF16;
    return expert_gate_up + expert_down + permuted_tokens + moe_bwd_act;
}

/// Extra BF16 slack for ops whose backward peak is not yet modeled.
/// Inherits the legacy `extra_tmp = max(BT*C, BT*QKV, BT*MUp) * dtype_size`.
[[nodiscard]] long unmodeled_bwd_tmp(const BufferPlan& plan) {
    const long BT = plan.B * plan.T;
    const long dtype_bytes = static_cast<long>(get_dtype_size(plan.act_dtype));
    return std::max({BT * plan.C, BT * plan.QKV, BT * plan.MUp}) * dtype_bytes;
}

}  // namespace

// Heuristic sizing from the plan-level peak alone. The 2x multiplier
// compensates for the fact that plan-level coverage is ~55% of actual
// runtime peak — remaining gap is filled by `moe_extra`, `extra_tmp`,
// `safety_bytes`, and the various architecture-specific slacks below.
// (CPU training uses 1x because the stack resets every layer boundary.)
static long heuristic_required_bytes(const BufferPlan& plan,
                                     const PretrainedConfig& cfg,
                                     const RuntimeOptions& options,
                                     long moe_stack_slack) {
    const long plan_peak = plan.plan_stack_peak_bytes();
    const long base_multiplier = options.CpuTraining ? 1L : 2L;
    const long moe_extra = moe_op_internal_estimate(plan);
    const long safety_floor = plan.lora_only ? (32L * 1024 * 1024) : (64L * 1024 * 1024);
    const long safety_bytes = std::max(safety_floor, plan_peak / 8);
    const long extra_tmp = unmodeled_bwd_tmp(plan);

    long required = std::max(1024L * 1024, plan_peak * base_multiplier + moe_extra + safety_bytes + extra_tmp);

    const long slack_bytes = options.CpuTraining ? (128L * 1024 * 1024)
                             : plan.lora_only    ? (256L * 1024 * 1024)
                                                 : (512L * 1024 * 1024);
    required += slack_bytes;

    if (is_qwen3_hybrid_lora(plan, cfg)) {
        required += qwen3_hybrid_lora_heuristic_slack(plan, options);
    }

    required += moe_stack_slack;

    if (options.UseCudaGraphs) {
        const long graph_extra_slack = plan.lora_only ? (512L * 1024 * 1024) : (1024L * 1024 * 1024);
        required += graph_extra_slack;
    }
    return required;
}

// Graph-walk peak plus a graph-specific safety factor. Used only when the
// compiled backward graph is available — the graph walk accounts for op-
// internal temps (flash attention, Mamba scan, ChunkGatedDeltaRule, ...)
// that the plan-only estimate misses, so a much tighter safety margin
// suffices than the 2x multiplier in the heuristic path.
static long graph_required_bytes(long graph_peak, const RuntimeOptions& options) {
    if (graph_peak <= 0) return 0;
    const long safety = options.CpuTraining ? std::max(64L * 1024 * 1024, graph_peak / 8)
                                            : std::max(128L * 1024 * 1024, graph_peak / 3);
    return graph_peak + safety;
}

// Absolute minimum floors — LoRA 512 MiB, full fine-tune 3 GiB (CPU training
// 512 MiB). CUDA graphs bump these up since capture retains more transient
// state. The `SUROGATE_MIN_STACK_MB` env var, when set, replaces the
// computed floor entirely.
static long min_stack_floor(const BufferPlan& plan, const PretrainedConfig& cfg, const RuntimeOptions& options) {
    long floor = options.CpuTraining ? (512L * 1024 * 1024)
                 : plan.lora_only    ? (512L * 1024 * 1024)
                                     : (3L * 1024 * 1024 * 1024);
    if (options.UseCudaGraphs) {
        floor = std::max(floor, plan.lora_only ? (1024L * 1024 * 1024) : (4L * 1024 * 1024 * 1024));
    }
    floor = std::max(floor, qwen3_hybrid_lora_floor(plan, cfg, options));
    if (const char* env = std::getenv("SUROGATE_MIN_STACK_MB")) {
        const long mb = std::max(64L, std::atol(env));
        floor = mb * 1024 * 1024;
    }
    return floor;
}

long required_stack_bytes(const BufferPlan& plan,
                          const CompiledGraph* bwd_graph,
                          const PretrainedConfig& cfg,
                          const RuntimeOptions& options) {
    // Global MoE slack (applied to both heuristic path and floor). Env-
    // override kept for operator control when a model spikes beyond the
    // built-in 2 GiB allowance.
    long moe_stack_slack = plan.has_moe() ? (2048L * 1024 * 1024) : 0L;
    if (const char* env = std::getenv("SUROGATE_STACK_SLACK_MB")) {
        const long mb = std::max(0L, std::atol(env));
        moe_stack_slack = std::max(moe_stack_slack, mb * 1024 * 1024);
    }

    const long heuristic = heuristic_required_bytes(plan, cfg, options, moe_stack_slack);
    const long graph_peak = graph_backward_stack_peak(bwd_graph, plan);
    const long graph_based = graph_required_bytes(graph_peak, options);
    const long floor = min_stack_floor(plan, cfg, options) + moe_stack_slack;

    return std::max({heuristic, graph_based, floor});
}

}  // namespace dsl
