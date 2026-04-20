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

#include <unordered_set>
#include <vector>

#include "config/pretrained_config.h"
#include "runtime/dsl/dsl_runtime_config.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "runtime/training/runtime_options.h"
#include "utilities/utils.h"

namespace dsl {

struct CompiledGraph;  // fwd — defined in graph_compiler.h; used by stack-sizing helpers.

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
    // Forward-activation sharing (share_ln1, share_ln2, share_qkv,
    // share_att, share_att_out, share_mlp_up, share_swiglu,
    // share_residual_att, share_mlp_down, share_qk_rstd) was removed
    // as part of the Phase 4 M5 cleanup — every layer now allocates
    // its own per-slot buffer. See commit d3ec195.

    // ---------------- Gradient sharing decisions ----------------
    bool share_grads = false;          ///< d_ln1/d_ln2/d_res_att/d_att_out shared
    bool share_d_att = false;          ///< d_att shared only when attn dims are uniform
    bool share_res_ffn_grad = false;   ///< alternating d_res_ffn[0/1]
    bool share_mlp_down_grad = false;  ///< alternating d_mlp_down[0/1]

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
                            ETensorDType grad_dtype);
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
