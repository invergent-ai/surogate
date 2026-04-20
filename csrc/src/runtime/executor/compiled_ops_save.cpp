// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CompiledExecutor: tensor save / persistence methods.

#include "runtime/executor/compiled_ops.h"
#include "runtime/dsl/tensor_slot_dispatch.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/core/fp8_scaling_config.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_weights_manager.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "runtime/moe/moe_types.h"
#include "recipes/recipe.h"
#include "runtime/training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {

namespace {

// Per-block MoE slots (router_logits, routing_weights, permuted_input,
// expert_*, moe_out, ...) have dedicated TensorSlot enum values. The
// name-based tail covers MoE side-channels deliberately NOT modelled as
// per-block slots: `moe_expert_offsets` / `moe_gather_indices` (global
// scratch) and `ep_*` (expert-parallel internal comms). To add a new MoE
// tensor, extend the TensorSlot enum or update the tail list here.
bool is_moe_tensor_name(const std::string& n) {
    int lid = -1;
    std::string field;
    const std::string probe = parse_block_param(n, lid, field) ? strip_ssa_suffix(field) : strip_ssa_suffix(n);
    switch (builtin_slot_from_name(probe)) {
        case TensorSlot::BlockRouterLogits:
        case TensorSlot::BlockRouterProbs:
        case TensorSlot::BlockRoutingWeights:
        case TensorSlot::BlockRoutingIndices:
        case TensorSlot::BlockPermutedInput:
        case TensorSlot::BlockScatterIndices:
        case TensorSlot::BlockExpertGateUp:
        case TensorSlot::BlockExpertAct:
        case TensorSlot::BlockExpertDown:
        case TensorSlot::BlockMoeOut: return true;
        default: break;
    }
    return probe == "moe_expert_offsets" || probe == "moe_gather_indices" || n.rfind("ep_", 0) == 0;
}

}  // namespace

// Historical note: `strip_autodiff_accum_suffix` and
// `should_alias_autodiff_accum_name` lived here. Both were name-based
// heuristics that attempted to detect autodiff accumulator variants
// (`d_foo_from_N` / `d_foo_accum_N`) by string-matching and then alias them
// to the base `d_foo` slot. The heuristic misclassified intermediate
// gradients as parameter grads — exactly the bug class we retired — so both
// helpers have been deleted. `ensure_output_tensor` now consults
// `TensorMeta::kind == AccumTemp` directly; the classifier is authoritative
// and accumulators always get their own buffer.

void CompiledExecutor::save_moe_layer_tensors(int layer_idx) {
    // Copy MoE tensors from this layer to persistent storage before stack restore.
    // This allows stack memory to be reclaimed while preserving tensors for backward.
    if (mCapturing) {
        return;
    }
    if (mConfig.NumExperts == 0) {
        return;
    }

    if (!mCurrentGraph) return;

    // Build layer prefix pattern (e.g., "blocks[5]." or "layer5.")
    // Check the first block tensor name to determine naming convention
    std::string layer_prefix;
    for (const auto& [n, _] : mCurrentGraph->tensor_name_to_id) {
        if (n.rfind("layer", 0) == 0 && n.size() > 5 && std::isdigit(n[5])) {
            layer_prefix = "layer" + std::to_string(layer_idx) + ".";
            break;
        }
        if (n.rfind("blocks[", 0) == 0) {
            layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";
            break;
        }
    }
    if (layer_prefix.empty()) {
        layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";
    }

    // Iterate through compile-time tensor name map looking for MoE tensors from this layer
    for (const auto& [name, tid] : mCurrentGraph->tensor_name_to_id) {
        // Skip global MoE tensors - these are scratch space reused each layer
        // and are NOT needed for backward (backward uses mMoEExpertOffsetsGPU).
        if (name == "moe_expert_offsets" || name == "moe_gather_indices") {
            continue;
        }

        // Check if tensor belongs to this layer
        if (name.find(layer_prefix) != 0) {
            continue;
        }

        // Check if this is an MoE-related tensor that needs persistent storage
        bool is_moe_tensor =
            (name.find("moe_") != std::string::npos || name.find("ep_") != std::string::npos ||
             name.find("scatter_indices") != std::string::npos || name.find("routing_weights") != std::string::npos ||
             name.find("routing_indices") != std::string::npos || name.find("router_") != std::string::npos ||
             name.find("permuted") != std::string::npos || name.find("expert_") != std::string::npos);

        if (!is_moe_tensor) continue;
        if (tid < 0 || static_cast<std::size_t>(tid) >= mTensors.size()) continue;
        auto& tensor = mTensors[static_cast<std::size_t>(tid)];
        if (!tensor.Data) continue;

        const size_t bytes = tensor.bytes();
        if (bytes == 0) {
            continue;
        }

        // Allocate or resize persistent buffer if needed (Phase 3 subsystem #6
        // flip: arena bump + cudaMalloc fallback).
        auto buf_it = mMoeSavedBuffers.find(name);
        if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
            if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                auto ab_it = mMoeSavedArenaBacked.find(name);
                const bool old_was_arena = ab_it != mMoeSavedArenaBacked.end() && ab_it->second;
                if (!old_was_arena) CUDA_CHECK(cudaFree(buf_it->second));
            }
            auto a = allocate_moe_saved(bytes);
            mMoeSavedBuffers[name] = a.ptr;
            mMoeSavedSizes[name] = bytes;
            mMoeSavedArenaBacked[name] = a.arena_backed;
        }

        // Copy data to persistent buffer
        void* dst_buffer = mMoeSavedBuffers[name];
        CUDA_CHECK(cudaMemcpyAsync(dst_buffer, tensor.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));

        // Update all authoritative tables to point at the persistent buffer.
        // If we only patch mTensors, resolve_tensor() can still pick a stale
        // stack-backed alias from mNamedTensors first.
        Tensor saved_tensor = tensor;
        saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
        tensor = saved_tensor;
        mNamedTensors[name] = saved_tensor;
        if (mSaved) {
            auto saved_it = mSaved->find(name);
            if (saved_it != mSaved->end() || mSaveSet.find(name) != mSaveSet.end()) {
                (*mSaved)[name] = saved_tensor;
            }
        }
    }
}

void CompiledExecutor::prepare_saved_buffers_for_capture(const std::vector<std::string>& save_list,
                                                         const CompiledGraph* capture_graph) {
    // Only needed when recompute is enabled or MoE tensors require persistence.
    if (!mSaved) {
        return;
    }

    const bool recompute_enabled = mRecomputeEnabled;
    // When forward replay is active, ALL block tensors will be regenerated
    // by replay_layer_forward during backward — no persistent buffers needed.
    const bool forward_replay_active = recompute_enabled && static_cast<bool>(mRecomputeFn);

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        // Forward replay: all block tensors are replayed, no persistent save needed
        if (forward_replay_active) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(tensor_name, layer_idx, field)) {
                return true;
            }
        }
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(tensor_name, layer_idx, field)) {
            return mSlotRegistry->will_recompute(strip_ssa_suffix(field), lora_only_mode);
        }
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    auto is_shared_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(field))) {
                return entry->memory_hint == ActivationMemoryHint::Shared;
            }
            return std::nullopt;
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->memory_hint == ActivationMemoryHint::Shared;
        }
        return std::nullopt;
    };

    auto is_mapped_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int lid = -1;
        std::string fld;
        if (parse_block_param(name, lid, fld)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(fld))) {
                return entry->slot == TensorSlot::Mapped;
            }
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->slot == TensorSlot::Mapped;
        }
        return std::nullopt;
    };

    auto should_persist = [&](const std::string& name, bool prefer_live, bool force_persist) -> bool {
        if (force_persist) {
            return true;
        }
        if (!recompute_enabled || prefer_live) {
            return false;
        }
        auto mapped = is_mapped_slot(name);
        if (mapped.has_value() && mapped.value()) {
            return true;
        }
        auto shared = is_shared_slot(name);
        if (shared.has_value()) {
            return shared.value();
        }
        // Unknown tensors default to non-persistent to avoid over-allocating
        // during graph-capture preallocation.
        return false;
    };

    const bool debug_save_buffers = (std::getenv("SUROGATE_DEBUG_SAVE_BUFFERS") != nullptr);
    auto ensure_buffer = [&](const std::string& name, size_t bytes) {
        if (bytes == 0) {
            return;
        }
        auto buf_it = mMoeSavedBuffers.find(name);
        if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
            if (debug_save_buffers) {
                std::cerr << "[SAVE-BUF] alloc name=" << name << " bytes=" << bytes
                          << " old_bytes=" << (buf_it == mMoeSavedBuffers.end() ? 0 : mMoeSavedSizes[name])
                          << std::endl;
            }
            if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                auto ab_it = mMoeSavedArenaBacked.find(name);
                const bool old_was_arena = ab_it != mMoeSavedArenaBacked.end() && ab_it->second;
                if (!old_was_arena) CUDA_CHECK(cudaFree(buf_it->second));
            }
            auto a = allocate_moe_saved(bytes);
            mMoeSavedBuffers[name] = a.ptr;
            mMoeSavedSizes[name] = bytes;
            mMoeSavedArenaBacked[name] = a.arena_backed;
        }
    };

    // A `_flat` suffix on a block-level field means "view the tensor as 2D by
    // collapsing the first two dims" — this is just a reinterpretation of the
    // same storage. `flatten_2d()` applies that view when needed; we collapse
    // the field naming lookup into a single source of truth (the slot
    // registry) and keep the flat-ness as a separate boolean.
    auto flatten_2d = [&](Tensor t) -> Tensor {
        if (t.Rank >= 3) {
            return view_tensor(t, {t.Sizes[0] * t.Sizes[1], t.Sizes[2]});
        }
        return t;
    };

    auto resolve_source = [&](const std::string& name) -> std::optional<Tensor> {
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                return mTensors[tid];
            }
        }

        // Global / IO slots via the shared helper. `_flat` aliases aren't in
        // the static name→slot table (they have distinct 2D shapes);
        // `resolve_slot_with_flat` handles the suffix stripping centrally.
        bool name_is_flat = false;
        TensorSlot glob_slot = resolve_slot_with_flat(name, &name_is_flat);
        if (Tensor* t = global_activation_ptr(mRunState, glob_slot)) {
            return name_is_flat ? flatten_2d(*t) : *t;
        }
        // FreqCis opts out of the helper because it depends on the original
        // name (qualified or not) for multi-table models.
        if (glob_slot == TensorSlot::FreqCis) {
            return mRunState.rope_freqs(name);
        }
        // Qualified rope references (blocks[N].rope_freqs, …) still need the name.
        if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
            return mRunState.rope_freqs(name);
        }

        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(name, layer_idx, field)) {
            return std::nullopt;
        }
        if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
            return std::nullopt;
        }
        const std::string base_field = strip_ssa_suffix(field);
        const bool is_flat = (base_field.size() >= 5 && base_field.compare(base_field.size() - 5, 5, "_flat") == 0);
        const TensorSlot slot = builtin_slot_from_name(base_field);

        // Block-scope activation (including MoE slots) via the shared helper.
        // `is_flat` applies a 2D view when the caller requested `<field>_flat`
        // on a 3D buffer. MoE slots like `moe_out` are already 2D, so
        // flatten_2d is a no-op for them (guarded by Rank >= 3).
        if (Tensor* t = block_activation_ptr(mRunState, layer_idx, slot)) {
            return is_flat ? flatten_2d(*t) : *t;
        }

        return std::nullopt;
    };

    auto infer_block_bytes = [&](const std::string& name, std::size_t& out_bytes) -> bool {
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(name, layer_idx, field)) {
            return false;
        }
        std::string base_field = strip_ssa_suffix(field);
        const long B = (mB > 0) ? mB : mRunState.B;
        const long T = (mT > 0) ? mT : mRunState.T;
        if (B <= 0 || T <= 0) {
            return false;
        }
        const long C = mConfig.HiddenSize;
        const long D = mRunState.buffer_plan().layer_intermediate(layer_idx);
        const long Hq = mConfig.NumQueryHeads;
        const long Hkv = mConfig.NumKeyValHeads;
        const long QKV = mRunState.buffer_plan().layer_qkv(layer_idx);
        const long AttnDim = mRunState.buffer_plan().layer_attn_dim(layer_idx);
        const long MUp = mRunState.buffer_plan().layer_mlp_up(layer_idx);

        // `_flat` fields share the backing buffer of their 3D counterpart but
        // carry a 2D shape (B*T, innermost). Strip the suffix so the slot
        // enum dispatch covers both variants in one table.
        const bool is_flat = (base_field.size() >= 5 && base_field.compare(base_field.size() - 5, 5, "_flat") == 0);
        if (is_flat) {
            base_field = base_field.substr(0, base_field.size() - 5);
        }

        std::vector<long> shape;
        ETensorDType dtype = ETensorDType::BF16;
        if (mConfig.NumLayers > 0) {
            dtype = mRunState.simplified_acts(0).ln1.DType;
        }

        // Dispatch the per-slot shape formula. `base_field` at this point has
        // both the SSA suffix and any `_flat` suffix stripped, so a single
        // `builtin_slot_from_name()` call covers every alias.
        switch (builtin_slot_from_name(base_field)) {
            case TensorSlot::BlockLN1:
            case TensorSlot::BlockLN2:
            case TensorSlot::BlockAttOut:
            case TensorSlot::BlockMLPDown:
            case TensorSlot::BlockResidualAtt:
            case TensorSlot::BlockResidualFFN:
            case TensorSlot::BlockHOut:
                shape = is_flat ? std::vector<long>{B * T, C} : std::vector<long>{B, T, C};
                break;
            case TensorSlot::BlockMLPUp:
                shape = is_flat ? std::vector<long>{B * T, MUp} : std::vector<long>{B, T, MUp};
                break;
            case TensorSlot::BlockSwiGLU:
                shape = is_flat ? std::vector<long>{B * T, D} : std::vector<long>{B, T, D};
                break;
            case TensorSlot::BlockQKV:
            case TensorSlot::BlockQKVRoPE:
                shape = is_flat ? std::vector<long>{B * T, QKV} : std::vector<long>{B, T, QKV};
                break;
            case TensorSlot::BlockAtt:
                shape = is_flat ? std::vector<long>{B * T, AttnDim} : std::vector<long>{B, T, AttnDim};
                break;
            case TensorSlot::BlockLN1RSTD:
            case TensorSlot::BlockLN2RSTD:
                shape = {B, T};
                dtype = ETensorDType::FP32;
                break;
            case TensorSlot::BlockQRSTD:
                shape = {B, T, Hq};
                dtype = ETensorDType::FP32;
                break;
            case TensorSlot::BlockKRSTD:
                shape = {B, T, Hkv};
                dtype = ETensorDType::FP32;
                break;
            case TensorSlot::BlockLSE:
                shape = {B, Hq, T};
                dtype = ETensorDType::FP32;
                break;
            default: return false;  // unknown slot — can't infer bytes
        }

        const std::size_t nelem = shape_nelem(shape);
        out_bytes = nelem * static_cast<std::size_t>(get_dtype_size(dtype));
        return out_bytes > 0;
    };

    auto infer_graph_ref_bytes = [&](const std::string& name, std::size_t& out_bytes) -> bool {
        const CompiledGraph* graph = capture_graph ? capture_graph : mCurrentGraph;
        if (!graph) {
            return false;
        }
        auto ref_bytes = [&](const TensorRef& ref, std::size_t& bytes_out) -> bool {
            if (ref.name != name || ref.shape.empty()) {
                return false;
            }
            const std::size_t nelem = shape_nelem(ref.shape);
            if (nelem == 0) {
                return false;
            }
            bytes_out = nelem * static_cast<std::size_t>(get_dtype_size(ref.dtype));
            return bytes_out > 0;
        };
        // Prefer producer shape (outputs), then fallback to consumer refs (inputs).
        for (const auto& op : graph->ops) {
            for (const auto& ref : op.outputs) {
                if (ref_bytes(ref, out_bytes)) {
                    return true;
                }
            }
        }
        for (const auto& op : graph->ops) {
            for (const auto& ref : op.inputs) {
                if (ref_bytes(ref, out_bytes)) {
                    return true;
                }
            }
        }
        return false;
    };

    // Same predicates as save_tensors(), duplicated here because this function
    // (prepare_saved_buffers_for_capture) runs in a different context. They
    // share a slot-enum definition; names are NOT re-compared as strings.
    auto is_lora_hook_activation = [&](const std::string& tensor_name) -> bool {
        if (!mLoRAConfig) {
            return false;
        }
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(tensor_name, layer_idx, field)) {
            return false;
        }
        const TensorSlot slot = builtin_slot_from_name(strip_ssa_suffix(field));
        return slot == TensorSlot::BlockSwiGLU || slot == TensorSlot::BlockAtt;
    };
    auto is_forced_persist_global = [&](const std::string& n) -> bool {
        switch (resolve_slot_with_flat(n)) {
            case TensorSlot::LNFinal:
            case TensorSlot::LNFinalRSTD:
            case TensorSlot::FinalResidual: return true;
            default: return false;
        }
    };
    // is_moe_tensor_name hoisted to file-scope helper above.

    for (const auto& name : save_list) {
        if (mWeights.has(name)) {
            continue;
        }

        // LoRA hook activations (att, swiglu) need persistence for backward LoRA
        // gradient computation — BUT only when forward replay is NOT active.
        // When replay regenerates these tensors, no persistent buffer is needed.
        const bool force_lora_hook = !forward_replay_active && is_lora_hook_activation(name);
        const bool force_persist_name = is_forced_persist_global(name) || force_lora_hook;
        const bool is_moe_tensor = is_moe_tensor_name(name);
        const bool prefer_live = prefer_live_tensor(name);
        const bool force_persist = is_moe_tensor && mConfig.NumExperts > 0;
        const bool need_persist = should_persist(name, prefer_live, force_persist || force_persist_name);

        // Block-level tensors may be stack-backed and persisted by
        // persist_saved_layer_tensors() during execute_forward. Pre-allocate
        // buffers for those only when they can actually require persistence.
        int layer_idx = -1;
        std::string field;
        const bool is_block_tensor = parse_block_param(name, layer_idx, field);
        if (!need_persist && (!is_block_tensor || prefer_live)) {
            continue;
        }

        auto src_opt = resolve_source(name);
        // For capture safety, preallocate block save tensors even when they
        // appear non-stack-backed at prep time. Runtime dispatch may route
        // them through stack-backed storage depending on recompute/layout.
        if (src_opt.has_value() && src_opt->Data) {
            ensure_buffer(name, src_opt->bytes());
            continue;
        }
        std::size_t inferred_bytes = 0;
        if (infer_block_bytes(name, inferred_bytes)) {
            ensure_buffer(name, inferred_bytes);
            continue;
        }
        if (infer_graph_ref_bytes(name, inferred_bytes)) {
            ensure_buffer(name, inferred_bytes);
            continue;
        }
        // Infer sizes for non-block tensors. Dispatch on the slot enum so
        // every alias (including `_flat` suffixes not in the static name→slot
        // table) resolves to the same shape formula in one place.
        {
            const long B = (mB > 0) ? mB : mRunState.B;
            const long T = (mT > 0) ? mT : mRunState.T;
            const long C = mConfig.HiddenSize;
            if (B > 0 && T > 0 && C > 0) {
                TensorSlot slot = resolve_slot_with_flat(name);
                const std::size_t elem_bf16 = static_cast<std::size_t>(get_dtype_size(ETensorDType::BF16));
                std::size_t nbytes = 0;
                switch (slot) {
                    case TensorSlot::LNFinal:        // xF, ln_final (+ xF_flat / ln_final_flat via suffix strip)
                    case TensorSlot::FinalResidual:  // final_residual, residual_final
                    case TensorSlot::Encoded:        // x0, encoded
                        nbytes = static_cast<std::size_t>(B * T * C) * elem_bf16;
                        break;
                    case TensorSlot::LNFinalRSTD: nbytes = static_cast<std::size_t>(B * T) * sizeof(float); break;
                    default: break;
                }
                if (nbytes > 0) {
                    ensure_buffer(name, nbytes);
                }
            }
        }
    }

    // Some ops persist internal saved tensors with op-scoped names that are not
    // in forward.save (e.g., mamba_gated_rmsnorm saves "<op_id>.rstd"/".normed").
    // Pre-allocate those buffers before capture to avoid cudaMalloc in dispatch.
    const CompiledGraph* graph = capture_graph ? capture_graph : mCurrentGraph;
    if (graph) {
        for (const auto& op : graph->ops) {
            if (op.type != CompiledOpType::MambaGatedRMSNorm || op.op_id.empty()) {
                continue;
            }
            if (op.inputs.empty() || op.inputs[0].shape.empty()) {
                continue;
            }
            const auto& x_ref = op.inputs[0];
            const auto x_elems = shape_nelem(x_ref.shape);
            if (x_elems == 0) {
                continue;
            }

            const std::size_t normed_bytes = x_elems * static_cast<std::size_t>(get_dtype_size(x_ref.dtype));
            ensure_buffer(op.op_id + ".out_fallback", normed_bytes);
            ensure_buffer(op.op_id + ".normed", normed_bytes);

            const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;
            long rows = x_ref.shape[0];
            if (x_ref.shape.size() >= 3) {
                rows *= x_ref.shape[1];
            }
            if (rows > 0 && groups > 0) {
                const std::size_t rstd_bytes =
                    static_cast<std::size_t>(rows) * static_cast<std::size_t>(groups) * sizeof(float);
                ensure_buffer(op.op_id + ".rstd", rstd_bytes);
            }
        }
    }
}

void CompiledExecutor::save_tensors(const std::vector<std::string>& save_list, bool force_persist) {
    if (!mSaved) {
        return;
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture = (cudaStreamIsCapturing(mRunState.MainStream, &capture_status) == cudaSuccess &&
                             capture_status != cudaStreamCaptureStatusNone);
    const bool capturing = mCapturing || in_capture;

    // Recompute is only active when explicitly enabled for this execution.
    // This gate is set by GraphExecutor after validating runtime options + plan.
    const bool recompute_enabled = mRecomputeEnabled;
    const bool forward_replay_active = recompute_enabled && static_cast<bool>(mRecomputeFn);
    auto contains_ci = [](std::string_view haystack, std::string_view needle) {
        std::string h(haystack);
        std::string n(needle);
        std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return h.find(n) != std::string::npos;
    };
    const bool is_qwen3_5_model =
        contains_ci(mConfig.ModelTypeName, "qwen3_5") || contains_ci(mConfig.ModelTypeName, "qwen3.5") ||
        contains_ci(mConfig.ArchitectureName, "qwen3_5") || contains_ci(mConfig.ArchitectureName, "qwen3.5");

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (force_persist) {
            return false;
        }
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(tensor_name, layer_idx, field)) {
            const std::string base_field = strip_ssa_suffix(field);
            // When forward replay is active, ALL block tensors will be regenerated
            if (forward_replay_active) {
                // Qwen3.5 replay drifts at LN1 if we drop the exact original normalized
                // activation into metadata-only mode. Keep those tensors persistent so
                // replay consumes the saved forward values instead of recomputing them.
                // Dispatch via slot enum so any alias (`ln1_flat`, `ln`, `ln_flat`,
                // `ln1_rstd`, `ln_rstd`) maps to the same decision in one place.
                if (is_qwen3_5_model) {
                    const TensorSlot slot = builtin_slot_from_name(base_field);
                    if (slot == TensorSlot::BlockLN1 || slot == TensorSlot::BlockLN1RSTD) {
                        return false;
                    }
                }
                return true;
            }
            return mSlotRegistry->will_recompute(base_field, lora_only_mode);
        }
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    // Helper to copy tensor to persistent buffer when needed in recompute mode.
    // Returns true if tensor was copied to persistent storage, false if metadata-only save.
    auto is_shared_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(field))) {
                return entry->memory_hint == ActivationMemoryHint::Shared;
            }
            return std::nullopt;
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->memory_hint == ActivationMemoryHint::Shared;
        }
        return std::nullopt;
    };

    auto is_mapped_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(field))) {
                return entry->slot == TensorSlot::Mapped;
            }
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->slot == TensorSlot::Mapped;
        }
        return std::nullopt;
    };

    auto should_persist = [&](const std::string& name, bool prefer_live, bool force_persist_name) -> bool {
        if (force_persist || force_persist_name) {
            return true;
        }
        if (!recompute_enabled || prefer_live) {
            return false;
        }
        auto mapped = is_mapped_slot(name);
        if (mapped.has_value() && mapped.value()) {
            // Slot resolves to Mapped: no persistent buffer, so saved data must be copied.
            return true;
        }
        auto shared = is_shared_slot(name);
        if (shared.has_value()) {
            return shared.value();
        }
        // Unknown tensors default to non-persistent unless explicitly forced.
        return false;
    };

    auto save_tensor_with_policy =
        [&](const std::string& name, const Tensor& src, bool prefer_live, bool force_persist_name) -> void {
        if (force_persist) {
            prefer_live = false;
            force_persist_name = true;
        }
        if (prefer_live) {
            // Save metadata only - will resolve from live buffer or recompute
            Tensor meta = src;
            meta.Data = nullptr;
            (*mSaved)[name] = meta;
            return;
        }

        const bool src_stack_backed = src.Data && mRunState.Stack.owns(src.Data);
        const bool need_persist =
            (should_persist(name, prefer_live, force_persist_name) || src_stack_backed) && src.Data != nullptr;
        if (need_persist && src.Data != nullptr) {
            const size_t bytes = src.bytes();
            auto buf_it = mMoeSavedBuffers.find(name);
            if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
                if (capturing) {
                    // During CUDA graph capture we cannot allocate new buffers.
                    // Fall back to metadata-only save so backward can resolve
                    // from live/recomputed tensors instead of aborting capture.
                    Tensor meta = src;
                    meta.Data = nullptr;
                    (*mSaved)[name] = meta;
                    return;
                }
                if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                    auto ab_it = mMoeSavedArenaBacked.find(name);
                    const bool old_was_arena = ab_it != mMoeSavedArenaBacked.end() && ab_it->second;
                    if (!old_was_arena) CUDA_CHECK(cudaFree(buf_it->second));
                }
                auto a = allocate_moe_saved(bytes);
                mMoeSavedBuffers[name] = a.ptr;
                mMoeSavedSizes[name] = bytes;
                mMoeSavedArenaBacked[name] = a.arena_backed;
            }
            void* dst_buffer = mMoeSavedBuffers[name];
            CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved_tensor;
            saved_tensor.DType = src.DType;
            saved_tensor.Rank = src.Rank;
            for (int i = 0; i < src.Rank; ++i) {
                saved_tensor.Sizes[i] = src.Sizes[i];
            }
            saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
            (*mSaved)[name] = saved_tensor;
            return;
        }

        // Non-recompute mode: just store reference
        (*mSaved)[name] = src;
    };

    // LoRA-hook activations (att, swiglu) are the tensors the LoRA backward
    // pass reads directly — they must survive forward→backward even when
    // recompute is enabled. Dispatch on the slot enum: any block-scope slot
    // that corresponds to a LoRA input qualifies.
    auto is_lora_hook_activation = [&](const std::string& tensor_name) -> bool {
        if (!mLoRAConfig) {
            return false;
        }
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(tensor_name, layer_idx, field)) {
            return false;
        }
        const TensorSlot slot = builtin_slot_from_name(strip_ssa_suffix(field));
        return slot == TensorSlot::BlockSwiGLU || slot == TensorSlot::BlockAtt;
    };

    // The set of globals that must always be persisted (even when recompute
    // is enabled) — the lm-head backward reads from these and can't replay
    // them. Encoded here as slot-enum dispatch; `xF_flat` is the one name
    // without a dedicated TensorSlot (it's a 2D view of LNFinal).
    auto is_forced_persist_global = [&](const std::string& n) -> bool {
        // Resolve the slot, honouring `_flat` suffix aliases that aren't in
        // the static name→slot table (they carry their own 2D shape).
        switch (resolve_slot_with_flat(n)) {
            case TensorSlot::LNFinal:  // covers xF, ln_final, xF_flat, ln_final_flat
            case TensorSlot::LNFinalRSTD:
            case TensorSlot::FinalResidual: return true;
            default: return false;
        }
    };

    for (const auto& name : save_list) {
        // LoRA hook activations (att, swiglu) need persistence for backward LoRA
        // gradient computation — BUT only when forward replay is NOT active.
        // When replay regenerates these tensors, no persistent buffer is needed.
        const bool force_lora_hook = !forward_replay_active && is_lora_hook_activation(name);
        const bool force_persist_name = is_forced_persist_global(name) || force_lora_hook;
        const bool prefer_live = prefer_live_tensor(name);
        // Skip tensors already saved directly by dispatch (e.g. mamba_gated_rmsnorm saves rstd/normed),
        // unless we are forcing a persistent copy to replace metadata-only entries.
        auto saved_it = mSaved->find(name);
        if (saved_it != mSaved->end()) {
            // Layer-end replay bookkeeping pre-seeds metadata-only entries for block tensors.
            // If this tensor should not remain metadata-only, refresh it now with live data.
            const bool needs_refresh =
                (saved_it->second.Data == nullptr) && (force_persist || force_persist_name || !prefer_live);
            if (!needs_refresh) {
                continue;
            }
            mSaved->erase(saved_it);
        }

        // First check intermediate tensors via flat vector (fast path) or mirror map
        Tensor* found_tensor = nullptr;
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                found_tensor = &mTensors[tid];
            }
        }
        if (found_tensor) {
            const bool force_persist = is_moe_tensor_name(name) && mConfig.NumExperts > 0;
            save_tensor_with_policy(name, *found_tensor, prefer_live, force_persist || force_persist_name);
            continue;
        }

        // Route through the shared slot-dispatch helpers — every recognized
        // name resolves via builtin_slot_from_name() → {block_activation_ptr,
        // global_activation_ptr}, with the slot enum as the single source of
        // truth. Per-name dispatch (the previous 130-line if/else ladder) lived
        // in THIS file and drifted out of sync with the other resolvers; the
        // shared helpers make every new slot a one-line change in ONE file.
        // `prefer_live` already computed at loop top.

        // Global-scope via the shared slot-dispatch helper. `_flat` aliases
        // (xF_flat, ln_final_flat, …) carry their own 2D shape in the DSL
        // layout so they're NOT added to the static name→slot table — that
        // would clobber the 2D shape with the 3D parent's shape at compile
        // time. `resolve_slot_with_flat` handles the suffix stripping.
        bool is_flat_view = false;
        TensorSlot glob_slot = resolve_slot_with_flat(name, &is_flat_view);
        if (Tensor* t = global_activation_ptr(mRunState, glob_slot)) {
            if (is_flat_view && t->Rank >= 3) {
                Tensor flat = view_tensor(*t, {t->Sizes[0] * t->Sizes[1], t->Sizes[2]});
                save_tensor_with_policy(name, flat, prefer_live, force_persist_name);
            } else {
                save_tensor_with_policy(name, *t, prefer_live, force_persist_name);
            }
            continue;
        }
        if (glob_slot == TensorSlot::FreqCis) {
            save_tensor_with_policy(name, mRunState.rope_freqs(name), prefer_live, force_persist_name);
            continue;
        }
        // Qualified rope references (blocks[N].rope_freqs, …) still need the name.
        if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
            save_tensor_with_policy(name, mRunState.rope_freqs(name), prefer_live, force_persist_name);
            continue;
        }

        // Block-scope fields.
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            const std::string base_field = strip_ssa_suffix(field);
            const bool is_flat = (base_field.size() >= 5 && base_field.compare(base_field.size() - 5, 5, "_flat") == 0);
            const TensorSlot slot = builtin_slot_from_name(base_field);
            if (Tensor* t = block_activation_ptr(mRunState, layer_idx, slot)) {
                if (is_flat && t->Rank >= 3) {
                    Tensor flat = view_tensor(*t, {t->Sizes[0] * t->Sizes[1], t->Sizes[2]});
                    save_tensor_with_policy(name, flat, prefer_live, force_persist_name);
                } else {
                    save_tensor_with_policy(name, *t, prefer_live, force_persist_name);
                }
                continue;
            }

            // MoE slots now have dedicated TensorSlot enum values — the
            // block_activation_ptr call above already covers them. If the
            // helper returned nullptr, the field isn't a recognized block
            // activation; fall through to the weight-store check.

            if (mWeights.has(name)) {
                (*mSaved)[name] = mWeights.get(name);
                continue;
            }
            throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
        }

        if (mWeights.has(name)) {
            (*mSaved)[name] = mWeights.get(name);
            continue;
        }
        throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
    }

    // For MoE models, copy expert_offsets data to persistent storage for backward pass
    // The original tensor is stack-allocated and will be freed before backward runs
    if (mConfig.NumExperts > 0) {
        Tensor* moe_offsets = nullptr;
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id("moe_expert_offsets");
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                moe_offsets = &mTensors[tid];
            }
        }
        if (moe_offsets) {
            const Tensor& src = *moe_offsets;
            const int num_elements = static_cast<int>(src.nelem());
            mMoEExpertOffsetsData.resize(num_elements);
            CUDA_CHECK(
                cudaMemcpy(mMoEExpertOffsetsData.data(), src.Data, num_elements * sizeof(int), cudaMemcpyDeviceToHost));
            // Store metadata for reconstruction in backward
            mMoEExpertOffsets = src;           // Copy the tensor metadata (shape, dtype, etc.)
            mMoEExpertOffsets.Data = nullptr;  // Data will be restored from CPU storage
        }
    }
}

Tensor* CompiledExecutor::try_resolve_saved_live(const std::string& name, const Tensor& saved) {
    std::vector<long> shape;
    shape.reserve(static_cast<std::size_t>(saved.Rank));
    for (int i = 0; i < saved.Rank; ++i) {
        shape.push_back(saved.Sizes[i]);
    }

    auto map_view = [&](Tensor& base) -> Tensor* {
        if (!base.Data) {
            return nullptr;
        }
        if (shape.empty() || tensor_shape_matches(base, shape)) {
            return &base;
        }
        if (shape_nelem(shape) != base.nelem()) {
            return nullptr;
        }
        Tensor view = view_tensor(base, shape);
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
                mTensors[tid] = view;
                return &mTensors[tid];
            }
        }
        return &base;  // Can't cache view without tensor_id, return base
    };

    // Global / IO slots via the shared helper. `_flat` aliases resolve to the
    // same slot as their 3D counterpart; `map_view` already applies a shape
    // override when the saved shape is 2D, so no explicit flatten here.
    const TensorSlot glob_slot = builtin_slot_from_name(name);
    if (Tensor* t = global_activation_ptr(mRunState, glob_slot)) {
        return map_view(*t);
    }
    if (glob_slot == TensorSlot::FreqCis) {
        return map_view(mRunState.rope_freqs(name));
    }
    if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
        return map_view(mRunState.rope_freqs(name));
    }

    int layer_idx = -1;
    std::string field;
    if (parse_block_param(name, layer_idx, field)) {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
            return nullptr;
        }
        const std::string base_field = strip_ssa_suffix(field);
        const TensorSlot slot = builtin_slot_from_name(base_field);
        if (Tensor* t = block_activation_ptr(mRunState, layer_idx, slot)) {
            return map_view(*t);
        }

        // MoE slots are handled by block_activation_ptr above (via enum
        // dispatch) — no separate MoE branch needed here.

        if (base_field == "rope_freqs" || base_field == "freq_cis") {
            return map_view(mRunState.rope_freqs(name));
        }
    }

    return nullptr;
}

Tensor& CompiledExecutor::resolve_tensor(const TensorRef& ref) {
    auto& rs = mRunState;
    const int tid = ref.tensor_id;
    const bool debug_dtype = []() {
        const char* env = std::getenv("SUROGATE_DEBUG_DTYPE_RUNTIME");
        return env && std::string(env) == "1";
    }();
    const bool debug_name = debug_dtype && (ref.name.find("d_xF") != std::string::npos);
    auto log_tensor = [&](const Tensor& t, const char* tag) {
        if (debug_name) {
            fprintf(stderr,
                    "[DEBUG_DTYPE_RUNTIME] resolve_tensor %s %s dtype=%s\n",
                    ref.name.c_str(),
                    tag,
                    dtype_to_str(t.DType));
        }
    };

    // Phase 4 M3 Phase A: bypass mNamedTensors for Stack-backed rstd
    // slots under SUROGATE_RSTD_ON_STACK. Covers ln1, ln2, and qk-norm
    // rstd slots — all same-layer producer/consumer, FP32, small shape.
    const bool bypass_named_for_rstd = [&]() {
        switch (ref.slot) {
            case TensorSlot::BlockLN1RSTD:
            case TensorSlot::BlockLN2RSTD:
            case TensorSlot::BlockQRSTD:
            case TensorSlot::BlockKRSTD:
            case TensorSlot::BlockLSE:
            case TensorSlot::BlockLN1:
            case TensorSlot::BlockLN2:
            case TensorSlot::BlockAttOut:
            case TensorSlot::BlockHOut: break;
            default: return false;
        }
        static const bool enabled = []() {
            const char* e = std::getenv("SUROGATE_RSTD_ON_STACK");
            return e && std::string(e) == "1";
        }();
        return enabled;
    }();

    // Phase 4 M2: baked-view shortcut for SaveForBwd tids.
    //
    // persist_saved_layer_tensors() writes the arena-backed Tensor into
    // both mNamedTensors[name] (string-keyed) and mTensors[tid] (via
    // bind_tensor). The mNamedTensors path below works — but the hashmap
    // lookup is the first operation a backward op hits when reading a
    // saved activation. For tids classified as SaveForBwd we can go
    // straight to mTensors[tid] and skip the string hash, proving the
    // end-to-end baked operand path works before M3 migrates the full
    // FwdStack/BwdStack activation set.
    //
    // Modes (SUROGATE_BAKED_SAVES):
    //   "off"    (default)  — legacy path, mNamedTensors first
    //   "on"                — baked shortcut, skip mNamedTensors for
    //                         SaveForBwd tids
    //   "verify"            — shadow: take legacy path, but sanity-check
    //                         that mTensors[tid].Data == mNamedTensors[name].Data
    //                         for every SaveForBwd hit; log first
    //                         mismatch. Proves equivalence under
    //                         live-traffic without changing behavior.
    static const int baked_saves_mode = []() {
        const char* e = std::getenv("SUROGATE_BAKED_SAVES");
        if (!e) return 0;
        const std::string v(e);
        if (v == "1" || v == "on") return 1;
        if (v == "verify") return 2;
        return 0;
    }();
    if (baked_saves_mode != 0 && tid >= 0 && mCurrentGraph && static_cast<std::size_t>(tid) < mTensors.size()) {
        const auto& meta = mCurrentGraph->tensor_meta[static_cast<std::size_t>(tid)];
        if (meta.region == dsl::RegionKind::SaveForBwd) {
            Tensor& cached = mTensors[static_cast<std::size_t>(tid)];
            if (cached.Data) {
                if (baked_saves_mode == 2 && !ref.name.empty()) {
                    auto nit = mNamedTensors.find(ref.name);
                    if (nit != mNamedTensors.end() && nit->second.Data && nit->second.Data != cached.Data) {
                        static int mismatch_count = 0;
                        if (++mismatch_count <= 5) {
                            fprintf(stderr,
                                    "[baked-saves verify] MISMATCH %s tid=%d name=%p tid=%p\n",
                                    ref.name.c_str(),
                                    tid,
                                    static_cast<void*>(nit->second.Data),
                                    static_cast<void*>(cached.Data));
                        }
                    }
                } else if (baked_saves_mode == 1) {
                    log_tensor(cached, "baked_save");
                    return cached;
                }
            }
        }
    }

    if (!ref.name.empty() && !bypass_named_for_rstd) {
        auto name_it = mNamedTensors.find(ref.name);
        if (name_it != mNamedTensors.end() && name_it->second.Data) {
            log_tensor(name_it->second, "named");
            return name_it->second;
        }
    }

    // Only resolve gradient tensors via the parameter-grad store when the
    // classifier says kind == ParamGrad. For ActivationGrad / AccumTemp the
    // classifier returns nullopt, correctly skipping this path so the
    // gradient lands in its own per-layer slot / temp.
    if (ref.is_gradient && !ref.name.empty() && mCurrentGraph) {
        if (auto base = base_param_from_grad_kind(ref.tensor_id, *mCurrentGraph)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    Tensor resolved = *grad;
                    if (!ref.shape.empty() && shape_nelem(ref.shape) == static_cast<std::size_t>(grad->nelem())) {
                        resolved = view_tensor(*grad, ref.shape);
                    }
                    if (tid >= 0) {
                        mTensors[static_cast<std::size_t>(tid)] = resolved;
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    return *grad;
                }
            }
        }
    }

    // If shape is specified and this is a pre-allocated slot, we may need to create a view
    if (!ref.shape.empty() && ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Saved &&
        ref.slot != TensorSlot::Parameter && ref.slot != TensorSlot::Temporary) {
        // Need to create a view from the base tensor
        if (ref.name.find("att_flat") != std::string::npos && ref.layer_idx == 4) {
            fprintf(stderr,
                    "[RESOLVE] %s slot=%d layer=%d shape=[%ld,%ld] tid=%d\n",
                    ref.name.c_str(),
                    (int)ref.slot,
                    ref.layer_idx,
                    ref.shape.size() > 0 ? ref.shape[0] : -1,
                    ref.shape.size() > 1 ? ref.shape[1] : -1,
                    tid);
        }
        Tensor* base = nullptr;
        switch (ref.slot) {
            case TensorSlot::TokenIDs: base = &rs.Inputs; break;
            case TensorSlot::PositionIDs: base = &rs.PositionIDs; break;
            case TensorSlot::Targets: base = &rs.Targets; break;
            case TensorSlot::Losses: base = &rs.Losses; break;
            case TensorSlot::DLoss: base = &rs.scratch().cross_entropy_dloss; break;
            case TensorSlot::FreqCis: base = &rs.rope_freqs(ref.name); break;
            case TensorSlot::BlockDLN1: base = &rs.simplified_grads(ref.layer_idx).d_ln1; break;
            case TensorSlot::BlockDQKV: base = &rs.simplified_grads(ref.layer_idx).d_qkv; break;
            case TensorSlot::BlockDAtt: base = &rs.simplified_grads(ref.layer_idx).d_att; break;
            case TensorSlot::BlockDSwiGLU: base = &rs.simplified_grads(ref.layer_idx).d_swiglu; break;
            case TensorSlot::BlockDMLPUp: base = &rs.simplified_grads(ref.layer_idx).d_mlp_up; break;
            case TensorSlot::BlockDMLPDown: base = &rs.simplified_grads(ref.layer_idx).d_mlp_down; break;
            case TensorSlot::BlockDLN2: base = &rs.simplified_grads(ref.layer_idx).d_ln2; break;
            case TensorSlot::BlockDResAtt: base = &rs.simplified_grads(ref.layer_idx).d_res_att; break;
            case TensorSlot::BlockDAttOut: base = &rs.simplified_grads(ref.layer_idx).d_att_out; break;
            case TensorSlot::BlockDResFFN: base = &rs.simplified_grads(ref.layer_idx).d_res_ffn; break;
            case TensorSlot::BlockLN1: base = &rs.simplified_acts(ref.layer_idx).ln1; break;
            case TensorSlot::BlockLN2: base = &rs.simplified_acts(ref.layer_idx).ln2; break;
            case TensorSlot::BlockQKV: base = &rs.simplified_acts(ref.layer_idx).qkv; break;
            case TensorSlot::BlockAtt: base = &rs.simplified_acts(ref.layer_idx).att; break;
            case TensorSlot::BlockAttOut: base = &rs.simplified_acts(ref.layer_idx).att_out; break;
            case TensorSlot::BlockMLPUp: base = &rs.simplified_acts(ref.layer_idx).mlp_up; break;
            case TensorSlot::BlockSwiGLU: base = &rs.simplified_acts(ref.layer_idx).swiglu; break;
            case TensorSlot::BlockMLPDown: base = &rs.simplified_acts(ref.layer_idx).mlp_down; break;
            case TensorSlot::BlockRouterLogits: base = &rs.simplified_acts(ref.layer_idx).router_logits; break;
            case TensorSlot::BlockRouterProbs: base = &rs.simplified_acts(ref.layer_idx).router_probs; break;
            case TensorSlot::BlockRoutingWeights: base = &rs.simplified_acts(ref.layer_idx).routing_weights; break;
            case TensorSlot::BlockRoutingIndices: base = &rs.simplified_acts(ref.layer_idx).routing_indices; break;
            case TensorSlot::BlockPermutedInput: base = &rs.simplified_acts(ref.layer_idx).permuted_input; break;
            case TensorSlot::BlockScatterIndices: base = &rs.simplified_acts(ref.layer_idx).scatter_indices; break;
            case TensorSlot::BlockExpertGateUp: base = &rs.simplified_acts(ref.layer_idx).expert_gate_up; break;
            case TensorSlot::BlockExpertAct: base = &rs.simplified_acts(ref.layer_idx).expert_act; break;
            case TensorSlot::BlockExpertDown: base = &rs.simplified_acts(ref.layer_idx).expert_down; break;
            case TensorSlot::BlockMoeOut: base = &rs.simplified_acts(ref.layer_idx).moe_out; break;
            default: break;
        }
        if (base && base->Data) {
            Tensor view = view_for_shape(*base, ref.shape, ref.name);
            if (tid >= 0) {
                mTensors[static_cast<std::size_t>(tid)] = view;
                return mTensors[static_cast<std::size_t>(tid)];
            }
            return *base;
        }
    }

    // Check flat tensor vector first for cached/aliased tensors (e.g., view_backward aliases).
    // This is critical because view_backward stores aliases, and subsequent ops
    // (like rmsnorm_backward) must use that aliased tensor, not the pre-allocated simplified_grads buffer.
    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
        log_tensor(mTensors[static_cast<std::size_t>(tid)], "cached");
        return mTensors[static_cast<std::size_t>(tid)];
    }

    switch (ref.slot) {
        case TensorSlot::TokenIDs: return rs.Inputs;
        case TensorSlot::PositionIDs: return rs.PositionIDs;
        case TensorSlot::Targets: return rs.Targets;
        case TensorSlot::Losses: return rs.Losses;
        case TensorSlot::DLoss: return rs.scratch().cross_entropy_dloss;
        case TensorSlot::Encoded: return rs.non_block_activations().encoded;
        case TensorSlot::LNFinal: return rs.non_block_activations().ln_final;
        case TensorSlot::LNFinalRSTD: return rs.non_block_activations().ln_final_rstd;
        case TensorSlot::FinalResidual: return rs.get_final_residual();
        case TensorSlot::FreqCis: return rs.rope_freqs(ref.name);
        case TensorSlot::BlockLN1: return rs.simplified_acts(ref.layer_idx).ln1;
        case TensorSlot::BlockLN1RSTD: return rs.simplified_acts(ref.layer_idx).ln1_rstd;
        case TensorSlot::BlockLN2: return rs.simplified_acts(ref.layer_idx).ln2;
        case TensorSlot::BlockLN2RSTD: return rs.simplified_acts(ref.layer_idx).ln2_rstd;
        case TensorSlot::BlockQRSTD: return rs.simplified_acts(ref.layer_idx).q_rstd;
        case TensorSlot::BlockKRSTD: return rs.simplified_acts(ref.layer_idx).k_rstd;
        case TensorSlot::BlockQKV: return rs.simplified_acts(ref.layer_idx).qkv;
        case TensorSlot::BlockQKVRoPE: {
            auto& acts = rs.simplified_acts(ref.layer_idx);
            return acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
        }
        case TensorSlot::BlockLSE: return rs.simplified_acts(ref.layer_idx).lse;
        case TensorSlot::BlockAtt: return rs.simplified_acts(ref.layer_idx).att;
        case TensorSlot::BlockAttOut: return rs.simplified_acts(ref.layer_idx).att_out;
        case TensorSlot::BlockResidualAtt: return rs.simplified_acts(ref.layer_idx).residual_att;
        case TensorSlot::BlockMLPUp: return rs.simplified_acts(ref.layer_idx).mlp_up;
        case TensorSlot::BlockSwiGLU: return rs.simplified_acts(ref.layer_idx).swiglu;
        case TensorSlot::BlockMLPDown: return rs.simplified_acts(ref.layer_idx).mlp_down;
        case TensorSlot::BlockHOut: return rs.simplified_acts(ref.layer_idx).h_out;
        case TensorSlot::BlockResidualFFN: return rs.get_residual(ref.layer_idx, rs.MainStream);
        case TensorSlot::BlockRouterLogits: return rs.simplified_acts(ref.layer_idx).router_logits;
        case TensorSlot::BlockRouterProbs: return rs.simplified_acts(ref.layer_idx).router_probs;
        case TensorSlot::BlockRoutingWeights: return rs.simplified_acts(ref.layer_idx).routing_weights;
        case TensorSlot::BlockRoutingIndices: return rs.simplified_acts(ref.layer_idx).routing_indices;
        case TensorSlot::BlockPermutedInput: return rs.simplified_acts(ref.layer_idx).permuted_input;
        case TensorSlot::BlockScatterIndices: return rs.simplified_acts(ref.layer_idx).scatter_indices;
        case TensorSlot::BlockExpertGateUp: return rs.simplified_acts(ref.layer_idx).expert_gate_up;
        case TensorSlot::BlockExpertAct: return rs.simplified_acts(ref.layer_idx).expert_act;
        case TensorSlot::BlockExpertDown: return rs.simplified_acts(ref.layer_idx).expert_down;
        case TensorSlot::BlockMoeOut: return rs.simplified_acts(ref.layer_idx).moe_out;
        case TensorSlot::BlockDLN1: return rs.simplified_grads(ref.layer_idx).d_ln1;
        case TensorSlot::BlockDQKV: return rs.simplified_grads(ref.layer_idx).d_qkv;
        case TensorSlot::BlockDAtt: return rs.simplified_grads(ref.layer_idx).d_att;
        case TensorSlot::BlockDSwiGLU: return rs.simplified_grads(ref.layer_idx).d_swiglu;
        case TensorSlot::BlockDMLPUp: return rs.simplified_grads(ref.layer_idx).d_mlp_up;
        case TensorSlot::BlockDMLPDown: return rs.simplified_grads(ref.layer_idx).d_mlp_down;
        case TensorSlot::BlockDHOut: return rs.simplified_grads(ref.layer_idx).d_h_out;
        case TensorSlot::BlockDLN2: return rs.simplified_grads(ref.layer_idx).d_ln2;
        case TensorSlot::BlockDResAtt: return rs.simplified_grads(ref.layer_idx).d_res_att;
        case TensorSlot::BlockDAttOut: return rs.simplified_grads(ref.layer_idx).d_att_out;
        case TensorSlot::BlockDResFFN: return rs.simplified_grads(ref.layer_idx).d_res_ffn;
        case TensorSlot::Parameter: return mWeights.get(ref.name);
        case TensorSlot::Saved:
            if (mSaved) {
                auto it = mSaved->find(ref.name);
                if (it != mSaved->end()) {
                    // If the saved tensor has actual data, use it directly.
                    // Only resolve from live buffers when Data == nullptr (metadata-only mode).
                    if (it->second.Data != nullptr) {
                        return it->second;
                    }
                    // Metadata-only: resolve from current live tensors first.
                    if (Tensor* live = try_resolve_saved_live(ref.name, it->second)) {
                        return *live;
                    }
                    // As a last resort, reuse cached tensor-id only if it points to
                    // persistent (non-stack) memory; stack pointers can become stale
                    // across layer checkpoint restores under recompute/capture.
                    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data &&
                        !mRunState.Stack.owns(mTensors[static_cast<std::size_t>(tid)].Data)) {
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    return it->second;
                }
            }
            throw std::runtime_error("CompiledExecutor: saved tensor not found: " + ref.name);
        case TensorSlot::Mapped: {
            // Already checked mTensors[tid] above; if we get here, tensor was not found
            if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
                return mTensors[static_cast<std::size_t>(tid)];
            }
            // Materialize effective tensors on demand (e.g. *_weight_eff = *_weight + 1).
            if (!ref.name.empty() && ref.name.size() > 4 && ref.name.compare(ref.name.size() - 4, 4, "_eff") == 0) {
                const std::string base_name = ref.name.substr(0, ref.name.size() - 4);
                Tensor* base_ptr = nullptr;
                if (mWeights.has(base_name)) {
                    base_ptr = &mWeights.get(base_name);
                } else if (mSaved) {
                    auto it = mSaved->find(base_name);
                    if (it != mSaved->end() && it->second.Data) {
                        base_ptr = &it->second;
                    }
                }
                if (!base_ptr && mCurrentGraph) {
                    const int base_tid = mCurrentGraph->find_tensor_id(base_name);
                    if (base_tid >= 0 && static_cast<std::size_t>(base_tid) < mTensors.size() &&
                        mTensors[static_cast<std::size_t>(base_tid)].Data) {
                        base_ptr = &mTensors[static_cast<std::size_t>(base_tid)];
                    }
                }
                if (base_ptr && base_ptr->Data) {
                    std::vector<long> shape(base_ptr->Sizes.begin(), base_ptr->Sizes.begin() + base_ptr->Rank);
                    Tensor ones = mRunState.temp_alloc(base_ptr->DType, shape, "ones");
                    Tensor eff = mRunState.temp_alloc(base_ptr->DType, shape, "eff");
                    mTemps.push_back(ones);
                    mTemps.push_back(eff);
                    fill_constant(ones, 1.0f, static_cast<std::size_t>(ones.nelem()), mRunState.MainStream);
                    vector_add_sr(eff,
                                  *base_ptr,
                                  ones,
                                  1.0f,
                                  static_cast<long>(base_ptr->nelem()),
                                  0,
                                  mRunState.MainStream);
                    if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
                        mTensors[static_cast<std::size_t>(tid)] = eff;
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    throw std::runtime_error("CompiledExecutor: mapped effective tensor requires valid tensor_id: " +
                                             ref.name);
                }
            }
            throw std::runtime_error("CompiledExecutor: tensor not found: " + ref.name);
        }
        case TensorSlot::Temporary: throw std::runtime_error("CompiledExecutor: temporary slot requires allocation");
    }
    // P4.7 error rewriter: include tid context when available, so users see
    // "invalid tensor slot for tid=12 name='blocks[3].att' region=FwdStack"
    // instead of the bare "invalid tensor slot".
    std::ostringstream oss;
    oss << "CompiledExecutor: invalid tensor slot";
    if (mCurrentGraph && ref.tensor_id >= 0) {
        oss << " (" << mCurrentGraph->describe_tensor_id(ref.tensor_id) << ")";
    } else if (!ref.name.empty()) {
        oss << " (name='" << ref.name << "')";
    }
    throw std::runtime_error(oss.str());
}

Tensor& CompiledExecutor::ensure_output_tensor(const TensorRef& ref) {
    const int tid = ref.tensor_id;
    // `normalized_name` exists for historical compatibility with the old
    // accumulator-alias path (strip `_from_N` / `_accum_N`, then look up the
    // base name's buffer). The classifier now tags every such tensor as
    // AccumTemp at compile time, and we ALWAYS refuse to collapse an accum
    // temp onto its base — they're live simultaneously by construction, so
    // aliasing corrupts gradients. Non-accum tensors never have the suffix,
    // so there's nothing left to strip: normalized_name == ref.name always.
    const std::string& normalized_name = ref.name;
    const bool debug_mlp_ensure = []() {
        const char* env = std::getenv("SUROGATE_DEBUG_ENSURE_OUT");
        return env && std::string(env) != "0";
    }();
    auto log_mlp_ensure = [&](const char* path, const Tensor* t = nullptr) {
        if (!debug_mlp_ensure || ref.name.find("mlp_x_flat") == std::string::npos) {
            return;
        }
        std::fprintf(stderr,
                     "[ENSURE_OUT] name=%s normalized=%s path=%s tid=%d slot=%d ptr=%p shape_empty=%d\n",
                     ref.name.c_str(),
                     normalized_name.c_str(),
                     path,
                     tid,
                     static_cast<int>(ref.slot),
                     t ? static_cast<void*>(t->Data) : nullptr,
                     ref.shape.empty() ? 1 : 0);
    };
    const bool debug_dtype = []() {
        const char* env = std::getenv("SUROGATE_DEBUG_DTYPE_RUNTIME");
        return env && std::string(env) == "1";
    }();
    const bool debug_name = debug_dtype && (ref.name.find("d_xF") != std::string::npos);
    if (debug_name) {
        fprintf(stderr,
                "[DEBUG_DTYPE_RUNTIME] ensure_output_tensor enter %s slot=%d ref=%s\n",
                ref.name.c_str(),
                static_cast<int>(ref.slot),
                dtype_to_str(ref.dtype));
    }

    // Fast path: pre-allocated block slots with existing data bypass string parsing.
    // This covers most activation and gradient outputs during forward/backward.
    // Only Mapped/Temporary/Parameter/Saved slots need the string-heavy resolution below.
    if (ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Temporary && ref.slot != TensorSlot::Parameter &&
        ref.slot != TensorSlot::Saved) {
        Tensor& t = resolve_tensor(ref);
        if (t.Data) {
            Tensor resolved = t;
            if (!ref.shape.empty()) {
                resolved = view_for_shape(t, ref.shape, ref.name);
            }
            if (tid >= 0) {
                mTensors[static_cast<std::size_t>(tid)] = resolved;
                return mTensors[static_cast<std::size_t>(tid)];
            }
            return t;
        }
        // No data — fall through to slow path for alias resolution or temp allocation
    }

    // `ref.is_gradient` is set at compile time on every tensor whose name starts
    // with `d_` — consult it directly instead of re-matching the prefix here.
    if (!normalized_name.empty() && ref.is_gradient) {
        if (const Tensor* base = try_get_tensor(normalized_name)) {
            // try_get_tensor already filters by `base->Data != nullptr`, so
            // we know this path has real storage. Still, guard defensively.
            if (base->Data) {
                Tensor resolved = *base;
                if (!ref.shape.empty() && shape_nelem(ref.shape) == static_cast<std::size_t>(base->nelem())) {
                    resolved = view_tensor(*base, ref.shape);
                }
                if (tid >= 0) {
                    mTensors[static_cast<std::size_t>(tid)] = resolved;
                    log_mlp_ensure("grad_exact", &mTensors[static_cast<std::size_t>(tid)]);
                    return mTensors[static_cast<std::size_t>(tid)];
                }
                throw std::runtime_error("CompiledExecutor: ensure_output_tensor requires valid tensor_id for: " +
                                         ref.name);
            }
        }
        if (const Tensor* base = try_get_tensor_fuzzy(normalized_name)) {
            // try_get_tensor_fuzzy can legitimately return a pointer to a Tensor
            // with `Data == nullptr` — e.g., a simplified_grads slot that was
            // allocated lazily / configured for recompute and isn't populated
            // yet. Running an op against such a tensor silently corrupts
            // memory (illegal access at kernel dispatch). Only accept this
            // resolution when the buffer is real; otherwise fall through to
            // fresh allocation.
            if (base->Data) {
                Tensor resolved = *base;
                if (!ref.shape.empty() && shape_nelem(ref.shape) == static_cast<std::size_t>(base->nelem())) {
                    resolved = view_tensor(*base, ref.shape);
                }
                if (tid >= 0) {
                    mTensors[static_cast<std::size_t>(tid)] = resolved;
                    log_mlp_ensure("grad_fuzzy", &mTensors[static_cast<std::size_t>(tid)]);
                    return mTensors[static_cast<std::size_t>(tid)];
                }
                throw std::runtime_error("CompiledExecutor: ensure_output_tensor requires valid tensor_id for: " +
                                         ref.name);
            }
        }
    }

    if (!normalized_name.empty() && mCurrentGraph) {
        // Classifier-backed param-grad routing. Falls through for
        // ActivationGrad / AccumTemp / Scratch — those land in their own
        // storage via the Mapped path below.
        if (auto base = base_param_from_grad_kind(tid, *mCurrentGraph)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    Tensor resolved = *grad;
                    if (!ref.shape.empty() && shape_nelem(ref.shape) == static_cast<std::size_t>(grad->nelem())) {
                        resolved = view_tensor(*grad, ref.shape);
                    }
                    if (tid >= 0) {
                        mTensors[static_cast<std::size_t>(tid)] = resolved;
                        log_mlp_ensure("param_grad", &mTensors[static_cast<std::size_t>(tid)]);
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    return *grad;
                }
            }
        }
    }

    // DSL-driven aliasing: allow gradients to reuse existing activation buffers.
    if (mSlotRegistry && mSlotRegistry->has_dsl_layout() && !ref.name.empty()) {
        std::string base_name = strip_ssa_suffix(normalized_name);
        const bool is_grad_name = starts_with(base_name, "d_");
        std::string parse_name = is_grad_name ? base_name.substr(2) : base_name;
        int layer_idx = -1;
        std::string field;
        std::string lookup_name = base_name;
        if (parse_block_param(parse_name, layer_idx, field)) {
            const std::string base_field = strip_ssa_suffix(field);
            lookup_name = is_grad_name ? ("d_" + base_field) : base_field;
        }
        if (auto slot_entry = mSlotRegistry->lookup(lookup_name)) {
            if (!slot_entry->alias_of.empty()) {
                const std::string alias_field = slot_entry->alias_of;
                std::string alias_name = alias_field;
                if (layer_idx >= 0) {
                    // Reconstruct using the same naming convention as the source tensor
                    if (parse_name.rfind("layer", 0) == 0) {
                        alias_name = "layer" + std::to_string(layer_idx) + "." + alias_field;
                    } else {
                        alias_name = "blocks[" + std::to_string(layer_idx) + "]." + alias_field;
                    }
                }
                if (auto alias_entry = mSlotRegistry->lookup(alias_field)) {
                    TensorRef alias_ref;
                    alias_ref.name = alias_name;
                    alias_ref.layer_idx = layer_idx;
                    alias_ref.slot = alias_entry->slot;
                    alias_ref.shape = ref.shape;
                    alias_ref.dtype = ref.dtype;
                    // Resolve alias tensor_id from current graph
                    if (mCurrentGraph) {
                        alias_ref.tensor_id = mCurrentGraph->find_tensor_id(alias_name);
                    }
                    if (mSaveSet.find(alias_name) != mSaveSet.end()) {
                        alias_ref.slot = TensorSlot::Saved;
                    }
                    try {
                        Tensor& base = resolve_tensor(alias_ref);
                        Tensor view = ref.shape.empty() ? base : view_tensor(base, ref.shape);
                        if (tid >= 0) {
                            mTensors[static_cast<std::size_t>(tid)] = view;
                            log_mlp_ensure("slot_alias", &mTensors[static_cast<std::size_t>(tid)]);
                            return mTensors[static_cast<std::size_t>(tid)];
                        }
                        return base;
                    } catch (const std::exception&) {
                        // Fall through to normal allocation if alias resolution fails.
                    }
                }
            }
        }
    }

    // For pre-allocated slots, just return the tensor
    if (ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Temporary) {
        Tensor& t = resolve_tensor(ref);
        if (!t.Data) {
            mRunState.temp_acquire(t);
            mTemps.push_back(t);
        }
        if (!ref.shape.empty()) {
            Tensor view = view_tensor(t, ref.shape);
            if (tid >= 0) {
                mTensors[static_cast<std::size_t>(tid)] = view;
                log_mlp_ensure("prealloc_view", &mTensors[static_cast<std::size_t>(tid)]);
                return mTensors[static_cast<std::size_t>(tid)];
            }
            return t;  // tid < 0 should not happen for pre-allocated slots
        }
        log_mlp_ensure("prealloc_base", &t);
        return t;
    }

    // For mapped/temporary tensors, check flat vector first
    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
        if (debug_name) {
            const auto& t = mTensors[static_cast<std::size_t>(tid)];
            fprintf(stderr, "[DEBUG_DTYPE_RUNTIME] reuse tid=%d dtype=%s\n", tid, dtype_to_str(t.DType));
        }
        log_mlp_ensure("mapped_cached", &mTensors[static_cast<std::size_t>(tid)]);
        return mTensors[static_cast<std::size_t>(tid)];
    }

    Tensor t = mRunState.temp_alloc(ref.dtype, ref.shape, "t");

    // Zero gradient tensors to prevent stale values from accumulating.
    if (ref.is_gradient) {
        fill_zero(t, mRunState.MainStream);
    }

    mTemps.push_back(t);
    if (tid >= 0) {
        mTensors[static_cast<std::size_t>(tid)] = t;
        if (debug_name) {
            fprintf(stderr, "[DEBUG_DTYPE_RUNTIME] alloc tid=%d dtype=%s\n", tid, dtype_to_str(t.DType));
        }
        log_mlp_ensure("mapped_alloc", &mTensors[static_cast<std::size_t>(tid)]);
        return mTensors[static_cast<std::size_t>(tid)];
    }
    throw std::runtime_error("CompiledExecutor: ensure_output_tensor requires valid tensor_id for: " + ref.name);
}

}  // namespace dsl
