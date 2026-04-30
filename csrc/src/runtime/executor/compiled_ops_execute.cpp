// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CompiledExecutor replay and shared execution helpers.
// Forward/backward execution live in compiled_ops_execute_forward.cpp and
// compiled_ops_execute_backward.cpp.

#include "runtime/executor/compiled_ops.h"
#include "runtime/dsl/tensor_slot_dispatch.h"

#include "runtime/ep/ep_strategy.h"

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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/core.h>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/tensor_role.h"
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

std::optional<bool> rope_role_from_ref(const CompiledGraph* graph, const TensorRef& ref) {
    if (!graph) return std::nullopt;
    if (const TensorRole* role = graph->role_for_tensor_id(ref.tensor_id)) {
        return role->is_rope_freq();
    }
    if (!ref.name.empty()) {
        if (const TensorRole* role = graph->role_for_name(ref.name)) {
            return role->is_rope_freq();
        }
    }
    return std::nullopt;
}

bool is_rope_freq_ref(const CompiledGraph* graph, const TensorRef& ref, const char* context) {
    (void)context;
    return rope_role_from_ref(graph, ref).value_or(tensor_role_is_rope_name(ref.name));
}

}  // namespace

// ---------------------------------------------------------------------------
// replay_layer_forward — torch-style gradient checkpointing
//
// Re-execute a single layer's compiled forward ops during backward to
// regenerate activations. The data lives on the stack; the caller (backward)
// must restore the stack checkpoint after consuming the data.
// ---------------------------------------------------------------------------

void CompiledExecutor::replay_layer_forward(int layer_idx,
                                            long B,
                                            long T,
                                            const CompiledGraph& fwd_graph,
                                            const modules::ForwardHook* hook) {
    static const bool debug_replay = std::getenv("SUROGATE_DEBUG_REPLAY") != nullptr;
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
    if (debug_replay) {
        fprintf(stderr, "[REPLAY] replay_layer_forward layer=%d B=%ld T=%ld\n", layer_idx, B, T);
    }
    // Restore any previous deferred checkpoint before starting a new replay
    if (mHasDeferredReplayCheckpoint) {
        mRunState.Stack.restore(mDeferredReplayCheckpoint);
        if (mTemps.size() > mDeferredReplayTempMark) {
            mTemps.resize(mDeferredReplayTempMark);
        }
        mHasDeferredReplayCheckpoint = false;
    }
    // Previous replay copies are no longer needed once we start recomputing the
    // next lower layer, but any tables still pointing at them must be scrubbed
    // before freeing or they become dangling saved tensors.
    clear_replay_copied_refs();
    // Free persistent copies from previous replay (backward has consumed them)
    for (void* ptr : mReplayCopiedBuffers) {
        cudaFreeAsync(ptr, mRunState.MainStream);
    }
    mReplayCopiedBuffers.clear();

    // Save current execution state
    const CompiledGraph* saved_graph = mCurrentGraph;
    std::vector<Tensor> saved_tensors;
    std::unordered_map<std::string, Tensor> saved_named_tensors;
    saved_tensors.swap(mTensors);
    saved_named_tensors.swap(mNamedTensors);

    // Set replay mode
    mInReplay = true;
    mReplayLayerIdx = layer_idx;

    // Initialize fresh tensor storage for the forward graph
    mCurrentGraph = &fwd_graph;
    mTensors.assign(static_cast<std::size_t>(fwd_graph.num_tensors), Tensor{});
    mNamedTensors.clear();
    // Mirror the pre-bind that execute_forward does so replayed ops hit
    // the mTensors[tid] cache and write to the arena.
    populate_fwd_stack_bindings(fwd_graph);

    bind_runtime_bindings();

    // Take stack checkpoint — backward will restore this after consuming replay data
    auto replay_checkpoint = mRunState.Stack.checkpoint();
    auto replay_temp_mark = mTemps.size();

    // Find the op range for this layer
    if (layer_idx < 0 || static_cast<std::size_t>(layer_idx) >= fwd_graph.layer_start_indices.size() ||
        fwd_graph.layer_start_indices[static_cast<std::size_t>(layer_idx)] == SIZE_MAX) {
        // Layer not found in forward graph — restore state and return
        mTensors.swap(saved_tensors);
        mNamedTensors.swap(saved_named_tensors);
        mCurrentGraph = saved_graph;
        mInReplay = false;
        mReplayLayerIdx = -1;
        return;
    }

    const std::size_t start = fwd_graph.layer_start_indices[static_cast<std::size_t>(layer_idx)];
    const std::size_t end = (static_cast<std::size_t>(layer_idx) < fwd_graph.layer_end_indices.size())
                                ? fwd_graph.layer_end_indices[static_cast<std::size_t>(layer_idx)]
                                : fwd_graph.ops.size();

    // Collect tensor IDs produced within this layer's op range
    std::unordered_set<int> produced_ids;
    for (std::size_t idx = start; idx < end && idx < fwd_graph.ops.size(); ++idx) {
        for (const auto& out : fwd_graph.ops[idx].outputs) {
            if (out.tensor_id >= 0) {
                produced_ids.insert(out.tensor_id);
            }
        }
    }

    // Pre-bind external inputs: tensors consumed by this layer but produced before it.
    // These include the layer's input residual, previous block outputs, RoPE freqs, etc.
    for (std::size_t idx = start; idx < end && idx < fwd_graph.ops.size(); ++idx) {
        for (const auto& inp : fwd_graph.ops[idx].inputs) {
            if (inp.tensor_id < 0) continue;
            if (produced_ids.count(inp.tensor_id)) continue;
            // Already bound?
            if (static_cast<std::size_t>(inp.tensor_id) < mTensors.size() && mTensors[inp.tensor_id].Data) continue;

            // Try to resolve from known sources. The two routing paths
            // (slot-by-enum on the compiled TensorRef, and name-based
            // fallbacks for refs whose slot is Mapped) both go through
            // the shared `global_activation_ptr` / `block_activation_ptr`
            // helpers so new slots are picked up automatically.
            Tensor resolved{};

            // (1) TensorRef already carries the slot — use it directly.
            if (Tensor* gp = global_activation_ptr(mRunState, inp.slot)) {
                resolved = *gp;
            } else if (inp.slot == TensorSlot::FreqCis) {
                resolved = mRunState.rope_freqs(inp.name);
            }

            // (2) Mapped refs: parse name → slot and reuse the block/global helpers.
            if (!resolved.Data && !inp.name.empty()) {
                // Blocks-qualified: `blocks[N].<field>`
                int lyr = -1;
                const TensorSlot slot = resolve_block_slot(inp.name, &lyr);
                if (lyr >= 0) {
                    if (Tensor* bp = block_activation_ptr(mRunState, lyr, slot)) {
                        resolved = *bp;
                    }
                } else if (is_rope_freq_ref(mCurrentGraph, inp, "compiled_ops_execute::resolve_input")) {
                    // Rope frequencies come through unqualified global names;
                    // inp.slot would be FreqCis and line 217 already handled
                    // compile-classified refs. This fallback covers names the
                    // compiler left as Mapped (e.g., qualified substring hits).
                    resolved = mRunState.rope_freqs(inp.name);
                }

                // Cross-layer connector tensors: `layerN.<field>` (used by
                // HybridStackedBlocks to avoid the block-activation resolver).
                // Resolve via the block helper — same field vocabulary,
                // different name prefix.
                if (!resolved.Data && inp.name.rfind("layer", 0) == 0) {
                    auto dot = inp.name.find('.');
                    if (dot != std::string::npos) {
                        try {
                            int cross_lyr = std::stoi(inp.name.substr(5, dot - 5));
                            std::string cross_field = strip_ssa_suffix(inp.name.substr(dot + 1));
                            TensorSlot slot = builtin_slot_from_name(cross_field);
                            // `out` is a connector-specific alias for the block's
                            // final output (mlp_down slot in legacy models).
                            if (slot == TensorSlot::Mapped && (cross_field == "out" || cross_field == "out_flat")) {
                                slot = TensorSlot::BlockMLPDown;
                            }
                            if (Tensor* bp = block_activation_ptr(mRunState, cross_lyr, slot)) {
                                resolved = *bp;
                            }
                        } catch (...) {
                        }
                    }
                }
            }

            // For layer 0 input: the "zeros" residual — allocate a fresh
            // zero tensor on the stack. This is a Gemma4 / hybrid-stack
            // idiom: the first layer's residual is initialized from an
            // explicit zeros op whose name contains "zeros".
            if (!resolved.Data && !inp.name.empty() && inp.name.find("zeros") != std::string::npos) {
                long C = static_cast<long>(mConfig.HiddenSize);
                resolved = mRunState.temp_alloc(ETensorDType::BF16, {mB, mT, C}, "zeros");
                fill_zero(resolved, mRunState.MainStream);
            }

            // Embedding output (embed_1, embed_0, ...) routes to the Encoded
            // slot buffer via TensorRole while the slot registry catches up.
            if (!resolved.Data && !inp.name.empty() && tensor_role_is_embedding_name(inp.name)) {
                if (Tensor* gp = global_activation_ptr(mRunState, TensorSlot::Encoded)) {
                    resolved = *gp;
                }
            }

            // Last resort: check mSaved (forward saved tensors)
            if (!resolved.Data && mSaved && !inp.name.empty()) {
                auto it = mSaved->find(inp.name);
                if (it != mSaved->end() && it->second.Data) {
                    resolved = it->second;
                }
            }

            // Very last resort: check backward graph's named tensors
            if (!resolved.Data && !inp.name.empty()) {
                auto it = saved_named_tensors.find(inp.name);
                if (it != saved_named_tensors.end() && it->second.Data) {
                    resolved = it->second;
                }
            }

            // Tid-based fallback: inputs from model-scope ops (e.g., the
            // Gemma4 PLI `scale_N` → `narrow_pli_L` chain) are often
            // compiled with empty ref.name (TensorRef optimizes named-slot
            // strings away when the slot resolves by tid alone) AND aren't
            // in saved_named_tensors. They DO live in saved_tensors[tid]
            // after the swap, as long as prune_stack_tensors didn't evict
            // them during forward — which the `cross_layer_global` flag
            // prevents via mSaveMask. Without this branch, the chain above
            // misses them and throws "tensor not found".
            if (!resolved.Data && inp.tensor_id >= 0 &&
                static_cast<std::size_t>(inp.tensor_id) < saved_tensors.size()) {
                const Tensor& t = saved_tensors[static_cast<std::size_t>(inp.tensor_id)];
                if (t.Data) {
                    resolved = t;
                }
            }

            if (resolved.Data) {
                store_tensor(inp, resolved);
                if (!inp.name.empty()) {
                    mNamedTensors[inp.name] = resolved;
                }
            }
        }
    }

    // Replay the layer's forward ops.
    // Tile MLP during replay to avoid allocating full-size intermediates (swiglu_flat,
    // mlp_up). The tiled backward will recompute these per-chunk during backward execution.
    std::unordered_map<std::size_t, const MlpTileGroup*> replay_tile_groups;
    for (const auto& tg : fwd_graph.mlp_tile_groups) {
        if (tg.start_op_idx >= start && tg.end_op_idx <= end) {
            replay_tile_groups[tg.start_op_idx] = &tg;
        }
    }
    for (std::size_t idx = start; idx < end && idx < fwd_graph.ops.size(); ++idx) {
        const auto& op = fwd_graph.ops[idx];

        // Scope assertion: replay_layer_forward(L) must only execute ops
        // belonging to layer L. An op whose `layer_start` field is set to a
        // different layer's index means the loop bounds disagree with the
        // compiler's layer-membership tagging — historically this happened
        // when `idx <= end` (inclusive) pulled layer L+1's deferred-residual
        // fused_residual_rmsnorm into L's replay, clobbering L's activations
        // under shared-FwdStack-arena routing. Gated on SUROGATE_CHECK_REPLAY_SCOPE
        // to stay out of the hot path by default.
        if (op.layer_start >= 0 && op.layer_start != layer_idx) {
            if (const char* e = std::getenv("SUROGATE_CHECK_REPLAY_SCOPE"); e && (*e == '1' || *e == 'a')) {
                std::fprintf(stderr,
                             "[replay-scope] layer=%d replay running op idx=%zu with layer_start=%d (type=%s)\n",
                             layer_idx,
                             idx,
                             op.layer_start,
                             op_type_to_string(op.type));
                if (*e == 'a') {
                    throw std::runtime_error("SUROGATE_CHECK_REPLAY_SCOPE=abort: replay-scope violation");
                }
            }
        }

        // Skip loss ops — these should never be replayed
        if (op.type == CompiledOpType::CrossEntropyLoss || op.type == CompiledOpType::FusedLMHeadLoss) {
            continue;
        }

        // Tile MLP during replay — produces correct output without full intermediates.
        // The backward tiled execution will recompute intermediates per-chunk.
        if (!replay_tile_groups.empty()) {
            auto tg_it = replay_tile_groups.find(idx);
            if (tg_it != replay_tile_groups.end()) {
                execute_tiled_mlp(fwd_graph, *tg_it->second, B, T, hook);
                idx = tg_it->second->end_op_idx;
                continue;
            }
        }

        try {
            // Dispatch via the function pointer baked into op.fn at graph
            // compile time. Null fn means "no handler for this op in the
            // forward direction" — silently skip, preserving the old
            // replay_layer_forward `default: break` semantics.
            if (op.fn) {
                check_op_io_aliasing(op, idx, "replay");
                op.fn(*this, op, static_cast<const void*>(hook));
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "replay_layer_forward layer=" << layer_idx << " op=" << (idx - start)
                << " (type=" << op_type_to_string(op.type) << "): " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    // Persist replayed tensors into mSaved — save live pointers first, then
    // selectively copy any entries whose backing storage is not stable enough
    // to survive the replay checkpoint restore / subsequent layer execution.
    if (mSaved && mSaveList) {
        auto replay_preserve_existing_saved = [&](const std::string& tensor_name) -> bool {
            if (!is_qwen3_5_model) {
                return false;
            }
            int saved_layer = -1;
            std::string saved_field;
            if (!parse_block_param(tensor_name, saved_layer, saved_field) || saved_layer != layer_idx) {
                return false;
            }
            const TensorSlot slot = builtin_slot_from_name(strip_ssa_suffix(saved_field));
            return slot == TensorSlot::BlockLN1 || slot == TensorSlot::BlockLN1RSTD || slot == TensorSlot::BlockLN2 ||
                   slot == TensorSlot::BlockLN2RSTD;
        };
        for (const auto& name : *mSaveList) {
            {
                int lyr_check = -1;
                std::string fld_check;
                if (!parse_block_param(name, lyr_check, fld_check) || lyr_check != layer_idx) continue;
            }

            if (replay_preserve_existing_saved(name)) {
                auto existing_it = mSaved->find(name);
                if (existing_it != mSaved->end() && existing_it->second.Data) {
                    continue;
                }
            }

            // Try to find the tensor from the replayed forward graph
            int tid = fwd_graph.find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                (*mSaved)[name] = mTensors[tid];
                continue;
            }
            // Try SSA-stripped lookup
            auto ssa_it = fwd_graph.ssa_base_to_id.find(name);
            if (ssa_it != fwd_graph.ssa_base_to_id.end()) {
                int sid = ssa_it->second;
                if (sid >= 0 && static_cast<std::size_t>(sid) < mTensors.size() && mTensors[sid].Data) {
                    (*mSaved)[name] = mTensors[sid];
                    continue;
                }
            }
            // Fallback: resolve via the shared block-activation dispatch
            // (slot-keyed lookup through tensor_slot_dispatch).
            int lyr = -1;
            const TensorSlot slot = resolve_block_slot(name, &lyr);
            if (lyr >= 0) {
                if (Tensor* bp = block_activation_ptr(mRunState, lyr, slot)) {
                    if (bp->Data) {
                        (*mSaved)[name] = *bp;
                    }
                }
            }
        }
    }

    // Restore tensor storage — backward graph uses its own namespace
    mTensors.swap(saved_tensors);
    mNamedTensors.swap(saved_named_tensors);
    mCurrentGraph = saved_graph;
    mInReplay = false;
    mReplayLayerIdx = -1;

    // Eagerly restore the stack checkpoint by first copying any stack-resident
    // saved tensors to persistent GPU memory. This frees the replay's stack
    // allocation before backward ops start allocating their temps, preventing
    // stack OOM on memory-hungry backward ops (e.g., gated delta rule).
    if (mSaved) {
        auto replay_saved_slot_requires_persist = [&](const std::string& tensor_name) -> bool {
            if (!mSlotRegistry || tensor_name.empty()) {
                return false;
            }

            auto requires_persist = [&](const std::string& lookup_name) -> bool {
                if (auto entry = mSlotRegistry->lookup(lookup_name)) {
                    return entry->slot == TensorSlot::Mapped || entry->memory_hint == ActivationMemoryHint::Shared;
                }
                return false;
            };

            int saved_layer_idx = -1;
            std::string saved_field;
            if (parse_block_param(tensor_name, saved_layer_idx, saved_field)) {
                return requires_persist(strip_ssa_suffix(saved_field));
            }
            return requires_persist(strip_ssa_suffix(tensor_name));
        };

        auto replay_temp_backed = [&](const Tensor& tensor) -> bool {
            if (!tensor.Data || tensor.bytes() == 0) {
                return false;
            }
            const std::byte* ptr = tensor.Data;
            for (std::size_t i = replay_temp_mark; i < mTemps.size(); ++i) {
                const Tensor& tmp = mTemps[i];
                if (!tmp.Data) {
                    continue;
                }
                const std::size_t tmp_bytes = tmp.bytes();
                if (tmp_bytes == 0) {
                    continue;
                }
                const std::byte* begin = tmp.Data;
                const std::byte* end = begin + tmp_bytes;
                if (ptr >= begin && ptr < end) {
                    return true;
                }
            }
            return false;
        };

        for (auto& [name, tensor] : *mSaved) {
            if (!tensor.Data) continue;
            const bool stack_backed = mRunState.Stack.owns(tensor.Data);
            const bool temp_backed = replay_temp_backed(tensor);
            const bool slot_requires_persist = replay_saved_slot_requires_persist(name);
            if (!stack_backed && !temp_backed && !slot_requires_persist) {
                continue;
            }
            // Tensor would be invalidated by replay checkpoint restore / temp rollback,
            // or it lives in a shared/mapped activation slot that replay can overwrite
            // before backward consumes the saved value.
            const std::size_t bytes = tensor.bytes();
            if (bytes == 0) continue;
            // Prefer the replay-persist arena (stable pointer across CUDA graph
            // captures/replays). Fall back to cudaMallocAsync for over-size
            // requests — arena-exhausted fallback keeps the legacy semantics
            // on the rare path where a single tensor exceeds the arena.
            std::byte* persistent = allocate_replay_persist(bytes);
            if (!persistent) {
                void* raw = nullptr;
                CUDA_CHECK(cudaMallocAsync(&raw, bytes, mRunState.MainStream));
                persistent = static_cast<std::byte*>(raw);
                mReplayCopiedBuffers.push_back(persistent);
            }
            CUDA_CHECK(cudaMemcpyAsync(persistent, tensor.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
            tensor.Data = persistent;
        }

        // rstd / ln / attn_out / h_out slots live on the Stack. The
        // save-capture loop above copies live .Data into mSaved; without
        // the persist below, that .Data is the Stack pointer which
        // Stack.restore invalidates. Copy into a persistent cudaMalloc
        // buffer and rebind BOTH the slot AND any mSaved / mNamedTensors /
        // mTensors entries that captured the old Stack ptr.
        if (layer_idx >= 0) {
            auto persist_stack_slot = [&](Tensor& slot, const std::string& slot_name) {
                if (!slot.Data || slot.bytes() == 0) return;
                if (!mRunState.Stack.owns(slot.Data)) return;
                const std::size_t bytes = slot.bytes();
                std::byte* stack_ptr = slot.Data;
                std::byte* persistent_bytes_ptr = allocate_replay_persist(bytes);
                if (!persistent_bytes_ptr) {
                    void* raw = nullptr;
                    CUDA_CHECK(cudaMallocAsync(&raw, bytes, mRunState.MainStream));
                    persistent_bytes_ptr = static_cast<std::byte*>(raw);
                    mReplayCopiedBuffers.push_back(persistent_bytes_ptr);
                }
                CUDA_CHECK(cudaMemcpyAsync(persistent_bytes_ptr,
                                           stack_ptr,
                                           bytes,
                                           cudaMemcpyDeviceToDevice,
                                           mRunState.MainStream));
                slot.Data = persistent_bytes_ptr;

                if (mSaved) {
                    auto it = mSaved->find(slot_name);
                    if (it != mSaved->end() && it->second.Data == stack_ptr) {
                        it->second.Data = persistent_bytes_ptr;
                    }
                }
                if (auto nit = mNamedTensors.find(slot_name); nit != mNamedTensors.end()) {
                    if (nit->second.Data == stack_ptr) nit->second.Data = persistent_bytes_ptr;
                }
                if (mCurrentGraph) {
                    const int tid = mCurrentGraph->find_tensor_id(slot_name);
                    if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() &&
                        mTensors[static_cast<std::size_t>(tid)].Data == stack_ptr) {
                        mTensors[static_cast<std::size_t>(tid)].Data = persistent_bytes_ptr;
                    }
                }
            };
            const std::string prefix = "blocks[" + std::to_string(layer_idx) + "].";
            auto persist_by_slot = [&](TensorSlot s, const char* field_name) {
                if (Tensor* t = block_activation_ptr(mRunState, layer_idx, s)) {
                    persist_stack_slot(*t, prefix + field_name);
                }
            };
            persist_by_slot(TensorSlot::BlockLN1RSTD, "ln1_rstd");
            persist_by_slot(TensorSlot::BlockLN2RSTD, "ln2_rstd");
            persist_by_slot(TensorSlot::BlockQRSTD, "q_rstd");
            persist_by_slot(TensorSlot::BlockKRSTD, "k_rstd");
            persist_by_slot(TensorSlot::BlockLSE, "lse");
            persist_by_slot(TensorSlot::BlockAttOut, "att_out");
            persist_by_slot(TensorSlot::BlockLN1, "ln1");
            persist_by_slot(TensorSlot::BlockLN2, "ln2");
            persist_by_slot(TensorSlot::BlockHOut, "h_out");
        }
    }
    // Now safe to restore — stack-resident data has been copied
    mRunState.Stack.restore(replay_checkpoint);
    if (mTemps.size() > replay_temp_mark) {
        mTemps.resize(replay_temp_mark);
    }
    mHasDeferredReplayCheckpoint = false;

    // Debug: dump saved tensor states after replay
    if (debug_replay && mSaved) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        int null_count = 0, live_count = 0;
        for (const auto& [sname, stensor] : *mSaved) {
            int lyr = -1;
            std::string fld;
            if (parse_block_param(sname, lyr, fld) && lyr == layer_idx) {
                if (stensor.Data) {
                    live_count++;
                } else {
                    null_count++;
                    if (debug_replay) fprintf(stderr, "[REPLAY] layer=%d saved NULL: %s\n", layer_idx, sname.c_str());
                }
            }
        }
        if (debug_replay)
            fprintf(stderr, "[REPLAY] layer=%d saved stats: live=%d null=%d\n", layer_idx, live_count, null_count);
    }
}

void CompiledExecutor::bind_runtime_bindings() {
    if (!mRuntimeBindings) {
        return;
    }
    for (const auto& binding : *mRuntimeBindings) {
        bind_tensor(binding.name, binding.tensor);
    }
}

bool CompiledExecutor::needs_non_block_weight_transfer() const {
    return mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled());
}

}  // namespace dsl
