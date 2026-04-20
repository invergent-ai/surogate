// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CompiledExecutor: execute_forward / execute_backward / replay_layer_forward.
// Extracted from compiled_ops.cpp to reduce file size; behavior unchanged.

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

// ---------------------------------------------------------------------------
// replay_layer_forward — torch-style gradient checkpointing
//
// Re-execute a single layer's compiled forward ops during backward to
// regenerate activations. The data lives on the stack; the caller (backward)
// must restore the stack checkpoint after consuming the data.
// ---------------------------------------------------------------------------

static bool rstd_on_stack_enabled() {
    static const bool enabled = []() {
        const char* e = std::getenv("SUROGATE_RSTD_ON_STACK");
        return e && std::string(e) == "1";
    }();
    return enabled;
}

// Clear-.Data helpers for Stack-backed slots that Stack.restore invalidates.
// These patterns repeat 4+ times in the dispatch loop (flat-ops path,
// SegmentDispatch path, PhaseExit BwdBlock); hoist to file-scope so the
// slot list stays in sync.
static void clear_rstd_stack_slots(DslRunState& rs, int L) {
    auto clear = [&](TensorSlot s) {
        if (Tensor* t = block_activation_ptr(rs, L, s)) t->Data = nullptr;
    };
    clear(TensorSlot::BlockLN1RSTD);
    clear(TensorSlot::BlockLN2RSTD);
    clear(TensorSlot::BlockQRSTD);
    clear(TensorSlot::BlockKRSTD);
    clear(TensorSlot::BlockLSE);
    clear(TensorSlot::BlockAttOut);
    clear(TensorSlot::BlockLN1);
    clear(TensorSlot::BlockLN2);
    clear(TensorSlot::BlockHOut);
}

static void clear_ffn_temp_stack_slots(DslRunState& rs, int L) {
    if (Tensor* t = block_activation_ptr(rs, L, TensorSlot::BlockMLPUp)) t->Data = nullptr;
    if (Tensor* t = block_activation_ptr(rs, L, TensorSlot::BlockSwiGLU)) t->Data = nullptr;
}

static void clear_large_bwd_grad_stack_slots(DslRunState& rs, int L) {
    if (Tensor* t = block_gradient_ptr(rs, L, TensorSlot::BlockDQKV)) t->Data = nullptr;
    if (Tensor* t = block_gradient_ptr(rs, L, TensorSlot::BlockDMLPUp)) t->Data = nullptr;
    if (Tensor* t = block_gradient_ptr(rs, L, TensorSlot::BlockDSwiGLU)) t->Data = nullptr;
}

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

    // Bind known inputs
    bind_tensor("token_ids", mRunState.Inputs);
    bind_tensor("position_ids", mRunState.PositionIDs);
    if (mRunState.VisualPosMasks.Data) {
        bind_tensor("visual_pos_masks", mRunState.VisualPosMasks);
    }
    if (mRunState.VisualEmbeds.Data) {
        bind_tensor("visual_embeds", mRunState.VisualEmbeds);
    }
    bind_tensor("x0", mRunState.non_block_activations().encoded);

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
    for (std::size_t idx = start; idx <= end && idx < fwd_graph.ops.size(); ++idx) {
        for (const auto& out : fwd_graph.ops[idx].outputs) {
            if (out.tensor_id >= 0) {
                produced_ids.insert(out.tensor_id);
            }
        }
    }

    // Pre-bind external inputs: tensors consumed by this layer but produced before it.
    // These include the layer's input residual, previous block outputs, RoPE freqs, etc.
    for (std::size_t idx = start; idx <= end && idx < fwd_graph.ops.size(); ++idx) {
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
                } else {
                    // Unqualified global-level reference — try slot table first.
                    const TensorSlot slot = builtin_slot_from_name(inp.name);
                    if (Tensor* gp = global_activation_ptr(mRunState, slot)) {
                        resolved = *gp;
                    } else if (inp.name.find("freq_cis") != std::string::npos ||
                               inp.name.find("rope_freqs") != std::string::npos) {
                        resolved = mRunState.rope_freqs(inp.name);
                    }
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

            // Embedding output (embed_1, embed_0, …) — substring match because
            // this family of names isn't (yet) in the slot registry. Routes to
            // the Encoded slot buffer.
            if (!resolved.Data && !inp.name.empty() && inp.name.find("embed") != std::string::npos) {
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
    for (std::size_t idx = start; idx <= end && idx < fwd_graph.ops.size(); ++idx) {
        const auto& op = fwd_graph.ops[idx];

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
            const std::string base = strip_ssa_suffix(saved_field);
            return base == "ln1" || base == "ln1_flat" || base == "ln" || base == "ln_flat" || base == "ln1_rstd" ||
                   base == "ln_rstd";
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
            // Fallback: resolve from simplified_acts via the shared block-activation
            // dispatch. `builtin_slot_from_name()` maps every alias (`ln`, `ln_flat`,
            // `qkv_rope`, `res_att`, …) to a TensorSlot; the helper routes it to
            // the right RunState buffer. New slots are added in ONE place
            // (tensor_slot_dispatch.cpp) rather than duplicated here.
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

        // Phase 4 M3 Phase A: rstd slots (ln*_rstd, q_rstd, k_rstd) are
        // Stack-backed under SUROGATE_RSTD_ON_STACK. The save-capture
        // loop above copies live .Data into mSaved; without the persist
        // below, that .Data is the Stack pointer which Stack.restore
        // invalidates. Copy into a persistent cudaMalloc buffer and
        // rebind BOTH the slot AND any mSaved / mNamedTensors /
        // mTensors entries that captured the old Stack ptr.
        if (rstd_on_stack_enabled() && layer_idx >= 0) {
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

void CompiledExecutor::execute_forward(const CompiledGraph& graph,
                                       NCCLCommunicator& comm,
                                       bool full,
                                       const modules::ForwardHook* hook) {
    mComm = &comm;
    mCurrentGraph = &graph;
    mTemps.clear();
    mMoEHostOffsetsCache.clear();
    // cudaFree and cudaMemPoolTrimTo are prohibited during CUDA stream capture —
    // they invalidate the capture. Skip all cleanup when capturing; it will run
    // on the next eager (non-captured) step instead.
    // Check both the inner capture flag and the actual stream status (for outer
    // whole-step captures like train_step_graphed that don't set mCapturing).
    cudaStreamCaptureStatus cleanup_capture_status = cudaStreamCaptureStatusNone;
    const bool cleanup_capturing =
        mCapturing || (cudaStreamIsCapturing(mRunState.MainStream, &cleanup_capture_status) == cudaSuccess &&
                       cleanup_capture_status != cudaStreamCaptureStatusNone);
    if (!cleanup_capturing) {
        // Free retired shared EP buffers from previous steps (accumulated during reallocation).
        // Previous step is fully complete, so these are no longer referenced.
        mEpStrategy->buffer_pool().clear_retired();
        // Free EP buffer pool — temporary buffers with short lifetimes (acquired/released
        // within a single dispatch call). As routing imbalance changes during training,
        // buffer sizes drift and stale entries become unreusable zombies. Clearing per-step
        // prevents this accumulation; cudaMalloc overhead is negligible vs A2A/GEMM costs.
        mEpStrategy->buffer_pool().clear_pool();
        // Trim CUDA stream-ordered memory pool to release cached allocations.
        // cuBLAS cublasGemmGroupedBatchedEx internally uses cudaMallocAsync;
        // trimming reclaims unused cached blocks from previous steps.
        {
            int device;
            cudaGetDevice(&device);
            cudaMemPool_t pool;
            if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
                cudaMemPoolTrimTo(pool, 0);
            }
        }
    }
    // Initialize flat tensor vector indexed by compile-time tensor IDs
    mTensors.assign(static_cast<std::size_t>(graph.num_tensors), Tensor{});
    mNamedTensors.clear();
    // Scrub mSaved entries pointing into the Stack arena. Between steps the
    // Stack top is reset; any saved Tensor whose Data lived in the Stack is
    // now a dangling pointer. Persistent (cudaMalloc-backed) saves are
    // preserved. Without this scrub, replay-layer persistence iterates over
    // step N-1's stack saves at step N and faults.
    if (mSaved) {
        for (auto& [name, tensor] : *mSaved) {
            if (tensor.Data && mRunState.Stack.owns(tensor.Data)) {
                tensor.Data = nullptr;
            }
        }
    }
    mCurrentLayer = -1;
    mSegmentDispatchedUntil = 0;

    // Match GraphExecutor behavior: initialize loss/counter buffers for full forward runs.
    // This avoids stale accumulation when tests call CompiledExecutor directly.
    if (full) {
        bool has_loss_op = false;
        for (const auto& op : graph.ops) {
            if (op.type == CompiledOpType::CrossEntropyLoss || op.type == CompiledOpType::FusedLMHeadLoss) {
                has_loss_op = true;
                break;
            }
        }
        if (has_loss_op) {
            fill_zero(mRunState.Losses, mRunState.MainStream);
            fill_zero(mRunState.ValidTokenCount, mRunState.MainStream);
            fill_zero(mRunState.CorrectCount, mRunState.MainStream);
        }
    }
    const int num_layers = static_cast<int>(mConfig.NumLayers);
    // Reuse member vectors to avoid per-forward heap allocations.
    auto& layer_checkpoints = mLayerCheckpoints;
    auto& layer_temp_marks = mLayerTempMarks;
    auto& layer_active = mLayerActive;
    if (num_layers > 0) {
        layer_checkpoints.resize(static_cast<std::size_t>(num_layers));
        layer_temp_marks.resize(static_cast<std::size_t>(num_layers));
        layer_active.assign(static_cast<std::size_t>(num_layers), 0);
    }

    // Build save mask for fast per-ID lookup during pruning
    mSaveMask.assign(static_cast<std::size_t>(graph.num_tensors), false);
    if (mSaveList) {
        for (const auto& name : *mSaveList) {
            int sid = graph.find_tensor_id(name);
            if (sid >= 0) {
                mSaveMask[static_cast<std::size_t>(sid)] = true;
            }
        }
    }

    // Layer-boundary bookkeeping. on_fwd_layer_start checkpoints the Stack
    // and records the temp watermark so layer_end can restore both. Called
    // from PhaseEnter / flat-ops-loop / tiled-MLP-start paths. The tiled-MLP
    // end path has simpler cleanup (no MoE save, no stack-slot clears) and
    // stays inline.
    auto on_fwd_layer_start = [&](int L) {
        if (L >= 0 && L < num_layers && !layer_active[static_cast<std::size_t>(L)]) {
            layer_checkpoints[static_cast<std::size_t>(L)] = mRunState.Stack.checkpoint();
            layer_temp_marks[static_cast<std::size_t>(L)] = mTemps.size();
            layer_active[static_cast<std::size_t>(L)] = 1;
        }
        if (L >= 0) handle_layer_start(L);
    };
    auto prune_stack_tensors = [&]() {
        // Prune flat tensor vector using pre-computed metadata (no string parsing)
        for (int id = 0; id < graph.num_tensors; ++id) {
            auto& t = mTensors[static_cast<std::size_t>(id)];
            if (!t.Data) continue;
            // Skip tensors needed for backward (in save list)
            if (mSaveMask[static_cast<std::size_t>(id)]) continue;
            // Skip cross-layer connector tensors (layerN.out, layerN.res_in, etc.)
            const auto& meta = graph.tensor_meta[static_cast<std::size_t>(id)];
            if (meta.is_cross_layer()) continue;
            if (mRunState.Stack.owns(t.Data) && !mRunState.Stack.is_live(t.Data)) {
                t = Tensor{};
            }
        }
        for (auto it = mNamedTensors.begin(); it != mNamedTensors.end();) {
            const Tensor& t = it->second;
            if (t.Data && mRunState.Stack.owns(t.Data) && !mRunState.Stack.is_live(t.Data)) {
                it = mNamedTensors.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Detect if the stream is being captured (either by internal graphs via mCapturing,
    // or by an outer full-step graph from train_step_graphed in py_train.cpp).
    cudaStreamCaptureStatus fwd_capture_status = cudaStreamCaptureStatusNone;
    const bool fwd_stream_capturing =
        mCapturing || (cudaStreamIsCapturing(mRunState.MainStream, &fwd_capture_status) == cudaSuccess &&
                       fwd_capture_status != cudaStreamCaptureStatusNone);

    // Check if a tensor will be recomputed in backward (same logic as save_tensors).
    const bool recompute_enabled_flag = mRecomputeEnabled;
    auto will_recompute_tensor = [&](const std::string& tensor_name) -> bool {
        if (!recompute_enabled_flag || !mSlotRegistry) {
            return false;
        }
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int lyr = -1;
        std::string fld;
        if (parse_block_param(tensor_name, lyr, fld)) {
            return mSlotRegistry->will_recompute(strip_ssa_suffix(fld), lora_only_mode);
        }
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    // When forward replay is active, ALL block tensors will be regenerated by
    // replay_layer_forward during backward. Save metadata only — no D2D copies needed.
    const bool forward_replay_active = recompute_enabled_flag && static_cast<bool>(mRecomputeFn);

    auto name_belongs_to_layer = [](const std::string& name, int target_layer) -> bool {
        int lyr = -1;
        std::string fld;
        return parse_block_param(name, lyr, fld) && lyr == target_layer;
    };

    auto contains_ci_local = [](std::string_view haystack, std::string_view needle) {
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
    const bool is_qwen3_5_forward_replay_model = contains_ci_local(mConfig.ModelTypeName, "qwen3_5") ||
                                                 contains_ci_local(mConfig.ModelTypeName, "qwen3.5") ||
                                                 contains_ci_local(mConfig.ArchitectureName, "qwen3_5") ||
                                                 contains_ci_local(mConfig.ArchitectureName, "qwen3.5");

    int arena_persists = 0;
    int cudaMalloc_persists = 0;
    auto persist_saved_layer_tensors = [&](int layer_idx) {
        if (!mSaved || !mSaveList) {
            return;
        }

        auto q35_forward_replay_needs_persist = [&](const std::string& name) -> bool {
            if (!forward_replay_active || !is_qwen3_5_forward_replay_model) {
                return false;
            }
            int resolved_layer = -1;
            std::string field;
            if (!parse_block_param(name, resolved_layer, field) || resolved_layer != layer_idx) {
                return false;
            }
            const std::string base_field = strip_ssa_suffix(field);
            return base_field == "ln1" || base_field == "ln1_flat" || base_field == "ln" || base_field == "ln_flat" ||
                   base_field == "ln1_rstd" || base_field == "ln_rstd";
        };

        auto persist_saved_source_now = [&](const std::string& name, const Tensor& src) -> bool {
            if (!src.Data) {
                return false;
            }
            const size_t bytes = src.bytes();
            if (bytes == 0) {
                return false;
            }
            auto buf_it = mMoeSavedBuffers.find(name);
            if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
                if (fwd_stream_capturing) {
                    return false;
                }
                if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                    CUDA_CHECK(cudaFree(buf_it->second));
                }
                void* new_buffer = nullptr;
                CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                mMoeSavedBuffers[name] = new_buffer;
                mMoeSavedSizes[name] = bytes;
            }
            void* dst_buffer = mMoeSavedBuffers[name];
            CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved_tensor = src;
            saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
            (*mSaved)[name] = saved_tensor;
            bind_tensor(name, saved_tensor);
            return true;
        };

        auto resolve_saved_source = [&](const std::string& name) -> std::optional<Tensor> {
            // Prefer exact match from the flat tensor vector (O(1) lookup).
            if (mCurrentGraph) {
                int tid = mCurrentGraph->find_tensor_id(name);
                if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                    return mTensors[tid];
                }
                // Fall back to SSA-suffixed entries via pre-computed ssa_base_to_id (O(1) vs O(N) scan).
                auto ssa_it = mCurrentGraph->ssa_base_to_id.find(name);
                if (ssa_it != mCurrentGraph->ssa_base_to_id.end()) {
                    int sid = ssa_it->second;
                    if (sid >= 0 && static_cast<std::size_t>(sid) < mTensors.size() && mTensors[sid].Data) {
                        return mTensors[sid];
                    }
                }
            }

            int resolved_layer = -1;
            std::string field;
            if (!parse_block_param(name, resolved_layer, field)) {
                return std::nullopt;
            }
            const std::string base_field = strip_ssa_suffix(field);
            // Special cases with field aliases not in the name→slot table or
            // that need a shape override (qkv_norm, qkv_flat, swiglu_flat).
            if (base_field == "qkv_norm") {
                if (Tensor* t = block_activation_ptr(mRunState, resolved_layer, TensorSlot::BlockQKV)) return *t;
                return std::nullopt;
            }
            if (base_field == "qkv_flat") {
                if (Tensor* t = block_activation_ptr(mRunState, resolved_layer, TensorSlot::BlockQKV)) {
                    Tensor qkv = *t;
                    return view_tensor(qkv, {qkv.Sizes[0] * qkv.Sizes[1], qkv.Sizes[2]});
                }
                return std::nullopt;
            }
            if (base_field == "swiglu_flat") {
                if (Tensor* t = block_activation_ptr(mRunState, resolved_layer, TensorSlot::BlockSwiGLU)) {
                    Tensor swiglu = *t;
                    return view_tensor(swiglu, {swiglu.Sizes[0] * swiglu.Sizes[1], swiglu.Sizes[2]});
                }
                return std::nullopt;
            }
            // Primary dispatch: name→slot table covers every alias.
            // block_activation_ptr handles the qkv_rope fallback to qkv and
            // the BlockResidualFFN managed-residual acquisition.
            if (Tensor* t = block_activation_ptr(mRunState, resolved_layer, builtin_slot_from_name(base_field))) {
                return *t;
            }
            return std::nullopt;
        };

        // When forward replay is active, save metadata only for most block tensors.
        // Qwen3.5 LN1 replay is the exception: ln1/ln1_rstd sit in shared activation
        // space, so their exact forward values must be copied at layer end before
        // later ops overwrite the slot.
        if (forward_replay_active) {
            for (const auto& name : *mSaveList) {
                if (!name_belongs_to_layer(name, layer_idx)) continue;
                if (mSaved->find(name) != mSaved->end()) continue;
                if (q35_forward_replay_needs_persist(name)) {
                    auto src_opt = resolve_saved_source(name);
                    if (src_opt.has_value() && persist_saved_source_now(name, *src_opt)) {
                        continue;
                    }
                }
                Tensor meta{};
                (*mSaved)[name] = meta;
            }
            return;
        }

        int saved_count = 0;
        int recompute_count = 0;
        for (const auto& name : *mSaveList) {
            if (!name_belongs_to_layer(name, layer_idx)) {
                continue;
            }
            if (mSaved->find(name) != mSaved->end()) {
                continue;
            }
            // Skip tensors that will be recomputed in backward — save metadata only.
            // This avoids allocating persistent cudaMalloc buffers for tensors like
            // mlp_up and swiglu that are stack-backed but fully recomputable.
            if (will_recompute_tensor(name)) {
                auto src_opt = resolve_saved_source(name);
                if (src_opt.has_value() && src_opt->Data) {
                    Tensor meta = *src_opt;
                    meta.Data = nullptr;  // Metadata only — no data pointer
                    (*mSaved)[name] = meta;
                    recompute_count++;
                }
                continue;
            }
            auto src_opt = resolve_saved_source(name);
            if (!src_opt.has_value()) {
                continue;
            }
            const Tensor& src = *src_opt;
            if (!src.Data || !mRunState.Stack.owns(src.Data)) {
                continue;
            }
            const size_t bytes = src.bytes();
            if (bytes == 0) {
                continue;
            }

            // Arena-backed persist (M5.d): when the phase-tree arenas are
            // bound and the tid is classified as SaveForBwd, copy directly
            // into the pre-allocated arena slot instead of cudaMalloc'ing a
            // per-name persistent buffer. Skips cudaMalloc on the hot path.
            void* dst_buffer = nullptr;
            bool used_arena = false;
            if (mPhaseArenas && mPhaseArenas->allocated && mCurrentGraph) {
                const int tid = mCurrentGraph->find_tensor_id(name);
                if (tid >= 0) {
                    std::byte* arena_ptr = dsl::resolve_tid_in_arena(*mPhaseArenas, *mCurrentGraph, tid);
                    if (arena_ptr != nullptr) {
                        const auto& meta = mCurrentGraph->tensor_meta[static_cast<std::size_t>(tid)];
                        if (meta.region == dsl::RegionKind::SaveForBwd && meta.bytes >= bytes) {
                            dst_buffer = arena_ptr;
                            used_arena = true;
                        }
                    }
                }
            }

            if (!used_arena) {
                auto buf_it = mMoeSavedBuffers.find(name);
                if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
                    if (fwd_stream_capturing) {
                        // Cannot cudaMalloc during any CUDA graph capture (internal or outer).
                        // Skip this tensor — the outer capture warmup or
                        // prepare_saved_buffers_for_capture should have pre-allocated it.
                        continue;
                    }
                    if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                        CUDA_CHECK(cudaFree(buf_it->second));
                    }
                    void* new_buffer = nullptr;
                    CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                    mMoeSavedBuffers[name] = new_buffer;
                    mMoeSavedSizes[name] = bytes;
                }
                dst_buffer = mMoeSavedBuffers[name];
            }

            CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved_tensor = src;
            saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
            (*mSaved)[name] = saved_tensor;
            bind_tensor(name, saved_tensor);
            saved_count++;
            if (used_arena) {
                ++arena_persists;
            } else {
                ++cudaMalloc_persists;
            }
        }
    };

    // Layer-end bookkeeping shared by PhaseExit-FwdBlock and the flat-ops
    // loop. Persists stack-backed saves before Stack.restore, prunes dead
    // tensors, clears the per-layer stack-backed slots (rstd / ffn_temps).
    // Defined here (after persist_saved_layer_tensors) so the capture
    // resolves. The tiled-MLP path uses a simpler inline variant — no MoE
    // save, no stack-slot clears — and is intentionally not routed through
    // this helper.
    auto on_fwd_layer_end = [&](int L) {
        if (L >= 0 && L < num_layers && layer_active[static_cast<std::size_t>(L)]) {
            if (mDebugDumpLayerFn) mDebugDumpLayerFn(L);
            if (mConfig.NumExperts > 0) save_moe_layer_tensors(L);
            persist_saved_layer_tensors(L);
            mRunState.Stack.restore(layer_checkpoints[static_cast<std::size_t>(L)]);
            if (mTemps.size() > layer_temp_marks[static_cast<std::size_t>(L)]) {
                mTemps.resize(layer_temp_marks[static_cast<std::size_t>(L)]);
            }
            prune_stack_tensors();
            if (mRunState.ffn_temps_on_stack()) clear_ffn_temp_stack_slots(mRunState, L);
            if (rstd_on_stack_enabled()) clear_rstd_stack_slots(mRunState, L);
            layer_active[static_cast<std::size_t>(L)] = 0;
        }
        if (L >= 0) handle_layer_end(L);
    };

    // Bind known inputs (into both flat vector and write-through mirror)
    bind_tensor("token_ids", mRunState.Inputs);
    bind_tensor("position_ids", mRunState.PositionIDs);
    if (mRunState.VisualPosMasks.Data) {
        bind_tensor("visual_pos_masks", mRunState.VisualPosMasks);
    }
    if (mRunState.VisualEmbeds.Data) {
        bind_tensor("visual_embeds", mRunState.VisualEmbeds);
    }
    if (!mRunState.DeepstackVisualEmbeds.empty()) {
        for (std::size_t i = 0; i < mRunState.DeepstackVisualEmbeds.size(); ++i) {
            if (!mRunState.DeepstackVisualEmbeds[i].Data) {
                continue;
            }
            bind_tensor("deepstack_visual_embeds_" + std::to_string(i), mRunState.DeepstackVisualEmbeds[i]);
        }
    }
    bind_tensor("x0", mRunState.non_block_activations().encoded);

    // Ensure non-block weights are gathered if streaming/offload is enabled
    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_embeddings(comm, mRunState.MainStream);
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
    }

    // Prefetch layer 0 before loop
    mPrefetchDirection = 1;  // Forward traversal
    if (mConfig.NumLayers > 0 && !mCapturing) {
        if (mWeightManager && mWeightManager->needs_block_gather()) {
            mWeightManager->gather_block(0, comm, mRunState.side_stream());
        }
        // QLoRA offload: prefetch first layer's quantized weights
        if (auto* provider = mWeights.qlora_provider()) {
            if (provider->has_offloading()) {
                provider->prefetch_for_layer(0, mRunState.side_stream());
            }
        }
    }

    // Main dispatch loop - no string comparisons, direct function pointer dispatch
    const char* op_trace_env = std::getenv("SUROGATE_OP_TRACE");
    const bool op_trace = op_trace_env && std::string(op_trace_env) != "0";
    const int debug_nonfinite_mode = env_int("SUROGATE_DEBUG_CHECK_NONFINITE", 0);
    const bool debug_nonfinite_forward = (debug_nonfinite_mode & 0x1) != 0;
    const char* watch_tensor_env = std::getenv("SUROGATE_DEBUG_WATCH_TENSOR");
    const std::string watch_tensor_name = watch_tensor_env ? std::string(watch_tensor_env) : std::string();
    const bool watch_tensor_enabled = !watch_tensor_name.empty();
    const float watch_amax_delta = env_float("SUROGATE_DEBUG_WATCH_AMAX_DELTA", 1.0f);
    const float watch_alarm_amax = env_float("SUROGATE_DEBUG_WATCH_ALARM_AMAX", 1e6f);
    const bool watch_abort_on_alarm = env_int("SUROGATE_DEBUG_WATCH_ABORT", 0) != 0;
    int watch_tensor_id = -1;
    if (watch_tensor_enabled && mCurrentGraph) {
        const int tid = mCurrentGraph->find_tensor_id(watch_tensor_name);
        if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
            watch_tensor_id = tid;
        }
    }
    if (watch_tensor_enabled && watch_tensor_id < 0) {
        for (const auto& scan_op : graph.ops) {
            bool found = false;
            for (const auto& ref : scan_op.inputs) {
                if (ref.name == watch_tensor_name && ref.tensor_id >= 0 &&
                    static_cast<std::size_t>(ref.tensor_id) < mTensors.size()) {
                    watch_tensor_id = ref.tensor_id;
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
            for (const auto& ref : scan_op.outputs) {
                if (ref.name == watch_tensor_name && ref.tensor_id >= 0 &&
                    static_cast<std::size_t>(ref.tensor_id) < mTensors.size()) {
                    watch_tensor_id = ref.tensor_id;
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
        }
    }
    if (watch_tensor_enabled) {
        int watch_input_refs = 0;
        int watch_output_refs = 0;
        for (const auto& scan_op : graph.ops) {
            for (const auto& ref : scan_op.inputs) {
                if (ref.name == watch_tensor_name) {
                    watch_input_refs++;
                }
            }
            for (const auto& ref : scan_op.outputs) {
                if (ref.name == watch_tensor_name) {
                    watch_output_refs++;
                }
            }
        }
        std::cerr << "[WATCH_META] tensor='" << watch_tensor_name << "' tensor_id=" << watch_tensor_id
                  << " input_refs=" << watch_input_refs << " output_refs=" << watch_output_refs << std::endl;
    }
    auto try_bind_watch_tensor_id_from_ref = [&](const TensorRef& ref) {
        if (watch_tensor_id >= 0) {
            return;
        }
        if (ref.name != watch_tensor_name) {
            return;
        }
        if (ref.tensor_id < 0) {
            return;
        }
        if (static_cast<std::size_t>(ref.tensor_id) >= mTensors.size()) {
            return;
        }
        watch_tensor_id = ref.tensor_id;
    };
    auto try_get_watch_tensor = [&](const CompiledOp* op_ctx = nullptr) -> const Tensor* {
        if (!watch_tensor_enabled) {
            return nullptr;
        }
        if (op_ctx) {
            for (const auto& ref : op_ctx->inputs) {
                try_bind_watch_tensor_id_from_ref(ref);
            }
            for (const auto& ref : op_ctx->outputs) {
                try_bind_watch_tensor_id_from_ref(ref);
            }
        }
        if (watch_tensor_id >= 0 && static_cast<std::size_t>(watch_tensor_id) < mTensors.size() &&
            mTensors[static_cast<std::size_t>(watch_tensor_id)].Data) {
            return &mTensors[static_cast<std::size_t>(watch_tensor_id)];
        }
        if (mWeights.has(watch_tensor_name)) {
            Tensor& w = mWeights.get(watch_tensor_name);
            if (w.Data) {
                return &w;
            }
        }
        if (const Tensor* direct = try_get_tensor(watch_tensor_name)) {
            return direct;
        }
        return try_get_tensor_fuzzy(watch_tensor_name);
    };
    auto watch_non_finite_count = [&](const Tensor& t) -> int {
        if (t.DType != ETensorDType::BF16 && t.DType != ETensorDType::FP32) {
            return -1;
        }
        Tensor non_finite_count = mRunState.temp_alloc(ETensorDType::INT32, {1}, "non_finite_count");
        CUDA_CHECK(cudaMemsetAsync(non_finite_count.Data, 0, sizeof(int), mRunState.MainStream));
        count_non_finite(non_finite_count, t, mRunState.MainStream);
        int host_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_count,
                                   non_finite_count.get<int>(),
                                   sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        mRunState.temp_free(non_finite_count);
        return host_count;
    };
    auto watch_absmax = [&](const Tensor& t) -> float {
        if (t.DType != ETensorDType::BF16 && t.DType != ETensorDType::FP32) {
            return -1.0f;
        }
        Tensor amax = mRunState.temp_alloc(ETensorDType::FP32, {1}, "amax");
        CUDA_CHECK(cudaMemsetAsync(amax.Data, 0, sizeof(float), mRunState.MainStream));
        global_amax(amax.get<float>(),
                    t,
                    static_cast<std::size_t>(t.nelem()),
                    mRunState.DeviceProp,
                    mRunState.MainStream);
        float host_amax = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&host_amax,
                                   amax.get<float>(),
                                   sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        mRunState.temp_free(amax);
        return host_amax;
    };
    auto check_nonfinite_refs = [&](const CompiledOp& op, const std::vector<TensorRef>& refs) {
        if (!debug_nonfinite_forward) {
            return;
        }
        for (const auto& ref : refs) {
            if (ref.name.empty()) {
                continue;
            }
            const Tensor* t = try_get_tensor(ref.name);
            if (!t || !t->Data) {
                continue;
            }
            if (t->DType != ETensorDType::BF16 && t->DType != ETensorDType::FP32) {
                continue;
            }

            Tensor non_finite_count = mRunState.temp_alloc(ETensorDType::INT32, {1}, "non_finite_count");
            CUDA_CHECK(cudaMemsetAsync(non_finite_count.Data, 0, sizeof(int), mRunState.MainStream));
            count_non_finite(non_finite_count, *t, mRunState.MainStream);
            int host_count = 0;
            CUDA_CHECK(cudaMemcpyAsync(&host_count,
                                       non_finite_count.get<int>(),
                                       sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            mRunState.temp_free(non_finite_count);

            if (host_count > 0) {
                std::ostringstream oss;
                oss << "Non-finite detected in forward output tensor '" << ref.name << "' at op id=" << op.op_id
                    << " type=" << op_type_to_string(op.type) << " count=" << host_count
                    << " dtype=" << static_cast<int>(t->DType) << " shape=[";
                for (int d = 0; d < t->Rank; ++d) {
                    if (d > 0) oss << ",";
                    oss << t->Sizes[d];
                }
                oss << "]";
                throw std::runtime_error(oss.str());
            }
        }
    };
    // Build tile group lookup for long-context tiled MLP execution
    std::unordered_map<std::size_t, const MlpTileGroup*> tile_group_starts;
    for (const auto& tg : graph.mlp_tile_groups) {
        tile_group_starts[tg.start_op_idx] = &tg;
    }

    // Stream-driven forward execution. Reuses today's allocator, stack, and
    // op dispatch; drives them from graph.instruction_stream instead of the
    // flat ops list + op.layer_start/end flags. Phase 3 subsystem #1 flip:
    // SegmentDispatch honors graph.layer_segments when mSplitAttentionGraphs
    // is on, replicating today's per-segment graph-capture-or-eager dispatch.
    // Phase 4: stream-driven interpreter is now the default. Set
    // SUROGATE_LEGACY_EXECUTOR=1 to fall back to the flat-ops path
    // (kept for one release cycle as an escape hatch for regressions).
    const bool legacy_escape = [] {
        const char* e = std::getenv("SUROGATE_LEGACY_EXECUTOR");
        return e && std::string(e) == "1";
    }();
    const bool stream_driven = !legacy_escape && !graph.instruction_stream.empty() && !mCapturing;
    if (stream_driven) {
        if (const char* env = std::getenv("SUROGATE_DEBUG_PHASE_INTERPRETER")) {
            if (std::string(env) == "1") {
                std::cerr << "[phase-interp] forward: stream-driven (" << graph.instruction_stream.size()
                          << " instructions)\n";
            }
        }
        for (const auto& inst : graph.instruction_stream) {
            switch (inst.kind) {
                case dsl::InstKind::PhaseEnter:
                    if (inst.phase_kind == dsl::PhaseKind::FwdBlock && inst.block_index >= 0) {
                        on_fwd_layer_start(inst.block_index);
                        mCurrentLayer = inst.block_index;
                    }
                    break;
                case dsl::InstKind::SegmentDispatch: {
                    // Phase 3 subsystem #1 flip: honor graph.layer_segments when
                    // split-attention is active. Each layer's ops are dispatched
                    // as an ordered list of graph-captured / eager segments
                    // (replicates compiled_ops_execute.cpp:1228-1325 semantics
                    // in the stream-driven path).
                    const int L = inst.block_index;
                    const bool splits_disabled = std::getenv("SUROGATE_DISABLE_SPLIT_SEG") != nullptr;
                    const bool use_splits = !splits_disabled && mSplitAttentionGraphs &&
                                            inst.phase_kind == dsl::PhaseKind::FwdBlock && L >= 0 &&
                                            static_cast<std::size_t>(L) < graph.layer_segments.size() &&
                                            !graph.layer_segments[static_cast<std::size_t>(L)].empty();
                    if (use_splits) {
                        const auto& segs = graph.layer_segments[static_cast<std::size_t>(L)];
                        for (std::size_t s = 0; s < segs.size(); ++s) {
                            const auto& seg = segs[s];
                            if (seg.eager) {
                                const MlpTileGroup* tile_group = nullptr;
                                for (const auto& tg : graph.mlp_tile_groups) {
                                    if (tg.start_op_idx == seg.start_op) {
                                        tile_group = &tg;
                                        break;
                                    }
                                }
                                if (tile_group) {
                                    execute_tiled_mlp(graph, *tile_group, mB, mT, hook);
                                } else {
                                    for (std::size_t i = seg.start_op; i < seg.end_op; ++i) {
                                        dispatch_forward_op(graph.ops[i], hook);
                                    }
                                }
                            } else {
                                auto& sg = mFwdSegGraphs[static_cast<std::size_t>(L)][s];
                                const bool is_capture = (sg.exec == nullptr);
                                std::size_t saved_before = mSaved ? mSaved->size() : 0;
                                auto run = [&]() {
                                    for (std::size_t i = seg.start_op; i < seg.end_op; ++i) {
                                        dispatch_forward_op(graph.ops[i], hook);
                                    }
                                };
                                trace_or_execute_cuda_graph_with_stack(run,
                                                                       mRunState.MainStream,
                                                                       sg.exec,
                                                                       true,
                                                                       mRunState.Stack,
                                                                       sg.checkpoint);
                                if (is_capture) {
                                    sg.post_checkpoint = mRunState.Stack.checkpoint();
                                    sg.tensor_snapshot.clear();
                                    sg.named_snapshot.clear();
                                    sg.saved_snapshot.clear();
                                    for (int tid = 0; tid < static_cast<int>(mTensors.size()); ++tid) {
                                        if (mTensors[tid].Data) sg.tensor_snapshot.emplace_back(tid, mTensors[tid]);
                                    }
                                    for (const auto& [name, t] : mNamedTensors) {
                                        if (t.Data) sg.named_snapshot.emplace_back(name, t);
                                    }
                                    if (mSaved && mSaved->size() > saved_before) {
                                        for (const auto& [name, t] : *mSaved) {
                                            sg.saved_snapshot.emplace_back(name, t);
                                        }
                                    }
                                } else {
                                    mRunState.Stack.restore(sg.post_checkpoint);
                                    for (const auto& [tid, t] : sg.tensor_snapshot) {
                                        if (static_cast<std::size_t>(tid) < mTensors.size()) mTensors[tid] = t;
                                    }
                                    for (const auto& [name, t] : sg.named_snapshot)
                                        mNamedTensors[name] = t;
                                    if (mSaved) {
                                        for (const auto& [name, t] : sg.saved_snapshot) {
                                            if (mSaved->find(name) == mSaved->end()) (*mSaved)[name] = t;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for (std::size_t i = inst.op_start; i < inst.op_end; ++i) {
                            if (!full && !graph.required_mask.empty() && !graph.required_mask[i]) continue;
                            const auto& op = graph.ops[i];
                            if (!op.fn) continue;
                            op.fn(*this, op, static_cast<const void*>(hook));
                        }
                    }
                    break;
                }
                case dsl::InstKind::PruneByLastUse:
                    // Pass-through mode: pruning happens at PhaseExit for FwdBlock,
                    // matching today's layer_end semantics. Per-segment pruning is
                    // a future optimization.
                    break;
                case dsl::InstKind::RecomputeBlock:
                    // Forward direction: no-op. Recompute fires in backward
                    // PhaseEnter BwdBlock via subsystem #7's handler.
                    break;
                case dsl::InstKind::PhaseExit:
                    if (inst.phase_kind == dsl::PhaseKind::FwdBlock && inst.block_index >= 0) {
                        on_fwd_layer_end(inst.block_index);
                    }
                    break;
            }
        }
    }

    for (std::size_t idx = 0; !stream_driven && idx < graph.ops.size(); ++idx) {
        if (!full && !graph.required_mask.empty() && !graph.required_mask[idx]) {
            continue;
        }

        // Check if this op starts a tiled MLP group
        if (!tile_group_starts.empty()) {
            auto tg_it = tile_group_starts.find(idx);
            if (tg_it != tile_group_starts.end()) {
                const auto& tg = *tg_it->second;
                // Handle layer start if the first op has one
                const auto& first_op = graph.ops[tg.start_op_idx];
                if (first_op.layer_start >= 0) {
                    on_fwd_layer_start(first_op.layer_start);
                }
                execute_tiled_mlp(graph, tg, mB, mT, hook);
                // Handle layer end if the last op has one
                const auto& last_op = graph.ops[tg.end_op_idx];
                if (last_op.layer_end >= 0) {
                    if (last_op.layer_end < num_layers && layer_active[static_cast<std::size_t>(last_op.layer_end)]) {
                        if (mDebugDumpLayerFn) mDebugDumpLayerFn(last_op.layer_end);
                        persist_saved_layer_tensors(last_op.layer_end);
                        mRunState.Stack.restore(layer_checkpoints[static_cast<std::size_t>(last_op.layer_end)]);
                        if (mTemps.size() > layer_temp_marks[static_cast<std::size_t>(last_op.layer_end)]) {
                            mTemps.resize(layer_temp_marks[static_cast<std::size_t>(last_op.layer_end)]);
                        }
                        prune_stack_tensors();
                        layer_active[static_cast<std::size_t>(last_op.layer_end)] = 0;
                    }
                    handle_layer_end(last_op.layer_end);
                }
                idx = tg.end_op_idx;  // skip past the tile group (loop will ++idx)
                continue;
            }
        }

        const auto& op = graph.ops[idx];

        bool watch_pre_valid = false;
        int watch_pre_nf = -1;
        float watch_pre_amax = -1.0f;
        if (watch_tensor_enabled) {
            if (const Tensor* wt = try_get_watch_tensor(&op)) {
                if (wt->Data && (wt->DType == ETensorDType::BF16 || wt->DType == ETensorDType::FP32)) {
                    watch_pre_nf = watch_non_finite_count(*wt);
                    watch_pre_amax = watch_absmax(*wt);
                    watch_pre_valid = true;
                }
            }
        }

        if (op_trace) {
            std::cerr << "[OP " << idx << "] " << op_type_to_string(op.type) << " id=" << op.op_id << std::endl;
        }

        // Handle layer boundaries
        if (op.layer_start >= 0) {
            on_fwd_layer_start(op.layer_start);

            // Split-attention graph mode: pre-dispatch all layer ops via segments.
            // Non-attention segments are captured/replayed as CUDA graphs;
            // graph-breaking ops run eagerly. The normal loop still iterates
            // through these ops for layer_end handling, tensor persistence, and
            // pruning — but skips the per-op dispatch (already done here).
            if (mSplitAttentionGraphs && !graph.layer_segments.empty() &&
                static_cast<std::size_t>(op.layer_start) < graph.layer_segments.size() &&
                !graph.layer_segments[static_cast<std::size_t>(op.layer_start)].empty()) {
                const int L = op.layer_start;
                const auto& segs = graph.layer_segments[static_cast<std::size_t>(L)];
                for (std::size_t s = 0; s < segs.size(); ++s) {
                    const auto& seg = segs[s];
                    if (seg.eager) {
                        // Check if this eager segment matches an MLP tile group.
                        // compute_layer_segments emits these as single eager segments.
                        const MlpTileGroup* tile_group = nullptr;
                        for (const auto& tg : graph.mlp_tile_groups) {
                            if (tg.start_op_idx == seg.start_op) {
                                tile_group = &tg;
                                break;
                            }
                        }
                        if (tile_group) {
                            execute_tiled_mlp(graph, *tile_group, mB, mT, hook);
                        } else {
                            for (std::size_t i = seg.start_op; i < seg.end_op; ++i) {
                                dispatch_forward_op(graph.ops[i], hook);
                            }
                        }
                    } else {
                        auto& sg = mFwdSegGraphs[static_cast<std::size_t>(L)][s];
                        const bool is_capture = (sg.exec == nullptr);
                        // Track mSaved entries before segment to detect new ones
                        std::size_t saved_before = mSaved ? mSaved->size() : 0;
                        auto run = [&]() {
                            for (std::size_t i = seg.start_op; i < seg.end_op; ++i) {
                                dispatch_forward_op(graph.ops[i], hook);
                            }
                        };
                        trace_or_execute_cuda_graph_with_stack(run,
                                                               mRunState.MainStream,
                                                               sg.exec,
                                                               true,
                                                               mRunState.Stack,
                                                               sg.checkpoint);
                        if (is_capture) {
                            // Save post-dispatch stack state. On replay,
                            // trace_or_execute restores to sg.checkpoint (pre-alloc)
                            // before launching the graph. We must then advance
                            // past the graph's stack allocations so the next
                            // segment doesn't overlap.
                            sg.post_checkpoint = mRunState.Stack.checkpoint();

                            // Snapshot all tensor entries after capture. On replay
                            // dispatch doesn't run so mTensors/mNamedTensors/mSaved
                            // wouldn't be populated.
                            sg.tensor_snapshot.clear();
                            sg.named_snapshot.clear();
                            sg.saved_snapshot.clear();
                            for (int tid = 0; tid < static_cast<int>(mTensors.size()); ++tid) {
                                if (mTensors[tid].Data) {
                                    sg.tensor_snapshot.emplace_back(tid, mTensors[tid]);
                                }
                            }
                            for (const auto& [name, t] : mNamedTensors) {
                                if (t.Data) {
                                    sg.named_snapshot.emplace_back(name, t);
                                }
                            }
                            // Snapshot mSaved entries added by dispatch (e.g.
                            // MambaGatedRMSNorm writes rstd directly to mSaved)
                            if (mSaved && mSaved->size() > saved_before) {
                                for (const auto& [name, t] : *mSaved) {
                                    sg.saved_snapshot.emplace_back(name, t);
                                }
                            }
                        } else {
                            // Advance stack past graph's allocations so the next
                            // segment (eager attention) doesn't overlap.
                            mRunState.Stack.restore(sg.post_checkpoint);

                            // Restore tensor/saved entries from capture snapshot
                            for (const auto& [tid, t] : sg.tensor_snapshot) {
                                if (static_cast<std::size_t>(tid) < mTensors.size()) {
                                    mTensors[tid] = t;
                                }
                            }
                            for (const auto& [name, t] : sg.named_snapshot) {
                                mNamedTensors[name] = t;
                            }
                            if (mSaved) {
                                for (const auto& [name, t] : sg.saved_snapshot) {
                                    if (mSaved->find(name) == mSaved->end()) {
                                        (*mSaved)[name] = t;
                                    }
                                }
                            }
                        }
                    }
                }
                // Mark: layer ops already dispatched. The normal loop will still
                // iterate through them for layer_end handling but skip dispatch.
                mSegmentDispatchedUntil = graph.layer_end_indices[static_cast<std::size_t>(L)];
            }
        }

        // Skip dispatch for ops already handled by split-attention segment execution.
        // The normal loop still runs for these ops to handle layer_end boundaries,
        // tensor persistence, pruning, etc.
        if (idx < mSegmentDispatchedUntil) {
            goto skip_dispatch;
        }

        try {
            // Phase 2a: dispatch via the function pointer baked into
            // op.fn at graph compile time. One indirect call, no switch.
            if (!op.fn) {
                throw std::runtime_error(std::string("CompiledExecutor: no dispatch fn for forward op type ") +
                                         op_type_to_string(op.type));
            }
            op.fn(*this, op, static_cast<const void*>(hook));
            check_nonfinite_refs(op, op.outputs);
            if (watch_tensor_enabled) {
                bool watch_post_valid = false;
                int watch_post_nf = -1;
                float watch_post_amax = -1.0f;
                if (const Tensor* wt = try_get_watch_tensor(&op)) {
                    if (wt->Data && (wt->DType == ETensorDType::BF16 || wt->DType == ETensorDType::FP32)) {
                        watch_post_nf = watch_non_finite_count(*wt);
                        watch_post_amax = watch_absmax(*wt);
                        watch_post_valid = true;
                    }
                }

                const bool became_invalid = watch_pre_valid && !watch_post_valid;
                const bool became_valid = !watch_pre_valid && watch_post_valid;
                const bool nf_changed = watch_pre_valid && watch_post_valid && watch_pre_nf != watch_post_nf;
                const bool amax_changed = watch_pre_valid && watch_post_valid &&
                                          std::fabs(watch_post_amax - watch_pre_amax) > watch_amax_delta;
                const bool alarm = watch_post_valid && (watch_post_nf > 0 || !std::isfinite(watch_post_amax) ||
                                                        watch_post_amax >= watch_alarm_amax);

                if (became_invalid || became_valid || nf_changed || amax_changed || alarm) {
                    std::ostringstream in_list;
                    for (std::size_t ri = 0; ri < op.inputs.size(); ++ri) {
                        if (ri > 0) in_list << ",";
                        in_list << op.inputs[ri].name << "#" << op.inputs[ri].tensor_id;
                    }
                    std::ostringstream out_list;
                    for (std::size_t ro = 0; ro < op.outputs.size(); ++ro) {
                        if (ro > 0) out_list << ",";
                        out_list << op.outputs[ro].name << "#" << op.outputs[ro].tensor_id;
                    }
                    std::cerr << "[WATCH] tensor='" << watch_tensor_name << "' op_idx=" << idx << " op_id=" << op.op_id
                              << " type=" << op_type_to_string(op.type) << " pre_valid=" << (watch_pre_valid ? 1 : 0)
                              << " post_valid=" << (watch_post_valid ? 1 : 0) << " pre_nf=" << watch_pre_nf
                              << " post_nf=" << watch_post_nf << " pre_amax=" << watch_pre_amax
                              << " post_amax=" << watch_post_amax << " inputs=[" << in_list.str() << "]" << " outputs=["
                              << out_list.str() << "]" << std::endl;
                }
                if (alarm && watch_abort_on_alarm) {
                    std::ostringstream oss_watch;
                    oss_watch << "Watch tensor '" << watch_tensor_name << "' alarm after op idx=" << idx
                              << " id=" << op.op_id << " type=" << op_type_to_string(op.type) << " nf=" << watch_post_nf
                              << " amax=" << watch_post_amax;
                    throw std::runtime_error(oss_watch.str());
                }
            }
            // After each op, check for sticky CUDA errors (debug mode)
            if (op_trace) {
                auto post_err = cudaGetLastError();
                if (post_err != cudaSuccess) {
                    std::ostringstream oss2;
                    oss2 << "CompiledExecutor forward op " << idx << " (type=" << op_type_to_string(op.type)
                         << ", id=" << op.op_id << "): left sticky CUDA error: " << cudaGetErrorString(post_err);
                    throw std::runtime_error(oss2.str());
                }
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor forward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            throw std::runtime_error(oss.str());
        }

    skip_dispatch:

        // Handle layer end
        if (op.layer_end >= 0) {
            on_fwd_layer_end(op.layer_end);
        }
    }

    // Free temporaries
    for (auto it = mTemps.rbegin(); it != mTemps.rend(); ++it) {
        mRunState.temp_free(*it);
    }
    mTemps.clear();

    if (const char* env = std::getenv("SUROGATE_DEBUG_ARENA_COVERAGE")) {
        if (std::string(env) == "1") {
            std::cerr << "[arena-persist] arena=" << arena_persists << " cudaMalloc=" << cudaMalloc_persists << "\n";
        }
    }

    // Dump requested non-block tensors (e.g. xF/residual_final) after forward.
    // Per-layer block dumps are handled in the layer_end callback above.
    if (mDebugDumpFn) {
        static const char* dump_tensors_env = std::getenv("SUROGATE_DEBUG_DUMP_TENSORS");
        if (dump_tensors_env && *dump_tensors_env) {
            std::vector<std::string> names;
            {
                std::string token;
                std::stringstream ss(dump_tensors_env);
                while (std::getline(ss, token, ',')) {
                    // trim ASCII whitespace
                    std::size_t b = 0;
                    while (b < token.size() && std::isspace(static_cast<unsigned char>(token[b]))) {
                        ++b;
                    }
                    std::size_t e = token.size();
                    while (e > b && std::isspace(static_cast<unsigned char>(token[e - 1]))) {
                        --e;
                    }
                    if (e > b) {
                        names.emplace_back(token.substr(b, e - b));
                    }
                }
            }
            std::vector<std::string> global_names;
            global_names.reserve(names.size());
            for (const auto& name : names) {
                if (name.rfind("blocks[", 0) != 0) {
                    global_names.push_back(name);
                }
            }
            if (!global_names.empty()) {
                cudaStreamCaptureStatus dump_capture_status = cudaStreamCaptureStatusNone;
                const bool dump_capturing =
                    (cudaStreamIsCapturing(mRunState.MainStream, &dump_capture_status) == cudaSuccess &&
                     dump_capture_status != cudaStreamCaptureStatusNone);
                if (!dump_capturing) {
                    mDebugDumpFn(global_names, -1);
                }
            }
        }
    }

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_embeddings(mRunState.MainStream);
        mWeightManager->release_final_norm(mRunState.MainStream);
    }
}

void CompiledExecutor::execute_backward(const CompiledGraph& graph,
                                        NCCLCommunicator& comm,
                                        int grad_accum_steps,
                                        int micro_step,
                                        const modules::BackwardHook* hook,
                                        bool skip_zeroing) {
    mComm = &comm;
    mCurrentGraph = &graph;
    mRunState.reset_simplified_gradients();
    mTemps.clear();
    // Phase 3 subsystem #4: reset per-step bump cursor for bwd_cross_layer arena.
    mBwdCrossLayerBumpOffset = 0;
    // For EP models, keep forward-cached host offsets (populated by ep_dispatch).
    // During gradient checkpointing recompute, ep_dispatch is skipped (it's a
    // communication op), so the GPU persistent buffers may be stale. The forward
    // cache has the correct merged expert offsets for each layer.
    if (mConfig.EPSize <= 1) {
        mMoEHostOffsetsCache.clear();
    }
    mTensors.assign(static_cast<std::size_t>(graph.num_tensors), Tensor{});
    mNamedTensors.clear();
    mAccumulateTensors.clear();
    mCurrentLayer = -1;
    mLastRecomputeLayer = -1;
    mMicroStep = micro_step;
    if (!mPersistedBackwardTensors.empty()) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        for (auto* ptr : mPersistedBackwardTensors) {
            if (ptr) {
                cudaFree(ptr);
            }
        }
        mPersistedBackwardTensors.clear();
    }

    // Clear activation/non-block gradients for each micro-step.
    // When called from GraphExecutor::backward_with_hook(), the caller already zeroes these
    // buffers, so skip_zeroing=true avoids redundant GPU work.
    if (!skip_zeroing) {
        fill_zero(mRunState.non_block_gradients().d_ln_final, mRunState.MainStream);
        if (mRunState.non_block_gradients().d_embeddings.Data && !mRunState.is_lora_only_mode()) {
            fill_zero(mRunState.non_block_gradients().d_embeddings, mRunState.MainStream);
        }
        if (mConfig.NumLayers > 0) {
            if (Tensor* d_res_ffn =
                    block_gradient_ptr(mRunState, static_cast<int>(mConfig.NumLayers) - 1, TensorSlot::BlockDResFFN)) {
                fill_zero(*d_res_ffn, mRunState.MainStream);
            }
        }
        mRunState.zero_activation_gradients(mRunState.MainStream);
    }

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->gather_lm_head(comm, mRunState.MainStream);
        }
    }

    // Prefetch last layer before backward loop (layers processed in reverse)
    mPrefetchDirection = -1;  // Backward traversal
    if (mConfig.NumLayers > 0 && !mCapturing) {
        if (mWeightManager && mWeightManager->needs_block_gather()) {
            const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
            mWeightManager->gather_block(last_layer, comm, mRunState.side_stream());
        }
        // QLoRA offload: prefetch last layer's quantized weights for backward
        if (auto* provider = mWeights.qlora_provider()) {
            if (provider->has_offloading()) {
                const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
                provider->prefetch_for_layer(last_layer, mRunState.side_stream());
            }
        }
    }

    // Save stack checkpoint at start of backward - we'll restore per-layer to manage memory
    auto initial_checkpoint = mRunState.Stack.checkpoint();
    int last_layer_restored = -1;
    auto clear_shared_grads = [&](int layer_idx) {
        // No-op: d_ln2, d_att, d_ln1 are fully overwritten by their respective
        // backward ops before being read. Gradient data flows through mTensors[],
        // not through the pre-allocated simplified_grads buffers. The matmul backward
        // only temporarily remaps these pointers during LoRA hooks, then restores them.
        (void)layer_idx;
    };
    auto prune_stack_tensors = [&](int current_layer) {
        // Prune flat tensor vector using pre-computed metadata (no string parsing)
        for (int id = 0; id < graph.num_tensors; ++id) {
            auto& t = mTensors[static_cast<std::size_t>(id)];
            if (!t.Data) continue;
            const auto& meta = graph.tensor_meta[static_cast<std::size_t>(id)];
            // Skip MoE expert_offsets
            if (meta.is_moe_offsets()) continue;
            // Skip cross-layer gradients for earlier layers
            if (current_layer >= 0 && meta.is_d_blocks() && meta.block_layer_idx >= 0 &&
                meta.block_layer_idx < current_layer)
                continue;
            // Skip saved tensors for earlier layers
            if (current_layer >= 0 && meta.is_blocks() && meta.block_layer_idx >= 0 &&
                meta.block_layer_idx < current_layer)
                continue;
            // Skip tensors with unparseable layer index (be safe)
            if ((meta.is_d_blocks() || meta.is_blocks()) && meta.block_layer_idx < 0) continue;
            if (mRunState.Stack.owns(t.Data) && !mRunState.Stack.is_live(t.Data)) {
                t = Tensor{};
            }
        }
        for (auto it = mNamedTensors.begin(); it != mNamedTensors.end();) {
            const Tensor& t = it->second;
            if (t.Data && mRunState.Stack.owns(t.Data) && !mRunState.Stack.is_live(t.Data)) {
                it = mNamedTensors.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Bind initial gradient tensors (from loss computation)
    // d_logits is stored in the output buffer after loss backward (only when lmhead_chunks == 1)
    auto& output = mRunState.non_block_activations().output;
    if (!output.Data) {
        throw std::runtime_error("CompiledExecutor: output tensor has no data (B=" + std::to_string(mB) +
                                 ", T=" + std::to_string(mT) + ")");
    }

    if (mOptions.LMHeadChunks <= 1) {
        Tensor logits_view = view_tensor(output, {mB, mT, static_cast<long>(mConfig.VocabSize)});
        bind_tensor("d_logits", logits_view);
        // Also provide flattened version for matmul backward ops
        Tensor logits_flat = view_tensor(output, {mB * mT, static_cast<long>(mConfig.VocabSize)});
        if (logits_flat.Rank != 2) {
            throw std::runtime_error(
                "CompiledExecutor: d_logits_flat has wrong rank=" + std::to_string(logits_flat.Rank) + " expected 2");
        }
        bind_tensor("d_logits_flat", logits_flat);
    }

    // Bind gradient output buffers for final layer norm backward
    // DSL-driven: use slot registry to derive all mappings from gradient_of relationships
    Tensor& d_ln_final_buf = mRunState.non_block_gradients().d_ln_final;
    Tensor& d_embeddings_buf = mRunState.non_block_gradients().d_embeddings;

    Tensor d_ln_final_flat = view_tensor(d_ln_final_buf, {mB * mT, static_cast<long>(mConfig.HiddenSize)});

    // Helper: map a `gradient_of` field (the forward tensor this gradient
    // targets) to its persistent buffer. Dispatches on the slot enum so
    // every alias (`xF`/`ln_final`/`xF_flat`, `x0`/`encoded`) resolves in one
    // place. `embeddings` is a DSL gradient_of name for the embedding table's
    // output gradient; it has no slot enum entry so we keep it explicit.
    // `d_xN` / `d_residualN` are intentionally not mapped here — those are
    // the backward-input stubs that get computed on-the-fly.
    auto get_target_buffer = [&](const std::string& grad_of) -> Tensor* {
        switch (resolve_slot_with_flat(grad_of)) {
            case TensorSlot::LNFinal:
            case TensorSlot::FinalResidual: return &d_ln_final_buf;
            case TensorSlot::Encoded: return &d_embeddings_buf;
            default: break;
        }
        if (grad_of == "embeddings") {
            return &d_embeddings_buf;
        }
        return nullptr;
    };

    // Bind global gradient tensors - these are always needed regardless of DSL layout
    // The DSL gradient slots declare shape/dtype but the actual buffers come from RunState
    bind_tensor("d_xF_flat", d_ln_final_flat);
    bind_tensor("d_xF", d_ln_final_buf);
    bind_tensor("d_ln_final", d_ln_final_buf);
    bind_tensor("d_ln_final_flat", d_ln_final_flat);

    // Always bind embedding gradients to the persistent d_embeddings buffer, even in
    // LoRA-only mode. This prevents ensure_output_tensor from stack-allocating them,
    // which would block can_restore_stack for the entire backward pass.
    bind_tensor("d_encoded", d_embeddings_buf);
    bind_tensor("d_x0", d_embeddings_buf);

    // DSL-driven binding for any additional gradient slots declared in the Python model
    if (mSlotRegistry && mSlotRegistry->has_dsl_layout()) {
        mSlotRegistry->for_each([&](const std::string& slot_name, const TensorSlotRegistry::SlotEntry& entry) {
            if (entry.scope != ActivationScope::GlobalGradient) return;
            // Skip if already bound above
            if (mCurrentGraph) {
                int sid = mCurrentGraph->find_tensor_id(slot_name);
                if (sid >= 0 && static_cast<std::size_t>(sid) < mTensors.size() && mTensors[sid].Data) return;
            }

            Tensor* target_buf = get_target_buffer(entry.gradient_of);
            if (target_buf && target_buf->Data) {
                bind_tensor(slot_name, *target_buf);
            }
        });
    }

    // Ensure global block outputs (xN/residualN) map to the last block's gradients.
    // These gradients must survive layer-boundary stack restores in recompute mode.
    // Gemma4 routes the block output through a dedicated h_out slot, so the top
    // of the backward chain must seed d_h_out rather than the MLP/down or
    // residual-attention buffers.
    // Declared outside the if/else chain so the StackedBlocks fallback below
    // can reference them for d_<name> binding.
    Tensor* d_h_out = nullptr;
    Tensor* d_mlp_down = nullptr;
    Tensor* d_res_att = nullptr;
    bool has_h_out_grad = false;
    if (mConfig.NumLayers > 0) {
        const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
        d_h_out = block_gradient_ptr(mRunState, last_layer, TensorSlot::BlockDHOut);
        d_mlp_down = block_gradient_ptr(mRunState, last_layer, TensorSlot::BlockDMLPDown);
        d_res_att = block_gradient_ptr(mRunState, last_layer, TensorSlot::BlockDResAtt);
        has_h_out_grad = d_h_out && d_h_out->Data;
        if (has_h_out_grad) {
            bind_tensor("d_xN", *d_h_out);
            bind_tensor("d_residualN", *d_h_out);
            bind_tensor(fmt::format("d_blocks[{}].h_out", last_layer), *d_h_out);
        } else {
            if (d_mlp_down && d_mlp_down->Data) {
                bind_tensor("d_xN", *d_mlp_down);
            }
            if (d_res_att && d_res_att->Data) {
                bind_tensor("d_residualN", *d_res_att);
            }
        }

        // Heuristic aliasing for non-inlined StackedBlocks outputs
        // (e.g., "StackedBlocks_4" or "HybridStackedBlocks_10").
        if (mSaved) {
            std::vector<std::pair<int, std::string>> stacked;
            stacked.reserve(2);
            auto parse_stacked_output_index = [](const std::string& name, int& idx) -> bool {
                constexpr std::array<const char*, 2> prefixes = {"StackedBlocks_", "HybridStackedBlocks_"};
                for (const char* prefix : prefixes) {
                    if (name.rfind(prefix, 0) != 0) {
                        continue;
                    }
                    const char* s = name.c_str() + std::strlen(prefix);
                    if (!*s) {
                        return false;
                    }
                    char* end = nullptr;
                    long parsed = std::strtol(s, &end, 10);
                    if (end == s) {
                        return false;
                    }
                    idx = static_cast<int>(parsed);
                    return true;
                }
                return false;
            };
            for (const auto& kv : *mSaved) {
                const std::string& name = kv.first;
                int idx = -1;
                if (parse_stacked_output_index(name, idx)) {
                    stacked.emplace_back(idx, name);
                }
            }
            if (!stacked.empty()) {
                std::sort(stacked.begin(), stacked.end(), [](const auto& a, const auto& b) {
                    return a.first < b.first;
                });
                if (has_h_out_grad) {
                    for (const auto& [_, name] : stacked) {
                        bind_tensor("d_" + name, *d_h_out);
                    }
                } else if (stacked.size() == 1) {
                    if (d_res_att && d_res_att->Data) {
                        bind_tensor("d_" + stacked[0].second, *d_res_att);
                    }
                } else {
                    if (d_mlp_down && d_mlp_down->Data) {
                        bind_tensor("d_" + stacked[0].second, *d_mlp_down);
                    }
                    if (d_res_att && d_res_att->Data) {
                        bind_tensor("d_" + stacked[1].second, *d_res_att);
                    }
                }
            }
        }
    }

    // Bind autodiff-generated gradient names (d_embed_1, etc.) from forward embedding outputs.
    // Always bind even in LoRA-only mode to prevent stack-allocation (see d_embeddings comment above).
    for (const auto& emb_out : mEmbeddingOutputs) {
        std::string grad_name = "d_" + emb_out;
        bind_tensor(grad_name, d_embeddings_buf);
    }

    // Restore MoE expert_offsets from persistent CPU storage
    // This is needed by grouped GEMM backward ops for proper token routing
    if (mConfig.NumExperts > 0 && !mMoEExpertOffsetsData.empty()) {
        // Allocate PERSISTENT GPU buffer for expert_offsets (not stack-allocated)
        // This ensures the memory won't be invalidated by stack restores or temp_free calls
        const int num_elements = static_cast<int>(mMoEExpertOffsetsData.size());
        const size_t needed_bytes = num_elements * sizeof(int);

        // Allocate or resize GPU buffer if needed
        if (mMoEExpertOffsetsGPU == nullptr || mMoEExpertOffsetsGPUSize < needed_bytes) {
            if (mMoEExpertOffsetsGPU) {
                CUDA_CHECK(cudaFree(mMoEExpertOffsetsGPU));
            }
            CUDA_CHECK(cudaMalloc(&mMoEExpertOffsetsGPU, needed_bytes));
            mMoEExpertOffsetsGPUSize = needed_bytes;
        }

        // Copy data from CPU to GPU
        CUDA_CHECK(cudaMemcpyAsync(mMoEExpertOffsetsGPU,
                                   mMoEExpertOffsetsData.data(),
                                   needed_bytes,
                                   cudaMemcpyHostToDevice,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));

        // Create tensor wrapper pointing to persistent buffer
        Tensor expert_offsets;
        expert_offsets.DType = ETensorDType::INT32;
        expert_offsets.Rank = 1;
        expert_offsets.Sizes[0] = num_elements;
        expert_offsets.Data = static_cast<std::byte*>(mMoEExpertOffsetsGPU);

        bind_tensor("moe_expert_offsets", expert_offsets);
        // Note: NOT adding to mTemps since this is persistent memory managed separately
    }

    // Also bind standard inputs that backward ops may reference
    bind_tensor("token_ids", mRunState.Inputs);
    bind_tensor("position_ids", mRunState.PositionIDs);
    if (mRunState.VisualPosMasks.Data) {
        bind_tensor("visual_pos_masks", mRunState.VisualPosMasks);
    }
    if (mRunState.VisualEmbeds.Data) {
        bind_tensor("visual_embeds", mRunState.VisualEmbeds);
    }
    if (!mRunState.DeepstackVisualEmbeds.empty()) {
        for (std::size_t i = 0; i < mRunState.DeepstackVisualEmbeds.size(); ++i) {
            if (!mRunState.DeepstackVisualEmbeds[i].Data) {
                continue;
            }
            bind_tensor("deepstack_visual_embeds_" + std::to_string(i), mRunState.DeepstackVisualEmbeds[i]);
        }
    }

    // Build the set of gradients that require accumulation (not the first micro-step).
    // Also bind parameter gradient tensors so they're used instead of temporaries.
    for (const auto& param_name : mGrads.param_names()) {
        if (param_name.find("rope_freqs") != std::string::npos) {
            continue;
        }
        bool accumulate = false;
        Tensor* grad_tensor = mGrads.get_param_grad(param_name, accumulate);
        if (grad_tensor && grad_tensor->Data) {
            std::string grad_name = "d_" + param_name;
            bind_tensor(grad_name, *grad_tensor);
            if (accumulate) {
                mAccumulateTensors.insert(grad_name);
            }
        }
    }

    auto is_grad_ref = [](const TensorRef& ref) -> bool {
        if (!ref.name.empty() && ref.name.size() > 2 && ref.name[0] == 'd' && ref.name[1] == '_') {
            return true;
        }
        switch (ref.slot) {
            case TensorSlot::BlockDLN1:
            case TensorSlot::BlockDQKV:
            case TensorSlot::BlockDAtt:
            case TensorSlot::BlockDSwiGLU:
            case TensorSlot::BlockDMLPUp:
            case TensorSlot::BlockDMLPDown:
            case TensorSlot::BlockDHOut:
            case TensorSlot::BlockDLN2:
            case TensorSlot::BlockDResAtt:
            case TensorSlot::BlockDResFFN:
            case TensorSlot::DLoss: return true;
            default: return false;
        }
    };

    auto ref_layer_idx = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto ref_layer_idx_any = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto op_layer_idx = [&](const CompiledOp& op) -> int {
        int detected_non_grad = -1;
        for (const auto& ref : op.inputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        for (const auto& ref : op.outputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        return (detected_non_grad >= 0) ? detected_non_grad : -1;
    };

    auto op_layer_idx_any = [&](const CompiledOp& op) -> int {
        int detected_any = -1;
        for (const auto& ref : op.inputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        for (const auto& ref : op.outputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        if (op.attrs.layer_idx >= 0) {
            detected_any = std::max(detected_any, op.attrs.layer_idx);
        }
        return detected_any;
    };

    const bool skip_logits_grad = (mOptions.LMHeadChunks > 1);
    auto is_logits_grad_name = [](const std::string& name) {
        return name == "d_logits" || name == "d_logits_flat";
    };
    auto is_logits_grad_op = [&](const CompiledOp& op) {
        for (const auto& ref : op.inputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        for (const auto& ref : op.outputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        return false;
    };

    // Use pre-computed last-use data from graph compilation (avoids rebuilding every backward).
    const auto& last_use_names = graph.last_use_names;
    const auto& last_use = graph.last_use_index;
    std::unordered_map<int, std::byte*> persisted_backward_by_tid;
    std::unordered_map<std::byte*, int> persisted_backward_refcount;
    auto release_persisted_backward_ptr = [&](std::byte* ptr) {
        if (!ptr) {
            return;
        }
        for (auto& [tid, active_ptr] : persisted_backward_by_tid) {
            if (active_ptr == ptr) {
                active_ptr = nullptr;
            }
        }
        for (auto& tensor : mTensors) {
            if (tensor.Data == ptr) {
                tensor = Tensor{};
            }
        }
        for (auto it = mNamedTensors.begin(); it != mNamedTensors.end();) {
            if (it->second.Data == ptr) {
                it = mNamedTensors.erase(it);
            } else {
                ++it;
            }
        }
        // Phase 3 #4: arena-backed pointers live in the bwd_cross_layer arena,
        // not in mPersistedBackwardTensors. Only cudaFree pointers we actually
        // cudaMalloc'd. Nulling the slot in mPersistedBackwardTensors is also
        // the signal that it was cudaMalloc-backed.
        bool was_cudamalloc_backed = false;
        for (auto& active_ptr : mPersistedBackwardTensors) {
            if (active_ptr == ptr) {
                active_ptr = nullptr;
                was_cudamalloc_backed = true;
            }
        }
        if (was_cudamalloc_backed) {
            CUDA_CHECK(cudaFreeAsync(ptr, mRunState.MainStream));
        }
        persisted_backward_refcount.erase(ptr);
    };
    auto release_persisted_backward_tid = [&](int tid) {
        auto it = persisted_backward_by_tid.find(tid);
        if (it == persisted_backward_by_tid.end()) {
            return;
        }
        std::byte* ptr = it->second;
        persisted_backward_by_tid.erase(it);
        if (!ptr) {
            return;
        }
        auto ref_it = persisted_backward_refcount.find(ptr);
        if (ref_it == persisted_backward_refcount.end()) {
            return;
        }
        if (--ref_it->second == 0) {
            release_persisted_backward_ptr(ptr);
        }
    };
    auto prune_by_last_use = [&](std::size_t idx) {
        if (idx >= last_use_names.size()) {
            return;
        }
        for (const auto& name : last_use_names[idx]) {
            if (mCurrentGraph) {
                int tid = mCurrentGraph->find_tensor_id(name);
                if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
                    release_persisted_backward_tid(tid);
                    mTensors[tid] = Tensor{};
                }
            }
            mNamedTensors.erase(name);
        }
    };
    const int num_layers = static_cast<int>(mConfig.NumLayers);
    static const bool debug_replay = std::getenv("SUROGATE_DEBUG_REPLAY") != nullptr;
    const char* op_trace_env = std::getenv("SUROGATE_OP_TRACE");
    const bool op_trace = op_trace_env && std::string(op_trace_env) != "0";
    const char* bwd_filter_env = std::getenv("SUROGATE_DEBUG_BWD_FILTER");
    const std::string bwd_filter = bwd_filter_env ? std::string(bwd_filter_env) : std::string();
    const bool bwd_filter_enabled = !bwd_filter.empty();
    const bool bwd_filter_dump = env_int("SUROGATE_DEBUG_BWD_FILTER_DUMP", 0) != 0;
    const char* op_profile_env = std::getenv("SUROGATE_OP_PROFILE");
    const bool op_profile = op_profile_env && std::string(op_profile_env) != "0";
    const int debug_nonfinite_mode = env_int("SUROGATE_DEBUG_CHECK_NONFINITE", 0);
    const bool debug_nonfinite_backward = (debug_nonfinite_mode & 0x2) != 0;
    // Optional extreme-magnitude threshold: flag when any backward-op output
    // contains a finite value with |x| > this. Useful for finding ops that
    // produce sane-looking (non-NaN) but astronomical gradients.
    const float debug_extreme_threshold = env_float("SUROGATE_DEBUG_BACKWARD_EXTREME", 0.0f);
    const bool debug_extreme_backward = debug_extreme_threshold > 0.0f;
    auto count_nonfinite_of = [&](const Tensor& t) -> int {
        Tensor non_finite_count = mRunState.temp_alloc(ETensorDType::INT32, {1}, "non_finite_count");
        CUDA_CHECK(cudaMemsetAsync(non_finite_count.Data, 0, sizeof(int), mRunState.MainStream));
        count_non_finite(non_finite_count, t, mRunState.MainStream);
        int host_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_count,
                                   non_finite_count.get<int>(),
                                   sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        mRunState.temp_free(non_finite_count);
        return host_count;
    };
    auto count_above_of = [&](const Tensor& t, float threshold) -> int {
        if (t.DType != ETensorDType::BF16 && t.DType != ETensorDType::FP32) return 0;
        Tensor cnt = mRunState.temp_alloc(ETensorDType::INT32, {1}, "count_above_threshold");
        CUDA_CHECK(cudaMemsetAsync(cnt.Data, 0, sizeof(int), mRunState.MainStream));
        const int n = static_cast<int>(t.nelem());
        if (t.DType == ETensorDType::BF16) {
            count_above_threshold(cnt.get<int>(), t.get<nv_bfloat16>(), n, threshold, mRunState.MainStream);
        } else {
            count_above_threshold(cnt.get<int>(), t.get<float>(), n, threshold, mRunState.MainStream);
        }
        int host = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host, cnt.get<int>(), sizeof(int), cudaMemcpyDeviceToHost, mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        mRunState.temp_free(cnt);
        return host;
    };
    auto check_nonfinite_refs = [&](const CompiledOp& op, const std::vector<TensorRef>& refs) {
        if (!debug_nonfinite_backward && !debug_extreme_backward) {
            return;
        }
        for (const auto& ref : refs) {
            if (ref.name.empty()) {
                continue;
            }
            const Tensor* t = try_get_tensor(ref.name);
            if (!t || !t->Data) {
                continue;
            }
            if (t->DType != ETensorDType::BF16 && t->DType != ETensorDType::FP32) {
                continue;
            }

            int host_count = debug_nonfinite_backward ? count_nonfinite_of(*t) : 0;
            int extreme_count = debug_extreme_backward ? count_above_of(*t, debug_extreme_threshold) : 0;

            if (host_count == 0 && extreme_count > 0) {
                // Emit a diagnostic but do NOT throw — extreme magnitudes are
                // often legitimate mid-training. The goal is just to pinpoint
                // the first op in the backward chain that inflates grads.
                fprintf(
                    stderr,
                    "[EXTREME] backward op '%s' (id=%s type=%s): output '%s' has %d values with |x|>%.3g (nelem=%zu)\n",
                    op.op_id.c_str(),
                    op.op_id.c_str(),
                    op_type_to_string(op.type),
                    ref.name.c_str(),
                    extreme_count,
                    debug_extreme_threshold,
                    (size_t)t->nelem());
            }

            if (host_count > 0) {
                // Also report input non-finite counts so we can distinguish
                // upstream-propagated NaN from NaN introduced by this op.
                std::ostringstream inputs_oss;
                for (const auto& in_ref : op.inputs) {
                    if (in_ref.name.empty()) continue;
                    const Tensor* ti = nullptr;
                    if (in_ref.tensor_id >= 0 && static_cast<std::size_t>(in_ref.tensor_id) < mTensors.size() &&
                        mTensors[in_ref.tensor_id].Data) {
                        ti = &mTensors[in_ref.tensor_id];
                    }
                    if (!ti) ti = try_get_tensor(in_ref.name);
                    if (!ti) ti = try_get_tensor_fuzzy(in_ref.name);
                    if (!ti) {
                        // Last resort: try resolve_tensor (may throw for some refs).
                        try {
                            Tensor& r = resolve_tensor(in_ref);
                            ti = &r;
                        } catch (...) {
                            ti = nullptr;
                        }
                    }
                    if (!ti) {
                        inputs_oss << "\n  input '" << in_ref.name << "' <not resolvable>";
                        continue;
                    }
                    if (!ti->Data) {
                        inputs_oss << "\n  input '" << in_ref.name
                                   << "' <no data> dtype=" << static_cast<int>(ti->DType);
                        continue;
                    }
                    if (ti->DType != ETensorDType::BF16 && ti->DType != ETensorDType::FP32) {
                        inputs_oss << "\n  input '" << in_ref.name << "' <skipped dtype=" << static_cast<int>(ti->DType)
                                   << ">";
                        continue;
                    }
                    int ic = count_nonfinite_of(*ti);
                    inputs_oss << "\n  input '" << in_ref.name << "' nonfinite=" << ic
                               << " dtype=" << static_cast<int>(ti->DType);
                }
                std::ostringstream oss;
                oss << "Non-finite detected in backward output tensor '" << ref.name << "' at op id=" << op.op_id
                    << " type=" << op_type_to_string(op.type) << " count=" << host_count
                    << " dtype=" << static_cast<int>(t->DType) << " shape=[";
                for (int d = 0; d < t->Rank; ++d) {
                    if (d > 0) oss << ",";
                    oss << t->Sizes[d];
                }
                oss << "]" << inputs_oss.str();
                throw std::runtime_error(oss.str());
            }
        }
    };
    std::unordered_map<std::string, double> op_profile_total_ms;
    std::unordered_map<std::string, std::size_t> op_profile_counts;
    cudaEvent_t op_profile_start = nullptr;
    cudaEvent_t op_profile_end = nullptr;
    if (op_profile) {
        CUDA_CHECK(cudaEventCreateWithFlags(&op_profile_start, cudaEventDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&op_profile_end, cudaEventDefault));
    }
    cudaStreamCaptureStatus bwd_capture_status = cudaStreamCaptureStatusNone;
    const bool bwd_stream_capturing =
        mCapturing || (cudaStreamIsCapturing(mRunState.MainStream, &bwd_capture_status) == cudaSuccess &&
                       bwd_capture_status != cudaStreamCaptureStatusNone);

    std::vector<std::size_t> layer_start_indices(num_layers, SIZE_MAX);
    std::vector<bool> layer_seen_any(num_layers, false);
    for (const auto& op : graph.ops) {
        if (op.layer_start >= 0 && op.layer_start < num_layers) {
            layer_start_indices[op.layer_start] = &op - graph.ops.data();
        }
    }

    // Build backward tile group lookup for long-context tiled MLP execution.
    // Only include groups that contain MatmulBackward ops (not forward groups).
    std::unordered_map<std::size_t, const MlpTileGroup*> bwd_tile_group_starts;
    for (const auto& tg : graph.mlp_tile_groups) {
        for (std::size_t i = tg.start_op_idx; i <= tg.end_op_idx && i < graph.ops.size(); ++i) {
            if (graph.ops[i].type == CompiledOpType::MatmulBackward) {
                bwd_tile_group_starts[tg.start_op_idx] = &tg;
                break;
            }
        }
    }

    // Stream-driven backward execution (M5.d, flag-gated pass-through mode).
    // Minimal: no tiled MLP, no capture, no recompute, no bwd_filter / watch /
    // op_profile — matches the Llama-only scope of the forward stream path.
    // Phase 4: stream-driven backward default-on. Same SUROGATE_LEGACY_EXECUTOR=1 escape.
    const bool bwd_legacy_escape = [] {
        const char* e = std::getenv("SUROGATE_LEGACY_EXECUTOR");
        return e && std::string(e) == "1";
    }();
    const bool bwd_stream_driven = !bwd_legacy_escape && !graph.instruction_stream.empty() && !bwd_stream_capturing &&
                                   bwd_tile_group_starts.empty();

    // Per-op layer-end cleanup (shared between SegmentDispatch inline and
    // PhaseExit BwdBlock). Mirrors the flat-ops layer-end handler at lines
    // 2767-2895. Idempotent via last_layer_restored - safe to call multiple
    // times for the same layer. `idx` is the op index that triggered the
    // cleanup (used by BwdCrossLayer persist to decide which tids survive).
    auto bwd_layer_end_cleanup = [&](int L, std::size_t idx, bool capturing = false) {
        if (L < 0 || L == last_layer_restored) return;
        // Capture-unsafe work (BwdCrossLayer persist, handle_layer_end,
        // debug dump, grad reduce/offload/notify) is skipped during CUDA
        // stream capture — those APIs invalidate the capture. Stack.restore
        // + slot clears still run so peak memory stays bounded.
        if (!capturing) {
            // BwdCrossLayer persist (Phase 3 #4 arena-backed).
            std::unordered_map<std::uintptr_t, std::size_t> max_bytes_by_ptr;
            struct PendingPersist {
                int tid;
                std::byte* ptr;
                std::size_t bytes;
            };
            std::vector<PendingPersist> pending;
            for (const auto& [name, use_idx] : last_use) {
                if (use_idx <= idx) continue;
                int tid = graph.find_tensor_id(name);
                if (tid < 0 || static_cast<std::size_t>(tid) >= mTensors.size()) continue;
                const auto& t = mTensors[static_cast<std::size_t>(tid)];
                if (t.Data && mRunState.Stack.owns(t.Data)) {
                    pending.push_back({tid, t.Data, t.bytes()});
                    auto key = reinterpret_cast<std::uintptr_t>(t.Data);
                    auto it = max_bytes_by_ptr.find(key);
                    if (it == max_bytes_by_ptr.end())
                        max_bytes_by_ptr.emplace(key, t.bytes());
                    else
                        it->second = std::max(it->second, t.bytes());
                }
            }
            if (!pending.empty()) {
                if (env_enabled("SUROGATE_DEBUG_BACKWARD_PERSIST")) {
                    std::size_t total_bytes = 0;
                    for (const auto& [ptr_key, nbytes] : max_bytes_by_ptr)
                        total_bytes += nbytes;
                    std::cerr << "[BWD_PERSIST] layer=" << L << " unique_ptrs=" << max_bytes_by_ptr.size()
                              << " total_bytes=" << total_bytes << std::endl;
                }
                std::unordered_map<std::uintptr_t, std::byte*> persistent_by_ptr;
                persistent_by_ptr.reserve(max_bytes_by_ptr.size());
                for (const auto& [ptr_key, nbytes] : max_bytes_by_ptr) {
                    if (nbytes == 0) continue;
                    auto a = allocate_bwd_cross_layer(nbytes);
                    auto* original = reinterpret_cast<const std::byte*>(ptr_key);
                    CUDA_CHECK(
                        cudaMemcpyAsync(a.ptr, original, nbytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
                    persistent_by_ptr.emplace(ptr_key, a.ptr);
                    if (!a.arena_backed) mPersistedBackwardTensors.push_back(a.ptr);
                }
                for (const auto& p : pending) {
                    auto it = persistent_by_ptr.find(reinterpret_cast<std::uintptr_t>(p.ptr));
                    if (it == persistent_by_ptr.end()) continue;
                    release_persisted_backward_tid(p.tid);
                    persisted_backward_by_tid[p.tid] = it->second;
                    persisted_backward_refcount[it->second] += 1;
                    mTensors[static_cast<std::size_t>(p.tid)].Data = it->second;
                }
                for (auto& [name, tensor] : mNamedTensors) {
                    if (!tensor.Data) continue;
                    auto it = persistent_by_ptr.find(reinterpret_cast<std::uintptr_t>(tensor.Data));
                    if (it != persistent_by_ptr.end()) tensor.Data = it->second;
                }
            }

            handle_layer_end(L);
            if (mDebugDumpBackwardLayerFn) mDebugDumpBackwardLayerFn(L);

            if (mGrads.is_streaming_grads()) {
                if (mComm && mComm->world_size() > 1) {
                    CUDA_CHECK(cudaEventRecord(mRunState.side_stream_event(), mRunState.MainStream));
                    CUDA_CHECK(cudaStreamWaitEvent(mRunState.side_stream(), mRunState.side_stream_event(), 0));
                    mGrads.reduce_layer_grads(L, mRunState.side_stream(), *mComm);
                }
                mGrads.offload_layer_grads(L, mRunState.MainStream, mRunState.side_stream());
            } else if (mComm && mComm->world_size() > 1) {
                CUDA_CHECK(cudaEventRecord(mRunState.side_stream_event(), mRunState.MainStream));
                CUDA_CHECK(cudaStreamWaitEvent(mRunState.side_stream(), mRunState.side_stream_event(), 0));
                mGrads.notify_block(L, mRunState.side_stream(), *mComm);
            }
        }

        mRunState.Stack.restore(initial_checkpoint);
        mTemps.clear();
        prune_stack_tensors(L);
        if (rstd_on_stack_enabled()) clear_rstd_stack_slots(mRunState, L);
        if (mRunState.ffn_temps_on_stack()) clear_ffn_temp_stack_slots(mRunState, L);
        if (mRunState.large_bwd_temps_on_stack()) clear_large_bwd_grad_stack_slots(mRunState, L);
        last_layer_restored = L;
    };
    if (bwd_stream_driven) {
        if (const char* env = std::getenv("SUROGATE_DEBUG_PHASE_INTERPRETER")) {
            if (std::string(env) == "1") {
                std::cerr << "[phase-interp] backward: stream-driven (" << graph.instruction_stream.size()
                          << " instructions)\n";
            }
        }
        for (const auto& inst : graph.instruction_stream) {
            switch (inst.kind) {
                case dsl::InstKind::PhaseEnter:
                    if (inst.phase_kind == dsl::PhaseKind::BwdBlock && inst.block_index >= 0) {
                        const int L = inst.block_index;
                        handle_layer_start(L);
                        if (mGrads.is_streaming_grads()) {
                            mGrads.prepare_layer_grads(L, mRunState.MainStream);
                            for (const auto& pname : mGrads.layer_grad_names(L)) {
                                std::string gname = "d_" + pname;
                                bool dummy = false;
                                Tensor* grad = mGrads.get_param_grad(pname, dummy);
                                if (grad) bind_tensor(gname, *grad);
                            }
                        }
                        if (L < num_layers && !layer_seen_any[static_cast<std::size_t>(L)]) {
                            clear_shared_grads(L);
                            layer_seen_any[static_cast<std::size_t>(L)] = true;
                        }
                        // Phase 3 subsystem #7 flip: recompute moved out of
                        // PhaseEnter into the explicit RecomputeBlock
                        // instruction emitted immediately after.
                    }
                    break;

                case dsl::InstKind::RecomputeBlock:
                    // Phase 3 subsystem #7: explicit forward-block replay for
                    // gradient-checkpointed backward. No-op when recompute is
                    // off. Idempotent per block via mLastRecomputeLayer.
                    if (mRecomputeEnabled && mRecomputeFn && inst.block_index >= 0 &&
                        inst.block_index != mLastRecomputeLayer) {
                        mRecomputeFn(inst.block_index, mB, mT, mRecomputeUseGraphs);
                        mLastRecomputeLayer = inst.block_index;
                    }
                    break;

                case dsl::InstKind::SegmentDispatch:
                    for (std::size_t i = inst.op_start; i < inst.op_end; ++i) {
                        const auto& op = graph.ops[i];
                        if (skip_logits_grad && is_logits_grad_op(op)) continue;
                        if (!op.fn) continue;

                        // Per-op recompute trigger (MoE fix): mirrors the
                        // flat-ops backward at lines 2600-2621. Some backward
                        // ops live OUTSIDE any BwdBlock phase-tree bucket
                        // (e.g., GPT-OSS's LM-head + MoE EP dispatch path in
                        // the backward Prologue), so a PhaseEnter-only
                        // recompute trigger misses them. Detect the op's
                        // effective layer from its inputs/outputs and fire
                        // recompute there. Idempotent via mLastRecomputeLayer.
                        if (mRecomputeEnabled && mRecomputeFn) {
                            const int non_grad_layer = op_layer_idx(op);
                            const int any_layer = op_layer_idx_any(op);
                            const int effective_layer_idx = (non_grad_layer >= 0) ? non_grad_layer : any_layer;
                            if (effective_layer_idx >= 0 && effective_layer_idx != mLastRecomputeLayer) {
                                if (effective_layer_idx < num_layers &&
                                    !layer_seen_any[static_cast<std::size_t>(effective_layer_idx)]) {
                                    clear_shared_grads(effective_layer_idx);
                                    layer_seen_any[static_cast<std::size_t>(effective_layer_idx)] = true;
                                }
                                mRecomputeFn(effective_layer_idx, mB, mT, mRecomputeUseGraphs);
                                mLastRecomputeLayer = effective_layer_idx;
                            }
                        }

                        op.fn(*this, op, static_cast<const void*>(hook));
                        // LM-head post-dispatch cleanup: free the d_logits payload
                        // after the first matmul_backward (see line ~2372 comment).
                        if (op.type == CompiledOpType::MatmulBackward && i == 1) {
                            mRunState.temp_free(mRunState.non_block_activations().output);
                            mTemps.clear();
                            initial_checkpoint = mRunState.Stack.checkpoint();
                        }
                        // Phase 3 subsystem #5 revised: inline prune per op to
                        // free Stack slots as soon as a tid's last use passes.
                        // Batching pruning to the PruneByLastUse instruction
                        // caused a 3x+ Stack bloat on Prologue-heavy graphs
                        // (MoE, hybrid Mamba) because hundreds of Prologue ops
                        // accumulate before any gets freed.
                        prune_by_last_use(i);

                        // Per-op layer-end cleanup. Flat-ops does this after
                        // each op with op.layer_end set (line 2767). Without
                        // it, recompute-allocated forward activations
                        // accumulate on Stack across Prologue ops, producing
                        // 3x+ peak Stack usage on hybrid architectures. The
                        // lambda is idempotent via last_layer_restored.
                        if (op.layer_end >= 0) {
                            bwd_layer_end_cleanup(op.layer_end, i);
                        }
                    }
                    break;

                case dsl::InstKind::PruneByLastUse:
                    // Phase 3 subsystem #5: redundant with the inline
                    // prune_by_last_use above; kept as a no-op so the
                    // instruction stream shape remains stable for
                    // validation / M5.c's static validator.
                    break;

                case dsl::InstKind::PhaseExit:
                    // Phase-level cleanup (safety net - most BwdBlocks are
                    // already drained by the inline per-op bwd_layer_end_cleanup
                    // above when ops inside hit op.layer_end). Idempotent via
                    // last_layer_restored, so the duplicate call is a no-op.
                    if (inst.phase_kind == dsl::PhaseKind::BwdBlock && inst.block_index >= 0) {
                        const std::size_t block_end =
                            (inst.block_index < static_cast<int>(graph.layer_end_indices.size()))
                                ? graph.layer_end_indices[static_cast<std::size_t>(inst.block_index)]
                                : graph.ops.size();
                        const std::size_t idx = block_end > 0 ? block_end - 1 : 0;
                        bwd_layer_end_cleanup(inst.block_index, idx);
                    }
                    break;
            }
        }
    }

    for (std::size_t idx = 0; !bwd_stream_driven && idx < graph.ops.size(); ++idx) {
        const auto& op = graph.ops[idx];
        const int op_layer_any = op_layer_idx_any(op);
        if (skip_logits_grad && is_logits_grad_op(op)) {
            continue;
        }

        bool bwd_filter_matched = false;
        if (bwd_filter_enabled) {
            for (const auto& ref : op.inputs) {
                if (ref.name.find(bwd_filter) != std::string::npos) {
                    bwd_filter_matched = true;
                    break;
                }
            }
            if (!bwd_filter_matched) {
                for (const auto& ref : op.outputs) {
                    if (ref.name.find(bwd_filter) != std::string::npos) {
                        bwd_filter_matched = true;
                        break;
                    }
                }
            }
            if (bwd_filter_matched) {
                auto dump_refs = [](const std::vector<TensorRef>& refs) {
                    std::ostringstream oss;
                    for (std::size_t i = 0; i < refs.size(); ++i) {
                        if (i > 0) oss << ", ";
                        const auto& ref = refs[i];
                        oss << ref.name << "{slot=" << static_cast<int>(ref.slot) << ",layer=" << ref.layer_idx
                            << ",tid=" << ref.tensor_id << "}";
                    }
                    return oss.str();
                };
                std::cerr << "[BWD_FILTER] idx=" << idx << " op_id=" << op.op_id
                          << " type=" << op_type_to_string(op.type) << " layer_start=" << op.layer_start
                          << " layer_end=" << op.layer_end << " inputs=[" << dump_refs(op.inputs) << "]"
                          << " outputs=[" << dump_refs(op.outputs) << "]" << std::endl;
            }
        }

        // Check if this op starts a backward tiled MLP group
        if (!bwd_tile_group_starts.empty()) {
            auto tg_it = bwd_tile_group_starts.find(idx);
            if (tg_it != bwd_tile_group_starts.end()) {
                const auto& tg = *tg_it->second;
                // Handle layer start/recompute if the first op has one
                const auto& first_op = graph.ops[tg.start_op_idx];
                if (first_op.layer_start >= 0) {
                    handle_layer_start(first_op.layer_start);
                    if (mRecomputeEnabled && mRecomputeFn) {
                        const int layer_idx = first_op.layer_start;
                        if (layer_idx >= 0 && layer_idx != mLastRecomputeLayer) {
                            if (layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(layer_idx)]) {
                                clear_shared_grads(layer_idx);
                                layer_seen_any[static_cast<std::size_t>(layer_idx)] = true;
                            }
                            mRecomputeFn(layer_idx, mB, mT, mRecomputeUseGraphs);
                            mLastRecomputeLayer = layer_idx;
                        }
                    }
                }
                execute_tiled_mlp_backward(graph, tg, mB, mT, hook);
                // Prune tensors for all ops in the group
                for (std::size_t gi = tg.start_op_idx; gi <= tg.end_op_idx; ++gi) {
                    prune_by_last_use(gi);
                }
                idx = tg.end_op_idx;
                continue;
            }
        }

        if (op_profile) {
            CUDA_CHECK(cudaEventRecord(op_profile_start, mRunState.MainStream));
        }

        if (op.layer_start >= 0) {
            handle_layer_start(op.layer_start);

            // CPU-RAM centric: bind rotating GPU gradient buffer for this layer
            if (mGrads.is_streaming_grads()) {
                mGrads.prepare_layer_grads(op.layer_start, mRunState.MainStream);
                // Rebind gradient tensors in compiled executor (updates mTensors + mNamedTensors)
                for (const auto& pname : mGrads.layer_grad_names(op.layer_start)) {
                    std::string gname = "d_" + pname;
                    bool dummy = false;
                    Tensor* grad = mGrads.get_param_grad(pname, dummy);
                    if (grad) bind_tensor(gname, *grad);
                }
            }

            if (mRecomputeEnabled && mRecomputeFn) {
                const int layer_idx = op.layer_start;
                if (layer_idx >= 0 && layer_idx != mLastRecomputeLayer) {
                    if (debug_replay) {
                        fprintf(stderr,
                                "[BWD] layer_start=%d for op %zu type=%s\n",
                                layer_idx,
                                idx,
                                op_type_to_string(op.type));
                    }
                    if (layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(layer_idx)]) {
                        clear_shared_grads(layer_idx);
                        layer_seen_any[static_cast<std::size_t>(layer_idx)] = true;
                    }
                    mRecomputeFn(layer_idx, mB, mT, mRecomputeUseGraphs);
                    mLastRecomputeLayer = layer_idx;
                }
            }

            // Note: backward always runs through the normal dispatch loop (no segment
            // graph shortcut) because backward tensor lifetime management (cross-layer
            // persistence, prune_by_last_use, deferred recompute checkpoints) is too
            // complex to replicate in the segmented path. The forward segmented path
            // provides the graph capture benefit; backward runs fully eager.
        }

        if (mRecomputeEnabled && mRecomputeFn) {
            const int layer_idx = op_layer_idx(op);
            const int layer_idx_any = op_layer_idx_any(op);
            const int effective_layer_idx = (layer_idx >= 0) ? layer_idx : layer_idx_any;
            if (effective_layer_idx >= 0 && effective_layer_idx != mLastRecomputeLayer) {
                if (debug_replay) {
                    fprintf(stderr,
                            "[BWD] op_layer_detect=%d (non_grad=%d any=%d) for op %zu type=%s\n",
                            effective_layer_idx,
                            layer_idx,
                            layer_idx_any,
                            idx,
                            op_type_to_string(op.type));
                }
                if (effective_layer_idx < num_layers &&
                    !layer_seen_any[static_cast<std::size_t>(effective_layer_idx)]) {
                    clear_shared_grads(effective_layer_idx);
                    layer_seen_any[static_cast<std::size_t>(effective_layer_idx)] = true;
                }
                mRecomputeFn(effective_layer_idx, mB, mT, mRecomputeUseGraphs);
                mLastRecomputeLayer = effective_layer_idx;
            }
        }

        try {
            // Phase 2a: dispatch via the function pointer baked into
            // op.fn at backward-graph compile time. One indirect call,
            // no switch.
            if (!op.fn) {
                std::ostringstream oss;
                oss << "CompiledExecutor: no dispatch fn for backward op at idx " << idx
                    << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id << ")";
                throw std::runtime_error(oss.str());
            }
            op.fn(*this, op, static_cast<const void*>(hook));
            if (bwd_filter_matched && bwd_filter_dump && mDebugDumpFn) {
                std::vector<std::string> dump_names;
                dump_names.reserve(op.inputs.size() + op.outputs.size());
                for (const auto& ref : op.inputs) {
                    if (!ref.name.empty()) {
                        dump_names.push_back(ref.name);
                    }
                }
                for (const auto& ref : op.outputs) {
                    if (!ref.name.empty()) {
                        dump_names.push_back(ref.name);
                    }
                }
                mDebugDumpFn(dump_names, op_layer_any);
            }
            if (bwd_filter_matched) {
                auto log_live_refs = [&](const char* label, const std::vector<TensorRef>& refs) {
                    std::ostringstream oss;
                    for (std::size_t ri = 0; ri < refs.size(); ++ri) {
                        if (ri > 0) {
                            oss << ", ";
                        }
                        const auto& ref = refs[ri];
                        const Tensor* t = nullptr;
                        if (ref.tensor_id >= 0 && static_cast<std::size_t>(ref.tensor_id) < mTensors.size() &&
                            mTensors[static_cast<std::size_t>(ref.tensor_id)].Data) {
                            t = &mTensors[static_cast<std::size_t>(ref.tensor_id)];
                        }
                        if (!t) {
                            t = try_get_tensor(ref.name);
                        }
                        if (!t) {
                            t = try_get_tensor_fuzzy(ref.name);
                        }
                        oss << ref.name << "{tid=" << ref.tensor_id;
                        if (t && t->Data) {
                            oss << ",ptr=" << t->Data << ",dtype=" << dtype_to_str(t->DType) << ",shape=[";
                            for (int di = 0; di < t->Rank; ++di) {
                                if (di > 0) {
                                    oss << ",";
                                }
                                oss << t->Sizes[di];
                            }
                            oss << "]";
                        } else {
                            oss << ",missing";
                        }
                        oss << "}";
                    }
                    std::cerr << "[BWD_FILTER_LIVE] idx=" << idx << " op_id=" << op.op_id << " label=" << label
                              << " refs=[" << oss.str() << "]" << std::endl;
                };
                log_live_refs("inputs", op.inputs);
                log_live_refs("outputs", op.outputs);
            }

            // Post-dispatch side effect: after the first matmul_backward
            // (LM-head backward) free the output tensor to reclaim the
            // ~1.2 GB of stack memory its d_logits payload occupies.
            // This depends on local backward-loop state
            // (initial_checkpoint, mTemps) and cannot live inside the
            // dispatch function, so we key it off op.type here.
            if (op.type == CompiledOpType::MatmulBackward && idx == 1) {
                mRunState.temp_free(mRunState.non_block_activations().output);
                mTemps.clear();
                initial_checkpoint = mRunState.Stack.checkpoint();
            }
            check_nonfinite_refs(op, op.outputs);

            // Per-op CUDA error check in trace mode: detects illegal memory accesses from launched kernels
            if (op_trace) {
                auto op_err = cudaDeviceSynchronize();
                if (op_err != cudaSuccess) {
                    std::ostringstream oss;
                    oss << "CompiledExecutor backward op " << idx << " (type=" << op_type_to_string(op.type)
                        << ", id=" << op.op_id << "): CUDA error after execution: " << cudaGetErrorString(op_err);
                    throw std::runtime_error(oss.str());
                }
            }

            if (op_profile) {
                CUDA_CHECK(cudaEventRecord(op_profile_end, mRunState.MainStream));
                CUDA_CHECK(cudaEventSynchronize(op_profile_end));
                float elapsed_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, op_profile_start, op_profile_end));
                const std::string op_name = op_type_to_string(op.type);
                op_profile_total_ms[op_name] += static_cast<double>(elapsed_ms);
                op_profile_counts[op_name] += 1;
            }

            // Memory management - prune tensors after last use, then restore stack at layer boundaries.
            // If live cross-layer tensors exist on the stack, persist them to allocated memory first.
            prune_by_last_use(idx);
            if (op.layer_end >= 0 && op.layer_end != last_layer_restored) {
                bwd_layer_end_cleanup(op.layer_end, idx, bwd_stream_capturing);
            }
            // Every N ops as fallback (catches non-annotated layers)
            // NOTE: When recompute is disabled, we cannot aggressively prune tensors because
            // the backward graph may reference intermediate tensors (like d_blocks[N].view_K)
            // that were produced earlier but are still needed. The stack restore + prune
            // would remove these tensors from mTensors, causing "tensor not found" errors.
            // For now, skip periodic cleanup when recompute is disabled to preserve correctness.
            // Memory usage will be higher but the backward pass will complete successfully.

            // After each backward op, check for CUDA errors (lightweight, non-blocking)
            {
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::ostringstream oss2;
                    oss2 << "CompiledExecutor backward op " << idx << " (type=" << op_type_to_string(op.type)
                         << ", id=" << op.op_id << "): CUDA error: " << cudaGetErrorString(err);
                    throw std::runtime_error(oss2.str());
                }
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor backward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            // Add inputs/outputs for debugging
            oss << "\n  inputs: [";
            for (size_t i = 0; i < op.inputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.inputs[i].name << "(slot=" << static_cast<int>(op.inputs[i].slot) << ")";
            }
            oss << "]";
            oss << "\n  outputs: [";
            for (size_t i = 0; i < op.outputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.outputs[i].name << "(slot=" << static_cast<int>(op.outputs[i].slot) << ")";
            }
            oss << "]";
            throw std::runtime_error(oss.str());
        }
    }

    if (op_profile && !op_profile_total_ms.empty()) {
        std::vector<std::pair<std::string, double>> totals(op_profile_total_ms.begin(), op_profile_total_ms.end());
        std::sort(totals.begin(), totals.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::cerr << "[OP PROFILE][backward] totals:\n";
        for (const auto& [name, total_ms] : totals) {
            const std::size_t count = op_profile_counts[name];
            const double avg_ms = count > 0 ? (total_ms / static_cast<double>(count)) : 0.0;
            std::cerr << "  " << name << " total=" << total_ms << "ms" << " count=" << count << " avg=" << avg_ms
                      << "ms\n";
        }
    }
    if (op_profile) {
        if (op_profile_start) cudaEventDestroy(op_profile_start);
        if (op_profile_end) cudaEventDestroy(op_profile_end);
    }

    // Restore any deferred replay checkpoint before final cleanup
    if (mHasDeferredReplayCheckpoint) {
        mRunState.Stack.restore(mDeferredReplayCheckpoint);
        if (mTemps.size() > mDeferredReplayTempMark) {
            mTemps.resize(mDeferredReplayTempMark);
        }
        mHasDeferredReplayCheckpoint = false;
    }

    clear_replay_copied_refs();
    // Free persistent copies from last replay
    for (void* ptr : mReplayCopiedBuffers) {
        cudaFreeAsync(ptr, mRunState.MainStream);
    }
    mReplayCopiedBuffers.clear();

    // Final cleanup - pass -1 to allow full pruning (backward complete)
    mRunState.Stack.restore(initial_checkpoint);
    prune_stack_tensors(-1);
    mTemps.clear();

    // Free persisted cross-layer backward tensors
    // Sync main stream first to ensure all consumers have completed
    if (!mPersistedBackwardTensors.empty()) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        for (auto* ptr : mPersistedBackwardTensors) {
            if (ptr) {
                cudaFree(ptr);
            }
        }
        mPersistedBackwardTensors.clear();
    }

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_final_norm(mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->release_lm_head(mRunState.MainStream);
        }
    }

    // CPU-RAM centric: offload non-block gradients (embedding, lm_head, final_norm) to CPU
    if (mGrads.is_streaming_grads()) {
        mGrads.offload_non_block_grads(mRunState.MainStream);
    }
}

}  // namespace dsl
