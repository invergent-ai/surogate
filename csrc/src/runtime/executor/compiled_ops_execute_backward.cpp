// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CompiledExecutor backward execution.
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

bool is_moe_ep_sync_boundary(CompiledOpType type) {
    switch (type) {
        case CompiledOpType::MoETopK:
        case CompiledOpType::MoEPermute:
        case CompiledOpType::MoEGroupedGemm:
        case CompiledOpType::MoEGroupedGemmGateUp:
        case CompiledOpType::MoEGroupedGemmDown:
        case CompiledOpType::MoEUnpermute:
        case CompiledOpType::MoEExpertBiasAdd:
        case CompiledOpType::MoETopKBackward:
        case CompiledOpType::MoEPermuteBackward:
        case CompiledOpType::MoEGroupedGemmBackward:
        case CompiledOpType::MoEGroupedGemmGateUpBackward:
        case CompiledOpType::MoEGroupedGemmDownBackward:
        case CompiledOpType::MoEUnpermuteBackward:
        case CompiledOpType::MoEExpertBiasAddBackward:
        case CompiledOpType::EpDispatch:
        case CompiledOpType::EpCombine:
        case CompiledOpType::EpDispatchBackward:
        case CompiledOpType::EpCombineBackward: return true;
        default: return false;
    }
}

// Clear-.Data helpers for Stack-backed gradient slots that Stack.restore
// invalidates across layer boundaries. block_gradient_ptr routes tid-first
// so the clear lands in mTensors[tid].
void clear_large_bwd_grad_stack_slots(dsl::DslRunState& rs, int L) {
    auto clear = [&](TensorSlot s) {
        if (Tensor* t = block_gradient_ptr(rs, L, s)) t->Data = nullptr;
    };
    clear(TensorSlot::BlockDQKV);
    clear(TensorSlot::BlockDMLPUp);
    clear(TensorSlot::BlockDSwiGLU);
}

}  // namespace

void CompiledExecutor::initialize_backward_execution(const CompiledGraph& graph,
                                                     NCCLCommunicator& comm,
                                                     int micro_step) {
    mComm = &comm;
    mCurrentGraph = &graph;
    mTemps.clear();
    // Reset per-step bump cursor for bwd_cross_layer arena. Free any
    // cudaMalloc fallbacks from the previous backward call — arena-backed
    // pointers auto-reset via the bump cursor, but fallback allocations
    // must be explicitly released.
    mBwdCrossLayerBumpOffset = 0;
    mBwdCrossLayerCurrentFallbackBytes = 0;
    mBwdCrossLayerCurrentLiveBytes = 0;
    mBwdCrossLayerFreeBlocks.clear();
    mBwdCrossLayerAllocBytes.clear();
    // cudaFree during stream capture invalidates the capture, so defer
    // the fallback-cleanup to the next non-capturing entry. Fallbacks
    // only ever get appended in the bwd_cross_layer-arena overflow path
    // (eager-only — that allocator throws under capture), so it's safe
    // to skip the free loop entirely here. The vector is cleared
    // unconditionally so the bump cursor reset stays consistent.
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(mRunState.MainStream, &cap_status);
    const bool capturing = (cap_status != cudaStreamCaptureStatusNone);
    if (!capturing) {
        for (std::byte* ptr : mBwdCrossLayerFallbacks) {
            if (ptr) {
                (void)cudaFree(ptr);
            }
        }
        mBwdCrossLayerFallbacks.clear();
    }
    // Under capture: leave any existing fallback pointers in the
    // vector. They'll be freed on the next non-capturing entry.

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

    // Register as active executor. Forward activations reach backward via
    // (a) mSaved pre-bind at backward entry for SaveForBwd tids,
    // (b) replay_layer_forward for recompute tids, or (c) the snapshot/
    // restore block below for FwdStack arena-resident tids.
    mRunState.set_active_executor(this);
}

void CompiledExecutor::restore_forward_snapshot_for_backward(const CompiledGraph& graph) {
    // Restore forward's end-state mTensors for FwdStack/SaveForBwd
    // arena-resident tids. Restricts to arena-backed activations
    // (exclude Stack-backed temps like ones/zeros/views whose Stack
    // pointer would trigger `Stack.owns(t.Data)` at
    // bwd_layer_end_cleanup's cross-layer persist, blowing the 64 MiB
    // BwdCrossLayer arena budget). Arena-backed tids have stable
    // cross-phase pointers, enabling the resolve_tensor tid-cache fast
    // path for backward-consumed forward activations.
    if (mForwardTensorsSnapshot.empty() || !mForwardGraph || !mPhaseArenas) {
        return;
    }
    const std::size_t n =
        std::min({mForwardTensorsSnapshot.size(), mTensors.size(), mForwardGraph->tensor_meta.size()});
    std::byte* fwd_lo = mPhaseArenas->fwd_stack_ptr;
    std::byte* fwd_hi = fwd_lo + mPhaseArenas->fwd_stack_bytes;
    std::byte* save_lo = mPhaseArenas->save_for_bwd_ptr;
    std::byte* save_hi = save_lo + mPhaseArenas->save_for_bwd_bytes;
    const bool has_fwd = fwd_lo && mPhaseArenas->fwd_stack_bytes > 0;
    const bool has_save = save_lo && mPhaseArenas->save_for_bwd_bytes > 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& meta = mForwardGraph->tensor_meta[i];
        std::byte* data = mForwardTensorsSnapshot[i].Data;
        if (!data) continue;
        // Only restore arena-resident bindings: Stack-owned pointers would
        // trigger bwd_layer_end_cleanup's cross-layer persist (Stack.owns
        // → true), which allocates per-tid copies into the 64 MiB
        // BwdCrossLayer arena and overflows on Q3.5's hybrid blocks.
        // FwdStack-region tids live in fwd_stack arena; under no-recompute,
        // finalize_save_for_bwd promotes some to SaveForBwd, whose bindings
        // live in save_for_bwd arena.
        const bool in_fwd = has_fwd && meta.region == dsl::RegionKind::FwdStack && data >= fwd_lo && data < fwd_hi;
        const bool in_save =
            has_save && meta.region == dsl::RegionKind::SaveForBwd && data >= save_lo && data < save_hi;
        // Cross-layer globals (e.g. Gemma4 per_layer_inputs) aren't in
        // either arena at runtime — their data is stack-temp-alloc'd at
        // forward time — but they must still be visible to backward
        // resolution. The snapshot carries the pointer (which remains
        // valid under the unified_stack-rebase regime); restore it.
        const bool is_clg = meta.cross_layer_global && mRunState.Stack.owns(data);
        // Block-scope FwdStack tids whose Data falls outside the
        // narrow fwd_stack_ptr sub-range can still need restoration.
        // Examples in Gemma4: compiler-synthesized view/narrow outputs
        // (`pli_flat`, `pli_narrow_layer*`) whose ensure_output_tensor
        // slow path landed them in Stack temps outside the arena's
        // tight window. Without restore, mTensors[tid] stays as
        // Tensor{} after execute_backward's assign-clear, and
        // micro-step N+1's backward (which runs C++ but no preceding
        // C++ forward under full-step capture — graph replay skips
        // host dispatch) reads wrong-shape zero/stale data. Snapshot
        // carries the correct metadata from the forward that last
        // ran through host code.
        const bool fwd_region = meta.region == dsl::RegionKind::FwdStack && data != nullptr;
        if (!in_fwd && !in_save && !is_clg && !fwd_region) continue;
        mTensors[i] = mForwardTensorsSnapshot[i];
    }
    (void)graph;
}

void CompiledExecutor::bind_saved_tensors_for_backward() {
    // Pre-populate mTensors for every saved-source tid. mSaved was filled
    // by persist_saved_layer_tensors at forward's layer_end; its entries
    // hold the arena-backed (SaveForBwd / moe_saved) Tensors that backward
    // ops need. The backward graph's Saved refs use the stripped name
    // (`blocks[L].x` — no `saved.` prefix) as their tid key, so a
    // single bind_tensor per saved entry routes every subsequent
    // resolve_tensor through the cached mTensors[tid] path without
    // touching the mSaved hashmap on the hot path. Metadata-only entries
    // (Data == nullptr) stay on the slow path — try_resolve_saved_live
    // handles them.
    if (!mSaved) {
        return;
    }
    for (const auto& kv : *mSaved) {
        if (!kv.second.Data) continue;
        bind_tensor(kv.first, kv.second);
    }
}

void CompiledExecutor::zero_backward_entry_gradients(bool skip_zeroing) {
    // Clear activation/non-block gradients for each micro-step.
    // When the graph executor has already zeroed these buffers, skip_zeroing=true
    // avoids redundant GPU work.
    if (skip_zeroing) {
        return;
    }
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

void CompiledExecutor::gather_backward_non_block_weights(NCCLCommunicator& comm) {
    if (!needs_non_block_weight_transfer() || !mExecutionRequest) {
        return;
    }
    for (const auto& group : mExecutionRequest->gather_before_backward_weight_groups) {
        mWeightManager->gather_non_block_group(group, comm, mRunState.MainStream);
    }
}

void CompiledExecutor::prefetch_backward_last_layer(NCCLCommunicator& comm) {
    // Prefetch last layer before backward loop (layers processed in reverse).
    mPrefetchDirection = -1;  // Backward traversal
    if (mConfig.NumLayers <= 0 || mCapturing) {
        return;
    }
    if (mWeightManager && mWeightManager->needs_block_gather()) {
        const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
        mWeightManager->gather_block(last_layer, comm, mRunState.side_stream());
    }
    // QLoRA offload: prefetch last layer's quantized weights for backward.
    if (auto* provider = mWeights.qlora_provider()) {
        if (provider->has_offloading()) {
            const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
            provider->prefetch_for_layer(last_layer, mRunState.side_stream());
        }
    }
}

void CompiledExecutor::release_backward_non_block_weights() {
    if (!needs_non_block_weight_transfer() || !mExecutionRequest) {
        return;
    }
    for (const auto& group : mExecutionRequest->release_after_backward_weight_groups) {
        mWeightManager->release_non_block_group(group, mRunState.MainStream);
    }
}

void CompiledExecutor::bind_backward_entry_gradient_tensors() {
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
}

void CompiledExecutor::restore_moe_expert_offsets_for_backward() {
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
}

void CompiledExecutor::bind_param_gradient_tensors_for_backward() {
    // Build the set of gradients that require accumulation (not the first micro-step).
    // Also bind parameter gradient tensors so they're used instead of temporaries.
    for (const auto& param_name : mGrads.param_names()) {
        if (tensor_role_is_rope_name(param_name)) {
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
}

void CompiledExecutor::report_backward_op_profile(const std::unordered_map<std::string, double>& totals_by_op,
                                                  const std::unordered_map<std::string, std::size_t>& counts_by_op,
                                                  cudaEvent_t start_event,
                                                  cudaEvent_t end_event) {
    if (!totals_by_op.empty()) {
        std::vector<std::pair<std::string, double>> totals(totals_by_op.begin(), totals_by_op.end());
        std::sort(totals.begin(), totals.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::cerr << "[OP PROFILE][backward] totals:\n";
        for (const auto& [name, total_ms] : totals) {
            const auto count_it = counts_by_op.find(name);
            const std::size_t count = count_it == counts_by_op.end() ? 0 : count_it->second;
            const double avg_ms = count > 0 ? (total_ms / static_cast<double>(count)) : 0.0;
            std::cerr << "  " << name << " total=" << total_ms << "ms" << " count=" << count << " avg=" << avg_ms
                      << "ms\n";
        }
    }
    if (start_event) cudaEventDestroy(start_event);
    if (end_event) cudaEventDestroy(end_event);
}

void CompiledExecutor::cleanup_backward_replay_buffers() {
    // Restore any deferred replay checkpoint before final cleanup.
    if (mHasDeferredReplayCheckpoint) {
        mRunState.Stack.restore(mDeferredReplayCheckpoint);
        if (mTemps.size() > mDeferredReplayTempMark) {
            mTemps.resize(mDeferredReplayTempMark);
        }
        mHasDeferredReplayCheckpoint = false;
    }

    clear_replay_copied_refs();
    // Free persistent copies from last replay.
    for (void* ptr : mReplayCopiedBuffers) {
        cudaFreeAsync(ptr, mRunState.MainStream);
    }
    mReplayCopiedBuffers.clear();
}

void CompiledExecutor::stabilize_backward_observable_outputs(const std::unordered_set<std::string>& output_names) {
    if (!mRunState.Allocator) {
        return;
    }
    for (const auto& name : output_names) {
        Tensor* tensor = nullptr;
        int tid = -1;
        if (mCurrentGraph) {
            tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() &&
                mTensors[static_cast<std::size_t>(tid)].Data) {
                tensor = &mTensors[static_cast<std::size_t>(tid)];
            }
        }
        if (!tensor) {
            auto it = mNamedTensors.find(name);
            if (it != mNamedTensors.end() && it->second.Data) {
                tensor = &it->second;
            }
        }
        if (!tensor || !tensor->Data || !mRunState.Stack.owns(tensor->Data)) {
            continue;
        }
        std::vector<long> shape(tensor->Sizes.begin(), tensor->Sizes.begin() + tensor->Rank);
        Tensor stable = mRunState.Allocator->allocate(tensor->DType,
                                                      ("backward_output_" + name).c_str(),
                                                      EAllocationType::ON_DEVICE,
                                                      shape);
        CUDA_CHECK(cudaMemcpyAsync(stable.Data,
                                   tensor->Data,
                                   tensor->bytes(),
                                   cudaMemcpyDeviceToDevice,
                                   mRunState.MainStream));
        if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
            mTensors[static_cast<std::size_t>(tid)] = stable;
        }
        mNamedTensors[name] = stable;
    }
}

void CompiledExecutor::clear_large_backward_stack_slots_after_replay() {
    // Per-layer bwd_layer_end_cleanup nulls layer L's Stack-resident slots,
    // but replay_layer_forward(L) writes blocks[L+1].ln1 / ln1_rstd via the
    // end-of-layer fused residual+rmsnorm — a cross-layer side effect that
    // the per-layer clear can't see. Since we process backward in reverse
    // (27→0), layer L+1's cleanup has already run by the time layer L's
    // replay pollutes blocks[L+1]. Sweep every layer here so no Stack-range
    // pointer survives into the optimizer / next step.
    if (!mRunState.large_bwd_temps_on_stack()) {
        return;
    }
    for (int L = 0; L < static_cast<int>(mConfig.NumLayers); ++L) {
        clear_large_bwd_grad_stack_slots(mRunState, L);
    }
}

void CompiledExecutor::execute_backward(const CompiledGraph& graph,
                                        NCCLCommunicator& comm,
                                        int grad_accum_steps,
                                        int micro_step,
                                        const modules::BackwardHook* hook,
                                        bool skip_zeroing) {
    initialize_backward_execution(graph, comm, micro_step);
    restore_forward_snapshot_for_backward(graph);

    // Seed mTensors[tid] with arena pointers for every block-scope
    // gradient slot (and build the zero-list on first compile).
    populate_bwd_stack_bindings(graph);

    bind_runtime_bindings();
    bind_saved_tensors_for_backward();
    zero_backward_entry_gradients(skip_zeroing);
    gather_backward_non_block_weights(comm);
    prefetch_backward_last_layer(comm);

    // Save stack checkpoint at start of backward - we'll restore per-layer to manage memory
    auto initial_checkpoint = mRunState.Stack.checkpoint();
    int last_layer_restored = -1;
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

    bind_backward_entry_gradient_tensors();

    restore_moe_expert_offsets_for_backward();

    bind_runtime_bindings();
    bind_param_gradient_tensors_for_backward();

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

    const bool skip_profile_backward_tensors = mSkippedBackwardTensors && !mSkippedBackwardTensors->empty();
    auto is_skipped_backward_tensor = [&](const std::string& name) {
        return std::find(mSkippedBackwardTensors->begin(), mSkippedBackwardTensors->end(), name) !=
               mSkippedBackwardTensors->end();
    };
    auto is_skipped_backward_op = [&](const CompiledOp& op) {
        for (const auto& ref : op.inputs) {
            if (is_skipped_backward_tensor(ref.name)) return true;
        }
        for (const auto& ref : op.outputs) {
            if (is_skipped_backward_tensor(ref.name)) return true;
        }
        return false;
    };

    // Use pre-computed last-use data from graph compilation (avoids rebuilding every backward).
    const auto& last_use_names = graph.last_use_names;
    const auto& last_use = graph.last_use_index;
    std::unordered_set<std::string> consumed_backward_names;
    std::unordered_set<std::string> externally_observable_backward_outputs;
    for (const auto& op : graph.ops) {
        for (const auto& ref : op.inputs) {
            if (!ref.name.empty()) {
                consumed_backward_names.insert(ref.name);
            }
        }
    }
    for (const auto& op : graph.ops) {
        for (const auto& ref : op.outputs) {
            if (!ref.name.empty() && consumed_backward_names.find(ref.name) == consumed_backward_names.end()) {
                externally_observable_backward_outputs.insert(ref.name);
            }
        }
    }
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
        // Arena-backed pointers in the bwd_cross_layer arena are reclaimed
        // into a per-backward free list so later cross-layer persists can
        // reuse the memory before the next bump reset.
        release_bwd_cross_layer(ptr);
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
            if (externally_observable_backward_outputs.find(name) != externally_observable_backward_outputs.end()) {
                continue;
            }
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
    const char* op_trace_sync_env = std::getenv("SUROGATE_OP_TRACE_SYNC");
    const bool op_trace_sync = op_trace_sync_env && std::string(op_trace_sync_env) != "0";
    const bool sync_moe_ep_ops =
        env_int("SUROGATE_SYNC_CAPTURE_UNSAFE_OPS",
                (mWeights.qlora_provider() && mWeights.qlora_provider()->has_offloading()) ? 1 : 0) != 0;
    auto sync_after_backward_op = [&](const CompiledOp& op) {
        if (op_trace_sync || (sync_moe_ep_ops && is_moe_ep_sync_boundary(op.type))) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        }
    };
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

    // Stream-driven backward execution (flag-gated pass-through mode).
    // Minimal: no tiled MLP, no capture, no recompute, no bwd_filter / watch /
    // op_profile — matches the Llama-only scope of the forward stream path.
    const bool bwd_stream_driven =
        !graph.instruction_stream.empty() && !bwd_stream_capturing && bwd_tile_group_starts.empty();

    // Per-op layer-end cleanup (shared between SegmentDispatch inline and
    // PhaseExit BwdBlock). Mirrors the flat-ops layer-end handler at lines
    // 2767-2895. Idempotent via last_layer_restored - safe to call multiple
    // times for the same layer. `idx` is the op index that triggered the
    // cleanup (used by BwdCrossLayer persist to decide which tids survive).
    auto bwd_layer_end_cleanup = [&](int L, std::size_t idx, bool capturing = false) {
        if (L < 0 || L == last_layer_restored) return;

        // BwdCrossLayer arena-backed persist runs in BOTH eager and capture
        // modes. The arena is grown to the eager-measured high-water mark
        // by `prepare_bwd_cross_layer_for_capture` before capture begins,
        // so `allocate_bwd_cross_layer` doesn't fall back to cudaMalloc
        // here and the path is capture-safe. Skipping persist under
        // capture would drop cross-layer-global gradients (Gemma4
        // d_scale_8, Qwen3.5 d_blocks[N].lin_conv_w2d) at layer-end
        // Stack.restore, breaking the next layer's resolution.
        {
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
                    std::byte* arena_ptr = allocate_bwd_cross_layer(nbytes);
                    auto* original = reinterpret_cast<const std::byte*>(ptr_key);
                    CUDA_CHECK(
                        cudaMemcpyAsync(arena_ptr, original, nbytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
                    persistent_by_ptr.emplace(ptr_key, arena_ptr);
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
        }

        // Eager-only side-stream / host-callback work. handle_layer_end
        // self-skips when mCapturing; debug dump and grad streaming/reduce/
        // notify/offload do their own host-side recording that's not
        // permitted during CUDA stream capture.
        if (!capturing) {
            handle_layer_end(L);
            if (mDebugDumpBackwardLayerFn) mDebugDumpBackwardLayerFn(L);

            if (mGrads.is_streaming_grads()) {
                GradientOffloadHookPayload offload_payload;
                offload_payload.grads = &mGrads;
                offload_payload.comm = mComm;
                offload_payload.compute_stream = mRunState.MainStream;
                offload_payload.copy_stream = mRunState.side_stream();
                offload_payload.sync_event = mRunState.side_stream_event();
                offload_payload.capturing = capturing;
                dispatch_schema_layer_hooks(HookEventKind::AfterAllReduce, L, &offload_payload);
                if (!offload_payload.offloaded && mComm && mComm->world_size() > 1) {
                    CUDA_CHECK(cudaEventRecord(mRunState.side_stream_event(), mRunState.MainStream));
                    CUDA_CHECK(cudaStreamWaitEvent(mRunState.side_stream(), mRunState.side_stream_event(), 0));
                    mGrads.reduce_layer_grads(L, mRunState.side_stream(), *mComm);
                }
                if (!offload_payload.offloaded) {
                    mGrads.offload_layer_grads(L, mRunState.MainStream, mRunState.side_stream());
                }
            } else if (mComm && mComm->world_size() > 1) {
                CUDA_CHECK(cudaEventRecord(mRunState.side_stream_event(), mRunState.MainStream));
                CUDA_CHECK(cudaStreamWaitEvent(mRunState.side_stream(), mRunState.side_stream_event(), 0));
                mGrads.notify_block(L, mRunState.side_stream(), *mComm);
            }
        }

        mRunState.Stack.restore(initial_checkpoint);
        mTemps.clear();
        prune_stack_tensors(L);
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
                        // Recompute runs via the explicit RecomputeBlock
                        // instruction emitted immediately after.
                    }
                    break;

                case dsl::InstKind::RecomputeBlock:
                    // Explicit forward-block replay for gradient-checkpointed
                    // backward. No-op when recompute is off. Idempotent per
                    // block via mLastRecomputeLayer.
                    if (mRecomputeEnabled && mRecomputeFn && inst.block_index >= 0 &&
                        inst.block_index != mLastRecomputeLayer) {
                        mRecomputeFn(inst.block_index, mB, mT, mRecomputeUseGraphs);
                        mLastRecomputeLayer = inst.block_index;
                    }
                    break;

                case dsl::InstKind::SegmentDispatch:
                    for (std::size_t i = inst.op_start; i < inst.op_end; ++i) {
                        const auto& op = graph.ops[i];
                        if (skip_profile_backward_tensors && is_skipped_backward_op(op)) continue;
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
                                mRecomputeFn(effective_layer_idx, mB, mT, mRecomputeUseGraphs);
                                mLastRecomputeLayer = effective_layer_idx;
                            }
                        }

                        check_op_io_aliasing(op, i, "bwd");
                        if (op_trace) {
                            std::cerr << "[BWD OP " << i << "] " << op_type_to_string(op.type) << " id=" << op.op_id
                                      << std::endl;
                        }
                        try {
                            op.fn(*this, op, static_cast<const void*>(hook));
                            sync_after_backward_op(op);
                        } catch (const std::exception& e) {
                            std::ostringstream oss;
                            oss << "execute_backward stream op=" << i << " type=" << op_type_to_string(op.type)
                                << " id=" << op.op_id << ": " << e.what();
                            throw std::runtime_error(oss.str());
                        }
                        // Profile-controlled memory cleanup: after the first
                        // backward matmul consumes the output payload, release
                        // it so the remaining backward pass can reuse memory.
                        if (op.type == CompiledOpType::MatmulBackward && i == 1) {
                            mRunState.temp_free(mRunState.non_block_activations().output);
                            mTemps.clear();
                            initial_checkpoint = mRunState.Stack.checkpoint();
                        }
                        // Inline prune per op to free Stack slots as soon as
                        // a tid's last use passes. Prologue-heavy graphs (MoE,
                        // hybrid Mamba) need per-op pruning; batching to a
                        // single PruneByLastUse instruction would accumulate
                        // hundreds of Prologue allocations before any gets
                        // freed, producing a 3x+ Stack bloat.
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
                    // Redundant with the inline prune_by_last_use above;
                    // kept as a no-op so the instruction stream shape
                    // remains stable for the static validator.
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
        if (skip_profile_backward_tensors && is_skipped_backward_op(op)) {
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
                          << " layer_end=" << op.layer_end << " inputs=[" << dump_refs(op.inputs) << "]" << " outputs=["
                          << dump_refs(op.outputs) << "]" << std::endl;
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
                    << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id
                    << ", semantic=" << op_semantic_kind_name(op.semantic_kind)
                    << ", comm=" << communication_kind_name(op.comm_profile.kind)
                    << ", distribution=" << distribution_kind_name(op.distribution_kind)
                    << ", caps=" << op_capability_flags_string(op.default_caps)
                    << ", matmul_caps=" << matmul_capability_flags_string(op.matmul_caps)
                    << ", moe_caps=" << moe_capability_flags_string(op.moe_caps)
                    << ", epilogue=" << epilogue_support_flags_string(op.epilogue_support)
                    << ", storage=" << storage_compatibility_flags_string(op.storage_compat) << ")";
                throw std::runtime_error(oss.str());
            }
            check_op_io_aliasing(op, idx, "bwd");
            if (op_trace) {
                std::cerr << "[BWD OP " << idx << "] " << op_type_to_string(op.type) << " id=" << op.op_id << std::endl;
            }
            op.fn(*this, op, static_cast<const void*>(hook));
            sync_after_backward_op(op);
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

            // Post-dispatch side effect: after the first backward matmul frees
            // the profile-bound output payload, reclaim its stack memory.
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

    if (op_profile) {
        report_backward_op_profile(op_profile_total_ms, op_profile_counts, op_profile_start, op_profile_end);
    }

    cleanup_backward_replay_buffers();
    stabilize_backward_observable_outputs(externally_observable_backward_outputs);

    // Final cleanup - pass -1 to allow full pruning (backward complete)
    mRunState.Stack.restore(initial_checkpoint);
    prune_stack_tensors(-1);
    mTemps.clear();

    clear_large_backward_stack_slots_after_replay();

    release_backward_non_block_weights();

    // CPU-RAM centric: offload non-block gradients (embedding, lm_head, final_norm) to CPU
    if (mGrads.is_streaming_grads()) {
        mGrads.offload_non_block_grads(mRunState.MainStream);
    }

    // Deregister on backward exit.
    mRunState.set_active_executor(nullptr);
}

}  // namespace dsl
