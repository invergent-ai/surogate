// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CompiledExecutor forward execution.
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

}  // namespace

void CompiledExecutor::initialize_forward_execution(const CompiledGraph& graph, NCCLCommunicator& comm, bool full) {
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
        int device;
        cudaGetDevice(&device);
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
            cudaMemPoolTrimTo(pool, 0);
        }
    }

    // Initialize flat tensor vector indexed by compile-time tensor IDs.
    mTensors.assign(static_cast<std::size_t>(graph.num_tensors), Tensor{});
    mNamedTensors.clear();
    // Pre-bind every FwdStack tid to its arena slot so block-scope
    // Mapped-slot ops route through the arena at ensure_output_tensor
    // instead of Stack temp_alloc.
    populate_fwd_stack_bindings(graph);
    // Register as active executor only after populate, so
    // block_activation_ptr's tid-first path sees a coherent mTensors on
    // every call.
    mRunState.set_active_executor(this);

    // Scrub mSaved entries pointing into the Stack arena or the
    // replay-persist arena. Between steps both are reset — Stack top rolls
    // back, replay-persist offset rewinds — so any saved Tensor whose Data
    // lived in either region is now a dangling pointer. Persistent
    // (cudaMalloc-backed) saves are preserved. Without this scrub, the
    // next step's replay-layer persistence or backward dispatch iterates
    // over step N-1's stale pointers and faults / silently reads garbage.
    if (mSaved) {
        const std::byte* rp_begin = mReplayPersistArena;
        const std::byte* rp_end = rp_begin + mReplayPersistCapacity;
        for (auto& [name, tensor] : *mSaved) {
            if (!tensor.Data) continue;
            const bool in_stack = mRunState.Stack.owns(tensor.Data);
            const bool in_replay_persist = rp_begin && tensor.Data >= rp_begin && tensor.Data < rp_end;
            if (in_stack || in_replay_persist) {
                tensor.Data = nullptr;
            }
        }
    }
    // Reset replay-persist arena bump so step N's replays overwrite step N-1's
    // slots. clear_replay_copied_refs resets during backward replay; this
    // handles the eager (non-replay) forward path too, preventing monotonic
    // arena growth on runs that never hit the backward-replay path.
    mReplayPersistOffset = 0;
    // Eagerly allocate the arena BEFORE any CUDA graph capture begins. If we
    // let allocate_replay_persist() cudaMalloc lazily during backward, the
    // first cudaMalloc happens inside the captured backward graph — CUDA
    // stream capture doesn't allow synchronous host allocations, and the
    // captured graph ends up with a stale pointer. Allocating here (outside
    // any capture) gives a stable base pointer for all subsequent captures.
    ensure_replay_persist_arena();
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
    if (num_layers > 0) {
        mLayerCheckpoints.resize(static_cast<std::size_t>(num_layers));
        mLayerTempMarks.resize(static_cast<std::size_t>(num_layers));
        mLayerActive.assign(static_cast<std::size_t>(num_layers), 0);
    }

    // Build save mask for fast per-ID lookup during pruning.
    mSaveMask.assign(static_cast<std::size_t>(graph.num_tensors), false);
    if (mSaveList) {
        for (const auto& name : *mSaveList) {
            int sid = graph.find_tensor_id(name);
            if (sid >= 0) {
                mSaveMask[static_cast<std::size_t>(sid)] = true;
            }
        }
    }
    // Cross-layer globals (model-scope tensors consumed by multiple layers,
    // e.g. Gemma4 per_layer_inputs) must NOT be pruned at layer-end — each
    // subsequent layer's compiler-synthesized narrow op reads them, and the
    // forward-to-backward snapshot carry-over also depends on them staying
    // in mTensors. They aren't in the runtime save list, so mark them in
    // the mask directly. Narrow: only preserves the tensor across layer
    // boundaries; does not trigger save_tensors or any other persistence.
    for (int tid = 0; tid < graph.num_tensors; ++tid) {
        if (graph.tensor_meta[static_cast<std::size_t>(tid)].cross_layer_global) {
            mSaveMask[static_cast<std::size_t>(tid)] = true;
        }
    }
}

void CompiledExecutor::gather_forward_non_block_weights(NCCLCommunicator& comm) {
    if (!needs_non_block_weight_transfer() || !mExecutionRequest) {
        return;
    }
    for (const auto& group : mExecutionRequest->gather_before_forward_weight_groups) {
        mWeightManager->gather_non_block_group(group, comm, mRunState.MainStream);
    }
}

void CompiledExecutor::prefetch_forward_layer_zero(NCCLCommunicator& comm) {
    // Prefetch layer 0 before loop.
    mPrefetchDirection = 1;  // Forward traversal
    if (mConfig.NumLayers <= 0 || mCapturing) {
        return;
    }
    if (mWeightManager && mWeightManager->needs_block_gather()) {
        mWeightManager->gather_block(0, comm, mRunState.side_stream());
    }
    // QLoRA offload: prefetch first layer's quantized weights.
    if (auto* provider = mWeights.qlora_provider()) {
        if (provider->has_offloading()) {
            provider->prefetch_for_layer(0, mRunState.side_stream());
        }
    }
}

void CompiledExecutor::release_forward_non_block_weights() {
    if (!needs_non_block_weight_transfer() || !mExecutionRequest) {
        return;
    }
    for (const auto& group : mExecutionRequest->release_after_forward_weight_groups) {
        mWeightManager->release_non_block_group(group, mRunState.MainStream);
    }
}

void CompiledExecutor::persist_forward_saved_layer_tensors(const CompiledGraph& graph,
                                                           int layer_idx,
                                                           bool fwd_stream_capturing,
                                                           bool forward_replay_active,
                                                           int& arena_persists,
                                                           int& cudaMalloc_persists) {
    if (!mSaved || !mSaveList) {
        return;
    }

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
    auto name_belongs_to_layer = [](const std::string& name, int target_layer) -> bool {
        int lyr = -1;
        std::string fld;
        return parse_block_param(name, lyr, fld) && lyr == target_layer;
    };
    auto q35_forward_replay_needs_persist = [&](const std::string& name) -> bool {
        if (!forward_replay_active || !is_qwen3_5_forward_replay_model) {
            return false;
        }
        int resolved_layer = -1;
        std::string field;
        if (!parse_block_param(name, resolved_layer, field) || resolved_layer != layer_idx) {
            return false;
        }
        const TensorSlot slot = builtin_slot_from_name(strip_ssa_suffix(field));
        return slot == TensorSlot::BlockLN1 || slot == TensorSlot::BlockLN1RSTD || slot == TensorSlot::BlockLN2 ||
               slot == TensorSlot::BlockLN2RSTD;
    };
    auto is_lora_hook_activation = [&](const std::string& name) -> bool {
        if (!mLoRAConfig) {
            return false;
        }
        const TensorSlot slot = resolve_block_slot(name);
        return slot == TensorSlot::BlockSwiGLU || slot == TensorSlot::BlockAtt;
    };
    auto will_recompute_tensor = [&](const std::string& tensor_name) -> bool {
        if (!mRecomputeEnabled || !mSlotRegistry) {
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
                // Capture can record the D2D copy below, but it cannot
                // allocate the persistent destination. The graph executor
                // must preallocate these buffers before capture.
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
        const bool force_lora_hook = is_lora_hook_activation(name);
        if (auto saved_it = mSaved->find(name); saved_it != mSaved->end()) {
            if (!force_lora_hook) {
                continue;
            }
            mSaved->erase(saved_it);
        }
        // Skip tensors that will be recomputed in backward — save metadata only.
        // This avoids allocating persistent cudaMalloc buffers for tensors like
        // mlp_up and swiglu that are stack-backed but fully recomputable.
        if (!force_lora_hook && will_recompute_tensor(name)) {
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

        // Arena-backed persist: when the phase-tree arenas are bound
        // and the tid is classified as SaveForBwd, copy directly into
        // the pre-allocated arena slot instead of cudaMalloc'ing a
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
    (void)graph;
    (void)saved_count;
    (void)recompute_count;
}

void CompiledExecutor::dump_forward_debug_tensors() {
    // Dump requested non-block tensors (e.g. xF/residual_final) after forward.
    // Per-layer block dumps are handled in the layer_end callback above.
    if (!mDebugDumpFn) {
        return;
    }
    static const char* dump_tensors_env = std::getenv("SUROGATE_DEBUG_DUMP_TENSORS");
    if (!dump_tensors_env || !*dump_tensors_env) {
        return;
    }

    std::vector<std::string> names;
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

    std::vector<std::string> global_names;
    global_names.reserve(names.size());
    for (const auto& name : names) {
        if (name.rfind("blocks[", 0) != 0) {
            global_names.push_back(name);
        }
    }
    if (global_names.empty()) {
        return;
    }
    cudaStreamCaptureStatus dump_capture_status = cudaStreamCaptureStatusNone;
    const bool dump_capturing = (cudaStreamIsCapturing(mRunState.MainStream, &dump_capture_status) == cudaSuccess &&
                                 dump_capture_status != cudaStreamCaptureStatusNone);
    if (!dump_capturing) {
        mDebugDumpFn(global_names, -1);
    }
}

void CompiledExecutor::snapshot_forward_execution_state() {
    // Snapshot mTensors + mNamedTensors at the end of forward execution.
    // Captures the authoritative per-tid runtime state produced by the
    // forward dispatchers — including matmul_swiglu's live-buffer rebinds,
    // store_tensor updates from metadata ops, Stack-backed temps, and view
    // aliases. Restored at execute_backward entry so backward resolves
    // route through the tid-cache fast path.
    mForwardTensorsSnapshot = mTensors;
    mForwardNamedTensorsSnapshot = mNamedTensors;
}

void CompiledExecutor::execute_forward(const CompiledGraph& graph,
                                       NCCLCommunicator& comm,
                                       bool full,
                                       const modules::ForwardHook* hook) {
    initialize_forward_execution(graph, comm, full);
    const int num_layers = static_cast<int>(mConfig.NumLayers);
    // Reuse member vectors to avoid per-forward heap allocations.
    auto& layer_checkpoints = mLayerCheckpoints;
    auto& layer_temp_marks = mLayerTempMarks;
    auto& layer_active = mLayerActive;

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

    // When forward replay is active, ALL block tensors will be regenerated by
    // replay_layer_forward during backward. Save metadata only — no D2D copies needed.
    const bool forward_replay_active = mRecomputeEnabled && static_cast<bool>(mRecomputeFn);

    int arena_persists = 0;
    int cudaMalloc_persists = 0;
    // Layer-end bookkeeping shared by PhaseExit-FwdBlock and the flat-ops
    // loop. Persists stack-backed saves before Stack.restore, prunes dead
    // tensors, clears the per-layer stack-backed slots (rstd / ffn_temps).
    // The tiled-MLP path uses a simpler inline variant — no MoE
    // save, no stack-slot clears — and is intentionally not routed through
    // this helper.
    auto on_fwd_layer_end = [&](int L) {
        if (L >= 0 && L < num_layers && layer_active[static_cast<std::size_t>(L)]) {
            if (mDebugDumpLayerFn) mDebugDumpLayerFn(L);
            if (mConfig.NumExperts > 0) save_moe_layer_tensors(L);
            persist_forward_saved_layer_tensors(graph,
                                                L,
                                                fwd_stream_capturing,
                                                forward_replay_active,
                                                arena_persists,
                                                cudaMalloc_persists);
            mRunState.Stack.restore(layer_checkpoints[static_cast<std::size_t>(L)]);
            if (mTemps.size() > layer_temp_marks[static_cast<std::size_t>(L)]) {
                mTemps.resize(layer_temp_marks[static_cast<std::size_t>(L)]);
            }
            prune_stack_tensors();
            layer_active[static_cast<std::size_t>(L)] = 0;
        }
        if (L >= 0) handle_layer_end(L);
    };

    bind_runtime_bindings();

    // Ensure non-block weights are gathered if streaming/offload is enabled
    gather_forward_non_block_weights(comm);
    prefetch_forward_layer_zero(comm);

    // Main dispatch loop - no string comparisons, direct function pointer dispatch
    const char* op_trace_env = std::getenv("SUROGATE_OP_TRACE");
    const bool op_trace = op_trace_env && std::string(op_trace_env) != "0";
    const char* op_trace_sync_env = std::getenv("SUROGATE_OP_TRACE_SYNC");
    const bool op_trace_sync = op_trace_sync_env && std::string(op_trace_sync_env) != "0";
    const bool sync_moe_ep_ops =
        env_int("SUROGATE_SYNC_CAPTURE_UNSAFE_OPS",
                (mWeights.qlora_provider() && mWeights.qlora_provider()->has_offloading()) ? 1 : 0) != 0;
    auto sync_after_forward_op = [&](const CompiledOp& op) {
        if (op_trace_sync || (sync_moe_ep_ops && is_moe_ep_sync_boundary(op.type))) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        }
    };
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

    // Stream-driven forward execution. Reuses the allocator, stack, and op
    // dispatch; drives them from graph.instruction_stream instead of the
    // flat ops list + op.layer_start/end flags. SegmentDispatch honors
    // graph.layer_segments when mSplitAttentionGraphs is on. The flat-ops
    // loop below remains the path when capturing (CUDA graph capture needs
    // the single-pass op walk) or when the compiler did not emit an
    // instruction stream.
    const bool stream_driven = !graph.instruction_stream.empty() && !mCapturing;
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
                    // Honor graph.layer_segments when split-attention is
                    // active. Each layer's ops are dispatched as an ordered
                    // list of graph-captured / eager segments.
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
                            check_op_io_aliasing(op, i, "fwd");
                            if (op_trace) {
                                std::cerr << "[OP " << i << "] " << op_type_to_string(op.type) << " id=" << op.op_id
                                          << std::endl;
                            }
                            try {
                                op.fn(*this, op, static_cast<const void*>(hook));
                                sync_after_forward_op(op);
                            } catch (const std::exception& e) {
                                std::ostringstream oss;
                                oss << "execute_forward stream op=" << i << " type=" << op_type_to_string(op.type)
                                    << " id=" << op.op_id << ": " << e.what();
                                throw std::runtime_error(oss.str());
                            }
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
                        persist_forward_saved_layer_tensors(graph,
                                                            last_op.layer_end,
                                                            fwd_stream_capturing,
                                                            forward_replay_active,
                                                            arena_persists,
                                                            cudaMalloc_persists);
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
                std::ostringstream oss;
                oss << "CompiledExecutor: no dispatch fn for forward op type " << op_type_to_string(op.type)
                    << " (semantic=" << op_semantic_kind_name(op.semantic_kind)
                    << ", comm=" << communication_kind_name(op.comm_profile.kind)
                    << ", distribution=" << distribution_kind_name(op.distribution_kind)
                    << ", caps=" << op_capability_flags_string(op.default_caps)
                    << ", matmul_caps=" << matmul_capability_flags_string(op.matmul_caps)
                    << ", moe_caps=" << moe_capability_flags_string(op.moe_caps)
                    << ", epilogue=" << epilogue_support_flags_string(op.epilogue_support)
                    << ", storage=" << storage_compatibility_flags_string(op.storage_compat) << ")";
                throw std::runtime_error(oss.str());
            }
            check_op_io_aliasing(op, idx, "fwd");
            op.fn(*this, op, static_cast<const void*>(hook));
            sync_after_forward_op(op);
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

    dump_forward_debug_tensors();

    release_forward_non_block_weights();

    mRunState.set_active_executor(nullptr);
    snapshot_forward_execution_state();
}

}  // namespace dsl
