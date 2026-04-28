// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL gradient store implementation.

#include "runtime/dsl/dsl_grad_store.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <cuda_bf16.h>

#include "runtime/dsl/dsl_param_store.h"
#include "runtime/dsl/graph_compiler.h"
#include "runtime/dsl/hook_registry.h"
#include "runtime/dsl/tensor_role.h"
#include "kernels/kernels.h"
#include "utilities/comm.h"
#include "utilities/utils.h"

namespace dsl {

namespace {

// Helper to check if a parameter name is an embedding name
bool is_embedding_name(const std::string& name) {
    return name == "embedding" || name == "embeddings" || name == "embed_tokens";
}

// Helper to check if a parameter name is an lm_head name
bool is_lm_head_name(const std::string& name) {
    return name == "lm_head" || name == "lm_head_weight";
}

bool use_expert_parallel_grad_route(const std::string& name, const char* context) {
    (void)context;
    return tensor_role_is_expert_parallel_name(name);
}

}  // namespace

DslGradStore::DslGradStore(const DslParamStore& params,
                           const std::shared_ptr<TensorAllocator>& allocator,
                           bool offload_grads,
                           EAllocationType offload_alloc,
                           int num_shards,
                           bool tied_embeddings,
                           std::optional<ETensorDType> grad_dtype_override,
                           bool cpu_training)
    : mAllocator(allocator),
      mOffloadGrads(offload_grads),
      mCpuTraining(cpu_training),
      mOffloadAlloc(offload_alloc),
      mGradDtypeOverride(grad_dtype_override) {
    mSchemaHookDispatchEnabled = schema_hook_dispatch_enabled();

    if (!mAllocator) {
        throw std::runtime_error("DslGradStore: allocator is null");
    }

    // Full gradients are always allocated on device for NCCL compatibility.
    // Sharded gradients (for ZeRO-2) are allocated later in configure() and can be offloaded.
    const EAllocationType grad_alloc = EAllocationType::ON_DEVICE;

    // Helper: check if a parameter name is a block (per-layer) parameter.
    auto is_block_name = [](const std::string& n) -> bool {
        for (const char* prefix : {"blocks.", "blocks[", "layers.", "model.layers.", "model.blocks."}) {
            if (n.find(prefix) != std::string::npos) return true;
        }
        return false;
    };

    // First pass: find embedding gradient name if present (for weight tying)
    std::string embedding_grad_name;
    if (tied_embeddings) {
        for (const auto& name : params.param_names()) {
            if (params.is_trainable(name) && is_embedding_name(name)) {
                embedding_grad_name = name;
                break;
            }
        }
    }

    for (const auto& name : params.param_names()) {
        if (!params.is_trainable(name)) {
            continue;
        }

        // When embeddings are tied, make lm_head gradient point to same tensor as embedding
        if (tied_embeddings && is_lm_head_name(name) && !embedding_grad_name.empty()) {
            // lm_head shares gradient with embedding - add alias to existing gradient
            auto emb_it = mGrads.find(embedding_grad_name);
            if (emb_it != mGrads.end()) {
                mGrads.emplace(name, emb_it->second);  // Same tensor, different key
                mParamOrder.push_back(name);
                continue;
            }
            // Fall through to allocate if embedding not found yet (shouldn't happen)
        }

        const Tensor& weight = params.template_tensor(name);
        std::vector<long> shape(weight.Sizes.begin(), weight.Sizes.begin() + weight.Rank);
        const ETensorDType grad_dtype = mGradDtypeOverride.value_or(weight.DType);

        if (cpu_training && is_block_name(name)) {
            // CPU-RAM centric: skip GPU allocation for block grads.
            // Register with shape metadata but null Data — enable_streaming() will
            // provide rotating GPU buffers and CPU pinned storage.
            Tensor placeholder{};
            placeholder.DType = grad_dtype;
            placeholder.Rank = static_cast<int>(shape.size());
            for (int d = 0; d < placeholder.Rank; ++d) {
                placeholder.Sizes[d] = shape[d];
            }
            placeholder.Data = nullptr;
            mGrads.emplace(name, placeholder);
        } else {
            Tensor grad = mAllocator->allocate(grad_dtype, ("d_" + name).c_str(), grad_alloc, shape);
            mGrads.emplace(name, grad);
        }
        mParamOrder.push_back(name);
    }
    std::sort(mParamOrder.begin(), mParamOrder.end());

    // Build bulk zero segments so zero_all() uses a single kernel launch
    // (block grads with null Data are automatically skipped)
    build_zero_segments();

    // Initialize double-buffer block states
    mBlockStates[0] = {-1, false, nullptr};
    mBlockStates[1] = {-1, false, nullptr};
}

void DslGradStore::configure(const DslGradStoreConfig& config) {
    mConfig = config;

    // Validate: legacy offload_grads (non-streaming) requires shard_gradients (ZeRO-2).
    // In streaming mode (cpu_training), per-layer D2H replaces ZeRO-2 sharding — skip this check.
    // Note: mStreamGrads is set later by enable_streaming(), so we can't check it here.
    // Instead, we allow offload_grads without shard_gradients when num_shards==1
    // (single-GPU cpu_training path; Python validation ensures this is only cpu_training).
    if (mOffloadGrads && !mConfig.shard_gradients && mConfig.num_shards > 1 && !mCpuTraining) {
        throw std::logic_error(
            "offload_grads on multi-GPU requires shard_gradients=true (ZeRO-2) or cpu_training=true.");
    }

    // Early exit if no gradients to manage (e.g., LoRA-only training)
    if (mParamOrder.empty()) {
        return;
    }
    if (mConfig.num_layers > 0) {
        build_layer_grad_map();
        // Only create events if we have layer gradients to reduce
        if (mHasLayerGrads) {
            create_layer_events(mConfig.num_layers);
        }
    }

    // ZeRO-2 gradient offloading: allocate sharded gradient storage on host.
    // This mirrors the old ModularGradientManager behavior where:
    // - Full gradients stay on device (for backward + NCCL reduce-scatter)
    // - Sharded gradients can be offloaded to host (for optimizer)
    if (mOffloadGrads && mConfig.shard_gradients && mConfig.num_shards > 1) {
        allocate_sharded_grads();
    }
}

void DslGradStore::set_schema_hook_registry(const HookRegistry* registry,
                                            std::vector<std::string> schema_ids_by_layer) {
    mSchemaHookRegistry = registry;
    mHookSchemaIdsByLayer = std::move(schema_ids_by_layer);
}

void DslGradStore::build_layer_grad_map() {
    mLayerGradNames.clear();
    mLayerGradNames.resize(mConfig.num_layers);
    mHasLayerGrads = false;

    // Parse gradient names to determine which layer they belong to.
    // Naming convention: "blocks.{layer_idx}.{component}" or "layers.{layer_idx}.{component}"
    for (const auto& name : mParamOrder) {
        int layer_idx = -1;

        // Check for "blocks.N." or "layers.N." pattern
        auto extract_layer = [&](const std::string& prefix) -> bool {
            auto pos = name.find(prefix);
            if (pos != std::string::npos) {
                pos += prefix.size();
                auto dot_pos = name.find('.', pos);
                if (dot_pos != std::string::npos) {
                    try {
                        layer_idx = std::stoi(name.substr(pos, dot_pos - pos));
                        return true;
                    } catch (...) {
                    }
                }
            }
            return false;
        };

        if (extract_layer("blocks.") || extract_layer("layers.") || extract_layer("model.layers.") ||
            extract_layer("model.blocks.")) {
            if (layer_idx >= 0 && layer_idx < mConfig.num_layers) {
                mLayerGradNames[layer_idx].push_back(name);
                mHasLayerGrads = true;
            }
        }
        // Non-layer params (embeddings, lm_head, final_norm) are not tracked per-layer
    }
}

void DslGradStore::build_zero_segments() {
    // Free previous segment arrays if they exist (e.g., rebuild after enable_streaming)
    if (mZeroPtrs.Data) {
        mAllocator->free(mZeroPtrs);
    }
    if (mZeroSizes.Data) {
        mAllocator->free(mZeroSizes);
    }
    mZeroPtrs = {};
    mZeroSizes = {};
    mZeroCount = 0;

    if (!mAllocator || mGrads.empty()) {
        return;
    }

    std::vector<std::uint64_t> ptrs;
    std::vector<std::uint64_t> sizes;
    ptrs.reserve(mGrads.size());
    sizes.reserve(mGrads.size());

    // Deduplicate: tied embeddings share the same tensor
    std::unordered_set<std::byte*> seen;
    seen.reserve(mGrads.size());

    for (const auto& kv : mGrads) {
        const Tensor& t = kv.second;
        if (!t.Data) continue;
        const std::size_t bytes = static_cast<std::size_t>(t.bytes());
        if (bytes == 0) continue;
        if (!seen.insert(t.Data).second) continue;  // skip duplicate (tied weights)
        ptrs.push_back(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t.Data)));
        sizes.push_back(static_cast<std::uint64_t>(bytes));
    }

    mZeroCount = static_cast<int>(ptrs.size());
    if (mZeroCount <= 0) {
        return;
    }

    const long bytes = static_cast<long>(static_cast<std::size_t>(mZeroCount) * sizeof(std::uint64_t));
    mZeroPtrs = mAllocator->allocate(ETensorDType::BYTE, "grad_zero_ptrs", EAllocationType::ON_DEVICE, {bytes});
    mZeroSizes = mAllocator->allocate(ETensorDType::BYTE, "grad_zero_sizes", EAllocationType::ON_DEVICE, {bytes});

    CUDA_CHECK(cudaMemcpy(mZeroPtrs.Data, ptrs.data(), static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mZeroSizes.Data, sizes.data(), static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice));
}

void DslGradStore::create_layer_events(int num_layers) {
    destroy_layer_events();
    mLayerReduceEvents.resize(static_cast<std::size_t>(num_layers), nullptr);
    for (auto& ev : mLayerReduceEvents) {
        CUDA_CHECK(cudaEventCreate(&ev));
    }

    // Create events for double-buffer states
    for (auto& state : mBlockStates) {
        if (!state.Event) {
            CUDA_CHECK(cudaEventCreate(&state.Event));
        }
    }
}

void DslGradStore::destroy_layer_events() noexcept {
    for (auto& ev : mLayerReduceEvents) {
        if (ev) {
            cudaEventDestroy(ev);
            ev = nullptr;
        }
    }
    mLayerReduceEvents.clear();

    for (auto& state : mBlockStates) {
        if (state.Event) {
            cudaEventDestroy(state.Event);
            state.Event = nullptr;
        }
    }
}

void DslGradStore::start_micro_step(cudaStream_t stream, int micro_step, int total_steps) {
    mMicroStep = micro_step;
    mIsLastMicroStep = (micro_step == total_steps - 1);
    mAccumulate = micro_step > 0;
    if (!mAccumulate) {
        zero_all(stream);
        // Zero CPU gradient storage for streaming mode
        if (mStreamGrads) {
            for (auto& [name, cpu_grad] : mCpuGrads) {
                std::memset(cpu_grad.Data, 0, cpu_grad.bytes());
            }
            // Zero the norm accumulator
            if (mLayerNormAccum.Data) {
                CUDA_CHECK(cudaMemsetAsync(mLayerNormAccum.Data, 0, sizeof(float), stream));
            }
        }
    } else if (mStreamGrads) {
        // Zero only the norm accumulator (CPU grads keep accumulated values)
        if (mLayerNormAccum.Data) {
            CUDA_CHECK(cudaMemsetAsync(mLayerNormAccum.Data, 0, sizeof(float), stream));
        }
    }

    // Reset streaming slot tracking
    if (mStreamGrads) {
        mActiveGradSlot = 0;
        mPendingD2HEvents.clear();
    }

    // Reset block states for new micro-step (only needed when overlapped reduction is active)
    if (mHasLayerGrads) {
        for (auto& state : mBlockStates) {
            state.LayerIdx = -1;
            state.NeedsAccumulation = false;
        }
    }
}

void DslGradStore::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    (void)stream;
    (void)comm;
    // Note: For ZeRO-2, any pending accumulations would be handled here.
    // Currently we wait for all reductions inline via wait_for_block_reduce.
}

Tensor* DslGradStore::get_param_grad(const std::string& name, bool& accumulate) {
    auto it = mGrads.find(name);
    if (it == mGrads.end()) {
        return nullptr;
    }
    // Streaming block grads: always overwrite on GPU (cross-micro-step accumulation on CPU)
    if (mStreamGrads && is_block_grad(name)) {
        accumulate = false;
    } else {
        accumulate = mAccumulate;
    }
    return &it->second;
}

void DslGradStore::zero_all(cudaStream_t stream) {
    if (mZeroCount > 0 && mZeroPtrs.Data && mZeroSizes.Data) {
        zero_device_segments(reinterpret_cast<const std::uint64_t*>(mZeroPtrs.Data),
                             reinterpret_cast<const std::uint64_t*>(mZeroSizes.Data),
                             mZeroCount,
                             stream);
    }
}

void DslGradStore::reduce_all(NCCLCommunicator& comm, cudaStream_t stream) {
    const bool ep_active = comm.ep_enabled();
    for (auto& kv : mGrads) {
        // EP: expert weight gradients average across DP group only (same experts),
        // everything else (dense, router, norms) averages across all GPUs.
        if (ep_active && use_expert_parallel_grad_route(kv.first, "DslGradStore::reduce_all")) {
            comm.all_reduce_avg_dp(kv.second, stream);
        } else {
            comm.all_reduce_avg(kv.second, stream);
        }
    }
    GradientOffloadHookPayload payload;
    payload.grads = this;
    payload.comm = &comm;
    payload.compute_stream = stream;
    payload.copy_stream = stream;
    payload.all_reduced = true;
    for (int layer = 0; layer < static_cast<int>(mHookSchemaIdsByLayer.size()); ++layer) {
        dispatch_schema_layer_hooks(HookEventKind::AfterAllReduce, layer, stream, &payload);
    }
    mReducePending = false;
}

void DslGradStore::reduce_all_async(NCCLCommunicator& comm, cudaStream_t stream, cudaEvent_t done_event) {
    const bool ep_active = comm.ep_enabled();
    const bool ep_only = ep_active && (comm.dp_size() == 1);

    // Helper: reduce a single gradient using the appropriate communicator
    auto reduce_one = [&](const std::string& name, Tensor& grad) {
        if (ep_active && use_expert_parallel_grad_route(name, "DslGradStore::reduce_all_async")) {
            comm.all_reduce_avg_dp(grad, stream);
        } else {
            comm.all_reduce_avg(grad, stream);
        }
    };

    // If overlapped reduction is enabled and we have layer gradients that were already reduced,
    // only reduce non-layer gradients (embeddings, lm_head, final_norm)
    if (is_overlapped_enabled() && mHasLayerGrads && !ep_only) {
        // Collect non-layer gradient names (those not in any layer)
        std::unordered_set<std::string> layer_grads;
        for (const auto& layer_names : mLayerGradNames) {
            for (const auto& name : layer_names) {
                layer_grads.insert(name);
            }
        }

        // Reduce non-layer gradients
        for (auto& kv : mGrads) {
            if (layer_grads.find(kv.first) == layer_grads.end()) {
                reduce_one(kv.first, kv.second);
            }
        }
    } else {
        // Fallback: reduce all gradients (original behavior)
        for (auto& kv : mGrads) {
            reduce_one(kv.first, kv.second);
        }
    }

    // Record completion event so optimizer can wait on it
    if (done_event) {
        CUDA_CHECK(cudaEventRecord(done_event, stream));
    }
    mReducePending = true;
}

void DslGradStore::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    // Single-GPU: nothing to reduce
    if (mConfig.num_shards == 1) return;

    // No layer gradient map: can't do per-layer reduction
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) return;
    if (mLayerGradNames[layer_idx].empty()) return;
    // EP-only (dp_size=1): defer layer grad reduction to reduce_all_async().
    // Layer-end overlap can race with late gradient writes in EP backward paths.
    if (comm.ep_enabled() && comm.dp_size() == 1) return;

    if (!mConfig.shard_gradients) {
        // ZeRO-1: reduce-scatter once per optimizer step (on the last micro-step)
        if (!mIsLastMicroStep) return;
        scatter_reduce_layer(layer_idx, stream, comm);
        return;
    }

    // ZeRO-2: reduce-scatter on every micro-step
    // Use double-buffering to overlap reduction with next layer's compute
    auto& state = mBlockStates[layer_idx % 2];

    // Wait for previous layer using this buffer slot to finish its reduction
    if (state.NeedsAccumulation && state.Event) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, state.Event, 0));
        state.NeedsAccumulation = false;
    }

    state.LayerIdx = layer_idx;
    scatter_reduce_layer(layer_idx, stream, comm);
    state.NeedsAccumulation = true;
}

void DslGradStore::wait_for_block_reduce(int layer_idx, cudaStream_t stream) {
    if (mConfig.num_shards == 1) return;
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerReduceEvents.size())) return;

    cudaEvent_t ev = mLayerReduceEvents[layer_idx];
    if (ev) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, ev, 0));
    }
}

std::vector<Tensor*> DslGradStore::get_layer_grads(int layer_idx) {
    std::vector<Tensor*> result;
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) {
        return result;
    }

    for (const auto& name : mLayerGradNames[layer_idx]) {
        auto it = mGrads.find(name);
        if (it != mGrads.end()) {
            result.push_back(&it->second);
        }
    }
    return result;
}

void DslGradStore::scatter_reduce_layer(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) return;

    const auto& grad_names = mLayerGradNames[layer_idx];
    if (grad_names.empty()) return;

    cudaEvent_t ev =
        (layer_idx < static_cast<int>(mLayerReduceEvents.size())) ? mLayerReduceEvents[layer_idx] : nullptr;

    const bool ep_active = comm.ep_enabled();

    // EP: expert gradients use DP-only all-reduce (separate from global transaction).
    // Dense/router gradients use the standard global comm transaction.
    if (ep_active) {
        // First pass: reduce dense/router grads via global comm transaction
        bool has_dense = false;
        for (const auto& name : grad_names) {
            if (!use_expert_parallel_grad_route(name, "DslGradStore::scatter_reduce_layer.has_dense")) {
                has_dense = true;
                break;
            }
        }
        if (has_dense) {
            comm.begin_transaction(stream);
            for (const auto& name : grad_names) {
                if (use_expert_parallel_grad_route(name, "DslGradStore::scatter_reduce_layer.dense")) continue;
                auto it = mGrads.find(name);
                if (it != mGrads.end() && it->second.Data && it->second.nelem() > 0) {
                    if (mConfig.shard_gradients) {
                        comm.schedule_reduce_scatter(it->second);
                    } else {
                        comm.schedule_all_reduce_avg(it->second);
                    }
                }
            }
            comm.execute_transaction(ev);
        }

        // Second pass: reduce expert grads via DP comm (individual calls)
        for (const auto& name : grad_names) {
            if (!use_expert_parallel_grad_route(name, "DslGradStore::scatter_reduce_layer.expert")) continue;
            auto it = mGrads.find(name);
            if (it != mGrads.end() && it->second.Data && it->second.nelem() > 0) {
                comm.all_reduce_avg_dp(it->second, stream);
            }
        }
    } else {
        // No EP: original path — all grads use global comm
        comm.begin_transaction(stream);
        for (const auto& name : grad_names) {
            auto it = mGrads.find(name);
            if (it != mGrads.end() && it->second.Data && it->second.nelem() > 0) {
                if (mConfig.shard_gradients) {
                    comm.schedule_reduce_scatter(it->second);
                } else {
                    comm.schedule_all_reduce_avg(it->second);
                }
            }
        }
        comm.execute_transaction(ev);
    }

    GradientOffloadHookPayload reduce_payload;
    reduce_payload.grads = this;
    reduce_payload.compute_stream = stream;
    reduce_payload.copy_stream = stream;
    reduce_payload.reduce_scattered = mConfig.shard_gradients;
    dispatch_schema_layer_hooks(mConfig.shard_gradients ? HookEventKind::AfterReduceScatter
                                                        : HookEventKind::AfterAllReduce,
                                layer_idx,
                                stream,
                                &reduce_payload);

    // ZeRO-2 with offloading: accumulate reduced shards into host storage.
    if (mConfig.shard_gradients && !reduce_payload.sharded_accumulated && !mShardedGrads.empty()) {
        accumulate_to_sharded(layer_idx, stream);
    }
}

int DslGradStore::dispatch_schema_layer_hooks(HookEventKind event, int layer_idx, cudaStream_t stream, void* payload) {
    if (!mSchemaHookDispatchEnabled || !mSchemaHookRegistry || layer_idx < 0 ||
        layer_idx >= static_cast<int>(mHookSchemaIdsByLayer.size())) {
        return 0;
    }
    const std::string& schema_id = mHookSchemaIdsByLayer[static_cast<std::size_t>(layer_idx)];
    if (schema_id.empty()) {
        return 0;
    }
    int dispatched = 0;
    for (const HookRegistration& registration : mSchemaHookRegistry->registrations()) {
        if (registration.event != event || registration.target.schema_id != schema_id) {
            continue;
        }
        HookContext context;
        context.layer_idx = layer_idx;
        context.target = registration.target;
        context.event = event;
        context.stream = stream;
        context.payload = payload;
        if (registration.callback) {
            registration.callback(context);
        }
        ++dispatched;
    }
    return dispatched;
}

void DslGradStore::accumulate_layer_to_sharded(int layer_idx, cudaStream_t stream) {
    if (!mConfig.shard_gradients || mShardedGrads.empty()) {
        return;
    }
    accumulate_to_sharded(layer_idx, stream);
}

void DslGradStore::allocate_sharded_grads() {
    // Allocate sharded gradient storage for ZeRO-2 offloading.
    // Each rank stores only its local shard (1/num_shards of full gradient).
    // Storage can be on host (pinned/write-combined) for memory savings.
    for (const auto& kv : mGrads) {
        const std::string& name = kv.first;
        const Tensor& full = kv.second;

        // Compute shard size (first dimension is sharded)
        std::vector<long> shard_shape(full.Sizes.begin(), full.Sizes.begin() + full.Rank);
        long full_size = shard_shape[0];
        long shard_size = (full_size + mConfig.num_shards - 1) / mConfig.num_shards;
        shard_shape[0] = shard_size;

        Tensor shard = mAllocator->allocate(full.DType, ("ds_" + name).c_str(), mOffloadAlloc, shard_shape);
        mShardedGrads.emplace(name, shard);
    }
}

void DslGradStore::accumulate_to_sharded(int layer_idx, cudaStream_t stream) {
    // After reduce-scatter, copy the local shard from full gradient buffer to sharded storage.
    // This is the ZeRO-2 accumulation step from the old ModularGradientManager.
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) return;

    const auto& grad_names = mLayerGradNames[layer_idx];
    for (const auto& name : grad_names) {
        auto full_it = mGrads.find(name);
        auto shard_it = mShardedGrads.find(name);
        if (full_it == mGrads.end() || shard_it == mShardedGrads.end()) continue;

        const Tensor& full = full_it->second;
        Tensor& shard = shard_it->second;

        // After reduce-scatter, the local shard is in [0:shard_size] of the full buffer.
        // We need to copy/accumulate it to the sharded storage.
        size_t shard_bytes = shard.bytes();

        if (mMicroStep == 0) {
            // First micro-step: copy (overwrite)
            CUDA_CHECK(cudaMemcpyAsync(shard.Data, full.Data, shard_bytes, cudaMemcpyDeviceToHost, stream));
        } else {
            // Subsequent micro-steps: accumulate
            // For host-resident shards, we use a simple kernel that writes to pinned memory.
            // Pinned memory is accessible from GPU via zero-copy.
            // Use layer_idx + micro_step as seed for stochastic rounding.
            unsigned seed = static_cast<unsigned>(layer_idx * 1000 + mMicroStep);
            vector_add_sr(shard, shard, full, 1.0f, static_cast<long>(shard.nelem()), seed, stream);
        }
    }
}

std::vector<Tensor*> DslGradStore::get_layer_sharded_grads(int layer_idx) {
    std::vector<Tensor*> result;
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) {
        return result;
    }

    // Return sharded grads if offloading is active, otherwise return full grads
    auto& grad_map = mShardedGrads.empty() ? mGrads : mShardedGrads;

    for (const auto& name : mLayerGradNames[layer_idx]) {
        auto it = grad_map.find(name);
        if (it != grad_map.end()) {
            result.push_back(&it->second);
        }
    }
    return result;
}

// ============================================================================
// CPU-RAM centric gradient streaming
// ============================================================================

std::string DslGradStore::base_grad_name(const std::string& name) {
    // "blocks.5.attn_q_weight" → "blocks[].attn_q_weight"
    // Find "blocks.N." or "layers.N." and replace the number with []
    for (const char* prefix : {"blocks.", "layers.", "model.layers.", "model.blocks."}) {
        auto pos = name.find(prefix);
        if (pos != std::string::npos) {
            auto num_start = pos + std::strlen(prefix);
            auto dot_pos = name.find('.', num_start);
            if (dot_pos != std::string::npos) {
                return name.substr(0, pos) + std::string(prefix, std::strlen(prefix) - 1) + "[]" + name.substr(dot_pos);
            }
        }
    }
    return name;  // Non-block param, return as-is
}

bool DslGradStore::is_block_grad(const std::string& name) const {
    for (const auto& layer_names : mLayerGradNames) {
        for (const auto& n : layer_names) {
            if (n == name) return true;
        }
    }
    return false;
}

void DslGradStore::enable_streaming(const DslParamStore& params) {
    if (mLayerGradNames.empty()) {
        throw std::runtime_error("DslGradStore::enable_streaming: call configure() first to build layer map");
    }
    mStreamGrads = true;

    // Block grads were already skipped during construction (cpu_training=true in constructor).
    // Their entries in mGrads have null Data. enable_streaming allocates:
    // - Rotating GPU buffers (2 slots, max-layer sized) for compute
    // - CPU pinned storage for all grads (accumulation + optimizer reads)
    allocate_streaming_buffers(params);
    create_streaming_events();

    // Rebuild zero segments — block grads have null Data (excluded from zeroing).
    // Only non-block grads (embedding, lm_head, norm) remain for GPU-side zeroing.
    build_zero_segments();
}

void DslGradStore::allocate_streaming_buffers(const DslParamStore& params) {
    // 1. Allocate CPU pinned storage for ALL parameter gradients
    for (const auto& kv : mGrads) {
        const auto& name = kv.first;
        const auto& gpu_grad = kv.second;
        std::vector<long> shape(gpu_grad.Sizes.begin(), gpu_grad.Sizes.begin() + gpu_grad.Rank);
        mCpuGrads.emplace(
            name,
            mAllocator->allocate(gpu_grad.DType, ("cpu_d_" + name).c_str(), EAllocationType::PINNED, shape));
    }

    // 2. Allocate CPU staging buffer (for micro-step > 0 accumulation)
    //    Sized for max layer; keyed by base_name
    std::unordered_map<std::string, std::vector<long>> max_shapes;
    std::unordered_map<std::string, ETensorDType> base_dtypes;
    for (const auto& layer_names : mLayerGradNames) {
        for (const auto& name : layer_names) {
            auto base = base_grad_name(name);
            auto it = mGrads.find(name);
            if (it == mGrads.end()) continue;
            const auto& grad = it->second;
            std::vector<long> shape(grad.Sizes.begin(), grad.Sizes.begin() + grad.Rank);
            auto sit = max_shapes.find(base);
            long existing_nelem = 0;
            if (sit != max_shapes.end()) {
                existing_nelem = 1;
                for (auto d : sit->second)
                    existing_nelem *= d;
            }
            if (sit == max_shapes.end() || grad.nelem() > existing_nelem) {
                max_shapes[base] = shape;
                base_dtypes[base] = grad.DType;
            }
        }
    }

    for (const auto& [base, shape] : max_shapes) {
        mCpuStagingBuffer.emplace(
            base,
            mAllocator->allocate(base_dtypes[base], ("staging_d_" + base).c_str(), EAllocationType::PINNED, shape));
    }

    // 3. Allocate double-buffered GPU gradient slots (sized for max layer)
    for (int s = 0; s < kNumGradSlots; ++s) {
        auto& slot = mGradSlots[s];
        for (const auto& [base, shape] : max_shapes) {
            slot.buffers.emplace(base,
                                 mAllocator->allocate(base_dtypes[base],
                                                      ("grad_slot" + std::to_string(s) + "_" + base).c_str(),
                                                      EAllocationType::ON_DEVICE,
                                                      shape));
        }
    }

    // 4. Allocate gradient norm accumulator (single float on GPU)
    mLayerNormAccum = mAllocator->allocate(ETensorDType::FP32, "grad_norm_accum", EAllocationType::ON_DEVICE, {1});
}

void DslGradStore::create_streaming_events() {
    for (int s = 0; s < kNumGradSlots; ++s) {
        auto& slot = mGradSlots[s];
        CUDA_CHECK(cudaEventCreateWithFlags(&slot.d2h_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&slot.compute_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&slot.reduce_done, cudaEventDisableTiming));
        // Pre-signal so first prepare doesn't deadlock
        CUDA_CHECK(cudaEventRecord(slot.d2h_done, nullptr));
    }
}

void DslGradStore::destroy_streaming_events() noexcept {
    for (int s = 0; s < kNumGradSlots; ++s) {
        auto& slot = mGradSlots[s];
        if (slot.d2h_done) {
            cudaEventDestroy(slot.d2h_done);
            slot.d2h_done = nullptr;
        }
        if (slot.compute_done) {
            cudaEventDestroy(slot.compute_done);
            slot.compute_done = nullptr;
        }
        if (slot.reduce_done) {
            cudaEventDestroy(slot.reduce_done);
            slot.reduce_done = nullptr;
        }
    }
}

void DslGradStore::prepare_layer_grads(int layer_idx, cudaStream_t stream) {
    if (!mStreamGrads || layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) {
        return;
    }

    int slot_idx = mActiveGradSlot;
    auto& slot = mGradSlots[slot_idx];

    // Wait for previous D2H to complete (so we can reuse this GPU buffer)
    CUDA_CHECK(cudaStreamWaitEvent(stream, slot.d2h_done, 0));

    // Rebind: point mGrads[name].Data to the GPU buffer slot
    for (const auto& name : mLayerGradNames[layer_idx]) {
        auto base = base_grad_name(name);
        auto buf_it = slot.buffers.find(base);
        if (buf_it == slot.buffers.end()) continue;
        auto grad_it = mGrads.find(name);
        if (grad_it == mGrads.end()) continue;
        // Point the gradient tensor's data to the rotating buffer
        grad_it->second.Data = buf_it->second.Data;
    }

    slot.layer_idx = layer_idx;
}

void DslGradStore::reduce_layer_grads(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    if (!mStreamGrads || comm.world_size() <= 1) return;
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) return;

    int slot_idx = mActiveGradSlot;
    auto& slot = mGradSlots[slot_idx];

    // NCCL all-reduce on side_stream
    comm.begin_transaction(stream);
    for (const auto& name : mLayerGradNames[layer_idx]) {
        auto grad_it = mGrads.find(name);
        if (grad_it == mGrads.end() || !grad_it->second.Data) continue;
        comm.schedule_all_reduce_avg(grad_it->second);
    }
    comm.execute_transaction(slot.reduce_done);
}

void DslGradStore::offload_layer_grads(int layer_idx, cudaStream_t compute_stream, cudaStream_t copy_stream) {
    if (!mStreamGrads || layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) {
        return;
    }

    int slot_idx = mActiveGradSlot;
    auto& slot = mGradSlots[slot_idx];

    // Record that backward (+ optional reduce) is complete on MainStream
    CUDA_CHECK(cudaEventRecord(slot.compute_done, compute_stream));

    // On copy_stream: wait for compute, then D2H
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream, slot.compute_done, 0));

    // If multi-GPU, also wait for NCCL reduce
    if (slot.reduce_done) {
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, slot.reduce_done, 0));
    }

    // Accumulate per-layer gradient norm on GPU (before D2H)
    for (const auto& name : mLayerGradNames[layer_idx]) {
        auto grad_it = mGrads.find(name);
        if (grad_it == mGrads.end() || !grad_it->second.Data) continue;
        // TODO: call global_norm_squared(mLayerNormAccum, grad, ...) here
        // For now, norm will be computed on CPU after D2H in wait_all_offloads
    }

    // D2H copy
    if (mMicroStep == 0) {
        // First micro-step: D2H directly to CPU storage
        for (const auto& name : mLayerGradNames[layer_idx]) {
            auto grad_it = mGrads.find(name);
            auto cpu_it = mCpuGrads.find(name);
            if (grad_it == mGrads.end() || cpu_it == mCpuGrads.end()) continue;
            CUDA_CHECK(cudaMemcpyAsync(cpu_it->second.Data,
                                       grad_it->second.Data,
                                       grad_it->second.bytes(),
                                       cudaMemcpyDeviceToHost,
                                       copy_stream));
        }
    } else {
        // Subsequent micro-steps: D2H to staging, then CPU accumulate in wait_all_offloads
        for (const auto& name : mLayerGradNames[layer_idx]) {
            auto grad_it = mGrads.find(name);
            auto base = base_grad_name(name);
            auto staging_it = mCpuStagingBuffer.find(base);
            if (grad_it == mGrads.end() || staging_it == mCpuStagingBuffer.end()) continue;
            CUDA_CHECK(cudaMemcpyAsync(staging_it->second.Data,
                                       grad_it->second.Data,
                                       grad_it->second.bytes(),
                                       cudaMemcpyDeviceToHost,
                                       copy_stream));
        }
    }

    // Record D2H done
    CUDA_CHECK(cudaEventRecord(slot.d2h_done, copy_stream));
    mPendingD2HEvents.push_back(slot.d2h_done);

    // Advance to next slot
    mActiveGradSlot = (mActiveGradSlot + 1) % kNumGradSlots;
}

void DslGradStore::offload_non_block_grads(cudaStream_t stream) {
    if (!mStreamGrads) return;

    for (const auto& name : mParamOrder) {
        if (is_block_grad(name)) continue;
        auto grad_it = mGrads.find(name);
        auto cpu_it = mCpuGrads.find(name);
        if (grad_it == mGrads.end() || cpu_it == mCpuGrads.end()) continue;
        if (!grad_it->second.Data) continue;

        if (mMicroStep == 0) {
            CUDA_CHECK(cudaMemcpyAsync(cpu_it->second.Data,
                                       grad_it->second.Data,
                                       grad_it->second.bytes(),
                                       cudaMemcpyDeviceToHost,
                                       stream));
        } else {
            // For non-block grads, accumulate was done on GPU via mAccumulateTensors
            // so GPU has the accumulated result. Just copy it.
            CUDA_CHECK(cudaMemcpyAsync(cpu_it->second.Data,
                                       grad_it->second.Data,
                                       grad_it->second.bytes(),
                                       cudaMemcpyDeviceToHost,
                                       stream));
        }
    }
}

void DslGradStore::wait_all_offloads(cudaStream_t stream) {
    if (!mStreamGrads) return;

    // Wait for all pending D2H copies
    for (auto ev : mPendingD2HEvents) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, ev, 0));
    }
    // Synchronize to ensure all D2H is visible on CPU
    CUDA_CHECK(cudaStreamSynchronize(stream));
    mPendingD2HEvents.clear();

    // CPU-side accumulation for micro_step > 0
    if (mMicroStep > 0) {
        for (const auto& layer_names : mLayerGradNames) {
            for (const auto& name : layer_names) {
                auto cpu_it = mCpuGrads.find(name);
                auto base = base_grad_name(name);
                auto staging_it = mCpuStagingBuffer.find(base);
                if (cpu_it == mCpuGrads.end() || staging_it == mCpuStagingBuffer.end()) continue;

                // cpu_grad += staging
                const auto nelem = static_cast<std::size_t>(cpu_it->second.nelem());
                if (cpu_it->second.DType == ETensorDType::FP32) {
                    auto* dst = cpu_it->second.template get<float>();
                    const auto* src = staging_it->second.template get<float>();
                    for (std::size_t i = 0; i < nelem; ++i) {
                        dst[i] += src[i];
                    }
                } else if (cpu_it->second.DType == ETensorDType::BF16) {
                    auto* dst = reinterpret_cast<nv_bfloat16*>(cpu_it->second.Data);
                    const auto* src = reinterpret_cast<const nv_bfloat16*>(staging_it->second.Data);
                    for (std::size_t i = 0; i < nelem; ++i) {
                        float val = static_cast<float>(dst[i]) + static_cast<float>(src[i]);
                        dst[i] = static_cast<nv_bfloat16>(val);
                    }
                }
            }
        }
    }
}

const Tensor& DslGradStore::get_cpu_grad(const std::string& name) const {
    auto it = mCpuGrads.find(name);
    if (it == mCpuGrads.end()) {
        throw std::runtime_error("DslGradStore::get_cpu_grad: not found: " + name);
    }
    return it->second;
}

const std::vector<std::string>& DslGradStore::layer_grad_names(int layer_idx) const {
    static const std::vector<std::string> empty;
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) {
        return empty;
    }
    return mLayerGradNames[layer_idx];
}

namespace {

bool is_device_resident_grad(const Tensor& t) {
    return t.Data != nullptr && t.Device >= 0;
}

}  // namespace

std::size_t DslGradStore::rebindable_accumulator_bytes(const CompiledGraph& graph) const {
    // Whole-or-nothing gate: if any gradient path is non-local, the
    // allocator keeps ownership of every grad. Offload / streaming /
    // ZeRO-2 sharding all intersperse device and non-device storage, so
    // clamping to just the local high-water mark isn't sound (the
    // compiler's Accumulator offsets cover every grad tid).
    if (mStreamGrads || mCpuTraining || mOffloadGrads) return 0;
    if (!mShardedGrads.empty()) return 0;

    std::size_t high_water = 0;
    for (const auto& kv : mGrads) {
        const Tensor& t = kv.second;
        if (!is_device_resident_grad(t)) return 0;
        const std::string grad_name = "d_" + kv.first;
        const int tid = graph.find_tensor_id(grad_name);
        if (tid < 0) continue;
        const auto& meta = graph.tensor_meta[static_cast<std::size_t>(tid)];
        if (meta.region != RegionKind::Accumulator || meta.offset == SIZE_MAX) continue;
        const std::size_t tensor_bytes = t.bytes();
        if (tensor_bytes == 0 || meta.bytes < tensor_bytes) continue;
        high_water = std::max(high_water, meta.offset + tensor_bytes);
    }
    return high_water;
}

void DslGradStore::rebind_to_accumulator_arena(const CompiledGraph& graph,
                                               const PhaseArenas& arenas,
                                               cudaStream_t stream) {
    if (!arenas.allocated || arenas.accumulator_ptr == nullptr || arenas.accumulator_bytes == 0) return;
    if (mStreamGrads || mCpuTraining || mOffloadGrads || !mShardedGrads.empty()) return;

    std::size_t rebound = 0;
    std::size_t skipped_no_tid = 0;
    std::size_t skipped_non_accumulator = 0;
    std::size_t skipped_size_mismatch = 0;

    // Dedup aliased gradients (tied embedding → lm_head grad share
    // a single Tensor in mGrads). Rebind the Data pointer once; later
    // aliases repoint without re-freeing.
    std::unordered_map<const std::byte*, std::byte*> rebind_map;

    for (const auto& name : mParamOrder) {
        auto it = mGrads.find(name);
        if (it == mGrads.end()) continue;
        Tensor& grad = it->second;
        if (!is_device_resident_grad(grad)) continue;

        auto cached = rebind_map.find(grad.Data);
        if (cached != rebind_map.end()) {
            // Aliased: repoint Data to the slot the original rebind chose.
            grad.Data = cached->second;
            continue;
        }

        const std::string grad_name = "d_" + name;
        const int tid = graph.find_tensor_id(grad_name);
        if (tid < 0) {
            ++skipped_no_tid;
            continue;
        }
        const auto& meta = graph.tensor_meta[static_cast<std::size_t>(tid)];
        if (meta.region != RegionKind::Accumulator || meta.offset == SIZE_MAX) {
            ++skipped_non_accumulator;
            continue;
        }
        const std::size_t tensor_bytes = grad.bytes();
        if (tensor_bytes == 0 || meta.bytes < tensor_bytes || meta.offset + tensor_bytes > arenas.accumulator_bytes) {
            ++skipped_size_mismatch;
            continue;
        }

        std::byte* arena_ptr = arenas.accumulator_ptr + meta.offset;
        CUDA_CHECK(cudaMemcpyAsync(arena_ptr, grad.Data, tensor_bytes, cudaMemcpyDeviceToDevice, stream));
        float* preserved_stats = grad.Stats;
        const int device = grad.Device;
        std::byte* orig = grad.Data;
        mAllocator->free(grad);
        grad.Data = arena_ptr;
        grad.Device = device;
        grad.Stats = preserved_stats;
        rebind_map.emplace(orig, arena_ptr);
        ++rebound;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Grad pointers were rebound; the cached pointer/size packed arrays
    // in mZeroPtrs / mZeroSizes (used by the bulk-zero kernel) still
    // hold pre-rebind addresses, which are now freed. Rebuild from the
    // current mGrads state.
    if (rebound > 0) {
        build_zero_segments();
    }

    if (const char* dbg = std::getenv("SUROGATE_DEBUG_ARENA_CONSUME")) {
        if (std::string(dbg) == "1") {
            std::cerr << "[arena-consume accumulator] rebound=" << rebound << " skipped_no_tid=" << skipped_no_tid
                      << " skipped_non_accumulator=" << skipped_non_accumulator
                      << " skipped_size_mismatch=" << skipped_size_mismatch
                      << " arena_bytes=" << arenas.accumulator_bytes << "\n";
        }
    }
}

}  // namespace dsl
