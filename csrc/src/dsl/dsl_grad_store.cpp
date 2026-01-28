// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL gradient store implementation.

#include "dsl/dsl_grad_store.h"

#include <stdexcept>
#include <unordered_set>

#include "dsl/dsl_param_store.h"
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

} // namespace

DslGradStore::DslGradStore(const DslParamStore& params,
                           const std::shared_ptr<TensorAllocator>& allocator,
                           bool offload_grads,
                           EAllocationType offload_alloc,
                           int num_shards,
                           bool tied_embeddings)
    : mAllocator(allocator) {
    if (!mAllocator) {
        throw std::runtime_error("DslGradStore: allocator is null");
    }

    const bool allow_offload = offload_grads && num_shards == 1;
    const EAllocationType grad_alloc = allow_offload ? offload_alloc : EAllocationType::ON_DEVICE;

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
        Tensor grad = mAllocator->allocate(weight.DType, ("d_" + name).c_str(), grad_alloc, shape);
        mGrads.emplace(name, grad);
        mParamOrder.push_back(name);
    }
    std::sort(mParamOrder.begin(), mParamOrder.end());

    // Initialize double-buffer block states
    mBlockStates[0] = {-1, false, nullptr};
    mBlockStates[1] = {-1, false, nullptr};
}

void DslGradStore::configure(const DslGradStoreConfig& config) {
    mConfig = config;
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
                    } catch (...) {}
                }
            }
            return false;
        };

        if (extract_layer("blocks.") || extract_layer("layers.") ||
            extract_layer("model.layers.") || extract_layer("model.blocks.")) {
            if (layer_idx >= 0 && layer_idx < mConfig.num_layers) {
                mLayerGradNames[layer_idx].push_back(name);
                mHasLayerGrads = true;
            }
        }
        // Non-layer params (embeddings, lm_head, final_norm) are not tracked per-layer
    }
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
    accumulate = mAccumulate;
    return &it->second;
}

void DslGradStore::zero_all(cudaStream_t stream) {
    for (auto& kv : mGrads) {
        fill_zero(kv.second, stream);
    }
}

void DslGradStore::reduce_all(NCCLCommunicator& comm, cudaStream_t stream) {
    for (auto& kv : mGrads) {
        comm.all_reduce_avg(kv.second, stream);
    }
    mReducePending = false;
}

void DslGradStore::reduce_all_async(NCCLCommunicator& comm, cudaStream_t stream, cudaEvent_t done_event) {
    // If overlapped reduction is enabled and we have layer gradients that were already reduced,
    // only reduce non-layer gradients (embeddings, lm_head, final_norm)
    if (is_overlapped_enabled() && mHasLayerGrads) {
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
                comm.all_reduce_avg(kv.second, stream);
            }
        }
    } else {
        // Fallback: reduce all gradients (original behavior)
        for (auto& kv : mGrads) {
            comm.all_reduce_avg(kv.second, stream);
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

    cudaEvent_t ev = (layer_idx < static_cast<int>(mLayerReduceEvents.size()))
                         ? mLayerReduceEvents[layer_idx]
                         : nullptr;

    if (mConfig.shard_gradients) {
        // ZeRO-2: Use reduce-scatter (batched transaction)
        comm.begin_transaction(stream);
        for (const auto& name : grad_names) {
            auto it = mGrads.find(name);
            if (it != mGrads.end() && it->second.Data && it->second.nelem() > 0) {
                comm.schedule_reduce_scatter(it->second);
            }
        }
        comm.execute_transaction(ev);
    } else {
        // DDP/ZeRO-1: Use all-reduce (one per tensor, as the API requires)
        for (const auto& name : grad_names) {
            auto it = mGrads.find(name);
            if (it != mGrads.end() && it->second.Data && it->second.nelem() > 0) {
                comm.all_reduce_avg(it->second, stream);
            }
        }
        // Record event after all reductions
        if (ev) {
            CUDA_CHECK(cudaEventRecord(ev, stream));
        }
    }
}

} // namespace dsl
