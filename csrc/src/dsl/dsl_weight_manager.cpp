// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL weight manager implementation.

#include "dsl/dsl_weight_manager.h"

#include <algorithm>
#include <regex>
#include <stdexcept>
#include <string_view>

#include "config/pretrained_config.h"
#include "kernels/kernels.h"
#include "modules/lora/lora_config.h"
#include "training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"

namespace dsl {
namespace {

bool is_rope_param(const std::string& name) {
    return name.find("rope_freqs") != std::string::npos;
}

bool is_router_param(const std::string& name) {
    return name.find("router") != std::string::npos;
}

std::string_view trim_optional(std::string_view name) {
    if (!name.empty() && name.back() == '?') {
        return name.substr(0, name.size() - 1);
    }
    return name;
}

bool is_embedding_name(std::string_view name) {
    const std::string_view clean = trim_optional(name);
    return clean == "embedding" || clean == "embeddings" || clean == "embed_tokens";
}

bool is_lm_head_name(std::string_view name) {
    const std::string_view clean = trim_optional(name);
    return clean == "lm_head" || clean == "lm_head_weight";
}

// Augment shape env with model config values (same as dsl_runtime.cpp)
void augment_shape_env(ShapeEnv& env, const AttrMap& config) {
    auto get_long = [&](std::string_view key) -> std::optional<long> {
        auto it = config.find(std::string(key));
        if (it == config.end()) return std::nullopt;
        if (auto v = std::get_if<std::int64_t>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        if (auto v = std::get_if<double>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        return std::nullopt;
    };

    auto d_model = get_long("d_model");
    if (!d_model) d_model = get_long("hidden_size");
    auto num_q = get_long("num_query_heads");
    if (!num_q) num_q = get_long("num_attention_heads");
    auto num_kv = get_long("num_kv_heads");
    if (!num_kv) num_kv = get_long("num_key_value_heads");
    auto head_size = get_long("head_size");
    if (!head_size) head_size = get_long("head_dim");
    auto d_ff = get_long("d_ff");
    if (!d_ff) d_ff = get_long("intermediate_size");
    auto vocab = get_long("vocab_size");
    if (!vocab) vocab = get_long("vocab");
    auto max_seq = get_long("max_seq");
    if (!max_seq) max_seq = get_long("max_position_embeddings");

    if (d_model) env.values.emplace("C", *d_model);
    if (max_seq) env.values.emplace("MaxSeq", *max_seq);
    if (num_q) env.values.emplace("Hq", *num_q);
    if (num_kv) {
        env.values.emplace("Hkv", *num_kv);
    } else if (num_q) {
        env.values.emplace("Hkv", *num_q);
    }
    long Hq = env.values.count("Hq") ? env.values.at("Hq") : 0;
    long Hkv = env.values.count("Hkv") ? env.values.at("Hkv") : 0;
    long C = env.values.count("C") ? env.values.at("C") : 0;
    if (!head_size && Hq > 0 && C > 0) {
        head_size = C / Hq;
    }
    if (head_size) env.values.emplace("D", *head_size);
    if (d_ff) {
        env.values.emplace("M", *d_ff);
        env.values.emplace("MUp", 2 * (*d_ff));
    }
    if (vocab) env.values.emplace("V", *vocab);
    if (Hq > 0 && head_size) {
        env.values.emplace("AttnDim", Hq * (*head_size));
    }
    if (head_size && Hq > 0 && Hkv > 0) {
        env.values.emplace("QKV", (Hq + 2 * Hkv) * (*head_size));
    }

    // MoE dimensions
    auto num_experts = get_long("num_experts");
    auto num_experts_per_tok = get_long("num_experts_per_tok");
    if (!num_experts_per_tok) num_experts_per_tok = get_long("num_selected_experts");
    auto shared_expert_intermediate = get_long("shared_expert_intermediate");
    if (!shared_expert_intermediate) shared_expert_intermediate = get_long("shared_expert_intermediate_size");

    if (num_experts) {
        env.values.emplace("E", *num_experts);
    }
    if (num_experts_per_tok) {
        env.values.emplace("K", *num_experts_per_tok);
    }
    if (shared_expert_intermediate && *shared_expert_intermediate > 0) {
        env.values.emplace("SharedM", *shared_expert_intermediate);
        env.values.emplace("SharedMUp", 2 * (*shared_expert_intermediate));
    } else if (d_ff) {
        // Default shared expert size to regular intermediate size if not specified
        env.values.emplace("SharedM", *d_ff);
        env.values.emplace("SharedMUp", 2 * (*d_ff));
    }
}

} // namespace

DslWeightManager::DslWeightManager(const Module& module,
                                   const Graph& graph,
                                   const RuntimeOptions& options,
                                   const PretrainedConfig& config,
                                   const std::shared_ptr<TensorAllocator>& allocator,
                                   const modules::ModularLoRAConfig* lora_config,
                                   int shard_idx,
                                   int num_shards)
    : mAllocator(allocator) {
    if (!mAllocator) {
        throw std::runtime_error("DslWeightManager: allocator is null");
    }

    // Build configuration
    mConfig.num_layers = config.NumLayers;
    mConfig.hidden_size = config.HiddenSize;
    mConfig.vocab_size = config.VocabSize;
    mConfig.master_dtype = options.MasterDType.value_or(config.DType);
    mConfig.work_dtype = options.ModelType.value_or(config.DType);
    mConfig.shard_idx = shard_idx;
    mConfig.num_shards = num_shards;
    mConfig.shard_weights = options.ShardWeights;
    mConfig.offload_master = options.OffloadMaster;
    mConfig.offload_quants = options.OffloadQuants;
    mConfig.persistent_quants = options.PersistentQuants;
    mConfig.use_zero_copy = options.UseZeroCopy;
    mConfig.enable_fp8_forward = options.fp8_forward_enabled();

    // Enable streaming if sharding weights with multiple GPUs
    mStreamWeights = mConfig.shard_weights && mConfig.num_shards > 1;

    // Allocate weights
    allocate_weights(module, graph, lora_config);

    // Resolve non-block parameter names (embeddings/final_norm/lm_head)
    resolve_non_block_names();

    // Allocate prefetch buffers if streaming
    if (mStreamWeights || mConfig.offload_master) {
        allocate_prefetch_buffers();
    }

    create_cuda_resources();
}

DslWeightManager::~DslWeightManager() {
    release_cuda_resources();
}

void DslWeightManager::allocate_weights(const Module& module,
                                        const Graph& graph,
                                        const modules::ModularLoRAConfig* lora_config) {
    ShapeEnv env = make_shape_env(module, /*B=*/1, /*T=*/1);
    augment_shape_env(env, module.config);

    const bool freeze_base = lora_config && lora_config->enabled();
    const bool train_router = freeze_base && lora_config->train_router;

    // Prepare per-layer param name lists
    mBlockParamNames.resize(mConfig.num_layers);

    const bool sharded_master = mConfig.shard_weights && mConfig.num_shards > 1;

    for (const auto& kv : graph.params) {
        const std::string& name = kv.first;
        const TensorInfo& info = kv.second;

        if (is_rope_param(name)) {
            // RoPE frequencies are provided by run state
            continue;
        }

        ETensorDType dtype = info.dtype.value_or(mConfig.master_dtype);
        std::vector<long> shape = resolve_shape(info.shape, env);

        DslWeightEntry entry;
        entry.trainable = !is_rope_param(name);
        if (freeze_base) {
            entry.trainable = train_router && is_router_param(name);
        }

        // Parse layer index for block weights
        int layer_idx = -1;
        if (parse_layer_index(name, layer_idx)) {
            entry.is_block = true;
            entry.layer_idx = layer_idx;
            if (layer_idx >= 0 && layer_idx < mConfig.num_layers) {
                mBlockParamNames[layer_idx].push_back(name);
            }
        }

        // Cache global (unsharded) shape
        entry.global_shape = shape;

        // Determine allocation location for master weights
        EAllocationType master_alloc = EAllocationType::ON_DEVICE;
        if (mConfig.offload_master && entry.is_block) {
            // PINNED gives cudaHostAlloc with mapped flag, enabling zero-copy access from GPU
            master_alloc = EAllocationType::PINNED;
        }

        // Replicate embeddings and lm_head when sharding to avoid per-step all-gather
        // before forward output projection (memory trade for comm reduction).
        const bool replicate_non_block =
            sharded_master && (is_embedding_name(name) || is_lm_head_name(name));
        const bool entry_sharded = sharded_master && !replicate_non_block;

        // Allocate master weight (sharded if enabled)
        if (entry_sharded) {
            TensorShard shard = mAllocator->allocate_shard(dtype, mConfig.shard_idx, mConfig.num_shards,
                                                           name.c_str(), shape, master_alloc);
            entry.master = static_cast<Tensor>(shard);
        } else {
            entry.master = mAllocator->allocate(dtype, name.c_str(), master_alloc, shape);
        }
        entry.master_sharded = entry_sharded;

        // Allocate work weight (always on device)
        // If not streaming/offloading, master and work can share storage
        if (!mStreamWeights && !mConfig.offload_master) {
            entry.work = entry.master;  // Alias
        } else if (entry.is_block) {
            // Work weights for blocks are allocated in prefetch buffers
            entry.work = Tensor{};  // Will be set during gather
        } else {
            // Non-block weights: always keep work copy on device
            entry.work = mAllocator->allocate(mConfig.work_dtype, (name + "_work").c_str(),
                                              EAllocationType::ON_DEVICE, shape);
        }

        mWeights.emplace(name, std::move(entry));
        mParamOrder.push_back(name);
    }

    // Sort for deterministic ordering
    std::sort(mParamOrder.begin(), mParamOrder.end());
    for (auto& layer_names : mBlockParamNames) {
        std::sort(layer_names.begin(), layer_names.end());
    }
}

const DslWeightEntry* DslWeightManager::find_entry_by_name(const std::string& name) const {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        return nullptr;
    }
    return &it->second;
}

DslWeightEntry* DslWeightManager::find_entry_by_name(const std::string& name) {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        return nullptr;
    }
    return &it->second;
}

void DslWeightManager::resolve_non_block_names() {
    auto match_name = [&](const std::vector<std::string_view>& aliases) -> std::string {
        for (const auto& kv : mWeights) {
            const std::string_view clean = trim_optional(kv.first);
            for (auto alias : aliases) {
                if (clean == alias) {
                    return kv.first;
                }
            }
        }
        return {};
    };

    mEmbeddingName = match_name({"embedding", "embeddings", "embed_tokens"});
    mFinalNormName = match_name({"final_norm", "final_norm_weight", "norm"});
    mLmHeadName = match_name({"lm_head", "lm_head_weight"});
}

void DslWeightManager::allocate_prefetch_buffers() {
    if (!mStreamWeights && !mConfig.offload_master) {
        return;
    }

    // Find max buffer size needed per layer
    std::size_t max_layer_bytes = 0;
    for (int l = 0; l < mConfig.num_layers; ++l) {
        std::size_t layer_bytes = 0;
        for (const auto& name : mBlockParamNames[l]) {
            auto it = mWeights.find(name);
            if (it != mWeights.end()) {
                const auto& entry = it->second;
                if (!entry.global_shape.empty()) {
                    Tensor tmp = Tensor::empty(entry.master.DType, entry.global_shape);
                    layer_bytes += tmp.bytes();
                } else {
                    layer_bytes += entry.master.bytes();
                }
            }
        }
        max_layer_bytes = std::max(max_layer_bytes, layer_bytes);
    }

    // Allocate double buffers for prefetching
    for (int i = 0; i < kNumPrefetchBuffers; ++i) {
        mPrefetchStatus[i].layer_idx = -1;
        mPrefetchStatus[i].is_ready = true;
        mPrefetchStatus[i].fetch_pending = false;
        mPrefetchStatus[i].version = -1;

        // Allocate individual tensors for each block weight
        for (int l = 0; l < mConfig.num_layers; ++l) {
            for (const auto& name : mBlockParamNames[l]) {
                auto it = mWeights.find(name);
                if (it == mWeights.end()) continue;
                const auto& entry = it->second;
                std::vector<long> shape = entry.global_shape;
                if (shape.empty()) {
                    shape.assign(entry.master.Sizes.begin(),
                                 entry.master.Sizes.begin() + entry.master.Rank);
                }
                std::string buf_name = "prefetch_" + std::to_string(i) + "_" + name;
                Tensor buf = mAllocator->allocate(mConfig.work_dtype, buf_name.c_str(),
                                                  EAllocationType::ON_DEVICE, shape);
                mPrefetchBuffers[i].emplace(name, std::move(buf));
            }
        }
    }
}

void DslWeightManager::create_cuda_resources() {
    for (int i = 0; i < kNumPrefetchBuffers; ++i) {
        CUDA_CHECK(cudaEventCreate(&mGatherEvents[i]));
        mPrefetchStatus[i].done_event = mGatherEvents[i];
        CUDA_CHECK(cudaEventRecord(mGatherEvents[i], 0));
    }
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaEventCreate(&mNonBlockEvents[i]));
        CUDA_CHECK(cudaEventRecord(mNonBlockEvents[i], 0));
    }
    mEmbeddingsStatus.done_event = mNonBlockEvents[0];
    mFinalNormStatus.done_event = mNonBlockEvents[1];
    mLmHeadStatus.done_event = mNonBlockEvents[2];
}

void DslWeightManager::release_cuda_resources() noexcept {
    for (int i = 0; i < kNumPrefetchBuffers; ++i) {
        if (mGatherEvents[i]) {
            cudaEventDestroy(mGatherEvents[i]);
            mGatherEvents[i] = nullptr;
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (mNonBlockEvents[i]) {
            cudaEventDestroy(mNonBlockEvents[i]);
            mNonBlockEvents[i] = nullptr;
        }
    }
}

Tensor& DslWeightManager::get(const std::string& name) {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        throw std::runtime_error("DslWeightManager: missing parameter " + name);
    }
    // Return work tensor if available, otherwise master
    if (it->second.work.Data) {
        return it->second.work;
    }
    return it->second.master;
}

const Tensor& DslWeightManager::get(const std::string& name) const {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        throw std::runtime_error("DslWeightManager: missing parameter " + name);
    }
    if (it->second.work.Data) {
        return it->second.work;
    }
    return it->second.master;
}

bool DslWeightManager::has(const std::string& name) const {
    return mWeights.find(name) != mWeights.end();
}

bool DslWeightManager::is_trainable(const std::string& name) const {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) return false;
    return it->second.trainable;
}

bool DslWeightManager::is_sharded(const std::string& name) const {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) return false;
    return it->second.master_sharded;
}

Tensor& DslWeightManager::get_master(const std::string& name) {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        throw std::runtime_error("DslWeightManager: missing parameter " + name);
    }
    return it->second.master;
}

void DslWeightManager::synchronize_master(const std::string& name, cudaStream_t stream) {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) return;

    auto& entry = it->second;
    if (entry.work.Data && entry.work.Data != entry.master.Data) {
        // Copy work back to master (for offloaded weights after optimizer update)
        if (entry.master.DType == entry.work.DType) {
            CUDA_CHECK(cudaMemcpyAsync(entry.master.Data, entry.work.Data,
                                       entry.work.bytes(), cudaMemcpyDefault, stream));
        } else {
            // Dtype conversion needed
            convert_dtype(entry.master.get<float>(),
                          reinterpret_cast<const nv_bfloat16*>(entry.work.Data),
                          entry.work.nelem(), stream);
        }
    }
}

void DslWeightManager::gather_block(int layer_idx, NCCLCommunicator& comm, cudaStream_t stream) {
    if (!mStreamWeights && !mConfig.offload_master) {
        // No streaming - weights are already on device
        return;
    }

    if (layer_idx < 0 || layer_idx >= mConfig.num_layers) {
        return;
    }

    // Find available prefetch buffer
    int buf_idx = -1;
    for (int i = 0; i < kNumPrefetchBuffers; ++i) {
        auto& status = mPrefetchStatus[i];
        if (status.layer_idx == layer_idx && status.version == mVersion) {
            // Already fetched and up-to-date
            return;
        }
        if (status.is_ready && buf_idx < 0) {
            buf_idx = i;
        }
    }

    if (buf_idx < 0) {
        // No buffer available - wait for one
        buf_idx = mCurrentPrefetchBuffer;
        auto& status = mPrefetchStatus[buf_idx];
        if (status.fetch_pending) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, status.done_event, 0));
        }
    }

    auto& status = mPrefetchStatus[buf_idx];
    status.layer_idx = layer_idx;
    status.is_ready = false;
    status.fetch_pending = true;
    status.version = mVersion;

    // Begin NCCL transaction for sharded gather
    cudaEvent_t ready_event = nullptr;
    if (mConfig.shard_weights && mConfig.num_shards > 1) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ready_event, stream));
        comm.begin_transaction(ready_event);
    }

    // Copy/convert each weight in this layer
    for (const auto& name : mBlockParamNames[layer_idx]) {
        auto it = mWeights.find(name);
        if (it == mWeights.end()) continue;

        auto& entry = it->second;
        auto buf_it = mPrefetchBuffers[buf_idx].find(name);
        if (buf_it == mPrefetchBuffers[buf_idx].end()) continue;

        Tensor& work = buf_it->second;

        if (entry.master_sharded) {
            // Sharded: copy/convert local shard into staging buffer, then all-gather into full work tensor.
            std::vector<long> shard_shape(entry.master.Sizes.begin(),
                                          entry.master.Sizes.begin() + entry.master.Rank);
            Tensor staging = Tensor::from_pointer(work.Data, work.Device, work.DType, shard_shape);
            convert_to_work(entry.master, staging, stream);

            std::vector<long> global_shape = entry.global_shape;
            if (global_shape.empty()) {
                global_shape.assign(work.Sizes.begin(), work.Sizes.begin() + work.Rank);
            }
            TensorShard local(staging, mConfig.shard_idx, mConfig.num_shards, global_shape);
            comm.schedule_all_gather(local, work);
        } else {
            // Not sharded or single GPU: just copy/convert
            convert_to_work(entry.master, work, stream);
        }

        // Update entry's work pointer to prefetch buffer
        entry.work = work;
    }

    // Execute NCCL transaction and record completion
    if (mConfig.shard_weights && mConfig.num_shards > 1) {
        comm.execute_transaction(status.done_event);
        if (ready_event) {
            cudaEventDestroy(ready_event);
        }
    } else {
        // Record completion event for non-sharded path
        CUDA_CHECK(cudaEventRecord(status.done_event, stream));
    }

    mCurrentPrefetchBuffer = (buf_idx + 1) % kNumPrefetchBuffers;
}

void DslWeightManager::release_block(int layer_idx, cudaStream_t stream) {
    if (!mStreamWeights && !mConfig.offload_master) {
        return;
    }

    // Find and release the buffer holding this layer
    for (int i = 0; i < kNumPrefetchBuffers; ++i) {
        auto& status = mPrefetchStatus[i];
        if (status.layer_idx == layer_idx) {
            // Wait for any pending operations
            if (status.fetch_pending) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, status.done_event, 0));
                status.fetch_pending = false;
            }
            status.is_ready = true;
            break;
        }
    }
}

void DslWeightManager::wait_for_gather(int layer_idx, cudaStream_t stream) {
    if (!mStreamWeights && !mConfig.offload_master) {
        return;
    }

    for (int i = 0; i < kNumPrefetchBuffers; ++i) {
        auto& status = mPrefetchStatus[i];
        if (status.layer_idx == layer_idx && status.fetch_pending) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, status.done_event, 0));
            status.fetch_pending = false;
            break;
        }
    }
}

void DslWeightManager::gather_embeddings(NCCLCommunicator& comm, cudaStream_t stream) {
    if ((!mStreamWeights && !mConfig.offload_master) || mEmbeddingName.empty()) {
        return;
    }
    if (mEmbeddingsStatus.version == mVersion) {
        return;
    }

    auto* entry = find_entry_by_name(mEmbeddingName);
    if (!entry) {
        return;
    }

    CUDA_CHECK(cudaStreamWaitEvent(stream, mEmbeddingsStatus.done_event, 0));
    mEmbeddingsStatus.fetch_pending = true;
    mEmbeddingsStatus.is_ready = false;
    mEmbeddingsStatus.version = mVersion;

    Tensor& work = entry->work;
    if (entry->master_sharded) {
        std::vector<long> shard_shape(entry->master.Sizes.begin(),
                                      entry->master.Sizes.begin() + entry->master.Rank);
        Tensor staging = Tensor::from_pointer(work.Data, work.Device, work.DType, shard_shape);
        convert_to_work(entry->master, staging, stream);

        std::vector<long> global_shape = entry->global_shape;
        if (global_shape.empty()) {
            global_shape.assign(work.Sizes.begin(), work.Sizes.begin() + work.Rank);
        }
        comm.begin_transaction(stream);
        TensorShard local(staging, mConfig.shard_idx, mConfig.num_shards, global_shape);
        comm.schedule_all_gather(local, work);
        comm.execute_transaction(mEmbeddingsStatus.done_event);
        return;
    }

    convert_to_work(entry->master, work, stream);
    CUDA_CHECK(cudaEventRecord(mEmbeddingsStatus.done_event, stream));
}

void DslWeightManager::release_embeddings(cudaStream_t stream) {
    if ((!mStreamWeights && !mConfig.offload_master) || mEmbeddingName.empty()) {
        return;
    }
    CUDA_CHECK(cudaEventRecord(mEmbeddingsStatus.done_event, stream));
    mEmbeddingsStatus.fetch_pending = false;
    mEmbeddingsStatus.is_ready = true;
}

void DslWeightManager::gather_final_norm(NCCLCommunicator& comm, cudaStream_t stream) {
    if ((!mStreamWeights && !mConfig.offload_master) || mFinalNormName.empty()) {
        return;
    }
    if (mFinalNormStatus.version == mVersion) {
        return;
    }

    auto* entry = find_entry_by_name(mFinalNormName);
    if (!entry) {
        return;
    }

    CUDA_CHECK(cudaStreamWaitEvent(stream, mFinalNormStatus.done_event, 0));
    mFinalNormStatus.fetch_pending = true;
    mFinalNormStatus.is_ready = false;
    mFinalNormStatus.version = mVersion;

    Tensor& work = entry->work;
    if (entry->master_sharded) {
        std::vector<long> shard_shape(entry->master.Sizes.begin(),
                                      entry->master.Sizes.begin() + entry->master.Rank);
        Tensor staging = Tensor::from_pointer(work.Data, work.Device, work.DType, shard_shape);
        convert_to_work(entry->master, staging, stream);

        std::vector<long> global_shape = entry->global_shape;
        if (global_shape.empty()) {
            global_shape.assign(work.Sizes.begin(), work.Sizes.begin() + work.Rank);
        }
        comm.begin_transaction(stream);
        TensorShard local(staging, mConfig.shard_idx, mConfig.num_shards, global_shape);
        comm.schedule_all_gather(local, work);
        comm.execute_transaction(mFinalNormStatus.done_event);
        return;
    }

    convert_to_work(entry->master, work, stream);
    CUDA_CHECK(cudaEventRecord(mFinalNormStatus.done_event, stream));
}

void DslWeightManager::release_final_norm(cudaStream_t stream) {
    if ((!mStreamWeights && !mConfig.offload_master) || mFinalNormName.empty()) {
        return;
    }
    CUDA_CHECK(cudaEventRecord(mFinalNormStatus.done_event, stream));
    mFinalNormStatus.fetch_pending = false;
    mFinalNormStatus.is_ready = true;
}

void DslWeightManager::gather_lm_head(NCCLCommunicator& comm, cudaStream_t stream) {
    if ((!mStreamWeights && !mConfig.offload_master) || mLmHeadName.empty()) {
        return;
    }
    if (mLmHeadStatus.version == mVersion) {
        return;
    }

    auto* entry = find_entry_by_name(mLmHeadName);
    if (!entry) {
        return;
    }

    CUDA_CHECK(cudaStreamWaitEvent(stream, mLmHeadStatus.done_event, 0));
    mLmHeadStatus.fetch_pending = true;
    mLmHeadStatus.is_ready = false;
    mLmHeadStatus.version = mVersion;

    Tensor& work = entry->work;
    if (entry->master_sharded) {
        std::vector<long> shard_shape(entry->master.Sizes.begin(),
                                      entry->master.Sizes.begin() + entry->master.Rank);
        Tensor staging = Tensor::from_pointer(work.Data, work.Device, work.DType, shard_shape);
        convert_to_work(entry->master, staging, stream);

        std::vector<long> global_shape = entry->global_shape;
        if (global_shape.empty()) {
            global_shape.assign(work.Sizes.begin(), work.Sizes.begin() + work.Rank);
        }
        comm.begin_transaction(stream);
        TensorShard local(staging, mConfig.shard_idx, mConfig.num_shards, global_shape);
        comm.schedule_all_gather(local, work);
        comm.execute_transaction(mLmHeadStatus.done_event);
        return;
    }

    convert_to_work(entry->master, work, stream);
    CUDA_CHECK(cudaEventRecord(mLmHeadStatus.done_event, stream));
}

void DslWeightManager::release_lm_head(cudaStream_t stream) {
    if ((!mStreamWeights && !mConfig.offload_master) || mLmHeadName.empty()) {
        return;
    }
    CUDA_CHECK(cudaEventRecord(mLmHeadStatus.done_event, stream));
    mLmHeadStatus.fetch_pending = false;
    mLmHeadStatus.is_ready = true;
}

void DslWeightManager::invalidate() {
    ++mVersion;
}

const std::vector<std::string>& DslWeightManager::block_param_names(int layer_idx) const {
    static const std::vector<std::string> empty;
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mBlockParamNames.size())) {
        return empty;
    }
    return mBlockParamNames[layer_idx];
}

void DslWeightManager::iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) {
    for (const auto& name : mParamOrder) {
        auto it = mWeights.find(name);
        if (it == mWeights.end()) continue;
        if (it->second.master_sharded && !it->second.global_shape.empty()) {
            callback(name, TensorShard(it->second.master, mConfig.shard_idx, mConfig.num_shards, it->second.global_shape));
        } else {
            callback(name, TensorShard(it->second.master));
        }
    }
}

void DslWeightManager::convert_to_work(const Tensor& master, Tensor& work, cudaStream_t stream) {
    if (!master.Data || !work.Data || master.nelem() == 0) return;

    // Same pointer - no copy needed
    if (master.Data == work.Data) return;

    // Same dtype - direct copy
    if (master.DType == work.DType) {
        CUDA_CHECK(cudaMemcpyAsync(work.Data, master.Data, work.bytes(), cudaMemcpyDefault, stream));
        return;
    }

    // Dtype conversion
    if (master.DType == ETensorDType::FP32 && work.DType == ETensorDType::BF16) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(work.Data),
                      master.get<float>(), master.nelem(), stream);
        return;
    }
    if (master.DType == ETensorDType::BF16 && work.DType == ETensorDType::FP32) {
        convert_dtype(work.get<float>(),
                      reinterpret_cast<const nv_bfloat16*>(master.Data),
                      master.nelem(), stream);
        return;
    }

    // FP8 quantization
    if (work.DType == ETensorDType::FP8_E4M3 || work.DType == ETensorDType::FP8_E5M2) {
        if (!master.Stats) {
            throw std::runtime_error("DslWeightManager: FP8 conversion requires Stats pointer");
        }
        // Note: This assumes Stats contains [abs_max, scale] at Stats[0] and Stats[1]
        // The actual implementation would need proper abs_max computation
        CUDA_CHECK(cudaMemcpyAsync(work.Data, master.Data, work.bytes(), cudaMemcpyDefault, stream));
        return;
    }

    throw std::runtime_error("DslWeightManager: unsupported dtype conversion");
}

bool DslWeightManager::parse_layer_index(const std::string& name, int& layer_idx) {
    // Match patterns like "blocks[0].qkv_weight" or "blocks.0.qkv_weight"
    static const std::regex block_pattern(R"(blocks[\[.](\d+)[\].]?.*)");
    std::smatch match;
    if (std::regex_match(name, match, block_pattern)) {
        layer_idx = std::stoi(match[1].str());
        return true;
    }
    layer_idx = -1;
    return false;
}

} // namespace dsl
