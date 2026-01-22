// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL weight manager with support for sharding, offloading, and streaming.

#ifndef SUROGATE_SRC_DSL_DSL_WEIGHT_MANAGER_H
#define SUROGATE_SRC_DSL_DSL_WEIGHT_MANAGER_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "dsl/ir.h"
#include "utilities/allocator.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"

struct RuntimeOptions;
struct PretrainedConfig;
class NCCLCommunicator;
namespace modules { struct ModularLoRAConfig; }

namespace dsl {

/**
 * @brief Status tracking for double-buffered weight prefetching.
 */
struct WeightGatherStatus {
    int layer_idx = -1;                ///< Which layer is stored in this buffer
    cudaEvent_t done_event = nullptr;  ///< Event to synchronize on
    bool fetch_pending = false;        ///< Whether a gather is in progress
    bool is_ready = true;              ///< Whether buffer is available for reuse
    int version = -1;                  ///< Cache version for invalidation
};

/**
 * @brief Configuration for DslWeightManager.
 */
struct DslWeightManagerConfig {
    int num_layers = 0;
    int hidden_size = 0;
    int vocab_size = 0;

    // Data types
    ETensorDType master_dtype = ETensorDType::BF16;  ///< Master weight dtype (typically BF16 or FP32)
    ETensorDType work_dtype = ETensorDType::BF16;    ///< Working dtype for computation

    // Sharding (ZeRO-3 style)
    int shard_idx = 0;
    int num_shards = 1;
    bool shard_weights = false;  ///< Enable weight sharding across GPUs

    // Offloading
    bool offload_master = false;     ///< Offload master weights to CPU
    bool offload_quants = false;     ///< Offload quantized weights to CPU
    bool persistent_quants = false;  ///< Keep quantized weights instead of re-quantizing
    bool use_zero_copy = false;      ///< Use zero-copy for CPU-GPU transfers

    // FP8 caching
    bool enable_fp8_forward = false;
};

/**
 * @brief Weight entry with master and work tensors.
 */
struct DslWeightEntry {
    Tensor master;           ///< Master weight (may be on CPU if offloaded)
    Tensor work;             ///< Work weight (always on GPU during computation)
    bool trainable = true;   ///< Whether this weight is trainable
    bool is_block = false;   ///< Whether this is a per-layer block weight
    int layer_idx = -1;      ///< Layer index for block weights (-1 for non-block)
};

/**
 * @brief Manages model weights with support for sharding, offloading, and streaming.
 *
 * This class provides a dual-storage model:
 * - Master weights: Full precision, may be sharded and/or offloaded to CPU
 * - Work weights: Potentially quantized, always on GPU during forward/backward
 *
 * The gather/release protocol enables efficient prefetching:
 * - gather_block(l): Load layer l weights to GPU (may overlap with compute)
 * - get_block(l): Get ready-to-use weights for layer l
 * - release_block(l): Mark layer l weights as available for reuse
 */
class DslWeightManager final : public ITensorContainer {
public:
    DslWeightManager(const Module& module,
                     const Graph& graph,
                     const RuntimeOptions& options,
                     const PretrainedConfig& config,
                     const std::shared_ptr<TensorAllocator>& allocator,
                     const modules::ModularLoRAConfig* lora_config = nullptr);
    ~DslWeightManager();

    // Weight access (simple path - all weights resident)
    Tensor& get(const std::string& name);
    const Tensor& get(const std::string& name) const;
    bool has(const std::string& name) const;
    bool is_trainable(const std::string& name) const;

    // Layer-based weight streaming protocol
    void gather_block(int layer_idx, NCCLCommunicator& comm, cudaStream_t stream);
    void release_block(int layer_idx, cudaStream_t stream);
    void gather_embeddings(NCCLCommunicator& comm, cudaStream_t stream);
    void release_embeddings(cudaStream_t stream);
    void gather_final_norm(NCCLCommunicator& comm, cudaStream_t stream);
    void release_final_norm(cudaStream_t stream);
    void gather_lm_head(NCCLCommunicator& comm, cudaStream_t stream);
    void release_lm_head(cudaStream_t stream);

    // Synchronization helpers
    void wait_for_gather(int layer_idx, cudaStream_t stream);
    void invalidate();  ///< Invalidate all cached weights (call on optimizer update)

    // Master weight access for optimizer
    Tensor& get_master(const std::string& name);
    void synchronize_master(const std::string& name, cudaStream_t stream);

    // Metadata
    const std::vector<std::string>& param_names() const { return mParamOrder; }
    const std::vector<std::string>& block_param_names(int layer_idx) const;
    int num_layers() const { return mConfig.num_layers; }
    bool is_streaming_enabled() const { return mStreamWeights; }
    bool is_offload_enabled() const { return mConfig.offload_master || mConfig.offload_quants; }

    // ITensorContainer interface (for checkpointing)
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

private:
    void allocate_weights(const Module& module, const Graph& graph,
                          const modules::ModularLoRAConfig* lora_config);
    void allocate_prefetch_buffers();
    void create_cuda_resources();
    void release_cuda_resources() noexcept;

    // Helper to convert master -> work (dtype conversion, H2D, etc.)
    void convert_to_work(const Tensor& master, Tensor& work, cudaStream_t stream);

    // Helper to parse layer index from weight name
    static bool parse_layer_index(const std::string& name, int& layer_idx);

    std::shared_ptr<TensorAllocator> mAllocator;
    DslWeightManagerConfig mConfig;

    // Weight storage
    std::unordered_map<std::string, DslWeightEntry> mWeights;
    std::vector<std::string> mParamOrder;
    std::vector<std::vector<std::string>> mBlockParamNames;  ///< Per-layer weight names

    // Streaming state
    bool mStreamWeights = false;
    int mVersion = 0;  ///< Incremented on invalidate()

    // Double-buffered prefetch (for streaming mode)
    static constexpr int kNumPrefetchBuffers = 2;
    std::array<WeightGatherStatus, kNumPrefetchBuffers> mPrefetchStatus;
    std::array<std::unordered_map<std::string, Tensor>, kNumPrefetchBuffers> mPrefetchBuffers;
    int mCurrentPrefetchBuffer = 0;

    // Non-block weight status
    WeightGatherStatus mEmbeddingsStatus;
    WeightGatherStatus mFinalNormStatus;
    WeightGatherStatus mLmHeadStatus;

    // CUDA resources
    cudaEvent_t mGatherEvents[kNumPrefetchBuffers] = {nullptr, nullptr};
    cudaEvent_t mNonBlockEvents[3] = {nullptr, nullptr, nullptr};  // emb, final_norm, lm_head
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_DSL_WEIGHT_MANAGER_H
