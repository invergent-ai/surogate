// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL gradient store - manages parameter gradients for DSL execution.

#ifndef SUROGATE_SRC_DSL_DSL_GRAD_STORE_H
#define SUROGATE_SRC_DSL_DSL_GRAD_STORE_H

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "utilities/allocator.h"
#include "utilities/tensor.h"

namespace dsl {

class DslParamStore;

}

class NCCLCommunicator;

namespace dsl {

/// Configuration for gradient management (mirrors ModularGradientManager::Config)
struct DslGradStoreConfig {
    int num_shards = 1;              ///< Number of ZeRO shards (world_size)
    int shard_idx = 0;               ///< This rank's shard index
    bool shard_gradients = false;    ///< ZeRO-2: shard gradients across ranks
    bool use_all_to_all_reduce = false; ///< Use all-to-all instead of reduce-scatter
    int num_layers = 0;              ///< Number of transformer layers
};

// Stores parameter gradients for DSL execution.
class DslGradStore {
public:
    DslGradStore(const DslParamStore& params,
                 const std::shared_ptr<TensorAllocator>& allocator);

    /// Configure multi-GPU gradient reduction
    void configure(const DslGradStoreConfig& config);

    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm);

    Tensor* get_param_grad(const std::string& name, bool& accumulate);

    void zero_all(cudaStream_t stream);
    void reduce_all(NCCLCommunicator& comm, cudaStream_t stream);

    /// Start async all-reduce on all gradients (non-blocking).
    /// Call wait_for_reduce() or synchronize on AllReduceDone event before using gradients.
    void reduce_all_async(NCCLCommunicator& comm, cudaStream_t stream, cudaEvent_t done_event);

    /// Notify that a layer's backward pass is complete.
    /// For ZeRO-1: triggers reduce-scatter on the last micro-step.
    /// For ZeRO-2: triggers reduce-scatter every micro-step with deferred accumulation.
    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm);

    /// Wait for a layer's reduction to complete (call before optimizer uses sharded grads)
    void wait_for_block_reduce(int layer_idx, cudaStream_t stream);

    /// Check if async reduce has been started (for avoiding redundant reduce in update)
    bool is_reduce_pending() const { return mReducePending; }
    void clear_reduce_pending() { mReducePending = false; }

    /// Check if per-layer overlapped reduction is enabled
    /// Returns true only if multi-GPU AND we have layer gradients to reduce
    bool is_overlapped_enabled() const { return mConfig.num_shards > 1 && mHasLayerGrads; }

    [[nodiscard]] bool is_first_micro_step() const { return mMicroStep == 0; }
    [[nodiscard]] bool is_last_micro_step() const { return mIsLastMicroStep; }

    const std::vector<std::string>& param_names() const { return mParamOrder; }
    const std::unordered_map<std::string, Tensor>& grads() const { return mGrads; }

    /// Get gradients for a specific layer (for per-layer operations)
    std::vector<Tensor*> get_layer_grads(int layer_idx);

private:
    void build_layer_grad_map();
    void scatter_reduce_layer(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm);
    void create_layer_events(int num_layers);
    void destroy_layer_events() noexcept;

    std::shared_ptr<TensorAllocator> mAllocator;
    std::unordered_map<std::string, Tensor> mGrads;
    std::vector<std::string> mParamOrder;
    bool mAccumulate = false;
    bool mReducePending = false;  ///< True if async reduce has been started
    int mMicroStep = 0;
    bool mIsLastMicroStep = false;

    // Per-layer gradient organization (for overlapped reduction)
    DslGradStoreConfig mConfig;
    std::vector<std::vector<std::string>> mLayerGradNames;  ///< Gradient names per layer
    std::vector<cudaEvent_t> mLayerReduceEvents;            ///< One event per layer
    bool mHasLayerGrads = false;  ///< True if we have any layer gradients (false for LoRA-only)

    // ZeRO-2 double-buffering state (for deferred accumulation)
    struct BlockState {
        int LayerIdx = -1;
        bool NeedsAccumulation = false;
        cudaEvent_t Event = nullptr;
    };
    std::array<BlockState, 2> mBlockStates;  ///< Double-buffer for overlapped layers
    std::vector<Tensor> mShardedGrads;       ///< ZeRO-2: per-layer sharded gradient storage
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_DSL_GRAD_STORE_H
