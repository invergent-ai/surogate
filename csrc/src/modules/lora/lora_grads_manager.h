// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_GRADS_MANAGER_H
#define SUROGATE_SRC_MODULES_LORA_LORA_GRADS_MANAGER_H

#include <memory>

#include "lora_types.h"
#include "utilities/tensor.h"
#include "utilities/dtype.h"

class TensorAllocator;
class NCCLCommunicator;

namespace modules {

/**
 * @brief Modular LoRA gradients manager
 *
 * Manages gradient storage for LoRA adapter training.
 */
class ModularLoRAGradsManager {
public:
    struct Config {
        int num_layers;
        int hidden_size;
        int intermediate_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        ModularLoRAConfig lora_config;
        ETensorDType grad_dtype;
        int shard_idx = 0;
        int num_shards = 1;
        bool is_moe = false;  ///< True for MoE models

        // MoE-specific configuration (only used when is_moe = true)
        int num_experts = 0;              ///< Number of experts per layer
        int moe_intermediate_size = 0;    ///< Per-expert intermediate size (0 = use intermediate_size)

        [[nodiscard]] int effective_moe_intermediate() const {
            return moe_intermediate_size > 0 ? moe_intermediate_size : intermediate_size;
        }
    };

    ModularLoRAGradsManager(const Config& config, const std::shared_ptr<TensorAllocator>& allocator);
    ~ModularLoRAGradsManager();

    /**
     * @brief Start a micro-step (for gradient accumulation)
     */
    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);

    /**
     * @brief End a micro-step
     */
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm);

    /**
     * @brief Get full gradients for backward pass
     */
    LoRABlockWeights<Tensor>& get_block_full(int layer_idx, cudaStream_t stream,
                                              NCCLCommunicator& comm, bool& accumulate);

    /**
     * @brief Get sharded gradients for optimizer
     */
    LoRABlockWeights<TensorShard>& get_block_shard(int layer_idx, cudaStream_t stream);

    /**
     * @brief Notify gradient computation complete for a block
     */
    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm);

    [[nodiscard]] bool is_first_micro_step() const { return mIsFirstMicroStep; }
    [[nodiscard]] bool is_last_micro_step() const { return mIsLastMicroStep; }

private:
    Config mConfig;
    std::shared_ptr<TensorAllocator> mAllocator;

    // Full gradients (for backward computation)
    LoRAWeightsSet<Tensor> mFullGrads;

    // Sharded gradients (after all-reduce, for optimizer)
    LoRAWeightsSet<TensorShard> mShardedGrads;

    bool mIsFirstMicroStep = true;
    bool mIsLastMicroStep = false;

    void allocate_gradients();
    void reduce_gradients(cudaStream_t stream, NCCLCommunicator& comm);
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_GRADS_MANAGER_H
