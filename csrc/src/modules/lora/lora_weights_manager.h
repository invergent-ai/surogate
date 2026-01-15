// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_WEIGHTS_MANAGER_H
#define SUROGATE_SRC_MODULES_LORA_LORA_WEIGHTS_MANAGER_H

#include <string>
#include <vector>
#include <functional>

#include "lora_types.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"
#include "utilities/dtype.h"

class TensorAllocator;
class NCCLCommunicator;

namespace modules {

/**
 * @brief Modular LoRA weights manager
 *
 * Manages LoRA adapter weights for all layers, supporting:
 * - Sharded storage for multi-GPU
 * - Random initialization (Kaiming for A, zeros for B)
 * - Import/export from safetensors
 */
class ModularLoRAWeightsManager : public ITensorContainer {
public:
    struct Config {
        int num_layers;
        int hidden_size;
        int intermediate_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        ModularLoRAConfig lora_config;
        ETensorDType work_dtype = ETensorDType::BF16;  // compute dtype (typically base model dtype)
        int shard_idx = 0;
        int num_shards = 1;
        bool is_moe = false;  ///< True for MoE models

        // MoE-specific configuration (only used when is_moe = true)
        int num_experts = 0;              ///< Number of experts per layer
        int moe_intermediate_size = 0;    ///< Per-expert intermediate size (0 = use intermediate_size)
        bool train_router = false;         ///< Train MoE router gate during LoRA fine-tuning

        [[nodiscard]] int effective_moe_intermediate() const {
            return moe_intermediate_size > 0 ? moe_intermediate_size : intermediate_size;
        }
    };

    ModularLoRAWeightsManager(const Config& config, TensorAllocator& allocator);
    ~ModularLoRAWeightsManager() = default;

    /**
     * @brief Initialize LoRA weights randomly
     *
     * A matrices: Kaiming uniform initialization
     * B matrices: Zeros (so initial output is zero)
     */
    void random_init(int seed, NCCLCommunicator& comm);

    /**
     * @brief Import LoRA adapter from safetensors file
     */
    void import_from_file(const std::string& file_name, NCCLCommunicator& comm);

    /**
     * @brief Export LoRA adapter to safetensors file
     */
    void export_to_file(const std::string& file_name, NCCLCommunicator& comm) const;

    /**
     * @brief Get block weights for forward/backward pass
     */
    LoRABlockWeights<Tensor>& get_block(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get master block weights for optimizer
     */
    LoRABlockWeights<TensorShard>& get_master_block(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get LoRA scaling factor
     */
    [[nodiscard]] float scaling() const { return mConfig.lora_config.scaling(); }

    /**
     * @brief Check if LoRA is enabled
     */
    [[nodiscard]] bool enabled() const { return mConfig.lora_config.enabled(); }

    /**
     * @brief Get the LoRA configuration
     */
    [[nodiscard]] const ModularLoRAConfig& lora_config() const { return mConfig.lora_config; }

    /**
     * @brief Check if router training is enabled
     */
    [[nodiscard]] bool train_router() const { return mConfig.train_router; }

    /**
     * @brief Copy router weights from base model (for train_router mode)
     *
     * @param layer_idx Layer index
     * @param router_weight Source router weight from base model
     * @param stream CUDA stream
     */
    void copy_router_from_base(int layer_idx, const Tensor& router_weight, cudaStream_t stream);

    /**
     * @brief Get number of trainable parameters
     */
    [[nodiscard]] std::size_t num_parameters() const;

    // ITensorContainer interface
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

private:
    Config mConfig;
    TensorAllocator* mAllocator;

    // Master weights (sharded for multi-GPU)
    LoRAWeightsSet<TensorShard> mMaster;

    // Working weights (full precision for compute)
    LoRAWeightsSet<Tensor> mWork;

    void allocate_layer_weights(LoRALayerWeights<TensorShard>& shard,
                                 LoRALayerWeights<Tensor>& work,
                                 int in_features, int out_features,
                                 const std::string& name);
    void allocate_block_weights(int layer_idx);
    void allocate_grouped_moe_weights(LoRAGroupedExpertWeights<TensorShard>& master_moe,
                                      LoRAGroupedExpertWeights<Tensor>& work_moe,
                                      int layer_idx);
    void allocate_expert_weights(LoRAExpertWeights<TensorShard>& master_expert,
                                  LoRAExpertWeights<Tensor>& work_expert,
                                  int layer_idx, int expert_idx);
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_WEIGHTS_MANAGER_H
