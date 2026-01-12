// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3MoEModel - Model class for Qwen3 Mixture-of-Experts architectures
//
// Extends Qwen3Model with MoE layer support. All Qwen3 MoE models use pure MoE
// architecture (decoder_sparse_step=1, mlp_only_layers=[]).
//
// Model inheritance hierarchy:
//   LlamaModel
//   └── Qwen2Model
//       └── Qwen3Model
//           └── Qwen3MoEModel
//

#ifndef SUROGATE_SRC_MODELS_QWEN3MOE_QWEN3_MOE_MODEL_H
#define SUROGATE_SRC_MODELS_QWEN3MOE_QWEN3_MOE_MODEL_H

#include "models/qwen3moe/config.h"
#include "models/qwen3moe/qwen3_moe_block.h"
#include "modules/model/modular_model.h"

namespace modules {

/**
 * @brief Model class for Qwen3 MoE architectures
 *
 * All Qwen3 MoE models use pure MoE architecture where every layer is MoE
 * (decoder_sparse_step=1, mlp_only_layers=[]).
 *
 * Qwen3MoEBlock features:
 * - QK normalization in attention (inherited from Qwen3)
 * - Qwen3RouterModule with norm_topk_prob=true
 * - Optional shared expert
 *
 * Weight I/O uses the "qwen3_moe" weight mapping, which extends "qwen3" mapping
 * with router, expert, and shared_expert weights.
 */
class Qwen3MoEModel : public ModularTransformerModel<Qwen3MoEBlock> {
public:
    using Base = ModularTransformerModel<Qwen3MoEBlock>;
    using BlockType = Qwen3MoEBlock;

    /**
     * @brief Construct a Qwen3MoEModel (pure MoE - all layers MoE)
     *
     * @param config Model configuration (should be Qwen3MoEConfig)
     * @param options Runtime options
     * @param rank Process rank for sharding
     * @param world World size
     * @param alloc Optional tensor allocator
     */
    Qwen3MoEModel(const ModelConfig& config, const ModelOptions& options,
                  int rank, int world,
                  const std::shared_ptr<TensorAllocator>& alloc = nullptr)
        : Base(config, options, rank, world, alloc) {}

    // ========================================================================
    // Architecture identification
    // ========================================================================

    /**
     * @brief Get the HuggingFace model type identifier
     */
    [[nodiscard]] std::string_view model_type() const override {
        return "Qwen3MoeForCausalLM";
    }

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] static constexpr bool is_moe_model() {
        return true;
    }

    /**
     * @brief Get the config architecture ID
     */
    [[nodiscard]] static constexpr PretrainedConfig::ArchitectureId architecture_id() {
        return PretrainedConfig::QWEN3_MOE;
    }
};

/**
 * @brief Factory function to create a Qwen3 MoE model from Qwen3MoEConfig
 *
 * @param config Qwen3MoEConfig instance
 * @param options Runtime options
 * @param rank Process rank for sharding
 * @param world World size
 * @param alloc Optional tensor allocator
 * @return Unique pointer to Qwen3MoEModel (as IModel)
 */
inline std::unique_ptr<IModel> create_qwen3_moe_model(
    const Qwen3MoEConfig& config,
    const ModelOptions& options,
    int rank, int world,
    const std::shared_ptr<TensorAllocator>& alloc = nullptr) {

    ModelConfig mod_config = ModelConfig::from_pretrained_config(config);
    return std::make_unique<Qwen3MoEModel>(mod_config, options, rank, world, alloc);
}

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN3MOE_QWEN3_MOE_MODEL_H
