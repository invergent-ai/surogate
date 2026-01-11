// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3Model - Model class for Qwen3 dense architectures
//
// Inherits from Qwen2Model concept and uses Qwen3TransformerBlock.
// Extends Qwen2Model with QK normalization for improved training stability.
//
// Model inheritance hierarchy:
//   LlamaModel
//   └── Qwen2Model
//       └── Qwen3Model      <-- This class
//           └── Qwen3MoEModel
//

#ifndef SUROGATE_SRC_MODELS_QWEN3_QWEN3_MODEL_H
#define SUROGATE_SRC_MODELS_QWEN3_QWEN3_MODEL_H

#include "models/qwen3/config.h"
#include "models/qwen3/transformer_block.h"
#include "modules/model/modular_model.h"

namespace modules {

/**
 * @brief Model class for Qwen3 dense architectures
 *
 * Uses Qwen3TransformerBlock which composes:
 * - Qwen3AttentionModule (GQA with QK normalization + RoPE)
 * - SwiGLUModule (standard activation)
 * - Qwen3FusedResidualRMSNormModule (fused residual + norm)
 *
 * Extends Qwen2Model with:
 * - QK normalization for improved training stability at larger scales
 * - Explicit head_dim support (may differ from hidden_size/num_heads)
 *
 * Weight I/O uses the "qwen3" weight mapping, which extends "qwen2" mapping
 * with q_norm and k_norm weight tensors.
 */
class Qwen3Model : public ModularTransformerModel<Qwen3TransformerBlock> {
public:
    using Base = ModularTransformerModel<Qwen3TransformerBlock>;
    using BlockType = Qwen3TransformerBlock;

    /**
     * @brief Construct a Qwen3Model
     *
     * @param config Model configuration (should be Qwen3Config or derived)
     * @param options Runtime options
     * @param rank Process rank for sharding
     * @param world World size
     * @param alloc Optional tensor allocator
     */
    Qwen3Model(const ModelConfig& config, const ModelOptions& options,
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
        return "Qwen3ForCausalLM";
    }

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] static constexpr bool is_moe_model() {
        return false;
    }

    /**
     * @brief Get the config architecture ID
     */
    [[nodiscard]] static constexpr PretrainedConfig::ArchitectureId architecture_id() {
        return PretrainedConfig::QWEN3;
    }
};

/**
 * @brief Factory function to create a Qwen3Model from Qwen3Config
 *
 * @param config Qwen3Config instance
 * @param options Runtime options
 * @param rank Process rank for sharding
 * @param world World size
 * @param alloc Optional tensor allocator
 * @return Unique pointer to Qwen3Model (as IModel)
 */
inline std::unique_ptr<IModel> create_qwen3_model(
    const Qwen3Config& config,
    const ModelOptions& options,
    int rank, int world,
    const std::shared_ptr<TensorAllocator>& alloc = nullptr) {

    ModelConfig mod_config = ModelConfig::from_pretrained_config(config);
    return std::make_unique<Qwen3Model>(mod_config, options, rank, world, alloc);
}

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN3_QWEN3_MODEL_H
