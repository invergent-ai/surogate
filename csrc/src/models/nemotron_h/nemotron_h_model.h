// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// NemotronHModel - Model class for Nemotron-H hybrid architectures

#ifndef SUROGATE_SRC_MODELS_NEMOTRON_H_NEMOTRON_H_MODEL_H
#define SUROGATE_SRC_MODELS_NEMOTRON_H_NEMOTRON_H_MODEL_H

#include "models/nemotron_h/config.h"
#include "modules/composite/transformer_block.h"
#include "modules/model/modular_model.h"

namespace modules {

/**
 * @brief Model class for Nemotron-H hybrid architectures
 *
 * Uses DenseTransformerBlock with BlockSpec-driven execution to support
 * per-layer Attention / MLP / Mamba variants via ModelConfig overrides.
 */
class NemotronHModel : public ModularTransformerModel<DenseTransformerBlock<>> {
public:
    using Base = ModularTransformerModel<DenseTransformerBlock<>>;
    using BlockType = DenseTransformerBlock<>;

    NemotronHModel(const ModelConfig& config, const ModelOptions& options,
                   int rank, int world,
                   const std::shared_ptr<TensorAllocator>& alloc = nullptr)
        : Base(config, options, rank, world, alloc) {}

    [[nodiscard]] std::string_view model_type() const override {
        return "NemotronHForCausalLM";
    }

    [[nodiscard]] static constexpr bool is_moe_model() {
        return false;
    }

    [[nodiscard]] static constexpr PretrainedConfig::ArchitectureId architecture_id() {
        return PretrainedConfig::NEMOTRON_H;
    }
};

inline std::unique_ptr<IModel> create_nemotron_h_model(
    const NemotronHConfig& config,
    const ModelOptions& options,
    int rank, int world,
    const std::shared_ptr<TensorAllocator>& alloc = nullptr) {

    ModelConfig mod_config = ModelConfig::from_pretrained_config(config);
    return std::make_unique<NemotronHModel>(mod_config, options, rank, world, alloc);
}

} // namespace modules

#endif // SUROGATE_SRC_MODELS_NEMOTRON_H_NEMOTRON_H_MODEL_H
