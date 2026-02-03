// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MODEL_FACTORY_H
#define SUROGATE_SRC_MODULES_MODEL_FACTORY_H

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "config/pretrained_config.h"
#include "runtime/dsl/dsl_model.h"
#include "runtime/lora/lora_config.h"
#include "runtime/qlora/qlora_config.h"
#include "runtime/training/model.h"
#include "runtime/training/runtime_options.h"
#include "utilities/allocator.h"

namespace modules {

/**
 * @brief Factory for creating DSL-backed models.
 *
 * The legacy modular C++ model path has been removed; all models are created
 * through the DSL backend using a precompiled IR JSON payload.
 */
class ModelFactory {
public:
    /**
     * @brief Create a DSL model from PretrainedConfig.
     *
     * @param config PretrainedConfig (or derived) instance
     * @param options Runtime options (must include DslIrJson)
     * @param rank Process rank for sharding
     * @param world World size
     * @param alloc Optional tensor allocator
     * @return Unique pointer to model (as IModel)
     */
    static std::unique_ptr<IModel> create_from_pretrained_config(
        const PretrainedConfig& config,
        const RuntimeOptions& options,
        int rank,
        int world,
        const std::shared_ptr<TensorAllocator>& alloc = nullptr) {
        require_dsl_ir(options);
        return std::make_unique<dsl::DslModel>(config, options, options.DslIrJson, alloc,
                                               std::nullopt, modules::QLoRAConfig{}, rank, world);
    }

    /**
     * @brief Create a DSL LoRA model from PretrainedConfig.
     *
     * @param config PretrainedConfig (or derived) instance
     * @param lora_config LoRA configuration (rank, alpha, targets, etc.)
     * @param options Runtime options (must include DslIrJson)
     * @param comm NCCL communicator for distributed setup
     * @param alloc Tensor allocator
     * @param qlora_config Optional QLoRA configuration for quantized base weights
     * @return Unique pointer to LoRA-wrapped model (as IModel)
     */
    static std::unique_ptr<IModel> create_lora_from_pretrained_config(
        const PretrainedConfig& config,
        const ModularLoRAConfig& lora_config,
        const RuntimeOptions& options,
        NCCLCommunicator& comm,
        const std::shared_ptr<TensorAllocator>& alloc,
        const QLoRAConfig& qlora_config = QLoRAConfig{}) {
        require_dsl_ir(options);
        return std::make_unique<dsl::DslModel>(config, options, options.DslIrJson, alloc,
                                               std::optional<ModularLoRAConfig>{lora_config},
                                               qlora_config, comm.rank(), comm.world_size());
    }

private:
    static void require_dsl_ir(const RuntimeOptions& options) {
        if (options.DslIrJson.empty()) {
            throw std::runtime_error("DSL model requires RuntimeOptions.dsl_ir_json to be set");
        }
    }
};

}  // namespace modules

#endif  // SUROGATE_SRC_MODULES_MODEL_FACTORY_H
