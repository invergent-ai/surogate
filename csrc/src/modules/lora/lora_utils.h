// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_UTILS_H
#define SUROGATE_SRC_MODULES_LORA_LORA_UTILS_H

#include <cstddef>
#include "modules/model_config.h"
#include "lora_config.h"

namespace modules {

/**
 * @brief Calculate number of LoRA parameters
 */
std::size_t lora_num_parameters(const ModelConfig& model_config, const ModularLoRAConfig& lora_config);

/**
 * @brief Calculate bytes required for LoRA adapter
 */
std::size_t lora_bytes(const ModelConfig& model_config, const ModularLoRAConfig& lora_config);

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_UTILS_H
