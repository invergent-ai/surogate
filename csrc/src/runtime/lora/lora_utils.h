// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_UTILS_H
#define SUROGATE_SRC_MODULES_LORA_LORA_UTILS_H

#include <cstddef>
#include <functional>
#include <vector>
#include <string>
#include "runtime/core/model_config.h"
#include "lora_config.h"
#include "utilities/tensor_container.h"

namespace modules {

namespace detail {

struct EmptyTensorContainer final : public ITensorContainer {
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>&) override {}
};

inline ITensorContainer& empty_tensor_container() {
    static EmptyTensorContainer instance;
    return instance;
}

inline std::vector<std::string> targets_to_peft_names(const ModularLoRAConfig& cfg) {
    std::vector<std::string> out;
    out.reserve(10);
    if (cfg.applies_to_q()) out.emplace_back("q_proj");
    if (cfg.applies_to_k()) out.emplace_back("k_proj");
    if (cfg.applies_to_v()) out.emplace_back("v_proj");
    if (cfg.applies_to_o()) out.emplace_back("o_proj");
    if (cfg.applies_to_gate()) out.emplace_back("gate_proj");
    if (cfg.applies_to_gate_up()) out.emplace_back("gate_up_proj");
    if (cfg.applies_to_up()) out.emplace_back("up_proj");
    if (cfg.applies_to_down()) out.emplace_back("down_proj");
    // MoE router gate (when train_router is enabled)
    if (cfg.train_router) out.emplace_back("mlp.gate");
    return out;
}

} // namespace detail

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
