// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_utils.h"
#include "utilities/dtype.h"

namespace modules {

std::size_t lora_num_parameters(const ModelConfig& model_config, const ModularLoRAConfig& lora_config) {
    if (!lora_config.enabled()) return 0;

    const std::size_t r = static_cast<std::size_t>(lora_config.rank);
    const std::size_t C = static_cast<std::size_t>(model_config.HiddenSize);
    const std::size_t D = static_cast<std::size_t>(model_config.IntermediateSize);
    const std::size_t Hq = static_cast<std::size_t>(model_config.NumQueryHeads);
    const std::size_t Hkv = static_cast<std::size_t>(model_config.NumKeyValHeads);
    const std::size_t Hs = static_cast<std::size_t>(model_config.head_size());
    const std::size_t q_out = Hq * Hs;
    const std::size_t kv_out = Hkv * Hs;

    std::size_t per_layer = 0;
    if (lora_config.applies_to_q()) per_layer += r * C + q_out * r;
    if (lora_config.applies_to_k()) per_layer += r * C + kv_out * r;
    if (lora_config.applies_to_v()) per_layer += r * C + kv_out * r;
    if (lora_config.applies_to_o()) per_layer += r * q_out + C * r;
    if (lora_config.applies_to_gate()) per_layer += r * C + D * r;
    if (lora_config.applies_to_up()) per_layer += r * C + D * r;
    if (lora_config.applies_to_down()) per_layer += r * D + C * r;

    return per_layer * static_cast<std::size_t>(model_config.NumLayers);
}

std::size_t lora_bytes(const ModelConfig& model_config, const ModularLoRAConfig& lora_config) {
    return lora_num_parameters(model_config, lora_config) * get_dtype_size(lora_config.dtype);
}

} // namespace modules
