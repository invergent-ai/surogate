// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_WEIGHTS_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_WEIGHTS_H

#include "lora_model_core.h"
#include "lora_utils.h"

namespace modules {

template<typename Block>
ITensorContainer& ModularLoRAModel<Block>::weights() {
    if (lora_enabled()) return *mLoRAWeights;
    return mBaseModel->weights();
}

template<typename Block>
ITensorContainer& ModularLoRAModel<Block>::opt_momentum() {
    if (lora_enabled()) return detail::empty_tensor_container();
    return mBaseModel->opt_momentum();
}

template<typename Block>
ITensorContainer& ModularLoRAModel<Block>::opt_momentum_scales() {
    if (lora_enabled()) return detail::empty_tensor_container();
    return mBaseModel->opt_momentum_scales();
}

template<typename Block>
ITensorContainer& ModularLoRAModel<Block>::opt_variance() {
    if (lora_enabled()) return detail::empty_tensor_container();
    return mBaseModel->opt_variance();
}

template<typename Block>
ITensorContainer& ModularLoRAModel<Block>::opt_variance_scales() {
    if (lora_enabled()) return detail::empty_tensor_container();
    return mBaseModel->opt_variance_scales();
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_WEIGHTS_H
