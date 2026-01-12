// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// LlamaConfig - Configuration for LLaMA family models.

#ifndef SUROGATE_SRC_MODELS_LLAMA_CONFIG_H
#define SUROGATE_SRC_MODELS_LLAMA_CONFIG_H

#include "config/pretrained_config.h"

/**
 * @brief LlamaConfig - Configuration for LLaMA family models.
 *
 * Inherits all base PretrainedConfig fields.
 * LLaMA uses standard RoPE, RMSNorm, SwiGLU, and GQA.
 */
struct LlamaConfig : public PretrainedConfig {
    LlamaConfig() {
        Architecture = LLAMA;
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<LlamaConfig>(*this);
    }
};

#endif // SUROGATE_SRC_MODELS_LLAMA_CONFIG_H
