// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen2Config - Configuration for Qwen2 family models.

#ifndef SUROGATE_SRC_MODELS_QWEN25_CONFIG_H
#define SUROGATE_SRC_MODELS_QWEN25_CONFIG_H

#include "models/llama/config.h"

/**
 * @brief Qwen2Config - Configuration for Qwen2 family models.
 *
 * Inherits from LlamaConfig and adds sliding window attention support.
 */
struct Qwen2Config : public LlamaConfig {
    // Sliding window attention (0 = disabled)
    int SlidingWindow = 0;

    Qwen2Config() {
        Architecture = QWEN2;
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<Qwen2Config>(*this);
    }

    [[nodiscard]] bool has_sliding_window() const override { return SlidingWindow > 0; }
};

#endif // SUROGATE_SRC_MODELS_QWEN25_CONFIG_H
