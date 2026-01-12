// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3Config - Configuration for Qwen3 dense models.

#ifndef SUROGATE_SRC_MODELS_QWEN3_CONFIG_H
#define SUROGATE_SRC_MODELS_QWEN3_CONFIG_H

#include "models/qwen25/config.h"

/**
 * @brief Qwen3Config - Configuration for Qwen3 dense models.
 *
 * Inherits from Qwen2Config and adds QK normalization.
 * Qwen3 uses explicit head_dim and QK norm by default.
 */
struct Qwen3Config : public Qwen2Config {
    Qwen3Config() {
        Architecture = QWEN3;
        UseQKNorm = true;  // Qwen3 uses QK norm by default
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<Qwen3Config>(*this);
    }

    [[nodiscard]] bool has_qk_norm() const override { return UseQKNorm; }
};

#endif // SUROGATE_SRC_MODELS_QWEN3_CONFIG_H
