// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// NemotronHConfig - Configuration for Nemotron-H hybrid models.

#ifndef SUROGATE_SRC_MODELS_NEMOTRON_H_CONFIG_H
#define SUROGATE_SRC_MODELS_NEMOTRON_H_CONFIG_H

#include <string>
#include <vector>

#include "config/pretrained_config.h"

/**
 * @brief NemotronHConfig - Configuration for Nemotron-H hybrid models.
 *
 * Extends PretrainedConfig with hybrid (attention/MLP/Mamba) layer pattern
 * and Mamba-specific hyperparameters.
 */
struct NemotronHConfig : public PretrainedConfig {
    // Hybrid layer pattern
    std::vector<std::string> LayersBlockType;  ///< Per-layer block types: "mamba", "attention", "mlp"
    std::string HybridOverridePattern;         ///< Optional compact pattern (e.g., "M*-*")

    // Mamba-specific configuration
    int MambaNumHeads = 0;
    int MambaHeadDim = 0;
    int SsmStateSize = 0;
    int ConvKernel = 0;
    int NGroups = 1;
    int ChunkSize = 0;
    float TimeStepMin = 0.0f;
    float TimeStepMax = 0.0f;
    float TimeStepFloor = 0.0f;
    float TimeStepLimit = 0.0f;

    bool UseBias = false;
    bool UseConvBias = false;

    // MLP / attention configuration
    bool MlpBias = false;
    bool AttentionBias = false;
    float AttentionDropout = 0.0f;
    bool ResidualInFp32 = false;

    std::string MambaHiddenAct = "silu";
    std::string MlpHiddenAct = "silu";

    NemotronHConfig() {
        Architecture = NEMOTRON_H;
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<NemotronHConfig>(*this);
    }
};

#endif // SUROGATE_SRC_MODELS_NEMOTRON_H_CONFIG_H
