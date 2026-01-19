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

    // MoE-specific configuration (Nemotron-H hybrid MoE)
    int NRoutedExperts = 0;                ///< Number of routed experts (0 = not MoE)
    int NumExpertsPerTok = 0;              ///< Top-k experts per token
    int MoeIntermediateSize = 0;           ///< Per-expert intermediate size (0 = use IntermediateSize)
    int MoeSharedExpertIntermediateSize = 0; ///< Shared expert size (0 = use MoeIntermediateSize)
    int NSharedExperts = 0;                ///< Number of shared experts (0 = none)
    bool NormTopkProb = false;             ///< Normalize top-k weights after selection
    float RoutedScalingFactor = 1.0f;      ///< Scale factor for routed expert outputs
    int TopkGroup = 1;                     ///< Grouped top-k selection (router)
    int NGroup = 1;                        ///< Number of expert groups (router)
    float RouterAuxLossCoef = 0.01f;       ///< Router auxiliary loss coefficient
    float RouterZLossCoef = 0.001f;        ///< Router z-loss coefficient

    NemotronHConfig() {
        Architecture = NEMOTRON_H;
    }

    [[nodiscard]] std::unique_ptr<PretrainedConfig> clone() const override {
        return std::make_unique<NemotronHConfig>(*this);
    }
};

#endif // SUROGATE_SRC_MODELS_NEMOTRON_H_CONFIG_H
