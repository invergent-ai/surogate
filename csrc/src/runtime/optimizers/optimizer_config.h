// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_OPTIMIZERS_OPTIMIZER_CONFIG_H
#define SUROGATE_SRC_MODULES_OPTIMIZERS_OPTIMIZER_CONFIG_H

#include "optimizer_base.h"

namespace optimizers {

/**
 * @brief Configuration for optimizers
 *
 * Contains all hyperparameters for supported optimizers.
 * Parameters for unused optimizers are ignored.
 */
struct OptimizerConfig {
    OptimizerType type = OptimizerType::ADAMW_8BIT;

    // Common parameters
    float learning_rate = 2e-4f;
    float weight_decay = 0.1f;
    float grad_clip = 0.0f;

    // AdamW-specific parameters
    float adamw_beta1 = 0.9f;
    float adamw_beta2 = 0.999f;
    float adamw_epsilon = 1e-8f;

    // Muon-specific parameters (future)
    float muon_momentum = 0.95f;

    // SGD-specific parameters (future)
    float sgd_momentum = 0.9f;
    bool sgd_nesterov = false;

    // NorMuon-specific parameters
    // NorMuon uses a hybrid approach: AdamW for embeddings/norms/lm_head,
    // NorMuon (orthogonalized momentum) for 2D weight matrices
    float normuon_momentum = 0.95f;     // β₁ for SGD momentum
    float normuon_beta2 = 0.95f;        // β₂ for variance EMA
    float normuon_lr = 0.02f;           // Higher default LR than AdamW
    bool normuon_cautious_wd = true;    // Use cautious (sign-aware) weight decay

    /**
     * @brief Create default AdamW 8-bit config
     */
    static OptimizerConfig adamw_8bit(float lr = 2e-4f, float beta1 = 0.9f,
                                       float beta2 = 0.999f, float epsilon = 1e-8f,
                                       float weight_decay = 0.1f, float grad_clip = 0.0f) {
        OptimizerConfig config;
        config.type = OptimizerType::ADAMW_8BIT;
        config.learning_rate = lr;
        config.adamw_beta1 = beta1;
        config.adamw_beta2 = beta2;
        config.adamw_epsilon = epsilon;
        config.weight_decay = weight_decay;
        config.grad_clip = grad_clip;
        return config;
    }

    /**
     * @brief Create default NorMuon config
     *
     * NorMuon uses a hybrid approach:
     * - AdamW 8-bit for embeddings, norms, lm_head, and 0D/1D parameters
     * - Orthogonalized momentum (NorMuon) for 2D weight matrices
     */
    static OptimizerConfig normuon(float lr = 0.02f, float momentum = 0.95f,
                                    float beta2 = 0.95f, float weight_decay = 0.01f,
                                    float grad_clip = 0.0f, bool cautious_wd = true) {
        OptimizerConfig config;
        config.type = OptimizerType::NORMUON;
        config.learning_rate = lr;
        config.normuon_momentum = momentum;
        config.normuon_beta2 = beta2;
        config.normuon_lr = lr;
        config.weight_decay = weight_decay;
        config.grad_clip = grad_clip;
        config.normuon_cautious_wd = cautious_wd;

        // Also set AdamW params for hybrid usage (embeddings, norms, etc.)
        config.adamw_beta1 = 0.9f;
        config.adamw_beta2 = 0.999f;
        config.adamw_epsilon = 1e-8f;

        return config;
    }
};

} // namespace optimizers

#endif // SUROGATE_SRC_MODULES_OPTIMIZERS_OPTIMIZER_CONFIG_H
