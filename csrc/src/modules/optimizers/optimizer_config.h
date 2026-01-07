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

    // NormUon-specific parameters (future)
    float normuon_momentum = 0.95f;

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
};

} // namespace optimizers

#endif // SUROGATE_SRC_MODULES_OPTIMIZERS_OPTIMIZER_CONFIG_H
