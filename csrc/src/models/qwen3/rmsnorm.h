// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3RMSNormModule - RMSNorm module for Qwen3 family models
//
// Qwen3 uses the same RMSNorm as Qwen2/LLaMA, provided here as a type alias
// for consistency with the inheritance pattern.

#ifndef SUROGATE_SRC_MODELS_QWEN3_RMSNORM_H
#define SUROGATE_SRC_MODELS_QWEN3_RMSNORM_H

#include "models/qwen25/rmsnorm.h"

namespace modules {

/**
 * @brief RMSNorm module for Qwen3 family models
 *
 * Qwen3 uses the same RMSNorm implementation as Qwen2/LLaMA.
 * Note: Qwen3's QK normalization is handled separately in the attention module.
 */
using Qwen3RMSNormModule = Qwen2RMSNormModule;

/**
 * @brief Fused Residual + RMSNorm module for Qwen3 family models
 */
using Qwen3FusedResidualRMSNormModule = Qwen2FusedResidualRMSNormModule;

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN3_RMSNORM_H
