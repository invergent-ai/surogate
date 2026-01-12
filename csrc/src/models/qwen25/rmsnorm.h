// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen2RMSNormModule - RMSNorm module for Qwen2 family models
//
// Qwen2 uses the same RMSNorm as LLaMA, provided here as a type alias
// for consistency with the inheritance pattern.

#ifndef SUROGATE_SRC_MODELS_QWEN2_RMSNORM_H
#define SUROGATE_SRC_MODELS_QWEN2_RMSNORM_H

#include "models/llama/rmsnorm.h"

namespace modules {

/**
 * @brief RMSNorm module for Qwen2 family models
 *
 * Qwen2 uses the same RMSNorm implementation as LLaMA.
 * This is provided as a distinct type to match the inheritance pattern
 * and allow for future Qwen2-specific customizations.
 */
using Qwen2RMSNormModule = LlamaRMSNormModule;

/**
 * @brief Fused Residual + RMSNorm module for Qwen2 family models
 */
using Qwen2FusedResidualRMSNormModule = LlamaFusedResidualRMSNormModule;

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN2_RMSNORM_H
