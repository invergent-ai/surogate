// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3TransformerBlock - Transformer block type alias for Qwen3 family models

#ifndef SUROGATE_SRC_MODELS_QWEN3_TRANSFORMER_BLOCK_H
#define SUROGATE_SRC_MODELS_QWEN3_TRANSFORMER_BLOCK_H

#include "modules/composite/transformer_block.h"
#include "models/qwen3/attention.h"
#include "models/qwen3/rmsnorm.h"

namespace modules {

/**
 * @brief Transformer block for Qwen3 family models
 *
 * Composes:
 * - Qwen3AttentionModule (GQA with QK normalization + RoPE)
 * - SwiGLUModule (standard activation)
 * - Qwen3FusedResidualRMSNormModule (fused residual + norm)
 *
 * Extends Qwen2TransformerBlock with QK normalization for improved
 * training stability at larger scales.
 */
using Qwen3TransformerBlock = DenseTransformerBlock<
    Qwen3AttentionModule,
    SwiGLUModule,
    Qwen3FusedResidualRMSNormModule
>;

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN3_TRANSFORMER_BLOCK_H
