// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen2TransformerBlock - Transformer block type alias for Qwen2 family models

#ifndef SUROGATE_SRC_MODELS_QWEN2_TRANSFORMER_BLOCK_H
#define SUROGATE_SRC_MODELS_QWEN2_TRANSFORMER_BLOCK_H

#include "modules/composite/transformer_block.h"
#include "models/qwen25/attention.h"
#include "models/qwen25/rmsnorm.h"

namespace modules {

/**
 * @brief Transformer block for Qwen2 family models
 *
 * Composes:
 * - Qwen2AttentionModule (GQA with RoPE, optional sliding window)
 * - SwiGLUModule (standard activation)
 * - Qwen2FusedResidualRMSNormModule (fused residual + norm)
 *
 * Extends LlamaTransformerBlock with sliding window attention support.
 */
using Qwen2TransformerBlock = DenseTransformerBlock<
    Qwen2AttentionModule,
    SwiGLUModule,
    Qwen2FusedResidualRMSNormModule
>;

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN2_TRANSFORMER_BLOCK_H
