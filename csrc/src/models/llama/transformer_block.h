// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// LlamaTransformerBlock - Transformer block type alias for LLaMA family models

#ifndef SUROGATE_SRC_MODELS_LLAMA_TRANSFORMER_BLOCK_H
#define SUROGATE_SRC_MODELS_LLAMA_TRANSFORMER_BLOCK_H

#include "modules/composite/transformer_block.h"
#include "models/llama/attention.h"
#include "models/llama/rmsnorm.h"

namespace modules {

/**
 * @brief Transformer block for LLaMA family models
 *
 * Composes:
 * - LlamaAttentionModule (GQA with RoPE)
 * - SwiGLUModule (standard activation)
 * - LlamaFusedResidualRMSNormModule (fused residual + norm)
 *
 * This is the standard pre-norm transformer block used by LLaMA models:
 *   x = x + Attention(RMSNorm(x))
 *   x = x + MLP(RMSNorm(x))
 */
using LlamaTransformerBlock = DenseTransformerBlock<
    LlamaAttentionModule,
    SwiGLUModule,
    LlamaFusedResidualRMSNormModule
>;

} // namespace modules

#endif // SUROGATE_SRC_MODELS_LLAMA_TRANSFORMER_BLOCK_H
