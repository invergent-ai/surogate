// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_RUN_STATE_CONFIG_H
#define SUROGATE_SRC_MODULES_RUN_STATE_CONFIG_H

#include "utilities/dtype.h"
#include "config/pretrained_config.h"
#include "fp8_scaling_config.h"

namespace modules {

/**
 * @brief Configuration for run state manager
 *
 * @tparam BlockConfig The configuration type for the model blocks
 */
template<typename BlockConfig>
struct ModularRunStateConfig {
    int num_layers;
    BlockConfig block_config;

    // Dimensions
    int batch_size;
    int seq_length;
    int hidden_size;
    int vocab_size;

    // Data types
    ETensorDType activation_dtype;   ///< Activation dtype (typically BF16)
    ETensorDType grad_dtype;         ///< Activation-gradient value dtype (typically BF16)
    ETensorDType matmul_dtype;       ///< Matmul/quant dtype for activations (may be FP8)
    ETensorDType grad_quant_dtype;   ///< Matmul/quant dtype for activation gradients (may be FP8)

    // FP8 forward-only mode: use FP8 for forward pass, BF16 for backward
    bool enable_fp8_forward = false;
    ETensorDType forward_matmul_dtype = ETensorDType::BF16;  ///< Forward pass matmul dtype (FP8 when enabled)

    // FP8 HYBRID mode with delayed scaling
    // Automatically enabled when enable_fp8_forward=true and grad_quant_dtype=E5M2
    bool enable_fp8_hybrid_delayed = false;
    FP8ScalingConfig fp8_scaling_config;  ///< Delayed scaling parameters (history len, margin, algo)

    // FP4 Training Options (requires Blackwell SM100+)
    // FP4 uses E2M1 format with two-level block scaling for extreme memory efficiency.

    // FP4 forward-only: quantize activations to FP4 for forward matmuls
    // Gradients remain in BF16/FP8 for stability.
    bool enable_fp4_forward = false;

    // FP4 forward+backward: use FP4 for both forward and backward matmuls
    // Backward uses stochastic rounding for gradient quantization.
    bool enable_fp4_backward = false;

    // Memory optimization
    bool offload_residuals = false;  ///< Offload residuals to CPU between layers
    int num_residual_buffers = 2;    ///< Number of double-buffered residual slots

    // Recomputation flags
    bool recompute_rmsnorm = false;
    bool recompute_qkv = false;
    bool recompute_attention = false;
    bool recompute_ffn = false;
    bool recompute_swiglu = false;
    bool recompute_block = false;    ///< Full block recomputation

    // LM head chunking for memory efficiency
    int lmhead_chunks = 1;           ///< Number of chunks for LM head computation
    int attention_bwd_chunks = 1;    ///< Number of chunks for attention backward (chunked over B)

    // Fused RoPE: compute cos/sin on-the-fly, skip precomputed freq_cis allocation
    bool use_fused_rope = false;

    // LoRA-only mode: skip storing activations not needed for LoRA backward
    // When enabled, we only store activations needed for LoRA gradient computation:
    // - ln1, ln2 (LoRA QKV/MLP inputs)
    // - att (LoRA O input + attention backward)
    // - swiglu (LoRA down input)
    // - qkv, lse, mlp_up, residual_att (for base model backward gradient flow)
    // We skip storing:
    // - ln1_rstd, ln2_rstd, q_rstd, k_rstd (only needed for frozen LN weight gradients)
    // - att_out, mlp_down (not needed for LoRA)
    bool lora_only_mode = false;

    // LoRA activation recomputation mode: recompute ln1/ln2/att during LoRA backward
    // instead of storing per-layer. This allows activation buffer sharing across layers
    // even in LoRA mode, trading compute for memory (~600MB savings for 4B model).
    // When enabled:
    // - ln1/ln2 are recomputed from residual stream during LoRA QKV/MLP backward
    // - att is recomputed during LoRA O backward
    // - swiglu is recomputed during LoRA down backward (uses existing recompute_swiglu)
    // Requires: recompute_block or (recompute_attention + recompute_ffn)
    bool recompute_lora = false;

    // Train MoE router gate during LoRA fine-tuning
    // When enabled, router gradients are computed even in lora_only mode
    bool train_router = false;

    // MoE model flag (set when model has MoE blocks)
    bool is_moe = false;

    // MoE-specific configuration (when is_moe=true)
    int num_experts = 0;

    // PretrainedConfig for IRunState base class initialization
    PretrainedConfig pretrained_config;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_RUN_STATE_CONFIG_H
