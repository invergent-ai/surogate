// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_RUN_STATE_TYPES_H
#define SUROGATE_SRC_MODULES_RUN_STATE_TYPES_H

#include <cuda_runtime.h>
#include <array>
#include <cstddef>
#include <vector>
#include "runtime/dsl/tensor_slot.h"
#include "utilities/tensor.h"
#include "fp8_run_state.h"
#include "fp4_run_state.h"

namespace modules {

/**
 * @brief Optional quantized backward gradients used by FP8/int8 matmuls.
 *
 * These are scratch/shared across layers in the simplified backward path.
 * When grad_quant_dtype == grad_dtype, these tensors are left empty (Data == nullptr).
 */
struct SimplifiedQuantGradients {
    Tensor d_res_ffn;  ///< (B, T, C) in grad_quant_dtype
    Tensor d_res_att;  ///< (B, T, C) in grad_quant_dtype
    Tensor d_mlp_up;   ///< (B, T, 2*D) in grad_quant_dtype
    Tensor d_qkv;      ///< (B, T, QKV_C) in grad_quant_dtype
};

/**
 * @brief Non-block activations (embeddings, final norm, output)
 */
struct NonBlockActivations {
    Tensor encoded;        ///< (B, T, C) after embedding lookup
    Tensor freq_cis;       ///< (T, 2*head_size) RoPE frequencies
    Tensor output;         ///< (B, T, V) final logits
    Tensor ln_final;       ///< (B, T, C) after final norm
    Tensor ln_final_rstd;  ///< (B, T) final norm reciprocal std
};

/**
 * @brief Non-block gradient buffers
 */
struct NonBlockGradientBuffers {
    Tensor d_ln_final;    ///< (B, T, C) gradient through final norm
    Tensor d_embeddings;  ///< (B, T, C) gradient to embeddings
};

/**
 * @brief Scratch buffers for various operations
 */
struct ScratchBuffers {
    Tensor rmsnorm_scratch;                ///< For RMSNorm backward
    Tensor matmul_bias_scratch;            ///< For fused matmul+bias
    Tensor cross_entropy_dloss;            ///< [B*T] per-token d_loss (filled with scalar)
    Tensor cross_entropy_logsumexp;        ///< [B*T] final logsumexp per token
    Tensor cross_entropy_chunk_logsumexp;  ///< [B*T, n_chunks] intermediate logsumexp per chunk
    // Attention fallback buffers (FP32). Used when cuDNN backward is unavailable (e.g., GQA).
    Tensor attn_qkv_f32;
    Tensor attn_out_f32;
    Tensor attn_d_out_f32;
    Tensor attn_d_qkv_f32;
    Tensor attn_lse_f32;
    // cuDNN attention backward workspace.
    // Like legacy `LLamaRunState::CuDNNWorkspace`, this is a descriptor that is backed by the DeviceMemoryStack
    // via temp_acquire/temp_free, so it can overlap with other temporaries (e.g. output logits chunks).
    Tensor cudnn_workspace;
    Tensor encoder_bwd_scratch;  ///< For encoder backward
    Tensor encoder_bwd_indices;  ///< CPU tensor for encoder scheduling
    Tensor encoder_bwd_info;     ///< CPU tensor for encoder scheduling
    Tensor norm_buffer;          ///< For gradient norm computation
    Tensor matmul_scales;        ///< For FP8 scaling
};

/**
 * @brief State for individual residual buffers (for offloading)
 */
struct ResidualState {
    cudaEvent_t event = nullptr;
    cudaEvent_t ready_event = nullptr;
    int layer_idx = -1;
    bool is_ready = false;
};

}  // namespace modules

#endif  // SUROGATE_SRC_MODULES_RUN_STATE_TYPES_H
