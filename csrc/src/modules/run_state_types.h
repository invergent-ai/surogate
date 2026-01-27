// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_RUN_STATE_TYPES_H
#define SUROGATE_SRC_MODULES_RUN_STATE_TYPES_H

#include <cuda_runtime.h>
#include "utilities/tensor.h"
#include "fp8_run_state.h"
#include "fp4_run_state.h"

namespace modules {

/**
 * @brief Simplified layer activations for forward/backward
 *
 * This mirrors sLLamaLayerActivations but with simplified structure.
 * Used for initial implementation - can be replaced with modular activations later.
 */
struct SimplifiedLayerActivations {
    Tensor ln1_rstd;         ///< (B, T) - RMSNorm reciprocal std
    Tensor ln1;              ///< (B, T, C) - normalized input
    Tensor ln2_rstd;         ///< (B, T) - RMSNorm reciprocal std
    Tensor ln2;              ///< (B, T, C) - normalized input
    Tensor q_rstd;           ///< (B, T, Hq) - optional Q head RMSNorm rstd (Qwen3)
    Tensor k_rstd;           ///< (B, T, Hkv) - optional K head RMSNorm rstd (Qwen3)
    Tensor qkv;              ///< (B, T, QKV_C) - after QKV projection (+ optional QK-norm); pre-RoPE if qkv_rope is used
    Tensor qkv_rope;         ///< (B, T, QKV_C) - optional post-RoPE packed QKV (for faster QK-norm backward)
    Tensor lse;              ///< (B, num_heads, T) - log-sum-exp from attention
    Tensor att;              ///< (B, T, Hq*Hs) - attention output (pre out-proj)
    Tensor att_out;          ///< (B, T, C) - after output projection
    Tensor residual_att;     ///< (B, T, C) - residual + attention
    Tensor mlp_up;           ///< (B, T, 2*D) - gate+up projection
    Tensor swiglu;           ///< (B, T, D) - SwiGLU output
    Tensor swiglu_scale;     ///< (B*T,) - per-row scale from scaled SwiGLU (nvfp4-simple recipe)
    Tensor mlp_down;         ///< (B, T, C) - down projection

    // Mamba / SSM activations (Nemotron-H)
    Tensor mamba_gate;       ///< (B, T, D) - gate projection
    Tensor mamba_conv_in;    ///< (B, conv_dim, T) - input to conv1d (channel-first)
    Tensor mamba_u;          ///< (B, D, T) - selective scan input (hidden, transposed)
    Tensor mamba_delta;      ///< (B, D, T) - expanded dt (transposed)
    Tensor mamba_B;          ///< (B, G, N, T) - B parameters (packed)
    Tensor mamba_C;          ///< (B, G, N, T) - C parameters (packed)
    Tensor mamba_scan_out;   ///< (B, T, D) - scan output (transposed back)
    Tensor mamba_gated;      ///< (B, T, D) - scan_out * silu(gate)
    Tensor mamba_normed;     ///< (B, T, D) - RMSNorm output (input to out_proj)
    Tensor mamba_rstd;       ///< (B, T, G) - group RMSNorm rstd (float)
    Tensor mamba_x;          ///< (B, D, n_chunks, N*2) - scan intermediates (float)
};

/**
 * @brief Optional quantized forward activations used by FP8/int8 matmuls.
 *
 * When matmul_dtype == activation_dtype, these tensors are left empty (Data == nullptr).
 */
struct SimplifiedLayerQuantActivations {
    Tensor ln1;      ///< (B, T, C) in matmul_dtype
    Tensor ln2;      ///< (B, T, C) in matmul_dtype
    Tensor att;      ///< (B, T, Hq*Hs) in matmul_dtype (shared across layers)
    Tensor swiglu;   ///< (B, T, D) in matmul_dtype
};

/**
 * @brief Simplified per-layer activation gradients (for simplified backward path)
 *
 * Mirrors the legacy LLamaRunState per-layer gradient buffers closely enough
 * for the modular "simplified" backward implementation in model/modular_model.h.
 */
struct SimplifiedLayerGradients {
    Tensor d_res_ffn;   ///< (B, T, C) gradient w.r.t. (residual_att + mlp_down)
    Tensor d_res_att;   ///< (B, T, C) gradient w.r.t. residual input to attention (and att_out)
    Tensor d_ln2;       ///< (B, T, C) gradient w.r.t. LN2 output
    Tensor d_mlp_up;    ///< (B, T, 2*D) gradient w.r.t. MLP up (gate+up) output
    Tensor d_swiglu;    ///< (B, T, D) gradient w.r.t. SwiGLU output
    Tensor d_mlp_down;  ///< (B, T, C) gradient w.r.t. MLP down output (block output)
    Tensor d_att;       ///< (B, T, Hq*Hs) gradient w.r.t. attention output (pre out-proj)
    Tensor d_qkv;       ///< (B, T, QKV_C) gradient w.r.t. QKV (post RoPE)
    Tensor d_ln1;       ///< (B, T, C) gradient w.r.t. LN1 output

    // Mamba / SSM gradients (Nemotron-H)
    Tensor d_mamba_normed;   ///< (B, T, D) gradient w.r.t. normed output
    Tensor d_mamba_gated;    ///< (B, T, D) gradient w.r.t. gated input to RMSNorm
    Tensor d_mamba_scan_out; ///< (B, T, D) gradient w.r.t. scan output
    Tensor d_mamba_u;        ///< (B, D, T) gradient w.r.t. selective scan input
    Tensor d_mamba_delta;    ///< (B, D, T) gradient w.r.t. expanded dt
    Tensor d_mamba_B;        ///< (B, G, N, T) - FP32 (selective scan backward)
    Tensor d_mamba_C;        ///< (B, G, N, T) - FP32 (selective scan backward)
    Tensor d_mamba_conv_out; ///< (B, conv_dim, T) gradient w.r.t. conv1d output
};

/**
 * @brief Optional quantized backward gradients used by FP8/int8 matmuls.
 *
 * These are scratch/shared across layers in the simplified backward path.
 * When grad_quant_dtype == grad_dtype, these tensors are left empty (Data == nullptr).
 */
struct SimplifiedQuantGradients {
    Tensor d_res_ffn;   ///< (B, T, C) in grad_quant_dtype
    Tensor d_res_att;   ///< (B, T, C) in grad_quant_dtype
    Tensor d_mlp_up;    ///< (B, T, 2*D) in grad_quant_dtype
    Tensor d_qkv;       ///< (B, T, QKV_C) in grad_quant_dtype
};

/**
 * @brief Non-block activations (embeddings, final norm, output)
 */
struct NonBlockActivations {
    Tensor encoded;          ///< (B, T, C) after embedding lookup
    Tensor freq_cis;         ///< (T, 2*head_size) RoPE frequencies
    Tensor output;           ///< (B, T, V) final logits
    Tensor ln_final;         ///< (B, T, C) after final norm
    Tensor ln_final_rstd;    ///< (B, T) final norm reciprocal std
};

/**
 * @brief Non-block gradient buffers
 */
struct NonBlockGradientBuffers {
    Tensor d_ln_final;       ///< (B, T, C) gradient through final norm
    Tensor d_embeddings;     ///< (B, T, C) gradient to embeddings
};

/**
 * @brief Scratch buffers for various operations
 */
struct ScratchBuffers {
    Tensor rmsnorm_scratch;      ///< For RMSNorm backward
    Tensor matmul_bias_scratch;  ///< For fused matmul+bias
    Tensor cross_entropy_dloss;  ///< [B*T] per-token d_loss (filled with scalar)
    Tensor cross_entropy_logsumexp;       ///< [B*T] final logsumexp per token
    Tensor cross_entropy_chunk_logsumexp; ///< [B*T, n_chunks] intermediate logsumexp per chunk
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

} // namespace modules

#endif // SUROGATE_SRC_MODULES_RUN_STATE_TYPES_H
