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
 * @brief Simplified layer activations for forward/backward
 *
 * This mirrors sLLamaLayerActivations but with simplified structure.
 * Used for initial implementation - can be replaced with modular activations later.
 */
struct SimplifiedLayerActivations {
    // Backing storage indexed by dsl::TensorSlot. Only block-activation
    // slot indices (BlockLN1 .. BlockMoeOut) are written — the rest of the
    // enum range is zero-initialized Tensor{} padding (~4 KiB per layer).
    //
    // Access: acts[TensorSlot::BlockLN1] = ... / acts[TensorSlot::BlockLN1].Data
    //
    // Slot documentation (see TensorSlot enum):
    //   BlockLN1RSTD/LN2RSTD: (B, T) FP32 - RMSNorm reciprocal std
    //   BlockLN1/LN2: (B, T, C) - normalized input
    //   BlockQRSTD/KRSTD: (B, T, Hq|Hkv) FP32 - optional QK-norm rstd (Qwen3)
    //   BlockQKV: (B, T, QKV_C) - after QKV projection; pre-RoPE if qkv_rope used
    //   BlockQKVRoPE: (B, T, QKV_C) - optional post-RoPE packed QKV
    //   BlockLSE: (B, num_heads, T) - log-sum-exp from attention
    //   BlockAtt: (B, T, Hq*Hs) - attention output (pre out-proj)
    //   BlockAttOut: (B, T, C) - after output projection
    //   BlockResidualAtt: (B, T, C) - residual + attention
    //   BlockMLPUp: (B, T, 2*D) - gate+up projection
    //   BlockSwiGLU: (B, T, D) - SwiGLU output
    //   BlockMLPDown: (B, T, C) - down projection
    //   BlockHOut: (B, T, C) - final block output (Gemma4)
    //   BlockRouter*/Routing*/Expert*/Permuted*/Scatter*/MoeOut: MoE slots
    //
    // Mamba / SSM ops route per-layer tensors through resolve_tensor
    // ("blocks[N].mamba_*") — no per-field struct storage needed.
    static constexpr std::size_t kSize = static_cast<std::size_t>(dsl::TensorSlot::Mapped) + 1;
    std::array<Tensor, kSize> slots{};

    Tensor& operator[](dsl::TensorSlot s) {
        return slots[static_cast<std::size_t>(s)];
    }
    const Tensor& operator[](dsl::TensorSlot s) const {
        return slots[static_cast<std::size_t>(s)];
    }
};

/**
 * @brief Simplified per-layer activation gradients (for simplified backward path)
 *
 * Mirrors the legacy LLamaRunState per-layer gradient buffers closely enough
 * for the modular "simplified" backward implementation in model/modular_model.h.
 */
struct SimplifiedLayerGradients {
    // Backing storage indexed by dsl::TensorSlot. Only BlockD* slot indices
    // (BlockDLN1 .. BlockDResFFN) are written — the rest of the enum range
    // is zero-initialized Tensor{} padding.
    //
    // Access: grads[TensorSlot::BlockDLN1] = ... / grads[TensorSlot::BlockDLN1].Data
    //
    // Slot documentation:
    //   BlockDResFFN: (B, T, C) gradient w.r.t. (residual_att + mlp_down)
    //   BlockDResAtt: (B, T, C) gradient w.r.t. residual input to attention
    //   BlockDAttOut: (B, T, C) gradient w.r.t. attention output projection
    //   BlockDLN2: (B, T, C) gradient w.r.t. LN2 output
    //   BlockDMLPUp: (B, T, 2*D) gradient w.r.t. MLP up output
    //   BlockDSwiGLU: (B, T, D) gradient w.r.t. SwiGLU output
    //   BlockDMLPDown: (B, T, C) gradient w.r.t. MLP down output
    //   BlockDHOut: (B, T, C) gradient w.r.t. block final output (Gemma4)
    //   BlockDAtt: (B, T, Hq*Hs) gradient w.r.t. attention output
    //   BlockDQKV: (B, T, QKV_C) gradient w.r.t. QKV (post RoPE)
    //   BlockDLN1: (B, T, C) gradient w.r.t. LN1 output
    //
    // Mamba / SSM gradients route through resolve_tensor
    // ("d_blocks[N].mamba_*") — no per-field struct storage needed.
    static constexpr std::size_t kSize = static_cast<std::size_t>(dsl::TensorSlot::Mapped) + 1;
    std::array<Tensor, kSize> slots{};
    /// Mirrors SimplifiedLayerActivations::persist_across_layer_end for
    /// gradient slots backed by a persistent arena (Accumulator, BwdStack).
    std::array<bool, kSize> persist_across_layer_end{};

    Tensor& operator[](dsl::TensorSlot s) {
        return slots[static_cast<std::size_t>(s)];
    }
    const Tensor& operator[](dsl::TensorSlot s) const {
        return slots[static_cast<std::size_t>(s)];
    }
};

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
