// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_RUN_STATE_H
#define SUROGATE_SRC_MODULES_RUN_STATE_H

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <vector>

#include <cuda_bf16.h>
#include <fmt/format.h>

#include "fp8_scaling_config.h"
#include "fp8_scaling_state.h"
#include "module_concept.h"
#include "modules/matmul_context.h"
#include "training/model.h"  // For IRunState base class
#include "utilities/allocator.h"
#include "utilities/tensor.h"
#include "utilities/stack.h"
#include "utilities/utils.h"
#include "kernels/kernels.h"  // For precompute_freqs_cis

namespace modules {

/**
 * @brief Modular run state manager template
 *
 * Manages activation tensors for a model composed of modular blocks. Handles:
 * - Per-layer activation storage during forward pass
 * - Per-layer gradient buffers during backward pass
 * - Residual offloading for memory efficiency
 * - Double-buffered prefetching for offloaded residuals
 *
 * The run state provides storage that mirrors the block structure,
 * enabling layer-by-layer processing with recomputation support.
 *
 * @tparam Block The transformer block module type
 */
template<typename Block>
class ModularRunState : public IRunState {
public:
    using BlockActivations = typename Block::Activations;
    using BlockGradients = typename Block::Gradients;

    /**
     * @brief Configuration for run state manager
     */
    struct Config {
        int num_layers;
        typename Block::Config block_config;

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

        // Random Hadamard Transform (RHT) before FP4 quantization
        // RHT spreads outliers across channels, improving quantization accuracy.
        // Enabled by default when FP4 is used; can be disabled for debugging.
        bool enable_fp4_hadamard = true;

        // Scaled SwiGLU: use per-row scaling for FP4 numerical stability
        // When enabled, SwiGLU output is normalized and scale is applied after down_proj.
        // Required by nvfp4-simple recipe for training stability.
        bool enable_scaled_swiglu = false;

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

        // PretrainedConfig for IRunState base class initialization
        PretrainedConfig pretrained_config;
    };

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
    };

    /**
     * @brief Optional quantized forward activations used by FP8/int8 matmuls.
     *
     * Mirrors legacy QuantizableTensor usage for LN1/LN2/Att/SwiGLU.
     * When matmul_dtype == activation_dtype, these tensors are left empty (Data == nullptr).
     */
    struct SimplifiedLayerQuantActivations {
        Tensor ln1;      ///< (B, T, C) in matmul_dtype
        Tensor ln2;      ///< (B, T, C) in matmul_dtype
        Tensor att;      ///< (B, T, Hq*Hs) in matmul_dtype (shared across layers)
        Tensor swiglu;   ///< (B, T, D) in matmul_dtype
    };

    /**
     * @brief FP8 forward-only activation buffers (when enable_fp8_forward is set).
     *
     * These buffers are used for FP8 quantization during forward pass only.
     * They are transient and can be shared across layers since the FP8 data
     * is consumed immediately by the matmul and not needed for backward.
     * Backward pass uses BF16 cached activations for stability.
     */
    struct FP8ForwardQuantActivations {
        Tensor ln1;      ///< (B, T, C) in FP8 E4M3 - input to QKV projection
        Tensor ln2;      ///< (B, T, C) in FP8 E4M3 - input to MLP up projection
        Tensor att;      ///< (B, T, Hq*Hs) in FP8 E4M3 - input to output projection
        Tensor swiglu;   ///< (B, T, D) in FP8 E4M3 - input to MLP down projection
    };

    /**
     * @brief FP4 forward quantization buffers (when enable_fp4_forward is set).
     *
     * FP4 uses E2M1 format with two-level block scaling:
     * - Level 1: FP8 E4M3 scale per 16 consecutive values
     * - Level 2: FP32 global per-tensor scale (amax)
     *
     * These buffers are transient and shared across layers since FP4 data
     * is consumed immediately by the matmul and not needed for backward.
     * Backward pass uses BF16 cached activations for stability (or stochastic
     * rounding when enable_fp4_backward is set).
     *
     * Data layout: (M, K/2) bytes - 2 FP4 values packed per byte
     * Scale layout: (ceil(M/128), ceil(ceil(K/16)/4)*4) for F8_128x4 swizzling
     */
    struct FP4ForwardQuantActivations {
        // LN1 -> QKV projection input
        Tensor ln1_data;          ///< (B*T, C/2) packed FP4 E2M1
        Tensor ln1_scales;        ///< FP8 E4M3 block scales
        float* ln1_global_amax;   ///< Device pointer to global amax

        // LN2 -> MLP up projection input
        Tensor ln2_data;          ///< (B*T, C/2) packed FP4 E2M1
        Tensor ln2_scales;        ///< FP8 E4M3 block scales
        float* ln2_global_amax;

        // Att -> output projection input
        Tensor att_data;          ///< (B*T, AttC/2) packed FP4 E2M1
        Tensor att_scales;        ///< FP8 E4M3 block scales
        float* att_global_amax;

        // SwiGLU -> MLP down projection input
        Tensor swiglu_data;       ///< (B*T, D/2) packed FP4 E2M1
        Tensor swiglu_scales;     ///< FP8 E4M3 block scales
        float* swiglu_global_amax;

        // Hadamard transform workspace (reused across all activations)
        Tensor hadamard_workspace;  ///< (B*T, max_dim) BF16 temporary

        // Global amax storage (4 floats for ln1, ln2, att, swiglu)
        Tensor global_amax_buffer;
    };

    /**
     * @brief Simplified per-layer activation gradients (for simplified backward path)
     *
     * Mirrors the legacy LLamaRunState per-layer gradient buffers closely enough
     * for the modular "simplified" backward implementation in modular_model.h.
     */
    struct SimplifiedLayerGradients {
        Tensor d_res_ffn;   ///< (B, T, C) gradient w.r.t. (residual_att + mlp_down)
        Tensor d_res_att;   ///< (B, T, C) gradient w.r.t. residual input to attention (and att_out)
        Tensor d_ln2;       ///< (B, T, C) gradient w.r.t. LN2 output
        Tensor d_mlp_up;    ///< (B, T, 2*D) gradient w.r.t. MLP up (gate+up) output
        Tensor d_swiglu;    ///< (B, T, D) gradient w.r.t. SwiGLU output
        Tensor d_att;       ///< (B, T, Hq*Hs) gradient w.r.t. attention output (pre out-proj)
        Tensor d_qkv;       ///< (B, T, QKV_C) gradient w.r.t. QKV (post RoPE)
        Tensor d_ln1;       ///< (B, T, C) gradient w.r.t. LN1 output
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

    ModularRunState(const Config& config, DeviceMemoryStack& stack,
                    const std::shared_ptr<TensorAllocator>& allocator);
    ~ModularRunState();

    // Non-copyable
    ModularRunState(const ModularRunState&) = delete;
    ModularRunState& operator=(const ModularRunState&) = delete;

    // Movable
    ModularRunState(ModularRunState&& other) noexcept;
    ModularRunState& operator=(ModularRunState&& other) noexcept;

    // ========================================================================
    // Block activation access
    // ========================================================================

    /**
     * @brief Get activation storage for a block
     *
     * Returns mutable reference to activation storage for the given layer.
     * Storage persists across forward pass for use in backward.
     */
    BlockActivations& get_block_activations(int layer_idx) {
        return mBlockActivations[layer_idx];
    }

    /**
     * @brief Get gradient buffer for a block
     *
     * Returns mutable reference to gradient buffer for the given layer.
     * Used during backward pass computation.
     */
    BlockGradients& get_block_gradients(int layer_idx) {
        return mBlockGradients[layer_idx];
    }

    // ========================================================================
    // Residual management (with optional offloading)
    // ========================================================================

    /**
     * @brief Initiate prefetch of residual from CPU
     *
     * Only does work if residuals are offloaded to CPU.
     */
    void fetch_residual(int layer_idx, cudaStream_t fetch_stream);

    /**
     * @brief Store residual to CPU
     *
     * Only does work if residuals are offloaded to CPU.
     */
    void put_residual(int layer_idx, cudaStream_t put_stream);

    /**
     * @brief Get residual tensor for a layer
     *
     * Waits for any pending fetch to complete.
     */
    Tensor& get_residual(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get the final residual buffer (for output of fused_residual_rmsnorm of last layer).
     *
     * This buffer must not alias any per-layer residual buffers, because those are
     * needed for backward (LN1 backward for layer l consumes res_ffn(l-1)).
     */
    Tensor& get_final_residual() {
        return mFinalResidual;
    }

    /**
     * @brief Mark residual as ready for offloading
     */
    void mark_residual_ready(int layer_idx, cudaStream_t stream);

    /**
     * @brief Release residual (allow buffer reuse)
     */
    void release_residual(int layer_idx, cudaStream_t stream);

    // ========================================================================
    // MLP up buffer management (memory optimization)
    // ========================================================================

    /**
     * @brief Acquire MLP up buffer for a layer
     *
     * For memory efficiency, MLP up tensors can be shared between layers
     * when using recomputation (only one needed at a time).
     */
    Tensor acquire_mlp_up(int layer_idx);

    /**
     * @brief Release MLP up buffer
     */
    void release_mlp_up(Tensor& mlp_up);

    // ========================================================================
    // Non-block state access
    // ========================================================================

    NonBlockActivations& non_block_activations() { return mNonBlockActivations; }
    NonBlockGradientBuffers& non_block_gradients() { return mNonBlockGradients; }
    ScratchBuffers& scratch() { return mScratch; }
    [[nodiscard]] bool ffn_temps_on_stack() const { return mConfig.recompute_block; }
    [[nodiscard]] bool large_bwd_temps_on_stack() const { return mConfig.recompute_block; }

    /**
     * @brief Get simplified layer activations for a block
     *
     * Used for the initial forward/backward implementation until
     * the full modular activation system is in place.
     */
    SimplifiedLayerActivations& simplified_acts(int layer_idx) {
        return mSimplifiedActivations[layer_idx];
    }

    SimplifiedLayerGradients& simplified_grads(int layer_idx) {
        return mSimplifiedGradients[layer_idx];
    }

    SimplifiedLayerQuantActivations& simplified_quant_acts(int layer_idx) {
        return mSimplifiedQuantActivations[layer_idx];
    }

    SimplifiedQuantGradients& simplified_quant_grads() {
        return mSimplifiedQuantGrads;
    }

    [[nodiscard]] bool is_lora_only_mode() const { return mConfig.lora_only_mode; }

    // ========================================================================
    // IRunState virtual method overrides (for recipe-driven dispatch)
    // ========================================================================

    [[nodiscard]] bool has_activation_quants() const override { return mConfig.matmul_dtype != mConfig.activation_dtype; }
    [[nodiscard]] bool has_grad_quants() const override { return mConfig.grad_quant_dtype != mConfig.grad_dtype; }

    // FP8 forward-only mode accessors
    [[nodiscard]] bool has_fp8_forward() const override { return mConfig.enable_fp8_forward; }
    FP8ForwardQuantActivations& fp8_forward_quants() { return mFP8ForwardQuants; }

    // FP8 HYBRID backward mode: uses E4M3 weights × E5M2 gradients for backward matmuls
    // Enabled when: enable_fp8_forward=true AND grad_quant_dtype=E5M2
    [[nodiscard]] bool has_fp8_hybrid_backward() const override {
        return mConfig.enable_fp8_forward && mConfig.grad_quant_dtype == ETensorDType::FP8_E5M2;
    }

    // FP8 delayed scaling state (TransformerEngine-style)
    // Enabled automatically when FP8 HYBRID mode is active
    [[nodiscard]] bool has_fp8_delayed_scaling() const override { return mFP8ScalingState != nullptr; }
    [[nodiscard]] FP8ScalingState* get_fp8_scaling_state() override { return mFP8ScalingState.get(); }
    FP8ScalingState& fp8_scaling_state() { return *mFP8ScalingState; }
    const FP8ScalingState& fp8_scaling_state() const { return *mFP8ScalingState; }

    // FP4 forward-only mode accessors (Blackwell SM100+)
    [[nodiscard]] bool has_fp4_forward() const override {
        return mConfig.enable_fp4_forward || mConfig.enable_fp4_backward;
    }
    [[nodiscard]] bool has_fp4_backward() const override { return mConfig.enable_fp4_backward; }
    [[nodiscard]] bool has_fp4_hadamard() const override { return mConfig.enable_fp4_hadamard; }
    [[nodiscard]] bool has_scaled_swiglu() const override { return mConfig.enable_scaled_swiglu; }
    FP4ForwardQuantActivations& fp4_forward_quants() { return mFP4ForwardQuants; }

    /// @brief Provide FP4 forward buffers to recipes (data, scales, global amax).
    ///
    /// Note: The returned (data, scales) buffers are allocated for the cuDNN FP4 layout.
    /// Recipes that only need the global amax (e.g., CUTLASS NVFP4) can still reuse the
    /// amax pointer to avoid a separate abs_max reduction.
    [[nodiscard]] std::tuple<Tensor*, Tensor*, float*> get_fp4_forward_buffers(int op) override {
        if (!has_fp4_forward()) return {nullptr, nullptr, nullptr};
        const auto matmul_op = static_cast<MatmulOp>(op);
        switch (matmul_op) {
            case MatmulOp::QKV:
                return {&mFP4ForwardQuants.ln1_data, &mFP4ForwardQuants.ln1_scales, mFP4ForwardQuants.ln1_global_amax};
            case MatmulOp::MLPUp:
                return {&mFP4ForwardQuants.ln2_data, &mFP4ForwardQuants.ln2_scales, mFP4ForwardQuants.ln2_global_amax};
            case MatmulOp::AttnOut:
                return {&mFP4ForwardQuants.att_data, &mFP4ForwardQuants.att_scales, mFP4ForwardQuants.att_global_amax};
            case MatmulOp::MLPDown:
                return {&mFP4ForwardQuants.swiglu_data, &mFP4ForwardQuants.swiglu_scales, mFP4ForwardQuants.swiglu_global_amax};
            default:
                return {nullptr, nullptr, nullptr};
        }
    }

    [[nodiscard]] Tensor* get_hadamard_workspace() override {
        if (!has_fp4_hadamard()) return nullptr;
        if (!mFP4ForwardQuants.hadamard_workspace.Data) return nullptr;
        return &mFP4ForwardQuants.hadamard_workspace;
    }

    // ========================================================================
    // CUDA synchronization resources
    // ========================================================================

    cudaStream_t side_stream() const { return mSideStream; }
    cudaEvent_t side_stream_event() const { return mSideStreamEvent; }
    cudaEvent_t opt_embeddings_done() const { return mOptEmbeddingsDone; }
    cudaEvent_t layer_update_done(int layer_idx) const { return mLayerUpdateDone[layer_idx]; }

    // CUDA graphs (exec instances live in the run state so they persist across steps).
    cudaGraphExec_t& forward_block_graph(int layer_idx) { return mForwardBlockGraphs.at((std::size_t)layer_idx); }
    // Select backward graph based on accumulate flag: [0]=first micro-step, [1]=accumulating
    cudaGraphExec_t& backward_block_graph(int layer_idx, bool accumulate) {
        return mBackwardBlockGraphs.at((std::size_t)layer_idx)[accumulate ? 1 : 0];
    }
    cudaGraphExec_t& global_norm_graph() { return mGlobalNormGraph; }

    // Stack checkpoints for graphs (ensures temp_alloc returns consistent addresses on graph replay).
    DeviceMemoryStack::Checkpoint& forward_block_stack_checkpoint(int layer_idx) {
        return mForwardBlockStackCheckpoints.at((std::size_t)layer_idx);
    }
    DeviceMemoryStack::Checkpoint& backward_block_stack_checkpoint(int layer_idx, bool accumulate) {
        return mBackwardBlockStackCheckpoints.at((std::size_t)layer_idx)[accumulate ? 1 : 0];
    }

    void configure_forward_graphs(bool hooked) {
        if (mForwardGraphsHooked == hooked) return;
        reset_graphs(mForwardBlockGraphs);
        mForwardGraphsHooked = hooked;
    }

    void configure_backward_graphs(bool hooked) {
        if (mBackwardGraphsHooked == hooked) return;
        reset_graphs(mBackwardBlockGraphs);
        mBackwardGraphsHooked = hooked;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int batch_size() const { return mConfig.batch_size; }
    [[nodiscard]] int seq_length() const { return mConfig.seq_length; }

private:
    Config mConfig;
    std::shared_ptr<TensorAllocator> mAllocator;

    // Per-block storage
    std::vector<BlockActivations> mBlockActivations;
    std::vector<BlockGradients> mBlockGradients;

    // Simplified layer activations (for initial forward/backward implementation)
    std::vector<SimplifiedLayerActivations> mSimplifiedActivations;
    std::vector<SimplifiedLayerGradients> mSimplifiedGradients;
    std::vector<SimplifiedLayerQuantActivations> mSimplifiedQuantActivations;
    SimplifiedQuantGradients mSimplifiedQuantGrads{};
    Tensor mQuantStats{};

    // FP8 forward-only buffers (shared across layers, transient)
    FP8ForwardQuantActivations mFP8ForwardQuants{};
    Tensor mFP8ForwardStats{};  ///< Stats buffer for FP8 forward quants (4 pairs = 8 floats)

    // FP8 delayed scaling state (TransformerEngine-style)
    // Allocated when FP8 HYBRID mode is enabled
    std::unique_ptr<FP8ScalingState> mFP8ScalingState;

    // FP4 forward buffers (shared across layers, transient)
    // FP4 uses E2M1 format with two-level block scaling (requires Blackwell SM100+)
    FP4ForwardQuantActivations mFP4ForwardQuants{};

    // Shared activation buffers for recomputation modes (checkpointing).
    // These are used to avoid storing large per-layer intermediates when recomputation is enabled.
    Tensor mSharedLn1;
    Tensor mSharedLn2;
    Tensor mSharedQKV;
    Tensor mSharedAtt;
    Tensor mSharedAttOut;
    Tensor mSharedMlpUp;
    Tensor mSharedSwiGlu;
    Tensor mSharedResidualAtt;
    Tensor mSharedMlpDown;
    // Shared rstd buffers for LoRA-only mode (can reuse across layers since we don't need per-layer values)
    Tensor mSharedLn1Rstd;
    Tensor mSharedLn2Rstd;
    Tensor mSharedQRstd;
    Tensor mSharedKRstd;
    // Shared activation quant buffers (FP8/int8) for matmuls.
    Tensor mSharedLn1Quant;
    Tensor mSharedLn2Quant;
    Tensor mSharedAttQuant;
    Tensor mSharedSwiGluQuant;
    std::array<Tensor, 2> mSharedDResFFN{};
    Tensor mSharedDResAtt;
    Tensor mSharedDLn2;
    Tensor mSharedDMlpUp;
    Tensor mSharedDSwiGlu;
    Tensor mSharedDAtt;
    Tensor mSharedDQKV;
    Tensor mSharedDLn1;

    // Non-block storage
    NonBlockActivations mNonBlockActivations;
    NonBlockGradientBuffers mNonBlockGradients;
    ScratchBuffers mScratch;

    // Residual management (for offloading)
    std::vector<Tensor> mDeviceResiduals;
    std::vector<Tensor> mOffloadedResiduals;
    Tensor mFinalResidual;

    struct ResidualState {
        cudaEvent_t event = nullptr;
        cudaEvent_t ready_event = nullptr;
        int layer_idx = -1;
        bool is_ready = false;
    };
    std::vector<ResidualState> mResidualState;

    // MLP up buffer pool (for memory optimization with recomputation)
    std::vector<Tensor> mMlpUpPool;
    std::vector<bool> mMlpUpInUse;

    // CUDA synchronization
    cudaStream_t mSideStream = nullptr;
    cudaEvent_t mSideStreamEvent = nullptr;
    cudaEvent_t mOptEmbeddingsDone = nullptr;
    std::vector<cudaEvent_t> mLayerUpdateDone;

    // CUDA graph executables (optional, enabled by ModelOptions::use_cuda_graphs).
    std::vector<cudaGraphExec_t> mForwardBlockGraphs;
    // Two backward graphs per layer: [0] for accumulate=false (first micro-step),
    // [1] for accumulate=true (subsequent micro-steps with gradient accumulation).
    std::vector<std::array<cudaGraphExec_t, 2>> mBackwardBlockGraphs;
    cudaGraphExec_t mGlobalNormGraph = nullptr;
    bool mForwardGraphsHooked = false;
    bool mBackwardGraphsHooked = false;

    // Stack checkpoints for CUDA graph compatibility.
    // When graphs use temp_alloc, we must restore the stack to the same state
    // before each graph replay so that allocations return consistent addresses.
    std::vector<DeviceMemoryStack::Checkpoint> mForwardBlockStackCheckpoints;
    // Two checkpoints per layer corresponding to the two backward graphs.
    std::vector<std::array<DeviceMemoryStack::Checkpoint, 2>> mBackwardBlockStackCheckpoints;

    static void reset_graphs(std::vector<cudaGraphExec_t>& graphs) noexcept {
        for (auto& g : graphs) {
            if (g) {
                (void)cudaGraphExecDestroy(g);
                g = nullptr;
            }
        }
    }

    static void reset_graphs(std::vector<std::array<cudaGraphExec_t, 2>>& graphs) noexcept {
        for (auto& arr : graphs) {
            for (auto& g : arr) {
                if (g) {
                    (void)cudaGraphExecDestroy(g);
                    g = nullptr;
                }
            }
        }
    }

    // Internal helpers
    void allocate_block_activations(BlockActivations& acts);
    void allocate_block_gradients(BlockGradients& grads);
    void allocate_simplified_activations();
    void allocate_simplified_gradients();
    void allocate_simplified_quant_buffers();
    void allocate_non_block_state();
    void allocate_scratch_buffers(DeviceMemoryStack& stack);
    void allocate_residual_buffers();
    void create_cuda_resources();
    void release_cuda_resources() noexcept;
    int find_free_residual_buffer() const;
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Block>
ModularRunState<Block>::ModularRunState(
    const Config& config, DeviceMemoryStack& stack,
    const std::shared_ptr<TensorAllocator>& allocator)
    : IRunState(config.pretrained_config, config.batch_size, config.seq_length, allocator)
    , mConfig(config)
    , mAllocator(allocator) {

    // Bind the passed-in stack reference to IRunState::Stack for measurement purposes.
    // The modular_model.h code passes a dummy stack first to measure required size,
    // then replaces Stack with a properly allocated one.
    Stack = std::move(stack);

    // Allocate per-block storage
    mBlockActivations.resize(config.num_layers);
    mBlockGradients.resize(config.num_layers);

    for (int i = 0; i < config.num_layers; ++i) {
        allocate_block_activations(mBlockActivations[i]);
        allocate_block_gradients(mBlockGradients[i]);
    }

    // Allocate simplified activations for initial forward/backward implementation
    allocate_simplified_activations();
    allocate_simplified_gradients();
    allocate_simplified_quant_buffers();

    // Allocate non-block state
    allocate_non_block_state();

    // Allocate scratch buffers
    allocate_scratch_buffers(Stack);

    // Allocate residual buffers (always needed - per-layer buffers to avoid aliasing)
    allocate_residual_buffers();

    // Create CUDA synchronization resources
    create_cuda_resources();
}

template<typename Block>
ModularRunState<Block>::~ModularRunState() {
    release_cuda_resources();
}

template<typename Block>
ModularRunState<Block>::ModularRunState(ModularRunState&& other) noexcept
    : IRunState(std::move(other))
    , mConfig(std::move(other.mConfig))
    , mAllocator(std::move(other.mAllocator))
    , mBlockActivations(std::move(other.mBlockActivations))
    , mBlockGradients(std::move(other.mBlockGradients))
    , mSimplifiedActivations(std::move(other.mSimplifiedActivations))
    , mSimplifiedGradients(std::move(other.mSimplifiedGradients))
    , mSimplifiedQuantActivations(std::move(other.mSimplifiedQuantActivations))
    , mSimplifiedQuantGrads(std::move(other.mSimplifiedQuantGrads))
    , mQuantStats(std::move(other.mQuantStats))
    , mFP8ForwardQuants(std::move(other.mFP8ForwardQuants))
    , mFP8ForwardStats(std::move(other.mFP8ForwardStats))
    , mFP8ScalingState(std::move(other.mFP8ScalingState))
    , mFP4ForwardQuants(std::move(other.mFP4ForwardQuants))
    , mSharedLn1(std::move(other.mSharedLn1))
    , mSharedLn2(std::move(other.mSharedLn2))
    , mSharedQKV(std::move(other.mSharedQKV))
    , mSharedAtt(std::move(other.mSharedAtt))
    , mSharedAttOut(std::move(other.mSharedAttOut))
    , mSharedMlpUp(std::move(other.mSharedMlpUp))
    , mSharedSwiGlu(std::move(other.mSharedSwiGlu))
    , mSharedResidualAtt(std::move(other.mSharedResidualAtt))
    , mSharedMlpDown(std::move(other.mSharedMlpDown))
    , mSharedLn1Rstd(std::move(other.mSharedLn1Rstd))
    , mSharedLn2Rstd(std::move(other.mSharedLn2Rstd))
    , mSharedQRstd(std::move(other.mSharedQRstd))
    , mSharedKRstd(std::move(other.mSharedKRstd))
    , mSharedLn1Quant(std::move(other.mSharedLn1Quant))
    , mSharedLn2Quant(std::move(other.mSharedLn2Quant))
    , mSharedAttQuant(std::move(other.mSharedAttQuant))
    , mSharedSwiGluQuant(std::move(other.mSharedSwiGluQuant))
    , mSharedDResFFN(std::move(other.mSharedDResFFN))
    , mSharedDResAtt(std::move(other.mSharedDResAtt))
    , mSharedDLn2(std::move(other.mSharedDLn2))
    , mSharedDMlpUp(std::move(other.mSharedDMlpUp))
    , mSharedDSwiGlu(std::move(other.mSharedDSwiGlu))
    , mSharedDAtt(std::move(other.mSharedDAtt))
    , mSharedDQKV(std::move(other.mSharedDQKV))
    , mSharedDLn1(std::move(other.mSharedDLn1))
    , mNonBlockActivations(std::move(other.mNonBlockActivations))
    , mNonBlockGradients(std::move(other.mNonBlockGradients))
    , mScratch(std::move(other.mScratch))
    , mDeviceResiduals(std::move(other.mDeviceResiduals))
    , mOffloadedResiduals(std::move(other.mOffloadedResiduals))
    , mFinalResidual(std::move(other.mFinalResidual))
    , mResidualState(std::move(other.mResidualState))
    , mMlpUpPool(std::move(other.mMlpUpPool))
    , mMlpUpInUse(std::move(other.mMlpUpInUse))
    , mSideStream(other.mSideStream)
    , mSideStreamEvent(other.mSideStreamEvent)
    , mOptEmbeddingsDone(other.mOptEmbeddingsDone)
    , mLayerUpdateDone(std::move(other.mLayerUpdateDone))
    , mForwardBlockGraphs(std::move(other.mForwardBlockGraphs))
    , mBackwardBlockGraphs(std::move(other.mBackwardBlockGraphs))
    , mGlobalNormGraph(other.mGlobalNormGraph)
    , mForwardGraphsHooked(other.mForwardGraphsHooked)
    , mBackwardGraphsHooked(other.mBackwardGraphsHooked)
    , mForwardBlockStackCheckpoints(std::move(other.mForwardBlockStackCheckpoints))
    , mBackwardBlockStackCheckpoints(std::move(other.mBackwardBlockStackCheckpoints)) {
    // Clear source pointers
    other.mSideStream = nullptr;
    other.mSideStreamEvent = nullptr;
    other.mOptEmbeddingsDone = nullptr;
    other.mGlobalNormGraph = nullptr;
    other.mForwardGraphsHooked = false;
    other.mBackwardGraphsHooked = false;
}

template<typename Block>
ModularRunState<Block>& ModularRunState<Block>::operator=(ModularRunState&& other) noexcept {
    if (this != &other) {
        release_cuda_resources();
        IRunState::operator=(std::move(other));

        mConfig = std::move(other.mConfig);
        mAllocator = std::move(other.mAllocator);
        mBlockActivations = std::move(other.mBlockActivations);
        mBlockGradients = std::move(other.mBlockGradients);
        mSimplifiedActivations = std::move(other.mSimplifiedActivations);
        mSimplifiedGradients = std::move(other.mSimplifiedGradients);
        mSimplifiedQuantActivations = std::move(other.mSimplifiedQuantActivations);
        mSimplifiedQuantGrads = std::move(other.mSimplifiedQuantGrads);
        mQuantStats = std::move(other.mQuantStats);
        mFP8ForwardQuants = std::move(other.mFP8ForwardQuants);
        mFP8ForwardStats = std::move(other.mFP8ForwardStats);
        mFP8ScalingState = std::move(other.mFP8ScalingState);
        mFP4ForwardQuants = std::move(other.mFP4ForwardQuants);
        mSharedLn1 = std::move(other.mSharedLn1);
        mSharedLn2 = std::move(other.mSharedLn2);
        mSharedQKV = std::move(other.mSharedQKV);
        mSharedAtt = std::move(other.mSharedAtt);
        mSharedAttOut = std::move(other.mSharedAttOut);
        mSharedMlpUp = std::move(other.mSharedMlpUp);
        mSharedSwiGlu = std::move(other.mSharedSwiGlu);
        mSharedResidualAtt = std::move(other.mSharedResidualAtt);
        mSharedMlpDown = std::move(other.mSharedMlpDown);
        mSharedLn1Rstd = std::move(other.mSharedLn1Rstd);
        mSharedLn2Rstd = std::move(other.mSharedLn2Rstd);
        mSharedQRstd = std::move(other.mSharedQRstd);
        mSharedKRstd = std::move(other.mSharedKRstd);
        mSharedLn1Quant = std::move(other.mSharedLn1Quant);
        mSharedLn2Quant = std::move(other.mSharedLn2Quant);
        mSharedAttQuant = std::move(other.mSharedAttQuant);
        mSharedSwiGluQuant = std::move(other.mSharedSwiGluQuant);
        mSharedDResFFN = std::move(other.mSharedDResFFN);
        mSharedDResAtt = std::move(other.mSharedDResAtt);
        mSharedDLn2 = std::move(other.mSharedDLn2);
        mSharedDMlpUp = std::move(other.mSharedDMlpUp);
        mSharedDSwiGlu = std::move(other.mSharedDSwiGlu);
        mSharedDAtt = std::move(other.mSharedDAtt);
        mSharedDQKV = std::move(other.mSharedDQKV);
        mSharedDLn1 = std::move(other.mSharedDLn1);
        mNonBlockActivations = std::move(other.mNonBlockActivations);
        mNonBlockGradients = std::move(other.mNonBlockGradients);
        mScratch = std::move(other.mScratch);
        mDeviceResiduals = std::move(other.mDeviceResiduals);
        mOffloadedResiduals = std::move(other.mOffloadedResiduals);
        mFinalResidual = std::move(other.mFinalResidual);
        mResidualState = std::move(other.mResidualState);
        mMlpUpPool = std::move(other.mMlpUpPool);
        mMlpUpInUse = std::move(other.mMlpUpInUse);
        mSideStream = other.mSideStream;
        mSideStreamEvent = other.mSideStreamEvent;
        mOptEmbeddingsDone = other.mOptEmbeddingsDone;
        mLayerUpdateDone = std::move(other.mLayerUpdateDone);
        mForwardBlockGraphs = std::move(other.mForwardBlockGraphs);
        mBackwardBlockGraphs = std::move(other.mBackwardBlockGraphs);
        mGlobalNormGraph = other.mGlobalNormGraph;
        mForwardGraphsHooked = other.mForwardGraphsHooked;
        mBackwardGraphsHooked = other.mBackwardGraphsHooked;
        mForwardBlockStackCheckpoints = std::move(other.mForwardBlockStackCheckpoints);
        mBackwardBlockStackCheckpoints = std::move(other.mBackwardBlockStackCheckpoints);

        other.mSideStream = nullptr;
        other.mSideStreamEvent = nullptr;
        other.mOptEmbeddingsDone = nullptr;
        other.mGlobalNormGraph = nullptr;
        other.mForwardGraphsHooked = false;
        other.mBackwardGraphsHooked = false;
    }
    return *this;
}

template<typename Block>
void ModularRunState<Block>::fetch_residual(int layer_idx, cudaStream_t fetch_stream) {
    if (!mConfig.offload_residuals) {
        return;  // No fetch needed
    }

    const int buf_idx = layer_idx % mConfig.num_residual_buffers;
    auto& status = mResidualState.at((std::size_t)buf_idx);

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.event, 0));
    status.layer_idx = layer_idx;
    status.is_ready = false;

    // Host -> device prefetch into the staging buffer.
    const size_t size = mOffloadedResiduals.at(layer_idx).bytes();
    CUDA_CHECK(cudaMemcpyAsync(
        mDeviceResiduals.at((std::size_t)buf_idx).Data,
        mOffloadedResiduals.at(layer_idx).Data,
        size,
        cudaMemcpyHostToDevice,
        fetch_stream));

    CUDA_CHECK(cudaEventRecord(status.event, fetch_stream));
}

template<typename Block>
void ModularRunState<Block>::put_residual(int layer_idx, cudaStream_t put_stream) {
    if (!mConfig.offload_residuals) {
        return;  // No put needed
    }

    const int buf_idx = layer_idx % mConfig.num_residual_buffers;
    auto& status = mResidualState.at((std::size_t)buf_idx);
    status.is_ready = false;
    if (status.layer_idx != layer_idx) {
        throw std::logic_error("ModularRunState::put_residual: mismatched layer index");
    }

    // Wait until the producing stream has marked the residual as ready.
    CUDA_CHECK(cudaStreamWaitEvent(put_stream, status.ready_event, 0));

    const size_t size = mDeviceResiduals.at((std::size_t)buf_idx).bytes();
    CUDA_CHECK(cudaMemcpyAsync(
        mOffloadedResiduals.at(layer_idx).Data,
        mDeviceResiduals.at((std::size_t)buf_idx).Data,
        size,
        cudaMemcpyDeviceToHost,
        put_stream));

    CUDA_CHECK(cudaEventRecord(status.event, put_stream));
}

template<typename Block>
Tensor& ModularRunState<Block>::get_residual(int layer_idx, cudaStream_t stream) {
    if (!mConfig.offload_residuals) {
        return mDeviceResiduals[layer_idx];
    }

    const int buf_idx = layer_idx % mConfig.num_residual_buffers;
    auto& status = mResidualState.at((std::size_t)buf_idx);
    if (!status.is_ready) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, status.event, 0));
        status.is_ready = true;
    }
    return mDeviceResiduals.at((std::size_t)buf_idx);
}

template<typename Block>
void ModularRunState<Block>::mark_residual_ready(int layer_idx, cudaStream_t stream) {
    if (!mConfig.offload_residuals) {
        return;
    }

    const int buf_idx = layer_idx % mConfig.num_residual_buffers;
    auto& status = mResidualState.at((std::size_t)buf_idx);
    status.layer_idx = layer_idx;
    CUDA_CHECK(cudaEventRecord(status.ready_event, stream));
}

template<typename Block>
void ModularRunState<Block>::release_residual(int layer_idx, cudaStream_t stream) {
    if (!mConfig.offload_residuals) {
        return;
    }

    const int buf_idx = layer_idx % mConfig.num_residual_buffers;
    auto& status = mResidualState.at((std::size_t)buf_idx);
    status.is_ready = false;
    CUDA_CHECK(cudaEventRecord(status.event, stream));
}

template<typename Block>
Tensor ModularRunState<Block>::acquire_mlp_up(int layer_idx) {
    // Find a free buffer in the pool
    for (size_t i = 0; i < mMlpUpPool.size(); ++i) {
        if (!mMlpUpInUse[i]) {
            mMlpUpInUse[i] = true;
            return mMlpUpPool[i];
        }
    }

    // Allocate a new buffer if none available
    // This shouldn't happen with proper recomputation scheduling
    long B = mConfig.batch_size;
    long T = mConfig.seq_length;
    long intermediate_size = mConfig.block_config.intermediate_size * 2;  // For gate + up

    Tensor new_buffer = mAllocator->allocate(
        mConfig.activation_dtype, "mlp_up_pool",
        EAllocationType::ON_DEVICE,
        {B, T, intermediate_size});

    mMlpUpPool.push_back(new_buffer);
    mMlpUpInUse.push_back(true);
    return new_buffer;
}

template<typename Block>
void ModularRunState<Block>::release_mlp_up(Tensor& mlp_up) {
    for (size_t i = 0; i < mMlpUpPool.size(); ++i) {
        if (mMlpUpPool[i].Data == mlp_up.Data) {
            mMlpUpInUse[i] = false;
            return;
        }
    }
}

template<typename Block>
void ModularRunState<Block>::allocate_block_activations(BlockActivations& acts) {
    // Stub - actual implementation depends on Block::Activations structure
    // Each block type defines its own activation tensors
}

template<typename Block>
void ModularRunState<Block>::allocate_block_gradients(BlockGradients& grads) {
    // Stub - actual implementation depends on Block::Gradients structure
    // Each block type defines its own gradient tensors
}

template<typename Block>
void ModularRunState<Block>::allocate_simplified_activations() {
    long B = mConfig.batch_size;
    long T = mConfig.seq_length;
    long C = mConfig.hidden_size;
    long D = mConfig.block_config.intermediate_size;
    int Hq = mConfig.block_config.num_query_heads;
    int Hkv = mConfig.block_config.num_kv_heads;
    int HS = mConfig.block_config.head_size;
    long AttC = static_cast<long>(HS) * static_cast<long>(Hq);
    long qkv_channels = HS * (Hq + 2 * Hkv);
    const bool use_qk_norm = [&]() -> bool {
        if constexpr (requires { mConfig.block_config.use_qk_norm; }) return mConfig.block_config.use_qk_norm;
        return false;
    }();

    auto dtype = mConfig.activation_dtype;
    auto kind = EAllocationType::ON_DEVICE;

    // Mirror legacy recompute dependencies:
    // - LN1 can be reused if recompute-norm or recompute-att or recompute-block
    // - LN2 can be reused if recompute-norm or recompute-ffn or recompute-block
    // - QKV can be reused if recompute-qkv or recompute-att or recompute-block
    // - SwiGLU can be reused if recompute-swiglu or recompute-ffn or recompute-block
    //
    // IMPORTANT: In LoRA-only mode, we CANNOT share ln1, ln2, att, swiglu across layers because
    // LoRA backward hooks read these activations from each layer. If they're shared, they get
    // overwritten by the forward pass of subsequent layers before we can use them in backward.
    // EXCEPTION: When recompute_lora is enabled, we CAN share ln1/ln2 because LoRA backward will
    // recompute them on-the-fly instead of reading stored per-layer values.
    // NOTE: att is still kept per-layer in LoRA mode even with recompute_lora because recomputing
    // attention is expensive (requires full QKV + attention forward). This is only needed when
    // O projection has LoRA adapters. Future optimization: skip sharing att only when O is targeted.
    const bool lora_only = mConfig.lora_only_mode;
    const bool lora_can_share_ln = !lora_only || mConfig.recompute_lora;  // ln1/ln2 can be shared with recompute_lora
    const bool lora_can_share_att = !lora_only;  // att still needs per-layer storage in LoRA mode (for O proj)
    const bool lora_can_share_swiglu = !lora_only;  // swiglu needs per-layer storage in LoRA mode (for down proj)
    const bool share_ln1 = lora_can_share_ln && (mConfig.recompute_rmsnorm || mConfig.recompute_attention || mConfig.recompute_block);
    const bool share_ln2 = lora_can_share_ln && (mConfig.recompute_rmsnorm || mConfig.recompute_ffn || mConfig.recompute_block);
    const bool share_qkv = mConfig.recompute_qkv || mConfig.recompute_attention || mConfig.recompute_block;
    const bool share_att = lora_can_share_att && (mConfig.recompute_attention || mConfig.recompute_block);
    const bool share_mlp_up = mConfig.recompute_ffn || mConfig.recompute_block;
    const bool share_swiglu = lora_can_share_swiglu && (mConfig.recompute_swiglu || mConfig.recompute_ffn || mConfig.recompute_block);
    const bool ffn_temps_on_stack = mConfig.recompute_block;
    const bool share_residual_intermediates = mConfig.recompute_block;

    // Allocate shared buffers once if needed.
    if (share_ln1 && !mSharedLn1.Data) {
        mSharedLn1 = mAllocator->allocate(dtype, "ln1_shared", kind, {B, T, C});
    }
    if (share_ln2 && !mSharedLn2.Data) {
        mSharedLn2 = mAllocator->allocate(dtype, "ln2_shared", kind, {B, T, C});
    }
    if (share_qkv && !mSharedQKV.Data) {
        mSharedQKV = mAllocator->allocate(dtype, "qkv_shared", kind, {B, T, qkv_channels});
    }
    if (share_att && !mSharedAtt.Data) {
        mSharedAtt = mAllocator->allocate(dtype, "att_shared", kind, {B, T, AttC});
        mSharedAttOut = mAllocator->allocate(dtype, "att_out_shared", kind, {B, T, C});
    }
    if (share_mlp_up && !ffn_temps_on_stack && !mSharedMlpUp.Data) {
        mSharedMlpUp = mAllocator->allocate(dtype, "mlp_up_shared", kind, {B, T, 2 * D});
    }
    if (share_swiglu && !ffn_temps_on_stack && !mSharedSwiGlu.Data) {
        mSharedSwiGlu = mAllocator->allocate(dtype, "swiglu_shared", kind, {B, T, D});
    }
    if (share_residual_intermediates && !mSharedResidualAtt.Data) {
        mSharedResidualAtt = mAllocator->allocate(dtype, "residual_att_shared", kind, {B, T, C});
        mSharedMlpDown = mAllocator->allocate(dtype, "mlp_down_shared", kind, {B, T, C});
    }

    mSimplifiedActivations.resize(mConfig.num_layers);

    // Performance optimization for Qwen3-style QK-norm:
    // When we are not recomputing QKV/attention, keep a separate post-RoPE packed QKV buffer
    // so backward can consume pre-RoPE activations directly (avoids an extra rope_backward on activations).
    const bool cache_pre_rope_qkv = use_qk_norm && !share_qkv;

    // LoRA-only mode: skip allocating activations not needed for LoRA backward.
    // We still need: ln1, ln2, att, swiglu (LoRA inputs), qkv, lse, mlp_up, residual_att, ln1_rstd, ln2_rstd (for base backward flow).
    // We can skip: q_rstd, k_rstd (only for QK-norm weight grads), att_out, mlp_down.
    // Note: lora_only is already declared above where we disable sharing for LoRA-needed activations.
    // IMPORTANT: We CANNOT share ln1_rstd/ln2_rstd across layers even in LoRA mode because
    // rmsnorm_backward needs the per-layer rstd to compute input gradients (d_res_att, d_ln1).
    // The comment about "not needing per-layer values for backward weight gradients" was incorrect -
    // we still need them for the input gradient computation which is always required for gradient flow.

    for (int i = 0; i < mConfig.num_layers; ++i) {
        auto& acts = mSimplifiedActivations[i];

        // RMSNorm rstd values - ALWAYS needed per-layer for rmsnorm_backward input gradient
        acts.ln1_rstd = mAllocator->allocate(
            ETensorDType::FP32, "ln1_rstd", kind, {B, T});
        // ln1 is always needed - it's the input X for LoRA QKV backward
        acts.ln1 = share_ln1 ? mSharedLn1 : mAllocator->allocate(
            dtype, "ln1", kind, {B, T, C});

        acts.ln2_rstd = mAllocator->allocate(
            ETensorDType::FP32, "ln2_rstd", kind, {B, T});
        // ln2 is always needed - it's the input X for LoRA MLP up/gate backward
        acts.ln2 = share_ln2 ? mSharedLn2 : mAllocator->allocate(
            dtype, "ln2", kind, {B, T, C});

        // QK-norm rstd values - needed for forward pass but can be shared across layers in LoRA mode
        // (since we don't need to save them per-layer for backward weight gradients)
        if (use_qk_norm) {
            if (lora_only) {
                // Share rstd buffers across layers in LoRA-only mode
                if (!mSharedQRstd.Data) {
                    mSharedQRstd = mAllocator->allocate(ETensorDType::FP32, "q_rstd_shared", kind, {B, T, (long)Hq});
                    mSharedKRstd = mAllocator->allocate(ETensorDType::FP32, "k_rstd_shared", kind, {B, T, (long)Hkv});
                }
                acts.q_rstd = mSharedQRstd;
                acts.k_rstd = mSharedKRstd;
            } else {
                acts.q_rstd = mAllocator->allocate(ETensorDType::FP32, "q_rstd", kind, {B, T, (long)Hq});
                acts.k_rstd = mAllocator->allocate(ETensorDType::FP32, "k_rstd", kind, {B, T, (long)Hkv});
            }
        } else {
            acts.q_rstd = {};
            acts.k_rstd = {};
        }

        // qkv is needed for attention backward (cuDNN)
        acts.qkv = share_qkv ? mSharedQKV : mAllocator->allocate(
            dtype, "qkv", kind, {B, T, qkv_channels});
        if (cache_pre_rope_qkv && !lora_only) {
            // In LoRA-only mode, we can skip the separate pre-RoPE QKV buffer
            // since we don't need to compute QK-norm weight gradients
            acts.qkv_rope = mAllocator->allocate(dtype, "qkv_rope", kind, {B, T, qkv_channels});
        } else {
            acts.qkv_rope = {};
        }

        // lse is needed for attention backward (cuDNN)
        acts.lse = mAllocator->allocate(
            ETensorDType::FP32, "lse", kind, {B, T, (long)Hq});

        // att is needed - it's the input X for LoRA O backward AND for attention backward
        acts.att = share_att ? mSharedAtt : mAllocator->allocate(
            dtype, "att", kind, {B, T, AttC});

        // att_out is NOT needed for LoRA backward - we can skip it
        if (lora_only) {
            // In LoRA-only mode, we still need a buffer for forward computation,
            // but we can share it across layers since we don't need to save it for backward
            if (!mSharedAttOut.Data) {
                mSharedAttOut = mAllocator->allocate(dtype, "att_out_shared", kind, {B, T, C});
            }
            acts.att_out = mSharedAttOut;
        } else {
            acts.att_out = share_att ? mSharedAttOut : mAllocator->allocate(
                dtype, "att_out", kind, {B, T, C});
        }

        // residual_att is needed for LN2 backward (residual gradient flow)
        acts.residual_att = share_residual_intermediates ? mSharedResidualAtt : mAllocator->allocate(
            dtype, "residual_att", kind, {B, T, C});

        // mlp_up is needed for SwiGLU backward
        if (ffn_temps_on_stack) {
            const std::array<long, 3> mlp_up_shape{B, T, 2 * D};
            const std::array<long, 3> swiglu_shape{B, T, D};
            acts.mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, mlp_up_shape);
            acts.swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, swiglu_shape);
        } else {
            acts.mlp_up = share_mlp_up ? mSharedMlpUp : mAllocator->allocate(
                dtype, "mlp_up", kind, {B, T, 2 * D});
            // swiglu is needed - it's the input X for LoRA down backward
            acts.swiglu = share_swiglu ? mSharedSwiGlu : mAllocator->allocate(
                dtype, "swiglu", kind, {B, T, D});
        }

        // Allocate swiglu_scale for scaled SwiGLU (nvfp4-simple recipe)
        if (mConfig.enable_scaled_swiglu) {
            acts.swiglu_scale = mAllocator->allocate(
                ETensorDType::FP32, "swiglu_scale", kind, {B * T});
        }

        // mlp_down is NOT needed for LoRA backward - we can share across layers
        if (lora_only) {
            if (!mSharedMlpDown.Data) {
                mSharedMlpDown = mAllocator->allocate(dtype, "mlp_down_shared", kind, {B, T, C});
            }
            acts.mlp_down = mSharedMlpDown;
        } else {
            acts.mlp_down = share_residual_intermediates ? mSharedMlpDown : mAllocator->allocate(
                dtype, "mlp_down", kind, {B, T, C});
        }
    }
}

template<typename Block>
void ModularRunState<Block>::allocate_simplified_gradients() {
    long B = mConfig.batch_size;
    long T = mConfig.seq_length;
    long C = mConfig.hidden_size;
    long D = mConfig.block_config.intermediate_size;
    int Hq = mConfig.block_config.num_query_heads;
    int Hkv = mConfig.block_config.num_kv_heads;
    int HS = mConfig.block_config.head_size;
    long AttC = static_cast<long>(HS) * static_cast<long>(Hq);
    long qkv_channels = HS * (Hq + 2 * Hkv);

    auto dtype = mConfig.grad_dtype;
    auto kind = EAllocationType::ON_DEVICE;

    const bool share_grads =
        mConfig.recompute_attention || mConfig.recompute_ffn || mConfig.recompute_block;
    const bool share_res_ffn = mConfig.recompute_block;
    const bool large_temps_on_stack = mConfig.recompute_block;

    // Allocate shared gradient intermediates if we're in a recompute mode.
    // These buffers are only used within a single layer's backward, so they can be reused.
    if (share_grads && !mSharedDResAtt.Data) {
        if (share_res_ffn) {
            mSharedDResFFN[0] = mAllocator->allocate(dtype, "d_res_ffn_a", kind, {B, T, C});
            mSharedDResFFN[1] = mAllocator->allocate(dtype, "d_res_ffn_b", kind, {B, T, C});
        }
        mSharedDResAtt = mAllocator->allocate(dtype, "d_res_att_shared", kind, {B, T, C});
        mSharedDLn2 = mAllocator->allocate(dtype, "d_ln2_shared", kind, {B, T, C});
        mSharedDAtt = mAllocator->allocate(dtype, "d_att_shared", kind, {B, T, AttC});
        mSharedDLn1 = mAllocator->allocate(dtype, "d_ln1_shared", kind, {B, T, C});
        if (!large_temps_on_stack) {
            mSharedDMlpUp = mAllocator->allocate(dtype, "d_mlp_up_shared", kind, {B, T, 2 * D});
            mSharedDSwiGlu = mAllocator->allocate(dtype, "d_swiglu_shared", kind, {B, T, D});
            mSharedDQKV = mAllocator->allocate(dtype, "d_qkv_shared", kind, {B, T, qkv_channels});
        }
    }

    mSimplifiedGradients.resize(mConfig.num_layers);
    for (int i = 0; i < mConfig.num_layers; ++i) {
        auto& g = mSimplifiedGradients[i];
        g.d_res_ffn = share_res_ffn ? mSharedDResFFN[i % 2] : mAllocator->allocate(dtype, "d_res_ffn", kind, {B, T, C});
        g.d_res_att = share_grads ? mSharedDResAtt : mAllocator->allocate(dtype, "d_res_att", kind, {B, T, C});
        g.d_ln2 = share_grads ? mSharedDLn2 : mAllocator->allocate(dtype, "d_ln2", kind, {B, T, C});
        g.d_att = share_grads ? mSharedDAtt : mAllocator->allocate(dtype, "d_att", kind, {B, T, AttC});
        g.d_ln1 = share_grads ? mSharedDLn1 : mAllocator->allocate(dtype, "d_ln1", kind, {B, T, C});
        if (large_temps_on_stack) {
            const std::array<long, 3> d_mlp_up_shape{B, T, 2 * D};
            const std::array<long, 3> d_swiglu_shape{B, T, D};
            const std::array<long, 3> d_qkv_shape{B, T, qkv_channels};
            g.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, d_mlp_up_shape);
            g.d_swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, d_swiglu_shape);
            g.d_qkv = Tensor::from_pointer(nullptr, DeviceId, dtype, d_qkv_shape);
        } else {
            // d_mlp_up is handled in-place on the mlp_up buffer (matches legacy LLamaModel).
            // This saves memory and ensures consistent in-place behavior across all modes.
            g.d_mlp_up = mSimplifiedActivations[i].mlp_up;
            g.d_swiglu = share_grads ? mSharedDSwiGlu : mAllocator->allocate(dtype, "d_swiglu", kind, {B, T, D});
            g.d_qkv = share_grads ? mSharedDQKV : mAllocator->allocate(dtype, "d_qkv", kind, {B, T, qkv_channels});
        }
    }
}

template<typename Block>
void ModularRunState<Block>::allocate_simplified_quant_buffers() {
    long B = mConfig.batch_size;
    long T = mConfig.seq_length;
    long C = mConfig.hidden_size;
    long D = mConfig.block_config.intermediate_size;
    int Hq = mConfig.block_config.num_query_heads;
    int Hkv = mConfig.block_config.num_kv_heads;
    int HS = mConfig.block_config.head_size;
    long AttC = static_cast<long>(HS) * static_cast<long>(Hq);
    long qkv_channels = HS * (Hq + 2 * Hkv);

    const bool need_act_quants = (mConfig.matmul_dtype != mConfig.activation_dtype);
    const bool need_grad_quants = (mConfig.grad_quant_dtype != mConfig.grad_dtype);

    // Always size the vector so call sites can index without branching.
    mSimplifiedQuantActivations.resize(mConfig.num_layers);

    // FP8 forward-only buffers: allocate shared buffers when enable_fp8_forward is set.
    // These are transient (consumed immediately by matmul), so one set is shared across all layers.
    // NOTE: Must be allocated BEFORE the early return for no-quant case.
    if (mConfig.enable_fp8_forward) {
        const auto fp8_dtype = mConfig.forward_matmul_dtype;  // Should be FP8_E4M3

        // Allocate stats buffer: 4 pairs (abs_max, scale) = 8 floats
        mFP8ForwardStats = mAllocator->allocate(ETensorDType::FP32, "fp8_fwd_stats",
                                                 EAllocationType::ON_DEVICE, {8L});
        float* fp8_stats = mFP8ForwardStats.get<float>();

        // LN1 -> QKV projection input
        mFP8ForwardQuants.ln1 = mAllocator->allocate(fp8_dtype, "fp8_fwd_ln1",
                                                      EAllocationType::ON_DEVICE, {B, T, C});
        mFP8ForwardQuants.ln1.Stats = fp8_stats + 0;

        // LN2 -> MLP up projection input
        mFP8ForwardQuants.ln2 = mAllocator->allocate(fp8_dtype, "fp8_fwd_ln2",
                                                      EAllocationType::ON_DEVICE, {B, T, C});
        mFP8ForwardQuants.ln2.Stats = fp8_stats + 2;

        // Att -> output projection input
        mFP8ForwardQuants.att = mAllocator->allocate(fp8_dtype, "fp8_fwd_att",
                                                      EAllocationType::ON_DEVICE, {B, T, AttC});
        mFP8ForwardQuants.att.Stats = fp8_stats + 4;

        // SwiGLU -> MLP down projection input
        mFP8ForwardQuants.swiglu = mAllocator->allocate(fp8_dtype, "fp8_fwd_swiglu",
                                                         EAllocationType::ON_DEVICE, {B, T, D});
        mFP8ForwardQuants.swiglu.Stats = fp8_stats + 6;
    }

    // Allocate FP8 delayed scaling state when HYBRID mode is enabled
    if (mConfig.enable_fp8_hybrid_delayed) {
        mFP8ScalingState = std::make_unique<FP8ScalingState>(
            mConfig.fp8_scaling_config, mAllocator, DeviceId, mConfig.num_layers);
    }

    // Allocate FP4 forward buffers when FP4 mode is enabled (Blackwell SM100+)
    // FP4 uses E2M1 format with two-level block scaling:
    // - Data: 2 FP4 values packed per byte, shape (M, K/2)
    // - Scales: FP8 E4M3 per 16 values, shape (ceil(M/128), ceil(ceil(K/16)/4)*4)
    if (mConfig.enable_fp4_forward || mConfig.enable_fp4_backward) {
        const auto fp4_dtype = ETensorDType::BYTE;  // Packed FP4 (2 values per byte)
        const auto scale_dtype = ETensorDType::FP8_E4M3;
        const long M = B * T;  // Batch dimension flattened

        // Compute scale tensor dimensions for FP4 block quantization
        // Scale tensor uses F8_128x4 swizzled layout:
        // - Each row of input has K/16 block scales (one per 16 elements)
        // - Rows are grouped into 128-row base blocks
        // - Columns are grouped into 4-column groups
        // The swizzled layout requires: rows aligned to 128, cols aligned to 4
        auto compute_scale_shape = [](long rows, long cols) -> std::pair<long, long> {
            const long scale_rows = ((rows + 127) / 128) * 128;  // Align to 128-row base blocks
            const long scale_cols_raw = (cols + 15) / 16;  // 16-element blocks
            const long scale_cols = ((scale_cols_raw + 3) / 4) * 4;  // 4-column alignment
            return {scale_rows, scale_cols};
        };

        // Global amax buffer: 4 floats for ln1, ln2, att, swiglu
        mFP4ForwardQuants.global_amax_buffer = mAllocator->allocate(
            ETensorDType::FP32, "fp4_global_amax", EAllocationType::ON_DEVICE, {4L});
        float* amax_ptr = mFP4ForwardQuants.global_amax_buffer.template get<float>();
        mFP4ForwardQuants.ln1_global_amax = amax_ptr + 0;
        mFP4ForwardQuants.ln2_global_amax = amax_ptr + 1;
        mFP4ForwardQuants.att_global_amax = amax_ptr + 2;
        mFP4ForwardQuants.swiglu_global_amax = amax_ptr + 3;

        // LN1 -> QKV projection input: (B*T, C)
        {
            auto [sr, sc] = compute_scale_shape(M, C);
            mFP4ForwardQuants.ln1_data = mAllocator->allocate(
                fp4_dtype, "fp4_fwd_ln1_data", EAllocationType::ON_DEVICE, {M, C / 2});
            mFP4ForwardQuants.ln1_scales = mAllocator->allocate(
                scale_dtype, "fp4_fwd_ln1_scales", EAllocationType::ON_DEVICE, {sr, sc});
        }

        // LN2 -> MLP up projection input: (B*T, C)
        {
            auto [sr, sc] = compute_scale_shape(M, C);
            mFP4ForwardQuants.ln2_data = mAllocator->allocate(
                fp4_dtype, "fp4_fwd_ln2_data", EAllocationType::ON_DEVICE, {M, C / 2});
            mFP4ForwardQuants.ln2_scales = mAllocator->allocate(
                scale_dtype, "fp4_fwd_ln2_scales", EAllocationType::ON_DEVICE, {sr, sc});
        }

        // Att -> output projection input: (B*T, AttC)
        {
            auto [sr, sc] = compute_scale_shape(M, AttC);
            mFP4ForwardQuants.att_data = mAllocator->allocate(
                fp4_dtype, "fp4_fwd_att_data", EAllocationType::ON_DEVICE, {M, AttC / 2});
            mFP4ForwardQuants.att_scales = mAllocator->allocate(
                scale_dtype, "fp4_fwd_att_scales", EAllocationType::ON_DEVICE, {sr, sc});
        }

        // SwiGLU -> MLP down projection input: (B*T, D)
        {
            auto [sr, sc] = compute_scale_shape(M, D);
            mFP4ForwardQuants.swiglu_data = mAllocator->allocate(
                fp4_dtype, "fp4_fwd_swiglu_data", EAllocationType::ON_DEVICE, {M, D / 2});
            mFP4ForwardQuants.swiglu_scales = mAllocator->allocate(
                scale_dtype, "fp4_fwd_swiglu_scales", EAllocationType::ON_DEVICE, {sr, sc});
        }

        // Hadamard transform workspace: largest dimension among C, AttC, D
        // Only allocated when RHT is enabled
        if (mConfig.enable_fp4_hadamard) {
            const long max_dim = std::max({C, AttC, D});
            mFP4ForwardQuants.hadamard_workspace = mAllocator->allocate(
                mConfig.activation_dtype, "fp4_hadamard_ws", EAllocationType::ON_DEVICE, {M, max_dim});
        }
    }

    // Leave tensors empty (Data == nullptr) when no quants are needed.
    if (!need_act_quants && !need_grad_quants) {
        const std::array<long, 3> ln_shape{B, T, C};
        const std::array<long, 3> att_shape{B, T, AttC};
        const std::array<long, 3> swiglu_shape{B, T, D};
        const std::array<long, 3> qkv_shape{B, T, qkv_channels};
        const std::array<long, 3> mlp_up_shape{B, T, 2 * D};
        for (int i = 0; i < mConfig.num_layers; ++i) {
            auto& q = mSimplifiedQuantActivations[i];
            q.ln1 = Tensor::from_pointer(nullptr, DeviceId, mConfig.matmul_dtype, ln_shape);
            q.ln2 = Tensor::from_pointer(nullptr, DeviceId, mConfig.matmul_dtype, ln_shape);
            q.att = Tensor::from_pointer(nullptr, DeviceId, mConfig.matmul_dtype, att_shape);
            q.swiglu = Tensor::from_pointer(nullptr, DeviceId, mConfig.matmul_dtype, swiglu_shape);
        }
        mSimplifiedQuantGrads.d_res_ffn = Tensor::from_pointer(nullptr, DeviceId, mConfig.grad_quant_dtype, ln_shape);
        mSimplifiedQuantGrads.d_res_att = Tensor::from_pointer(nullptr, DeviceId, mConfig.grad_quant_dtype, ln_shape);
        mSimplifiedQuantGrads.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, mConfig.grad_quant_dtype, mlp_up_shape);
        mSimplifiedQuantGrads.d_qkv = Tensor::from_pointer(nullptr, DeviceId, mConfig.grad_quant_dtype, qkv_shape);
        return;
    }

    // Stats layout:
    // - Forward activation quants: per-layer (LN1/LN2/Att/SwiGLU) = 4 pairs = 8 floats per layer.
    //   Note: data buffers may be shared across layers, but Stats must be per-layer (legacy parity).
    // - Backward gradient quants: shared scratch = 4 pairs = 8 floats.
    const long act_stats_floats = need_act_quants ? 8L * mConfig.num_layers : 0L;
    const long grad_stats_floats = need_grad_quants ? 8L : 0L;
    const long total_stats_floats = act_stats_floats + grad_stats_floats;

    mQuantStats = mAllocator->allocate(ETensorDType::FP32, "qmm_stats_modular", EAllocationType::ON_DEVICE, {total_stats_floats});
    float* stats = mQuantStats.get<float>();

    auto alloc = [&](ETensorDType dtype, const std::string& name, const std::vector<long>& shape) -> Tensor {
        return mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
    };

    const bool share_ln_data = mConfig.recompute_rmsnorm;
    const bool share_swiglu_data = mConfig.recompute_swiglu || mConfig.recompute_ffn || mConfig.recompute_block;

    if (need_act_quants) {
        if (share_ln_data) {
            mSharedLn1Quant = alloc(mConfig.matmul_dtype, "ln1_q_shared", {B, T, C});
            mSharedLn2Quant = alloc(mConfig.matmul_dtype, "ln2_q_shared", {B, T, C});
        }
        // Attention quant data is always shared (legacy parity). Stats are per-layer.
        mSharedAttQuant = alloc(mConfig.matmul_dtype, "att_q_shared", {B, T, AttC});
        if (share_swiglu_data) {
            mSharedSwiGluQuant = alloc(mConfig.matmul_dtype, "swiglu_q_shared", {B, T, D});
        }

        for (int i = 0; i < mConfig.num_layers; ++i) {
            auto& q = mSimplifiedQuantActivations[i];
            float* layer_stats = stats + 8L * i;

            // LN1
            if (share_ln_data) {
                q.ln1 = mSharedLn1Quant;
            } else {
                q.ln1 = alloc(mConfig.matmul_dtype, "ln1_q_l" + std::to_string(i), {B, T, C});
            }
            q.ln1.Stats = layer_stats + 0;

            // LN2
            if (share_ln_data) {
                q.ln2 = mSharedLn2Quant;
            } else {
                q.ln2 = alloc(mConfig.matmul_dtype, "ln2_q_l" + std::to_string(i), {B, T, C});
            }
            q.ln2.Stats = layer_stats + 2;

            // Attention
            q.att = mSharedAttQuant;
            q.att.Stats = layer_stats + 4;

            // SwiGLU
            if (share_swiglu_data) {
                q.swiglu = mSharedSwiGluQuant;
            } else {
                q.swiglu = alloc(mConfig.matmul_dtype, "swiglu_q_l" + std::to_string(i), {B, T, D});
            }
            q.swiglu.Stats = layer_stats + 6;
        }
    } else {
        const std::array<long, 3> ln_shape{B, T, C};
        const std::array<long, 3> att_shape{B, T, AttC};
        const std::array<long, 3> swiglu_shape{B, T, D};
        for (int i = 0; i < mConfig.num_layers; ++i) {
            auto& q = mSimplifiedQuantActivations[i];
            q.ln1 = Tensor::from_pointer(nullptr, DeviceId, mConfig.matmul_dtype, ln_shape);
            q.ln2 = Tensor::from_pointer(nullptr, DeviceId, mConfig.matmul_dtype, ln_shape);
            q.att = Tensor::from_pointer(nullptr, DeviceId, mConfig.matmul_dtype, att_shape);
            q.swiglu = Tensor::from_pointer(nullptr, DeviceId, mConfig.matmul_dtype, swiglu_shape);
        }
    }

    if (need_grad_quants) {
        float* gstats = stats + act_stats_floats;
        mSimplifiedQuantGrads.d_res_ffn = alloc(mConfig.grad_quant_dtype, "d_res_ffn_q", {B, T, C});
        mSimplifiedQuantGrads.d_res_ffn.Stats = gstats + 0;
        mSimplifiedQuantGrads.d_res_att = alloc(mConfig.grad_quant_dtype, "d_res_att_q", {B, T, C});
        mSimplifiedQuantGrads.d_res_att.Stats = gstats + 2;
        mSimplifiedQuantGrads.d_mlp_up = alloc(mConfig.grad_quant_dtype, "d_mlp_up_q", {B, T, 2 * D});
        mSimplifiedQuantGrads.d_mlp_up.Stats = gstats + 4;
        mSimplifiedQuantGrads.d_qkv = alloc(mConfig.grad_quant_dtype, "d_qkv_q", {B, T, qkv_channels});
        mSimplifiedQuantGrads.d_qkv.Stats = gstats + 6;
    } else {
        const std::array<long, 3> ln_shape{B, T, C};
        const std::array<long, 3> qkv_shape{B, T, qkv_channels};
        const std::array<long, 3> mlp_up_shape{B, T, 2 * D};
        mSimplifiedQuantGrads.d_res_ffn = Tensor::from_pointer(nullptr, DeviceId, mConfig.grad_quant_dtype, ln_shape);
        mSimplifiedQuantGrads.d_res_att = Tensor::from_pointer(nullptr, DeviceId, mConfig.grad_quant_dtype, ln_shape);
        mSimplifiedQuantGrads.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, mConfig.grad_quant_dtype, mlp_up_shape);
        mSimplifiedQuantGrads.d_qkv = Tensor::from_pointer(nullptr, DeviceId, mConfig.grad_quant_dtype, qkv_shape);
    }
}

template<typename Block>
void ModularRunState<Block>::allocate_non_block_state() {
    long B = mConfig.batch_size;
    long T = mConfig.seq_length;
    long C = mConfig.hidden_size;
    long V = mConfig.vocab_size;
    auto dtype = mConfig.activation_dtype;
    auto kind = EAllocationType::ON_DEVICE;

    // Activations
    mNonBlockActivations.encoded = mAllocator->allocate(
        dtype, "encoded", kind, {B, T, C});
    mNonBlockActivations.ln_final = mAllocator->allocate(
        dtype, "ln_final", kind, {B, T, C});
    mNonBlockActivations.ln_final_rstd = mAllocator->allocate(
        ETensorDType::FP32, "ln_final_rstd", kind, {B, T});

    // Output tensor is chunked - lazily allocated via temp_acquire
    // Size is (B*T/lmhead_chunks, V) not (B, T, V)
    long lmhead_chunks = mConfig.lmhead_chunks;
    long out_size = (B * T) / lmhead_chunks;

    // Get device id from IRunState
    int dev = DeviceId;

    // Create tensor descriptor without data - will be filled by temp_acquire
    mNonBlockActivations.output = Tensor{dtype, {out_size, V}, nullptr, nullptr, 2, dev};

    // Simulate stack usage for the output tensor (for stack size measurement)
    // This will be freed immediately, but records the max utilization
    std::byte* simulated_output = Stack.allocate(mNonBlockActivations.output.bytes(), "output_simulate");
    Stack.free(simulated_output);

    // RoPE frequencies - precomputed for max position embeddings (capped at T for now)
    // Shape: (max_seq_len, 2*head_size) - contains cos/sin interleaved
    // Skip allocation when using fused RoPE (computes cos/sin on-the-fly)
    if (!mConfig.use_fused_rope) {
        int max_seq_len = std::min((int)T, mConfig.pretrained_config.MaxPositionEmbeddings);  // Cap at seq length
        int head_size = mConfig.block_config.head_size;
        float rope_theta = mConfig.block_config.rope_theta;

        mNonBlockActivations.freq_cis = mAllocator->allocate(
            dtype, "freq_cis", kind, {(long)max_seq_len, (long)(2 * head_size)});

        // Generate frequencies on CPU then copy to device
        if (dtype == ETensorDType::BF16) {
            std::vector<nv_bfloat16> freq_cpu(max_seq_len * 2 * head_size);
            precompute_freqs_cis(freq_cpu.data(), head_size, max_seq_len, rope_theta);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data, freq_cpu.data(),
                                  freq_cpu.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        } else if (dtype == ETensorDType::FP32) {
            std::vector<float> freq_cpu(max_seq_len * 2 * head_size);
            precompute_freqs_cis(freq_cpu.data(), head_size, max_seq_len, rope_theta);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data, freq_cpu.data(),
                                  freq_cpu.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    } else {
        // Leave freq_cis empty - fused RoPE computes cos/sin on-the-fly
        mNonBlockActivations.freq_cis = {};
    }

    // Gradients
    mNonBlockGradients.d_ln_final = mAllocator->allocate(
        mConfig.grad_dtype, "d_ln_final", kind, {B, T, C});
    mNonBlockGradients.d_embeddings = mAllocator->allocate(
        mConfig.grad_dtype, "d_embeddings", kind, {B, T, C});
}

template<typename Block>
void ModularRunState<Block>::allocate_scratch_buffers(DeviceMemoryStack& stack) {
    long B = mConfig.batch_size;
    long T = mConfig.seq_length;
    long C = mConfig.hidden_size;

    // RMSNorm backward scratch (shared across layers).
    const long rmsnorm_scratch_bytes = static_cast<long>(get_rmsnorm_backward_scratch_size((int)C, DeviceProp));
    mScratch.rmsnorm_scratch = mAllocator->allocate(
        ETensorDType::BYTE, "rmsnorm_scratch",
        EAllocationType::ON_DEVICE, {rmsnorm_scratch_bytes});

    // Bias backward scratch (shared across layers), sized for the largest OC we use (QKV).
    {
        int Hq = mConfig.block_config.num_query_heads;
        int Hkv = mConfig.block_config.num_kv_heads;
        int HS = mConfig.block_config.head_size;
        int qkv_channels = HS * (Hq + 2 * Hkv);
        const long bias_scratch_bytes =
            static_cast<long>(get_bias_backward_scratch_size(mConfig.grad_dtype, qkv_channels, DeviceProp));
        mScratch.matmul_bias_scratch = mAllocator->allocate(
            ETensorDType::FP32, "bias_scratch",
            EAllocationType::ON_DEVICE, {bias_scratch_bytes / (long)sizeof(float)});
    }

    // Norm buffer for gradient norm/clipping. Must be sized for global_norm_squared partial sums.
    const long num_block_sums = std::max<long>(2, static_cast<long>(get_max_num_block_sums(DeviceProp)));
    mScratch.norm_buffer = mAllocator->allocate(
        ETensorDType::FP32, "norm_buffer",
        EAllocationType::ON_DEVICE, {num_block_sums});

    // Matmul scales for FP8
    mScratch.matmul_scales = mAllocator->allocate(
        ETensorDType::FP32, "matmul_scales",
        EAllocationType::ON_DEVICE, {2L});

    // Encoder backward scratch (match legacy LLamaRunState sizing)
    // encoder_backward uses these host buffers for deterministic bucketing/scheduling.
    const long group_width = (long)(16 / get_dtype_size(mConfig.grad_dtype) * 32);
    const long num_c_groups = (C + group_width - 1) / group_width;
    mScratch.encoder_bwd_scratch = mAllocator->allocate(
        ETensorDType::INT32, "encoder_bwd_scratch",
        EAllocationType::ON_DEVICE, {B, T, num_c_groups * 5});
    mScratch.encoder_bwd_indices = mAllocator->allocate(
        ETensorDType::INT32, "encoder_bwd_indices",
        EAllocationType::PINNED, {B, T, num_c_groups});
    mScratch.encoder_bwd_info = mAllocator->allocate(
        ETensorDType::INT32, "encoder_bwd_info",
        EAllocationType::PINNED, {B, T, 4 * num_c_groups});

    // cuDNN workspace - size depends on attention configuration.
    // Keep it as a stack-backed descriptor (like legacy) so it can overlap with other temporaries.
    int Hq = mConfig.block_config.num_query_heads;
    int Hkv = mConfig.block_config.num_kv_heads;
    int HS = mConfig.block_config.head_size;
    const int attn_chunks = mConfig.attention_bwd_chunks;
    if (attn_chunks < 1) {
        throw std::invalid_argument("attention_bwd_chunks must be >= 1");
    }
    if (attn_chunks > 1 && B % attn_chunks != 0) {
        throw std::invalid_argument(fmt::format(
            "attn_bwd_chunks ({}) must evenly divide per_device_train_batch_size ({}). "
            "Either increase batch size to a multiple of {} or reduce attn_bwd_chunks.",
            attn_chunks, B, attn_chunks));
    }
    const long attn_ws_batch_size =
        (attn_chunks == 1) ? B : div_exact(B, static_cast<long>(attn_chunks));
    long cudnn_ws_size = static_cast<long>(
        cudnn_get_workspace_size(static_cast<int>(attn_ws_batch_size), static_cast<int>(T), Hq, Hkv, HS, CudnnHandle));
    mScratch.cudnn_workspace = Tensor{ETensorDType::BYTE, {cudnn_ws_size}, nullptr, nullptr, 1, DeviceId};

    // Simulate stack usage so the stack high-water mark accounts for the attention workspace.
    // When recompute_block=true, d_qkv is also stack-backed and allocated simultaneously with
    // cudnn_workspace during attention_backward. We must simulate this combined usage.
    if (mConfig.recompute_block) {
        int qkv_channels = HS * (Hq + 2 * Hkv);
        long d_qkv_bytes = B * T * qkv_channels * get_dtype_size(mConfig.grad_dtype);
        std::byte* simulated_d_qkv = stack.allocate(d_qkv_bytes, "d_qkv_simulate");
        std::byte* simulated_ws = stack.allocate(mScratch.cudnn_workspace.bytes(), "workspace");
        stack.free(simulated_ws);
        stack.free(simulated_d_qkv);
    } else {
        // (When output logits dominate, this won't increase the max, but it remains correct for other configs.)
        std::byte* simulated_ws = stack.allocate(mScratch.cudnn_workspace.bytes(), "workspace");
        stack.free(simulated_ws);
    }

    // Simulate mlp_up, swiglu and d_swiglu stack usage when recompute_block is enabled.
    // These tensors are allocated via temp_acquire during forward and backward pass
    // (not during RunState construction) so we must simulate them here to ensure the stack is sized correctly.
    // The peak during backward is: mlp_up + swiglu + d_swiglu (all live simultaneously during MLP backward).
    if (mConfig.recompute_block) {
        long D = mConfig.block_config.intermediate_size;
        long mlp_up_bytes = B * T * 2 * D * get_dtype_size(mConfig.activation_dtype);
        long swiglu_bytes = B * T * D * get_dtype_size(mConfig.activation_dtype);
        long d_swiglu_bytes = B * T * D * get_dtype_size(mConfig.grad_dtype);
        std::byte* simulated_mlp_up = stack.allocate(mlp_up_bytes, "mlp_up_simulate");
        std::byte* simulated_swiglu = stack.allocate(swiglu_bytes, "swiglu_simulate");
        std::byte* simulated_d_swiglu = stack.allocate(d_swiglu_bytes, "d_swiglu_simulate");

        // FP8 backward_matmul allocates additional temporaries on the stack during MLP backward.
        // We need to simulate both mlp_down and mlp_up backward to find the true peak:
        //
        // Peak 1 (mlp_down backward): mlp_up + swiglu + d_swiglu + FP8_down_temps
        //   - weight_tp: (D, C) for transposed mlp_down_weight
        //   - act_tp: (D, B*T) for quantized swiglu input
        //   - grad_tp: (C, B*T) for transposed gradient
        //
        // Peak 2 (mlp_up backward, after freeing swiglu/d_swiglu): mlp_up + FP8_up_temps
        //   - weight_tp: (2*D, C) for transposed mlp_up_weight (LARGER!)
        //   - act_tp: (C, B*T) for quantized ln2 input
        //   - grad_tp: (2*D, B*T) for transposed gradient (LARGER!)
        //
        // We simulate peak 1 first, then check if peak 2 is larger.
        if (mConfig.enable_fp8_hybrid_delayed || mConfig.enable_fp8_forward) {
            long stats_bytes = 2 * sizeof(float);

            // Simulate peak 1: mlp_down backward (with swiglu/d_swiglu still on stack)
            // MLP down: weight is (C, D), input is (B*T, D), output is (B*T, C)
            //
            // When weights are BF16, backward_matmul allocates:
            // - weight_e4m3 (N, K) = (C, D): quantized weight
            // - weight_stats: scale for quantized weight
            // - weight_tp (K, N) = (D, C): transposed weight
            // - weight_tp_stats: scale for transposed weight
            long down_weight_e4m3_bytes = C * D * sizeof(__nv_fp8_e4m3);  // (N, K)
            long down_weight_tp_bytes = D * C * sizeof(__nv_fp8_e4m3);    // (K, N)
            long down_act_tp_bytes = D * B * T * sizeof(__nv_fp8_e4m3);
            long down_grad_tp_bytes = C * B * T * sizeof(__nv_fp8_e5m2);

            // Peak during dinp computation: mlp_up + swiglu + d_swiglu + weight_e4m3 + weight_stats + weight_tp + weight_tp_stats
            std::byte* sim_down_weight_e4m3 = stack.allocate(down_weight_e4m3_bytes, "fp8_down_weight_e4m3");
            std::byte* sim_down_weight_e4m3_stats = stack.allocate(stats_bytes, "fp8_down_weight_e4m3_stats");
            std::byte* sim_down_weight_tp = stack.allocate(down_weight_tp_bytes, "fp8_down_weight_tp");
            std::byte* sim_down_weight_tp_stats = stack.allocate(stats_bytes, "fp8_down_weight_tp_stats");

            // Free after dinp
            stack.free(sim_down_weight_tp_stats);
            stack.free(sim_down_weight_tp);
            stack.free(sim_down_weight_e4m3_stats);
            stack.free(sim_down_weight_e4m3);

            // Then dweight computation temps
            std::byte* sim_down_act_tp = stack.allocate(down_act_tp_bytes, "fp8_down_act_tp");
            std::byte* sim_down_act_stats = stack.allocate(stats_bytes, "fp8_down_act_stats");
            std::byte* sim_down_grad_tp = stack.allocate(down_grad_tp_bytes, "fp8_down_grad_tp");
            std::byte* sim_down_grad_stats = stack.allocate(stats_bytes, "fp8_down_grad_stats");
            stack.free(sim_down_grad_stats);
            stack.free(sim_down_grad_tp);
            stack.free(sim_down_act_stats);
            stack.free(sim_down_act_tp);

            // Free swiglu and d_swiglu (as happens between mlp_down and mlp_up backward)
            stack.free(simulated_d_swiglu);
            stack.free(simulated_swiglu);

            // Simulate peak 2: mlp_up backward (only mlp_up activation remains on stack)
            // MLP up: weight is (2*D, C), input is (B*T, C), output is (B*T, 2*D)
            //
            // When weights are BF16, backward_matmul allocates:
            // - weight_e4m3 (N, K) = (2*D, C): quantized weight
            // - weight_stats: scale for quantized weight
            // - weight_tp (K, N) = (C, 2*D): transposed weight
            // - weight_tp_stats: scale for transposed weight
            // All four are live simultaneously during dinp computation!
            //
            // After dinp: free weight_tp_stats, weight_tp, weight_stats, weight_e4m3
            // Then for dweight: activation_tp, act_stats, grad_tp, grad_stats
            long up_weight_e4m3_bytes = 2 * D * C * sizeof(__nv_fp8_e4m3);  // (N, K)
            long up_weight_tp_bytes = C * 2 * D * sizeof(__nv_fp8_e4m3);    // (K, N) - same size
            long up_act_tp_bytes = C * B * T * sizeof(__nv_fp8_e4m3);
            long up_grad_tp_bytes = 2 * D * B * T * sizeof(__nv_fp8_e5m2);

            // Peak during dinp computation: mlp_up + weight_e4m3 + weight_stats + weight_tp + weight_tp_stats
            std::byte* sim_up_weight_e4m3 = stack.allocate(up_weight_e4m3_bytes, "fp8_up_weight_e4m3");
            std::byte* sim_up_weight_e4m3_stats = stack.allocate(stats_bytes, "fp8_up_weight_e4m3_stats");
            std::byte* sim_up_weight_tp = stack.allocate(up_weight_tp_bytes, "fp8_up_weight_tp");
            std::byte* sim_up_weight_tp_stats = stack.allocate(stats_bytes, "fp8_up_weight_tp_stats");

            // Free after dinp
            stack.free(sim_up_weight_tp_stats);
            stack.free(sim_up_weight_tp);
            stack.free(sim_up_weight_e4m3_stats);
            stack.free(sim_up_weight_e4m3);

            // Then dweight computation temps
            std::byte* sim_up_act_tp = stack.allocate(up_act_tp_bytes, "fp8_up_act_tp");
            std::byte* sim_up_act_stats = stack.allocate(stats_bytes, "fp8_up_act_stats");
            std::byte* sim_up_grad_tp = stack.allocate(up_grad_tp_bytes, "fp8_up_grad_tp");
            std::byte* sim_up_grad_stats = stack.allocate(stats_bytes, "fp8_up_grad_stats");
            stack.free(sim_up_grad_stats);
            stack.free(sim_up_grad_tp);
            stack.free(sim_up_act_stats);
            stack.free(sim_up_act_tp);

            // Free mlp_up (as happens after mlp_up backward)
            stack.free(simulated_mlp_up);
        } else {
            // Non-FP8: just free the simulated tensors
            stack.free(simulated_d_swiglu);
            stack.free(simulated_swiglu);
            stack.free(simulated_mlp_up);
        }
    }
}

template<typename Block>
void ModularRunState<Block>::allocate_residual_buffers() {
    long B = mConfig.batch_size;
    long T = mConfig.seq_length;
    long C = mConfig.hidden_size;
    auto dtype = mConfig.activation_dtype;

    // Final residual is always needed and must not alias any per-layer residual buffer.
    mFinalResidual = mAllocator->allocate(
        dtype, "final_residual", EAllocationType::ON_DEVICE, {B, T, C});

    if (mConfig.offload_residuals) {
        // With offloading: double-buffered device + per-layer host
        int num_buffers = mConfig.num_residual_buffers;

        // Device buffers (double-buffered)
        mDeviceResiduals.resize(num_buffers);
        for (int i = 0; i < num_buffers; ++i) {
            mDeviceResiduals[i] = mAllocator->allocate(
                dtype, ("device_residual_" + std::to_string(i)).c_str(),
                EAllocationType::ON_DEVICE, {B, T, C});
        }

        // Host buffers (one per layer)
        mOffloadedResiduals.resize(mConfig.num_layers);
        for (int i = 0; i < mConfig.num_layers; ++i) {
            mOffloadedResiduals[i] = mAllocator->allocate(
                dtype, ("offloaded_residual_" + std::to_string(i)).c_str(),
                EAllocationType::PINNED, {B, T, C});
        }

        // State tracking for double-buffering
        mResidualState.resize(num_buffers);
        for (int i = 0; i < num_buffers; ++i) {
            CUDA_CHECK(cudaEventCreate(&mResidualState[i].event));
            CUDA_CHECK(cudaEventCreate(&mResidualState[i].ready_event));
            mResidualState[i].layer_idx = -1;
            mResidualState[i].is_ready = false;
            // Mark the buffer event as "done" initially so first wait doesn't stall.
            CUDA_CHECK(cudaEventRecord(mResidualState[i].event, MainStream));
            CUDA_CHECK(cudaEventRecord(mResidualState[i].ready_event, MainStream));
        }
    } else {
        // Without offloading: one device buffer per layer
        mDeviceResiduals.resize(mConfig.num_layers);
        for (int i = 0; i < mConfig.num_layers; ++i) {
            mDeviceResiduals[i] = mAllocator->allocate(
                dtype, ("res_ffn_" + std::to_string(i)).c_str(),
                EAllocationType::ON_DEVICE, {B, T, C});
        }
    }
}

template<typename Block>
void ModularRunState<Block>::create_cuda_resources() {
    CUDA_CHECK(cudaStreamCreate(&mSideStream));
    CUDA_CHECK(cudaEventCreate(&mSideStreamEvent));
    CUDA_CHECK(cudaEventCreate(&mOptEmbeddingsDone));

    mLayerUpdateDone.resize(mConfig.num_layers);
    for (int i = 0; i < mConfig.num_layers; ++i) {
        CUDA_CHECK(cudaEventCreate(&mLayerUpdateDone[i]));
    }

    // One graph executable per layer for forward blocks, two for backward (accumulate=false/true).
    // This avoids repeatedly updating a single graph exec with per-layer pointers (which is slow and often unsupported),
    // and enables stable replay across steps.
    mForwardBlockGraphs.assign((std::size_t)mConfig.num_layers, nullptr);
    mBackwardBlockGraphs.resize((std::size_t)mConfig.num_layers);
    for (auto& arr : mBackwardBlockGraphs) {
        arr = {nullptr, nullptr};
    }

    // Stack checkpoints for graphs (used to restore stack state before graph replay).
    mForwardBlockStackCheckpoints.resize((std::size_t)mConfig.num_layers);
    mBackwardBlockStackCheckpoints.resize((std::size_t)mConfig.num_layers);
}

template<typename Block>
void ModularRunState<Block>::release_cuda_resources() noexcept {
    auto destroy_graph_exec = [](cudaGraphExec_t& graph) noexcept {
        if (graph) {
            (void)cudaGraphExecDestroy(graph);
            graph = nullptr;
        }
    };

    for (auto& graph : mForwardBlockGraphs) destroy_graph_exec(graph);
    for (auto& arr : mBackwardBlockGraphs) {
        for (auto& graph : arr) destroy_graph_exec(graph);
    }
    mForwardBlockGraphs.clear();
    mBackwardBlockGraphs.clear();
    destroy_graph_exec(mGlobalNormGraph);

    if (mSideStream) {
        cudaStreamDestroy(mSideStream);
        mSideStream = nullptr;
    }
    if (mSideStreamEvent) {
        cudaEventDestroy(mSideStreamEvent);
        mSideStreamEvent = nullptr;
    }
    if (mOptEmbeddingsDone) {
        cudaEventDestroy(mOptEmbeddingsDone);
        mOptEmbeddingsDone = nullptr;
    }

    for (auto& event : mLayerUpdateDone) {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    mLayerUpdateDone.clear();

    for (auto& state : mResidualState) {
        if (state.event) {
            cudaEventDestroy(state.event);
        }
        if (state.ready_event) {
            cudaEventDestroy(state.ready_event);
        }
    }
    mResidualState.clear();
}

template<typename Block>
int ModularRunState<Block>::find_free_residual_buffer() const {
    for (size_t i = 0; i < mResidualState.size(); ++i) {
        if (mResidualState[i].layer_idx < 0) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_RUN_STATE_H
