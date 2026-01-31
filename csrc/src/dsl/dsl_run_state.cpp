// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL run state implementation.

#include "dsl/dsl_run_state.h"

#include <algorithm>
#include <stdexcept>

#include "kernels/kernels.h"
#include "training/runtime_options.h"
#include "modules/fp8_run_state.h"
#include "modules/fp8_scaling_config.h"
#include "modules/fp8_scaling_state.h"
#include "modules/matmul_context.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {

DslRunState::DslRunState(const PretrainedConfig& config,
                         const RuntimeOptions& options,
                         int B, int T,
                         const std::shared_ptr<TensorAllocator>& allocator,
                         bool lora_only_mode,
                         std::size_t stack_bytes,
                         bool allocate_stack)
    : IRunState(config.clone(), B, T, allocator),
      mAllocator(allocator),
      mRecomputeLevel(options.Recompute),
      mLoraOnlyMode(lora_only_mode),
      mNumLayers(config.NumLayers),
      mPerLayerGraphsEnabled(options.UseCudaGraphs) {
    if (!mAllocator) {
        throw std::runtime_error("DslRunState: allocator is null");
    }

    mActivationDtype = options.ModelType.value_or(config.DType);
    if (is_fp8_dtype(mActivationDtype)) {
        mActivationDtype = ETensorDType::BF16;
    }
    mGradDtype = mActivationDtype;
    mMatmulDtype = options.MatmulType.value_or(options.ModelType.value_or(config.DType));
    if (options.TrainingRecipe && options.TrainingRecipe->is_fp8_hybrid()) {
        mGradQuantDtype = ETensorDType::FP8_E5M2;
    } else {
        mGradQuantDtype = options.GradientType.value_or(mMatmulDtype);
    }
    mEnableFp8Forward = options.fp8_forward_enabled();
    if (options.LMHeadChunks < 1) {
        throw std::runtime_error("lmhead_chunks must be >= 1");
    }
    if (options.AttBwdChunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    mLMHeadChunks = options.LMHeadChunks;
    mAttnBwdChunks = options.AttBwdChunks;
    mStackSimulate = !allocate_stack;

    const std::size_t stack_capacity = (stack_bytes > 0) ? stack_bytes : kDefaultStackBytes;
    if (allocate_stack) {
        // Allocate stack memory (heuristic size).
        mStackBuffer = mAllocator->allocate(
            ETensorDType::BYTE, "dsl_stack", EAllocationType::ON_DEVICE,
            {static_cast<long>(stack_capacity)});
        Stack = DeviceMemoryStack(mStackBuffer.Data, stack_capacity, DeviceId);
    } else {
        // Dummy stack for sizing pass (no device allocation).
        Stack = DeviceMemoryStack(nullptr, stack_capacity, DeviceId);
    }

    create_cuda_resources();
    allocate_non_block_state(config);
    allocate_simplified_activations(config);
    allocate_simplified_gradients(config);
    allocate_simplified_quant_buffers(config, options);
    allocate_residual_buffers(config, options.OffloadResidual);
    allocate_scratch_buffers(config);

    // Allocate per-layer CUDA graph arrays
    allocate_graph_arrays(config.NumLayers);
}

DslRunState::~DslRunState() {
    destroy_cuda_graphs();
    release_cuda_resources();
}

void DslRunState::set_stack_buffer(Tensor buffer, const DeviceMemoryStack::AllocationList& high_mark) {
    if (!buffer.Data || buffer.bytes() == 0) {
        throw std::runtime_error("DslRunState::set_stack_buffer: invalid stack buffer");
    }
    mStackBuffer = std::move(buffer);
    Stack = DeviceMemoryStack(mStackBuffer.Data, static_cast<std::size_t>(mStackBuffer.bytes()), DeviceId);
    if (!high_mark.empty()) {
        Stack.set_high_mark(high_mark);
    }
    mStackSimulate = false;
}

Tensor& DslRunState::get_residual(int layer_idx, cudaStream_t stream) {
    if (!mResidualManager) {
        throw std::runtime_error("DslRunState: residual manager not initialized");
    }
    return mResidualManager->get_residual(layer_idx, stream);
}

Tensor& DslRunState::get_final_residual() {
    if (!mResidualManager) {
        throw std::runtime_error("DslRunState: residual manager not initialized");
    }
    return mResidualManager->get_final_residual();
}

void DslRunState::allocate_non_block_state(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long V = cfg.VocabSize;
    const auto dtype = mActivationDtype;

    mNonBlockActivations.encoded = mAllocator->allocate(dtype, "encoded", EAllocationType::ON_DEVICE, {B, T, C});
    mNonBlockActivations.ln_final = mAllocator->allocate(dtype, "ln_final", EAllocationType::ON_DEVICE, {B, T, C});
    mNonBlockActivations.ln_final_rstd = mAllocator->allocate(ETensorDType::FP32, "ln_final_rstd", EAllocationType::ON_DEVICE, {B, T});

    // Output buffer (persistent; avoids large stack pressure for full fine-tuning).
    const long lmhead_chunks = static_cast<long>(mLMHeadChunks);
    const long out_size = (B * T) / lmhead_chunks;
    mNonBlockActivations.output = mAllocator->allocate(dtype, "output", EAllocationType::ON_DEVICE, {out_size, V});

    // RoPE frequencies (if not using fused RoPE).
    const int max_seq_len = std::min(static_cast<int>(T), cfg.MaxPositionEmbeddings);
    if (max_seq_len > 0) {
        const int head_size = cfg.head_size();
        if (dtype == ETensorDType::BF16) {
            mNonBlockActivations.freq_cis = mAllocator->allocate(
                dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            std::vector<nv_bfloat16> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            precompute_freqs_cis(freq_cpu.data(), head_size, max_seq_len, cfg.RopeTheta);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data, freq_cpu.data(),
                                  freq_cpu.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        } else if (dtype == ETensorDType::FP32) {
            mNonBlockActivations.freq_cis = mAllocator->allocate(
                dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            std::vector<float> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            precompute_freqs_cis(freq_cpu.data(), head_size, max_seq_len, cfg.RopeTheta);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data, freq_cpu.data(),
                                  freq_cpu.size() * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            // Default: allocate in model dtype and leave zeroed.
            mNonBlockActivations.freq_cis = mAllocator->allocate(
                dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            fill_zero(mNonBlockActivations.freq_cis, MainStream);
        }
    }

    mNonBlockGradients.d_ln_final = mAllocator->allocate(mGradDtype, "d_ln_final", EAllocationType::ON_DEVICE, {B, T, C});
    // Skip d_embeddings allocation in LoRA-only mode - embedding backward is skipped entirely
    if (!mLoraOnlyMode) {
        mNonBlockGradients.d_embeddings = mAllocator->allocate(mGradDtype, "d_embeddings", EAllocationType::ON_DEVICE, {B, T, C});
    }
}

void DslRunState::allocate_simplified_activations(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    const long AttnDim = Hq * D;
    const long QKV = D * (Hq + 2 * Hkv);
    const long M = cfg.IntermediateSize;
    const long MUp = 2 * M;
    const bool use_qk_norm = cfg.UseQKNorm;

    const auto dtype = mActivationDtype;
    const auto kind = EAllocationType::ON_DEVICE;

    // Activation sharing logic - based on recompute setting
    // When recompute is enabled, intermediates can be shared because backward
    // will recompute them from checkpoints.
    //
    // recompute: false - Save everything, no sharing (maximum memory)
    // recompute: true  - Share intermediates, recompute from checkpoints (saves ~17% VRAM)
    //
    const bool recompute_enabled = mRecomputeLevel >= RecomputeLevel::Enabled;
    const bool lora_only = mLoraOnlyMode;

    // LN outputs can be shared when recompute is enabled
    const bool share_ln1 = recompute_enabled;
    const bool share_ln2 = recompute_enabled;

    // QKV sharing: only in LoRA mode where attention is recomputed.
    // In FFT mode, we skip attention recompute and need saved QKV per-layer for bit-exact gradients.
    // If QKV is shared, all layers would use the last layer's QKV data during backward!
    const bool share_qkv = recompute_enabled && lora_only;

    // Attention output sharing:
    // IMPORTANT: att/att_out/lse must NEVER be shared when recompute is enabled.
    // - FFT mode: Need original forward values for bit-exact gradients
    // - LoRA mode: LoRA O proj backward hook needs original att values per-layer
    //              cuDNN attention is non-deterministic, so recomputed att != forward att
    // This matches modular model behavior (see run_state_impl.tpp:568-572).
    const bool share_att = false;

    // MLP intermediates sharing:
    // - FFT mode: Need per-layer for bit-exact gradients
    // - LoRA mode: LoRA down_proj backward hook needs original swiglu values per-layer
    // This matches modular model behavior (lora_can_share_swiglu = !lora_only).
    const bool share_mlp_up = false;
    const bool share_swiglu = false;

    // Residual intermediates: In LoRA mode they can be shared. In FFT mode, save per-layer.
    const bool share_residual_intermediates = recompute_enabled && lora_only;

    // FFN temps: Never use stack for mlp_up/swiglu when we need per-layer values.
    // - FFT mode: per-layer needed for bit-exact gradients
    // - LoRA mode: per-layer needed for down_proj LoRA backward (swiglu input)
    // This matches modular model behavior.
    const bool ffn_temps_on_stack = false;
    mFfnTempsOnStack = ffn_temps_on_stack;
    if (mStackSimulate && ffn_temps_on_stack) {
        const long mlp_up_bytes = B * T * MUp * get_dtype_size(dtype);
        const long swiglu_bytes = B * T * M * get_dtype_size(dtype);
        auto* sim_mlp_up = Stack.allocate(static_cast<std::size_t>(mlp_up_bytes), "mlp_up_simulate");
        auto* sim_swiglu = Stack.allocate(static_cast<std::size_t>(swiglu_bytes), "swiglu_simulate");
        Stack.free(sim_swiglu);
        Stack.free(sim_mlp_up);
    }

    Tensor shared_ln1{}, shared_ln2{}, shared_qkv{}, shared_att{}, shared_att_out{}, shared_lse{};
    Tensor shared_mlp_up{}, shared_swiglu{}, shared_residual_att{}, shared_mlp_down{};

    if (share_ln1) shared_ln1 = mAllocator->allocate(dtype, "ln1_shared", kind, {B, T, C});
    if (share_ln2) shared_ln2 = mAllocator->allocate(dtype, "ln2_shared", kind, {B, T, C});
    if (share_qkv) shared_qkv = mAllocator->allocate(dtype, "qkv_shared", kind, {B, T, QKV});
    // LSE sharing: only in lora_only mode. FFT needs per-layer LSE for bit-exact gradients.
    if (share_att) shared_lse = mAllocator->allocate(ETensorDType::FP32, "lse_shared", kind, {B, Hq, T});
    if (share_att) {
        shared_att = mAllocator->allocate(dtype, "att_shared", kind, {B, T, AttnDim});
        shared_att_out = mAllocator->allocate(dtype, "att_out_shared", kind, {B, T, C});
    }
    // att_out sharing is handled by share_att when recompute is enabled.
    if (share_mlp_up && !ffn_temps_on_stack) shared_mlp_up = mAllocator->allocate(dtype, "mlp_up_shared", kind, {B, T, MUp});
    if (share_swiglu && !ffn_temps_on_stack) shared_swiglu = mAllocator->allocate(dtype, "swiglu_shared", kind, {B, T, M});
    if (share_residual_intermediates) {
        shared_residual_att = mAllocator->allocate(dtype, "residual_att_shared", kind, {B, T, C});
    }
    // In lora_only mode, mlp_down can be shared (not needed for LoRA backward)
    if (lora_only) {
        shared_mlp_down = mAllocator->allocate(dtype, "mlp_down_shared", kind, {B, T, C});
    } else if (share_residual_intermediates) {
        shared_mlp_down = mAllocator->allocate(dtype, "mlp_down_shared", kind, {B, T, C});
    }

    mSimplifiedActivations.resize(cfg.NumLayers);
    for (int i = 0; i < cfg.NumLayers; ++i) {
        auto& acts = mSimplifiedActivations[i];
        acts.ln1_rstd = mAllocator->allocate(ETensorDType::FP32, "ln1_rstd", kind, {B, T});
        acts.ln1 = share_ln1 ? shared_ln1 : mAllocator->allocate(dtype, "ln1", kind, {B, T, C});

        acts.ln2_rstd = mAllocator->allocate(ETensorDType::FP32, "ln2_rstd", kind, {B, T});
        acts.ln2 = share_ln2 ? shared_ln2 : mAllocator->allocate(dtype, "ln2", kind, {B, T, C});

        if (use_qk_norm) {
            acts.q_rstd = mAllocator->allocate(ETensorDType::FP32, "q_rstd", kind, {B, T, Hq});
            acts.k_rstd = mAllocator->allocate(ETensorDType::FP32, "k_rstd", kind, {B, T, Hkv});
        } else {
            acts.q_rstd = {};
            acts.k_rstd = {};
        }

        acts.qkv = share_qkv ? shared_qkv : mAllocator->allocate(dtype, "qkv", kind, {B, T, QKV});
        // In FFT mode with recompute enabled, we need BOTH qkv (original) and qkv_rope (transformed)
        // because the qk_norm_rope backward kernel expects qkv_rope (the transformed output).
        // The kernel applies inverse RoPE internally to compute the gradient.
        // Without separate buffers, the in-place forward transform overwrites the qkv buffer,
        // so we can't distinguish between original qkv and transformed qkv_rope.
        // In LoRA mode, qkv_rope has lora_only policy so it gets recomputed during backward.
        // In no-recompute mode, the qkv buffer contains qkv_rope after forward (in-place transform).
        const bool need_separate_qkv_rope = recompute_enabled && !lora_only && use_qk_norm;
        acts.qkv_rope = need_separate_qkv_rope
            ? mAllocator->allocate(dtype, "qkv_rope", kind, {B, T, QKV})
            : Tensor{};

        acts.lse = share_att ? shared_lse
                             : mAllocator->allocate(ETensorDType::FP32, "lse", kind, {B, Hq, T});
        acts.att = share_att ? shared_att : mAllocator->allocate(dtype, "att", kind, {B, T, AttnDim});
        // When recompute is enabled, att_out can be shared across layers.
        acts.att_out = share_att ? shared_att_out : mAllocator->allocate(dtype, "att_out", kind, {B, T, C});

        // residual_att can be shared when recompute_block=true
        acts.residual_att = share_residual_intermediates ? shared_residual_att
                                                          : mAllocator->allocate(dtype, "residual_att", kind, {B, T, C});

        if (ffn_temps_on_stack) {
            acts.mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, MUp});
            acts.swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, M});
        } else {
            acts.mlp_up = share_mlp_up ? shared_mlp_up : mAllocator->allocate(dtype, "mlp_up", kind, {B, T, MUp});
            acts.swiglu = share_swiglu ? shared_swiglu : mAllocator->allocate(dtype, "swiglu", kind, {B, T, M});
        }

        // mlp_down can be shared in lora_only mode (not needed for LoRA backward)
        if (lora_only || share_residual_intermediates) {
            acts.mlp_down = shared_mlp_down;
        } else {
            acts.mlp_down = mAllocator->allocate(dtype, "mlp_down", kind, {B, T, C});
        }
    }

    // Allocate temporary buffers for recomputation
    // This prevents overwriting saved values when recomputing forward activations
    if (recompute_enabled) {
        mRecomputeRstd = mAllocator->allocate(ETensorDType::FP32, "recompute_rstd", kind, {B, T});
        // LSE buffer for attention recomputation - same shape as acts.lse [B, Hq, T]
        mRecomputeLSE = mAllocator->allocate(ETensorDType::FP32, "recompute_lse", kind, {B, Hq, T});
    }
}

void DslRunState::allocate_simplified_gradients(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    const long AttnDim = Hq * D;
    const long QKV = D * (Hq + 2 * Hkv);
    const long M = cfg.IntermediateSize;
    const long MUp = 2 * M;

    const auto dtype = mGradDtype;
    const auto kind = EAllocationType::ON_DEVICE;

    // Gradient sharing flags - based on recompute level
    // IMPORTANT: In FFT mode (not lora_only), gradient buffer sharing can cause issues because
    // the DSL backward graph structure differs from LoRA mode. The gradients may be needed
    // at different points in the backward pass, and sharing can cause corruption.
    // For safety, disable all gradient sharing in FFT mode.
    const bool recompute_enabled = mRecomputeLevel >= RecomputeLevel::Enabled;
    const bool share_grads = recompute_enabled && mLoraOnlyMode;
    const bool share_res_ffn = recompute_enabled && mLoraOnlyMode;
    const bool share_mlp_down = recompute_enabled && mLoraOnlyMode;
    const bool large_bwd_temps_on_stack = recompute_enabled;

    if (mStackSimulate && large_bwd_temps_on_stack) {
        const long d_qkv_bytes = B * T * QKV * get_dtype_size(dtype);
        const long d_mlp_up_bytes = B * T * MUp * get_dtype_size(dtype);
        const long d_swiglu_bytes = B * T * M * get_dtype_size(dtype);
        const long d_up_bytes = B * T * MUp * get_dtype_size(dtype);
        auto* sim_d_qkv = Stack.allocate(static_cast<std::size_t>(d_qkv_bytes), "d_qkv_simulate");
        auto* sim_d_mlp_up = Stack.allocate(static_cast<std::size_t>(d_mlp_up_bytes), "d_mlp_up_simulate");
        auto* sim_d_swiglu = Stack.allocate(static_cast<std::size_t>(d_swiglu_bytes), "d_swiglu_simulate");
        auto* sim_d_up = Stack.allocate(static_cast<std::size_t>(d_up_bytes), "d_up_simulate");
        Stack.free(sim_d_up);
        Stack.free(sim_d_swiglu);
        Stack.free(sim_d_mlp_up);
        Stack.free(sim_d_qkv);
    }

    // Allocate shared gradient buffers if recompute_block is enabled
    if (share_grads && !mSharedDResAtt.Data) {
        if (share_res_ffn) {
            mSharedDResFFN[0] = mAllocator->allocate(dtype, "d_res_ffn_a", kind, {B, T, C});
            mSharedDResFFN[1] = mAllocator->allocate(dtype, "d_res_ffn_b", kind, {B, T, C});
        }
        if (share_mlp_down) {
            mSharedDMlpDown[0] = mAllocator->allocate(dtype, "d_mlp_down_a", kind, {B, T, C});
            mSharedDMlpDown[1] = mAllocator->allocate(dtype, "d_mlp_down_b", kind, {B, T, C});
        }
        mSharedDResAtt = mAllocator->allocate(dtype, "d_res_att_shared", kind, {B, T, C});
        mSharedDLn2 = mAllocator->allocate(dtype, "d_ln2_shared", kind, {B, T, C});
        mSharedDAtt = mAllocator->allocate(dtype, "d_att_shared", kind, {B, T, AttnDim});
        mSharedDLn1 = mAllocator->allocate(dtype, "d_ln1_shared", kind, {B, T, C});
    }

    mSimplifiedGradients.resize(cfg.NumLayers);
    for (int i = 0; i < cfg.NumLayers; ++i) {
        auto& g = mSimplifiedGradients[i];

        // d_res_ffn uses alternating buffers (i % 2) to avoid aliasing in adjacent layers
        g.d_res_ffn = share_res_ffn ? mSharedDResFFN[static_cast<std::size_t>(i % 2)]
                                    : mAllocator->allocate(dtype, "d_res_ffn", kind, {B, T, C});
        g.d_res_att = share_grads ? mSharedDResAtt : mAllocator->allocate(dtype, "d_res_att", kind, {B, T, C});
        g.d_ln2 = share_grads ? mSharedDLn2 : mAllocator->allocate(dtype, "d_ln2", kind, {B, T, C});

        if (large_bwd_temps_on_stack) {
            g.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, MUp});
            g.d_swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, M});
            g.d_qkv = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, QKV});
        } else {
            g.d_mlp_up = mAllocator->allocate(dtype, "d_mlp_up", kind, {B, T, MUp});
            g.d_swiglu = mAllocator->allocate(dtype, "d_swiglu", kind, {B, T, M});
            g.d_qkv = mAllocator->allocate(dtype, "d_qkv", kind, {B, T, QKV});
        }

        // d_mlp_down uses alternating buffers (i % 2) to avoid aliasing in adjacent layers
        g.d_mlp_down = share_mlp_down ? mSharedDMlpDown[static_cast<std::size_t>(i % 2)]
                                      : mAllocator->allocate(dtype, "d_mlp_down", kind, {B, T, C});
        g.d_att = share_grads ? mSharedDAtt : mAllocator->allocate(dtype, "d_att", kind, {B, T, AttnDim});
        g.d_ln1 = share_grads ? mSharedDLn1 : mAllocator->allocate(dtype, "d_ln1", kind, {B, T, C});
    }

    // Preserve the original buffer pointers so we can restore them if the
    // compiled executor temporarily aliases gradients to stack-backed temps.
    mSimplifiedGradientsBase = mSimplifiedGradients;
}

void DslRunState::reset_simplified_gradients() {
    if (mSimplifiedGradientsBase.size() != mSimplifiedGradients.size()) {
        return;
    }
    for (std::size_t i = 0; i < mSimplifiedGradients.size(); ++i) {
        auto& dst = mSimplifiedGradients[i];
        const auto& src = mSimplifiedGradientsBase[i];

        dst.d_res_ffn.Data = src.d_res_ffn.Data;
        dst.d_res_att.Data = src.d_res_att.Data;
        dst.d_ln2.Data = src.d_ln2.Data;
        dst.d_mlp_up.Data = src.d_mlp_up.Data;
        dst.d_swiglu.Data = src.d_swiglu.Data;
        dst.d_mlp_down.Data = src.d_mlp_down.Data;
        dst.d_att.Data = src.d_att.Data;
        dst.d_qkv.Data = src.d_qkv.Data;
        dst.d_ln1.Data = src.d_ln1.Data;

        dst.d_mamba_normed.Data = src.d_mamba_normed.Data;
        dst.d_mamba_gated.Data = src.d_mamba_gated.Data;
        dst.d_mamba_scan_out.Data = src.d_mamba_scan_out.Data;
        dst.d_mamba_u.Data = src.d_mamba_u.Data;
        dst.d_mamba_delta.Data = src.d_mamba_delta.Data;
        dst.d_mamba_B.Data = src.d_mamba_B.Data;
        dst.d_mamba_C.Data = src.d_mamba_C.Data;
        dst.d_mamba_conv_out.Data = src.d_mamba_conv_out.Data;
    }
}

void DslRunState::zero_activation_gradients(cudaStream_t stream) {
    // Zero ALL activation gradient buffers to prevent stale gradients from accumulating.
    // Many backward kernels accumulate (+=) to their output buffers. If any buffer contains
    // stale values from a previous micro-batch, gradients will explode.
    // Zeroing ensures a clean slate for each backward pass.
    for (std::size_t i = 0; i < mSimplifiedGradients.size(); ++i) {
        auto& g = mSimplifiedGradients[i];
        // d_res_ffn: The last layer's is zeroed separately (receives gradient from loss).
        if (i < mSimplifiedGradients.size() - 1 && g.d_res_ffn.Data) {
            fill_zero(g.d_res_ffn, stream);
        }
        // Zero all other activation gradient buffers
        if (g.d_res_att.Data) fill_zero(g.d_res_att, stream);
        if (g.d_ln2.Data) fill_zero(g.d_ln2, stream);
        if (g.d_mlp_up.Data) fill_zero(g.d_mlp_up, stream);
        if (g.d_swiglu.Data) fill_zero(g.d_swiglu, stream);
        if (g.d_mlp_down.Data) fill_zero(g.d_mlp_down, stream);
        if (g.d_att.Data) fill_zero(g.d_att, stream);
        if (g.d_qkv.Data) fill_zero(g.d_qkv, stream);
        if (g.d_ln1.Data) fill_zero(g.d_ln1, stream);
    }
}

void DslRunState::allocate_simplified_quant_buffers(const PretrainedConfig& cfg, const RuntimeOptions& options) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    const long AttnDim = Hq * D;
    const long QKV = D * (Hq + 2 * Hkv);
    const long M = cfg.IntermediateSize;
    const long MUp = 2 * M;

    if (mEnableFp8Forward) {
        modules::allocate_fp8_forward_buffers(
            mFP8ForwardQuants, mFP8ForwardStats, *mAllocator,
            B, T, C, M, AttnDim, options.forward_matmul_dtype());
    }

    if (options.fp8_hybrid_enabled()) {
        modules::FP8ScalingConfig fp8_cfg{};
        fp8_cfg.amax_history_len = options.RecipeOptions.fp8_amax_history_len;
        fp8_cfg.margin = static_cast<float>(options.RecipeOptions.fp8_margin);
        mFP8ScalingState = std::make_unique<modules::FP8ScalingState>(
            fp8_cfg, mAllocator, DeviceId, cfg.NumLayers);
    }

    if (mGradQuantDtype == mGradDtype) {
        const std::array<long, 3> ln_shape{B, T, C};
        const std::array<long, 3> mlp_up_shape{B, T, MUp};
        const std::array<long, 3> qkv_shape{B, T, QKV};
        mSimplifiedQuantGrads.d_res_ffn = Tensor::from_pointer(nullptr, DeviceId, mGradQuantDtype, ln_shape);
        mSimplifiedQuantGrads.d_res_att = Tensor::from_pointer(nullptr, DeviceId, mGradQuantDtype, ln_shape);
        mSimplifiedQuantGrads.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, mGradQuantDtype, mlp_up_shape);
        mSimplifiedQuantGrads.d_qkv = Tensor::from_pointer(nullptr, DeviceId, mGradQuantDtype, qkv_shape);
        return;
    }

    mGradQuantStats = mAllocator->allocate(ETensorDType::FP32, "dsl_grad_quant_stats",
                                           EAllocationType::ON_DEVICE, {8L});
    float* stats = mGradQuantStats.get<float>();

    auto alloc = [&](ETensorDType dtype, const std::string& name, const std::vector<long>& shape) -> Tensor {
        return mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
    };

    mSimplifiedQuantGrads.d_res_ffn = alloc(mGradQuantDtype, "dsl_d_res_ffn_q", {B, T, C});
    mSimplifiedQuantGrads.d_res_ffn.Stats = stats + 0;
    mSimplifiedQuantGrads.d_res_att = alloc(mGradQuantDtype, "dsl_d_res_att_q", {B, T, C});
    mSimplifiedQuantGrads.d_res_att.Stats = stats + 2;
    mSimplifiedQuantGrads.d_mlp_up = alloc(mGradQuantDtype, "dsl_d_mlp_up_q", {B, T, MUp});
    mSimplifiedQuantGrads.d_mlp_up.Stats = stats + 4;
    mSimplifiedQuantGrads.d_qkv = alloc(mGradQuantDtype, "dsl_d_qkv_q", {B, T, QKV});
    mSimplifiedQuantGrads.d_qkv.Stats = stats + 6;
}

void DslRunState::allocate_scratch_buffers(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    const long QKV = D * (Hq + 2 * Hkv);

    const long rmsnorm_scratch_bytes = static_cast<long>(get_rmsnorm_backward_scratch_size(static_cast<int>(C), DeviceProp));
    mScratch.rmsnorm_scratch = mAllocator->allocate(
        ETensorDType::BYTE, "rmsnorm_scratch", EAllocationType::ON_DEVICE, {rmsnorm_scratch_bytes});

    const long M = cfg.IntermediateSize;
    const long MUp = 2 * M;
    const long V = cfg.VocabSize;
    const long max_bias_channels = std::max<long>(QKV, std::max<long>(C, std::max<long>(MUp, V)));
    const long bias_scratch_bytes =
        static_cast<long>(get_bias_backward_scratch_size(mGradDtype, static_cast<int>(max_bias_channels), DeviceProp));
    mScratch.matmul_bias_scratch = mAllocator->allocate(
        ETensorDType::FP32, "bias_scratch", EAllocationType::ON_DEVICE, {bias_scratch_bytes / static_cast<long>(sizeof(float))});

    const long num_block_sums = std::max<long>(2, static_cast<long>(get_max_num_block_sums(DeviceProp)));
    mScratch.norm_buffer = mAllocator->allocate(
        ETensorDType::FP32, "norm_buffer", EAllocationType::ON_DEVICE, {num_block_sums});

    mScratch.matmul_scales = mAllocator->allocate(
        ETensorDType::FP32, "matmul_scales", EAllocationType::ON_DEVICE, {2L});

    const long BT = B * T;
    mScratch.cross_entropy_dloss = mAllocator->allocate(
        ETensorDType::FP32, "cross_entropy_dloss", EAllocationType::ON_DEVICE, {BT});
    mScratch.cross_entropy_logsumexp = mAllocator->allocate(
        ETensorDType::FP32, "cross_entropy_logsumexp", EAllocationType::ON_DEVICE, {BT});
    const int n_chunks = static_cast<int>(
        (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE);
    if (n_chunks > 1) {
        mScratch.cross_entropy_chunk_logsumexp = mAllocator->allocate(
            ETensorDType::FP32, "cross_entropy_chunk_logsumexp", EAllocationType::ON_DEVICE,
            {BT, n_chunks});
    }

    // Encoder backward scratch buffers - skip in LoRA-only mode since embedding backward is skipped entirely
    if (!mLoraOnlyMode) {
        const long group_width = static_cast<long>(16 / get_dtype_size(mGradDtype) * 32);
        const long num_c_groups = (C + group_width - 1) / group_width;
        mScratch.encoder_bwd_scratch = mAllocator->allocate(
            ETensorDType::INT32, "encoder_bwd_scratch", EAllocationType::ON_DEVICE, {B, T, num_c_groups * 5});
        mScratch.encoder_bwd_indices = mAllocator->allocate(
            ETensorDType::INT32, "encoder_bwd_indices", EAllocationType::PINNED, {B, T, num_c_groups});
        mScratch.encoder_bwd_info = mAllocator->allocate(
            ETensorDType::INT32, "encoder_bwd_info", EAllocationType::PINNED, {B, T, 4 * num_c_groups});
    }

    const int attn_chunks = mAttnBwdChunks;
    if (attn_chunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    const long attn_ws_batch_size =
        (attn_chunks == 1) ? B : div_exact(B, static_cast<long>(attn_chunks));
    const bool cudnn_ok = (D > 0 && Hq > 0 && Hkv > 0 && (D % 8 == 0) && D <= 128);
    if (cudnn_ok) {
        const long cudnn_ws_size = static_cast<long>(
            cudnn_get_workspace_size(static_cast<int>(attn_ws_batch_size), static_cast<int>(T),
                                     static_cast<int>(Hq), static_cast<int>(Hkv),
                                     static_cast<int>(D), CudnnHandle));
        // Pre-allocate cudnn_workspace using the persistent allocator to avoid overlap with
        // stack-allocated gradient buffers. The workspace is large (~192MB) and if allocated
        // from the temp stack, checkpoint restores during backward can cause it to be reallocated
        // in a region that overlaps with gradient buffers.
        mScratch.cudnn_workspace = mAllocator->allocate(
            ETensorDType::BYTE, "cudnn_workspace", EAllocationType::ON_DEVICE, {cudnn_ws_size});
    } else {
        // Leave an empty descriptor; attention ops will fail later if invoked with invalid head size.
        mScratch.cudnn_workspace = Tensor::empty(ETensorDType::BYTE, {0});
    }

    // Note: Stack simulation no longer needed for workspace since it's persistently allocated
    if (mStackSimulate) {
        if (mRecomputeLevel >= RecomputeLevel::Enabled) {
            const long d_qkv_bytes = B * T * QKV * get_dtype_size(mGradDtype);
            auto* simulated_d_qkv = Stack.allocate(static_cast<std::size_t>(d_qkv_bytes), "d_qkv_simulate");
            Stack.free(simulated_d_qkv);
        }
    }
}

Tensor* DslRunState::get_fp8_forward_buffer(int op) {
    if (!has_fp8_forward()) return nullptr;
    auto matmul_op = static_cast<modules::MatmulOp>(op);
    switch (matmul_op) {
        case modules::MatmulOp::QKV:
            return &mFP8ForwardQuants.ln1;
        case modules::MatmulOp::MLPUp:
            return &mFP8ForwardQuants.ln2;
        case modules::MatmulOp::AttnOut:
            return &mFP8ForwardQuants.att;
        case modules::MatmulOp::MLPDown:
            return &mFP8ForwardQuants.swiglu;
        default:
            return nullptr;
    }
}

Tensor* DslRunState::get_gradient_quant_buffer(int op) {
    if (!has_grad_quants()) return nullptr;
    auto matmul_op = static_cast<modules::MatmulOp>(op);
    switch (matmul_op) {
        case modules::MatmulOp::QKV:
            return &mSimplifiedQuantGrads.d_qkv;
        case modules::MatmulOp::MLPUp:
            return &mSimplifiedQuantGrads.d_mlp_up;
        case modules::MatmulOp::AttnOut:
            return &mSimplifiedQuantGrads.d_res_att;
        case modules::MatmulOp::MLPDown:
            return &mSimplifiedQuantGrads.d_res_ffn;
        default:
            return nullptr;
    }
}

void DslRunState::allocate_residual_buffers(const PretrainedConfig& cfg, bool offload_residuals) {
    mOffloadResiduals = offload_residuals;
    mResidualManager = std::make_unique<modules::ResidualManager>(
        mAllocator,
        cfg.NumLayers,
        static_cast<int>(B),
        static_cast<int>(T),
        cfg.HiddenSize,
        cfg.DType,
        offload_residuals,
        /*num_residual_buffers=*/2,
        MainStream);
}

void DslRunState::fetch_residual(int layer_idx, cudaStream_t stream) {
    if (mResidualManager) {
        mResidualManager->fetch_residual(layer_idx, stream);
    }
}

void DslRunState::put_residual(int layer_idx, cudaStream_t stream) {
    if (mResidualManager) {
        mResidualManager->put_residual(layer_idx, stream);
    }
}

void DslRunState::mark_residual_ready(int layer_idx, cudaStream_t stream) {
    if (mResidualManager) {
        mResidualManager->mark_residual_ready(layer_idx, stream);
    }
}

void DslRunState::release_residual(int layer_idx, cudaStream_t stream) {
    if (mResidualManager) {
        mResidualManager->release_residual(layer_idx, stream);
    }
}

void DslRunState::create_cuda_resources() {
    CUDA_CHECK(cudaStreamCreate(&mSideStream));
    CUDA_CHECK(cudaEventCreate(&mSideStreamEvent));
    CUDA_CHECK(cudaEventCreate(&mAllReduceDone));
    CUBLAS_CHECK(cublasCreate(&mCublasHandle));
    CUBLAS_CHECK(cublasSetMathMode(mCublasHandle, CUBLAS_TF32_TENSOR_OP_MATH));
}

void DslRunState::release_cuda_resources() noexcept {
    if (mCublasHandle) {
        cublasDestroy(mCublasHandle);
        mCublasHandle = nullptr;
    }
    if (mAllReduceDone) {
        cudaEventDestroy(mAllReduceDone);
        mAllReduceDone = nullptr;
    }
    if (mSideStreamEvent) {
        cudaEventDestroy(mSideStreamEvent);
        mSideStreamEvent = nullptr;
    }
    if (mSideStream) {
        cudaStreamDestroy(mSideStream);
        mSideStream = nullptr;
    }
}

void DslRunState::allocate_graph_arrays(int num_layers) {
    mForwardBlockGraphs.resize(static_cast<std::size_t>(num_layers), nullptr);
    mBackwardBlockGraphs.resize(static_cast<std::size_t>(num_layers), {nullptr, nullptr});
    mForwardBlockStackCheckpoints.resize(static_cast<std::size_t>(num_layers));
    mBackwardBlockStackCheckpoints.resize(static_cast<std::size_t>(num_layers));
}

void DslRunState::destroy_cuda_graphs() noexcept {
    for (auto& g : mForwardBlockGraphs) {
        if (g) {
            (void)cudaGraphExecDestroy(g);
            g = nullptr;
        }
    }
    for (auto& arr : mBackwardBlockGraphs) {
        for (auto& g : arr) {
            if (g) {
                (void)cudaGraphExecDestroy(g);
                g = nullptr;
            }
        }
    }
}

void DslRunState::reset_cuda_graphs() {
    destroy_cuda_graphs();
    // Reset checkpoints to default
    for (auto& cp : mForwardBlockStackCheckpoints) {
        cp = DeviceMemoryStack::Checkpoint{};
    }
    for (auto& arr : mBackwardBlockStackCheckpoints) {
        arr[0] = DeviceMemoryStack::Checkpoint{};
        arr[1] = DeviceMemoryStack::Checkpoint{};
    }
}

void DslRunState::configure_forward_graphs(bool hooked) {
    if (mForwardGraphsHooked == hooked) {
        return;
    }
    // Graph topology changes when hooks are added/removed - must re-capture
    for (auto& g : mForwardBlockGraphs) {
        if (g) {
            (void)cudaGraphExecDestroy(g);
            g = nullptr;
        }
    }
    mForwardGraphsHooked = hooked;
}

void DslRunState::configure_backward_graphs(bool hooked) {
    if (mBackwardGraphsHooked == hooked) {
        return;
    }
    // Graph topology changes when hooks are added/removed - must re-capture
    for (auto& arr : mBackwardBlockGraphs) {
        for (auto& g : arr) {
            if (g) {
                (void)cudaGraphExecDestroy(g);
                g = nullptr;
            }
        }
    }
    mBackwardGraphsHooked = hooked;
}

} // namespace dsl
