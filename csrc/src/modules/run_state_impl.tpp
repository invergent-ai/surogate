// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_RUN_STATE_IMPL_TPP
#define SUROGATE_SRC_MODULES_RUN_STATE_IMPL_TPP

#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdio>
#include <fmt/format.h>

#include "modules/mlp_utils.h"

namespace modules {

template<typename Block>
ModularRunState<Block>::ModularRunState(
    const typename ModularRunState<Block>::Config& config, DeviceMemoryStack& stack,
    const std::shared_ptr<TensorAllocator>& allocator)
    : IRunState(config.pretrained_config->clone(), config.batch_size, config.seq_length, allocator)
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
    , mSharedDMlpDown(std::move(other.mSharedDMlpDown))
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
    , mResidualManager(std::move(other.mResidualManager))
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
        mSharedDMlpDown = std::move(other.mSharedDMlpDown);
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
        mResidualManager = std::move(other.mResidualManager);
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
    mResidualManager->fetch_residual(layer_idx, fetch_stream);
}

template<typename Block>
void ModularRunState<Block>::put_residual(int layer_idx, cudaStream_t put_stream) {
    mResidualManager->put_residual(layer_idx, put_stream);
}

template<typename Block>
Tensor& ModularRunState<Block>::get_residual(int layer_idx, cudaStream_t stream) {
    return mResidualManager->get_residual(layer_idx, stream);
}

template<typename Block>
void ModularRunState<Block>::mark_residual_ready(int layer_idx, cudaStream_t stream) {
    mResidualManager->mark_residual_ready(layer_idx, stream);
}

template<typename Block>
void ModularRunState<Block>::release_residual(int layer_idx, cudaStream_t stream) {
    mResidualManager->release_residual(layer_idx, stream);
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
    long intermediate_size = mlp_up_rows(mConfig.block_config);

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
    long M = mlp_up_rows(mConfig.block_config);
    int Hq = mConfig.block_config.num_query_heads;
    int Hkv = mConfig.block_config.num_kv_heads;
    int HS = mConfig.block_config.head_size;
    long AttC = static_cast<long>(HS) * static_cast<long>(Hq);
    long qkv_channels = HS * (Hq + 2 * Hkv);
    const bool has_mamba = mConfig.has_mamba;
    const int mamba_dim = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_dim(); }) {
            return mConfig.block_config.mamba_dim();
        }
        return 0;
    }();
    const int mamba_heads = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_num_heads; }) {
            return mConfig.block_config.mamba_num_heads;
        }
        return 0;
    }();
    const int mamba_state = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_ssm_state_size; }) {
            return mConfig.block_config.mamba_ssm_state_size;
        }
        return 0;
    }();
    const int mamba_groups = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_n_groups; }) {
            return mConfig.block_config.mamba_n_groups;
        }
        return 0;
    }();
    const int mamba_conv_dim = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_conv_dim(); }) {
            return mConfig.block_config.mamba_conv_dim();
        }
        return 0;
    }();
    const int mamba_chunk_size = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_chunk_size; }) {
            return mConfig.block_config.mamba_chunk_size > 0 ? mConfig.block_config.mamba_chunk_size : 2048;
        }
        return 2048;
    }();
    const int mamba_chunks = (has_mamba && mamba_chunk_size > 0) ? (T + mamba_chunk_size - 1) / mamba_chunk_size : 0;
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
    // EXCEPTION: When recompute_block is enabled, we CAN share ln1/ln2 because LoRA backward will
    // recompute them on-the-fly instead of reading stored per-layer values.
    // NOTE: att is still kept per-layer in LoRA mode even with recompute_block because recomputing
    // attention is expensive (requires full QKV + attention forward). This is only needed when
    // O projection has LoRA adapters. Future optimization: skip sharing att only when O is targeted.
    const bool lora_only = mConfig.lora_only_mode;
    const bool lora_can_share_ln = !lora_only || mConfig.recompute_block;  // ln1/ln2 can be shared with recompute_block
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
        mSharedMlpUp = mAllocator->allocate(dtype, "mlp_up_shared", kind, {B, T, M});
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
        // Note: shape metadata is {B, T, Hq} but memory layout is {B, Hq, T} as cuDNN expects
        // The total size B*T*Hq is the same, and cuDNN uses raw pointers with explicit dimensions
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
            const std::array<long, 3> mlp_up_shape_arr{B, T, M};
            const std::array<long, 3> swiglu_shape_arr{B, T, D};
            acts.mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, mlp_up_shape_arr);
            acts.swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, swiglu_shape_arr);
        } else {
            acts.mlp_up = share_mlp_up ? mSharedMlpUp : mAllocator->allocate(
                dtype, "mlp_up", kind, {B, T, M});
            // swiglu is needed - it's the input X for LoRA down backward
            acts.swiglu = share_swiglu ? mSharedSwiGlu : mAllocator->allocate(
                dtype, "swiglu", kind, {B, T, D});
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

        // Mamba / SSM activations (only for Mamba layers)
        const bool is_mamba_layer = has_mamba && (i < static_cast<int>(mConfig.layer_is_mamba.size()))
            && (mConfig.layer_is_mamba[static_cast<std::size_t>(i)] != 0);
        if (is_mamba_layer) {
            acts.mamba_gate = mAllocator->allocate(dtype, "mamba_gate", kind, {B, T, (long)mamba_dim});
            acts.mamba_conv_in = mAllocator->allocate(dtype, "mamba_conv_in", kind, {B, (long)mamba_conv_dim, T});
            acts.mamba_u = mAllocator->allocate(dtype, "mamba_u", kind, {B, (long)mamba_dim, T});
            acts.mamba_delta = mAllocator->allocate(dtype, "mamba_delta", kind, {B, (long)mamba_dim, T});
            acts.mamba_B = mAllocator->allocate(dtype, "mamba_B", kind, {B, (long)mamba_groups, (long)mamba_state, T});
            acts.mamba_C = mAllocator->allocate(dtype, "mamba_C", kind, {B, (long)mamba_groups, (long)mamba_state, T});
            acts.mamba_scan_out = mAllocator->allocate(dtype, "mamba_scan_out", kind, {B, T, (long)mamba_dim});
            acts.mamba_gated = mAllocator->allocate(dtype, "mamba_gated", kind, {B, T, (long)mamba_dim});
            acts.mamba_normed = mAllocator->allocate(dtype, "mamba_normed", kind, {B, T, (long)mamba_dim});
            acts.mamba_rstd = mAllocator->allocate(ETensorDType::FP32, "mamba_rstd", kind, {B, T, (long)mamba_groups});
            acts.mamba_x = mAllocator->allocate(ETensorDType::FP32, "mamba_x", kind,
                                                {B, (long)mamba_dim, (long)mamba_chunks, (long)mamba_state * 2});
        } else {
            acts.mamba_gate = {};
            acts.mamba_conv_in = {};
            acts.mamba_u = {};
            acts.mamba_delta = {};
            acts.mamba_B = {};
            acts.mamba_C = {};
            acts.mamba_scan_out = {};
            acts.mamba_gated = {};
            acts.mamba_normed = {};
            acts.mamba_rstd = {};
            acts.mamba_x = {};
        }
    }
}

template<typename Block>
void ModularRunState<Block>::allocate_simplified_gradients() {
    long B = mConfig.batch_size;
    long T = mConfig.seq_length;
    long C = mConfig.hidden_size;
    long D = mConfig.block_config.intermediate_size;
    long M = mlp_up_rows(mConfig.block_config);
    int Hq = mConfig.block_config.num_query_heads;
    int Hkv = mConfig.block_config.num_kv_heads;
    int HS = mConfig.block_config.head_size;
    long AttC = static_cast<long>(HS) * static_cast<long>(Hq);
    long qkv_channels = HS * (Hq + 2 * Hkv);
    const bool has_mamba = mConfig.has_mamba;
    const int mamba_dim = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_dim(); }) {
            return mConfig.block_config.mamba_dim();
        }
        return 0;
    }();
    const int mamba_state = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_ssm_state_size; }) {
            return mConfig.block_config.mamba_ssm_state_size;
        }
        return 0;
    }();
    const int mamba_groups = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_n_groups; }) {
            return mConfig.block_config.mamba_n_groups;
        }
        return 0;
    }();
    const int mamba_conv_dim = [&]() -> int {
        if constexpr (requires { mConfig.block_config.mamba_conv_dim(); }) {
            return mConfig.block_config.mamba_conv_dim();
        }
        return 0;
    }();

    auto dtype = mConfig.grad_dtype;
    auto kind = EAllocationType::ON_DEVICE;

    const bool share_grads =
        mConfig.recompute_attention || mConfig.recompute_ffn || mConfig.recompute_block;
    const bool share_res_ffn = mConfig.recompute_block;
    const bool share_mlp_down = mConfig.recompute_block;
    const bool large_temps_on_stack = mConfig.recompute_block;

    // Allocate shared gradient intermediates if we're in a recompute mode.
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
        mSharedDAtt = mAllocator->allocate(dtype, "d_att_shared", kind, {B, T, AttC});
        mSharedDLn1 = mAllocator->allocate(dtype, "d_ln1_shared", kind, {B, T, C});
        if (!large_temps_on_stack) {
            mSharedDMlpUp = mAllocator->allocate(dtype, "d_mlp_up_shared", kind, {B, T, M});
            mSharedDSwiGlu = mAllocator->allocate(dtype, "d_swiglu_shared", kind, {B, T, D});
            mSharedDQKV = mAllocator->allocate(dtype, "d_qkv_shared", kind, {B, T, qkv_channels});
        }
    }

    mSimplifiedGradients.resize(mConfig.num_layers);
    for (int i = 0; i < mConfig.num_layers; ++i) {
        auto& g = mSimplifiedGradients[i];
        g.d_res_ffn = share_res_ffn ? mSharedDResFFN[static_cast<std::size_t>(i % 2)] : mAllocator->allocate(dtype, "d_res_ffn", kind, {B, T, C});
        g.d_res_att = share_grads ? mSharedDResAtt : mAllocator->allocate(dtype, "d_res_att", kind, {B, T, C});
        g.d_ln2 = share_grads ? mSharedDLn2 : mAllocator->allocate(dtype, "d_ln2", kind, {B, T, C});
        g.d_att = share_grads ? mSharedDAtt : mAllocator->allocate(dtype, "d_att", kind, {B, T, AttC});
        g.d_ln1 = share_grads ? mSharedDLn1 : mAllocator->allocate(dtype, "d_ln1", kind, {B, T, C});
        if (large_temps_on_stack) {
            const std::array<long, 3> d_mlp_up_shape_arr{B, T, M};
            const std::array<long, 3> d_swiglu_shape_arr{B, T, D};
            const std::array<long, 3> d_qkv_shape_arr{B, T, qkv_channels};
            g.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, d_mlp_up_shape_arr);
            g.d_swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, d_swiglu_shape_arr);
            g.d_qkv = Tensor::from_pointer(nullptr, DeviceId, dtype, d_qkv_shape_arr);
        } else {
            // d_mlp_up is handled in-place on the mlp_up buffer (matches legacy LLamaModel).
            // This saves memory and ensures consistent in-place behavior across all modes.
            g.d_mlp_up = mSimplifiedActivations[i].mlp_up;
            g.d_swiglu = share_grads ? mSharedDSwiGlu : mAllocator->allocate(dtype, "d_swiglu", kind, {B, T, D});
            g.d_qkv = share_grads ? mSharedDQKV : mAllocator->allocate(dtype, "d_qkv", kind, {B, T, qkv_channels});
        }
        g.d_mlp_down = share_mlp_down ? mSharedDMlpDown[static_cast<std::size_t>(i % 2)] : mAllocator->allocate(dtype, "d_mlp_down", kind, {B, T, C});

        // Mamba gradients (only for Mamba layers)
        const bool is_mamba_layer = has_mamba && (i < static_cast<int>(mConfig.layer_is_mamba.size()))
            && (mConfig.layer_is_mamba[static_cast<std::size_t>(i)] != 0);
        if (is_mamba_layer) {
            g.d_mamba_normed = mAllocator->allocate(dtype, "d_mamba_normed", kind, {B, T, (long)mamba_dim});
            g.d_mamba_gated = mAllocator->allocate(dtype, "d_mamba_gated", kind, {B, T, (long)mamba_dim});
            g.d_mamba_scan_out = mAllocator->allocate(dtype, "d_mamba_scan_out", kind, {B, T, (long)mamba_dim});
            g.d_mamba_u = mAllocator->allocate(dtype, "d_mamba_u", kind, {B, (long)mamba_dim, T});
            g.d_mamba_delta = mAllocator->allocate(dtype, "d_mamba_delta", kind, {B, (long)mamba_dim, T});
            g.d_mamba_B = mAllocator->allocate(ETensorDType::FP32, "d_mamba_B", kind, {B, (long)mamba_groups, (long)mamba_state, T});
            g.d_mamba_C = mAllocator->allocate(ETensorDType::FP32, "d_mamba_C", kind, {B, (long)mamba_groups, (long)mamba_state, T});
            g.d_mamba_conv_out = mAllocator->allocate(dtype, "d_mamba_conv_out", kind, {B, (long)mamba_conv_dim, T});
        } else {
            g.d_mamba_normed = {};
            g.d_mamba_gated = {};
            g.d_mamba_scan_out = {};
            g.d_mamba_u = {};
            g.d_mamba_delta = {};
            g.d_mamba_B = {};
            g.d_mamba_C = {};
            g.d_mamba_conv_out = {};
        }
    }
}

template<typename Block>
void ModularRunState<Block>::allocate_simplified_quant_buffers() {
    long B = mConfig.batch_size;
    long T = mConfig.seq_length;
    long C = mConfig.hidden_size;
    long D = mConfig.block_config.intermediate_size;
    long M = mlp_up_rows(mConfig.block_config);
    int Hq = mConfig.block_config.num_query_heads;
    int Hkv = mConfig.block_config.num_kv_heads;
    int HS = mConfig.block_config.head_size;
    long AttC = static_cast<long>(HS) * static_cast<long>(Hq);
    long qkv_channels = HS * (Hq + 2 * Hkv);

    const bool need_act_quants = (mConfig.matmul_dtype != mConfig.activation_dtype);
    const bool need_grad_quants = (mConfig.grad_quant_dtype != mConfig.grad_dtype);

    mSimplifiedQuantActivations.resize(mConfig.num_layers);

    if (mConfig.enable_fp8_forward) {
        allocate_fp8_forward_buffers(
            mFP8ForwardQuants, mFP8ForwardStats, *mAllocator,
            B, T, C, D, AttC, mConfig.forward_matmul_dtype);
    }

    if (mConfig.enable_fp8_hybrid_delayed) {
        mFP8ScalingState = std::make_unique<FP8ScalingState>(
            mConfig.fp8_scaling_config, mAllocator, DeviceId, mConfig.num_layers);
    }

    if (mConfig.enable_fp4_forward || mConfig.enable_fp4_backward) {
        allocate_fp4_forward_buffers(
            mFP4ForwardQuants, *mAllocator,
            B, T, C, D, AttC, mConfig.activation_dtype);
    }

    if (!need_act_quants && !need_grad_quants) {
        const std::array<long, 3> ln_shape{B, T, C};
        const std::array<long, 3> att_shape{B, T, AttC};
        const std::array<long, 3> swiglu_shape{B, T, D};
        const std::array<long, 3> qkv_shape{B, T, qkv_channels};
        const std::array<long, 3> mlp_up_shape{B, T, M};
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
        mSharedAttQuant = alloc(mConfig.matmul_dtype, "att_q_shared", {B, T, AttC});
        if (share_swiglu_data) {
            mSharedSwiGluQuant = alloc(mConfig.matmul_dtype, "swiglu_q_shared", {B, T, D});
        }

        for (int i = 0; i < mConfig.num_layers; ++i) {
            auto& q = mSimplifiedQuantActivations[i];
            float* layer_stats = stats + 8L * i;

            if (share_ln_data) {
                q.ln1 = mSharedLn1Quant;
            } else {
                q.ln1 = alloc(mConfig.matmul_dtype, "ln1_q_l" + std::to_string(i), {B, T, C});
            }
            q.ln1.Stats = layer_stats + 0;

            if (share_ln_data) {
                q.ln2 = mSharedLn2Quant;
            } else {
                q.ln2 = alloc(mConfig.matmul_dtype, "ln2_q_l" + std::to_string(i), {B, T, C});
            }
            q.ln2.Stats = layer_stats + 2;

            q.att = mSharedAttQuant;
            q.att.Stats = layer_stats + 4;

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
        mSimplifiedQuantGrads.d_mlp_up = alloc(mConfig.grad_quant_dtype, "d_mlp_up_q", {B, T, M});
        mSimplifiedQuantGrads.d_mlp_up.Stats = gstats + 4;
        mSimplifiedQuantGrads.d_qkv = alloc(mConfig.grad_quant_dtype, "d_qkv_q", {B, T, qkv_channels});
        mSimplifiedQuantGrads.d_qkv.Stats = gstats + 6;
    } else {
        const std::array<long, 3> ln_shape{B, T, C};
        const std::array<long, 3> qkv_shape{B, T, qkv_channels};
        const std::array<long, 3> mlp_up_shape{B, T, M};
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

    mNonBlockActivations.encoded = mAllocator->allocate(
        dtype, "encoded", kind, {B, T, C});
    mNonBlockActivations.ln_final = mAllocator->allocate(
        dtype, "ln_final", kind, {B, T, C});
    mNonBlockActivations.ln_final_rstd = mAllocator->allocate(
        ETensorDType::FP32, "ln_final_rstd", kind, {B, T});

    long lmhead_chunks = mConfig.lmhead_chunks;
    long out_size = (B * T) / lmhead_chunks;
    int dev = DeviceId;
    mNonBlockActivations.output = Tensor{dtype, {out_size, V}, nullptr, nullptr, 2, dev};

    std::byte* simulated_output = Stack.allocate(mNonBlockActivations.output.bytes(), "output_simulate");
    Stack.free(simulated_output);

    if (!mConfig.use_fused_rope) {
        int max_seq_len = std::min((int)T, mConfig.pretrained_config->MaxPositionEmbeddings);
        int head_size = mConfig.block_config.head_size;
        float rope_theta = mConfig.block_config.rope.theta;

        mNonBlockActivations.freq_cis = mAllocator->allocate(
            dtype, "freq_cis", kind, {(long)max_seq_len, (long)(2 * head_size)});

        if (dtype == ETensorDType::BF16) {
            std::vector<nv_bfloat16> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            precompute_freqs_cis(freq_cpu.data(), head_size, max_seq_len, rope_theta);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data, freq_cpu.data(),
                                  freq_cpu.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        } else if (dtype == ETensorDType::FP32) {
            std::vector<float> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            precompute_freqs_cis(freq_cpu.data(), head_size, max_seq_len, rope_theta);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data, freq_cpu.data(),
                                  freq_cpu.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    } else {
        mNonBlockActivations.freq_cis = {};
    }

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

    const long rmsnorm_scratch_bytes = static_cast<long>(get_rmsnorm_backward_scratch_size((int)C, DeviceProp));
    mScratch.rmsnorm_scratch = mAllocator->allocate(
        ETensorDType::BYTE, "rmsnorm_scratch",
        EAllocationType::ON_DEVICE, {rmsnorm_scratch_bytes});

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

    const long num_block_sums = std::max<long>(2, static_cast<long>(get_max_num_block_sums(DeviceProp)));
    mScratch.norm_buffer = mAllocator->allocate(
        ETensorDType::FP32, "norm_buffer",
        EAllocationType::ON_DEVICE, {num_block_sums});

    mScratch.matmul_scales = mAllocator->allocate(
        ETensorDType::FP32, "matmul_scales",
        EAllocationType::ON_DEVICE, {2L});

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

    int Hq = mConfig.block_config.num_query_heads;
    int Hkv = mConfig.block_config.num_kv_heads;
    int HS = mConfig.block_config.head_size;
    const int attn_chunks = mConfig.attention_bwd_chunks;
    if (attn_chunks < 1) {
        throw std::runtime_error("attention_bwd_chunks must be >= 1");
    }
    const long attn_ws_batch_size =
        (attn_chunks == 1) ? B : div_exact(B, static_cast<long>(attn_chunks));
    long cudnn_ws_size = static_cast<long>(
        cudnn_get_workspace_size(static_cast<int>(attn_ws_batch_size), static_cast<int>(T), Hq, Hkv, HS, CudnnHandle));
    mScratch.cudnn_workspace = Tensor{ETensorDType::BYTE, {cudnn_ws_size}, nullptr, nullptr, 1, DeviceId};

    if (mConfig.recompute_block) {
        int qkv_channels = HS * (Hq + 2 * Hkv);
        long d_qkv_bytes = B * T * qkv_channels * get_dtype_size(mConfig.grad_dtype);
        std::byte* simulated_d_qkv = stack.allocate(d_qkv_bytes, "d_qkv_simulate");
        std::byte* simulated_ws = stack.allocate(mScratch.cudnn_workspace.bytes(), "workspace");
        stack.free(simulated_ws);
        stack.free(simulated_d_qkv);
    } else {
        std::byte* simulated_ws = stack.allocate(mScratch.cudnn_workspace.bytes(), "workspace");
        stack.free(simulated_ws);
    }

    if (mConfig.recompute_block) {
        long D = mConfig.block_config.intermediate_size;
        long M = mlp_up_rows(mConfig.block_config);
        long mlp_up_bytes = B * T * M * get_dtype_size(mConfig.activation_dtype);
        long swiglu_bytes = B * T * D * get_dtype_size(mConfig.activation_dtype);
        long d_swiglu_bytes = B * T * D * get_dtype_size(mConfig.grad_dtype);
        std::byte* simulated_mlp_up = stack.allocate(mlp_up_bytes, "mlp_up_simulate");
        std::byte* simulated_swiglu = stack.allocate(swiglu_bytes, "swiglu_simulate");
        std::byte* simulated_d_swiglu = stack.allocate(d_swiglu_bytes, "d_swiglu_simulate");

        if (mConfig.enable_fp8_hybrid_delayed || mConfig.enable_fp8_forward) {
            long stats_bytes = 2 * sizeof(float);
            long down_weight_e4m3_bytes = C * D * sizeof(__nv_fp8_e4m3);
            long down_weight_tp_bytes = D * C * sizeof(__nv_fp8_e4m3);
            long down_act_tp_bytes = D * B * T * sizeof(__nv_fp8_e4m3);
            long down_grad_tp_bytes = C * B * T * sizeof(__nv_fp8_e5m2);

            std::byte* sim_down_weight_e4m3 = stack.allocate(down_weight_e4m3_bytes, "fp8_down_weight_e4m3");
            std::byte* sim_down_weight_e4m3_stats = stack.allocate(stats_bytes, "fp8_down_weight_e4m3_stats");
            std::byte* sim_down_weight_tp = stack.allocate(down_weight_tp_bytes, "fp8_down_weight_tp");
            std::byte* sim_down_weight_tp_stats = stack.allocate(stats_bytes, "fp8_down_weight_tp_stats");

            stack.free(sim_down_weight_tp_stats);
            stack.free(sim_down_weight_tp);
            stack.free(sim_down_weight_e4m3_stats);
            stack.free(sim_down_weight_e4m3);

            std::byte* sim_down_act_tp = stack.allocate(down_act_tp_bytes, "fp8_down_act_tp");
            std::byte* sim_down_act_stats = stack.allocate(stats_bytes, "fp8_down_act_stats");
            std::byte* sim_down_grad_tp = stack.allocate(down_grad_tp_bytes, "fp8_down_grad_tp");
            std::byte* sim_down_grad_stats = stack.allocate(stats_bytes, "fp8_down_grad_stats");
            stack.free(sim_down_grad_stats);
            stack.free(sim_down_grad_tp);
            stack.free(sim_down_act_stats);
            stack.free(sim_down_act_tp);

            stack.free(simulated_d_swiglu);
            stack.free(simulated_swiglu);

            long up_weight_e4m3_bytes = M * C * sizeof(__nv_fp8_e4m3);
            long up_weight_tp_bytes = C * M * sizeof(__nv_fp8_e4m3);
            long up_act_tp_bytes = C * B * T * sizeof(__nv_fp8_e4m3);
            long up_grad_tp_bytes = M * B * T * sizeof(__nv_fp8_e5m2);

            std::byte* sim_up_weight_e4m3 = stack.allocate(up_weight_e4m3_bytes, "fp8_up_weight_e4m3");
            std::byte* sim_up_weight_e4m3_stats = stack.allocate(stats_bytes, "fp8_up_weight_e4m3_stats");
            std::byte* sim_up_weight_tp = stack.allocate(up_weight_tp_bytes, "fp8_up_weight_tp");
            std::byte* sim_up_weight_tp_stats = stack.allocate(stats_bytes, "fp8_up_weight_tp_stats");

            stack.free(sim_up_weight_tp_stats);
            stack.free(sim_up_weight_tp);
            stack.free(sim_up_weight_e4m3_stats);
            stack.free(sim_up_weight_e4m3);

            std::byte* sim_up_act_tp = stack.allocate(up_act_tp_bytes, "fp8_up_act_tp");
            std::byte* sim_up_act_stats = stack.allocate(stats_bytes, "fp8_up_act_stats");
            std::byte* sim_up_grad_tp = stack.allocate(up_grad_tp_bytes, "fp8_up_grad_tp");
            std::byte* sim_up_grad_stats = stack.allocate(stats_bytes, "fp8_up_grad_stats");
            stack.free(sim_up_grad_stats);
            stack.free(sim_up_grad_tp);
            stack.free(sim_up_act_stats);
            stack.free(sim_up_act_tp);

            stack.free(simulated_mlp_up);
        } else {
            stack.free(simulated_d_swiglu);
            stack.free(simulated_swiglu);
            stack.free(simulated_mlp_up);
        }
    }
}

template<typename Block>
void ModularRunState<Block>::allocate_residual_buffers() {
    mResidualManager = std::make_unique<ResidualManager>(
        mAllocator,
        mConfig.num_layers,
        mConfig.batch_size,
        mConfig.seq_length,
        mConfig.hidden_size,
        mConfig.activation_dtype,
        mConfig.offload_residuals,
        mConfig.num_residual_buffers,
        MainStream
    );
}

template<typename Block>
void ModularRunState<Block>::create_cuda_resources() {
    CUDA_CHECK(cudaStreamCreate(&mSideStream));
    CUDA_CHECK(cudaEventCreate(&mSideStreamEvent));
    CUDA_CHECK(cudaEventCreate(&mOptEmbeddingsDone));

    CUBLAS_CHECK(cublasCreate(&CublasHandle));

    mLayerUpdateDone.resize(static_cast<std::size_t>(mConfig.num_layers));
    for (int i = 0; i < mConfig.num_layers; ++i) {
        CUDA_CHECK(cudaEventCreate(&mLayerUpdateDone[static_cast<std::size_t>(i)]));
    }

    mForwardBlockGraphs.assign((std::size_t)mConfig.num_layers, nullptr);
    mBackwardBlockGraphs.resize((std::size_t)mConfig.num_layers);
    for (auto& arr : mBackwardBlockGraphs) {
        arr = {nullptr, nullptr};
    }

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

    if (CublasHandle) {
        cublasDestroy(CublasHandle);
        CublasHandle = nullptr;
    }

    for (auto& event : mLayerUpdateDone) {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    mLayerUpdateDone.clear();
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_RUN_STATE_IMPL_TPP
