// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_STATE_MANAGEMENT_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_STATE_MANAGEMENT_H

#include "lora_model_core.h"
#include "kernels/kernels.h"

namespace modules {

template<typename Block>
ModularLoRAModel<Block>::ModularLoRAModel(std::unique_ptr<ModularTransformerModel<Block>> base_model,
                                         const ModularLoRAConfig& lora_config,
                                         const RuntimeOptions& options,
                                         NCCLCommunicator& comm,
                                         const std::shared_ptr<TensorAllocator>& allocator,
                                         const QLoRAConfig& qlora_config)
    : mBaseModel(std::move(base_model))
    , mLoRAConfig(lora_config)
    , mQLoRAConfig(qlora_config)
    , mOptions(options)
    , mAllocator(allocator ? allocator : std::make_shared<TensorAllocator>())
    , mLoRAOptimizerRNG(42) {

    if (!lora_enabled()) {
        return;
    }

    const auto& cfg = mBaseModel->config();

    // Check if this is an MoE model - per-expert LoRA is used instead of MLP LoRA
    mIsMoEModel = (cfg.architecture == ArchitectureType::MoE) || cfg.moe_config.has_value();

    ModularLoRAWeightsManager::Config wm{};
    wm.num_layers = cfg.NumLayers;
    wm.hidden_size = cfg.HiddenSize;
    wm.intermediate_size = cfg.IntermediateSize;
    wm.num_query_heads = cfg.NumQueryHeads;
    wm.num_kv_heads = cfg.NumKeyValHeads;
    wm.head_size = cfg.head_size();
    wm.lora_config = mLoRAConfig;
    wm.work_dtype = cfg.DType;
    wm.shard_idx = comm.rank();
    wm.num_shards = comm.world_size();
    wm.is_moe = mIsMoEModel;
    if (mIsMoEModel && cfg.moe_config.has_value()) {
        wm.num_experts = cfg.moe_config->num_experts;
        wm.moe_intermediate_size = cfg.moe_config->moe_intermediate_size > 0
                                    ? cfg.moe_config->moe_intermediate_size
                                    : cfg.IntermediateSize;
        wm.train_router = mLoRAConfig.train_router;
    }
    mLoRAWeights = std::make_unique<ModularLoRAWeightsManager>(wm, *mAllocator);

    ModularLoRAGradsManager::Config gm{};
    gm.num_layers = cfg.NumLayers;
    gm.hidden_size = cfg.HiddenSize;
    gm.intermediate_size = cfg.IntermediateSize;
    gm.num_query_heads = cfg.NumQueryHeads;
    gm.num_kv_heads = cfg.NumKeyValHeads;
    gm.head_size = cfg.head_size();
    gm.lora_config = mLoRAConfig;
    gm.grad_dtype = mLoRAConfig.dtype;
    gm.shard_idx = comm.rank();
    gm.num_shards = comm.world_size();
    gm.is_moe = mIsMoEModel;
    if (mIsMoEModel && cfg.moe_config.has_value()) {
        gm.num_experts = cfg.moe_config->num_experts;
        gm.moe_intermediate_size = cfg.moe_config->moe_intermediate_size > 0
                                    ? cfg.moe_config->moe_intermediate_size
                                    : cfg.IntermediateSize;
        gm.train_router = mLoRAConfig.train_router;
    }
    mLoRAGrads = std::make_unique<ModularLoRAGradsManager>(gm, mAllocator);
}

template<typename Block>
void ModularLoRAModel<Block>::allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm, int B, int T, bool allocate_optimizer) {
    (void)allocate_optimizer;
    // Set the recipe on base model from RuntimeOptions (critical for FP8/FP4 recipes)
    if (options.TrainingRecipe) {
        mBaseModel->set_recipe(options.TrainingRecipe);
    }
    // Convert RuntimeOptions to ModelOptions and enable LoRA optimizations for memory efficiency
    auto model_opts = ModelOptions::from_runtime_options(options);
    if (lora_enabled()) {
        model_opts.lora_only_mode = true;
        model_opts.skip_base_gradients = true;
        // Pass train_router flag to base model so it allocates router gradient buffer
        model_opts.train_router = mLoRAConfig.train_router;
    }
    // Disable CUDA graphs for QLoRA FP4
    if (qlora_enabled() && mFP4WeightProvider) {
        model_opts.use_cuda_graphs = false;
    }
    // recompute_lora + offload_residuals: CUDA graphs must be disabled.
    if (model_opts.recompute_lora && model_opts.offload_residuals) {
        model_opts.use_cuda_graphs = false;
    }
    // MoE + LoRA: CUDA graphs must be disabled because the MoE forward pass
    // requires cudaStreamSynchronize to copy expert offsets to host, which is
    // not permitted during graph capture.
    // Check both architecture type and moe_config presence for robustness.
    const auto& base_cfg = mBaseModel->config();
    const bool is_moe = mIsMoEModel || base_cfg.moe_config.has_value() ||
                        base_cfg.architecture == ArchitectureType::MoE;
    if (is_moe && lora_enabled()) {
        model_opts.use_cuda_graphs = false;
    }
    // Use the ModelOptions overload to apply LoRA-specific flags
    mBaseModel->allocate_run_state(model_opts, comm, B, T, /*allocate_optimizer=*/false);

    allocate_lora_run_state(comm, B, T);

    // Allocate 8-bit AdamW optimizer state for LoRA
    if (lora_enabled() && !mLoRAAdamW8BitState) {
        mLoRAAdamW8BitState = std::make_unique<LoRAAdamW8BitState>();
        mLoRAAdamW8BitState->initialized = false;

        // Allocate quantization maps (256 entries each)
        mLoRAAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles1", {256});
        mLoRAAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles2", {256});

        // Initialize quantization maps on host then copy to device
        std::vector<float> h_quantiles1(256), h_quantiles2(256);
        create_adamw8bit_quantiles1(h_quantiles1.data());
        create_adamw8bit_quantiles2(h_quantiles2.data());
        CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles1.Data, h_quantiles1.data(),
                              256 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles2.Data, h_quantiles2.data(),
                              256 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

template<typename Block>
void ModularLoRAModel<Block>::init_weights(NCCLCommunicator& comm) {
    mBaseModel->init_weights(comm);
    if (lora_enabled()) {
        // LoRA weights are randomly initialized (A with kaiming uniform, B with zeros)
        // Router LoRA is also initialized this way for PEFT compatibility
        mLoRAWeights->random_init(42, comm);
    }
}

template<typename Block>
void ModularLoRAModel<Block>::allocate_lora_run_state(NCCLCommunicator& comm, int B, int T) {
    (void)comm;
    if (!lora_enabled()) return;

    mLoRARunState = std::make_unique<LoRARunState>();
    mLoRARunState->B = B;
    mLoRARunState->T = T;

    auto ctx = mAllocator->with_context("Modular_LoRA_RunState");

    const auto& cfg = mBaseModel->config();
    const int rank = mLoRAConfig.rank;
    const int BT = B * T;
    const int max_features = std::max((int)cfg.HiddenSize, (int)cfg.IntermediateSize);

    ETensorDType work_dtype = cfg.DType;

    mLoRARunState->intermediate = mAllocator->allocate(
        work_dtype, "lora_intermediate", EAllocationType::ON_DEVICE, {BT, rank});
    mLoRARunState->intermediate2 = mAllocator->allocate(
        work_dtype, "lora_intermediate2", EAllocationType::ON_DEVICE, {BT, rank});
    mLoRARunState->slice = mAllocator->allocate(
        work_dtype, "lora_slice", EAllocationType::ON_DEVICE, {BT, max_features});

    auto& rs = mBaseModel->run_state();
    const long num_block_sums = std::max<long>(2, static_cast<long>(get_max_num_block_sums(rs.DeviceProp)));

    // For MoE LoRA, use the activation dtype (BF16) instead of model dtype
    // to match the dtype of activation tensors like expert_gate_up
    ETensorDType moe_work_dtype = rs.config().activation_dtype;
    mLoRARunState->norm_buffer = mAllocator->allocate(
        ETensorDType::FP32, "lora_norm_buffer", EAllocationType::ON_DEVICE, {num_block_sums + 2});

    // Allocate recompute buffers only when recompute_lora is enabled
    if (rs.config().recompute_lora) {
        const int C = (int)cfg.HiddenSize;
        mLoRARunState->recompute_ln = mAllocator->allocate(
            work_dtype, "lora_recompute_ln", EAllocationType::ON_DEVICE, {B, T, C});
        mLoRARunState->recompute_rstd = mAllocator->allocate(
            ETensorDType::FP32, "lora_recompute_rstd", EAllocationType::ON_DEVICE, {B, T});
    }

    if (mIsMoEModel) {
        assert(cfg.moe_config.has_value());
        const auto& moe_cfg = *cfg.moe_config;
        const int top_k = moe_cfg.top_k;
        const int total_tokens = BT * top_k;
        const int expert_D = moe_cfg.moe_intermediate_size > 0 ? moe_cfg.moe_intermediate_size : (int)cfg.IntermediateSize;
        const int moe_M = (is_gated_activation(cfg.activation_type) ? 2 : 1) * expert_D;

        mLoRARunState->moe_lora_intermediate1 = mAllocator->allocate(
            moe_work_dtype, "moe_lora_intermediate1", EAllocationType::ON_DEVICE, {total_tokens, rank});
        mLoRARunState->moe_lora_intermediate2 = mAllocator->allocate(
            moe_work_dtype, "moe_lora_intermediate2", EAllocationType::ON_DEVICE, {total_tokens, expert_D});
        // moe_lora_gate and moe_lora_up are contiguous buffers for split gate_up values
        mLoRARunState->moe_lora_gate = mAllocator->allocate(
            moe_work_dtype, "moe_lora_gate", EAllocationType::ON_DEVICE, {total_tokens, expert_D});
        mLoRARunState->moe_lora_up = mAllocator->allocate(
            moe_work_dtype, "moe_lora_up", EAllocationType::ON_DEVICE, {total_tokens, expert_D});
        mLoRARunState->moe_lora_gate_up = mAllocator->allocate(
            moe_work_dtype, "moe_lora_gate_up", EAllocationType::ON_DEVICE, {total_tokens, moe_M});
    }
}

template<typename Block>
void ModularLoRAModel<Block>::ensure_lora_run_state(NCCLCommunicator& comm, int B, int T) {
    if (!lora_enabled()) return;
    if (!mLoRARunState || mLoRARunState->B != B || mLoRARunState->T != T) {
        allocate_lora_run_state(comm, B, T);
    }
}

template<typename Block>
Tensor ModularLoRAModel<Block>::recompute_rmsnorm(const Tensor& residual, const Tensor& ln_weight, float epsilon, int B, int T, int C, cudaStream_t stream) {
    rmsnorm_forward(mLoRARunState->recompute_ln, mLoRARunState->recompute_rstd,
                    residual, ln_weight, nullptr, epsilon, (long)B, (long)T, (long)C, stream);
    return mLoRARunState->recompute_ln;
}


} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_STATE_MANAGEMENT_H
