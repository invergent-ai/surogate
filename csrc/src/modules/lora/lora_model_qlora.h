// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_QLORA_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_QLORA_H

#include "lora_model_core.h"
#include "utilities/utils.h"

namespace modules {

template<typename Block>
void ModularLoRAModel<Block>::import_weights_qlora(const std::string& file_name, NCCLCommunicator& comm) {
    const auto& cfg = mBaseModel->config();
    auto fill_mamba_config = [&](auto& out_cfg) {
        out_cfg.layer_is_mamba.resize(cfg.NumLayers);
        out_cfg.has_mamba = false;
        for (int i = 0; i < cfg.NumLayers; ++i) {
            const bool is_mamba = (cfg.get_block_type(i) == BlockType::Mamba);
            out_cfg.layer_is_mamba[i] = static_cast<std::uint8_t>(is_mamba ? 1 : 0);
            out_cfg.has_mamba = out_cfg.has_mamba || is_mamba;
        }
        out_cfg.mamba_num_heads = cfg.MambaNumHeads;
        out_cfg.mamba_head_dim = cfg.MambaHeadDim;
        out_cfg.mamba_ssm_state_size = cfg.MambaSsmStateSize;
        out_cfg.mamba_conv_kernel = cfg.MambaConvKernel;
        out_cfg.mamba_n_groups = cfg.MambaNGroups;
        out_cfg.mamba_intermediate_size = cfg.MambaIntermediateSize;
        out_cfg.mamba_use_bias = cfg.MambaUseBias;
        out_cfg.mamba_use_conv_bias = cfg.MambaUseConvBias;
    };
    // Sync QLoRA MoE settings from base config if caller didn't provide them.
    if (cfg.moe_config.has_value()) {
        const auto& moe = cfg.moe_config.value();
        if (mQLoRAConfig.num_experts == 0) {
            mQLoRAConfig.num_experts = moe.num_experts;
        }
        if (mQLoRAConfig.num_experts_per_tok == 0) {
            mQLoRAConfig.num_experts_per_tok = moe.top_k;
        }
        if (mQLoRAConfig.moe_intermediate_size == 0) {
            mQLoRAConfig.moe_intermediate_size = moe.moe_intermediate_size;
        }
        if (mQLoRAConfig.num_shared_experts == 0 && moe.use_shared_expert) {
            mQLoRAConfig.num_shared_experts = 1;
        }
        if (mQLoRAConfig.moe_shared_expert_intermediate_size == 0 && moe.use_shared_expert) {
            mQLoRAConfig.moe_shared_expert_intermediate_size = moe.shared_expert_size;
        }
    }
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_props{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));

    cudaStream_t quant_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&quant_stream));

    if (mQLoRAConfig.is_fp8()) {
        typename FP8WeightProvider<Block>::Config qwp_config{};
        qwp_config.num_layers = cfg.NumLayers;
        qwp_config.hidden_size = cfg.HiddenSize;
        qwp_config.intermediate_size = cfg.IntermediateSize;
        qwp_config.num_query_heads = cfg.NumQueryHeads;
        qwp_config.num_kv_heads = cfg.NumKeyValHeads;
        qwp_config.head_size = cfg.head_size();
        qwp_config.vocab_size = cfg.VocabSize;
        qwp_config.mlp_up_factor = cfg.mlp_up_factor();
        qwp_config.qlora_config = mQLoRAConfig;
        qwp_config.lora_config = mLoRAConfig;
        qwp_config.model_dtype = cfg.DType;
        qwp_config.use_qk_norm = cfg.UseQKNorm;
        qwp_config.tied_embeddings = cfg.TiedWordEmbeddings;
        qwp_config.shard_idx = comm.rank();
        qwp_config.num_shards = comm.world_size();
        qwp_config.enable_fp8_forward = mOptions.fp8_forward_enabled();
        qwp_config.enable_fp8_hybrid = mOptions.fp8_hybrid_enabled();
        fill_mamba_config(qwp_config);

        mFP8WeightProvider = std::make_unique<FP8WeightProvider<Block>>(qwp_config, *mAllocator, device_props);
        mFP8WeightProvider->import_and_quantize(file_name, comm, quant_stream);

        auto& weights_manager = mBaseModel->weights_manager();
        weights_manager.set_weight_provider([this](int layer_idx, cudaStream_t stream) -> typename Block::Weights& {
            return mFP8WeightProvider->get_block(layer_idx, stream);
        });
        weights_manager.set_embeddings_provider([this](cudaStream_t stream) -> Tensor& { return mFP8WeightProvider->get_embeddings(stream); });
        weights_manager.set_final_norm_provider([this](cudaStream_t stream) -> Tensor& { return mFP8WeightProvider->get_final_norm(stream); });
        weights_manager.set_lm_head_provider([this](cudaStream_t stream) -> Tensor& { return mFP8WeightProvider->get_lm_head(stream); });

        if (mFP8WeightProvider->has_fp8_forward_cache()) {
            weights_manager.set_fp8_cache_provider([this]() -> typename ModularWeightManager<Block>::FP8WeightCache& {
                return mFP8WeightProvider->get_fp8_cache();
            });
        }
    } else if (mQLoRAConfig.is_fp4()) {
        import_weights_fp4_qlora(file_name, comm);
    } else if (mQLoRAConfig.is_bnb()) {
        import_weights_bnb_qlora(file_name, comm);
    }

    CUDA_CHECK(cudaStreamSynchronize(quant_stream));
    CUDA_CHECK(cudaStreamDestroy(quant_stream));
}

template<typename Block>
void ModularLoRAModel<Block>::import_weights_fp4_qlora(const std::string& file_name, NCCLCommunicator& comm) {
    const auto& cfg = mBaseModel->config();
    auto fill_mamba_config = [&](auto& out_cfg) {
        out_cfg.layer_is_mamba.resize(cfg.NumLayers);
        out_cfg.has_mamba = false;
        for (int i = 0; i < cfg.NumLayers; ++i) {
            const bool is_mamba = (cfg.get_block_type(i) == BlockType::Mamba);
            out_cfg.layer_is_mamba[i] = static_cast<std::uint8_t>(is_mamba ? 1 : 0);
            out_cfg.has_mamba = out_cfg.has_mamba || is_mamba;
        }
        out_cfg.mamba_num_heads = cfg.MambaNumHeads;
        out_cfg.mamba_head_dim = cfg.MambaHeadDim;
        out_cfg.mamba_ssm_state_size = cfg.MambaSsmStateSize;
        out_cfg.mamba_conv_kernel = cfg.MambaConvKernel;
        out_cfg.mamba_n_groups = cfg.MambaNGroups;
        out_cfg.mamba_intermediate_size = cfg.MambaIntermediateSize;
        out_cfg.mamba_use_bias = cfg.MambaUseBias;
        out_cfg.mamba_use_conv_bias = cfg.MambaUseConvBias;
    };
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_props{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));

    cudaStream_t quant_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&quant_stream));

    typename FP4WeightProvider<Block>::Config fp4_config{};
    fp4_config.num_layers = cfg.NumLayers;
    fp4_config.hidden_size = cfg.HiddenSize;
    fp4_config.intermediate_size = cfg.IntermediateSize;
    fp4_config.num_query_heads = cfg.NumQueryHeads;
    fp4_config.num_kv_heads = cfg.NumKeyValHeads;
    fp4_config.head_size = cfg.head_size();
    fp4_config.vocab_size = cfg.VocabSize;
    fp4_config.mlp_up_factor = cfg.mlp_up_factor();
    fp4_config.qlora_config = mQLoRAConfig;
    fp4_config.lora_config = mLoRAConfig;
    fp4_config.model_dtype = cfg.DType;
    fp4_config.use_qk_norm = cfg.UseQKNorm;
    fp4_config.tied_embeddings = cfg.TiedWordEmbeddings;
    fp4_config.shard_idx = comm.rank();
    fp4_config.num_shards = comm.world_size();
    fp4_config.selective_expert_dequant = mOptions.SelectiveExpertDequant;
    fp4_config.offload_experts = mOptions.OffloadExperts;
    fill_mamba_config(fp4_config);

    mFP4WeightProvider = std::make_unique<FP4WeightProvider<Block>>(fp4_config, *mAllocator, device_props);
    mFP4WeightProvider->import_and_quantize(file_name, comm, quant_stream);

    auto& weights_manager = mBaseModel->weights_manager();
    weights_manager.set_weight_provider([this](int layer_idx, cudaStream_t stream) -> typename Block::Weights& {
        return mFP4WeightProvider->get_block(layer_idx, stream);
    });
    weights_manager.set_embeddings_provider([this](cudaStream_t stream) -> Tensor& { return mFP4WeightProvider->get_embeddings(stream); });
    weights_manager.set_final_norm_provider([this](cudaStream_t stream) -> Tensor& { return mFP4WeightProvider->get_final_norm(stream); });
    weights_manager.set_lm_head_provider([this](cudaStream_t stream) -> Tensor& { return mFP4WeightProvider->get_lm_head(stream); });

    CUDA_CHECK(cudaStreamSynchronize(quant_stream));
    CUDA_CHECK(cudaStreamDestroy(quant_stream));
}

template<typename Block>
void ModularLoRAModel<Block>::import_weights_bnb_qlora(const std::string& file_name, NCCLCommunicator& comm) {
    const auto& cfg = mBaseModel->config();
    auto fill_mamba_config = [&](auto& out_cfg) {
        out_cfg.layer_is_mamba.resize(cfg.NumLayers);
        out_cfg.has_mamba = false;
        for (int i = 0; i < cfg.NumLayers; ++i) {
            const bool is_mamba = (cfg.get_block_type(i) == BlockType::Mamba);
            out_cfg.layer_is_mamba[i] = static_cast<std::uint8_t>(is_mamba ? 1 : 0);
            out_cfg.has_mamba = out_cfg.has_mamba || is_mamba;
        }
        out_cfg.mamba_num_heads = cfg.MambaNumHeads;
        out_cfg.mamba_head_dim = cfg.MambaHeadDim;
        out_cfg.mamba_ssm_state_size = cfg.MambaSsmStateSize;
        out_cfg.mamba_conv_kernel = cfg.MambaConvKernel;
        out_cfg.mamba_n_groups = cfg.MambaNGroups;
        out_cfg.mamba_intermediate_size = cfg.MambaIntermediateSize;
        out_cfg.mamba_use_bias = cfg.MambaUseBias;
        out_cfg.mamba_use_conv_bias = cfg.MambaUseConvBias;
    };
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_props{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));

    cudaStream_t quant_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&quant_stream));

    typename BnBWeightProvider<Block>::Config bnb_config{};
    bnb_config.num_layers = cfg.NumLayers;
    bnb_config.hidden_size = cfg.HiddenSize;
    bnb_config.intermediate_size = cfg.IntermediateSize;
    bnb_config.num_query_heads = cfg.NumQueryHeads;
    bnb_config.num_kv_heads = cfg.NumKeyValHeads;
    bnb_config.head_size = cfg.head_size();
    bnb_config.vocab_size = cfg.VocabSize;
    bnb_config.mlp_up_factor = cfg.mlp_up_factor();
    bnb_config.qlora_config = mQLoRAConfig;
    bnb_config.lora_config = mLoRAConfig;
    bnb_config.model_dtype = cfg.DType;
    bnb_config.use_qk_norm = cfg.UseQKNorm;
    bnb_config.tied_embeddings = cfg.TiedWordEmbeddings;
    bnb_config.shard_idx = comm.rank();
    bnb_config.num_shards = comm.world_size();
    bnb_config.selective_expert_dequant = mOptions.SelectiveExpertDequant;
    bnb_config.offload_experts = mOptions.OffloadExperts;
    fill_mamba_config(bnb_config);

    mBnBWeightProvider = std::make_unique<BnBWeightProvider<Block>>(bnb_config, *mAllocator, device_props);
    mBnBWeightProvider->import_and_quantize(file_name, comm, quant_stream);

    auto& weights_manager = mBaseModel->weights_manager();
    weights_manager.set_weight_provider([this](int layer_idx, cudaStream_t stream) -> typename Block::Weights& {
        return mBnBWeightProvider->get_block(layer_idx, stream);
    });
    weights_manager.set_embeddings_provider([this](cudaStream_t stream) -> Tensor& { return mBnBWeightProvider->get_embeddings(stream); });
    weights_manager.set_final_norm_provider([this](cudaStream_t stream) -> Tensor& { return mBnBWeightProvider->get_final_norm(stream); });
    weights_manager.set_lm_head_provider([this](cudaStream_t stream) -> Tensor& { return mBnBWeightProvider->get_lm_head(stream); });

    CUDA_CHECK(cudaStreamSynchronize(quant_stream));
    CUDA_CHECK(cudaStreamDestroy(quant_stream));
}

template<typename Block>
std::size_t ModularLoRAModel<Block>::qlora_quantized_weights_bytes() const {
    if (mFP4WeightProvider) return mFP4WeightProvider->quantized_weights_bytes();
    if (mFP8WeightProvider) return mFP8WeightProvider->quantized_weights_bytes();
    if (mBnBWeightProvider) return mBnBWeightProvider->quantized_weights_bytes();
    return 0;
}

template<typename Block>
float ModularLoRAModel<Block>::qlora_memory_savings_ratio() const {
    if (mFP4WeightProvider) return mFP4WeightProvider->memory_savings_ratio();
    if (mFP8WeightProvider) return mFP8WeightProvider->memory_savings_ratio();
    if (mBnBWeightProvider) return mBnBWeightProvider->memory_savings_ratio();
    return 1.0f;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_QLORA_H
