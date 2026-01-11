// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_weights_manager.h"

#include <cmath>
#include <fmt/format.h>

#include "kernels/kernels.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"

namespace modules {

ModularLoRAWeightsManager::ModularLoRAWeightsManager(const Config& config, TensorAllocator& allocator)
    : mConfig(config), mAllocator(&allocator) {
    mMaster.config = config.lora_config;
    mWork.config = config.lora_config;

    if (!enabled()) {
        return;
    }

    auto ctx = mAllocator->with_context("Modular_LoRA_Weights");
    mMaster.blocks.resize(config.num_layers);
    mWork.blocks.resize(config.num_layers);
    for (int l = 0; l < config.num_layers; ++l) {
        allocate_block_weights(l);
    }
}

void ModularLoRAWeightsManager::allocate_layer_weights(
    LoRALayerWeights<TensorShard>& shard,
    LoRALayerWeights<Tensor>& work,
    int in_features,
    int out_features,
    const std::string& name) {

    const int r = mConfig.lora_config.rank;
    const ETensorDType master_dtype = mConfig.lora_config.dtype;
    const ETensorDType work_dtype = mConfig.work_dtype;

    // Data-parallel LoRA: replicate weights on all ranks (no sharding yet).
    shard.A = TensorShard(mAllocator->allocate(master_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {r, in_features}));
    shard.B = mAllocator->allocate_shard(master_dtype, /*shard_idx=*/0, /*num_shards=*/1, (name + "_B").c_str(), {out_features, r});

    work.A = mAllocator->allocate(work_dtype, (name + "_A_work").c_str(), EAllocationType::ON_DEVICE, {r, in_features});
    work.B = mAllocator->allocate(work_dtype, (name + "_B_work").c_str(), EAllocationType::ON_DEVICE, {out_features, r});
}

void ModularLoRAWeightsManager::allocate_block_weights(int layer_idx) {
    if (!enabled()) return;

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int Hq = mConfig.num_query_heads;
    const int Hkv = mConfig.num_kv_heads;
    const int Hs = mConfig.head_size;
    const int q_out = Hq * Hs;
    const int kv_out = Hkv * Hs;

    auto& master = mMaster.blocks[layer_idx];
    auto& work = mWork.blocks[layer_idx];

    const std::string prefix = fmt::format("lora_layer_{}", layer_idx);

    if (mConfig.lora_config.applies_to_q()) {
        master.attention.q.emplace();
        work.attention.q.emplace();
        allocate_layer_weights(*master.attention.q, *work.attention.q, /*in=*/C, /*out=*/q_out, prefix + "_q");
    }
    if (mConfig.lora_config.applies_to_k()) {
        master.attention.k.emplace();
        work.attention.k.emplace();
        allocate_layer_weights(*master.attention.k, *work.attention.k, /*in=*/C, /*out=*/kv_out, prefix + "_k");
    }
    if (mConfig.lora_config.applies_to_v()) {
        master.attention.v.emplace();
        work.attention.v.emplace();
        allocate_layer_weights(*master.attention.v, *work.attention.v, /*in=*/C, /*out=*/kv_out, prefix + "_v");
    }
    if (mConfig.lora_config.applies_to_o()) {
        master.attention.o.emplace();
        work.attention.o.emplace();
        allocate_layer_weights(*master.attention.o, *work.attention.o, /*in=*/q_out, /*out=*/C, prefix + "_o");
    }

    // MLP LoRA: For dense models, use standard MLP LoRA. For MoE models, use per-expert LoRA.
    if (mConfig.is_moe && mConfig.num_experts > 0) {
        // Allocate per-expert LoRA weights for MoE models
        const bool has_mlp_lora = mConfig.lora_config.applies_to_gate() ||
                                   mConfig.lora_config.applies_to_up() ||
                                   mConfig.lora_config.applies_to_down();
        if (has_mlp_lora) {
            master.moe.experts.resize(mConfig.num_experts);
            work.moe.experts.resize(mConfig.num_experts);
            for (int e = 0; e < mConfig.num_experts; ++e) {
                allocate_expert_weights(master.moe.experts[e], work.moe.experts[e], layer_idx, e);
            }
        }
    } else {
        // Dense model: standard MLP LoRA
        if (mConfig.lora_config.applies_to_gate()) {
            master.mlp.gate.emplace();
            work.mlp.gate.emplace();
            allocate_layer_weights(*master.mlp.gate, *work.mlp.gate, /*in=*/C, /*out=*/D, prefix + "_gate");
        }
        if (mConfig.lora_config.applies_to_up()) {
            master.mlp.up.emplace();
            work.mlp.up.emplace();
            allocate_layer_weights(*master.mlp.up, *work.mlp.up, /*in=*/C, /*out=*/D, prefix + "_up");
        }
        if (mConfig.lora_config.applies_to_down()) {
            master.mlp.down.emplace();
            work.mlp.down.emplace();
            allocate_layer_weights(*master.mlp.down, *work.mlp.down, /*in=*/D, /*out=*/C, prefix + "_down");
        }
    }
}

void ModularLoRAWeightsManager::allocate_expert_weights(
    LoRAExpertWeights<TensorShard>& master_expert,
    LoRAExpertWeights<Tensor>& work_expert,
    int layer_idx, int expert_idx) {

    const int C = mConfig.hidden_size;
    const int D = mConfig.effective_moe_intermediate();
    const std::string prefix = fmt::format("lora_layer_{}_expert_{}", layer_idx, expert_idx);

    if (mConfig.lora_config.applies_to_gate()) {
        master_expert.gate.emplace();
        work_expert.gate.emplace();
        allocate_layer_weights(*master_expert.gate, *work_expert.gate, /*in=*/C, /*out=*/D, prefix + "_gate");
    }
    if (mConfig.lora_config.applies_to_up()) {
        master_expert.up.emplace();
        work_expert.up.emplace();
        allocate_layer_weights(*master_expert.up, *work_expert.up, /*in=*/C, /*out=*/D, prefix + "_up");
    }
    if (mConfig.lora_config.applies_to_down()) {
        master_expert.down.emplace();
        work_expert.down.emplace();
        allocate_layer_weights(*master_expert.down, *work_expert.down, /*in=*/D, /*out=*/C, prefix + "_down");
    }
}

void ModularLoRAWeightsManager::random_init(int seed, NCCLCommunicator& comm) {
    if (!enabled()) return;

    auto init_layer = [&](std::optional<LoRALayerWeights<TensorShard>>& layer,
                          int in_features,
                          unsigned long long subsequence) {
        if (!layer.has_value()) return;
        // std consistent with kaiming_uniform_(a=sqrt(5)) => bound = 1/sqrt(fan_in)
        float std_a = 1.0f / std::sqrt(3.0f * static_cast<float>(in_features));
        fill_normal(layer->A, layer->A.nelem(), 0.0f, std_a, seed, subsequence, nullptr);
        fill_zero(layer->B, nullptr);
    };

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int D_moe = mConfig.effective_moe_intermediate();
    const int q_out = mConfig.num_query_heads * mConfig.head_size;
    const int E = mConfig.num_experts;

    for (int l = 0; l < mConfig.num_layers; ++l) {
        auto& b = mMaster.blocks[l];
        unsigned long long base = static_cast<unsigned long long>(l) * 32ULL;
        init_layer(b.attention.q, C, base + 0);
        init_layer(b.attention.k, C, base + 1);
        init_layer(b.attention.v, C, base + 2);
        init_layer(b.attention.o, q_out, base + 3);

        // Dense MLP LoRA
        init_layer(b.mlp.gate, C, base + 4);
        init_layer(b.mlp.up, C, base + 5);
        init_layer(b.mlp.down, D, base + 6);

        // MoE expert LoRA
        for (int e = 0; e < (int)b.moe.experts.size(); ++e) {
            auto& expert = b.moe.experts[e];
            // Use separate subsequence space for each expert to avoid correlation
            unsigned long long expert_base = base + 8ULL + static_cast<unsigned long long>(e) * 4ULL;
            init_layer(expert.gate, C, expert_base + 0);
            init_layer(expert.up, C, expert_base + 1);
            init_layer(expert.down, D_moe, expert_base + 2);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();
}

void ModularLoRAWeightsManager::import_from_file(const std::string& file_name, NCCLCommunicator& comm) {
    if (!enabled()) return;
    load_safetensors(file_name, *this, /*allow_cast=*/true);
    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();
}

void ModularLoRAWeightsManager::export_to_file(const std::string& file_name, NCCLCommunicator& comm) const {
    if (!enabled()) return;
    if (comm.rank() == 0) {
        write_safetensors(file_name, const_cast<ModularLoRAWeightsManager&>(*this));
    }
    comm.barrier();
}

LoRABlockWeights<Tensor>& ModularLoRAWeightsManager::get_block(int layer_idx, cudaStream_t stream) {
    auto& work = mWork.blocks[layer_idx];
    if (!enabled()) return work;

    auto& master = mMaster.blocks[layer_idx];

    auto sync_tensor = [&](Tensor& dst_t, const TensorShard& src_t, const char* name) {
        if (!dst_t.Data || !src_t.Data) return;
        if (dst_t.nelem() != src_t.nelem()) {
            throw std::logic_error(fmt::format("ModularLoRAWeightsManager::get_block: {} nelem mismatch (dst={}, src={})",
                                               name, dst_t.nelem(), src_t.nelem()));
        }

        if (dst_t.DType == src_t.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst_t.Data, src_t.Data, dst_t.bytes(), cudaMemcpyDeviceToDevice, stream));
            return;
        }

        if (dst_t.DType == ETensorDType::BF16 && src_t.DType == ETensorDType::FP32) {
            convert_dtype(dst_t.get<nv_bfloat16>(), src_t.get<float>(), dst_t.nelem(), stream);
            return;
        }
        if (dst_t.DType == ETensorDType::FP32 && src_t.DType == ETensorDType::BF16) {
            convert_dtype(dst_t.get<float>(), src_t.get<nv_bfloat16>(), dst_t.nelem(), stream);
            return;
        }

        throw std::logic_error(fmt::format(
            "ModularLoRAWeightsManager::get_block: unsupported dtype cast for {} (src={}, dst={})",
            name, dtype_to_str(src_t.DType), dtype_to_str(dst_t.DType)));
    };

    auto sync_layer = [&](std::optional<LoRALayerWeights<Tensor>>& dst,
                          const std::optional<LoRALayerWeights<TensorShard>>& src,
                          const char* layer_name) {
        if (!dst.has_value() || !src.has_value()) return;
        sync_tensor(dst->A, src->A, (std::string(layer_name) + ".A").c_str());
        sync_tensor(dst->B, src->B, (std::string(layer_name) + ".B").c_str());
    };

    sync_layer(work.attention.q, master.attention.q, "q_proj");
    sync_layer(work.attention.k, master.attention.k, "k_proj");
    sync_layer(work.attention.v, master.attention.v, "v_proj");
    sync_layer(work.attention.o, master.attention.o, "o_proj");

    // Dense MLP LoRA
    sync_layer(work.mlp.gate, master.mlp.gate, "gate_proj");
    sync_layer(work.mlp.up, master.mlp.up, "up_proj");
    sync_layer(work.mlp.down, master.mlp.down, "down_proj");

    // MoE expert LoRA
    for (int e = 0; e < (int)master.moe.experts.size(); ++e) {
        auto& master_expert = master.moe.experts[e];
        auto& work_expert = work.moe.experts[e];
        std::string expert_prefix = fmt::format("expert_{}", e);
        sync_layer(work_expert.gate, master_expert.gate, (expert_prefix + "_gate").c_str());
        sync_layer(work_expert.up, master_expert.up, (expert_prefix + "_up").c_str());
        sync_layer(work_expert.down, master_expert.down, (expert_prefix + "_down").c_str());
    }

    return work;
}

LoRABlockWeights<TensorShard>& ModularLoRAWeightsManager::get_master_block(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mMaster.blocks[layer_idx];
}

std::size_t ModularLoRAWeightsManager::num_parameters() const {
    if (!enabled()) return 0;

    const std::size_t r = static_cast<std::size_t>(mConfig.lora_config.rank);
    const std::size_t C = static_cast<std::size_t>(mConfig.hidden_size);
    const std::size_t D = static_cast<std::size_t>(mConfig.intermediate_size);
    const std::size_t D_moe = static_cast<std::size_t>(mConfig.effective_moe_intermediate());
    const std::size_t Hq = static_cast<std::size_t>(mConfig.num_query_heads);
    const std::size_t Hkv = static_cast<std::size_t>(mConfig.num_kv_heads);
    const std::size_t Hs = static_cast<std::size_t>(mConfig.head_size);
    const std::size_t q_out = Hq * Hs;
    const std::size_t kv_out = Hkv * Hs;
    const std::size_t E = static_cast<std::size_t>(mConfig.num_experts);

    std::size_t per_layer = 0;

    // Attention LoRA parameters
    if (mConfig.lora_config.applies_to_q()) per_layer += r * C + q_out * r;
    if (mConfig.lora_config.applies_to_k()) per_layer += r * C + kv_out * r;
    if (mConfig.lora_config.applies_to_v()) per_layer += r * C + kv_out * r;
    if (mConfig.lora_config.applies_to_o()) per_layer += r * q_out + C * r;

    // MLP LoRA parameters (dense or MoE)
    if (mConfig.is_moe && E > 0) {
        // Per-expert LoRA for MoE models
        std::size_t per_expert = 0;
        if (mConfig.lora_config.applies_to_gate()) per_expert += r * C + D_moe * r;
        if (mConfig.lora_config.applies_to_up()) per_expert += r * C + D_moe * r;
        if (mConfig.lora_config.applies_to_down()) per_expert += r * D_moe + C * r;
        per_layer += per_expert * E;
    } else {
        // Dense MLP LoRA
        if (mConfig.lora_config.applies_to_gate()) per_layer += r * C + D * r;
        if (mConfig.lora_config.applies_to_up()) per_layer += r * C + D * r;
        if (mConfig.lora_config.applies_to_down()) per_layer += r * D + C * r;
    }

    return per_layer * static_cast<std::size_t>(mConfig.num_layers);
}

void ModularLoRAWeightsManager::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {
    if (!enabled()) return;

    for (int l = 0; l < (int)mMaster.blocks.size(); ++l) {
        std::string prefix = fmt::format("base_model.model.model.layers.{}", l);
        auto& block = mMaster.blocks[l];

        if (block.attention.q.has_value()) {
            callback(prefix + ".self_attn.q_proj.lora_A.weight", block.attention.q->A);
            callback(prefix + ".self_attn.q_proj.lora_B.weight", block.attention.q->B);
        }
        if (block.attention.k.has_value()) {
            callback(prefix + ".self_attn.k_proj.lora_A.weight", block.attention.k->A);
            callback(prefix + ".self_attn.k_proj.lora_B.weight", block.attention.k->B);
        }
        if (block.attention.v.has_value()) {
            callback(prefix + ".self_attn.v_proj.lora_A.weight", block.attention.v->A);
            callback(prefix + ".self_attn.v_proj.lora_B.weight", block.attention.v->B);
        }
        if (block.attention.o.has_value()) {
            callback(prefix + ".self_attn.o_proj.lora_A.weight", block.attention.o->A);
            callback(prefix + ".self_attn.o_proj.lora_B.weight", block.attention.o->B);
        }

        // Dense MLP LoRA
        if (block.mlp.gate.has_value()) {
            callback(prefix + ".mlp.gate_proj.lora_A.weight", block.mlp.gate->A);
            callback(prefix + ".mlp.gate_proj.lora_B.weight", block.mlp.gate->B);
        }
        if (block.mlp.up.has_value()) {
            callback(prefix + ".mlp.up_proj.lora_A.weight", block.mlp.up->A);
            callback(prefix + ".mlp.up_proj.lora_B.weight", block.mlp.up->B);
        }
        if (block.mlp.down.has_value()) {
            callback(prefix + ".mlp.down_proj.lora_A.weight", block.mlp.down->A);
            callback(prefix + ".mlp.down_proj.lora_B.weight", block.mlp.down->B);
        }

        // MoE expert LoRA (HuggingFace naming convention: .mlp.experts.{e}.{proj})
        for (int e = 0; e < (int)block.moe.experts.size(); ++e) {
            auto& expert = block.moe.experts[e];
            std::string expert_prefix = fmt::format("{}.mlp.experts.{}", prefix, e);

            if (expert.gate.has_value()) {
                callback(expert_prefix + ".gate_proj.lora_A.weight", expert.gate->A);
                callback(expert_prefix + ".gate_proj.lora_B.weight", expert.gate->B);
            }
            if (expert.up.has_value()) {
                callback(expert_prefix + ".up_proj.lora_A.weight", expert.up->A);
                callback(expert_prefix + ".up_proj.lora_B.weight", expert.up->B);
            }
            if (expert.down.has_value()) {
                callback(expert_prefix + ".down_proj.lora_A.weight", expert.down->A);
                callback(expert_prefix + ".down_proj.lora_B.weight", expert.down->B);
            }
        }
    }
}

} // namespace modules
