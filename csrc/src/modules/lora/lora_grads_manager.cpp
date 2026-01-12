// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_grads_manager.h"

#include <fmt/format.h>

#include "kernels/kernels.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"

namespace modules {

ModularLoRAGradsManager::ModularLoRAGradsManager(const Config& config, const std::shared_ptr<TensorAllocator>& allocator)
    : mConfig(config), mAllocator(allocator) {
    mFullGrads.config = config.lora_config;
    mShardedGrads.config = config.lora_config;

    if (!config.lora_config.enabled()) return;
    allocate_gradients();
}

ModularLoRAGradsManager::~ModularLoRAGradsManager() = default;

void ModularLoRAGradsManager::allocate_gradients() {
    auto ctx = mAllocator->with_context("Modular_LoRA_Grads");
    mFullGrads.blocks.resize(mConfig.num_layers);
    mShardedGrads.blocks.resize(mConfig.num_layers);

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int D_moe = mConfig.effective_moe_intermediate();
    const int q_out = mConfig.num_query_heads * mConfig.head_size;
    const int kv_out = mConfig.num_kv_heads * mConfig.head_size;
    const int r = mConfig.lora_config.rank;
    const int E = mConfig.num_experts;

    auto alloc_full = [&](int in_f, int out_f, const std::string& name) -> LoRALayerWeights<Tensor> {
        LoRALayerWeights<Tensor> w;
        w.A = mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {r, in_f});
        w.B = mAllocator->allocate(mConfig.grad_dtype, (name + "_B").c_str(), EAllocationType::ON_DEVICE, {out_f, r});
        return w;
    };
    auto alloc_shard = [&](int in_f, int out_f, const std::string& name) -> LoRALayerWeights<TensorShard> {
        LoRALayerWeights<TensorShard> w;
        w.A = TensorShard(mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {r, in_f}));
        w.B = mAllocator->allocate_shard(mConfig.grad_dtype, /*shard_idx=*/0, /*num_shards=*/1, (name + "_B").c_str(), {out_f, r});
        return w;
    };

    auto alloc_grouped_full = [&](int in_f, int out_f, const std::string& name) -> LoRAGroupedLayerWeights<Tensor> {
        LoRAGroupedLayerWeights<Tensor> w;
        w.A = mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {E, r, in_f});
        w.B = mAllocator->allocate(mConfig.grad_dtype, (name + "_B").c_str(), EAllocationType::ON_DEVICE, {E, out_f, r});
        return w;
    };
    auto alloc_grouped_shard = [&](int in_f, int out_f, const std::string& name) -> LoRAGroupedLayerWeights<TensorShard> {
        LoRAGroupedLayerWeights<TensorShard> w;
        w.A = TensorShard(mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {E, r, in_f}));
        w.B = TensorShard(mAllocator->allocate(mConfig.grad_dtype, (name + "_B").c_str(), EAllocationType::ON_DEVICE, {E, out_f, r}));
        return w;
    };

    for (int l = 0; l < mConfig.num_layers; ++l) {
        std::string prefix = fmt::format("lora_grad_layer_{}", l);
        auto& full = mFullGrads.blocks[l];
        auto& shard = mShardedGrads.blocks[l];

        if (mConfig.lora_config.applies_to_q()) {
            full.attention.q = alloc_full(C, q_out, prefix + "_q");
            shard.attention.q = alloc_shard(C, q_out, prefix + "_q_shard");
        }
        if (mConfig.lora_config.applies_to_k()) {
            full.attention.k = alloc_full(C, kv_out, prefix + "_k");
            shard.attention.k = alloc_shard(C, kv_out, prefix + "_k_shard");
        }
        if (mConfig.lora_config.applies_to_v()) {
            full.attention.v = alloc_full(C, kv_out, prefix + "_v");
            shard.attention.v = alloc_shard(C, kv_out, prefix + "_v_shard");
        }
        if (mConfig.lora_config.applies_to_o()) {
            full.attention.o = alloc_full(q_out, C, prefix + "_o");
            shard.attention.o = alloc_shard(q_out, C, prefix + "_o_shard");
        }

        // MLP LoRA gradients
        if (mConfig.is_moe && E > 0) {
            const bool has_mlp_lora = mConfig.lora_config.applies_to_gate() ||
                                       mConfig.lora_config.applies_to_up() ||
                                       mConfig.lora_config.applies_to_down();
            if (has_mlp_lora) {
                full.moe.use_grouped = true;
                shard.moe.use_grouped = true;

                std::string exp_prefix = prefix + "_moe_grouped";
                if (mConfig.lora_config.applies_to_gate()) {
                    full.moe.grouped.gate = alloc_grouped_full(C, D_moe, exp_prefix + "_gate");
                    shard.moe.grouped.gate = alloc_grouped_shard(C, D_moe, exp_prefix + "_gate_shard");
                }
                if (mConfig.lora_config.applies_to_up()) {
                    full.moe.grouped.up = alloc_grouped_full(C, D_moe, exp_prefix + "_up");
                    shard.moe.grouped.up = alloc_grouped_shard(C, D_moe, exp_prefix + "_up_shard");
                }
                if (mConfig.lora_config.applies_to_down()) {
                    full.moe.grouped.down = alloc_grouped_full(D_moe, C, exp_prefix + "_down");
                    shard.moe.grouped.down = alloc_grouped_shard(D_moe, C, exp_prefix + "_down_shard");
                }
            }
        } else {
            // Dense MLP LoRA gradients
            if (mConfig.lora_config.applies_to_gate()) {
                full.mlp.gate = alloc_full(C, D, prefix + "_gate");
                shard.mlp.gate = alloc_shard(C, D, prefix + "_gate_shard");
            }
            if (mConfig.lora_config.applies_to_up()) {
                full.mlp.up = alloc_full(C, D, prefix + "_up");
                shard.mlp.up = alloc_shard(C, D, prefix + "_up_shard");
            }
            if (mConfig.lora_config.applies_to_down()) {
                full.mlp.down = alloc_full(D, C, prefix + "_down");
                shard.mlp.down = alloc_shard(D, C, prefix + "_down_shard");
            }
        }
    }
}

void ModularLoRAGradsManager::start_micro_step(cudaStream_t stream, int micro_step, int total_steps) {
    mIsFirstMicroStep = (micro_step == 0);
    mIsLastMicroStep = (micro_step == total_steps - 1);

    if (!mConfig.lora_config.enabled()) return;

    if (mIsFirstMicroStep) {
        for (auto& block : mFullGrads.blocks) {
            auto zero_layer = [stream](auto& opt_layer) {
                if (!opt_layer.has_value()) return;
                if (opt_layer->A.Data) fill_zero(opt_layer->A, stream);
                if (opt_layer->B.Data) fill_zero(opt_layer->B, stream);
            };
            zero_layer(block.attention.q);
            zero_layer(block.attention.k);
            zero_layer(block.attention.v);
            zero_layer(block.attention.o);
            zero_layer(block.mlp.gate);
            zero_layer(block.mlp.up);
            zero_layer(block.mlp.down);

            if (block.moe.use_grouped) {
                zero_layer(block.moe.grouped.gate);
                zero_layer(block.moe.grouped.up);
                zero_layer(block.moe.grouped.down);
            } else {
                // MoE expert LoRA gradients
                for (auto& expert : block.moe.experts) {
                    zero_layer(expert.gate);
                    zero_layer(expert.up);
                    zero_layer(expert.down);
                }
            }
        }
    }
}

void ModularLoRAGradsManager::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    if (!mConfig.lora_config.enabled()) return;
    if (mIsLastMicroStep) {
        reduce_gradients(stream, comm);
    }
}

LoRABlockWeights<Tensor>& ModularLoRAGradsManager::get_block_full(
    int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    (void)stream;
    (void)comm;
    accumulate = !mIsFirstMicroStep;
    return mFullGrads.blocks[layer_idx];
}

LoRABlockWeights<TensorShard>& ModularLoRAGradsManager::get_block_shard(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mShardedGrads.blocks[layer_idx];
}

void ModularLoRAGradsManager::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    (void)layer_idx;
    (void)stream;
    (void)comm;
    // No-op for now (reduction batched in end_micro_step).
}

void ModularLoRAGradsManager::reduce_gradients(cudaStream_t stream, NCCLCommunicator& comm) {
    if (comm.world_size() == 1) return;

    auto all_reduce_layer = [&](std::optional<LoRALayerWeights<Tensor>>& layer) {
        if (!layer.has_value()) return;
        if (layer->A.Data) comm.all_reduce_avg(layer->A, stream);
        if (layer->B.Data) comm.all_reduce_avg(layer->B, stream);
    };

    auto all_reduce_grouped_layer = [&](std::optional<LoRAGroupedLayerWeights<Tensor>>& layer) {
        if (!layer.has_value()) return;
        if (layer->A.Data) comm.all_reduce_avg(layer->A, stream);
        if (layer->B.Data) comm.all_reduce_avg(layer->B, stream);
    };

    for (auto& block : mFullGrads.blocks) {
        all_reduce_layer(block.attention.q);
        all_reduce_layer(block.attention.k);
        all_reduce_layer(block.attention.v);
        all_reduce_layer(block.attention.o);
        all_reduce_layer(block.mlp.gate);
        all_reduce_layer(block.mlp.up);
        all_reduce_layer(block.mlp.down);

        if (block.moe.use_grouped) {
            all_reduce_grouped_layer(block.moe.grouped.gate);
            all_reduce_grouped_layer(block.moe.grouped.up);
            all_reduce_grouped_layer(block.moe.grouped.down);
        } else {
            // MoE expert LoRA gradients
            for (auto& expert : block.moe.experts) {
                all_reduce_layer(expert.gate);
                all_reduce_layer(expert.up);
                all_reduce_layer(expert.down);
            }
        }
    }
}

} // namespace modules
