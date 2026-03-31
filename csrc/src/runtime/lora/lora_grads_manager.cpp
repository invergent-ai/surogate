// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_grads_manager.h"

#include <algorithm>
#include <cctype>
#include <fmt/format.h>
#include <string>

#include "kernels/kernels.h"
#include "runtime/core/model_config.h"
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
    // No runtime path currently consumes mShardedGrads. Keeping a second full device copy
    // of all LoRA grad buffers materially increases VRAM pressure for EP+MoE models, so
    // leave the sharded set as empty metadata until a caller actually needs it.
    constexpr bool kAllocateLegacyShardedGradStorage = false;

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int D_moe = mConfig.effective_moe_intermediate();
    const int q_out = mConfig.num_query_heads * mConfig.head_size;
    const int kv_out = mConfig.num_kv_heads * mConfig.head_size;
    const int r = mConfig.lora_config.rank;
    const int E = mConfig.num_experts;
    auto contains_ci = [](std::string_view haystack, std::string_view needle) {
        std::string h(haystack);
        std::string n(needle);
        std::transform(h.begin(), h.end(), h.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::transform(n.begin(), n.end(), n.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return h.find(n) != std::string::npos;
    };
    const bool model_is_qwen3_5 =
        mConfig.model_config &&
        (contains_ci(mConfig.model_config->ModelTypeName, "qwen3_5") ||
         contains_ci(mConfig.model_config->ModelTypeName, "qwen3.5") ||
         contains_ci(mConfig.model_config->ArchitectureName, "qwen3_5") ||
         contains_ci(mConfig.model_config->ArchitectureName, "qwen3.5"));
    const bool use_shared_expert = mConfig.model_config &&
                                   mConfig.model_config->moe_config.has_value() &&
                                   mConfig.model_config->moe_config->use_shared_expert;
    const int shared_D = use_shared_expert && mConfig.model_config->moe_config->shared_expert_size > 0
                             ? mConfig.model_config->moe_config->shared_expert_size
                             : D_moe;

    auto alloc_full = [&](int in_f, int out_f, const std::string& name) -> LoRALayerWeights<Tensor> {
        LoRALayerWeights<Tensor> w;
        w.A = mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {r, in_f});
        w.B = mAllocator->allocate(mConfig.grad_dtype, (name + "_B").c_str(), EAllocationType::ON_DEVICE, {out_f, r});
        return w;
    };
    auto alloc_shard = [&](int in_f, int out_f, const std::string& name) -> LoRALayerWeights<TensorShard> {
        LoRALayerWeights<TensorShard> w;
        if constexpr (!kAllocateLegacyShardedGradStorage) {
            return w;
        }
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
        if constexpr (!kAllocateLegacyShardedGradStorage) {
            return w;
        }
        w.A = TensorShard(mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {E, r, in_f}));
        w.B = TensorShard(mAllocator->allocate(mConfig.grad_dtype, (name + "_B").c_str(), EAllocationType::ON_DEVICE, {E, out_f, r}));
        return w;
    };

    for (int l = 0; l < mConfig.num_layers; ++l) {
        std::string prefix = fmt::format("lora_grad_layer_{}", l);
        auto& full = mFullGrads.blocks[l];
        auto& shard = mShardedGrads.blocks[l];

        // Determine block type for this layer (hybrid-aware)
        BlockType bt = BlockType::Dense;
        bool is_hybrid = false;
        bool is_qwen3_hybrid = false;
        if (mConfig.model_config) {
            bt = mConfig.model_config->get_block_type(l);
            is_hybrid = (mConfig.model_config->architecture == ArchitectureType::Hybrid);
            const bool is_qwen3_family =
                contains_ci(mConfig.model_config->ModelTypeName, "qwen3") ||
                contains_ci(mConfig.model_config->ArchitectureName, "qwen3");
            is_qwen3_hybrid = is_hybrid && is_qwen3_family;
        }
        const int q_lora_out = model_is_qwen3_5 ? (2 * q_out) : q_out;

        // Attention LoRA grads: Dense always, Attention always, MoE/SwitchMoE only in non-hybrid.
        // Non-hybrid MoE layers contain both attention AND MoE; hybrid MoE layers have only MoE.
        const bool has_attention = (bt == BlockType::Dense || bt == BlockType::Attention ||
                                   ((bt == BlockType::MoE || bt == BlockType::SwitchMoE) && !is_hybrid));
        if (has_attention) {
            if (mConfig.lora_config.applies_to_q()) {
                full.attention.q = alloc_full(C, q_lora_out, prefix + "_q");
                shard.attention.q = alloc_shard(C, q_lora_out, prefix + "_q_shard");
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
        }

        // MoE LoRA grads: enable for MoE block types or Dense blocks in global MoE models.
        // Hybrid MoE blocks are supported via grouped GEMM LoRA hooks.
        const bool has_global_moe = (mConfig.num_experts > 0);
        const bool layer_is_moe =
            (bt == BlockType::MoE || bt == BlockType::SwitchMoE) ||
            (bt == BlockType::Dense && has_global_moe);
        // Qwen3.5 hybrid blocks (both linear-attention and full-attention)
        // contain standard MLP projections that should support LoRA.
        const bool layer_is_qwen3_linear_mlp = (bt == BlockType::Mamba) && is_qwen3_hybrid;
        const bool layer_is_qwen3_attention_mlp = (bt == BlockType::Attention) && is_qwen3_hybrid;
        const bool layer_is_dense_mlp = (bt == BlockType::MLP) ||
                                         (bt == BlockType::Dense && !has_global_moe) ||
                                         layer_is_qwen3_linear_mlp ||
                                         layer_is_qwen3_attention_mlp;

        if (layer_is_moe && E > 0) {
            const bool has_mlp_lora = mConfig.lora_config.applies_to_gate() ||
                                       mConfig.lora_config.applies_to_gate_up() ||
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
                if (mConfig.lora_config.applies_to_gate_up()) {
                    full.moe.grouped.gate_up = alloc_grouped_full(C, 2 * D_moe, exp_prefix + "_gate_up");
                    shard.moe.grouped.gate_up = alloc_grouped_shard(C, 2 * D_moe, exp_prefix + "_gate_up_shard");
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

            if (use_shared_expert) {
                const bool has_shared_lora = mConfig.lora_config.applies_to_up() ||
                                             mConfig.lora_config.applies_to_down();
                if (has_shared_lora) {
                    full.moe.shared.emplace();
                    shard.moe.shared.emplace();
                    if (mConfig.lora_config.applies_to_up()) {
                        full.moe.shared->up = alloc_full(C, shared_D, prefix + "_shared_up");
                        shard.moe.shared->up = alloc_shard(C, shared_D, prefix + "_shared_up_shard");
                    }
                    if (mConfig.lora_config.applies_to_down()) {
                        full.moe.shared->down = alloc_full(shared_D, C, prefix + "_shared_down");
                        shard.moe.shared->down = alloc_shard(shared_D, C, prefix + "_shared_down_shard");
                    }
                }
            }

            if (mConfig.train_router) {
                full.router = alloc_full(C, E, prefix + "_router");
                shard.router = alloc_shard(C, E, prefix + "_router_shard");
            }
        } else if (layer_is_dense_mlp) {
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
        // Non-Qwen3 Mamba/SSM blocks still do not have dedicated LoRA gradient coverage here.
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
            zero_layer(block.mlp.gate_up);
            zero_layer(block.mlp.up);
            zero_layer(block.mlp.down);

            if (block.moe.use_grouped) {
                zero_layer(block.moe.grouped.gate);
                zero_layer(block.moe.grouped.gate_up);
                zero_layer(block.moe.grouped.up);
                zero_layer(block.moe.grouped.down);
            } else {
                // MoE expert LoRA gradients
                for (auto& expert : block.moe.experts) {
                    zero_layer(expert.gate);
                    zero_layer(expert.gate_up);
                    zero_layer(expert.up);
                    zero_layer(expert.down);
                }
            }

            if (block.moe.shared.has_value()) {
                zero_layer(block.moe.shared->up);
                zero_layer(block.moe.shared->down);
            }

            // Router LoRA gradients
            zero_layer(block.router);
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
    const bool ep_active = comm.ep_enabled();

    auto all_reduce_layer = [&](std::optional<LoRALayerWeights<Tensor>>& layer) {
        if (!layer.has_value()) return;
        if (layer->A.Data) comm.all_reduce_avg(layer->A, stream);
        if (layer->B.Data) comm.all_reduce_avg(layer->B, stream);
    };

    auto all_reduce_layer_dp = [&](std::optional<LoRALayerWeights<Tensor>>& layer) {
        if (!layer.has_value()) return;
        if (layer->A.Data) comm.all_reduce_avg_dp(layer->A, stream);
        if (layer->B.Data) comm.all_reduce_avg_dp(layer->B, stream);
    };

    auto all_reduce_grouped_layer = [&](std::optional<LoRAGroupedLayerWeights<Tensor>>& layer) {
        if (!layer.has_value()) return;
        if (layer->A.Data) comm.all_reduce_avg(layer->A, stream);
        if (layer->B.Data) comm.all_reduce_avg(layer->B, stream);
    };

    auto all_reduce_grouped_layer_dp = [&](std::optional<LoRAGroupedLayerWeights<Tensor>>& layer) {
        if (!layer.has_value()) return;
        if (layer->A.Data) comm.all_reduce_avg_dp(layer->A, stream);
        if (layer->B.Data) comm.all_reduce_avg_dp(layer->B, stream);
    };

    for (auto& block : mFullGrads.blocks) {
        all_reduce_layer(block.attention.q);
        all_reduce_layer(block.attention.k);
        all_reduce_layer(block.attention.v);
        all_reduce_layer(block.attention.o);
        all_reduce_layer(block.mlp.gate);
        all_reduce_layer(block.mlp.gate_up);
        all_reduce_layer(block.mlp.up);
        all_reduce_layer(block.mlp.down);

        if (block.moe.use_grouped) {
            // Expert LoRA tensors are sharded by EP rank. Reducing across the full world
            // would mix different experts; reduce across DP group only when EP is active.
            if (ep_active) {
                all_reduce_grouped_layer_dp(block.moe.grouped.gate);
                all_reduce_grouped_layer_dp(block.moe.grouped.gate_up);
                all_reduce_grouped_layer_dp(block.moe.grouped.up);
                all_reduce_grouped_layer_dp(block.moe.grouped.down);
            } else {
                all_reduce_grouped_layer(block.moe.grouped.gate);
                all_reduce_grouped_layer(block.moe.grouped.gate_up);
                all_reduce_grouped_layer(block.moe.grouped.up);
                all_reduce_grouped_layer(block.moe.grouped.down);
            }
        } else {
            // MoE expert LoRA gradients
            for (auto& expert : block.moe.experts) {
                if (ep_active) {
                    all_reduce_layer_dp(expert.gate);
                    all_reduce_layer_dp(expert.gate_up);
                    all_reduce_layer_dp(expert.up);
                    all_reduce_layer_dp(expert.down);
                } else {
                    all_reduce_layer(expert.gate);
                    all_reduce_layer(expert.gate_up);
                    all_reduce_layer(expert.up);
                    all_reduce_layer(expert.down);
                }
            }
        }

        if (block.moe.shared.has_value()) {
            all_reduce_layer(block.moe.shared->up);
            all_reduce_layer(block.moe.shared->down);
        }

        // Router LoRA gradients
        all_reduce_layer(block.router);
    }
}

} // namespace modules
