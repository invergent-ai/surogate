// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_grads_manager.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <string_view>

#include "kernels/kernels.h"
#include "runtime/core/model_config.h"
#include "runtime/dsl/hook_registry.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"

namespace modules {

ModularLoRAGradsManager::ModularLoRAGradsManager(const Config& config,
                                                 const std::shared_ptr<TensorAllocator>& allocator)
    : mConfig(config),
      mAllocator(allocator) {
    if (const char* env = std::getenv("SUROGATE_ENABLE_SCHEMA_HOOK_DISPATCH")) {
        mSchemaHookDispatchEnabled = (std::string_view(env) == "1");
    }

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
    const int global_q_out = mConfig.num_query_heads * mConfig.head_size;
    const int global_kv_out = mConfig.num_kv_heads * mConfig.head_size;
    const int r = mConfig.lora_config.rank;
    const int E = mConfig.num_experts;

    // For hybrid models, per-layer attention / MLP dims may differ from the
    // global defaults. Resolve Q/K/V/O and MLP sizes per layer to avoid
    // mismatches between the forward LoRA weights (sized per-layer) and the
    // grad buffers.
    struct LayerDims {
        int q_out;
        int kv_out;
        int d_ff;
    };
    auto resolve_layer_dims = [&](int layer_idx) -> LayerDims {
        LayerDims out{global_q_out, global_kv_out, D};
        if (layer_idx >= 0 && static_cast<size_t>(layer_idx) < mConfig.per_layer_dims.size()) {
            const auto& d = mConfig.per_layer_dims[static_cast<size_t>(layer_idx)];
            if (d.attn_dim > 0) {
                out.q_out = static_cast<int>(d.attn_dim);
            }
            if (d.head_size > 0) {
                out.kv_out = mConfig.num_kv_heads * static_cast<int>(d.head_size);
            }
            if (d.intermediate > 0) {
                out.d_ff = static_cast<int>(d.intermediate);
            }
        }
        return out;
    };
    auto contains_ci = [](std::string_view haystack, std::string_view needle) {
        std::string h(haystack);
        std::string n(needle);
        std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return h.find(n) != std::string::npos;
    };
    const bool model_is_qwen3_5 =
        mConfig.model_config && (contains_ci(mConfig.model_config->ModelTypeName, "qwen3_5") ||
                                 contains_ci(mConfig.model_config->ModelTypeName, "qwen3.5") ||
                                 contains_ci(mConfig.model_config->ArchitectureName, "qwen3_5") ||
                                 contains_ci(mConfig.model_config->ArchitectureName, "qwen3.5"));
    const bool use_shared_expert = mConfig.model_config && mConfig.model_config->moe_config.has_value() &&
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
        w.A = TensorShard(
            mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {r, in_f}));
        w.B = mAllocator->allocate_shard(mConfig.grad_dtype,
                                         /*shard_idx=*/0,
                                         /*num_shards=*/1,
                                         (name + "_B").c_str(),
                                         {out_f, r});
        return w;
    };

    auto alloc_grouped_full = [&](int in_f, int out_f, const std::string& name) -> LoRAGroupedLayerWeights<Tensor> {
        LoRAGroupedLayerWeights<Tensor> w;
        w.A = mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {E, r, in_f});
        w.B =
            mAllocator->allocate(mConfig.grad_dtype, (name + "_B").c_str(), EAllocationType::ON_DEVICE, {E, out_f, r});
        return w;
    };
    auto alloc_grouped_shard =
        [&](int in_f, int out_f, const std::string& name) -> LoRAGroupedLayerWeights<TensorShard> {
        LoRAGroupedLayerWeights<TensorShard> w;
        if constexpr (!kAllocateLegacyShardedGradStorage) {
            return w;
        }
        w.A = TensorShard(
            mAllocator->allocate(mConfig.grad_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {E, r, in_f}));
        w.B = TensorShard(
            mAllocator->allocate(mConfig.grad_dtype, (name + "_B").c_str(), EAllocationType::ON_DEVICE, {E, out_f, r}));
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
            const bool is_qwen3_family = contains_ci(mConfig.model_config->ModelTypeName, "qwen3") ||
                                         contains_ci(mConfig.model_config->ArchitectureName, "qwen3");
            is_qwen3_hybrid = is_hybrid && is_qwen3_family;
        }
        const auto layer_dims = resolve_layer_dims(l);
        const int layer_q_out = layer_dims.q_out;
        const int layer_kv_out = layer_dims.kv_out;
        const int layer_d_ff = layer_dims.d_ff;
        const int q_lora_out = model_is_qwen3_5 ? (2 * layer_q_out) : layer_q_out;

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
                full.attention.k = alloc_full(C, layer_kv_out, prefix + "_k");
                shard.attention.k = alloc_shard(C, layer_kv_out, prefix + "_k_shard");
            }
            if (mConfig.lora_config.applies_to_v()) {
                full.attention.v = alloc_full(C, layer_kv_out, prefix + "_v");
                shard.attention.v = alloc_shard(C, layer_kv_out, prefix + "_v_shard");
            }
            if (mConfig.lora_config.applies_to_o()) {
                full.attention.o = alloc_full(layer_q_out, C, prefix + "_o");
                shard.attention.o = alloc_shard(layer_q_out, C, prefix + "_o_shard");
            }
        }

        // MoE LoRA grads: enable for MoE block types or Dense blocks in global MoE models.
        // Hybrid MoE blocks are supported via grouped GEMM LoRA hooks.
        const bool has_global_moe = (mConfig.num_experts > 0);
        const bool layer_is_moe =
            (bt == BlockType::MoE || bt == BlockType::SwitchMoE) || (bt == BlockType::Dense && has_global_moe);
        // Qwen3.5 hybrid blocks (both linear-attention and full-attention)
        // contain standard MLP projections that should support LoRA.
        const bool layer_is_qwen3_linear_mlp = (bt == BlockType::Mamba) && is_qwen3_hybrid;
        const bool layer_is_qwen3_attention_mlp = (bt == BlockType::Attention) && is_qwen3_hybrid;
        const bool layer_is_dense_mlp = (bt == BlockType::MLP) || (bt == BlockType::Dense && !has_global_moe) ||
                                        layer_is_qwen3_linear_mlp || layer_is_qwen3_attention_mlp;

        if (layer_is_moe && E > 0) {
            const bool has_mlp_lora = mConfig.lora_config.applies_to_gate() ||
                                      mConfig.lora_config.applies_to_gate_up() || mConfig.lora_config.applies_to_up() ||
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
                const bool has_shared_lora =
                    mConfig.lora_config.applies_to_up() || mConfig.lora_config.applies_to_down();
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
                full.mlp.gate = alloc_full(C, layer_d_ff, prefix + "_gate");
                shard.mlp.gate = alloc_shard(C, layer_d_ff, prefix + "_gate_shard");
            }
            if (mConfig.lora_config.applies_to_up()) {
                full.mlp.up = alloc_full(C, layer_d_ff, prefix + "_up");
                shard.mlp.up = alloc_shard(C, layer_d_ff, prefix + "_up_shard");
            }
            if (mConfig.lora_config.applies_to_down()) {
                full.mlp.down = alloc_full(layer_d_ff, C, prefix + "_down");
                shard.mlp.down = alloc_shard(layer_d_ff, C, prefix + "_down_shard");
            }
        }
        // Non-Qwen3 Mamba/SSM blocks still do not have dedicated LoRA gradient coverage here.
    }
}

void ModularLoRAGradsManager::zero_all(cudaStream_t stream) {
    if (!mConfig.lora_config.enabled()) return;

    for (auto& block : mFullGrads.blocks) {
        auto zero_layer = [stream](auto, auto& layer) {
            if (layer.A.Data) fill_zero(layer.A, stream);
            if (layer.B.Data) fill_zero(layer.B, stream);
        };
        for_each_lora_layer_weight(block, zero_layer);
    }
}

void ModularLoRAGradsManager::start_micro_step(cudaStream_t stream, int micro_step, int total_steps) {
    mIsFirstMicroStep = (micro_step == 0);
    mIsLastMicroStep = (micro_step == total_steps - 1);

    if (mIsFirstMicroStep) {
        zero_all(stream);
    }
}

void ModularLoRAGradsManager::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    if (!mConfig.lora_config.enabled()) return;
    if (mIsLastMicroStep) {
        reduce_gradients(stream, comm);
    }
}

LoRABlockWeights<Tensor>&
ModularLoRAGradsManager::get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
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

void ModularLoRAGradsManager::set_schema_hook_registry(const dsl::HookRegistry* registry,
                                                       std::vector<std::string> schema_ids_by_layer) {
    mSchemaHookRegistry = registry;
    mHookSchemaIdsByLayer = std::move(schema_ids_by_layer);
}

void ModularLoRAGradsManager::reduce_gradients(cudaStream_t stream, NCCLCommunicator& comm) {
    if (comm.world_size() == 1) return;

    for (int layer = 0; layer < static_cast<int>(mFullGrads.blocks.size()); ++layer) {
        dsl::GradientOffloadHookPayload payload;
        payload.lora_grads = this;
        payload.comm = &comm;
        payload.compute_stream = stream;
        payload.copy_stream = stream;
        payload.lora_gradients = true;
        dispatch_schema_layer_hooks(layer, stream, &payload);
        if (!payload.lora_reduced) {
            reduce_layer_gradients(layer, stream, comm);
        }
    }
}

void ModularLoRAGradsManager::reduce_layer_gradients(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mFullGrads.blocks.size())) {
        return;
    }
    const bool ep_active = comm.ep_enabled();
    auto all_reduce_layer = [&](LoRATargetId id, auto& layer) {
        // Expert adapters are EP-sharded; reducing them over the full world would mix experts.
        const bool reduce_dp_only = ep_active && lora_target_is_expert(id);
        if (layer.A.Data) {
            reduce_dp_only ? comm.all_reduce_avg_dp(layer.A, stream) : comm.all_reduce_avg(layer.A, stream);
        }
        if (layer.B.Data) {
            reduce_dp_only ? comm.all_reduce_avg_dp(layer.B, stream) : comm.all_reduce_avg(layer.B, stream);
        }
    };
    for_each_lora_layer_weight(mFullGrads.blocks[static_cast<std::size_t>(layer_idx)], all_reduce_layer);
}

int ModularLoRAGradsManager::dispatch_schema_layer_hooks(int layer_idx, cudaStream_t stream, void* payload) {
    if (!mSchemaHookDispatchEnabled || !mSchemaHookRegistry || layer_idx < 0 ||
        layer_idx >= static_cast<int>(mHookSchemaIdsByLayer.size())) {
        return 0;
    }
    const std::string& schema_id = mHookSchemaIdsByLayer[static_cast<std::size_t>(layer_idx)];
    if (schema_id.empty()) {
        return 0;
    }
    int dispatched = 0;
    for (const dsl::HookRegistration& registration : mSchemaHookRegistry->registrations()) {
        if (registration.event != dsl::HookEventKind::AfterAllReduce || registration.target.schema_id != schema_id) {
            continue;
        }
        dsl::HookContext context;
        context.layer_idx = layer_idx;
        context.target = registration.target;
        context.event = registration.event;
        context.stream = stream;
        context.payload = payload;
        if (registration.callback) {
            registration.callback(context);
        }
        ++dispatched;
    }
    return dispatched;
}

}  // namespace modules
