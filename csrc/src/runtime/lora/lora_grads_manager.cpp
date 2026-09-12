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
    mSchemaHookDispatchEnabled = dsl::schema_hook_dispatch_enabled();

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
    // EP: grad buffers mirror the weights manager's local expert shard.
    const int E = mConfig.effective_grouped_experts();

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

    // Full grads are shape-only here; their storage is carved from the two
    // gradient arenas once every layer has been visited (carve_gradient_arenas).
    auto alloc_full = [&](int in_f, int out_f, const std::string& name) -> LoRALayerWeights<Tensor> {
        (void)name;
        LoRALayerWeights<Tensor> w;
        w.A = Tensor::empty(mConfig.grad_dtype, {r, in_f});
        w.B = Tensor::empty(mConfig.grad_dtype, {out_f, r});
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
        (void)name;
        LoRAGroupedLayerWeights<Tensor> w;
        w.A = Tensor::empty(mConfig.grad_dtype, {E, r, in_f});
        w.B = Tensor::empty(mConfig.grad_dtype, {E, out_f, r});
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
        bool is_qwen3_5 = false;
        if (mConfig.model_config) {
            bt = mConfig.model_config->get_block_type(l);
            is_hybrid = (mConfig.model_config->architecture == ArchitectureType::Hybrid);
            const bool is_qwen3_family = contains_ci(mConfig.model_config->ModelTypeName, "qwen3") ||
                                         contains_ci(mConfig.model_config->ArchitectureName, "qwen3");
            is_qwen3_5 = is_hybrid && is_qwen3_family;
        }
        const auto layer_dims = resolve_layer_dims(l);
        const int layer_q_out = layer_dims.q_out;
        const int layer_kv_out = layer_dims.kv_out;
        const int layer_d_ff = layer_dims.d_ff;
        const int q_lora_out = mConfig.lora_config.fused_qkv ? layer_q_out + 2 * layer_kv_out
                                                          : (model_is_qwen3_5 ? 2 * layer_q_out : layer_q_out);

        // Attention LoRA grads: Dense always, Attention always, MoE/SwitchMoE only in non-hybrid.
        // Non-hybrid MoE layers contain both attention AND MoE; hybrid MoE layers have only MoE.
        const bool has_attention = (bt == BlockType::Dense || bt == BlockType::Attention ||
                                    ((bt == BlockType::MoE || bt == BlockType::SwitchMoE) && !is_hybrid));
        if (static_cast<std::size_t>(l) < mConfig.attention_shapes.size()) {
            const auto& shapes = mConfig.attention_shapes[l];
            auto allocate = [&](int i, bool enabled, auto& full_proj, auto& shard_proj, const char* name) {
                if (!enabled || shapes[i].input == 0 || shapes[i].output == 0) return;
                full_proj = alloc_full(shapes[i].input, shapes[i].output, prefix + name);
                shard_proj = alloc_shard(shapes[i].input, shapes[i].output, prefix + name + "_shard");
            };
            allocate(0, mConfig.lora_config.applies_to_q(), full.attention.q, shard.attention.q, "_q");
            allocate(1, mConfig.lora_config.applies_to_k(), full.attention.k, shard.attention.k, "_k");
            allocate(2, mConfig.lora_config.applies_to_v(), full.attention.v, shard.attention.v, "_v");
            allocate(3, mConfig.lora_config.applies_to_o(), full.attention.o, shard.attention.o, "_o");
        } else if (has_attention) {
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
        bool layer_is_moe =
            (bt == BlockType::MoE || bt == BlockType::SwitchMoE) || (bt == BlockType::Dense && has_global_moe);
        // Qwen3.5 hybrid blocks (both linear-attention and full-attention)
        // contain standard MLP projections that should support LoRA.
        const bool layer_is_qwen3_linear_mlp = (bt == BlockType::Mamba) && is_qwen3_5;
        const bool layer_is_qwen3_attention_mlp = (bt == BlockType::Attention) && is_qwen3_5;
        bool layer_is_dense_mlp = (bt == BlockType::MLP) || (bt == BlockType::Dense && !has_global_moe) ||
                                  layer_is_qwen3_linear_mlp || layer_is_qwen3_attention_mlp;

        // Per-layer MLP structure from the DSL graph overrides the block-type
        // heuristics — must mirror ModularLoRAWeightsManager exactly so grad
        // buffers line up with the allocated LoRA weights.
        if (static_cast<std::size_t>(l) < mConfig.layer_has_moe.size()) {
            layer_is_moe = mConfig.layer_has_moe[static_cast<std::size_t>(l)] != 0;
        }
        if (static_cast<std::size_t>(l) < mConfig.layer_has_dense_mlp.size()) {
            layer_is_dense_mlp = mConfig.layer_has_dense_mlp[static_cast<std::size_t>(l)] != 0;
        }

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
        }
        // Not else-if: hybrid blocks can carry BOTH a dense MLP and MoE
        // experts (Gemma4 MoE variants run them in parallel within one block).
        if (layer_is_dense_mlp) {
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

    carve_gradient_arenas();
}

void ModularLoRAGradsManager::carve_gradient_arenas() {
    constexpr std::size_t kAlign = 256;
    auto aligned = [](std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; };
    // Visits every allocated full-grad tensor in layer-major, target-major order;
    // the same order both times, so the offsets of the sizing pass are the
    // offsets of the carving pass.
    auto for_each_grad = [&](auto&& fn) {
        for (auto& block : mFullGrads.blocks) {
            for_each_lora_layer_weight(block, [&](LoRATargetId id, auto& layer) {
                fn(id, layer.A);
                fn(id, layer.B);
            });
        }
    };

    std::size_t dense_bytes = 0;
    std::size_t expert_bytes = 0;
    for_each_grad([&](LoRATargetId id, Tensor& t) {
        if (t.Rank <= 0 || t.nelem() == 0) return;
        (lora_target_is_expert(id) ? expert_bytes : dense_bytes) += aligned(t.bytes());
    });

    const std::size_t elem_bytes = get_dtype_size(mConfig.grad_dtype);
    auto allocate_arena = [&](std::size_t bytes, const char* name) -> Tensor {
        if (bytes == 0) return Tensor{};
        return mAllocator->allocate(
            mConfig.grad_dtype, name, EAllocationType::ON_DEVICE, {static_cast<long>(bytes / elem_bytes)});
    };
    mDenseArena = allocate_arena(dense_bytes, "lora_grad_arena_dense");
    mExpertArena = allocate_arena(expert_bytes, "lora_grad_arena_expert");

    std::size_t dense_off = 0;
    std::size_t expert_off = 0;
    for_each_grad([&](LoRATargetId id, Tensor& t) {
        if (t.Rank <= 0 || t.nelem() == 0) return;
        const bool expert = lora_target_is_expert(id);
        const Tensor& arena = expert ? mExpertArena : mDenseArena;
        std::size_t& off = expert ? expert_off : dense_off;
        t.Data = arena.Data + off;
        t.Device = arena.Device;
        off += aligned(t.bytes());
    });
}

void ModularLoRAGradsManager::zero_all(cudaStream_t stream) {
    if (!mConfig.lora_config.enabled()) return;
    if (mDenseArena.Data) fill_zero(mDenseArena, stream);
    if (mExpertArena.Data) fill_zero(mExpertArena, stream);
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

    // One collective per arena rather than one per adapter tensor (two per
    // target per layer before). Expert adapters are EP-sharded: under EP they
    // average over the DP group only, since reducing them over the full world
    // would mix experts; in pure EP runs (dp_size == 1) that call is a no-op
    // and the expert grads stay local by construction.
    if (mDenseArena.Data) {
        comm.all_reduce_avg(mDenseArena, stream);
    }
    if (mExpertArena.Data) {
        if (comm.ep_enabled()) {
            comm.all_reduce_avg_dp(mExpertArena, stream);
        } else {
            comm.all_reduce_avg(mExpertArena, stream);
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
