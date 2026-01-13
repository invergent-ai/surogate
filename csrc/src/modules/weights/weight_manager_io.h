// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Weight Manager I/O - Import/export weights from/to HuggingFace format
//
// This file provides the implementation of weight import/export for the
// ModularWeightManager. It handles:
// 1. Direct tensor mapping (fused QKV, fused gate+up)
// 2. Split tensor fusion from HuggingFace format (Q, K, V -> QKV)
// 3. Sharded weight loading for distributed training
// 4. Model-specific patterns via the weight mapping registry
//
// The registry-based system in weight_mapping.h provides an extensible approach
// for adding new model architectures through inheritance.

#ifndef LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_IO_H
#define LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_IO_H

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernels/kernels.h"
#include "models/llama/config.h"
#include "models/registry.h"
#include "modules/weights/weight_manager_helpers.h"
#include "modules/weights/weight_mapping.h"
#include "utilities/comm.h"
#include "utilities/philox.h"
#include "utilities/safetensors.h"
#include "utilities/utils.h"

namespace modules {

template<typename Block>
void ModularWeightManager<Block>::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {

    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;
    long qkv_rows = HS * (HQ + 2 * HKV);
    long q_rows = HS * HQ;

    auto shard = [&](const Tensor& t, const std::vector<long>& global_shape) -> TensorShard {
        if (mConfig.num_shards == 1) return TensorShard(t);
        return TensorShard(t, mConfig.shard_idx, mConfig.num_shards, global_shape);
    };

    // Non-block weights
    callback("model.embed_tokens.weight", mConfig.num_shards == 1
                                           ? TensorShard(mMasterNonBlock.embeddings)
                                           : TensorShard(mMasterNonBlock.embeddings, mConfig.shard_idx, mConfig.num_shards,
                                                        std::vector<long>{mConfig.vocab_size, mConfig.hidden_size}));
    callback("model.norm.weight", mConfig.num_shards == 1
                                     ? TensorShard(mMasterNonBlock.final_norm_weight)
                                     : TensorShard(mMasterNonBlock.final_norm_weight, mConfig.shard_idx, mConfig.num_shards,
                                                  std::vector<long>{mConfig.hidden_size}));
    if (!mConfig.tied_embeddings) {
        callback("lm_head.weight", mConfig.num_shards == 1
                                      ? TensorShard(mMasterNonBlock.lm_head)
                                      : TensorShard(mMasterNonBlock.lm_head, mConfig.shard_idx, mConfig.num_shards,
                                                   std::vector<long>{mConfig.vocab_size, mConfig.hidden_size}));
    }

    // Block weights
    for (int i = 0; i < mConfig.num_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        auto& block = mMasterBlocks[i];

        // Layer norms
        callback(prefix + ".input_layernorm.weight", shard(block.ln1.weight, {C}));
        callback(prefix + ".post_attention_layernorm.weight", shard(block.ln2.weight, {C}));

        // Attention weights (fused QKV stored as single tensor)
        callback(prefix + ".self_attn.qkv.weight", shard(block.attention.qkv_weight, {qkv_rows, C}));
        if (block.attention.qkv_bias.has_value()) {
            callback(prefix + ".self_attn.qkv.bias", shard(block.attention.qkv_bias.value(), {qkv_rows}));
        }
        callback(prefix + ".self_attn.o_proj.weight", shard(block.attention.out_weight, {C, q_rows}));
        // QK norm weights - only for attention modules that have them (Qwen3-style)
        if constexpr (has_qk_norm_weights<decltype(block.attention)>::value) {
            if (block.attention.q_norm_weight.has_value()) {
                callback(prefix + ".self_attn.q_norm.weight", shard(block.attention.q_norm_weight.value(), {HS}));
            }
            if (block.attention.k_norm_weight.has_value()) {
                callback(prefix + ".self_attn.k_norm.weight", shard(block.attention.k_norm_weight.value(), {HS}));
            }
        }

        // MLP weights - only for dense blocks
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            callback(prefix + ".mlp.up.weight", shard(block.mlp_up_weight, {2 * D, C}));
            callback(prefix + ".mlp.down_proj.weight", shard(block.mlp_down_weight, {C, D}));
        }

        // MoE-specific weights (router, experts) - only for MoE blocks
        if constexpr (has_moe_weights<BlockWeights>::value) {
            const auto& bcfg = mConfig.block_config;
            int num_experts = bcfg.num_experts;
            callback(prefix + ".mlp.router.gate.weight", shard(block.router.gate, {num_experts, C}));

            if (block.experts.use_batched) {
                callback(prefix + ".mlp.experts.gate_up_proj.weight",
                         shard(block.experts.gate_up_proj, {num_experts, 2 * D, C}));
                callback(prefix + ".mlp.experts.down_proj.weight",
                         shard(block.experts.down_proj, {num_experts, C, D}));
            } else {
                for (int e = 0; e < static_cast<int>(block.experts.experts.size()); ++e) {
                    std::string exp_prefix = prefix + ".mlp.experts." + std::to_string(e);
                    auto& expert = block.experts.experts[e];
                    callback(exp_prefix + ".gate_proj.weight", shard(expert.gate_proj, {D, C}));
                    callback(exp_prefix + ".up_proj.weight", shard(expert.up_proj, {D, C}));
                    callback(exp_prefix + ".down_proj.weight", shard(expert.down_proj, {C, D}));
                }
            }

            if (block.shared_expert.has_value()) {
                int shared_D = bcfg.shared_expert_intermediate > 0 ?
                               bcfg.shared_expert_intermediate : static_cast<int>(D);
                callback(prefix + ".mlp.shared_expert.gate_proj.weight",
                         shard(block.shared_expert->gate_proj, {shared_D, C}));
                callback(prefix + ".mlp.shared_expert.up_proj.weight",
                         shard(block.shared_expert->up_proj, {shared_D, C}));
                callback(prefix + ".mlp.shared_expert.down_proj.weight",
                         shard(block.shared_expert->down_proj, {C, shared_D}));
            }
        }
    }
}

template<typename Block>
void ModularWeightManager<Block>::random_init(int seed, NCCLCommunicator& comm) {
    if (mConfig.offload_master && mMasterNonBlock.embeddings.Device == -1 && !mConfig.use_zero_copy) {
        throw std::runtime_error("ModularWeightManager::random_init: --offload-master requires --use-zero-copy for random initialization");
    }

    Philox4x32 rng(seed);

    float scale = 0.02f;
    float residual_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(mConfig.num_layers));

    for (int l = 0; l < mConfig.num_layers; ++l) {
        auto local_seeds = rng.generate(comm.rank(), l);
        auto& layer = mMasterBlocks[l];

        fill_constant(layer.ln1.weight, 1.f, layer.ln1.weight.nelem(), nullptr);
        fill_constant(layer.ln2.weight, 1.f, layer.ln2.weight.nelem(), nullptr);
        if constexpr (has_qk_norm_weights<decltype(layer.attention)>::value) {
            if (layer.attention.q_norm_weight.has_value()) {
                fill_constant(layer.attention.q_norm_weight.value(), 1.f, layer.attention.q_norm_weight->nelem(), nullptr);
            }
            if (layer.attention.k_norm_weight.has_value()) {
                fill_constant(layer.attention.k_norm_weight.value(), 1.f, layer.attention.k_norm_weight->nelem(), nullptr);
            }
        }

        fill_normal(layer.attention.qkv_weight, layer.attention.qkv_weight.nelem(), 0.f, scale, seed, local_seeds[0], nullptr);
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            fill_normal(layer.mlp_up_weight, layer.mlp_up_weight.nelem(), 0.f, scale, seed, local_seeds[1], nullptr);
        }

        if (mConfig.block_config.use_qkv_bias && layer.attention.qkv_bias.has_value()) {
            fill_zero(layer.attention.qkv_bias.value(), nullptr);
        }

        if (mConfig.init_projections_to_zero) {
            fill_zero(layer.attention.out_weight, nullptr);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                fill_zero(layer.mlp_down_weight, nullptr);
            }
        } else {
            fill_normal(layer.attention.out_weight, layer.attention.out_weight.nelem(), 0.f, scale * residual_scale, seed, local_seeds[3], nullptr);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                fill_normal(layer.mlp_down_weight, layer.mlp_down_weight.nelem(), 0.f, scale * residual_scale, seed, local_seeds[2], nullptr);
            }
        }
    }

    auto local_seeds = rng.generate(comm.rank(), mConfig.num_layers);
    fill_normal(mMasterNonBlock.embeddings, mMasterNonBlock.embeddings.nelem(), 0.f, scale, seed, local_seeds[0], nullptr);
    if (!mConfig.tied_embeddings && mMasterNonBlock.lm_head.Data != mMasterNonBlock.embeddings.Data) {
        fill_normal(mMasterNonBlock.lm_head, mMasterNonBlock.lm_head.nelem(), 0.f, scale, seed, local_seeds[1], nullptr);
    }
    fill_constant(mMasterNonBlock.final_norm_weight, 1.f, mMasterNonBlock.final_norm_weight.nelem(), nullptr);

    synchronize_absmax(comm);
    comm.barrier();
}

namespace {
    /**
     * @brief Load only the intersection of a global element range into a sharded destination tensor.
     */
    inline void load_intersect(const TensorShard& dst, const SafeTensorEntry& src,
                               std::ptrdiff_t src_begin, std::ptrdiff_t src_end,
                               bool allow_cast) {
        if (src_begin >= src_end) return;

        std::ptrdiff_t dst_begin = dst.shard_offset();
        std::ptrdiff_t dst_end = dst_begin + static_cast<std::ptrdiff_t>(dst.nelem());

        if (src_begin >= dst_end) return;
        if (src_end <= dst_begin) return;

        std::ptrdiff_t slice_begin = src_begin < dst_begin ? dst_begin : src_begin;
        std::ptrdiff_t slice_end = src_end > dst_end ? dst_end : src_end;

        std::ptrdiff_t dst_offset = slice_begin - dst_begin;
        std::ptrdiff_t elements = slice_end - slice_begin;

        Tensor dst_slice = static_cast<const Tensor&>(dst);
        dst_slice.Sizes.fill(1);
        dst_slice.Rank = 1;
        dst_slice.Sizes[0] = elements;
        dst_slice.Data = dst.Data + dst_offset * get_dtype_size(dst.DType);

        src.read_raw(dst_slice, slice_begin - src_begin, elements, allow_cast);
    }

    /**
     * @brief Load full tensor (range {0,0} means entire tensor)
     */
    inline void load_full(const TensorShard& dst, const SafeTensorEntry& src, bool allow_cast) {
        load_intersect(dst, src, 0, static_cast<std::ptrdiff_t>(dst.global_nelem()), allow_cast);
    }
}

template<typename Block>
void ModularWeightManager<Block>::import_from_file(const std::string& filename, bool allow_cast, NCCLCommunicator& comm) {
    SafeTensorsReader reader{filename};

    const auto& cfg = mConfig.block_config;
    const long C = cfg.hidden_size;
    const long D = cfg.intermediate_size;
    const long HS = cfg.head_size;
    const long HQ = cfg.num_query_heads;
    const long HKV = cfg.num_kv_heads;
    const long q_rows = HS * HQ;
    const long kv_rows = HS * HKV;
    const long fused_rows = q_rows + 2 * kv_rows;

    const bool load_sharded =
        mStreamWeights ||
        mConfig.offload_master ||
        (mConfig.master_dtype != mConfig.model_dtype) ||
        (mConfig.master_dtype != mConfig.matmul_dtype);
    const int load_idx = load_sharded ? mConfig.shard_idx : 0;
    const int load_num = load_sharded ? mConfig.num_shards : 1;

    auto& dst_nonblock = load_sharded ? mMasterNonBlock : mWorkNonBlock;
    auto& dst_blocks = load_sharded ? mMasterBlocks : mWorkBlocks;

    // Helper to create TensorShard with appropriate sharding
    auto make_shard = [&](Tensor& t, const std::vector<long>& global_shape) -> TensorShard {
        if (load_num == 1) return TensorShard(t);
        return TensorShard(t, load_idx, load_num, global_shape);
    };

    // ========================================================================
    // Tensor Resolution: Map TensorTarget enum to actual TensorShard
    // ========================================================================

    // Resolve non-block tensors
    auto resolve_nonblock = [&](TensorTarget target) -> std::optional<TensorShard> {
        switch (target) {
            case TensorTarget::Embeddings:
                return make_shard(dst_nonblock.embeddings, {mConfig.vocab_size, mConfig.hidden_size});
            case TensorTarget::FinalNorm:
                return make_shard(dst_nonblock.final_norm_weight, {mConfig.hidden_size});
            case TensorTarget::LMHead:
                return make_shard(dst_nonblock.lm_head, {mConfig.vocab_size, mConfig.hidden_size});
            default:
                return std::nullopt;
        }
    };

    // Resolve block tensors (dense blocks)
    auto resolve_block = [&](TensorTarget target, int layer_idx) -> std::optional<TensorShard> {
        if (layer_idx < 0 || layer_idx >= mConfig.num_layers) return std::nullopt;
        auto& block = dst_blocks.at(layer_idx);

        switch (target) {
            case TensorTarget::LN1Weight:
                return make_shard(block.ln1.weight, {C});
            case TensorTarget::LN2Weight:
                return make_shard(block.ln2.weight, {C});
            case TensorTarget::QKVWeight:
                return make_shard(block.attention.qkv_weight, {fused_rows, C});
            case TensorTarget::QKVBias:
                if (block.attention.qkv_bias.has_value()) {
                    return make_shard(block.attention.qkv_bias.value(), {fused_rows});
                }
                return std::nullopt;
            case TensorTarget::OutWeight:
                return make_shard(block.attention.out_weight, {C, q_rows});
            case TensorTarget::QNormWeight:
                if constexpr (has_qk_norm_weights<decltype(block.attention)>::value) {
                    if (block.attention.q_norm_weight.has_value()) {
                        return make_shard(block.attention.q_norm_weight.value(), {HS});
                    }
                }
                return std::nullopt;
            case TensorTarget::KNormWeight:
                if constexpr (has_qk_norm_weights<decltype(block.attention)>::value) {
                    if (block.attention.k_norm_weight.has_value()) {
                        return make_shard(block.attention.k_norm_weight.value(), {HS});
                    }
                }
                return std::nullopt;
            case TensorTarget::MLPUpWeight:
                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    return make_shard(block.mlp_up_weight, {2 * D, C});
                }
                return std::nullopt;
            case TensorTarget::MLPDownWeight:
                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    return make_shard(block.mlp_down_weight, {C, D});
                }
                return std::nullopt;

            // MoE targets
            case TensorTarget::RouterGate:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    return make_shard(block.router.gate, {cfg.num_experts, C});
                }
                return std::nullopt;
            case TensorTarget::RouterBias:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    // bias: (num_experts,)
                    if (!block.router.bias.has_value()) {
                         // Allocate if not already there but mapped
                         block.router.bias = Tensor();
                         block.router.bias->DType = block.router.gate.DType;
                    }
                    return make_shard(*block.router.bias, {cfg.num_experts});
                }
                return std::nullopt;
            case TensorTarget::ExpertsGateUp:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.experts.use_batched) {
                        return make_shard(block.experts.gate_up_proj, {cfg.num_experts, 2 * D, C});
                    }
                }
                return std::nullopt;
            case TensorTarget::ExpertsDown:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.experts.use_batched) {
                        return make_shard(block.experts.down_proj, {cfg.num_experts, C, D});
                    }
                }
                return std::nullopt;
            case TensorTarget::SharedExpertGate:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.shared_expert.has_value()) {
                        int shared_D = cfg.shared_expert_intermediate > 0 ?
                                       cfg.shared_expert_intermediate : static_cast<int>(D);
                        return make_shard(block.shared_expert->gate_proj, {shared_D, C});
                    }
                }
                return std::nullopt;
            case TensorTarget::SharedExpertUp:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.shared_expert.has_value()) {
                        int shared_D = cfg.shared_expert_intermediate > 0 ?
                                       cfg.shared_expert_intermediate : static_cast<int>(D);
                        return make_shard(block.shared_expert->up_proj, {shared_D, C});
                    }
                }
                return std::nullopt;
            case TensorTarget::SharedExpertDown:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.shared_expert.has_value()) {
                        int shared_D = cfg.shared_expert_intermediate > 0 ?
                                       cfg.shared_expert_intermediate : static_cast<int>(D);
                        return make_shard(block.shared_expert->down_proj, {C, shared_D});
                    }
                }
                return std::nullopt;

            default:
                return std::nullopt;
        }
    };

    // Resolve per-expert tensors (non-batched MoE)
    auto resolve_expert = [&](TensorTarget target, int layer_idx, int expert_idx) -> std::optional<TensorShard> {
        if constexpr (has_moe_weights<BlockWeights>::value) {
            if (layer_idx < 0 || layer_idx >= mConfig.num_layers) return std::nullopt;
            auto& block = dst_blocks.at(layer_idx);

            // HF Qwen3-MoE checkpoints store experts as separate gate_proj/up_proj/down_proj tensors.
            // Internally we use a batched expert layout for grouped GEMM. When use_batched is enabled,
            // map per-expert tensors into the corresponding slice of the batched weights:
            //   - experts.gate_up_proj stores [up; gate] (rows 0..D-1 = up, rows D..2D-1 = gate)
            //   - experts.down_proj stores per-expert down weights (C, D)
            if (block.experts.use_batched) {
                const int E = cfg.num_experts;
                if (expert_idx < 0 || expert_idx >= E) return std::nullopt;

                // When loading sharded master weights, only materialize experts that belong to this shard.
                const int load_experts_per_shard = (load_num > 1) ? div_exact(E, load_num) : E;
                const int shard_begin = (load_num > 1) ? (load_idx * load_experts_per_shard) : 0;
                const int shard_end = shard_begin + load_experts_per_shard;
                if (expert_idx < shard_begin || expert_idx >= shard_end) return std::nullopt;
                const int local_expert = expert_idx - shard_begin;

                const std::size_t elem_size = get_dtype_size(block.experts.gate_up_proj.DType);
                if (target == TensorTarget::ExpertUp || target == TensorTarget::ExpertGate) {
                    // Slice into (E_local, 2*D, C) -> (D, C)
                    const long row_offset = (target == TensorTarget::ExpertUp) ? 0 : D;
                    Tensor slice = block.experts.gate_up_proj;
                    slice.Rank = 2;
                    slice.Sizes[0] = D;
                    slice.Sizes[1] = C;
                    const std::size_t base_elems = (static_cast<std::size_t>(local_expert) * static_cast<std::size_t>(2 * D) + static_cast<std::size_t>(row_offset)) * static_cast<std::size_t>(C);
                    slice.Data = block.experts.gate_up_proj.Data + base_elems * elem_size;
                    return TensorShard(slice);
                }
                if (target == TensorTarget::ExpertDown) {
                    // Slice into (E_local, C, D) -> (C, D)
                    Tensor slice = block.experts.down_proj;
                    slice.Rank = 2;
                    slice.Sizes[0] = C;
                    slice.Sizes[1] = D;
                    const std::size_t base_elems = static_cast<std::size_t>(local_expert) * static_cast<std::size_t>(C) * static_cast<std::size_t>(D);
                    slice.Data = block.experts.down_proj.Data + base_elems * elem_size;
                    return TensorShard(slice);
                }
                return std::nullopt;
            }

            // Non-batched expert layout: direct per-expert tensors.
            if (expert_idx < 0 || expert_idx >= static_cast<int>(block.experts.experts.size())) return std::nullopt;

            auto& expert = block.experts.experts[expert_idx];
            switch (target) {
                case TensorTarget::ExpertGate:
                    return make_shard(expert.gate_proj, {D, C});
                case TensorTarget::ExpertUp:
                    return make_shard(expert.up_proj, {D, C});
                case TensorTarget::ExpertDown:
                    return make_shard(expert.down_proj, {C, D});
                default:
                    return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
    };

    // ========================================================================
    // Create weight mapping registry for this model type
    // ========================================================================

    // Create a temporary PretrainedConfig-like object for range computation
    // We use a LlamaConfig as a proxy since the range functions only need
    // NumQueryHeads, NumKeyValHeads, HiddenSize, IntermediateSize
    LlamaConfig proxy_cfg;
    proxy_cfg.NumQueryHeads = static_cast<int>(HQ);
    proxy_cfg.NumKeyValHeads = static_cast<int>(HKV);
    proxy_cfg.HiddenSize = static_cast<int>(C);
    proxy_cfg.IntermediateSize = static_cast<int>(D);
    proxy_cfg.HeadDim = static_cast<int>(HS);

    // Get weight mapping from the registry using architecture ID
    auto mapping = models::create_weight_mapping(mConfig.architecture_id);

    // ========================================================================
    // Process each tensor using the registry
    // ========================================================================

    for (const auto& entry : reader.entries()) {
        const std::string& name = entry.name();

        // Match against registered patterns
        PatternMatch match = mapping->match(name, proxy_cfg, C);

        if (!match) {
            // No pattern matched - skip silently (could be optimizer state, etc.)
            continue;
        }

        // Resolve the target tensor
        std::optional<TensorShard> dst_opt;

        if (match.expert_idx >= 0) {
            // Per-expert tensor
            dst_opt = resolve_expert(match.pattern->target, match.layer_idx, match.expert_idx);
        } else if (match.layer_idx >= 0) {
            // Per-layer tensor
            dst_opt = resolve_block(match.pattern->target, match.layer_idx);
        } else {
            // Non-block tensor
            dst_opt = resolve_nonblock(match.pattern->target);
        }

        if (!dst_opt.has_value()) {
            // Target not available (optional tensor or type mismatch)
            if (!match.pattern->optional) {
                // Non-optional tensor should exist - this might indicate a problem
                // but we'll skip silently for robustness
            }
            continue;
        }

        const TensorShard& dst = dst_opt.value();

        // Load the tensor, applying range for fusion if needed
        if (match.range.is_full_tensor()) {
            // Direct load - full tensor
            load_full(dst, entry, allow_cast);
        } else {
            // Fusion load - load into a slice of the destination
            load_intersect(dst, entry, match.range.begin, match.range.end, allow_cast);
        }
    }

    synchronize_absmax(comm);
    comm.barrier();
}

template<typename Block>
void ModularWeightManager<Block>::export_to_file(const std::string& filename, NCCLCommunicator& comm) const {
    if (mStreamWeights && comm.world_size() > 1) {
        throw std::runtime_error("ModularWeightManager::export_to_file: export is not supported for ZeRO-3 streamed weights yet; use per-rank checkpoints or export with --gpus 1");
    }

    // Ensure full work weights are materialized before exporting.
    if (!mStreamWeights &&
        (comm.world_size() > 1 ||
         mConfig.offload_master ||
         (mConfig.master_dtype != mConfig.model_dtype) ||
         (mConfig.master_dtype != mConfig.matmul_dtype))) {
        auto* self = const_cast<ModularWeightManager*>(this);
        cudaStream_t s = comm.stream();
        self->gather_embeddings(comm, s);
        self->gather_final_norm(comm, s);
        self->gather_lm_head(comm, s);
        for (int i = 0; i < mConfig.num_layers; ++i) {
            self->gather_block(i, comm, s);
            (void)self->get_block(i, s);
            self->release_block(i, s);
        }
        comm.wait_on_comms(s);
        comm.barrier();
    }

    [[maybe_unused]] const auto& cfg = mConfig.block_config;
    const long D = cfg.intermediate_size;
    const long HS = cfg.head_size;
    const long HQ = cfg.num_query_heads;
    const long HKV = cfg.num_kv_heads;
    const long q_rows = HS * HQ;
    const long kv_rows = HS * HKV;

    // ========================================================================
    // Create weight mapping registry for export
    // ========================================================================

    // Get weight mapping from the registry using architecture ID
    auto mapping = models::create_weight_mapping(mConfig.architecture_id);

    // ========================================================================
    // Helper to compute row slices from marker values
    // Markers: -1=Q, -2=K, -3=V, -4=up, -5=gate
    // ========================================================================

    auto compute_slice = [&](long marker) -> std::pair<long, long> {
        switch (marker) {
            case -1: return {0, q_rows};                          // Q
            case -2: return {q_rows, q_rows + kv_rows};           // K
            case -3: return {q_rows + kv_rows, q_rows + 2 * kv_rows}; // V
            case -4: return {0, D};                                // up
            case -5: return {D, 2 * D};                           // gate
            default: return {0, 0};                               // full tensor
        }
    };

    // ========================================================================
    // Helper to resolve tensor from target
    // ========================================================================

    auto resolve_nonblock = [&](TensorTarget target) -> std::optional<Tensor> {
        switch (target) {
            case TensorTarget::Embeddings:
                return mWorkNonBlock.embeddings;
            case TensorTarget::FinalNorm:
                return mWorkNonBlock.final_norm_weight;
            case TensorTarget::LMHead:
                if (!mConfig.tied_embeddings) return mWorkNonBlock.lm_head;
                return std::nullopt;
            default:
                return std::nullopt;
        }
    };

    auto resolve_block = [&](TensorTarget target, int layer_idx, long marker) -> std::optional<Tensor> {
        if (layer_idx < 0 || layer_idx >= mConfig.num_layers) return std::nullopt;
        const auto& block = mWorkBlocks[layer_idx];

        auto [row_begin, row_end] = compute_slice(marker);

        switch (target) {
            case TensorTarget::LN1Weight:
                return block.ln1.weight;
            case TensorTarget::LN2Weight:
                return block.ln2.weight;
            case TensorTarget::QKVWeight:
                if (row_begin != row_end) {
                    return slice(block.attention.qkv_weight, 0, row_begin, row_end);
                }
                return block.attention.qkv_weight;
            case TensorTarget::QKVBias:
                if (block.attention.qkv_bias.has_value()) {
                    if (row_begin != row_end) {
                        return slice(block.attention.qkv_bias.value(), 0, row_begin, row_end);
                    }
                    return block.attention.qkv_bias.value();
                }
                return std::nullopt;
            case TensorTarget::OutWeight:
                return block.attention.out_weight;
            case TensorTarget::QNormWeight:
                if constexpr (has_qk_norm_weights<decltype(block.attention)>::value) {
                    if (block.attention.q_norm_weight.has_value()) {
                        return block.attention.q_norm_weight.value();
                    }
                }
                return std::nullopt;
            case TensorTarget::KNormWeight:
                if constexpr (has_qk_norm_weights<decltype(block.attention)>::value) {
                    if (block.attention.k_norm_weight.has_value()) {
                        return block.attention.k_norm_weight.value();
                    }
                }
                return std::nullopt;
            case TensorTarget::MLPUpWeight:
                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    if (row_begin != row_end) {
                        return slice(block.mlp_up_weight, 0, row_begin, row_end);
                    }
                    return block.mlp_up_weight;
                }
                return std::nullopt;
            case TensorTarget::MLPDownWeight:
                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    return block.mlp_down_weight;
                }
                return std::nullopt;

            // MoE targets
            case TensorTarget::RouterGate:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    return block.router.gate;
                }
                return std::nullopt;
            case TensorTarget::ExpertsGateUp:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.experts.use_batched) {
                        return block.experts.gate_up_proj;
                    }
                }
                return std::nullopt;
            case TensorTarget::ExpertsDown:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.experts.use_batched) {
                        return block.experts.down_proj;
                    }
                }
                return std::nullopt;
            case TensorTarget::SharedExpertGate:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.shared_expert.has_value()) {
                        return block.shared_expert->gate_proj;
                    }
                }
                return std::nullopt;
            case TensorTarget::SharedExpertUp:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.shared_expert.has_value()) {
                        return block.shared_expert->up_proj;
                    }
                }
                return std::nullopt;
            case TensorTarget::SharedExpertDown:
                if constexpr (has_moe_weights<BlockWeights>::value) {
                    if (block.shared_expert.has_value()) {
                        return block.shared_expert->down_proj;
                    }
                }
                return std::nullopt;

            default:
                return std::nullopt;
        }
    };

    // ========================================================================
    // Collect tensors for export
    // ========================================================================

    SafeTensorWriter writer{filename};

    // Register non-block tensors
    for (const auto& pattern : mapping->export_nonblock_patterns()) {
        auto tensor_opt = resolve_nonblock(pattern.source);
        if (tensor_opt.has_value()) {
            writer.register_tensor(pattern.hf_name_template, TensorShard(tensor_opt.value()));
        }
    }

    // Register per-layer tensors
    for (int layer_idx = 0; layer_idx < mConfig.num_layers; ++layer_idx) {
        for (const auto& pattern : mapping->export_layer_patterns()) {
            auto tensor_opt = resolve_block(pattern.source, layer_idx, pattern.row_begin);
            if (tensor_opt.has_value()) {
                std::string name = pattern.expand_name(layer_idx);
                writer.register_tensor(name, TensorShard(tensor_opt.value()));
            }
        }

        // Handle per-expert patterns for non-batched MoE
        if constexpr (has_moe_weights<BlockWeights>::value) {
            const auto& block = mWorkBlocks[layer_idx];
            if (!block.experts.use_batched) {
                std::string prefix = "model.layers." + std::to_string(layer_idx);
                for (int e = 0; e < static_cast<int>(block.experts.experts.size()); ++e) {
                    std::string exp_prefix = prefix + ".mlp.experts." + std::to_string(e);
                    const auto& expert = block.experts.experts[e];
                    writer.register_tensor(exp_prefix + ".gate_proj.weight", TensorShard(expert.gate_proj));
                    writer.register_tensor(exp_prefix + ".up_proj.weight", TensorShard(expert.up_proj));
                    writer.register_tensor(exp_prefix + ".down_proj.weight", TensorShard(expert.down_proj));
                }
            }
        }
    }

    writer.prepare_metadata(&comm);

    // ========================================================================
    // Write tensors
    // ========================================================================

    // Write non-block tensors
    for (const auto& pattern : mapping->export_nonblock_patterns()) {
        auto tensor_opt = resolve_nonblock(pattern.source);
        if (tensor_opt.has_value()) {
            writer.write_tensor(pattern.hf_name_template, TensorShard(tensor_opt.value()), &comm);
        }
    }

    // Write per-layer tensors
    for (int layer_idx = 0; layer_idx < mConfig.num_layers; ++layer_idx) {
        for (const auto& pattern : mapping->export_layer_patterns()) {
            auto tensor_opt = resolve_block(pattern.source, layer_idx, pattern.row_begin);
            if (tensor_opt.has_value()) {
                std::string name = pattern.expand_name(layer_idx);
                writer.write_tensor(name, TensorShard(tensor_opt.value()), &comm);
            }
        }

        // Handle per-expert patterns for non-batched MoE
        if constexpr (has_moe_weights<BlockWeights>::value) {
            const auto& block = mWorkBlocks[layer_idx];
            if (!block.experts.use_batched) {
                std::string prefix = "model.layers." + std::to_string(layer_idx);
                for (int e = 0; e < static_cast<int>(block.experts.experts.size()); ++e) {
                    std::string exp_prefix = prefix + ".mlp.experts." + std::to_string(e);
                    const auto& expert = block.experts.experts[e];
                    writer.write_tensor(exp_prefix + ".gate_proj.weight", TensorShard(expert.gate_proj), &comm);
                    writer.write_tensor(exp_prefix + ".up_proj.weight", TensorShard(expert.up_proj), &comm);
                    writer.write_tensor(exp_prefix + ".down_proj.weight", TensorShard(expert.down_proj), &comm);
                }
            }
        }
    }

    writer.finalize(&comm);
    comm.barrier();
}

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_IO_H
