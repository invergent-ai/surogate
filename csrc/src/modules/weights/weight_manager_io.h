// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

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
#include "modules/weights/weight_manager_helpers.h"
#include "utilities/comm.h"
#include "utilities/philox.h"
#include "utilities/safetensors.h"

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
        if (block.attention.q_norm_weight.has_value()) {
            callback(prefix + ".self_attn.q_norm.weight", shard(block.attention.q_norm_weight.value(), {HS}));
        }
        if (block.attention.k_norm_weight.has_value()) {
            callback(prefix + ".self_attn.k_norm.weight", shard(block.attention.k_norm_weight.value(), {HS}));
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
        if (layer.attention.q_norm_weight.has_value()) {
            fill_constant(layer.attention.q_norm_weight.value(), 1.f, layer.attention.q_norm_weight->nelem(), nullptr);
        }
        if (layer.attention.k_norm_weight.has_value()) {
            fill_constant(layer.attention.k_norm_weight.value(), 1.f, layer.attention.k_norm_weight->nelem(), nullptr);
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
}

template<typename Block>
void ModularWeightManager<Block>::import_from_file(const std::string& filename, bool allow_cast, NCCLCommunicator& comm) {
    SafeTensorsReader reader{filename};

    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;

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

    auto dst = [&](Tensor& t, const std::vector<long>& global_shape) -> TensorShard {
        if (load_num == 1) return TensorShard(t);
        return TensorShard(t, load_idx, load_num, global_shape);
    };

    // Build name -> destination mapping for direct matches.
    std::unordered_map<std::string, TensorShard> named_tensors;
    named_tensors.emplace("model.embed_tokens.weight", dst(dst_nonblock.embeddings, {mConfig.vocab_size, mConfig.hidden_size}));
    named_tensors.emplace("model.norm.weight", dst(dst_nonblock.final_norm_weight, {mConfig.hidden_size}));
    named_tensors.emplace("lm_head.weight", dst(dst_nonblock.lm_head, {mConfig.vocab_size, mConfig.hidden_size}));

    for (int i = 0; i < mConfig.num_layers; ++i) {
        auto& block = dst_blocks.at(i);
        std::string prefix = "model.layers." + std::to_string(i);

        named_tensors.emplace(prefix + ".input_layernorm.weight", dst(block.ln1.weight, {C}));
        named_tensors.emplace(prefix + ".post_attention_layernorm.weight", dst(block.ln2.weight, {C}));
        named_tensors.emplace(prefix + ".self_attn.qkv.weight", dst(block.attention.qkv_weight, {fused_rows, C}));
        if (block.attention.qkv_bias.has_value()) {
            named_tensors.emplace(prefix + ".self_attn.qkv.bias", dst(block.attention.qkv_bias.value(), {fused_rows}));
        }
        named_tensors.emplace(prefix + ".self_attn.o_proj.weight", dst(block.attention.out_weight, {C, q_rows}));
        if (block.attention.q_norm_weight.has_value()) {
            named_tensors.emplace(prefix + ".self_attn.q_norm.weight", dst(block.attention.q_norm_weight.value(), {HS}));
        }
        if (block.attention.k_norm_weight.has_value()) {
            named_tensors.emplace(prefix + ".self_attn.k_norm.weight", dst(block.attention.k_norm_weight.value(), {HS}));
        }

        if constexpr (has_mlp_weights<BlockWeights>::value) {
            named_tensors.emplace(prefix + ".mlp.up.weight", dst(block.mlp_up_weight, {2 * D, C}));
            named_tensors.emplace(prefix + ".mlp.down_proj.weight", dst(block.mlp_down_weight, {C, D}));
        }
    }

    for (const auto& entry : reader.entries()) {
        const std::string& name = entry.name();

        if (auto found = named_tensors.find(name); found != named_tensors.end()) {
            load_intersect(found->second, entry, 0, (std::ptrdiff_t)found->second.global_nelem(), allow_cast);
            continue;
        }

        // Handle split Q/K/V and gate/up projections from HuggingFace format
        if (name.starts_with("model.layers.")) {
            std::size_t chars = 0;
            auto layer_idx = std::stoi(name.c_str() + 13, &chars);
            std::string suffix = name.substr(13 + chars);
            auto& block = mMasterBlocks.at(layer_idx);

            TensorShard qkv_w = TensorShard(block.attention.qkv_weight, mConfig.shard_idx, mConfig.num_shards,
                                            std::vector<long>{fused_rows, C});
            std::optional<TensorShard> qkv_b{};
            if (block.attention.qkv_bias.has_value()) {
                qkv_b = TensorShard(block.attention.qkv_bias.value(), mConfig.shard_idx, mConfig.num_shards,
                                    std::vector<long>{fused_rows});
            }
            TensorShard mlp_up{};
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                mlp_up = TensorShard(block.mlp_up_weight, mConfig.shard_idx, mConfig.num_shards,
                                     std::vector<long>{2 * D, C});
            }

            // Global split positions in fused tensors (in elements).
            const std::ptrdiff_t q_begin = 0;
            const std::ptrdiff_t k_begin = q_rows * C;
            const std::ptrdiff_t v_begin = (q_rows + kv_rows) * C;
            const std::ptrdiff_t q_end = q_rows * C;
            const std::ptrdiff_t k_end = (q_rows + kv_rows) * C;
            const std::ptrdiff_t v_end = fused_rows * C;

            if (suffix == ".self_attn.q_proj.weight") {
                load_intersect(qkv_w, entry, q_begin, q_end, allow_cast);
            } else if (suffix == ".self_attn.k_proj.weight") {
                load_intersect(qkv_w, entry, k_begin, k_end, allow_cast);
            } else if (suffix == ".self_attn.v_proj.weight") {
                load_intersect(qkv_w, entry, v_begin, v_end, allow_cast);
            } else if (suffix == ".self_attn.q_proj.bias") {
                if (qkv_b.has_value()) load_intersect(qkv_b.value(), entry, 0, q_rows, allow_cast);
            } else if (suffix == ".self_attn.k_proj.bias") {
                if (qkv_b.has_value()) load_intersect(qkv_b.value(), entry, q_rows, q_rows + kv_rows, allow_cast);
            } else if (suffix == ".self_attn.v_proj.bias") {
                if (qkv_b.has_value()) load_intersect(qkv_b.value(), entry, q_rows + kv_rows, fused_rows, allow_cast);
            } else if (suffix == ".mlp.up_proj.weight") {
                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    load_intersect(mlp_up, entry, 0, D * C, allow_cast);
                }
            } else if (suffix == ".mlp.gate_proj.weight") {
                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    load_intersect(mlp_up, entry, D * C, 2 * D * C, allow_cast);
                }
            } else {
                // For other / MoE tensors, skip.
            }
        } else {
            throw std::runtime_error("Unexpected tensor name: " + name);
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

    SafeTensorWriter writer{filename};

    // Register non-split tensors.
    writer.register_tensor("model.embed_tokens.weight", TensorShard(mWorkNonBlock.embeddings));
    writer.register_tensor("model.norm.weight", TensorShard(mWorkNonBlock.final_norm_weight));
    if (!mConfig.tied_embeddings) {
        writer.register_tensor("lm_head.weight", TensorShard(mWorkNonBlock.lm_head));
    }

    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;

    // Register block tensors in HuggingFace format.
    for (int i = 0; i < mConfig.num_layers; ++i) {
        const auto& block = mWorkBlocks[i];
        std::string prefix = "model.layers." + std::to_string(i);

        const long q_rows = HS * HQ;
        const long kv_rows = HS * HKV;
        const long fused_rows = q_rows + 2 * kv_rows;

        writer.register_tensor(prefix + ".input_layernorm.weight", TensorShard(block.ln1.weight));
        writer.register_tensor(prefix + ".post_attention_layernorm.weight", TensorShard(block.ln2.weight));
        writer.register_tensor(prefix + ".self_attn.o_proj.weight", TensorShard(block.attention.out_weight));
        if (block.attention.q_norm_weight.has_value()) {
            writer.register_tensor(prefix + ".self_attn.q_norm.weight", TensorShard(block.attention.q_norm_weight.value()));
        }
        if (block.attention.k_norm_weight.has_value()) {
            writer.register_tensor(prefix + ".self_attn.k_norm.weight", TensorShard(block.attention.k_norm_weight.value()));
        }

        // Split QKV from fused tensor.
        writer.register_tensor(prefix + ".self_attn.q_proj.weight",
                               TensorShard(slice(block.attention.qkv_weight, 0, 0, q_rows)));
        writer.register_tensor(prefix + ".self_attn.k_proj.weight",
                               TensorShard(slice(block.attention.qkv_weight, 0, q_rows, q_rows + kv_rows)));
        writer.register_tensor(prefix + ".self_attn.v_proj.weight",
                               TensorShard(slice(block.attention.qkv_weight, 0, q_rows + kv_rows, fused_rows)));

        if (block.attention.qkv_bias.has_value()) {
            const auto& bias = block.attention.qkv_bias.value();
            writer.register_tensor(prefix + ".self_attn.q_proj.bias", TensorShard(slice(bias, 0, 0, q_rows)));
            writer.register_tensor(prefix + ".self_attn.k_proj.bias", TensorShard(slice(bias, 0, q_rows, q_rows + kv_rows)));
            writer.register_tensor(prefix + ".self_attn.v_proj.bias", TensorShard(slice(bias, 0, q_rows + kv_rows, fused_rows)));
        }

        if constexpr (has_mlp_weights<BlockWeights>::value) {
            writer.register_tensor(prefix + ".mlp.up_proj.weight", TensorShard(slice(block.mlp_up_weight, 0, 0, D)));
            writer.register_tensor(prefix + ".mlp.gate_proj.weight", TensorShard(slice(block.mlp_up_weight, 0, D, 2 * D)));
            writer.register_tensor(prefix + ".mlp.down_proj.weight", TensorShard(block.mlp_down_weight));
        }
    }

    writer.prepare_metadata(&comm);

    // Write non-block tensors
    writer.write_tensor("model.embed_tokens.weight", TensorShard(mWorkNonBlock.embeddings), &comm);
    writer.write_tensor("model.norm.weight", TensorShard(mWorkNonBlock.final_norm_weight), &comm);
    if (!mConfig.tied_embeddings) {
        writer.write_tensor("lm_head.weight", TensorShard(mWorkNonBlock.lm_head), &comm);
    }

    // Write block tensors
    for (int i = 0; i < mConfig.num_layers; ++i) {
        const auto& block = mWorkBlocks[i];
        std::string prefix = "model.layers." + std::to_string(i);

        const long q_rows = HS * HQ;
        const long kv_rows = HS * HKV;
        const long fused_rows = q_rows + 2 * kv_rows;

        writer.write_tensor(prefix + ".input_layernorm.weight", TensorShard(block.ln1.weight), &comm);
        writer.write_tensor(prefix + ".post_attention_layernorm.weight", TensorShard(block.ln2.weight), &comm);
        writer.write_tensor(prefix + ".self_attn.o_proj.weight", TensorShard(block.attention.out_weight), &comm);
        if (block.attention.q_norm_weight.has_value()) {
            writer.write_tensor(prefix + ".self_attn.q_norm.weight", TensorShard(block.attention.q_norm_weight.value()), &comm);
        }
        if (block.attention.k_norm_weight.has_value()) {
            writer.write_tensor(prefix + ".self_attn.k_norm.weight", TensorShard(block.attention.k_norm_weight.value()), &comm);
        }

        writer.write_tensor(prefix + ".self_attn.q_proj.weight",
                            TensorShard(slice(block.attention.qkv_weight, 0, 0, q_rows)), &comm);
        writer.write_tensor(prefix + ".self_attn.k_proj.weight",
                            TensorShard(slice(block.attention.qkv_weight, 0, q_rows, q_rows + kv_rows)), &comm);
        writer.write_tensor(prefix + ".self_attn.v_proj.weight",
                            TensorShard(slice(block.attention.qkv_weight, 0, q_rows + kv_rows, fused_rows)), &comm);

        if (block.attention.qkv_bias.has_value()) {
            const auto& bias = block.attention.qkv_bias.value();
            writer.write_tensor(prefix + ".self_attn.q_proj.bias", TensorShard(slice(bias, 0, 0, q_rows)), &comm);
            writer.write_tensor(prefix + ".self_attn.k_proj.bias", TensorShard(slice(bias, 0, q_rows, q_rows + kv_rows)), &comm);
            writer.write_tensor(prefix + ".self_attn.v_proj.bias", TensorShard(slice(bias, 0, q_rows + kv_rows, fused_rows)), &comm);
        }

        if constexpr (has_mlp_weights<BlockWeights>::value) {
            writer.write_tensor(prefix + ".mlp.up_proj.weight", TensorShard(slice(block.mlp_up_weight, 0, 0, D)), &comm);
            writer.write_tensor(prefix + ".mlp.gate_proj.weight", TensorShard(slice(block.mlp_up_weight, 0, D, 2 * D)), &comm);
            writer.write_tensor(prefix + ".mlp.down_proj.weight", TensorShard(block.mlp_down_weight), &comm);
        }
    }

    writer.finalize(&comm);
    comm.barrier();
}

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_IO_H
