// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_optimizer_state.h"

#include <algorithm>
#include <cstring>
#include <fmt/format.h>

#include "kernels/kernels.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"

namespace modules {

ModularLoRAOptimizerState::ModularLoRAOptimizerState(const Config& config, cudaStream_t stream,
                                                            NCCLCommunicator& comm, TensorAllocator& allocator)
    : mConfig(config), mAllocator(&allocator) {
    mMomentum.config = config.lora_config;
    mVariance.config = config.lora_config;
    mMomentumScales.config = config.lora_config;
    mVarianceScales.config = config.lora_config;

    if (!config.lora_config.enabled()) return;

    allocate_state();

    auto zero_layer = [stream](auto& opt_layer) {
        if (!opt_layer.has_value()) return;
        if (opt_layer->A.Data) {
            if (opt_layer->A.Device < 0) {
                std::memset(opt_layer->A.Data, 0, opt_layer->A.bytes());
            } else {
                fill_zero(opt_layer->A, stream);
            }
        }
        if (opt_layer->B.Data) {
            if (opt_layer->B.Device < 0) {
                std::memset(opt_layer->B.Data, 0, opt_layer->B.bytes());
            } else {
                fill_zero(opt_layer->B, stream);
            }
        }
    };

    for (auto& block : mMomentum.blocks) {
        zero_layer(block.attention.q);
        zero_layer(block.attention.k);
        zero_layer(block.attention.v);
        zero_layer(block.attention.o);
        zero_layer(block.mlp.gate);
        zero_layer(block.mlp.up);
        zero_layer(block.mlp.down);
    }
    for (auto& block : mVariance.blocks) {
        zero_layer(block.attention.q);
        zero_layer(block.attention.k);
        zero_layer(block.attention.v);
        zero_layer(block.attention.o);
        zero_layer(block.mlp.gate);
        zero_layer(block.mlp.up);
        zero_layer(block.mlp.down);
    }

    // Staging buffers are always device-resident.
    // Make sure any device-side zeroing has completed before proceeding.
    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();
}

ModularLoRAOptimizerState::~ModularLoRAOptimizerState() = default;

void ModularLoRAOptimizerState::allocate_state() {
    auto ctx = mAllocator->with_context("Modular_LoRA_OptState");
    mMomentum.blocks.resize(mConfig.num_layers);
    mVariance.blocks.resize(mConfig.num_layers);
    mMomentumScales.blocks.resize(mConfig.num_layers);
    mVarianceScales.blocks.resize(mConfig.num_layers);

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int q_out = mConfig.num_query_heads * mConfig.head_size;
    const int kv_out = mConfig.num_kv_heads * mConfig.head_size;
    const int r = mConfig.lora_config.rank;

    const EAllocationType kind_m = mConfig.offload_m ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;
    const EAllocationType kind_v = mConfig.offload_v ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;

    auto alloc_state = [&](ETensorDType dtype, EAllocationType kind, int in_f, int out_f, const std::string& name) -> LoRALayerWeights<TensorShard> {
        LoRALayerWeights<TensorShard> w;
        w.A = TensorShard(mAllocator->allocate(dtype, (name + "_A").c_str(), kind, {r, in_f}));
        w.B = mAllocator->allocate_shard(dtype, /*shard_idx=*/0, /*num_shards=*/1, (name + "_B").c_str(), {out_f, r}, kind);
        return w;
    };

    auto alloc_scales = [&](EAllocationType kind, int in_f, int out_f, const std::string& name) -> LoRALayerWeights<TensorShard> {
        LoRALayerWeights<TensorShard> w;
        const long a_elems = static_cast<long>(r) * static_cast<long>(in_f);
        const long b_elems = static_cast<long>(out_f) * static_cast<long>(r);
        const long a_blocks = div_ceil(a_elems, 128L);
        const long b_blocks = div_ceil(b_elems, 128L);
        w.A = TensorShard(mAllocator->allocate(ETensorDType::FP32, (name + "_A").c_str(), kind, {a_blocks}));
        w.B = TensorShard(mAllocator->allocate(ETensorDType::FP32, (name + "_B").c_str(), kind, {b_blocks}));
        return w;
    };

    for (int l = 0; l < mConfig.num_layers; ++l) {
        std::string prefix = fmt::format("lora_opt_layer_{}", l);
        auto& m = mMomentum.blocks[l];
        auto& v = mVariance.blocks[l];

        if (mConfig.lora_config.applies_to_q()) {
            m.attention.q = alloc_state(mConfig.m_dtype, kind_m, C, q_out, prefix + "_q_m");
            v.attention.q = alloc_state(mConfig.v_dtype, kind_v, C, q_out, prefix + "_q_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].attention.q = alloc_scales(kind_m, C, q_out, prefix + "_q_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].attention.q = alloc_scales(kind_v, C, q_out, prefix + "_q_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_k()) {
            m.attention.k = alloc_state(mConfig.m_dtype, kind_m, C, kv_out, prefix + "_k_m");
            v.attention.k = alloc_state(mConfig.v_dtype, kind_v, C, kv_out, prefix + "_k_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].attention.k = alloc_scales(kind_m, C, kv_out, prefix + "_k_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].attention.k = alloc_scales(kind_v, C, kv_out, prefix + "_k_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_v()) {
            m.attention.v = alloc_state(mConfig.m_dtype, kind_m, C, kv_out, prefix + "_v_m");
            v.attention.v = alloc_state(mConfig.v_dtype, kind_v, C, kv_out, prefix + "_v_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].attention.v = alloc_scales(kind_m, C, kv_out, prefix + "_v_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].attention.v = alloc_scales(kind_v, C, kv_out, prefix + "_v_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_o()) {
            m.attention.o = alloc_state(mConfig.m_dtype, kind_m, q_out, C, prefix + "_o_m");
            v.attention.o = alloc_state(mConfig.v_dtype, kind_v, q_out, C, prefix + "_o_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].attention.o = alloc_scales(kind_m, q_out, C, prefix + "_o_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].attention.o = alloc_scales(kind_v, q_out, C, prefix + "_o_v_scales");
            }
        }

        if (mConfig.lora_config.applies_to_gate()) {
            m.mlp.gate = alloc_state(mConfig.m_dtype, kind_m, C, D, prefix + "_gate_m");
            v.mlp.gate = alloc_state(mConfig.v_dtype, kind_v, C, D, prefix + "_gate_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].mlp.gate = alloc_scales(kind_m, C, D, prefix + "_gate_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].mlp.gate = alloc_scales(kind_v, C, D, prefix + "_gate_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_up()) {
            m.mlp.up = alloc_state(mConfig.m_dtype, kind_m, C, D, prefix + "_up_m");
            v.mlp.up = alloc_state(mConfig.v_dtype, kind_v, C, D, prefix + "_up_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].mlp.up = alloc_scales(kind_m, C, D, prefix + "_up_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].mlp.up = alloc_scales(kind_v, C, D, prefix + "_up_v_scales");
            }
        }
        if (mConfig.lora_config.applies_to_down()) {
            m.mlp.down = alloc_state(mConfig.m_dtype, kind_m, D, C, prefix + "_down_m");
            v.mlp.down = alloc_state(mConfig.v_dtype, kind_v, D, C, prefix + "_down_v");
            if (is_fp8_dtype(mConfig.m_dtype)) {
                mMomentumScales.blocks[l].mlp.down = alloc_scales(kind_m, D, C, prefix + "_down_m_scales");
            }
            if (is_fp8_dtype(mConfig.v_dtype)) {
                mVarianceScales.blocks[l].mlp.down = alloc_scales(kind_v, D, C, prefix + "_down_v_scales");
            }
        }
    }

    // Allocate device staging buffers when host offload is enabled.
    // Sized for the largest LoRA moment tensor (A or B) across modules.
    const int max_features = std::max({C, D, q_out, kv_out});
    const long max_elems = static_cast<long>(r) * static_cast<long>(max_features);
    const long max_scale_elems = div_ceil(max_elems, 128L);
    if (mConfig.offload_m && !mStagingM.Data) {
        mStagingM = mAllocator->allocate(mConfig.m_dtype, "lora_opt_m_stage", EAllocationType::ON_DEVICE, {max_elems});
    }
    if (mConfig.offload_v && !mStagingV.Data) {
        mStagingV = mAllocator->allocate(mConfig.v_dtype, "lora_opt_v_stage", EAllocationType::ON_DEVICE, {max_elems});
    }
    if (mConfig.offload_m && is_fp8_dtype(mConfig.m_dtype) && !mStagingMScales.Data) {
        mStagingMScales = mAllocator->allocate(ETensorDType::FP32, "lora_opt_m_scales_stage", EAllocationType::ON_DEVICE, {max_scale_elems});
    }
    if (mConfig.offload_v && is_fp8_dtype(mConfig.v_dtype) && !mStagingVScales.Data) {
        mStagingVScales = mAllocator->allocate(ETensorDType::FP32, "lora_opt_v_scales_stage", EAllocationType::ON_DEVICE, {max_scale_elems});
    }
}

LoRABlockWeights<TensorShard>& ModularLoRAOptimizerState::get_block_m(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mMomentum.blocks[layer_idx];
}

LoRABlockWeights<TensorShard>& ModularLoRAOptimizerState::get_block_v(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mVariance.blocks[layer_idx];
}

LoRABlockWeights<TensorShard>& ModularLoRAOptimizerState::get_block_m_scales(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mMomentumScales.blocks[layer_idx];
}

LoRABlockWeights<TensorShard>& ModularLoRAOptimizerState::get_block_v_scales(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mVarianceScales.blocks[layer_idx];
}

ITensorContainer& ModularLoRAOptimizerState::full_m() {
    return mMomentumContainer;
}

ITensorContainer& ModularLoRAOptimizerState::full_v() {
    return mVarianceContainer;
}

ITensorContainer& ModularLoRAOptimizerState::full_m_scales() {
    return mMomentumScalesContainer;
}

ITensorContainer& ModularLoRAOptimizerState::full_v_scales() {
    return mVarianceScalesContainer;
}

void ModularLoRAOptimizerState::StateContainer::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {
    if (!mSet) return;

    for (int l = 0; l < (int)mSet->blocks.size(); ++l) {
        std::string prefix = fmt::format("base_model.model.model.layers.{}", l);
        auto& block = mSet->blocks[l];

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

        // MoE expert LoRA
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
