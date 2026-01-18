// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Declarative builder for composable transformer block specs.

#ifndef SUROGATE_SRC_MODULES_COMPOSITE_BLOCK_BUILDER_H
#define SUROGATE_SRC_MODULES_COMPOSITE_BLOCK_BUILDER_H

#include "block_spec.h"
#include "modules/model_config.h"

namespace modules {

struct BlockBuilder {
    static BlockSpec build(const ModelConfig& config, int layer_idx) {
        switch (config.get_block_type(layer_idx)) {
            case BlockType::MoE:
            case BlockType::SwitchMoE:
                return moe(config);
            case BlockType::Attention:
                return attention_only(config);
            case BlockType::MLP:
                return mlp_only(config);
            case BlockType::Mamba:
                return mamba(config);
            case BlockType::Dense:
            case BlockType::Conv:
            default:
                return config.use_parallel_residual ? parallel(config) : dense(config);
        }
    }

    static BlockSpec dense(const ModelConfig& config) {
        BlockSpec spec;
        spec.variant = BlockVariant::Dense;
        spec.use_qk_norm = config.use_qk_norm;
        spec.ln2_on_residual_att = true;

        spec.push_op(BlockOp::LN1);
        spec.push_op(BlockOp::QKV);
        if (spec.use_qk_norm) {
            spec.push_op(BlockOp::QKNorm);
        }
        spec.push_op(BlockOp::RoPE);
        spec.push_op(BlockOp::Attention);
        spec.push_op(BlockOp::AttnOut);
        spec.push_op(BlockOp::ResidualLN2);
        spec.push_op(BlockOp::MLPUp);
        if (is_gated_activation(config.activation_type)) {
            spec.push_op(BlockOp::SwiGLU);
        } else {
            spec.push_op(BlockOp::MLPAct);
        }
        spec.push_op(BlockOp::MLPDown);
        return spec;
    }

    static BlockSpec parallel(const ModelConfig& config) {
        BlockSpec spec;
        spec.variant = BlockVariant::Parallel;
        spec.use_qk_norm = config.use_qk_norm;
        spec.ln2_on_residual_att = false;

        spec.push_op(BlockOp::LN1);
        spec.push_op(BlockOp::QKV);
        if (spec.use_qk_norm) {
            spec.push_op(BlockOp::QKNorm);
        }
        spec.push_op(BlockOp::RoPE);
        spec.push_op(BlockOp::Attention);
        spec.push_op(BlockOp::AttnOut);
        spec.push_op(BlockOp::ResidualAdd);
        spec.push_op(BlockOp::LN2);
        spec.push_op(BlockOp::MLPUp);
        if (is_gated_activation(config.activation_type)) {
            spec.push_op(BlockOp::SwiGLU);
        } else {
            spec.push_op(BlockOp::MLPAct);
        }
        spec.push_op(BlockOp::MLPDown);
        return spec;
    }

    static BlockSpec attention_only(const ModelConfig& config) {
        BlockSpec spec;
        spec.variant = BlockVariant::Dense;
        spec.use_qk_norm = config.use_qk_norm;
        spec.ln2_on_residual_att = false;

        spec.push_op(BlockOp::LN1);
        spec.push_op(BlockOp::QKV);
        if (spec.use_qk_norm) {
            spec.push_op(BlockOp::QKNorm);
        }
        spec.push_op(BlockOp::RoPE);
        spec.push_op(BlockOp::Attention);
        spec.push_op(BlockOp::AttnOut);
        spec.push_op(BlockOp::ResidualAdd);
        return spec;
    }

    static BlockSpec mlp_only(const ModelConfig& config) {
        BlockSpec spec;
        spec.variant = BlockVariant::Dense;
        spec.use_qk_norm = false;
        spec.ln2_on_residual_att = false;

        spec.push_op(BlockOp::LN1);
        spec.push_op(BlockOp::MLPUp);
        if (is_gated_activation(config.activation_type)) {
            spec.push_op(BlockOp::SwiGLU);
        } else {
            spec.push_op(BlockOp::MLPAct);
        }
        spec.push_op(BlockOp::MLPDown);
        spec.push_op(BlockOp::ResidualAdd);
        return spec;
    }

    static BlockSpec mamba(const ModelConfig& config) {
        BlockSpec spec;
        spec.variant = BlockVariant::Dense;
        spec.use_qk_norm = false;
        spec.ln2_on_residual_att = false;

        spec.push_op(BlockOp::LN1);
        spec.push_op(BlockOp::Mamba);
        spec.push_op(BlockOp::ResidualAdd);
        return spec;
    }

    static BlockSpec moe(const ModelConfig& config) {
        BlockSpec spec;
        spec.variant = BlockVariant::MoE;
        spec.use_qk_norm = config.use_qk_norm;
        spec.ln2_on_residual_att = true;

        spec.push_op(BlockOp::LN1);
        spec.push_op(BlockOp::QKV);
        if (spec.use_qk_norm) {
            spec.push_op(BlockOp::QKNorm);
        }
        spec.push_op(BlockOp::RoPE);
        spec.push_op(BlockOp::Attention);
        spec.push_op(BlockOp::AttnOut);
        spec.push_op(BlockOp::ResidualLN2);
        spec.push_op(BlockOp::Router);
        spec.push_op(BlockOp::Experts);
        spec.push_op(BlockOp::Combine);
        return spec;
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_COMPOSITE_BLOCK_BUILDER_H
