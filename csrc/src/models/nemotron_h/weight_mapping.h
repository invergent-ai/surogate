// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Nemotron-H Weight Mapping - HuggingFace tensor name patterns

#ifndef SUROGATE_SRC_MODELS_NEMOTRON_H_WEIGHT_MAPPING_H
#define SUROGATE_SRC_MODELS_NEMOTRON_H_WEIGHT_MAPPING_H

#include "modules/weights/weight_mapping.h"

namespace modules {

class NemotronHWeightMapping : public BaseWeightMapping {
public:
    void register_patterns() override {
        // Non-block weights
        add_pattern("model.embeddings.weight", TensorTarget::Embeddings);
        add_pattern("backbone.embeddings.weight", TensorTarget::Embeddings);

        add_pattern("model.norm_f.weight", TensorTarget::FinalNorm);
        add_pattern("backbone.norm_f.weight", TensorTarget::FinalNorm);

        add_pattern("lm_head.weight", TensorTarget::LMHead, nullptr, true);

        // Per-layer norm (single RMSNorm)
        add_layer_pattern("model.layers.{layer}.norm.weight", TensorTarget::LN1Weight);
        add_layer_pattern("backbone.layers.{layer}.norm.weight", TensorTarget::LN1Weight);

        // Attention (fused QKV)
        add_layer_pattern("model.layers.{layer}.mixer.q_proj.weight",
                          TensorTarget::QKVWeight, ranges::qkv_q_weight);
        add_layer_pattern("model.layers.{layer}.mixer.k_proj.weight",
                          TensorTarget::QKVWeight, ranges::qkv_k_weight);
        add_layer_pattern("model.layers.{layer}.mixer.v_proj.weight",
                          TensorTarget::QKVWeight, ranges::qkv_v_weight);
        add_layer_pattern("model.layers.{layer}.mixer.o_proj.weight", TensorTarget::OutWeight);

        add_layer_pattern("backbone.layers.{layer}.mixer.q_proj.weight",
                          TensorTarget::QKVWeight, ranges::qkv_q_weight);
        add_layer_pattern("backbone.layers.{layer}.mixer.k_proj.weight",
                          TensorTarget::QKVWeight, ranges::qkv_k_weight);
        add_layer_pattern("backbone.layers.{layer}.mixer.v_proj.weight",
                          TensorTarget::QKVWeight, ranges::qkv_v_weight);
        add_layer_pattern("backbone.layers.{layer}.mixer.o_proj.weight", TensorTarget::OutWeight);

        // Attention biases (optional)
        add_layer_pattern("model.layers.{layer}.mixer.q_proj.bias",
                          TensorTarget::QKVBias, ranges::qkv_q_bias, true);
        add_layer_pattern("model.layers.{layer}.mixer.k_proj.bias",
                          TensorTarget::QKVBias, ranges::qkv_k_bias, true);
        add_layer_pattern("model.layers.{layer}.mixer.v_proj.bias",
                          TensorTarget::QKVBias, ranges::qkv_v_bias, true);

        add_layer_pattern("backbone.layers.{layer}.mixer.q_proj.bias",
                          TensorTarget::QKVBias, ranges::qkv_q_bias, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.k_proj.bias",
                          TensorTarget::QKVBias, ranges::qkv_k_bias, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.v_proj.bias",
                          TensorTarget::QKVBias, ranges::qkv_v_bias, true);

        // MLP (non-gated)
        add_layer_pattern("model.layers.{layer}.mixer.up_proj.weight", TensorTarget::MLPUpWeight);
        add_layer_pattern("model.layers.{layer}.mixer.down_proj.weight", TensorTarget::MLPDownWeight);
        add_layer_pattern("backbone.layers.{layer}.mixer.up_proj.weight", TensorTarget::MLPUpWeight);
        add_layer_pattern("backbone.layers.{layer}.mixer.down_proj.weight", TensorTarget::MLPDownWeight);

        // Mamba (Nemotron-H)
        add_layer_pattern("model.layers.{layer}.mixer.in_proj.weight", TensorTarget::MambaInProjWeight);
        add_layer_pattern("model.layers.{layer}.mixer.in_proj.bias", TensorTarget::MambaInProjBias, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.out_proj.weight", TensorTarget::MambaOutProjWeight);
        add_layer_pattern("model.layers.{layer}.mixer.out_proj.bias", TensorTarget::MambaOutProjBias, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.conv1d.weight", TensorTarget::MambaConv1dWeight);
        add_layer_pattern("model.layers.{layer}.mixer.conv1d.bias", TensorTarget::MambaConv1dBias, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.A_log", TensorTarget::MambaALog);
        add_layer_pattern("model.layers.{layer}.mixer.D", TensorTarget::MambaD);
        add_layer_pattern("model.layers.{layer}.mixer.dt_bias", TensorTarget::MambaDtBias);
        add_layer_pattern("model.layers.{layer}.mixer.norm.weight", TensorTarget::MambaNormWeight);

        add_layer_pattern("backbone.layers.{layer}.mixer.in_proj.weight", TensorTarget::MambaInProjWeight);
        add_layer_pattern("backbone.layers.{layer}.mixer.in_proj.bias", TensorTarget::MambaInProjBias, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.out_proj.weight", TensorTarget::MambaOutProjWeight);
        add_layer_pattern("backbone.layers.{layer}.mixer.out_proj.bias", TensorTarget::MambaOutProjBias, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.conv1d.weight", TensorTarget::MambaConv1dWeight);
        add_layer_pattern("backbone.layers.{layer}.mixer.conv1d.bias", TensorTarget::MambaConv1dBias, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.A_log", TensorTarget::MambaALog);
        add_layer_pattern("backbone.layers.{layer}.mixer.D", TensorTarget::MambaD);
        add_layer_pattern("backbone.layers.{layer}.mixer.dt_bias", TensorTarget::MambaDtBias);
        add_layer_pattern("backbone.layers.{layer}.mixer.norm.weight", TensorTarget::MambaNormWeight);
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_NEMOTRON_H_WEIGHT_MAPPING_H
