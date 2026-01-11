// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Llama Weight Mapping - HuggingFace tensor name patterns for Llama models

#ifndef SUROGATE_SRC_MODELS_LLAMA_WEIGHT_MAPPING_H
#define SUROGATE_SRC_MODELS_LLAMA_WEIGHT_MAPPING_H

#include "modules/weights/weight_mapping.h"

namespace modules {

class LlamaWeightMapping : public BaseWeightMapping {
public:
    void register_patterns() override {
        // Non-block weights
        add_pattern("model.embed_tokens.weight", TensorTarget::Embeddings);
        add_pattern("model.norm.weight", TensorTarget::FinalNorm);
        add_pattern("lm_head.weight", TensorTarget::LMHead, nullptr, true);

        // Per-layer patterns - layer norms
        add_layer_pattern("model.layers.{layer}.input_layernorm.weight", TensorTarget::LN1Weight);
        add_layer_pattern("model.layers.{layer}.post_attention_layernorm.weight", TensorTarget::LN2Weight);

        // Attention - split Q/K/V (HuggingFace format) -> fused QKV
        add_layer_pattern("model.layers.{layer}.self_attn.q_proj.weight",
                          TensorTarget::QKVWeight, ranges::qkv_q_weight);
        add_layer_pattern("model.layers.{layer}.self_attn.k_proj.weight",
                          TensorTarget::QKVWeight, ranges::qkv_k_weight);
        add_layer_pattern("model.layers.{layer}.self_attn.v_proj.weight",
                          TensorTarget::QKVWeight, ranges::qkv_v_weight);
        add_layer_pattern("model.layers.{layer}.self_attn.o_proj.weight", TensorTarget::OutWeight);

        // Attention biases (optional)
        add_layer_pattern("model.layers.{layer}.self_attn.q_proj.bias",
                          TensorTarget::QKVBias, ranges::qkv_q_bias, true);
        add_layer_pattern("model.layers.{layer}.self_attn.k_proj.bias",
                          TensorTarget::QKVBias, ranges::qkv_k_bias, true);
        add_layer_pattern("model.layers.{layer}.self_attn.v_proj.bias",
                          TensorTarget::QKVBias, ranges::qkv_v_bias, true);

        // MLP - split gate/up (HuggingFace format) -> fused up
        add_layer_pattern("model.layers.{layer}.mlp.up_proj.weight",
                          TensorTarget::MLPUpWeight, ranges::mlp_up_weight);
        add_layer_pattern("model.layers.{layer}.mlp.gate_proj.weight",
                          TensorTarget::MLPUpWeight, ranges::mlp_gate_weight);
        add_layer_pattern("model.layers.{layer}.mlp.down_proj.weight", TensorTarget::MLPDownWeight);

        // Pre-fused patterns (internal format, direct copy)
        add_layer_pattern("model.layers.{layer}.self_attn.qkv.weight", TensorTarget::QKVWeight);
        add_layer_pattern("model.layers.{layer}.self_attn.qkv.bias", TensorTarget::QKVBias, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mlp.up.weight", TensorTarget::MLPUpWeight);
    }

    void register_export_patterns() override {
        // Non-block weights
        add_export_nonblock("model.embed_tokens.weight", TensorTarget::Embeddings);
        add_export_nonblock("model.norm.weight", TensorTarget::FinalNorm);
        add_export_nonblock("lm_head.weight", TensorTarget::LMHead, true);  // Optional if tied

        // Per-layer patterns - layer norms
        add_export_layer("model.layers.{layer}.input_layernorm.weight", TensorTarget::LN1Weight);
        add_export_layer("model.layers.{layer}.post_attention_layernorm.weight", TensorTarget::LN2Weight);

        // Attention - split fused QKV back to Q/K/V
        // Row slices are: Q=[0, q_rows), K=[q_rows, q_rows+kv_rows), V=[q_rows+kv_rows, q_rows+2*kv_rows)
        // We use special markers (-1, -2, -3) that will be computed at export time
        add_export_layer_slice("model.layers.{layer}.self_attn.q_proj.weight",
                               TensorTarget::QKVWeight, -1, -1);  // Q slice
        add_export_layer_slice("model.layers.{layer}.self_attn.k_proj.weight",
                               TensorTarget::QKVWeight, -2, -2);  // K slice
        add_export_layer_slice("model.layers.{layer}.self_attn.v_proj.weight",
                               TensorTarget::QKVWeight, -3, -3);  // V slice
        add_export_layer("model.layers.{layer}.self_attn.o_proj.weight", TensorTarget::OutWeight);

        // Attention biases (optional)
        add_export_layer_slice("model.layers.{layer}.self_attn.q_proj.bias",
                               TensorTarget::QKVBias, -1, -1, true);
        add_export_layer_slice("model.layers.{layer}.self_attn.k_proj.bias",
                               TensorTarget::QKVBias, -2, -2, true);
        add_export_layer_slice("model.layers.{layer}.self_attn.v_proj.bias",
                               TensorTarget::QKVBias, -3, -3, true);

        // MLP - split fused gate+up back to up and gate
        // Row slices: up=[0, D), gate=[D, 2*D)
        add_export_layer_slice("model.layers.{layer}.mlp.up_proj.weight",
                               TensorTarget::MLPUpWeight, -4, -4);  // up slice
        add_export_layer_slice("model.layers.{layer}.mlp.gate_proj.weight",
                               TensorTarget::MLPUpWeight, -5, -5);  // gate slice
        add_export_layer("model.layers.{layer}.mlp.down_proj.weight", TensorTarget::MLPDownWeight);
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_LLAMA_WEIGHT_MAPPING_H
