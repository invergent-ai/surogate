// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3 MoE Weight Mapping - HuggingFace tensor name patterns for Qwen3 MoE models

#ifndef SUROGATE_SRC_MODELS_QWEN3MOE_WEIGHT_MAPPING_H
#define SUROGATE_SRC_MODELS_QWEN3MOE_WEIGHT_MAPPING_H

#include "models/qwen3/weight_mapping.h"

namespace modules {

class Qwen3MoEWeightMapping : public Qwen3WeightMapping {
public:
    void register_patterns() override {
        Qwen3WeightMapping::register_patterns();

        // Router weights (both naming conventions)
        add_layer_pattern("model.layers.{layer}.mlp.gate.weight", TensorTarget::RouterGate);
        add_layer_pattern("model.layers.{layer}.mlp.router.gate.weight", TensorTarget::RouterGate);

        // Batched expert weights (pre-fused format)
        add_layer_pattern("model.layers.{layer}.mlp.experts.gate_up_proj.weight",
                          TensorTarget::ExpertsGateUp);
        add_layer_pattern("model.layers.{layer}.mlp.experts.down_proj.weight",
                          TensorTarget::ExpertsDown);

        // Per-expert weights (HuggingFace non-batched format)
        add_expert_pattern("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                           TensorTarget::ExpertGate);
        add_expert_pattern("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                           TensorTarget::ExpertUp);
        add_expert_pattern("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                           TensorTarget::ExpertDown);

        // Shared expert (optional)
        add_layer_pattern("model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                          TensorTarget::SharedExpertUp, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                          TensorTarget::SharedExpertDown, nullptr, true);

        // Fused shared expert pattern
        add_layer_pattern("model.layers.{layer}.mlp.shared_expert.gate_up_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);  // Reuse gate target for fused
    }

    void register_export_patterns() override {
        // Note: MoE layers don't have regular MLP patterns, so we don't call parent
        // Instead we register MoE-specific patterns

        // Non-block weights (from Llama)
        add_export_nonblock("model.embed_tokens.weight", TensorTarget::Embeddings);
        add_export_nonblock("model.norm.weight", TensorTarget::FinalNorm);
        add_export_nonblock("lm_head.weight", TensorTarget::LMHead, true);

        // Layer norms
        add_export_layer("model.layers.{layer}.input_layernorm.weight", TensorTarget::LN1Weight);
        add_export_layer("model.layers.{layer}.post_attention_layernorm.weight", TensorTarget::LN2Weight);

        // QKV split (same as Llama)
        add_export_layer_slice("model.layers.{layer}.self_attn.q_proj.weight",
                               TensorTarget::QKVWeight, -1, -1);
        add_export_layer_slice("model.layers.{layer}.self_attn.k_proj.weight",
                               TensorTarget::QKVWeight, -2, -2);
        add_export_layer_slice("model.layers.{layer}.self_attn.v_proj.weight",
                               TensorTarget::QKVWeight, -3, -3);
        add_export_layer("model.layers.{layer}.self_attn.o_proj.weight", TensorTarget::OutWeight);

        // QKV bias (optional)
        add_export_layer_slice("model.layers.{layer}.self_attn.q_proj.bias",
                               TensorTarget::QKVBias, -1, -1, true);
        add_export_layer_slice("model.layers.{layer}.self_attn.k_proj.bias",
                               TensorTarget::QKVBias, -2, -2, true);
        add_export_layer_slice("model.layers.{layer}.self_attn.v_proj.bias",
                               TensorTarget::QKVBias, -3, -3, true);

        // QK norm (Qwen3-specific)
        add_export_layer("model.layers.{layer}.self_attn.q_norm.weight", TensorTarget::QNormWeight, true);
        add_export_layer("model.layers.{layer}.self_attn.k_norm.weight", TensorTarget::KNormWeight, true);

        // MoE-specific exports
        add_export_layer("model.layers.{layer}.mlp.gate.weight", TensorTarget::RouterGate);

        // Batched experts (internal fused format)
        add_export_layer("model.layers.{layer}.mlp.experts.gate_up_proj.weight", TensorTarget::ExpertsGateUp);
        add_export_layer("model.layers.{layer}.mlp.experts.down_proj.weight", TensorTarget::ExpertsDown);

        // Per-expert patterns (non-batched format) - registered with expert_idx placeholder
        // These are handled specially in export_to_file since we need to iterate over experts

        // Shared expert
        add_export_layer("model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                         TensorTarget::SharedExpertGate, true);
        add_export_layer("model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                         TensorTarget::SharedExpertUp, true);
        add_export_layer("model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                         TensorTarget::SharedExpertDown, true);
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN3MOE_WEIGHT_MAPPING_H
