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

        // ====================================================================
        // MoE (Nemotron-H hybrid) - routed experts + optional shared expert
        // ====================================================================

        // Router weights (various naming conventions)
        add_layer_pattern("model.layers.{layer}.mlp.gate.weight", TensorTarget::RouterGate);
        add_layer_pattern("model.layers.{layer}.mlp.router.weight", TensorTarget::RouterGate);
        add_layer_pattern("model.layers.{layer}.mlp.router.gate.weight", TensorTarget::RouterGate);
        add_layer_pattern("model.layers.{layer}.mixer.gate.weight", TensorTarget::RouterGate);
        add_layer_pattern("model.layers.{layer}.mixer.router.weight", TensorTarget::RouterGate);
        add_layer_pattern("model.layers.{layer}.mixer.router.gate.weight", TensorTarget::RouterGate);
        add_layer_pattern("backbone.layers.{layer}.mlp.gate.weight", TensorTarget::RouterGate);
        add_layer_pattern("backbone.layers.{layer}.mlp.router.weight", TensorTarget::RouterGate);
        add_layer_pattern("backbone.layers.{layer}.mixer.gate.weight", TensorTarget::RouterGate);
        add_layer_pattern("backbone.layers.{layer}.mixer.router.weight", TensorTarget::RouterGate);

        // Router bias (optional)
        add_layer_pattern("model.layers.{layer}.mlp.router.bias", TensorTarget::RouterBias, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.router.bias", TensorTarget::RouterBias, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mlp.router.bias", TensorTarget::RouterBias, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.router.bias", TensorTarget::RouterBias, nullptr, true);

        // Batched expert weights (pre-fused format)
        add_layer_pattern("model.layers.{layer}.mlp.experts.gate_up_proj.weight",
                          TensorTarget::ExpertsGateUp, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mlp.experts.down_proj.weight",
                          TensorTarget::ExpertsDown, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.experts.gate_up_proj.weight",
                          TensorTarget::ExpertsGateUp, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.experts.down_proj.weight",
                          TensorTarget::ExpertsDown, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mlp.experts.gate_up_proj.weight",
                          TensorTarget::ExpertsGateUp, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mlp.experts.down_proj.weight",
                          TensorTarget::ExpertsDown, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.experts.gate_up_proj.weight",
                          TensorTarget::ExpertsGateUp, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.experts.down_proj.weight",
                          TensorTarget::ExpertsDown, nullptr, true);

        // Per-expert weights (non-batched format)
        add_expert_pattern("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                           TensorTarget::ExpertGate, nullptr, true);
        add_expert_pattern("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                           TensorTarget::ExpertUp);
        add_expert_pattern("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                           TensorTarget::ExpertDown);
        add_expert_pattern("model.layers.{layer}.mixer.experts.{expert}.gate_proj.weight",
                           TensorTarget::ExpertGate, nullptr, true);
        add_expert_pattern("model.layers.{layer}.mixer.experts.{expert}.up_proj.weight",
                           TensorTarget::ExpertUp);
        add_expert_pattern("model.layers.{layer}.mixer.experts.{expert}.down_proj.weight",
                           TensorTarget::ExpertDown);
        add_expert_pattern("backbone.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                           TensorTarget::ExpertGate, nullptr, true);
        add_expert_pattern("backbone.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                           TensorTarget::ExpertUp);
        add_expert_pattern("backbone.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                           TensorTarget::ExpertDown);
        add_expert_pattern("backbone.layers.{layer}.mixer.experts.{expert}.gate_proj.weight",
                           TensorTarget::ExpertGate, nullptr, true);
        add_expert_pattern("backbone.layers.{layer}.mixer.experts.{expert}.up_proj.weight",
                           TensorTarget::ExpertUp);
        add_expert_pattern("backbone.layers.{layer}.mixer.experts.{expert}.down_proj.weight",
                           TensorTarget::ExpertDown);

        // Shared expert (optional)
        add_layer_pattern("model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                          TensorTarget::SharedExpertUp, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                          TensorTarget::SharedExpertDown, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.shared_expert.gate_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.shared_expert.up_proj.weight",
                          TensorTarget::SharedExpertUp, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.shared_expert.down_proj.weight",
                          TensorTarget::SharedExpertDown, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mlp.shared_expert.up_proj.weight",
                          TensorTarget::SharedExpertUp, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mlp.shared_expert.down_proj.weight",
                          TensorTarget::SharedExpertDown, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.shared_expert.gate_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.shared_expert.up_proj.weight",
                          TensorTarget::SharedExpertUp, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.shared_expert.down_proj.weight",
                          TensorTarget::SharedExpertDown, nullptr, true);

        // Fused shared expert pattern (optional)
        add_layer_pattern("model.layers.{layer}.mlp.shared_expert.gate_up_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);
        add_layer_pattern("model.layers.{layer}.mixer.shared_expert.gate_up_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mlp.shared_expert.gate_up_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);
        add_layer_pattern("backbone.layers.{layer}.mixer.shared_expert.gate_up_proj.weight",
                          TensorTarget::SharedExpertGate, nullptr, true);

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
