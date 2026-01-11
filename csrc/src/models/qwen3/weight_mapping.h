// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3 Weight Mapping - HuggingFace tensor name patterns for Qwen3 models

#ifndef SUROGATE_SRC_MODELS_QWEN3_WEIGHT_MAPPING_H
#define SUROGATE_SRC_MODELS_QWEN3_WEIGHT_MAPPING_H

#include "models/qwen25/weight_mapping.h"

namespace modules {

class Qwen3WeightMapping : public Qwen2WeightMapping {
public:
    void register_patterns() override {
        Qwen2WeightMapping::register_patterns();

        // QK normalization weights (Qwen3-specific)
        add_layer_pattern("model.layers.{layer}.self_attn.q_norm.weight", TensorTarget::QNormWeight);
        add_layer_pattern("model.layers.{layer}.self_attn.k_norm.weight", TensorTarget::KNormWeight);
    }

    void register_export_patterns() override {
        Qwen2WeightMapping::register_export_patterns();

        // QK normalization weights (Qwen3-specific)
        add_export_layer("model.layers.{layer}.self_attn.q_norm.weight", TensorTarget::QNormWeight, true);
        add_export_layer("model.layers.{layer}.self_attn.k_norm.weight", TensorTarget::KNormWeight, true);
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN3_WEIGHT_MAPPING_H
