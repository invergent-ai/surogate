// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen2 Weight Mapping - HuggingFace tensor name patterns for Qwen2 models

#ifndef SUROGATE_SRC_MODELS_QWEN2_WEIGHT_MAPPING_H
#define SUROGATE_SRC_MODELS_QWEN2_WEIGHT_MAPPING_H

#include "models/llama/weight_mapping.h"

namespace modules {

class Qwen2WeightMapping : public LlamaWeightMapping {
public:
    void register_patterns() override {
        LlamaWeightMapping::register_patterns();
        // Qwen2 uses same weight patterns as Llama
        // Sliding window is handled at runtime, not via weight loading
    }

    void register_export_patterns() override {
        LlamaWeightMapping::register_export_patterns();
        // Qwen2 uses same export patterns as Llama
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN2_WEIGHT_MAPPING_H
