// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_CONFIG_LORA_ADAPTER_CONFIG_H
#define SUROGATE_SRC_CONFIG_LORA_ADAPTER_CONFIG_H

#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "utilities/dtype.h"

//! Which routed experts of each MoE layer get expert LoRA (`lora_expert_mask`).
//! Disabled (the default) = every expert of every MoE layer, today's behaviour. Enabled: layer `l`
//! adapts exactly `Layers[l]` (sorted, unique global expert ids); a layer past the end of `Layers`
//! adapts none, so `none` is an enabled mask with no layers. Shared-expert, attention, linear-attention
//! and router adapters do not depend on it.
struct LoRAExpertMask {
    bool Enabled = false;
    std::vector<std::vector<int>> Layers;
    std::string Source;  ///< "none", or the mask file's path: names the mask in messages and adapter_config.json
};

//! LoRA (Low-Rank Adaptation) adapter hyperparameters.
//! This is the user-facing configuration (CLI/python) for adapter training.
struct LoRAAdapterConfig {
    int Rank = 8;
    float Alpha = 16.0f;
    float Dropout = 0.0f;
    std::set<std::string> TargetModules = {"q_proj", "k_proj", "v_proj", "o_proj"};
    ETensorDType DType = ETensorDType::BF16;
    bool InitAKaimingUniform = true;
    bool UseRSLoRA = false;
    std::string FanInFanOut = "fan_in";
    bool TrainRouter = false;  ///< Router LoRA (rank Rank on mlp.gate) during LoRA fine-tuning
    LoRAExpertMask ExpertMask;  ///< Per-layer routed-expert selection for expert LoRA (default: all experts)

    [[nodiscard]] float scaling() const {
        if (UseRSLoRA) {
            return Alpha / std::sqrt(static_cast<float>(Rank));
        }
        return Alpha / static_cast<float>(Rank);
    }

    [[nodiscard]] bool applies_to(const std::string& module_name) const {
        if (TargetModules.count("all") > 0) {
            return true;
        }
        return TargetModules.count(module_name) > 0;
    }

    [[nodiscard]] bool applies_to_q() const {
        return applies_to("q_proj") || applies_to("q_k_v_proj");
    }
    [[nodiscard]] bool applies_to_k() const {
        return applies_to("k_proj");
    }
    [[nodiscard]] bool applies_to_v() const {
        return applies_to("v_proj");
    }
    [[nodiscard]] bool applies_to_o() const {
        return applies_to("o_proj") || applies_to("out_proj");
    }
    [[nodiscard]] bool applies_to_gate() const {
        return applies_to("gate_proj");
    }
    [[nodiscard]] bool applies_to_gate_up() const {
        return applies_to("gate_up_proj");
    }
    [[nodiscard]] bool applies_to_up() const {
        return applies_to("up_proj");
    }
    [[nodiscard]] bool applies_to_down() const {
        return applies_to("down_proj");
    }

    [[nodiscard]] bool applies_to_attention() const {
        return applies_to_q() || applies_to_k() || applies_to_v() || applies_to_o();
    }

    [[nodiscard]] bool applies_to_mlp() const {
        return applies_to_gate() || applies_to_gate_up() || applies_to_up() || applies_to_down();
    }
};

// Backwards-compatible alias.
using LoRAConfig = LoRAAdapterConfig;

#endif  // SUROGATE_SRC_CONFIG_LORA_ADAPTER_CONFIG_H
