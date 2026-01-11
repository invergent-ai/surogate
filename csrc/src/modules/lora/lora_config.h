// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_CONFIG_H
#define SUROGATE_SRC_MODULES_LORA_LORA_CONFIG_H

#include <cmath>
#include <set>
#include <string>

#include "config/lora_adapter_config.h"
#include "utilities/dtype.h"

namespace modules {

/**
 * @brief Target modules for LoRA adaptation
 */
enum class LoRATarget {
    Q_PROJ,      ///< Query projection
    K_PROJ,      ///< Key projection
    V_PROJ,      ///< Value projection
    O_PROJ,      ///< Output projection
    GATE_PROJ,   ///< MLP gate projection
    UP_PROJ,     ///< MLP up projection
    DOWN_PROJ    ///< MLP down projection
};

/**
 * @brief LoRA (Low-Rank Adaptation) configuration
 *
 * Defines hyperparameters for LoRA adapter training.
 * LoRA decomposes weight updates into low-rank matrices:
 *   W' = W + (alpha/r) * B @ A
 * where B is (out_features, rank) and A is (rank, in_features)
 */
struct ModularLoRAConfig {
    /// Rank of the low-rank decomposition
    int rank = 8;

    /// Scaling factor: output is multiplied by alpha/rank
    float alpha = 16.0f;

    /// Dropout probability applied to LoRA activations (0 = no dropout)
    float dropout = 0.0f;

    /// Data type for LoRA weights
    ETensorDType dtype = ETensorDType::BF16;

    /// Initialize A with Kaiming uniform, B with zeros (following PEFT)
    bool init_a_kaiming = true;

    /// Use RSLoRA scaling (alpha / sqrt(rank) instead of alpha / rank)
    bool use_rs_lora = false;

    /// Target modules for LoRA adaptation
    std::set<LoRATarget> targets = {
        LoRATarget::Q_PROJ,
        LoRATarget::K_PROJ,
        LoRATarget::V_PROJ,
        LoRATarget::O_PROJ
    };

    /**
     * @brief Get the LoRA scaling factor
     */
    [[nodiscard]] float scaling() const {
        if (use_rs_lora) {
            return alpha / std::sqrt(static_cast<float>(rank));
        }
        return alpha / static_cast<float>(rank);
    }

    /**
     * @brief Check if LoRA applies to a specific target
     */
    [[nodiscard]] bool applies_to(LoRATarget target) const {
        return targets.count(target) > 0;
    }

    // Convenience accessors
    [[nodiscard]] bool applies_to_q() const { return applies_to(LoRATarget::Q_PROJ); }
    [[nodiscard]] bool applies_to_k() const { return applies_to(LoRATarget::K_PROJ); }
    [[nodiscard]] bool applies_to_v() const { return applies_to(LoRATarget::V_PROJ); }
    [[nodiscard]] bool applies_to_o() const { return applies_to(LoRATarget::O_PROJ); }
    [[nodiscard]] bool applies_to_gate() const { return applies_to(LoRATarget::GATE_PROJ); }
    [[nodiscard]] bool applies_to_up() const { return applies_to(LoRATarget::UP_PROJ); }
    [[nodiscard]] bool applies_to_down() const { return applies_to(LoRATarget::DOWN_PROJ); }

    [[nodiscard]] bool applies_to_attention() const {
        return applies_to_q() || applies_to_k() || applies_to_v() || applies_to_o();
    }

    [[nodiscard]] bool applies_to_mlp() const {
        return applies_to_gate() || applies_to_up() || applies_to_down();
    }

    /**
     * @brief Check if LoRA is enabled (rank > 0)
     */
    [[nodiscard]] bool enabled() const { return rank > 0; }

    /**
     * @brief Enable all attention targets
     */
    ModularLoRAConfig& with_attention() {
        targets.insert(LoRATarget::Q_PROJ);
        targets.insert(LoRATarget::K_PROJ);
        targets.insert(LoRATarget::V_PROJ);
        targets.insert(LoRATarget::O_PROJ);
        return *this;
    }

    /**
     * @brief Enable all MLP targets
     */
    ModularLoRAConfig& with_mlp() {
        targets.insert(LoRATarget::GATE_PROJ);
        targets.insert(LoRATarget::UP_PROJ);
        targets.insert(LoRATarget::DOWN_PROJ);
        return *this;
    }

    /**
     * @brief Enable all targets
     */
    ModularLoRAConfig& with_all() {
        return with_attention().with_mlp();
    }

    /// Create from the user-facing LoRA adapter config (CLI/python).
    static ModularLoRAConfig from_adapter_config(const LoRAAdapterConfig& cfg);

    /// Convert to the user-facing LoRA adapter config (CLI/python).
    [[nodiscard]] LoRAAdapterConfig to_adapter_config() const;
};

/**
 * @brief Builder for LoRA configuration
 */
class LoRAConfigBuilder {
public:
    LoRAConfigBuilder() = default;

    LoRAConfigBuilder& rank(int r) { mConfig.rank = r; return *this; }
    LoRAConfigBuilder& alpha(float a) { mConfig.alpha = a; return *this; }
    LoRAConfigBuilder& dropout(float d) { mConfig.dropout = d; return *this; }
    LoRAConfigBuilder& dtype(ETensorDType dt) { mConfig.dtype = dt; return *this; }
    LoRAConfigBuilder& init_a_kaiming(bool v) { mConfig.init_a_kaiming = v; return *this; }
    LoRAConfigBuilder& use_rs_lora(bool v) { mConfig.use_rs_lora = v; return *this; }

    LoRAConfigBuilder& target(LoRATarget t) {
        mConfig.targets.insert(t);
        return *this;
    }

    LoRAConfigBuilder& attention() { mConfig.with_attention(); return *this; }
    LoRAConfigBuilder& mlp() { mConfig.with_mlp(); return *this; }
    LoRAConfigBuilder& all() { mConfig.with_all(); return *this; }

    LoRAConfigBuilder& clear_targets() {
        mConfig.targets.clear();
        return *this;
    }

    ModularLoRAConfig build() const { return mConfig; }

private:
    ModularLoRAConfig mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_CONFIG_H
