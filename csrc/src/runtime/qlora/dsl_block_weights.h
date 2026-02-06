// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL-only block weight layouts for QLoRA providers.

#ifndef SUROGATE_SRC_MODULES_QLORA_DSL_BLOCK_WEIGHTS_H
#define SUROGATE_SRC_MODULES_QLORA_DSL_BLOCK_WEIGHTS_H

#include <optional>
#include <vector>

#include "utilities/tensor.h"

namespace modules {

struct DslNormWeights {
    Tensor weight;
};

struct DslAttentionWeights {
    Tensor qkv_weight;
    Tensor out_weight;
    std::optional<Tensor> q_norm_weight;
    std::optional<Tensor> k_norm_weight;
};

struct DslRouterWeights {
    Tensor gate;
    std::optional<Tensor> bias;
};

struct DslExpertWeights {
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
};

struct DslExpertGroupWeights {
    std::vector<DslExpertWeights> experts;
    Tensor gate_up_proj;
    Tensor down_proj;
    bool use_batched = false;
    int num_active_experts = 0;
};

// Dense block layout used by the DSL QLoRA path (independent of composite blocks).
struct DslDenseBlock {
    struct Config {};

    struct Weights {
        // Norms
        DslNormWeights ln1;
        DslNormWeights ln2;

        // Attention
        DslAttentionWeights attention;

        // MLP (dense FFN)
        Tensor mlp_up_weight;
        Tensor mlp_down_weight;

        // MoE fields (optional for hybrid layers)
        DslRouterWeights router;
        DslExpertGroupWeights experts;
        std::optional<DslExpertWeights> shared_expert;

        // Mamba / SSM weights (optional for hybrid architectures)
        struct MambaWeights {
            Tensor in_proj_weight;             ///< (proj_size, hidden_size)
            std::optional<Tensor> in_proj_bias;
            Tensor out_proj_weight;            ///< (hidden_size, intermediate_size)
            std::optional<Tensor> out_proj_bias;
            Tensor conv1d_weight;              ///< (conv_dim, 1, kernel)
            std::optional<Tensor> conv1d_bias;  ///< (conv_dim)
            Tensor A_log;                      ///< (num_heads) FP32
            Tensor D;                          ///< (num_heads) FP32
            Tensor dt_bias;                    ///< (num_heads) FP32
            Tensor norm_weight;                ///< (intermediate_size)
        };
        std::optional<MambaWeights> mamba;
    };
};

// MoE block layout used by the DSL QLoRA path.
struct DslMoEBlock {
    struct Config {};

    struct Weights {
        // Norms
        DslNormWeights ln1;
        DslNormWeights ln2;

        // Attention
        DslAttentionWeights attention;

        // Router + experts
        DslRouterWeights router;
        DslExpertGroupWeights experts;
        std::optional<DslExpertWeights> shared_expert;

        // Mamba / SSM weights (for hybrid architectures like Nemotron-H
        // that interleave MoE and Mamba layers)
        using MambaWeights = DslDenseBlock::Weights::MambaWeights;
        std::optional<MambaWeights> mamba;
    };
};

}  // namespace modules

#endif  // SUROGATE_SRC_MODULES_QLORA_DSL_BLOCK_WEIGHTS_H
