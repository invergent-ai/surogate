// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Profile-owned run-state allocation requirements.

#ifndef SUROGATE_SRC_RUNTIME_CORE_RUN_STATE_REQUIREMENTS_H
#define SUROGATE_SRC_RUNTIME_CORE_RUN_STATE_REQUIREMENTS_H

namespace dsl {

struct RuntimeRunStateRequirements {
    bool token_inputs = true;
    bool position_ids = true;
    bool targets = true;
    bool visual_inputs = true;
    bool loss_metrics = true;

    bool encoded_activation = true;
    bool final_norm_activation = true;
    bool final_norm_rstd = true;
    bool logits_output = true;
    bool rope_freqs = true;
    bool final_norm_grad = true;
    bool embedding_grad = true;

    bool common_scratch = true;
    bool cross_entropy_scratch = true;
    bool encoder_backward_scratch = true;
    bool attention_workspace = true;
    bool transformer_quant_state = true;
    bool residual_buffers = true;
    bool per_layer_graph_state = true;

    [[nodiscard]] static RuntimeRunStateRequirements causal_lm() {
        return RuntimeRunStateRequirements{};
    }

    [[nodiscard]] static RuntimeRunStateRequirements embedding() {
        RuntimeRunStateRequirements req;
        req.position_ids = false;
        req.targets = false;
        req.visual_inputs = false;
        req.loss_metrics = false;
        req.final_norm_activation = false;
        req.final_norm_rstd = false;
        req.logits_output = false;
        req.rope_freqs = false;
        req.final_norm_grad = false;
        req.embedding_grad = false;
        req.common_scratch = false;
        req.cross_entropy_scratch = false;
        req.encoder_backward_scratch = false;
        req.attention_workspace = false;
        req.transformer_quant_state = false;
        req.residual_buffers = false;
        req.per_layer_graph_state = false;
        return req;
    }
};

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_CORE_RUN_STATE_REQUIREMENTS_H
