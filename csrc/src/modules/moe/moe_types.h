// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_MOE_TYPES_H
#define SUROGATE_SRC_MODULES_MOE_MOE_TYPES_H

#include "utilities/tensor.h"

namespace modules {

/**
 * @brief Context for MoE expert group computation
 *
 * This struct is passed to hooks during MoE block execution to enable
 * third-party modules (like LoRA) to perform specialized grouped computation.
 */
struct MoEGroupedContext {
    // Input/State tensors (from base model)
    const Tensor* expert_offsets;   ///< (num_experts + 1) int
    const Tensor* permuted_input;    ///< (total_tokens, C)
    const Tensor* expert_gate_up;    ///< (total_tokens, 2*D) output of base projection (Saved for backward)
    const Tensor* expert_outputs;    ///< (total_tokens, C) output of base down projection

    // Gradient tensors (during backward)
    const Tensor* d_expert_outputs;  ///< (total_tokens, C) gradient w.r.t expert outputs
    const Tensor* d_expert_gate_up;  ///< (total_tokens, 2*D) gradient w.r.t gate_up
    Tensor* d_permuted_input;        ///< (total_tokens, C) gradient w.r.t permuted input

    const int* host_offsets;        ///< Cached expert offsets on host
    int num_experts;
    int top_k;
    int total_tokens;

    // Optional: flag to indicate if the hook handled the computation
    bool handled = false;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_MOE_TYPES_H
