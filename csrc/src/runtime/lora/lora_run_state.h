// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_RUN_STATE_H
#define SUROGATE_SRC_MODULES_LORA_LORA_RUN_STATE_H

#include "utilities/tensor.h"

namespace modules {

/**
 * @brief Runtime state for LoRA execution
 */
struct LoRARunState {
    Tensor intermediate;   // (BT, rank) - first intermediate buffer
    Tensor intermediate2;  // (BT, rank) - second intermediate buffer for fused ops
    Tensor slice;
    Tensor norm_buffer;
    Tensor recompute_ln;   // (B, T, C) - buffer for recomputed ln1/ln2 activations
    Tensor recompute_rstd; // (B, T) - buffer for recomputed rstd (unused but required by kernel)
    int B = 0;
    int T = 0;

    // Grouped MoE LoRA scratch buffers
    Tensor moe_lora_intermediate1; // (total_tokens, rank)
    Tensor moe_lora_intermediate2; // (total_tokens, D)
    Tensor moe_lora_gate;          // (total_tokens, D) - contiguous buffer for gate projection
    Tensor moe_lora_up;            // (total_tokens, D) - contiguous buffer for up projection
    Tensor moe_lora_gate_up;       // (total_tokens, 2*D) - combined gate+up buffer

    // MoE expert LoRA state: pointers to current expert activations during hook execution.
    // These are set by the MoE block before calling expert hooks and read by the hook callback.
    // This avoids the need to pass activation pointers through the hook signature.

    // Dropout state: used to maintain consistent dropout masks between forward and backward passes
    bool is_training = true;              ///< Training mode (dropout applied) vs eval mode (no dropout)
    int micro_step = 0;                   ///< Current micro-step for seed computation
    unsigned int dropout_base_seed = 42;  ///< Base seed for dropout RNG
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_RUN_STATE_H
