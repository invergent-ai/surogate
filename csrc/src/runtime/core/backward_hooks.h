// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_BACKWARD_HOOKS_H
#define SUROGATE_SRC_MODULES_BACKWARD_HOOKS_H

#include <functional>

#include <cuda_runtime.h>

namespace modules {

/**
 * @brief Hook points during the backward pass
 *
 * These correspond to specific locations in the transformer block
 * where additional computation can be injected (e.g., LoRA gradient computation).
 */
enum class BackwardHookPoint {
    // Attention sub-block
    BeforeQKVBackward,      ///< Before QKV projection backward
    AfterQKVBackward,       ///< After QKV projection backward, gradients computed
    BeforeAttnOutBackward,  ///< Before attention output projection backward
    AfterAttnOutBackward,   ///< After attention output projection backward

    // MLP sub-block
    BeforeMLPDownBackward,  ///< Before MLP down projection backward
    AfterMLPDownBackward,   ///< After MLP down projection backward
    BeforeMLPUpBackward,    ///< Before MLP up projection backward
    AfterMLPUpBackward,     ///< After MLP up projection backward
    MoEExpertGroupManual,   ///< Manual expert group backward (fused)

    // MoE router
    AfterRouterBackward,  ///< After router backward (for router LoRA gradients)

    // Layer-level
    BeforeLayerBackward,  ///< Before any backward computation for this layer
    AfterLayerBackward,   ///< After all backward computation for this layer
};

/**
 * @brief Convert hook point to string for debugging
 */
constexpr const char* hook_point_name(BackwardHookPoint point) {
    switch (point) {
        case BackwardHookPoint::BeforeQKVBackward: return "BeforeQKVBackward";
        case BackwardHookPoint::AfterQKVBackward: return "AfterQKVBackward";
        case BackwardHookPoint::BeforeAttnOutBackward: return "BeforeAttnOutBackward";
        case BackwardHookPoint::AfterAttnOutBackward: return "AfterAttnOutBackward";
        case BackwardHookPoint::BeforeMLPDownBackward: return "BeforeMLPDownBackward";
        case BackwardHookPoint::AfterMLPDownBackward: return "AfterMLPDownBackward";
        case BackwardHookPoint::BeforeMLPUpBackward: return "BeforeMLPUpBackward";
        case BackwardHookPoint::AfterMLPUpBackward: return "AfterMLPUpBackward";
        case BackwardHookPoint::MoEExpertGroupManual: return "MoEExpertGroupManual";
        case BackwardHookPoint::AfterRouterBackward: return "AfterRouterBackward";
        case BackwardHookPoint::BeforeLayerBackward: return "BeforeLayerBackward";
        case BackwardHookPoint::AfterLayerBackward: return "AfterLayerBackward";
        default: return "Unknown";
    }
}

/**
 * @brief Callback signature for backward hooks
 *
 * @param layer_idx The transformer layer index (0-based)
 * @param point Where in the backward pass this hook is called
 * @param accumulate Whether gradients should be accumulated (vs overwritten)
 * @param stream CUDA stream for the computation
 *
 * Hooks are called synchronously during the backward pass. The hook
 * implementation should enqueue GPU work on the provided stream.
 */
using BackwardHook =
    std::function<void(int layer_idx, bool accumulate, cudaStream_t stream, BackwardHookPoint point, void* context)>;

// BackwardHookRegistry / BackwardHookGuard removed: all LoRA backward
// dispatch is now slice-driven via ``CompiledAttrs::lora_slices`` and
// ``apply_lora_slices_backward``. The ``BackwardHook`` callable type
// above is retained for ABI/signature stability with existing callers
// that still thread a hook pointer through op dispatch; all such
// pointers are null in the current build.

}  // namespace modules

#endif  // SUROGATE_SRC_MODULES_BACKWARD_HOOKS_H
