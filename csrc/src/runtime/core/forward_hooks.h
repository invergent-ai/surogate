// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_FORWARD_HOOKS_H
#define SUROGATE_SRC_MODULES_FORWARD_HOOKS_H

#include <functional>

#include <cuda_runtime.h>

namespace modules {

/**
 * @brief Hook points during the forward pass
 *
 * These correspond to specific locations in the transformer block forward
 * where additional computation can be injected (e.g., applying LoRA deltas).
 */
enum class ForwardHookPoint {
    AfterQKVProjection,      ///< After QKV matmul, before RoPE
    AfterAttnOutProjection,  ///< After attention out matmul, before residual+LN2
    AfterMLPUpProjection,    ///< After MLP up matmul, before SwiGLU
    AfterMLPDownProjection,  ///< After MLP down matmul
    MoEExpertGroupManual,    ///< Manual expert group execution (fused)
    AfterRouterProjection,   ///< After router logits matmul, before softmax (for router LoRA)
};

constexpr const char* hook_point_name(ForwardHookPoint point) {
    switch (point) {
        case ForwardHookPoint::AfterQKVProjection: return "AfterQKVProjection";
        case ForwardHookPoint::AfterAttnOutProjection: return "AfterAttnOutProjection";
        case ForwardHookPoint::AfterMLPUpProjection: return "AfterMLPUpProjection";
        case ForwardHookPoint::AfterMLPDownProjection: return "AfterMLPDownProjection";
        case ForwardHookPoint::MoEExpertGroupManual: return "MoEExpertGroupManual";
        case ForwardHookPoint::AfterRouterProjection: return "AfterRouterProjection";
        default: return "Unknown";
    }
}

using ForwardHook = std::function<void(int layer_idx, cudaStream_t stream, ForwardHookPoint point, void* context)>;

/**
 * @brief Hook points during MoE expert forward pass
 *
 * These correspond to specific locations within a single expert's forward pass
 * where per-expert LoRA can be applied.
 */
enum class MoEExpertHookPoint {
    AfterExpertUpProjection,   ///< After expert gate_up matmul, before SwiGLU
    AfterExpertDownProjection, ///< After expert down matmul
    ManualGroup,               ///< Manual expert group execution (fused)
};

constexpr const char* hook_point_name(MoEExpertHookPoint point) {
    switch (point) {
        case MoEExpertHookPoint::AfterExpertUpProjection: return "AfterExpertUpProjection";
        case MoEExpertHookPoint::AfterExpertDownProjection: return "AfterExpertDownProjection";
        case MoEExpertHookPoint::ManualGroup: return "ManualGroup";
        default: return "Unknown";
    }
}

/**
 * @brief Hook for MoE expert forward pass
 *
 * Called during per-expert computation with the expert index.
 * @param layer_idx The layer index
 * @param expert_idx The expert index within the layer (-1 for manual group)
 * @param point The hook point within the expert forward pass
 * @param stream CUDA stream
 * @param context Opaque context (MoEGroupedContext* for ManualGroup)
 */
using MoEExpertHook = std::function<void(int layer_idx, int expert_idx, MoEExpertHookPoint point, cudaStream_t stream, void* context)>;

} // namespace modules

#endif // SUROGATE_SRC_MODULES_FORWARD_HOOKS_H

