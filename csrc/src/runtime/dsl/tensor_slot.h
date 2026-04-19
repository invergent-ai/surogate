// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Pre-resolved tensor slot types for compiled operation dispatch.

#ifndef SUROGATE_SRC_DSL_TENSOR_SLOT_H
#define SUROGATE_SRC_DSL_TENSOR_SLOT_H

#include <cstdint>

namespace dsl {

// Tensor slot types for pre-resolution
enum class TensorSlot : std::uint8_t {
    // Activation slots (layer-indexed)
    BlockLN1,
    BlockLN1RSTD,
    BlockLN2,
    BlockLN2RSTD,
    BlockQRSTD,
    BlockKRSTD,
    BlockQKV,
    BlockQKVRoPE,
    BlockLSE,
    BlockAtt,
    BlockAttOut,
    BlockResidualAtt,
    BlockMLPUp,
    BlockSwiGLU,
    BlockMLPDown,
    BlockResidualFFN,
    BlockHOut,  ///< Block final output (post layer_scalar); used by Gemma4 to avoid mlp_down collision
    // MoE activation slots (layer-indexed, populated only when NumExperts > 0)
    BlockRouterLogits,
    BlockRouterProbs,
    BlockRoutingWeights,
    BlockRoutingIndices,
    BlockPermutedInput,
    BlockScatterIndices,
    BlockExpertGateUp,
    BlockExpertAct,
    BlockExpertDown,
    BlockMoeOut,  ///< View of mlp_down as flat (B*T, C) for MoE models
    // Gradient slots (layer-indexed)
    BlockDLN1,
    BlockDQKV,
    BlockDAtt,
    BlockDSwiGLU,
    BlockDMLPUp,
    BlockDMLPDown,
    BlockDHOut,
    BlockDLN2,
    BlockDResAtt,
    BlockDAttOut,
    BlockDResFFN,
    // Global activations
    Encoded,
    LNFinal,
    LNFinalRSTD,
    FinalResidual,
    FreqCis,
    // Inputs
    TokenIDs,
    PositionIDs,
    Targets,
    Losses,
    DLoss,
    // Named parameter (uses name lookup)
    Parameter,
    // Temporary (stack-allocated)
    Temporary,
    // Saved tensor (from forward pass)
    Saved,
    // Already in tensor map
    Mapped,
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_TENSOR_SLOT_H
