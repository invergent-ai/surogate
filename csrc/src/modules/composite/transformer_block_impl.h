// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DenseTransformerBlock implementation details

#ifndef SUROGATE_SRC_MODULES_COMPOSITE_TRANSFORMER_BLOCK_IMPL_H
#define SUROGATE_SRC_MODULES_COMPOSITE_TRANSFORMER_BLOCK_IMPL_H

#include "transformer_block.h"
#include "modules/run_state_types.h"
#include "modules/fp8_run_state.h"
#include "modules/fp4_run_state.h"
#include "modules/model_config.h"
#include "modules/matmul_context.h"
#include "modules/primitives/recipe_ops.h"
#include "modules/composite/block_builder.h"
#include "modules/composite/block_executor.h"
#include "modules/fp8_scaling_config.h"
#include "modules/weights/weight_manager_types.h"
#include "recipes/recipe.h"
#include "kernels/kernels.h"

// Forward declarations for template parameters
namespace modules {
template<typename Block> class ModularRunState;
template<typename Block> class ModularWeightManager;
}

namespace modules {

// ============================================================================
// Modular Block Execution (Production)
//
// These static methods provide a clean, composable implementation that uses
// recipe-driven matmul dispatch. They work with SimplifiedLayerActivations
// from the model's run state.
//
// The implementations closely follow model_forward.hpp / model_block_ops.hpp
// but are organized as static methods on the block type for cleaner API.
// ============================================================================

template<typename AttentionType, typename MLPType, typename NormType>
template<typename Block>
void DenseTransformerBlock<AttentionType, MLPType, NormType>::forward_block_modular(
    const ::recipes::Recipe& recipe,
    ModularRunState<Block>& rs,
    Weights& weights,
    SimplifiedLayerActivations& acts,
    SimplifiedLayerQuantActivations& quant_acts,
    Tensor& residual,
    int layer_idx,
    const ModelConfig& config,
    const ModelOptions& options,
    cudaStream_t stream,
    FP8ForwardQuantActivations* fp8_fwd_quants,
    FP4ForwardQuantActivations* fp4_fwd_quants,
    ModularWeightManager<Block>* weight_manager,
    bool allow_quant_layer,
    const ForwardHook* hook)
{
    const auto spec = BlockBuilder::build(config, layer_idx);
    BlockExecutor::forward(
        spec,
        recipe,
        rs,
        weights,
        acts,
        quant_acts,
        residual,
        layer_idx,
        config,
        options,
        stream,
        fp8_fwd_quants,
        fp4_fwd_quants,
        weight_manager,
        allow_quant_layer,
        hook);
}

template<typename AttentionType, typename MLPType, typename NormType>
template<typename Block>
void DenseTransformerBlock<AttentionType, MLPType, NormType>::backward_block_modular(
    const ::recipes::Recipe& recipe,
    ModularRunState<Block>& rs,
    Weights& weights,
    Gradients& grads,
    SimplifiedLayerActivations& acts,
    SimplifiedLayerGradients& d_acts,
    SimplifiedLayerQuantActivations& quant_acts,
    SimplifiedQuantGradients& quant_grads,
    int layer_idx,
    const ModelConfig& config,
    const ModelOptions& options,
    bool accumulate,
    cudaStream_t stream,
    bool allow_quant_layer,
    const BackwardHook* hook)
{
    const auto spec = BlockBuilder::build(config, layer_idx);
    BlockExecutor::backward(
        spec,
        recipe,
        rs,
        weights,
        grads,
        acts,
        d_acts,
        quant_acts,
        quant_grads,
        layer_idx,
        config,
        options,
        accumulate,
        stream,
        allow_quant_layer,
        hook);
}

template<typename AttentionType, typename MLPType, typename NormType>
template<typename Block>
void DenseTransformerBlock<AttentionType, MLPType, NormType>::recompute_block_modular(
    const ::recipes::Recipe& recipe,
    ModularRunState<Block>& rs,
    Weights& weights,
    SimplifiedLayerActivations& acts,
    SimplifiedLayerQuantActivations& quant_acts,
    Tensor& residual,
    int layer_idx,
    const ModelConfig& config,
    const ModelOptions& options,
    cudaStream_t stream,
    FP8ForwardQuantActivations* fp8_fwd_quants,
    FP4ForwardQuantActivations* fp4_fwd_quants,
    ModularWeightManager<Block>* weight_manager,
    bool allow_quant_layer)
{
    const auto spec = BlockBuilder::build(config, layer_idx);
    BlockExecutor::recompute(
        spec,
        recipe,
        rs,
        weights,
        acts,
        quant_acts,
        residual,
        layer_idx,
        config,
        options,
        stream,
        fp8_fwd_quants,
        fp4_fwd_quants,
        weight_manager,
        allow_quant_layer);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_COMPOSITE_TRANSFORMER_BLOCK_IMPL_H
