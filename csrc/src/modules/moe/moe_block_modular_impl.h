// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// MoETransformerBlock modular execution helpers

#ifndef SUROGATE_SRC_MODULES_MOE_MOE_BLOCK_MODULAR_IMPL_H
#define SUROGATE_SRC_MODULES_MOE_MOE_BLOCK_MODULAR_IMPL_H

#include "moe_block.h"
#include "modules/composite/block_builder.h"
#include "modules/composite/block_executor.h"
#include "modules/run_state_types.h"
#include "modules/model_config.h"
#include "modules/weights/weight_manager_types.h"
#include "recipes/recipe.h"

namespace modules {

template<typename AttentionType, typename RouterType, typename NormType>
template<typename Block>
void MoETransformerBlock<AttentionType, RouterType, NormType>::forward_block_modular(
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

template<typename AttentionType, typename RouterType, typename NormType>
template<typename Block>
void MoETransformerBlock<AttentionType, RouterType, NormType>::backward_block_modular(
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

template<typename AttentionType, typename RouterType, typename NormType>
template<typename Block>
void MoETransformerBlock<AttentionType, RouterType, NormType>::recompute_block_modular(
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

#endif // SUROGATE_SRC_MODULES_MOE_MOE_BLOCK_MODULAR_IMPL_H
