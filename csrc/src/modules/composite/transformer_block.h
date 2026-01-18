// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_COMPOSITE_TRANSFORMER_BLOCK_H
#define SUROGATE_SRC_MODULES_COMPOSITE_TRANSFORMER_BLOCK_H

#include <type_traits>

#include "config/rope_config.h"
#include "modules/module_base.h"
#include "modules/primitives/attention.h"
#include "modules/primitives/rmsnorm.h"
#include "modules/primitives/mlp.h"
#include "modules/forward_hooks.h"
#include "modules/backward_hooks.h"
#include "kernels/kernels.h"

// Forward declaration for recipe types (global ::recipes namespace)
namespace recipes { class Recipe; }

namespace modules {

// Forward declarations for the modular execution context
template<typename Block> class ModularRunState;
template<typename Block> class ModularWeightManager;

// Forward declarations for simplified activation/gradient types (from run_state_types.h)
struct SimplifiedLayerActivations;
struct SimplifiedLayerGradients;
struct SimplifiedLayerQuantActivations;
struct SimplifiedQuantGradients;

// Forward declarations for quantization activation buffers
struct FP8ForwardQuantActivations;
struct FP4ForwardQuantActivations;

// Forward declarations for model configuration (from model_config.h)
struct ModelConfig;
struct ModelOptions;

/**
 * @brief Dense Transformer Block
 *
 * Implements a standard pre-norm transformer block:
 *   x = x + Attention(RMSNorm(x))
 *   x = x + MLP(RMSNorm(x))
 *
 * Where MLP is: Linear (gate+up) -> SwiGLU -> Linear (down)
 *
 * This block uses the modular architecture where each sub-component
 * (attention, MLP, normalization) is a separate module following the
 * ModuleBase CRTP pattern. This enables:
 * - Zero-overhead static polymorphism via CRTP
 * - Compile-time customization of components
 * - Clean separation of concerns
 * - Reusable primitive modules
 *
 * Template parameters allow swapping components at compile time for
 * different architectures without runtime overhead:
 * - AttentionType: Different attention implementations (MHA, GQA, etc.)
 * - MLPType: Different MLP implementations (SwiGLU, GeGLU, etc.)
 * - NormType: Different normalizations (RMSNorm, LayerNorm, etc.)
 *
 * Residual connections are handled by the fused residual+norm operations
 * for efficiency. The pattern is:
 * - Block receives input residual
 * - LN1 normalizes the residual for attention
 * - Attention output is added to residual via fused LN2 operation
 * - MLP output is returned (caller handles next residual addition)
 *
 * == Modular Block Execution ==
 *
 * The model uses the static methods forward_block_modular(), backward_block_modular(),
 * and recompute_block_modular() as the production execution path.
 *
 * These methods provide a clean, composable implementation that is fully
 * compatible with the training infrastructure (recipes, FP8/FP4 quantization,
 * hooks, CUDA graphs, etc.).
 */
template<
    typename AttentionType = AttentionModule,
    typename MLPType = SwiGLUMLP,
    typename NormType = FusedResidualRMSNormModule
>
class DenseTransformerBlock : public ModuleBase<DenseTransformerBlock<AttentionType, MLPType, NormType>> {
public:
    /**
     * @brief Configuration for transformer block
     */
    struct Config {
        // Attention config
        int hidden_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;          // Usually hidden_size / num_query_heads
        RoPEConfig rope;        // Flexible RoPE configuration
        int max_seq_len;        // Maximum sequence length for RoPE
        bool use_qkv_bias = false;
        bool use_qk_norm = false;

        // MLP config
        int intermediate_size;

        // Norm config
        float rms_norm_eps = 1e-5f;

        // Derived configs for sub-modules
        [[nodiscard]] typename AttentionType::Config attention_config() const {
            typename AttentionType::Config cfg;
            cfg.hidden_size = hidden_size;
            cfg.num_query_heads = num_query_heads;
            cfg.num_kv_heads = num_kv_heads;
            cfg.rope = rope;
            cfg.use_qkv_bias = use_qkv_bias;
            cfg.use_qk_norm = use_qk_norm;
            cfg.qk_norm_eps = rms_norm_eps;
            cfg.head_size = head_size;
            return cfg;
        }

        [[nodiscard]] typename NormType::Config norm_config() const {
            return {
                .hidden_size = hidden_size,
                .epsilon = rms_norm_eps
            };
        }

        [[nodiscard]] typename MLPType::Config mlp_config() const {
            return {
                .hidden_size = hidden_size,
                .intermediate_size = intermediate_size
            };
        }
    };

    /**
     * @brief All weights for the transformer block
     */
    struct Weights {
        // Pre-attention norm
        typename NormType::Weights ln1;

        // Attention
        typename AttentionType::Weights attention;

        // Pre-MLP norm
        typename NormType::Weights ln2;

        // MLP weights - exposed directly for backward compatibility
        // Legacy code expects flat members: mlp_up_weight, mlp_down_weight
        Tensor mlp_up_weight;    ///< (2 * intermediate_size, hidden_size) fused gate+up
        Tensor mlp_down_weight;  ///< (hidden_size, intermediate_size) down projection

        // Accessor for new modular code that expects nested structure
        struct MLPWeightsProxy {
            Tensor& up_weight;
            Tensor& down_weight;
        };
        MLPWeightsProxy mlp_weights() { return {mlp_up_weight, mlp_down_weight}; }
    };

    /**
     * @brief Saved activations for backward pass
     *
     * Note: For production training, use SimplifiedLayerActivations from run_state_types.h
     * which is optimized for the training infrastructure. This type is kept for
     * API completeness and standalone module testing.
     */
    struct Activations {
        // Norm 1
        typename NormType::Activations ln1_acts;
        Tensor ln1_output;              ///< Output of LN1 (input to attention)

        // Attention
        typename AttentionType::Activations attention_acts;
        Tensor attention_output;        ///< Output of attention (before residual add)

        // Residual after attention (for backward)
        Tensor residual_att;            ///< residual + attention_output

        // Norm 2
        typename NormType::Activations ln2_acts;

        // MLP
        typename MLPType::Activations mlp_acts;
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        typename NormType::Gradients ln1_grads;
        typename AttentionType::Gradients attention_grads;
        typename NormType::Gradients ln2_grads;

        // MLP gradients - exposed directly for backward compatibility
        // Legacy code expects flat members: d_mlp_up_weight, d_mlp_down_weight
        Tensor d_mlp_up_weight;   ///< (2 * intermediate_size, hidden_size)
        Tensor d_mlp_down_weight; ///< (hidden_size, intermediate_size)

        // Accessor for new modular code that expects nested structure
        struct MLPGradientsProxy {
            Tensor& d_up_weight;
            Tensor& d_down_weight;
        };
        MLPGradientsProxy mlp_grads() { return {d_mlp_up_weight, d_mlp_down_weight}; }
    };

    explicit DenseTransformerBlock(Config config)
        : mConfig(config),
          mAttention(config.attention_config()) {}

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int hidden_size() const { return mConfig.hidden_size; }
    [[nodiscard]] int intermediate_size() const { return mConfig.intermediate_size; }
    [[nodiscard]] int num_query_heads() const { return mConfig.num_query_heads; }
    [[nodiscard]] int num_kv_heads() const { return mConfig.num_kv_heads; }

    // ========================================================================
    // Modular Block Execution (Production)
    //
    // These static methods provide a fully-featured implementation that works
    // with the training infrastructure. They use:
    // - SimplifiedLayerActivations for activation storage
    // - Recipe-driven matmul dispatch (BF16, FP8, FP4)
    // - Full quantization support
    // - Hook compatibility
    //
    // Usage: Default path for ModularTransformerModel
    // ========================================================================

    /**
     * @brief Modular forward pass for a transformer block layer
     *
     * Fully compatible with the training infrastructure. Computes:
     *   ln1 = RMSNorm(residual)
     *   qkv = ln1 @ qkv_weight + optional QK-norm + RoPE
     *   att = FlashAttention(qkv)
     *   att_out = att @ out_weight
     *   residual_att = residual + att_out
     *   ln2 = RMSNorm(residual_att)
     *   mlp_up = ln2 @ mlp_up_weight
     *   swiglu = SwiGLU(mlp_up)
     *   mlp_down = swiglu @ mlp_down_weight
     *
     * @tparam Block The transformer block type (for RunState/WeightManager templates)
     * @param recipe Training recipe (handles matmul dispatch, quantization)
     * @param rs ModularRunState with activation buffers
     * @param weights Block weights
     * @param acts SimplifiedLayerActivations for this layer
     * @param quant_acts SimplifiedLayerQuantActivations for FP8/FP4 inputs
     * @param residual Input residual tensor
     * @param layer_idx Layer index (for recipe quantizer indices)
     * @param config Model configuration
     * @param options Model options (recompute flags, quantization settings)
     * @param stream CUDA stream
     * @param fp8_fwd_quants Optional FP8 forward quant buffers
     * @param fp4_fwd_quants Optional FP4 forward quant buffers
     * @param weight_manager Weight manager (for FP8/FP4 cached weights)
     * @param allow_quant_layer Whether quantization is allowed for this layer
     */
    template<typename Block>
    static void forward_block_modular(
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
        const ForwardHook* hook = nullptr);

    /**
     * @brief Modular backward pass for a transformer block layer
     *
     * Computes gradients in reverse order through the block.
     * Uses recipe-driven backward matmul for proper FP8/FP4 handling.
     *
     * @tparam Block The transformer block type
     * @param recipe Training recipe
     * @param rs ModularRunState
     * @param weights Block weights
     * @param grads Block gradients (output)
     * @param acts SimplifiedLayerActivations from forward
     * @param d_acts SimplifiedLayerGradients for this layer
     * @param quant_acts SimplifiedLayerQuantActivations
     * @param quant_grads SimplifiedQuantGradients
     * @param layer_idx Layer index
     * @param config Model configuration
     * @param options Model options
     * @param accumulate Whether to accumulate into existing gradients
     * @param stream CUDA stream
     * @param allow_quant_layer Whether quantization is allowed for this layer
     */
    template<typename Block>
    static void backward_block_modular(
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
        const BackwardHook* hook = nullptr);

    /**
     * @brief Modular recomputation for gradient checkpointing
     *
     * Recomputes activations during backward pass when gradient checkpointing
     * is enabled. Only recomputes what's needed based on options.
     *
     * @tparam Block The transformer block type
     * @param recipe Training recipe
     * @param rs ModularRunState
     * @param weights Block weights
     * @param acts SimplifiedLayerActivations to populate
     * @param quant_acts SimplifiedLayerQuantActivations
     * @param residual Input residual
     * @param layer_idx Layer index
     * @param config Model configuration
     * @param options Model options (determines what to recompute)
     * @param stream CUDA stream
     * @param fp8_fwd_quants Optional FP8 forward quant buffers
     * @param fp4_fwd_quants Optional FP4 forward quant buffers
     * @param weight_manager Weight manager (for FP8/FP4 cached weights)
     * @param allow_quant_layer Whether quantization is allowed for this layer
     */
    template<typename Block>
    static void recompute_block_modular(
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
        bool allow_quant_layer);

private:
    Config mConfig;
    AttentionType mAttention;
};

// ============================================================================
// Type aliases for common configurations
// ============================================================================

/// Standard dense transformer block with SwiGLU activation
using StandardDenseBlock = DenseTransformerBlock<AttentionModule, SwiGLUMLP, FusedResidualRMSNormModule>;

/// Dense block with GeGLU activation (for models that use it)
using GeGLUDenseBlock = DenseTransformerBlock<AttentionModule, GeGLUMLP, FusedResidualRMSNormModule>;

// Trait: detect DenseTransformerBlock specializations for modular dispatch
template<typename T>
struct is_dense_transformer_block : std::false_type {};

template<typename AttentionType, typename MLPType, typename NormType>
struct is_dense_transformer_block<DenseTransformerBlock<AttentionType, MLPType, NormType>> : std::true_type {};

template<typename T>
inline constexpr bool is_dense_transformer_block_v = is_dense_transformer_block<T>::value;

} // namespace modules

// Include the implementation
#include "transformer_block_impl.h"

#endif // SUROGATE_SRC_MODULES_COMPOSITE_TRANSFORMER_BLOCK_H
