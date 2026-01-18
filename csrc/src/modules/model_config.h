// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MODEL_CONFIG_H
#define SUROGATE_SRC_MODULES_MODEL_CONFIG_H

#include <optional>
#include <string>
#include <vector>

#include "config/pretrained_config.h"
#include "fp8_scaling_config.h"
#include "models/qwen25/config.h"
#include "models/qwen3moe/config.h"
#include "training/runtime_options.h"
#include "utilities/dtype.h"

namespace modules {

/**
 * @brief Architecture types supported by the modular model system
 */
enum class ArchitectureType {
    Dense,      ///< Standard dense transformer (LLaMA, Qwen, etc.)
    MoE,        ///< Mixture of Experts (Mixtral, DeepSeek, etc.)
    Hybrid      ///< Some layers dense, some MoE (Nemotron, etc.)
};

/**
 * @brief Activation function types
 */
enum class ActivationType {
    SwiGLU,     ///< SiLU-gated linear unit (LLaMA, Qwen)
    GeGLU,      ///< GELU-gated linear unit
    ReLU,       ///< Rectified Linear Unit
    GeLU,       ///< Gaussian Error Linear Unit
    SiLU        ///< Sigmoid Linear Unit (Swish)
};

/**
 * @brief Normalization types
 */
enum class NormType {
    RMSNorm,    ///< Root Mean Square normalization (LLaMA, Qwen)
    LayerNorm   ///< Standard Layer Normalization
};

/**
 * @brief Attention types
 */
enum class AttentionType {
    MHA,        ///< Multi-Head Attention
    GQA,        ///< Grouped Query Attention
    MQA         ///< Multi-Query Attention
};

/**
 * @brief MoE configuration for MoE/Hybrid architectures
 */
struct MoEConfig {
    int num_experts = 8;            ///< Total number of experts
    int top_k = 2;                  ///< Number of experts to route to per token
    bool use_shared_expert = false; ///< Nemotron/DeepSeek shared expert
    int shared_expert_size = 0;     ///< Size of shared expert (0 = same as regular)
    float router_aux_loss_coef = 0.01f;  ///< Load balancing auxiliary loss coefficient
    float router_z_loss_coef = 0.001f;   ///< Router z-loss (logit regularization) coefficient
    bool router_jitter = false;     ///< Add noise during routing for training stability
    float capacity_factor = 1.25f;  ///< Expert capacity factor for load balancing

    // Qwen3 MoE-style layer configuration
    int decoder_sparse_step = 1;    ///< MoE layer frequency: MoE every N layers (1 = all MoE)
    std::vector<int> mlp_only_layers; ///< Explicit list of layer indices using dense MLP instead of MoE
    bool norm_topk_prob = false;    ///< Normalize top-k routing weights to sum to 1 (Qwen3 style)
    int moe_intermediate_size = 0;  ///< Per-expert intermediate size (0 = use IntermediateSize)
};

/**
 * @brief Block types for heterogeneous architectures
 */
enum class BlockType {
    Dense,       ///< Standard dense transformer block
    MoE,         ///< Mixture of Experts block
    Conv,        ///< Convolutional block (for LFM2)
    SwitchMoE    ///< Switch Transformer style (top-1 routing)
};

/**
 * @brief Per-layer configuration override for hybrid architectures
 */
struct LayerOverride {
    int layer_idx;                          ///< Which layer this override applies to
    BlockType block_type = BlockType::Dense; ///< Block type for this layer
    bool is_moe = false;                    ///< Whether this layer uses MoE (deprecated, use block_type)
    std::optional<int> num_experts;         ///< Override number of experts for this layer
    std::optional<int> intermediate_size;   ///< Override intermediate size
    std::optional<int> top_k;               ///< Override top-k for MoE layers

    // Convenience constructors
    static LayerOverride dense(int idx) {
        return LayerOverride{idx, BlockType::Dense, false};
    }

    static LayerOverride moe(int idx, int experts = 8, int k = 2) {
        LayerOverride override{idx, BlockType::MoE, true};
        override.num_experts = experts;
        override.top_k = k;
        return override;
    }

    static LayerOverride switch_moe(int idx, int experts = 128) {
        LayerOverride override{idx, BlockType::SwitchMoE, true};
        override.num_experts = experts;
        override.top_k = 1;
        return override;
    }

    static LayerOverride conv(int idx) {
        return LayerOverride{idx, BlockType::Conv, false};
    }
};

/**
 * @brief Extended model configuration supporting multiple architectures
 *
 * Inherits from PretrainedConfig for backwards compatibility while adding
 * support for MoE, different activation functions, and per-layer overrides.
 */
struct ModelConfig : public PretrainedConfig {
    // Original pretrained config (preserves derived type like Qwen3MoEConfig)
    // This is used for config saving to preserve all fields
    std::shared_ptr<PretrainedConfig> original_config;

    // Architecture specification
    ArchitectureType architecture = ArchitectureType::Dense;
    ActivationType activation_type = ActivationType::SwiGLU;
    NormType norm_type = NormType::RMSNorm;
    AttentionType attention_type = AttentionType::GQA;

    // MoE configuration (only used when architecture != Dense)
    std::optional<MoEConfig> moe_config;

    // Per-layer overrides for hybrid architectures
    std::vector<LayerOverride> layer_overrides;

    // Additional architecture features
    bool use_parallel_residual = false;  ///< GPT-NeoX style parallel attention + FFN
    bool use_qk_norm = false;            ///< Normalize Q and K in attention
    bool use_sliding_window = false;     ///< Sliding window attention (Mistral)
    int sliding_window_size = 4096;      ///< Size of sliding window

    // MoE convenience fields (copied from moe_config for direct access)
    // These are populated by from_pretrained_config when moe_config is set
    int NumExperts = 0;              ///< Number of experts (0 = dense model)
    int NumExpertsPerTok = 0;        ///< Top-K experts per token
    int MoeIntermediateSize = 0;     ///< Per-expert intermediate size

    // Extended RoPE configuration
    float rope_scaling_factor = 1.0f;    ///< RoPE scaling for longer contexts
    std::string rope_type = "default";   ///< RoPE type: "default", "linear", "dynamic", "yarn"

    /**
     * @brief Construct from existing PretrainedConfig
     *
     * Handles the config inheritance hierarchy, extracting MoE-specific
     * fields from Qwen3MoEConfig if applicable.
     */
    static ModelConfig from_pretrained_config(const PretrainedConfig& base) {
        ModelConfig config;

        // Store the original config to preserve derived type (e.g., Qwen3MoEConfig)
        config.original_config = base.clone();

        // Copy base fields
        config.Architecture = base.Architecture;
        config.BosTokenId = base.BosTokenId;
        config.EosTokenId = base.EosTokenId;
        config.PadTokenId = base.PadTokenId;
        config.HiddenSize = base.HiddenSize;
        config.IntermediateSize = base.IntermediateSize;
        config.VocabSize = base.VocabSize;
        config.NumQueryHeads = base.NumQueryHeads;
        config.NumKeyValHeads = base.NumKeyValHeads;
        config.NumLayers = base.NumLayers;
        config.HeadDim = base.HeadDim;
        config.MaxPositionEmbeddings = base.MaxPositionEmbeddings;
        config.RopeTheta = base.RopeTheta;
        config.Rope = base.Rope;
        config.RmsNormEps = base.RmsNormEps;
        config.TiedWordEmbeddings = base.TiedWordEmbeddings;
        config.UseQKVBias = base.UseQKVBias;
        config.UseQKNorm = base.UseQKNorm;
        config.DType = base.DType;

        // Copy QK norm setting
        config.use_qk_norm = base.has_qk_norm();

        // Check for sliding window (Qwen2Config and derived)
        if (const auto* qwen2 = dynamic_cast<const Qwen2Config*>(&base)) {
            config.use_sliding_window = qwen2->SlidingWindow > 0;
            config.sliding_window_size = qwen2->SlidingWindow;
        }

        // Infer attention type from head counts
        if (config.NumKeyValHeads == 1) {
            config.attention_type = AttentionType::MQA;
        } else if (config.NumKeyValHeads < config.NumQueryHeads) {
            config.attention_type = AttentionType::GQA;
        } else {
            config.attention_type = AttentionType::MHA;
        }

        // Check for MoE configuration (Qwen3MoEConfig)
        if (const auto* moe_cfg = dynamic_cast<const Qwen3MoEConfig*>(&base)) {
            if (moe_cfg->NumExperts > 0) {
                config.architecture = ArchitectureType::MoE;

                // Set up MoE config from Qwen3MoEConfig fields
                MoEConfig moe;
                moe.num_experts = moe_cfg->NumExperts;
                moe.top_k = moe_cfg->NumExpertsPerTok;
                moe.moe_intermediate_size = moe_cfg->MoeIntermediateSize;
                moe.decoder_sparse_step = moe_cfg->DecoderSparseStep;
                moe.mlp_only_layers = moe_cfg->MlpOnlyLayers;
                moe.norm_topk_prob = moe_cfg->NormTopkProb;
                moe.router_aux_loss_coef = moe_cfg->RouterAuxLossCoef;
                moe.router_z_loss_coef = moe_cfg->RouterZLossCoef;
                config.moe_config = moe;

                // Also populate convenience fields for direct access
                config.NumExperts = moe_cfg->NumExperts;
                config.NumExpertsPerTok = moe_cfg->NumExpertsPerTok;
                config.MoeIntermediateSize = moe_cfg->MoeIntermediateSize;
            }
        }

        return config;
    }

    /**
     * @brief Check if a specific layer uses MoE
     *
     * For MoE/Hybrid architectures, this follows the Qwen3 MoE pattern:
     * 1. If layer is in mlp_only_layers, it's NOT MoE (dense MLP)
     * 2. Otherwise, if (layer_idx + 1) % decoder_sparse_step == 0, it's MoE
     * 3. Layer overrides take precedence over the above rules
     */
    [[nodiscard]] bool is_layer_moe(int layer_idx) const {
        if (architecture == ArchitectureType::Dense) {
            return false;
        }

        // Check explicit layer overrides first (highest priority)
        for (const auto& override : layer_overrides) {
            if (override.layer_idx == layer_idx) {
                return override.block_type == BlockType::MoE ||
                       override.block_type == BlockType::SwitchMoE ||
                       override.is_moe;  // Backwards compatibility
            }
        }

        // For MoE/Hybrid architectures, use Qwen3-style pattern
        if (architecture == ArchitectureType::MoE || architecture == ArchitectureType::Hybrid) {
            if (moe_config.has_value()) {
                const auto& moe = moe_config.value();

                // Check if this layer is explicitly marked as dense (mlp_only_layers)
                for (int dense_layer : moe.mlp_only_layers) {
                    if (dense_layer == layer_idx) {
                        return false;
                    }
                }

                // Check decoder_sparse_step pattern: MoE if (layer_idx + 1) % step == 0
                // With step=1, all layers are MoE (default behavior)
                // With step=2, layers 1,3,5,... are MoE (every other layer starting from 1)
                if (moe.num_experts > 0) {
                    return (layer_idx + 1) % moe.decoder_sparse_step == 0;
                }
            }
        }

        // Default: MoE architecture means all layers are MoE, Hybrid means dense
        return architecture == ArchitectureType::MoE;
    }

    /**
     * @brief Get the block type for a specific layer
     */
    [[nodiscard]] BlockType get_block_type(int layer_idx) const {
        if (architecture == ArchitectureType::Dense) {
            return BlockType::Dense;
        }
        if (architecture == ArchitectureType::MoE) {
            return BlockType::MoE;
        }
        // Hybrid: check overrides
        for (const auto& override : layer_overrides) {
            if (override.layer_idx == layer_idx) {
                return override.block_type;
            }
        }
        return BlockType::Dense;  // Default to dense for hybrid
    }

    /**
     * @brief Get top-k for routing in a specific layer
     */
    [[nodiscard]] int get_top_k(int layer_idx) const {
        // Check for layer override
        for (const auto& override : layer_overrides) {
            if (override.layer_idx == layer_idx && override.top_k.has_value()) {
                return override.top_k.value();
            }
        }
        // Default from MoE config
        return moe_config.has_value() ? moe_config->top_k : 2;
    }

    /**
     * @brief Get number of experts for a specific layer
     */
    [[nodiscard]] int get_num_experts(int layer_idx) const {
        if (!is_layer_moe(layer_idx)) {
            return 1;
        }
        // Check for layer override
        for (const auto& override : layer_overrides) {
            if (override.layer_idx == layer_idx && override.num_experts.has_value()) {
                return override.num_experts.value();
            }
        }
        return moe_config.has_value() ? moe_config->num_experts : 8;
    }

    /**
     * @brief Get intermediate size for a specific layer
     */
    [[nodiscard]] int get_intermediate_size(int layer_idx) const {
        // Check for layer override
        for (const auto& override : layer_overrides) {
            if (override.layer_idx == layer_idx && override.intermediate_size.has_value()) {
                return override.intermediate_size.value();
            }
        }
        return IntermediateSize;
    }
};

/**
 * @brief Runtime options for modular models
 *
 * Extended from RuntimeOptions with module-specific settings.
 */
struct ModelOptions {
    // Recomputation flags
    bool recompute_swiglu = false;
    bool recompute_rmsnorm = false;
    bool recompute_ffn = false;
    bool recompute_qkv = false;
    bool recompute_attention = false;
    bool recompute_block = false;
    bool offload_residuals = false;

    // LM head settings
    int lmhead_chunks = 1;
    int attention_bwd_chunks = 1;

    // CUDA graphs
    bool use_cuda_graphs = false;
    bool trigger_timing_events = false;

    // Offloading
    bool offload_master = false;
    bool offload_quants = false;
    bool offload_optimizer = false;
    bool offload_grads = false;
    bool use_zero_copy = false;
    bool use_write_combined = false;

    // Sharding
    bool shard_weights = false;
    bool shard_gradients = false;
    bool use_all_to_all_reduce = false;
    bool persistent_quants = false;

    // Initialization
    bool init_projections_to_zero = false;

    // QLoRA: skip block weight allocation (weights provided externally)
    bool skip_block_allocation = false;

    // LoRA-only mode: optimize activation storage and skip computing frozen weight gradients
    // When enabled, only stores activations needed for LoRA gradient computation
    bool lora_only_mode = false;

    // LoRA activation recomputation: recompute ln1/ln2/att during LoRA backward
    // instead of storing per-layer. Enables activation sharing even in LoRA mode.
    bool recompute_lora = false;

    // Skip base model gradient allocation (used in LoRA mode where base weights are frozen)
    bool skip_base_gradients = false;

    // Train MoE router gate during LoRA fine-tuning (router gradients computed even in lora_only mode)
    bool train_router = false;

    // MoE loss coefficients (override model config when >= 0)
    float router_aux_loss_coef = -1.0f;  ///< Load balancing auxiliary loss coefficient (-1 = use model config)
    float router_z_loss_coef = -1.0f;    ///< Router z-loss (logit regularization) coefficient (-1 = use model config)

    // Data types
    std::optional<ETensorDType> model_dtype;
    std::optional<ETensorDType> matmul_dtype;
    std::optional<ETensorDType> gradient_dtype;
    std::optional<ETensorDType> master_dtype;

    // FP8 forward-only mode: use FP8 for forward pass matmuls, keep BF16 for backward
    bool enable_fp8_forward = false;

    // FP8 HYBRID format: E4M3 for forward, E5M2 for backward gradients
    // E5M2 has larger dynamic range (max=57344) suitable for gradients
    // When enabled, delayed scaling is automatically used for improved accuracy.
    bool enable_fp8_hybrid = false;

    // FP8 Delayed Scaling Configuration (TransformerEngine-style)
    // Automatically enabled when enable_fp8_hybrid is true.
    // These are hardcoded defaults following TransformerEngine's DelayedScaling recipe.
    FP8ScalingConfig fp8_scaling_config;

    // FP4 Training Options (requires Blackwell SM100+)
    // FP4 uses E2M1 format with two-level block scaling for extreme memory efficiency.

    // FP4 forward-only: quantize activations to FP4 for forward matmuls
    // Gradients remain in BF16/FP8 for stability.
    bool enable_fp4_forward = false;

    // FP4 forward+backward: use FP4 for both forward and backward matmuls
    // Backward uses stochastic rounding for gradient quantization.
    bool enable_fp4_backward = false;

    // Skip quantization for first/last N layers (embedding/lm_head layers)
    // Useful for FP4/FP8 stability - keeps these layers in BF16
    int skip_quant_first_layers = 0;
    int skip_quant_last_layers = 0;

    // Fused RoPE: compute cos/sin on-the-fly with shared memory caching (TransformerEngine-style).
    // Eliminates precomputed freq_cis tensor, reduces memory bandwidth.
    bool use_fused_rope = false;

    // Matmul backend selection for FP8 operations
    // AUTO: Let the system auto-detect (CUTLASS for SM120+ FP8, cuBLAS otherwise)
    // CUBLASLT: Force cuBLAS Lt (per-tensor FP8 scaling)
    // CUTLASS: Force CUTLASS (SM90: per-tensor, SM120+: block-scaled MX FP8)
    EMatmulBackend matmul_backend = EMatmulBackend::AUTO;

    // Use modular block implementation instead of legacy optimized kernel path.
    // When enabled, uses DenseTransformerBlock::forward_impl/backward_impl
    // instead of the hand-optimized kernel calls in model_forward.hpp/model_block_ops.hpp.
    // Default is enabled; set false to force the legacy path.
    bool use_modular_blocks = true;

    [[nodiscard]] ETensorDType get_matmul_dtype() const {
        return matmul_dtype.value_or(model_dtype.value());
    }

    [[nodiscard]] ETensorDType get_grad_dtype() const {
        // HYBRID format: use E5M2 for backward gradients (larger dynamic range)
        if (enable_fp8_hybrid) return ETensorDType::FP8_E5M2;
        return gradient_dtype.value_or(get_matmul_dtype());
    }

    // Returns FP8 E4M3 when FP8 forward (or HYBRID) is enabled, otherwise falls back to matmul_dtype
    [[nodiscard]] ETensorDType get_forward_matmul_dtype() const {
        if (enable_fp8_forward || enable_fp8_hybrid) return ETensorDType::FP8_E4M3;
        return get_matmul_dtype();
    }

    // Returns FP8 E5M2 when HYBRID mode is set, otherwise falls back to get_grad_dtype
    [[nodiscard]] ETensorDType get_backward_matmul_dtype() const {
        if (enable_fp8_hybrid) return ETensorDType::FP8_E5M2;
        return get_grad_dtype();
    }

    [[nodiscard]] EAllocationType get_offload_alloc() const {
        return use_write_combined ? EAllocationType::WRITE_CMB : EAllocationType::PINNED;
    }

    // FP4 helper methods
    [[nodiscard]] bool fp4_enabled() const {
        return enable_fp4_forward || enable_fp4_backward;
    }

    [[nodiscard]] bool fp4_forward_enabled() const {
        return enable_fp4_forward || enable_fp4_backward;
    }

    [[nodiscard]] bool fp4_backward_enabled() const {
        return enable_fp4_backward;
    }

    [[nodiscard]] RuntimeOptions to_runtime_options() const {
        RuntimeOptions opts;
        opts.RecomputeSwiGLu = recompute_swiglu;
        opts.RecomputeRMSNorm = recompute_rmsnorm;
        opts.RecomputeFFN = recompute_ffn;
        opts.RecomputeQKV = recompute_qkv;
        opts.RecomputeAtt = recompute_attention;
        opts.RecomputeBlock = recompute_block;
        opts.OffloadResidual = offload_residuals;
        opts.LMHeadChunks = lmhead_chunks;
        opts.AttBwdChunks = attention_bwd_chunks;
        opts.UseCudaGraphs = use_cuda_graphs;
        opts.TriggerTimingEvents = trigger_timing_events;
        opts.OffloadMaster = offload_master;
        opts.OffloadQuants = offload_quants;
        opts.OffloadOptimizer = offload_optimizer;
        opts.OffloadGrads = offload_grads;
        opts.UseZeroCopy = use_zero_copy;
        opts.UseWriteCombined = use_write_combined;
        opts.ShardWeights = shard_weights;
        opts.ShardGradients = shard_gradients;
        opts.UseAllToAllReduce = use_all_to_all_reduce;
        opts.PersistentQuants = persistent_quants;
        opts.InitProjectionsToZero = init_projections_to_zero;
        // Note: FP8/FP4 flags are now managed via TrainingRecipe in RuntimeOptions
        // The recipe should be set separately after calling to_runtime_options()
        opts.UseFusedRope = use_fused_rope;
        opts.MatmulBackend = matmul_backend;
        opts.UseModularBlocks = use_modular_blocks;
        opts.ModelType = model_dtype;
        opts.MatmulType = matmul_dtype;
        opts.GradientType = gradient_dtype;
        opts.MasterDType = master_dtype;
        return opts;
    }

    static ModelOptions from_runtime_options(const RuntimeOptions& opts) {
        ModelOptions options;
        options.recompute_swiglu = opts.RecomputeSwiGLu;
        options.recompute_rmsnorm = opts.RecomputeRMSNorm;
        options.recompute_ffn = opts.RecomputeFFN;
        options.recompute_qkv = opts.RecomputeQKV;
        options.recompute_attention = opts.RecomputeAtt;
        options.recompute_block = opts.RecomputeBlock;
        options.recompute_lora = opts.RecomputeLoRA;
        options.offload_residuals = opts.OffloadResidual;
        options.lmhead_chunks = opts.LMHeadChunks;
        options.attention_bwd_chunks = opts.AttBwdChunks;
        options.use_cuda_graphs = opts.UseCudaGraphs;
        options.trigger_timing_events = opts.TriggerTimingEvents;
        options.offload_master = opts.OffloadMaster;
        options.offload_quants = opts.OffloadQuants;
        options.offload_optimizer = opts.OffloadOptimizer;
        options.offload_grads = opts.OffloadGrads;
        options.use_zero_copy = opts.UseZeroCopy;
        options.use_write_combined = opts.UseWriteCombined;
        options.shard_weights = opts.ShardWeights;
        options.shard_gradients = opts.ShardGradients;
        options.use_all_to_all_reduce = opts.UseAllToAllReduce;
        options.persistent_quants = opts.PersistentQuants;
        options.init_projections_to_zero = opts.InitProjectionsToZero;
        // Derive FP8/FP4 flags from the recipe
        options.enable_fp8_hybrid = opts.fp8_hybrid_enabled();
        options.enable_fp8_forward = opts.fp8_forward_enabled();
        options.enable_fp4_forward = opts.fp4_forward_enabled();
        options.enable_fp4_backward = opts.fp4_backward_enabled();
        // Skip layers are now managed via CLI options
        options.skip_quant_first_layers = opts.RecipeOptions.skip_quant_first_layers;
        options.skip_quant_last_layers = opts.RecipeOptions.skip_quant_last_layers;
        // nvfp4-simple: check if recipe wants to skip embedding/lm_head
        if (opts.TrainingRecipe && opts.TrainingRecipe->skip_embedding_lmhead_quant()) {
            options.skip_quant_first_layers = std::max(options.skip_quant_first_layers, 1);
            options.skip_quant_last_layers = std::max(options.skip_quant_last_layers, 1);
        }
        options.use_fused_rope = opts.UseFusedRope;
        options.matmul_backend = opts.MatmulBackend;
        options.use_modular_blocks = opts.UseModularBlocks;
        options.model_dtype = opts.ModelType;
        options.matmul_dtype = opts.MatmulType;
        options.gradient_dtype = opts.GradientType;
        options.master_dtype = opts.MasterDType;
        // MoE loss coefficients
        options.router_aux_loss_coef = opts.RouterAuxLossCoef;
        options.router_z_loss_coef = opts.RouterZLossCoef;
        return options;
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODEL_CONFIG_H
