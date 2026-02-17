// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_TRAINING_RUNTIME_OPTIONS_H
#define SUROGATE_SRC_TRAINING_RUNTIME_OPTIONS_H

#include <memory>
#include <optional>
#include <string>

#include "runtime/training/matmul_backend.h"  // EMatmulBackend
#include "recipes/recipe.h"
#include "recipes/recipe_factory.h"
#include "utilities/allocator.h"
#include "utilities/dtype.h"

// ============================================================================
// Recomputation levels for activation checkpointing
// ============================================================================
// Recomputation trades compute for memory by recomputing activations during backward.
// Set via `recompute: true` or `recompute: false` in config.
//
// - false (None):    Save all activations. Maximum memory, fastest training.
//                    Guarantees bit-exact gradients.
// - true (Enabled):  Recompute intermediates from checkpoints (saves ~17% VRAM).
//                    In FFT mode, saves ln1/ln2 rstd and recomputes bit-exactly.
//                    In LoRA mode, fully recomputes attention and FFN segments.
//
enum class RecomputeLevel {
    None = 0,    ///< recompute: false - save all activations
    Enabled = 1, ///< recompute: true - recompute intermediates from checkpoints
};

// Runtime/training options used by the CLI and python bindings.
// The modular system consumes these via modules::ModelOptions::from_runtime_options().
struct RuntimeOptions {
    bool KeepAllActivations = false;

    // ========================================================================
    // Recomputation configuration
    // ========================================================================
    // Controls activation checkpointing strategy for memory-compute tradeoff.
    // See RecomputeLevel enum above for level descriptions.
    RecomputeLevel Recompute = RecomputeLevel::None;
    bool OffloadResidual = false;
    int LMHeadChunks = 1;
    int AttBwdChunks = 1;
    bool UseCudaGraphs = false;
    bool TriggerTimingEvents = false;

    bool OffloadMaster = false;
    bool OffloadQuants = false;
    bool OffloadOptimizer = false;
    bool OffloadGrads  = false;
    bool UseZeroCopy   = false;
    bool UseWriteCombined = false;
    bool ShardWeights = false;
    bool PersistentQuants = false;

    bool ShardGradients = false;
    bool UseAllToAllReduce = false;

    bool InitProjectionsToZero = false;

    // MoE optimization: Only dequantize selected experts (reduces memory from O(num_experts) to O(top_k))
    bool SelectiveExpertDequant = true;

    // MoE optimization: Offload expert NF4 weights to CPU, stream on-demand (saves ~12GB for 128-expert models)
    bool OffloadExperts = false;

    // Expert Parallelism: distribute MoE experts across GPUs (1 = no EP, all experts replicated)
    int EPSize = 1;

    // LLEP adaptive threshold: when max_gpu_load / mean_gpu_load exceeds this, LPT load balancing activates.
    // Only relevant when EPSize > 1. Values close to 1.0 = aggressive rebalancing, higher = less rebalancing.
    float EPLoadBalanceThreshold = 1.3f;

    // MoE loss coefficients (override model config when >= 0)
    float RouterAuxLossCoef = -1.0f;  ///< Load balancing auxiliary loss coefficient (-1 = use model config)
    float RouterZLossCoef = -1.0f;    ///< Router z-loss (logit regularization) coefficient (-1 = use model config)

    // Debug: print detailed memory breakdown after model allocation (useful for QLoRA optimization)
    bool DebugMemoryBreakdown = false;

    // Training recipe - defines quantization strategy for forward/backward passes.
    // Default is BF16 (no quantization). Set via --recipe=<name> CLI flag.
    std::shared_ptr<recipes::Recipe> TrainingRecipe;

    // Recipe-specific options parsed from CLI
    recipes::RecipeConfig RecipeOptions;

    // Fused RoPE: compute cos/sin on-the-fly with shared memory caching.
    // Eliminates precomputed freq_cis tensor, reduces memory bandwidth.
    bool UseFusedRope = false;

    // DSL IR execution (deprecated flag; DSL backend is always used).
    bool UseDslIr = true;
    std::string DslIrJson;

    // Matmul backend selection
    // AUTO: Let the system auto-detect (CUTLASS for SM120+ FP8, cuBLAS otherwise)
    // CUBLASLT: Force cuBLAS Lt (per-tensor FP8 scaling)
    // CUTLASS: Force CUTLASS (SM90: per-tensor, SM120+: block-scaled MX FP8)
    EMatmulBackend MatmulBackend = EMatmulBackend::AUTO;

    // ModelType is just a copy of the dtype set in config
    std::optional<ETensorDType> ModelType = std::nullopt;
    std::optional<ETensorDType> MatmulType = std::nullopt;
    std::optional<ETensorDType> GradientType = std::nullopt;
    std::optional<ETensorDType> MasterDType = std::nullopt;

    [[nodiscard]] ETensorDType matmul_dtype() const {
        return MatmulType.value_or(ModelType.value());
    }

    [[nodiscard]] ETensorDType grad_dtype() const {
        // FP8 HYBRID: use E5M2 for backward gradients (larger dynamic range)
        if (TrainingRecipe && TrainingRecipe->is_fp8_hybrid()) {
            return ETensorDType::FP8_E5M2;
        }
        return GradientType.value_or(matmul_dtype());
    }

    // Returns FP4 E2M1 when FP4 recipe is active, FP8 E4M3 when FP8 recipe is active,
    // otherwise falls back to matmul_dtype()
    [[nodiscard]] ETensorDType forward_matmul_dtype() const {
        if (TrainingRecipe && (TrainingRecipe->is_nvfp4() || TrainingRecipe->is_nvfp4_cutlass())) {
            return ETensorDType::FP4_E2M1;
        }
        if (TrainingRecipe && TrainingRecipe->is_fp8_hybrid()) {
            return ETensorDType::FP8_E4M3;
        }
        return matmul_dtype();
    }

    // Returns the actual compute dtype for speed-of-light (SOL) estimation.
    // For QLoRA, this returns BF16 because QLoRA dequantizes FP4/FP8 weights to BF16
    // before matmul (the quantized format is for storage, not compute).
    // For non-QLoRA FP8/FP4 recipes, returns the actual compute dtype.
    [[nodiscard]] ETensorDType sol_compute_dtype(bool is_qlora) const {
        if (is_qlora) {
            // QLoRA always dequantizes to BF16 for compute
            return ETensorDType::BF16;
        }
        return forward_matmul_dtype();
    }

    // Returns FP8 E5M2 when FP8 HYBRID recipe is set, otherwise falls back to grad_dtype()
    [[nodiscard]] ETensorDType backward_matmul_dtype() const {
        if (TrainingRecipe && TrainingRecipe->is_fp8_hybrid()) {
            return ETensorDType::FP8_E5M2;
        }
        return grad_dtype();
    }

    // Check if any FP4 recipe is active
    [[nodiscard]] bool fp4_enabled() const {
        return TrainingRecipe && (TrainingRecipe->is_nvfp4() || TrainingRecipe->is_nvfp4_cutlass());
    }

    // Check if FP4 forward pass is enabled
    [[nodiscard]] bool fp4_forward_enabled() const {
        return fp4_enabled();
    }

    // Check if FP4 backward pass is enabled
    [[nodiscard]] bool fp4_backward_enabled() const {
        return fp4_enabled();
    }

    // Check if FP8 forward is enabled
    [[nodiscard]] bool fp8_forward_enabled() const {
        return TrainingRecipe && TrainingRecipe->is_fp8_hybrid();
    }

    // Check if FP8 hybrid backward is enabled
    [[nodiscard]] bool fp8_hybrid_enabled() const {
        return TrainingRecipe && TrainingRecipe->is_fp8_hybrid();
    }

    // Get recipe name (or "bf16" if no recipe set)
    [[nodiscard]] std::string_view recipe_name() const {
        if (TrainingRecipe) {
            return TrainingRecipe->name();
        }
        return "bf16";
    }

    [[nodiscard]] EAllocationType offload_alloc() const {
        return UseWriteCombined ? EAllocationType::WRITE_CMB : EAllocationType::PINNED;
    }

    // ========================================================================
    // Recomputation level helpers
    // ========================================================================

    /// Returns true if any recomputation is enabled (level > None)
    [[nodiscard]] bool recompute_enabled() const {
        return Recompute != RecomputeLevel::None;
    }

    /// Returns the recompute level as a string for logging
    [[nodiscard]] std::string_view recompute_level_name() const {
        return Recompute == RecomputeLevel::Enabled ? "true" : "false";
    }

    /// Parse recompute level from string (for Python bindings)
    /// Primary values: "true"/"false". Legacy values also accepted for backward compatibility.
    static RecomputeLevel parse_recompute_level(const std::string& level) {
        if (level == "false" || level == "none" || level == "0") {
            return RecomputeLevel::None;
        }
        if (level == "true" || level == "1") {
            return RecomputeLevel::Enabled;
        }
        throw std::invalid_argument("Invalid recompute level: " + level +
                                    ". Valid values: true, false");
    }
};

// Backwards-compatible alias for existing user code/bindings.
using LLamaOptions = RuntimeOptions;

#endif // SUROGATE_SRC_TRAINING_RUNTIME_OPTIONS_H
