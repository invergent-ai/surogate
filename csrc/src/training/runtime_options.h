// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_TRAINING_RUNTIME_OPTIONS_H
#define SUROGATE_SRC_TRAINING_RUNTIME_OPTIONS_H

#include <memory>
#include <optional>
#include <string>

#include "training/matmul_backend.h"  // EMatmulBackend
#include "recipes/recipe.h"
#include "recipes/recipe_factory.h"
#include "utilities/allocator.h"
#include "utilities/dtype.h"

// Runtime/training options used by the CLI and python bindings.
// The modular system consumes these via modules::ModelOptions::from_runtime_options().
struct RuntimeOptions {
    bool KeepAllActivations = false;

    // ========================================================================
    // Fine-grained recomputation flags (per-component instead of per-layer)
    // ========================================================================
    // These provide memory-compute tradeoffs at different granularities.
    // When RecomputeBlock=true (default), all component flags are effectively enabled.
    //
    // Memory savings per component (approximate for 7B model, B=4, T=2048):
    //   - swiglu:    ~200MB  (B*T*2*D tensor saved)
    //   - mlp_up:    ~400MB  (B*T*2*D tensor saved)
    //   - qkv:       ~200MB  (B*T*QKV_C tensor saved)
    //   - att:       ~150MB  (B*T*Hq*Hs tensor saved)
    //   - att_out:   ~100MB  (B*T*C tensor saved)
    //   - ln1/ln2:    ~50MB each (B*T*C tensors + rstd)
    //
    // Compute cost per recompute (relative):
    //   - swiglu:    ~0.5%   (elementwise, very cheap)
    //   - rmsnorm:   ~1%     (reduction, cheap)
    //   - qkv:       ~15%    (large matmul)
    //   - att:       ~30%    (flash attention)
    //   - att_out:   ~10%    (matmul)
    //   - mlp_up:    ~15%    (large matmul)
    //   - mlp_down:  ~15%    (large matmul)
    //
    bool RecomputeSwiGLu = false;    ///< Recompute SwiGLU activation in FFN backward
    bool RecomputeRMSNorm = false;   ///< Recompute RMSNorm (LN1/LN2) forward
    bool RecomputeFFN = false;       ///< Recompute FFN/MLP up projection (implies RecomputeSwiGLu)
    bool RecomputeMLPDown = false;   ///< Recompute MLP down projection (rarely needed, output saved anyway)
    bool RecomputeQKV = false;       ///< Recompute QKV projection
    bool RecomputeQKNorm = false;    ///< Recompute QK head normalization (Qwen3-style)
    bool RecomputeRoPE = false;      ///< Recompute RoPE rotation (cheap, usually saved for QK-norm backward)
    bool RecomputeAtt = false;       ///< Recompute attention (flash attention forward) (implies RecomputeQKV)
    bool RecomputeOutProj = false;   ///< Recompute attention output projection
    bool RecomputeBlock = true;      ///< Recompute entire layer (enables all component flags)
    bool RecomputeLoRA = false;      ///< Recompute ln1/ln2 during LoRA backward (saves ~350MB for 4B models)
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

    // DSL IR execution (placeholder backend).
    bool UseDslIr = true;
    std::string DslIrJson;

    // Compiled DSL execution: eliminate operation dispatch overhead by pre-compiling
    // operations into direct function pointer calls with pre-resolved tensors/attrs.
    // This can improve throughput by 5-10% by eliminating:
    // - Runtime string comparisons (op_type == "embedding")
    // - Hash map lookups for tensor resolution (get_tensor())
    // - Attribute parsing (find_attr(), attr_double())
    // - Shape resolution (resolve_shape())
    bool UseCompiledDsl = true;

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
};

// Backwards-compatible alias for existing user code/bindings.
using LLamaOptions = RuntimeOptions;

#endif // SUROGATE_SRC_TRAINING_RUNTIME_OPTIONS_H
