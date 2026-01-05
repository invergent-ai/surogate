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
    bool RecomputeSwiGLu = false;
    bool RecomputeRMSNorm = false;
    bool RecomputeFFN = false;
    bool RecomputeQKV = false;
    bool RecomputeAtt = false;
    bool RecomputeBlock = true;
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

    // Returns FP8 E4M3 when FP8 recipe is active, otherwise falls back to matmul_dtype()
    [[nodiscard]] ETensorDType forward_matmul_dtype() const {
        if (TrainingRecipe && TrainingRecipe->is_fp8_hybrid()) {
            return ETensorDType::FP8_E4M3;
        }
        return matmul_dtype();
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

    // Check if scaled SwiGLU is required by the current recipe
    [[nodiscard]] bool scaled_swiglu_enabled() const {
        return TrainingRecipe && TrainingRecipe->requires_scaled_swiglu();
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
