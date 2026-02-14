// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MATMUL_CONTEXT_H
#define SUROGATE_SRC_MODULES_MATMUL_CONTEXT_H

#include <cstddef>
#include <optional>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "utilities/tensor.h"

// Forward declaration of the global IRunState (defined in runtime/training/model.h)
class IRunState;
typedef struct cudnnContext* cudnnHandle_t;

namespace modules {

/**
 * @brief Matmul operation identifiers within a transformer block
 *
 * Each enum value corresponds to a specific projection in the transformer.
 * Used to identify which matmul is being executed for recipe-specific handling.
 */
enum class MatmulOp {
    QKV,        ///< Query/Key/Value projection: (B,T,C) -> (B,T,QKV_C)
    AttnOut,    ///< Attention output projection: (B,T,Hq*Hs) -> (B,T,C)
    MLPUp,      ///< MLP gate+up projection: (B,T,C) -> (B,T,2*D)
    MLPDown,    ///< MLP down projection: (B,T,D) -> (B,T,C)
    Embedding,  ///< Embedding lookup (not a matmul, but included for completeness)
    LMHead,     ///< Language model head projection: (B,T,C) -> (B,T,V)
};

/**
 * @brief Convert MatmulOp to string for debugging
 */
constexpr const char* matmul_op_name(MatmulOp op) {
    switch (op) {
        case MatmulOp::QKV: return "QKV";
        case MatmulOp::AttnOut: return "AttnOut";
        case MatmulOp::MLPUp: return "MLPUp";
        case MatmulOp::MLPDown: return "MLPDown";
        case MatmulOp::Embedding: return "Embedding";
        case MatmulOp::LMHead: return "LMHead";
        default: return "Unknown";
    }
}

/**
 * @brief Context passed to recipe matmul methods
 *
 * Contains all information needed to execute a forward or backward matmul.
 * This struct is passed by reference to recipe methods, allowing them to
 * access tensors and runtime state without needing template parameters.
 *
 * Usage:
 * @code
 * MatmulContext ctx;
 * ctx.out = &acts.qkv;
 * ctx.inp = &acts.ln1;
 * ctx.weight = &weights.attention.qkv_weight;
 * ctx.B = B; ctx.T = T; ctx.C_in = C; ctx.C_out = qkv_channels;
 * ctx.run_state = &rs;
 * ctx.stream = main_stream;
 * ctx.layer_idx = l;
 * ctx.op = MatmulOp::QKV;
 * recipe->forward_matmul(ctx);
 * @endcode
 */
struct MatmulContext {
    // =========================================================================
    // Forward pass tensors
    // =========================================================================

    Tensor* out = nullptr;           ///< Output tensor (forward: result, backward: dinp)
    Tensor* inp = nullptr;           ///< Input activation
    Tensor* weight = nullptr;        ///< Weight matrix
    Tensor* bias = nullptr;          ///< Optional bias (nullptr if no bias)

    // =========================================================================
    // Backward pass tensors (only used for backward_matmul)
    // =========================================================================

    Tensor* dinp = nullptr;          ///< Gradient w.r.t. input
    Tensor* dweight = nullptr;       ///< Gradient w.r.t. weight
    Tensor* dbias = nullptr;         ///< Gradient w.r.t. bias (nullptr if no bias)
    Tensor* dout = nullptr;          ///< Upstream gradient

    // =========================================================================
    // Dimensions
    // =========================================================================

    int B = 0;                       ///< Batch size
    int T = 0;                       ///< Sequence length
    int C_in = 0;                    ///< Input channels (K dimension)
    int C_out = 0;                   ///< Output channels (N dimension)

    // =========================================================================
    // Runtime state
    // =========================================================================

    IRunState* run_state = nullptr;  ///< Run state with handles, workspace, etc.
    cudaStream_t stream = nullptr;   ///< CUDA stream for operations

    // =========================================================================
    // Operation metadata
    // =========================================================================

    int layer_idx = 0;               ///< Current transformer layer index (0-based)
    MatmulOp op = MatmulOp::QKV;     ///< Which matmul operation

    // =========================================================================
    // Backward-specific flags
    // =========================================================================

    bool accumulate = false;         ///< Accumulate into gradient buffers (vs overwrite)
    bool skip_weight_grad = false;   ///< Skip weight gradient computation (LoRA-only mode)
    unsigned int seed = 0;           ///< Random seed for stochastic rounding (FP4 backward)
    bool allow_fp4 = true;           ///< Allow FP4 quantization (false for skip_quant layers)
    bool allow_fp8 = true;           ///< Allow FP8 quantization (false for skip_quant layers)

    // =========================================================================
    // Quantization buffers (set by caller for FP8/FP4 recipes)
    // =========================================================================

    Tensor* inp_quant = nullptr;     ///< Pre-allocated quant buffer for input (FP8/FP4)
    const Tensor* cached_weight = nullptr;  ///< Cached quantized weight (FP8, optional)
    int delayed_quantizer_idx = -1;  ///< Quantizer index for delayed scaling (-1 = JIT scaling)

    // FP4 cached weight (CUTLASS layout) - used by NVFP4Recipe when available
    const Tensor* cached_fp4_data = nullptr;    ///< Cached FP4 packed data (N, K/2)
    const Tensor* cached_fp4_scales = nullptr;  ///< Cached FP4 block scales (FP8 E4M3, CUTLASS layout)
    const float* cached_fp4_amax = nullptr;     ///< Pointer to cached global amax (device memory)

    // =========================================================================
    // Backward-specific quantization buffers (for FP8 hybrid backward)
    // =========================================================================

    Tensor* dout_quant = nullptr;    ///< E5M2 gradient buffer for upstream gradient
    Tensor* bias_buffer = nullptr;   ///< Scratch buffer for bias gradient computation

    // =========================================================================
    // Recipe-specific state (optional, set by recipe)
    // =========================================================================

    void* recipe_state = nullptr;    ///< Opaque pointer for recipe-specific data

    // =========================================================================
    // Convenience methods
    // =========================================================================

    /**
     * @brief Check if this is a forward pass context
     */
    [[nodiscard]] bool is_forward() const {
        return dout == nullptr;
    }

    /**
     * @brief Check if this is a backward pass context
     */
    [[nodiscard]] bool is_backward() const {
        return dout != nullptr;
    }

    /**
     * @brief Check if bias is present
     */
    [[nodiscard]] bool has_bias() const {
        return bias != nullptr && bias->Data != nullptr;
    }

    /**
     * @brief Get total elements in the matmul (M * N)
     */
    [[nodiscard]] long output_elements() const {
        return static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C_out);
    }

    /**
     * @brief Get total elements in the input (M * K)
     */
    [[nodiscard]] long input_elements() const {
        return static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C_in);
    }
};

/**
 * @brief Context for SwiGLU activation
 *
 * Separate from MatmulContext since SwiGLU has different semantics
 * (element-wise operation, optional scaling).
 */
struct SwiGLUContext {
    Tensor* out = nullptr;           ///< Output tensor (B, T, D)
    Tensor* scale_out = nullptr;     ///< Per-row scale output for scaled SwiGLU (B*T,)
    const Tensor* inp = nullptr;     ///< Input tensor (B, T, 2*D) - gate || up concatenated
    float* abs_max_out = nullptr;    ///< Optional abs_max output for quantization

    // For backward
    Tensor* dinp = nullptr;          ///< Gradient w.r.t. input (B, T, 2*D)
    const Tensor* dout = nullptr;    ///< Upstream gradient (B, T, D)
    const Tensor* scale = nullptr;   ///< Per-row scale from forward (for scaled SwiGLU backward)

    int B = 0;                       ///< Batch size
    int T = 0;                       ///< Sequence length
    int D = 0;                       ///< Intermediate size (half of inp channels)

    cudaStream_t stream = nullptr;   ///< CUDA stream

    [[nodiscard]] bool is_forward() const { return dout == nullptr; }
    [[nodiscard]] bool is_backward() const { return dout != nullptr; }
    [[nodiscard]] bool has_scale() const { return scale_out != nullptr || scale != nullptr; }
};

/**
 * @brief Context for MoE grouped matmul (recipe-driven dispatch)
 *
 * Carries all information needed to execute forward and backward MoE grouped GEMM
 * through the recipe system, with full FP8 training support.
 */
struct MoeMatmulContext {
    // =========================================================================
    // Forward pass tensors
    // =========================================================================

    // BF16 tensors (always set for forward)
    nv_bfloat16* out = nullptr;          ///< Output (total_tokens, N)
    const nv_bfloat16* inp = nullptr;    ///< Input tokens (total_tokens, K), pre-permuted
    const nv_bfloat16* weights = nullptr;///< Expert weights (E, N, K) row-major, BF16
    const int* expert_offsets = nullptr; ///< Cumulative token offsets (num_experts+1) INT32

    // Optional FP8 weight-only quantization (WoQ) data.
    // When set, the recipe may use cuDNN FE block_scale_dequantize + moe_grouped_matmul
    // for fused FP8 WoQ execution. If nullptr, recipe falls back to BF16.
    const void* weights_fp8 = nullptr;   ///< FP8 E4M3 expert weights (E, N, K) row-major
    const float* fp8_block_scales = nullptr; ///< Per-block FP32 scales for FP8
    int fp8_block_size = 128;            ///< Block size for FP8 per-block quantization

    // Optional FP4 weight-only quantization (WoQ) data.
    // When set, the recipe may use cuDNN FE block_scale_dequantize + moe_grouped_matmul
    // for fused FP4 WoQ execution. Caller must pre-combine NVFP4's two-level scales
    // (FP8 block × FP32 global) into a single FP32 scale per block.
    const void* weights_fp4 = nullptr;   ///< FP4 E2M1 expert weights (E, N, K) packed
    const float* fp4_block_scales = nullptr; ///< Per-block FP32 combined scales for FP4
    int fp4_block_size = 16;             ///< Block size for FP4 per-block quantization

    // =========================================================================
    // Backward pass tensors (only set for backward_moe_matmul)
    // =========================================================================

    nv_bfloat16* dinp = nullptr;         ///< Gradient w.r.t. input (total_tokens, K)
    const nv_bfloat16* dout = nullptr;   ///< Upstream gradient (total_tokens, N)
    nv_bfloat16* dweight = nullptr;      ///< Gradient w.r.t. weights (E, N, K) - optional

    // =========================================================================
    // FP8 quantization buffers (for full FP8 training, not just WoQ)
    // =========================================================================

    Tensor* inp_quant = nullptr;         ///< E4M3 quantized input buffer (forward)
    Tensor* dout_quant = nullptr;        ///< E5M2 quantized gradient buffer (backward)
    int delayed_quantizer_idx = -1;      ///< Quantizer index for delayed scaling (-1 = JIT)

    // =========================================================================
    // Dimensions
    // =========================================================================

    int num_experts = 0;                 ///< Number of experts
    int N = 0;                           ///< Output dimension per expert
    int K = 0;                           ///< Input dimension per expert
    int total_tokens = 0;                ///< Total tokens across all experts
    int layer_idx = 0;                   ///< Current layer index (for delayed scaling)

    // =========================================================================
    // Runtime
    // =========================================================================

    IRunState* run_state = nullptr;      ///< Run state for temp buffer allocation
    cudnnHandle_t cudnn_handle = nullptr;
    void* cublas_handle = nullptr;       ///< cublasLtHandle_t for FP8×FP8 matmul
    std::byte* workspace = nullptr;      ///< cuDNN/cuBLAS workspace
    std::size_t workspace_size = 0;
    cudaStream_t stream = nullptr;

    // =========================================================================
    // Backward-specific flags
    // =========================================================================

    bool skip_weight_grad = false;       ///< Skip weight gradient computation (LoRA-only)
    bool allow_fp8 = true;               ///< Allow FP8 quantization (false for skip_quant)

    // =========================================================================
    // Host-side MoE metadata (cached for performance)
    // =========================================================================

    const int* host_offsets = nullptr;   ///< Host copy of expert_offsets (E+1)
    const int* active_experts = nullptr; ///< Active expert indices (compact mode)
    int num_active = -1;                 ///< Number of active experts (-1 = all)
    bool weight_is_compact = false;      ///< Weights use compact indexing

    // =========================================================================
    // Convenience methods
    // =========================================================================

    /// Check if this is a forward pass context
    [[nodiscard]] bool is_forward() const {
        return dout == nullptr;
    }

    /// Check if this is a backward pass context
    [[nodiscard]] bool is_backward() const {
        return dout != nullptr;
    }

    /// Check if FP8 WoQ data is available
    [[nodiscard]] bool has_fp8_weights() const {
        return weights_fp8 != nullptr && fp8_block_scales != nullptr;
    }

    /// Check if FP4 WoQ data is available
    [[nodiscard]] bool has_fp4_weights() const {
        return weights_fp4 != nullptr && fp4_block_scales != nullptr;
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MATMUL_CONTEXT_H
