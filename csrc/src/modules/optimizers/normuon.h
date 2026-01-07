// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// NorMuon optimizer: Momentum Orthogonalized by Newton-Schulz with variance reduction
// Based on implementation from train_gpt_medium.py
//
// This optimizer is not used for the embedding layer, the final
// fully connected layer (lm_head), or any {0,1}-D parameters; those are optimized by AdamW.

#ifndef SUROGATE_SRC_MODULES_OPTIMIZERS_NORMUON_H
#define SUROGATE_SRC_MODULES_OPTIMIZERS_NORMUON_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstddef>
#include <cmath>

namespace optimizers {

// Block size for NorMuon 8-bit optimizer (matches AdamW)
constexpr int NORMUON_BLOCK_SIZE = 2048;

// ----------------------------------------------------------------------------
// Variance Reduction
// ----------------------------------------------------------------------------

/**
 * @brief Compute row/column means of squared values for variance estimation
 *
 * For tall matrices (M >= N), reduces over columns: output shape (batch, M, 1)
 * For wide matrices (M < N), reduces over rows: output shape (batch, 1, N)
 *
 * @param v Input matrix (batch, M, N)
 * @param v_mean Output means (batch, M, 1) or (batch, 1, N)
 * @param batch Batch size
 * @param M Rows
 * @param N Cols
 * @param reduce_over_cols If true, reduce over columns (for tall matrices)
 * @param stream CUDA stream
 */
void compute_variance_mean(
    const nv_bfloat16* v,
    float* v_mean,
    int batch,
    int M,
    int N,
    bool reduce_over_cols,
    cudaStream_t stream
);

/**
 * @brief Apply NorMuon variance reduction to update tensor
 *
 * Implements Adafactor-style variance reduction:
 * 1. Compute row/column means of squared values
 * 2. Update EMA buffer: buf = beta2 * buf + (1 - beta2) * mean
 * 3. Compute adaptive scale and apply to v
 *
 * @param v Input/output update tensor (batch, M, N) - modified in place
 * @param variance_buffer Second moment EMA buffer - updated in place
 * @param batch Batch size
 * @param M Rows
 * @param N Cols
 * @param beta2 EMA decay factor (typically 0.95)
 * @param reduce_over_cols If true, reduce over columns (for tall matrices)
 * @param stream CUDA stream
 */
void apply_variance_reduction(
    nv_bfloat16* v,
    float* variance_buffer,
    int batch,
    int M,
    int N,
    float beta2,
    bool reduce_over_cols,
    cudaStream_t stream
);

// ----------------------------------------------------------------------------
// 8-bit Momentum State Management
// ----------------------------------------------------------------------------

/**
 * @brief Initialize NorMuon 8-bit momentum state tensors
 *
 * @param momentum_state 8-bit quantized momentum buffer
 * @param momentum_absmax Per-block absmax values
 * @param n Number of elements
 * @param stream CUDA stream
 */
void init_normuon_momentum_state(
    unsigned char* momentum_state,
    float* momentum_absmax,
    size_t n,
    cudaStream_t stream
);

/**
 * @brief Creates the signed quantization map for NorMuon momentum
 *
 * Same as AdamW first moment - maps values in [-1, 1]
 *
 * @param code Output array of 256 float values
 */
void create_normuon_quantiles(float* code);

// ----------------------------------------------------------------------------
// Cautious Weight Decay
// ----------------------------------------------------------------------------

/**
 * @brief Apply cautious weight decay with parameter update
 *
 * Cautious weight decay only applies decay when sign(update) == sign(param):
 *   mask = (v * p) >= 0
 *   p = p - (p * mask * wd * lr) - (v * lr)
 *
 * @param p Parameter tensor (modified in place)
 * @param v Update tensor (from orthogonalized, variance-reduced gradient)
 * @param n Number of elements
 * @param lr Learning rate
 * @param weight_decay Weight decay coefficient
 * @param stream CUDA stream
 */
void cautious_weight_decay_update(
    nv_bfloat16* p,
    const nv_bfloat16* v,
    size_t n,
    float lr,
    float weight_decay,
    cudaStream_t stream
);

void cautious_weight_decay_update(
    float* p,
    const nv_bfloat16* v,
    size_t n,
    float lr,
    float weight_decay,
    cudaStream_t stream
);

// ----------------------------------------------------------------------------
// Momentum Update with 8-bit Quantization
// ----------------------------------------------------------------------------

/**
 * @brief Update 8-bit quantized momentum buffer and output dequantized result
 *
 * Computes: momentum = beta1 * momentum + (1 - beta1) * gradient
 * Returns the momentum-smoothed gradient for further processing.
 *
 * @param gradient Input gradient tensor
 * @param momentum_state 8-bit quantized momentum state (updated in place)
 * @param momentum_out Output: dequantized updated momentum
 * @param n Number of elements
 * @param beta1 Momentum coefficient (typically 0.95)
 * @param quantiles 256-entry quantization map
 * @param absmax Per-block absmax values (updated in place)
 * @param stream CUDA stream
 */
void normuon_momentum_update_8bit(
    const nv_bfloat16* gradient,
    unsigned char* momentum_state,
    nv_bfloat16* momentum_out,
    size_t n,
    float beta1,
    const float* quantiles,
    float* absmax,
    cudaStream_t stream
);

void normuon_momentum_update_8bit(
    const float* gradient,
    unsigned char* momentum_state,
    float* momentum_out,
    size_t n,
    float beta1,
    const float* quantiles,
    float* absmax,
    cudaStream_t stream
);

// ----------------------------------------------------------------------------
// Full NorMuon Update Step
// ----------------------------------------------------------------------------

/**
 * @brief Complete NorMuon optimizer step for a single 2D weight tensor
 *
 * Performs the full NorMuon algorithm:
 * 1. Momentum update: m = beta1 * m + (1 - beta1) * g
 * 2. Polar Express orthogonalization
 * 3. Variance reduction
 * 4. Cautious weight decay + parameter update
 *
 * @param handle cuBLAS handle for Polar Express matrix operations
 * @param param Weight parameter (modified in place)
 * @param gradient Input gradient
 * @param momentum_state 8-bit quantized momentum (updated in place)
 * @param variance_buffer Variance EMA buffer (updated in place)
 * @param polar_workspace Workspace for Polar Express algorithm
 * @param M Weight rows (output features)
 * @param N Weight cols (input features)
 * @param lr Learning rate
 * @param beta1 Momentum coefficient
 * @param beta2 Variance EMA coefficient
 * @param weight_decay Weight decay coefficient
 * @param quantiles Momentum quantization map
 * @param absmax Per-block momentum absmax (updated in place)
 * @param stream CUDA stream
 */
void normuon_update_2d(
    cublasHandle_t handle,
    nv_bfloat16* param,
    const nv_bfloat16* gradient,
    unsigned char* momentum_state,
    float* variance_buffer,
    nv_bfloat16* polar_workspace,
    int M,
    int N,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    const float* quantiles,
    float* absmax,
    cudaStream_t stream
);

/**
 * @brief Calculate variance buffer size for a 2D weight tensor
 *
 * @param M Weight rows
 * @param N Weight cols
 * @return Size in floats
 */
inline size_t normuon_variance_buffer_size(int M, int N) {
    // For tall matrices (M >= N), reduce over cols: shape (M, 1)
    // For wide matrices (M < N), reduce over rows: shape (1, N)
    return M >= N ? M : N;
}

/**
 * @brief Determine optimal learning rate multiplier based on weight shape
 *
 * Follows the reference implementation: sqrt(max(1.0, M / N))
 * This boosts learning rate for parameters with many inputs relative to outputs.
 *
 * @param M Weight rows (output features)
 * @param N Weight cols (input features)
 * @return Learning rate multiplier
 */
inline float normuon_lr_multiplier(int M, int N) {
    float ratio = static_cast<float>(M) / static_cast<float>(N);
    return sqrtf(fmaxf(1.0f, ratio));
}

} // namespace optimizers

#endif // SUROGATE_SRC_MODULES_OPTIMIZERS_NORMUON_H
