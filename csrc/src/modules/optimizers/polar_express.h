// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Polar Express Sign Method for orthogonalizing gradient matrices
// Reference: https://arxiv.org/pdf/2505.16932
// Based on implementation by @byronxu99 and others from train_gpt_medium.py

#ifndef SUROGATE_SRC_MODULES_OPTIMIZERS_POLAR_EXPRESS_H
#define SUROGATE_SRC_MODULES_OPTIMIZERS_POLAR_EXPRESS_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstddef>

namespace optimizers {

// Number of Polar Express iterations (fixed from paper optimization)
constexpr int POLAR_EXPRESS_ITERATIONS = 5;

// Pre-computed coefficients for 5 iterations with num_iters=5, safety_factor=2e-2, cushion=2
// Each tuple is (a, b, c) where iteration computes:
//   A = X @ X.T
//   B = b*A + c*(A @ A)
//   X = a*X + B @ X
constexpr float POLAR_EXPRESS_COEFFS[5][3] = {
    {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
    {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
    {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
    {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
    {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f}
};

/**
 * @brief Compute symmetric matrix multiplication C = X @ X.T
 *
 * Optimized for symmetric output - only computes upper triangle and mirrors.
 * Supports batched operation for processing multiple matrices.
 *
 * @param X Input matrix of shape (batch, M, K) or (M, K)
 * @param C Output matrix of shape (batch, M, M) or (M, M)
 * @param batch Batch size (1 for non-batched)
 * @param M Number of rows in X (and rows/cols in output C)
 * @param K Number of columns in X
 * @param stream CUDA stream for async execution
 */
void XXT(
    const nv_bfloat16* X,
    nv_bfloat16* C,
    int batch,
    int M,
    int K,
    cudaStream_t stream
);

/**
 * @brief Compute fused operation C = beta*A + alpha*(A @ A.T)
 *
 * Used in Polar Express iteration where A is the result of XXT.
 * Optimized for symmetric matrices.
 *
 * @param A Input symmetric matrix of shape (batch, M, M) or (M, M)
 * @param C Output matrix of shape (batch, M, M) or (M, M)
 * @param batch Batch size
 * @param M Matrix dimension
 * @param alpha Coefficient for A @ A.T term
 * @param beta Coefficient for A term
 * @param stream CUDA stream
 */
void ba_plus_cAA(
    const nv_bfloat16* A,
    nv_bfloat16* C,
    int batch,
    int M,
    float alpha,
    float beta,
    cudaStream_t stream
);

/**
 * @brief Compute Frobenius norm of matrix and return scaling factor
 *
 * Computes norm for spectral normalization: scale = 1 / (norm * 1.02 + 1e-6)
 *
 * @param X Input matrix of shape (batch, M, K) or (M, K)
 * @param scale Output scaling factor per batch element
 * @param batch Batch size
 * @param M Rows
 * @param K Columns
 * @param stream CUDA stream
 */
void compute_spectral_scale(
    const nv_bfloat16* X,
    float* scale,
    int batch,
    int M,
    int K,
    cudaStream_t stream
);

/**
 * @brief Apply scaling factor to matrix: X = X * scale
 *
 * @param X Matrix to scale (in-place)
 * @param scale Per-batch scaling factors
 * @param batch Batch size
 * @param M Rows
 * @param K Columns
 * @param stream CUDA stream
 */
void apply_scale(
    nv_bfloat16* X,
    const float* scale,
    int batch,
    int M,
    int K,
    cudaStream_t stream
);

/**
 * @brief Compute matrix product C = A @ B
 *
 * General batched matrix multiplication for Polar Express.
 * Supports non-square matrices.
 *
 * @param A Input matrix (batch, M, K)
 * @param B Input matrix (batch, K, N)
 * @param C Output matrix (batch, M, N)
 * @param batch Batch size
 * @param M Rows of A and C
 * @param K Cols of A, rows of B
 * @param N Cols of B and C
 * @param stream CUDA stream
 */
void batched_matmul(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int batch,
    int M,
    int K,
    int N,
    cudaStream_t stream
);

/**
 * @brief Add scaled matrix: C = alpha * A + C
 *
 * @param A Input matrix
 * @param C Output matrix (accumulated in-place)
 * @param alpha Scaling coefficient
 * @param batch Batch size
 * @param M Rows
 * @param N Cols
 * @param stream CUDA stream
 */
void axpy_matrix(
    const nv_bfloat16* A,
    nv_bfloat16* C,
    float alpha,
    int batch,
    int M,
    int N,
    cudaStream_t stream
);

/**
 * @brief Full Polar Express orthogonalization
 *
 * Finds the nearest orthogonal matrix to the input gradient matrix using
 * 5 fixed-point iterations. This is used in NorMuon to orthogonalize
 * the momentum-updated gradients.
 *
 * Algorithm:
 * 1. If rows > cols, transpose
 * 2. Spectral normalize: X = X / (||X|| * 1.02 + 1e-6)
 * 3. For each of 5 iterations:
 *    A = X @ X.T
 *    B = b*A + c*(A @ A)
 *    X = a*X + B @ X
 * 4. If transposed, transpose back
 *
 * @param handle cuBLAS handle for batched GEMM operations
 * @param G Input gradient matrix of shape (batch, M, N), will be overwritten
 * @param workspace Workspace buffer, must be at least 4 * batch * max(M,N) * max(M,N) elements
 * @param batch Batch size
 * @param M Rows of input
 * @param N Cols of input
 * @param stream CUDA stream
 */
void polar_express(
    cublasHandle_t handle,
    nv_bfloat16* G,
    nv_bfloat16* workspace,
    int batch,
    int M,
    int N,
    cudaStream_t stream
);

/**
 * @brief Calculate workspace size needed for polar_express
 *
 * @param batch Batch size
 * @param M Max rows
 * @param N Max cols
 * @return Required workspace size in bytes
 */
size_t polar_express_workspace_size(int batch, int M, int N);

} // namespace optimizers

#endif // SUROGATE_SRC_MODULES_OPTIMIZERS_POLAR_EXPRESS_H
