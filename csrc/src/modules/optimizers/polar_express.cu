// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Polar Express Sign Method CUDA kernels
// Reference: https://arxiv.org/pdf/2505.16932
//
// Performance-optimized version using cuBLAS for matrix multiplications

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cmath>
#include <stdexcept>

#include "polar_express.h"
#include "kernels/kernel_utils.cuh"

namespace optimizers {

// ----------------------------------------------------------------------------
// Constants

constexpr int THREADS_PER_BLOCK = 256;

// cuBLAS error checking
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
        }                                                                      \
    } while (0)

// ----------------------------------------------------------------------------
// Frobenius Norm Kernel

/**
 * @brief Compute squared Frobenius norm per batch element
 *
 * Each thread block processes one batch element, reducing across all elements.
 */
template <int BLOCK_THREADS>
__global__ void kFrobeniusNormSquared(
    const nv_bfloat16* __restrict__ X,
    float* __restrict__ norm_sq,
    int M,
    int K
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_elems = M * K;

    const nv_bfloat16* X_batch = X + batch_idx * num_elems;

    // Each thread accumulates its portion
    float local_sum = 0.0f;
    for (int i = tid; i < num_elems; i += BLOCK_THREADS) {
        float val = __bfloat162float(X_batch[i]);
        local_sum += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Block-level reduction via shared memory
    __shared__ float smem[32];  // One per warp
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        local_sum = lane_id < (BLOCK_THREADS / 32) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane_id == 0) {
            norm_sq[batch_idx] = local_sum;
        }
    }
}

/**
 * @brief Convert norm squared to spectral scale: 1 / (sqrt(norm_sq) * 1.02 + 1e-6)
 */
__global__ void kNormToScale(
    const float* __restrict__ norm_sq,
    float* __restrict__ scale,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        float norm = sqrtf(norm_sq[idx]);
        scale[idx] = 1.0f / (norm * 1.02f + 1e-6f);
    }
}

void compute_spectral_scale(
    const nv_bfloat16* X,
    float* scale,
    int batch,
    int M,
    int K,
    cudaStream_t stream
) {
    // Launch one block per batch element
    kFrobeniusNormSquared<THREADS_PER_BLOCK><<<batch, THREADS_PER_BLOCK, 0, stream>>>(
        X, scale, M, K
    );

    // Convert norm to scale
    int blocks = (batch + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kNormToScale<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(scale, scale, batch);
}

// ----------------------------------------------------------------------------
// Scale Application Kernel

__global__ void kApplyScale(
    nv_bfloat16* __restrict__ X,
    const float* __restrict__ scale,
    int M,
    int K
) {
    const int batch_idx = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_elems = M * K;

    if (idx < num_elems) {
        float s = scale[batch_idx];
        int offset = batch_idx * num_elems + idx;
        float val = __bfloat162float(X[offset]);
        X[offset] = __float2bfloat16(val * s);
    }
}

void apply_scale(
    nv_bfloat16* X,
    const float* scale,
    int batch,
    int M,
    int K,
    cudaStream_t stream
) {
    int num_elems = M * K;
    int blocks_x = (num_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid(blocks_x, batch);
    kApplyScale<<<grid, THREADS_PER_BLOCK, 0, stream>>>(X, scale, M, K);
}

// ----------------------------------------------------------------------------
// cuBLAS-based XXT: Compute C = X @ X.T
//
// Row-major X[M,K] @ X[M,K]^T = C[M,M]
// Using cuBLAS with row-major data:
//   Call: CUBLAS_OP_T, CUBLAS_OP_N, m=M, n=M, k=K
//   lda=K (leading dim of X row-major)
//   ldb=K (leading dim of X row-major)
//   ldc=M (leading dim of C row-major)

void XXT(
    const nv_bfloat16* X,
    nv_bfloat16* C,
    int batch,
    int M,
    int K,
    cudaStream_t stream
) {
    // Create temporary cuBLAS handle for this call
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float one = 1.0f;
    const float zero = 0.0f;

    long long stride_X = M * K;
    long long stride_C = M * M;

    // For row-major X @ X^T:
    // cuBLAS: CUBLAS_OP_T, CUBLAS_OP_N with lda=ldb=K, ldc=M
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,     // transA
        CUBLAS_OP_N,     // transB
        M,               // m: rows of output C
        M,               // n: cols of output C
        K,               // k: inner dimension
        &one,
        X, CUDA_R_16BF, K, stride_X,   // lda=K
        X, CUDA_R_16BF, K, stride_X,   // ldb=K
        &zero,
        C, CUDA_R_16BF, M, stride_C,   // ldc=M
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));

    // Synchronize before destroying handle
    cudaStreamSynchronize(stream);
    CUBLAS_CHECK(cublasDestroy(handle));
}

// ----------------------------------------------------------------------------
// ba_plus_cAA: Compute C = beta*A + alpha*(A @ A)

void ba_plus_cAA(
    const nv_bfloat16* A,
    nv_bfloat16* C,
    int batch,
    int M,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // Legacy function kept for API compatibility
    // The actual implementation is done inline in polar_express with cuBLAS
}

// ----------------------------------------------------------------------------
// General Batched MatMul: C = A @ B (legacy)

void batched_matmul(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int batch,
    int M,
    int K,
    int N,
    cudaStream_t stream
) {
    // Legacy function kept for API compatibility
    // The actual implementation is done inline in polar_express with cuBLAS
}

// ----------------------------------------------------------------------------
// AXPY: C = alpha * A + C

__global__ void kAxpyMatrix(
    const nv_bfloat16* __restrict__ A,
    nv_bfloat16* __restrict__ C,
    float alpha,
    int num_elems
) {
    const int batch_idx = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elems) {
        int offset = batch_idx * num_elems + idx;
        float a_val = __bfloat162float(A[offset]);
        float c_val = __bfloat162float(C[offset]);
        C[offset] = __float2bfloat16(alpha * a_val + c_val);
    }
}

void axpy_matrix(
    const nv_bfloat16* A,
    nv_bfloat16* C,
    float alpha,
    int batch,
    int M,
    int N,
    cudaStream_t stream
) {
    int num_elems = M * N;
    int blocks_x = (num_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid(blocks_x, batch);
    kAxpyMatrix<<<grid, THREADS_PER_BLOCK, 0, stream>>>(A, C, alpha, num_elems);
}

// ----------------------------------------------------------------------------
// Matrix Transpose

__global__ void kTranspose(
    const nv_bfloat16* __restrict__ A,
    nv_bfloat16* __restrict__ B,
    int M,
    int N
) {
    const int batch_idx = blockIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int a_offset = batch_idx * M * N + row * N + col;
        int b_offset = batch_idx * N * M + col * M + row;
        B[b_offset] = A[a_offset];
    }
}

void transpose(
    const nv_bfloat16* A,
    nv_bfloat16* B,
    int batch,
    int M,
    int N,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16, batch);
    kTranspose<<<grid, block, 0, stream>>>(A, B, M, N);
}

// ----------------------------------------------------------------------------
// Fused kernel: C = beta*A + alpha*(matmul result already in C)

__global__ void kFusedBetaAPlusAlphaC(
    const nv_bfloat16* __restrict__ A,
    nv_bfloat16* __restrict__ C,
    float alpha,
    float beta,
    int num_elems
) {
    const int batch_idx = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elems) {
        int offset = batch_idx * num_elems + idx;
        float a_val = __bfloat162float(A[offset]);
        float c_val = __bfloat162float(C[offset]);
        // C = beta*A + alpha*C (where C contains matmul result)
        C[offset] = __float2bfloat16(beta * a_val + alpha * c_val);
    }
}

// ----------------------------------------------------------------------------
// Full Polar Express Algorithm with cuBLAS

size_t polar_express_workspace_size(int batch, int M, int N) {
    int max_dim = max(M, N);
    // Need: A (M x M), B (M x M), C (M x N), X_tmp (M x N), scale (batch floats)
    size_t matrix_size = batch * max_dim * max_dim * sizeof(nv_bfloat16);
    return 4 * matrix_size + batch * sizeof(float);
}

void polar_express(
    cublasHandle_t handle,
    nv_bfloat16* G,
    nv_bfloat16* workspace,
    int batch,
    int M,
    int N,
    cudaStream_t stream
) {
    // Set cuBLAS stream
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Determine if we need to transpose (algorithm works better with cols >= rows)
    bool need_transpose = M > N;
    int work_M = need_transpose ? N : M;
    int work_N = need_transpose ? M : N;

    // Workspace layout:
    // [A: work_M x work_M] [B: work_M x work_M] [C: work_M x work_N] [X_tmp: work_M x work_N] [scale: batch floats]
    size_t a_size = batch * work_M * work_M;
    size_t c_size = batch * work_M * work_N;

    nv_bfloat16* A_buf = workspace;
    nv_bfloat16* B_buf = A_buf + a_size;
    nv_bfloat16* C_buf = B_buf + a_size;
    nv_bfloat16* X_tmp = C_buf + c_size;
    float* scale_buf = reinterpret_cast<float*>(X_tmp + c_size);

    // X points to the current working matrix (starts as G, then alternates with C_buf)
    nv_bfloat16* X = G;

    // Step 1: Transpose if needed
    if (need_transpose) {
        transpose(G, X_tmp, batch, M, N, stream);
        X = X_tmp;
    }

    // Step 2: Spectral normalization
    compute_spectral_scale(X, scale_buf, batch, work_M, work_N, stream);
    apply_scale(X, scale_buf, batch, work_M, work_N, stream);

    // cuBLAS parameters for BF16 GEMM
    // cuBLAS uses column-major format. For row-major data:
    //   Row-major matrix A[M,K] is same as column-major A^T[K,M]
    //
    // For row-major C = A @ B where A[M,K], B[K,N], C[M,N]:
    //   (A @ B)^T = B^T @ A^T where A^T and B^T are the column-major views
    //   Call: CUBLAS_OP_N, CUBLAS_OP_N, m=N, n=M, k=K
    //   A_cublas=B with lda=N, B_cublas=A with ldb=K, C=output with ldc=N
    //
    // For row-major C = A @ B^T where A[M,K], B[N,K], C[M,N]:
    //   Use CUBLAS_OP_T, CUBLAS_OP_N, m=M, n=M, k=K
    //   (same as XXT function which works)

    const float one = 1.0f;
    const float zero = 0.0f;

    // Strides for batched operations
    long long stride_X = work_M * work_N;
    long long stride_A = work_M * work_M;

    // Step 3: 5 Polar Express iterations
    for (int iter = 0; iter < POLAR_EXPRESS_ITERATIONS; ++iter) {
        float a = POLAR_EXPRESS_COEFFS[iter][0];
        float b = POLAR_EXPRESS_COEFFS[iter][1];
        float c = POLAR_EXPRESS_COEFFS[iter][2];

        // A_out = X @ X^T (row-major: A[M,M] = X[M,N] @ X^T[N,M])
        // This is the same as the working XXT function:
        //   CUBLAS_OP_T, CUBLAS_OP_N, m=M, n=M, k=N, lda=N, ldb=N, ldc=M
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,     // transA
            CUBLAS_OP_N,     // transB
            work_M,          // m: rows of output A
            work_M,          // n: cols of output A
            work_N,          // k: inner dimension (cols of X)
            &one,
            X, CUDA_R_16BF, work_N, stride_X,     // lda=N (X row-major M x N)
            X, CUDA_R_16BF, work_N, stride_X,     // ldb=N (X row-major M x N)
            &zero,
            A_buf, CUDA_R_16BF, work_M, stride_A, // ldc=M (output row-major M x M)
            batch,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));

        // B = A @ A (row-major: B[M,M] = A[M,M] @ A[M,M])
        // For square matrix multiplication A @ A:
        // Row-major A @ A: call CUBLAS_OP_N, CUBLAS_OP_N, m=M, n=M, k=M
        // A_cublas=A with lda=M, B_cublas=A with ldb=M, C=B with ldc=M
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,     // transA
            CUBLAS_OP_N,     // transB
            work_M,          // m
            work_M,          // n
            work_M,          // k
            &one,
            A_buf, CUDA_R_16BF, work_M, stride_A,  // lda=M
            A_buf, CUDA_R_16BF, work_M, stride_A,  // ldb=M
            &zero,
            B_buf, CUDA_R_16BF, work_M, stride_A,  // ldc=M
            batch,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));

        // B = b*A + c*B (where B now contains A @ A)
        {
            int num_elems = work_M * work_M;
            int blocks_x = (num_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            dim3 grid(blocks_x, batch);
            kFusedBetaAPlusAlphaC<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
                A_buf, B_buf, c, b, num_elems
            );
        }

        // C = B @ X (row-major: C[M,N] = B[M,M] @ X[M,N])
        // For row-major C = A @ B where A[M,K], B[K,N], C[M,N]:
        //   CUBLAS_OP_N, CUBLAS_OP_N, m=N, n=M, k=K
        //   A_cublas=B with lda=N, B_cublas=A with ldb=K
        // Here: B[M,M] @ X[M,N] = C[M,N]
        //   m=N, n=M, k=M
        //   A_cublas=X with lda=N, B_cublas=B with ldb=M
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,     // transA
            CUBLAS_OP_N,     // transB
            work_N,          // m: cols of output (N)
            work_M,          // n: rows of output (M)
            work_M,          // k: inner dimension (M)
            &one,
            X, CUDA_R_16BF, work_N, stride_X,      // A=X, lda=N (X row-major M x N)
            B_buf, CUDA_R_16BF, work_M, stride_A,  // B=B, ldb=M (B row-major M x M)
            &zero,
            C_buf, CUDA_R_16BF, work_N, stride_X,  // C, ldc=N (C row-major M x N)
            batch,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));

        // C = a*X + C
        axpy_matrix(X, C_buf, a, batch, work_M, work_N, stream);

        // Swap X and C for next iteration
        nv_bfloat16* tmp = X;
        X = C_buf;
        C_buf = tmp;
    }

    // Step 4: Transpose back if needed, and copy result to G
    if (need_transpose) {
        transpose(X, G, batch, work_M, work_N, stream);
    } else if (X != G) {
        // Copy result back to G if it ended up in workspace
        cudaMemcpyAsync(G, X, batch * M * N * sizeof(nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
    }
}

} // namespace optimizers
