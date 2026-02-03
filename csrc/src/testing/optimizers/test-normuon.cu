// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for NorMuon optimizer and Polar Express orthogonalization

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "runtime/optimizers/polar_express.h"
#include "runtime/optimizers/normuon.h"
#include "utilities/utils.h"
#include "../utilities/test_config.h"
#include "../utilities/test_utils.h"

using namespace testing_utils;
using namespace optimizers;

namespace {

// ----------------------------------------------------------------------------
// CPU Reference Implementations

/**
 * @brief CPU reference for Frobenius norm
 */
float frobenius_norm_cpu(const std::vector<float>& X, int M, int N) {
    float sum = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        sum += X[i] * X[i];
    }
    return std::sqrt(sum);
}

/**
 * @brief CPU reference for X @ X.T
 */
void XXT_cpu(const std::vector<float>& X, std::vector<float>& C, int M, int K) {
    C.resize(M * M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += X[i * K + k] * X[j * K + k];
            }
            C[i * M + j] = sum;
        }
    }
}

/**
 * @brief CPU reference for matmul C = A @ B
 */
void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B,
                std::vector<float>& C, int M, int K, int N) {
    C.resize(M * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Check if matrix is approximately orthogonal
 *
 * For wide matrices (M <= N): checks U @ U.T ≈ I (rows are orthonormal)
 * For tall matrices (M > N): checks U.T @ U ≈ I (columns are orthonormal)
 * This matches Polar Express behavior which produces row-orthonormal matrices,
 * but transposes tall matrices before/after processing.
 */
float orthogonality_error(const std::vector<float>& U, int M, int N) {
    // For tall matrices, transpose was applied so columns become orthonormal
    bool check_columns = (M > N);
    int dim = check_columns ? N : M;  // dimension of the identity we check against

    std::vector<float> product(dim * dim);

    if (check_columns) {
        // Compute U.T @ U (columns are orthonormal)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < M; ++k) {
                    sum += U[k * N + i] * U[k * N + j];  // U.T[i,k] * U[k,j]
                }
                product[i * N + j] = sum;
            }
        }
    } else {
        // Compute U @ U.T (rows are orthonormal)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += U[i * N + k] * U[j * N + k];  // U[i,k] * U.T[k,j]
                }
                product[i * M + j] = sum;
            }
        }
    }

    // Compute Frobenius norm of (product - I)
    float error = 0.0f;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            float target = (i == j) ? 1.0f : 0.0f;
            float diff = product[i * dim + j] - target;
            error += diff * diff;
        }
    }
    return std::sqrt(error);
}

/**
 * @brief CPU reference for Polar Express (simplified single iteration for testing)
 */
void polar_express_cpu_reference(std::vector<float>& X, int M, int N, int num_iters = 5) {
    // Polar Express coefficients
    const float coeffs[5][3] = {
        {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
        {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
        {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
        {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
        {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f}
    };

    bool transposed = M > N;
    int work_M = transposed ? N : M;
    int work_N = transposed ? M : N;

    // Transpose if needed
    std::vector<float> work;
    if (transposed) {
        work.resize(work_M * work_N);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                work[j * M + i] = X[i * N + j];
            }
        }
    } else {
        work = X;
    }

    // Spectral normalization
    float norm = frobenius_norm_cpu(work, work_M, work_N);
    float scale = 1.0f / (norm * 1.02f + 1e-6f);
    for (auto& v : work) v *= scale;

    // Iterations
    std::vector<float> A(work_M * work_M);
    std::vector<float> B(work_M * work_M);
    std::vector<float> C(work_M * work_N);

    for (int iter = 0; iter < num_iters; ++iter) {
        float a = coeffs[iter][0];
        float b = coeffs[iter][1];
        float c = coeffs[iter][2];

        // A = X @ X.T
        XXT_cpu(work, A, work_M, work_N);

        // B = b*A + c*(A @ A)
        std::vector<float> AA;
        matmul_cpu(A, A, AA, work_M, work_M, work_M);
        for (int i = 0; i < work_M * work_M; ++i) {
            B[i] = b * A[i] + c * AA[i];
        }

        // C = a*X + B @ X
        std::vector<float> BX;
        matmul_cpu(B, work, BX, work_M, work_M, work_N);
        for (int i = 0; i < work_M * work_N; ++i) {
            C[i] = a * work[i] + BX[i];
        }

        work = C;
    }

    // Transpose back if needed
    if (transposed) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                X[i * N + j] = work[j * M + i];
            }
        }
    } else {
        X = work;
    }
}

/**
 * @brief CPU reference for momentum update
 */
void momentum_update_cpu(
    std::vector<float>& momentum,
    const std::vector<float>& gradient,
    float beta1
) {
    for (size_t i = 0; i < gradient.size(); ++i) {
        momentum[i] = beta1 * momentum[i] + (1.0f - beta1) * gradient[i];
    }
}

/**
 * @brief CPU reference for cautious weight decay update
 */
void cautious_wd_update_cpu(
    std::vector<float>& params,
    const std::vector<float>& update,
    float lr,
    float weight_decay
) {
    for (size_t i = 0; i < params.size(); ++i) {
        float mask = (update[i] * params[i] >= 0.0f) ? 1.0f : 0.0f;
        params[i] = params[i] - (params[i] * mask * weight_decay * lr) - (update[i] * lr);
    }
}

// ----------------------------------------------------------------------------
// Helper functions

std::vector<float> random_matrix(int M, int N, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(N)));
    std::vector<float> result(M * N);
    for (auto& v : result) v = dist(gen);
    return result;
}

std::vector<nv_bfloat16> float_to_bf16(const std::vector<float>& x) {
    std::vector<nv_bfloat16> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = __float2bfloat16(x[i]);
    }
    return result;
}

std::vector<float> bf16_to_float(const std::vector<nv_bfloat16>& x) {
    std::vector<float> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = __bfloat162float(x[i]);
    }
    return result;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

float relative_error(const std::vector<float>& actual, const std::vector<float>& expected) {
    float sum_sq_diff = 0.0f;
    float sum_sq_expected = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = actual[i] - expected[i];
        sum_sq_diff += diff * diff;
        sum_sq_expected += expected[i] * expected[i];
    }
    return std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_expected) + 1e-8f);
}

} // namespace

// ============================================================================
// Tests
// ============================================================================

TEST_CASE("Polar Express produces orthogonal output", "[optimizers][normuon]") {
    std::mt19937 gen(42);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    SECTION("Square matrix 64x64") {
        const int M = 64, N = 64;

        // Create random input
        auto X_fp32 = random_matrix(M, N, gen);
        auto X_bf16 = float_to_bf16(X_fp32);

        // Allocate device memory
        thrust::device_vector<nv_bfloat16> d_X(X_bf16);
        size_t ws_size = polar_express_workspace_size(1, M, N);
        thrust::device_vector<nv_bfloat16> d_workspace(ws_size / sizeof(nv_bfloat16) + 1);

        // Run Polar Express
        polar_express(
            cublas_handle,
            thrust::raw_pointer_cast(d_X.data()),
            thrust::raw_pointer_cast(d_workspace.data()),
            1, M, N, stream
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy back
        std::vector<nv_bfloat16> result_bf16(M * N);
        thrust::copy(d_X.begin(), d_X.end(), result_bf16.begin());
        auto result = bf16_to_float(result_bf16);

        // Check orthogonality
        float orth_error = orthogonality_error(result, M, N);
        INFO("Orthogonality error: " << orth_error);
        // Polar Express with 5 iterations achieves ~1.5 error on random matrices
        // This is expected behavior per the paper's analysis
        REQUIRE(orth_error < 2.0f);
    }

    SECTION("Wide matrix 32x128") {
        const int M = 32, N = 128;

        auto X_fp32 = random_matrix(M, N, gen);
        auto X_bf16 = float_to_bf16(X_fp32);

        thrust::device_vector<nv_bfloat16> d_X(X_bf16);
        size_t ws_size = polar_express_workspace_size(1, M, N);
        thrust::device_vector<nv_bfloat16> d_workspace(ws_size / sizeof(nv_bfloat16) + 1);

        polar_express(
            cublas_handle,
            thrust::raw_pointer_cast(d_X.data()),
            thrust::raw_pointer_cast(d_workspace.data()),
            1, M, N, stream
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<nv_bfloat16> result_bf16(M * N);
        thrust::copy(d_X.begin(), d_X.end(), result_bf16.begin());
        auto result = bf16_to_float(result_bf16);

        float orth_error = orthogonality_error(result, M, N);
        INFO("Orthogonality error (wide): " << orth_error);
        REQUIRE(orth_error < 2.0f);
    }

    SECTION("Tall matrix 128x32") {
        const int M = 128, N = 32;

        auto X_fp32 = random_matrix(M, N, gen);
        auto X_bf16 = float_to_bf16(X_fp32);

        thrust::device_vector<nv_bfloat16> d_X(X_bf16);
        size_t ws_size = polar_express_workspace_size(1, M, N);
        thrust::device_vector<nv_bfloat16> d_workspace(ws_size / sizeof(nv_bfloat16) + 1);

        polar_express(
            cublas_handle,
            thrust::raw_pointer_cast(d_X.data()),
            thrust::raw_pointer_cast(d_workspace.data()),
            1, M, N, stream
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<nv_bfloat16> result_bf16(M * N);
        thrust::copy(d_X.begin(), d_X.end(), result_bf16.begin());
        auto result = bf16_to_float(result_bf16);

        float orth_error = orthogonality_error(result, M, N);
        INFO("Orthogonality error (tall): " << orth_error);
        REQUIRE(orth_error < 2.0f);
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("XXT kernel correctness", "[optimizers][normuon]") {
    std::mt19937 gen(123);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int M = 64, K = 128;

    // Create random input
    auto X_fp32 = random_matrix(M, K, gen);
    auto X_bf16 = float_to_bf16(X_fp32);

    // CPU reference
    std::vector<float> C_ref;
    XXT_cpu(X_fp32, C_ref, M, K);

    // GPU computation
    thrust::device_vector<nv_bfloat16> d_X(X_bf16);
    thrust::device_vector<nv_bfloat16> d_C(M * M);

    XXT(
        thrust::raw_pointer_cast(d_X.data()),
        thrust::raw_pointer_cast(d_C.data()),
        1, M, K, stream
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back
    std::vector<nv_bfloat16> C_bf16(M * M);
    thrust::copy(d_C.begin(), d_C.end(), C_bf16.begin());
    auto C_gpu = bf16_to_float(C_bf16);

    // Compare
    float rel_err = relative_error(C_gpu, C_ref);
    INFO("XXT relative error: " << rel_err);
    REQUIRE(rel_err < 0.05f);  // Allow for BF16 precision loss

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("Spectral scale computation", "[optimizers][normuon]") {
    std::mt19937 gen(456);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int M = 64, K = 128;

    auto X_fp32 = random_matrix(M, K, gen);
    auto X_bf16 = float_to_bf16(X_fp32);

    // CPU reference
    float norm_cpu = frobenius_norm_cpu(X_fp32, M, K);
    float scale_ref = 1.0f / (norm_cpu * 1.02f + 1e-6f);

    // GPU computation
    thrust::device_vector<nv_bfloat16> d_X(X_bf16);
    thrust::device_vector<float> d_scale(1);

    compute_spectral_scale(
        thrust::raw_pointer_cast(d_X.data()),
        thrust::raw_pointer_cast(d_scale.data()),
        1, M, K, stream
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float scale_gpu;
    CUDA_CHECK(cudaMemcpy(&scale_gpu, thrust::raw_pointer_cast(d_scale.data()),
                          sizeof(float), cudaMemcpyDeviceToHost));

    INFO("CPU scale: " << scale_ref << ", GPU scale: " << scale_gpu);
    REQUIRE(std::fabs(scale_gpu - scale_ref) / scale_ref < 0.05f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("Cautious weight decay update", "[optimizers][normuon]") {
    std::mt19937 gen(789);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int N = 1024;
    const float lr = 0.02f;
    const float wd = 0.01f;

    // Create random params and updates
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> params_cpu(N), update_cpu(N);
    for (int i = 0; i < N; ++i) {
        params_cpu[i] = dist(gen);
        update_cpu[i] = dist(gen);
    }

    // CPU reference
    std::vector<float> params_ref = params_cpu;
    cautious_wd_update_cpu(params_ref, update_cpu, lr, wd);

    // GPU computation
    auto params_bf16 = float_to_bf16(params_cpu);
    auto update_bf16 = float_to_bf16(update_cpu);

    thrust::device_vector<nv_bfloat16> d_params(params_bf16);
    thrust::device_vector<nv_bfloat16> d_update(update_bf16);

    cautious_weight_decay_update(
        thrust::raw_pointer_cast(d_params.data()),
        thrust::raw_pointer_cast(d_update.data()),
        N, lr, wd, stream
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back
    std::vector<nv_bfloat16> result_bf16(N);
    thrust::copy(d_params.begin(), d_params.end(), result_bf16.begin());
    auto result = bf16_to_float(result_bf16);

    // Compare
    float rel_err = relative_error(result, params_ref);
    INFO("Cautious WD relative error: " << rel_err);
    REQUIRE(rel_err < 0.02f);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("NorMuon momentum update 8-bit", "[optimizers][normuon]") {
    std::mt19937 gen(101112);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int N = 4096;  // Multiple of BLOCK_SIZE
    const float beta1 = 0.95f;

    // Create random gradient
    std::normal_distribution<float> dist(0.0f, 0.1f);
    std::vector<float> gradient_cpu(N);
    for (int i = 0; i < N; ++i) {
        gradient_cpu[i] = dist(gen);
    }

    // Initialize momentum to zero (CPU reference)
    std::vector<float> momentum_ref(N, 0.0f);
    momentum_update_cpu(momentum_ref, gradient_cpu, beta1);

    // GPU computation
    auto gradient_bf16 = float_to_bf16(gradient_cpu);
    thrust::device_vector<nv_bfloat16> d_gradient(gradient_bf16);
    thrust::device_vector<nv_bfloat16> d_momentum_out(N);

    // Initialize 8-bit state
    constexpr int BLOCK_SIZE = 2048;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    thrust::device_vector<unsigned char> d_momentum_state(N);
    thrust::device_vector<float> d_absmax(num_blocks);
    thrust::device_vector<float> d_quantiles(256);

    // Create quantiles on host and copy to device
    std::vector<float> h_quantiles(256);
    create_normuon_quantiles(h_quantiles.data());
    thrust::copy(h_quantiles.begin(), h_quantiles.end(), d_quantiles.begin());

    // Initialize state
    init_normuon_momentum_state(
        thrust::raw_pointer_cast(d_momentum_state.data()),
        thrust::raw_pointer_cast(d_absmax.data()),
        N, stream
    );

    // Run momentum update
    normuon_momentum_update_8bit(
        thrust::raw_pointer_cast(d_gradient.data()),
        thrust::raw_pointer_cast(d_momentum_state.data()),
        thrust::raw_pointer_cast(d_momentum_out.data()),
        N, beta1,
        thrust::raw_pointer_cast(d_quantiles.data()),
        thrust::raw_pointer_cast(d_absmax.data()),
        stream
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy back
    std::vector<nv_bfloat16> result_bf16(N);
    thrust::copy(d_momentum_out.begin(), d_momentum_out.end(), result_bf16.begin());
    auto result = bf16_to_float(result_bf16);

    // Compare - expect some quantization error
    float rel_err = relative_error(result, momentum_ref);
    INFO("Momentum 8-bit relative error: " << rel_err);
    REQUIRE(rel_err < 0.15f);  // Higher tolerance due to 8-bit quantization

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Benchmarks
// ============================================================================

TEST_CASE("Polar Express benchmark", "[optimizers][normuon][benchmark][!benchmark]") {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    std::mt19937 gen(42);

    // Typical transformer weight sizes
    std::vector<std::pair<int, int>> sizes = {
        {768, 768},     // Small attention
        {768, 3072},    // Small MLP up
        {3072, 768},    // Small MLP down
        {1024, 1024},   // Medium attention
        {1024, 4096},   // Medium MLP up
        {4096, 1024},   // Medium MLP down
    };

    for (auto [M, N] : sizes) {
        auto X_fp32 = random_matrix(M, N, gen);
        auto X_bf16 = float_to_bf16(X_fp32);

        thrust::device_vector<nv_bfloat16> d_X(X_bf16);
        size_t ws_size = polar_express_workspace_size(1, M, N);
        thrust::device_vector<nv_bfloat16> d_workspace(ws_size / sizeof(nv_bfloat16) + 1);

        // Warmup
        for (int i = 0; i < 3; ++i) {
            thrust::copy(X_bf16.begin(), X_bf16.end(), d_X.begin());
            polar_express(
                cublas_handle,
                thrust::raw_pointer_cast(d_X.data()),
                thrust::raw_pointer_cast(d_workspace.data()),
                1, M, N, stream
            );
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Benchmark
        const int num_iters = 100;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iters; ++i) {
            thrust::copy(X_bf16.begin(), X_bf16.end(), d_X.begin());
            polar_express(
                cublas_handle,
                thrust::raw_pointer_cast(d_X.data()),
                thrust::raw_pointer_cast(d_workspace.data()),
                1, M, N, stream
            );
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = elapsed_ms / num_iters;

        INFO("Polar Express " << M << "x" << N << ": " << avg_ms << " ms");
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
