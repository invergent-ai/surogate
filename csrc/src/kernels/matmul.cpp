// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cublasLt.h>
#include <cublas_v2.h>
#include <fmt/core.h>
#include <mutex>
#include <type_traits>

#include "kernels.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

namespace {
std::once_flag g_fallback_cublas_once;
cublasHandle_t g_fallback_cublas_handle = nullptr;
}  // namespace

cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;

void init_cublas_fallback_handle() {
    std::call_once(g_fallback_cublas_once, []() {
        CUBLAS_CHECK(cublasCreate(&g_fallback_cublas_handle));
        // Allow tensor cores for BF16/FP16.
        (void)cublasSetMathMode(g_fallback_cublas_handle, CUBLAS_TENSOR_OP_MATH);
    });
}

static cublasHandle_t get_fallback_cublas_handle() {
    init_cublas_fallback_handle();
    return g_fallback_cublas_handle;
}

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
inline void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(fmt::format("cuBLAS ERROR ({}) at {}:{}", (int)status, file, line));
    }
}
#ifdef CUBLAS_CHECK
#undef CUBLAS_CHECK
#endif
#define CUBLAS_CHECK(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// Setup

cublasLtHandle_t create_cublaslt_handle() {
    cublasLtHandle_t handle;
    CUBLAS_CHECK(cublasLtCreate(&handle));
    return handle;
}

void destroy_cublaslt_handle(cublasLtHandle_t handle) noexcept {
    if (!handle) {
        return;
    }
    (void)cublasLtDestroy(handle);
}

// ----------------------------------------------------------------------------
// kernel launchers

/**
 * @brief Performs matrix multiplication using cuBLASLt: D = alpha * op(A) * op(B) + beta * C + bias.
 * 
 * Wrapper around cublasLtMatmul that is meant to support everything we need in llm.c
 * https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
 * 
 * This function wraps the cuBLASLt API to perform high-performance matrix multiplication,
 * supporting various data types (including FP8 via scaling factors), transposition modes,
 * and optional bias addition. It handles the creation of descriptors, layout definitions,
 * and heuristic search for the best algorithm.
 *
 * @tparam FloatC Type of the output matrix D and input accumulator C.
 * @tparam FloatA Type of the input matrix A.
 * @tparam FloatB Type of the input matrix B.
 * @tparam FloatBias Type of the bias vector.
 *
 * @param d Pointer to the output matrix D in device memory. Acts as both source (C) and destination (D).
 * @param a Pointer to the input matrix A in device memory.
 * @param b Pointer to the input matrix B in device memory.
 * @param bias Pointer to the bias vector in device memory. Can be nullptr if no bias is required.
 * @param workspace Pointer to device memory used as workspace for the operation.
 * @param workspace_size Size of the workspace buffer in bytes.
 * @param m Number of rows in the resulting matrix.
 * @param n Number of columns in the resulting matrix.
 * @param k Inner dimension of the matrix multiplication.
 * @param stream CUDA stream to execute the operation on.
 * @param handle Valid cuBLASLt handle.
 * @param scale_a Pointer to the scaling factor for matrix A (host or device). Used primarily for FP8 inputs.
 * @param scale_b Pointer to the scaling factor for matrix B (host or device). Used primarily for FP8 inputs.
 * @param mode Transposition mode for matrices A and B (e.g., NN, NT, TN, TT).
 * @param accumulate If true, accumulates the result into D (beta = 1.0). If false, overwrites D (beta = 0.0).
 * @param ldc_override Optional override for the leading dimension of C/D. If <= 0, defaults to m.
 *
 * @throws std::runtime_error If input pointers (a, b, d, bias) are not 16-byte aligned.
 * @throws std::runtime_error If scaling pointers are provided for non-byte-sized types (i.e., types other than FP8).
 * @throws std::runtime_error If no suitable cuBLASLt algorithm heuristic is found for the given configuration.
 */
template<class FloatC, class FloatA, class FloatB, class FloatBias>
void matmul_cublaslt(FloatC* d, const FloatA* a, const FloatB* b, const FloatBias* bias,
                     std::byte* workspace, std::size_t workspace_size,
                     int m, int n, int k, cudaStream_t stream, cublasLtHandle_t handle,
                     const float* scale_a, const float* scale_b, EMMTranspose mode, bool accumulate, int ldc_override = -1)
{
    bool has_bias = (bias != nullptr);

    // check alignment (some modes work unaligned, but it is always best to be aligned for performance)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        throw std::runtime_error("All cuBLASLt pointers must be aligned!");
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    bool transA = mode == EMMTranspose::TN || mode == EMMTranspose::TT;
    bool transB = mode == EMMTranspose::NT || mode == EMMTranspose::TT;

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<FloatA>, k, m, k));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&ALayout, to_cuda_lib_type_enum<FloatA>, m, k, m));
    }
    if (transB) {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<FloatB>, n, k, n));
    } else {
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&BLayout, to_cuda_lib_type_enum<FloatB>, k, n, k));
    }
    int ldc = (ldc_override > 0) ? ldc_override : m;
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&CLayout, to_cuda_lib_type_enum<FloatC>, m, n, ldc));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&DLayout, to_cuda_lib_type_enum<FloatC>, m, n, ldc));

    // create a preference handle with specified max workspace
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_size, sizeof(workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if(has_bias){
        epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = to_cuda_lib_type_enum<FloatBias>; // force BF16 bias for FP8 mode
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    if(scale_a) {
        if(sizeof(FloatA) != 1) {
            throw std::runtime_error("Scaling A is only supported for FP8");
        }
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(&scale_a)));
    }
    if(scale_b) {
        if(sizeof(FloatB) != 1) {
            throw std::runtime_error("Scaling B is only supported for FP8");
        }
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(&scale_b)));
    }

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    // Request multiple algorithms to allow fallback if primary fails
    constexpr int kMaxAlgos = 8;
    cublasLtMatmulHeuristicResult_t heuristics[kMaxAlgos];
    cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, kMaxAlgos, heuristics, &returnedResults);
    if (returnedResults == 0) {
        throw std::runtime_error(fmt::format("No cuBLASLt algorithm: m: {}, n: {}, k: {}, bias: {}", n, m, k, has_bias));
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    float one = 1.f;
    float zero = 0.f;
    float* alpha = &one;
    float* beta = accumulate ? &one : &zero;

    // Try algorithms in order until one succeeds.
    // Some heuristics can be reported with non-success state; skip those.
    cublasStatus_t matmul_status = CUBLAS_STATUS_NOT_SUPPORTED;
    cublasStatus_t gemm_status = CUBLAS_STATUS_NOT_SUPPORTED;
    bool fallback_tried = false;
    int algos_tried = 0;
    for (int i = 0; i < returnedResults; ++i) {
        if (heuristics[i].state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        matmul_status = cublasLtMatmul(handle, operationDesc,
                                   alpha, a, ALayout, b, BLayout, beta, d, CLayout, d, DLayout,
                                   &heuristics[i].algo, workspace, workspace_size, stream);
        ++algos_tried;
        if (matmul_status == CUBLAS_STATUS_SUCCESS) {
            break;
        }
    }
    if (matmul_status != CUBLAS_STATUS_SUCCESS) {
        // Fallback to cuBLAS GEMM for non-FP8 types when cuBLASLt fails.
        // This is slower but more robust for certain BF16 shapes.
        if constexpr (sizeof(FloatA) != 1 && sizeof(FloatB) != 1) {
            const bool has_bias = (bias != nullptr);
            const int ldc_fallback = (ldc_override > 0) ? ldc_override : m;

            // Strided output with bias isn't supported by the simple bias kernel.
            if (!has_bias || ldc_fallback == m) {
                fallback_tried = true;
                cublasHandle_t fallback_handle = get_fallback_cublas_handle();
                CUBLAS_CHECK(cublasSetStream(fallback_handle, stream));

                const cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
                const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
                const int lda = transA ? k : m;
                const int ldb = transB ? n : k;

                float one = 1.0f;
                float zero = 0.0f;
                float* alpha = &one;
                float* beta = accumulate ? &one : &zero;

                gemm_status = cublasGemmEx(
                    fallback_handle,
                    opA, opB,
                    m, n, k,
                    alpha,
                    a, to_cuda_lib_type_enum<FloatA>, lda,
                    b, to_cuda_lib_type_enum<FloatB>, ldb,
                    beta,
                    d, to_cuda_lib_type_enum<FloatC>, ldc_fallback,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                    // Retry without tensor ops for maximum compatibility.
                    gemm_status = cublasGemmEx(
                        fallback_handle,
                        opA, opB,
                        m, n, k,
                        alpha,
                        a, to_cuda_lib_type_enum<FloatA>, lda,
                        b, to_cuda_lib_type_enum<FloatB>, ldb,
                        beta,
                        d, to_cuda_lib_type_enum<FloatC>, ldc_fallback,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                }

                if (gemm_status == CUBLAS_STATUS_SUCCESS) {
                    if (has_bias) {
                        if constexpr (std::is_same_v<FloatC, float> && std::is_same_v<FloatBias, float>) {
                            // Treat output as row-major (n x m) for bias add; only product matters.
                            add_bias(d, bias, /*B=*/1, /*T=*/n, /*OC=*/m, stream);
                        } else if constexpr (std::is_same_v<FloatC, nv_bfloat16> &&
                                             std::is_same_v<FloatBias, nv_bfloat16>) {
                            add_bias(d, bias, /*B=*/1, /*T=*/n, /*OC=*/m, stream);
                        } else {
                            throw std::runtime_error("cuBLASLt failed and fallback bias add has incompatible dtype");
                        }
                    }
                    // Success via fallback.
                    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
                    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
                    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(ALayout));
                    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(BLayout));
                    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(CLayout));
                    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(DLayout));
                    CUDA_CHECK(cudaGetLastError());
                    return;
                }
            }
        }
        int device_id = -1;
        cudaGetDevice(&device_id);
        if (fallback_tried) {
            throw std::runtime_error(fmt::format(
                "cuBLAS ERROR ({}) at {}:{} - device: {}, m: {}, n: {}, k: {}, ws_size: {}, A_type: {}, B_type: {}, algos_tried: {}/{}, gemm_status: {}",
                (int)matmul_status, __FILE__, __LINE__, device_id, m, n, k, workspace_size,
                (int)to_cuda_lib_type_enum<FloatA>, (int)to_cuda_lib_type_enum<FloatB>, algos_tried, returnedResults,
                (int)gemm_status));
        }
        throw std::runtime_error(fmt::format(
            "cuBLAS ERROR ({}) at {}:{} - device: {}, m: {}, n: {}, k: {}, ws_size: {}, A_type: {}, B_type: {}, algos_tried: {}/{}",
            (int)matmul_status, __FILE__, __LINE__, device_id, m, n, k, workspace_size,
            (int)to_cuda_lib_type_enum<FloatA>, (int)to_cuda_lib_type_enum<FloatB>, algos_tried, returnedResults));
    }
    CUDA_CHECK(cudaGetLastError());

    // cleanups
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(ALayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(BLayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(CLayout));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(DLayout));
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Performs matrix multiplication using cuBLASLt.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute a matrix multiplication
 * operation of the form C = alpha * (A x B) + beta * C + bias, potentially with scaling factors.
 *
 * @param c Pointer to the output matrix C (device memory).
 * @param a Pointer to the input matrix A (device memory).
 * @param b Pointer to the input matrix B (device memory).
 * @param bias Pointer to the bias vector (device memory). Can be nullptr if no bias is applied.
 * @param scale_a Pointer to the scaling factor for matrix A (device memory). Used for quantized operations.
 * @param scale_b Pointer to the scaling factor for matrix B (device memory). Used for quantized operations.
 * @param handle The cuBLASLt handle used to manage the library context.
 * @param workspace Pointer to the workspace memory required by cuBLASLt (device memory).
 * @param workspace_size Size of the workspace memory in bytes.
 * @param M The number of rows in matrix A and C.
 * @param N The number of columns in matrix B and C.
 * @param K The number of columns in matrix A and rows in matrix B.
 * @param mode Enum specifying the transposition mode for the matrices (e.g., Transpose, NoTranspose).
 * @param accumulate If true, accumulates the result into the existing values of C (beta = 1). If false, overwrites C (beta = 0).
 * @param stream The CUDA stream on which the operation will be executed.
 */
void matmul(float* c, const float* a, const float* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

/**
 * @brief Performs a strided matrix multiplication using cuBLASLt.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute a matrix multiplication
 * operation of the form C = alpha * (A * B) + beta * C + bias, specifically handling
 * strided memory layouts and potential scaling factors for quantization.
 *
 * @param[out] c Pointer to the output matrix C (in device memory).
 * @param[in] a Pointer to the input matrix A (in device memory).
 * @param[in] b Pointer to the input matrix B (in device memory).
 * @param[in] bias Pointer to the bias vector (in device memory). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (host or device memory). Used for dequantization/scaling.
 * @param[in] scale_b Pointer to the scaling factor for matrix B (host or device memory). Used for dequantization/scaling.
 * @param[in] handle The cuBLASLt handle used to manage the operation context.
 * @param[in] workspace Pointer to the workspace memory allocated on the device.
 * @param[in] workspace_size Size of the workspace memory in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B.
 * @param[in] mode Transpose mode for the operation (e.g., indicating if A or B are transposed).
 * @param[in] accumulate If true, accumulates the result into C (beta != 0). If false, overwrites C (beta = 0).
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul_strided_c(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
                      cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                      int M, int N, int K, EMMTranspose mode, bool accumulate, int ldc, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, ldc);
}

void matmul_strided_c(float* c, const float* a, const float* b, const float* bias, const float* scale_a, const float* scale_b,
                      cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                      int M, int N, int K, EMMTranspose mode, bool accumulate, int ldc, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate, ldc);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt: C = alpha * (A x B) + beta * C + bias.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute a matrix multiplication
 * operation on the GPU. It supports mixed-precision inputs (nv_bfloat16) and float output,
 * along with optional bias addition and scaling.
 *
 * @param[out] c Pointer to the output matrix C (float).
 * @param[in] a Pointer to the input matrix A (nv_bfloat16).
 * @param[in] b Pointer to the input matrix B (nv_bfloat16).
 * @param[in] bias Pointer to the bias vector (float). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle used to manage the library context.
 * @param[in] workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param[in] workspace_size Size of the workspace in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B (inner dimension).
 * @param[in] mode Transposition mode for the operation (e.g., transpose A, transpose B).
 * @param[in] accumulate If true, accumulates the result into C (beta != 0). If false, overwrites C (beta = 0).
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul(float* c, const nv_bfloat16* a, const nv_bfloat16* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

/**
 * @brief Performs matrix multiplication using FP8 inputs and float output with cuBLASLt.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute the operation:
 * C = alpha * (op(A) * op(B)) + beta * C + bias
 * where A and B are 8-bit floating point matrices (e4m3), and C is a 32-bit floating point matrix.
 * It handles scaling factors for the quantized inputs and optional bias addition.
 *
 * @param c Pointer to the output matrix C (float).
 * @param a Pointer to the input matrix A (__nv_fp8_e4m3).
 * @param b Pointer to the input matrix B (__nv_fp8_e4m3).
 * @param bias Pointer to the bias vector (float). Can be nullptr if no bias is applied.
 * @param scale_a Pointer to the scaling factor for matrix A (float).
 * @param scale_b Pointer to the scaling factor for matrix B (float).
 * @param handle The cuBLASLt handle used to manage the library context.
 * @param workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param workspace_size Size of the workspace in bytes.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in A and rows in B (inner dimension).
 * @param mode Enum specifying the transposition mode for matrices A and B (e.g., Transpose, NoTranspose).
 * @param accumulate If true, accumulates the result into the existing values of C (beta = 1). If false, overwrites C (beta = 0).
 * @param stream The CUDA stream on which the operation will be executed.
 */
void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt with FP8 inputs and FP32 output.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute a matrix multiplication
 * operation of the form C = alpha * (op(A) * op(B)) + beta * C + bias, where A and B are
 * 8-bit floating point matrices, and C is a 32-bit floating point matrix.
 *
 * @param[out] c Pointer to the output matrix C (FP32).
 * @param[in] a Pointer to the input matrix A (__nv_fp8_e4m3).
 * @param[in] b Pointer to the input matrix B (__nv_fp8_e4m3).
 * @param[in] bias Pointer to the bias vector (nv_bfloat16). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle used to manage the library context.
 * @param[in] workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param[in] workspace_size Size of the workspace in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B (inner dimension).
 * @param[in] mode Enum specifying the transpose operation for matrices (e.g., Transpose, NoTranspose).
 * @param[in] accumulate If true, accumulates the result into the existing values of C (beta = 1.0).
 *                       If false, overwrites C (beta = 0.0).
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt with support for bfloat16 precision, bias addition, and scaling.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute the operation:
 * C = alpha * (op(A) * op(B)) + beta * C + bias
 * where alpha and beta are derived from scale factors and accumulation settings.
 *
 * @param[out] c Pointer to the output matrix C in GPU memory.
 * @param[in] a Pointer to the input matrix A in GPU memory (nv_bfloat16).
 * @param[in] b Pointer to the input matrix B in GPU memory (nv_bfloat16).
 * @param[in] bias Pointer to the bias vector in GPU memory (nv_bfloat16). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle used to manage the library context.
 * @param[in] workspace Pointer to the workspace memory allocated on the GPU.
 * @param[in] workspace_size Size of the workspace memory in bytes.
 * @param[in] M The number of rows in matrix A and C.
 * @param[in] N The number of columns in matrix B and C.
 * @param[in] K The number of columns in matrix A and rows in matrix B.
 * @param[in] mode Enum specifying the transpose operation for matrices A and B (e.g., Transpose, NoTranspose).
 * @param[in] accumulate If true, accumulates the result into the existing values of C (beta != 0). If false, overwrites C.
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt with FP8 inputs and BF16 output.
 *
 * This function acts as a wrapper around `matmul_cublaslt` to execute the operation:
 * C = alpha * (A * B) + beta * C + bias
 * where A and B are 8-bit floating point matrices, and C and bias are bfloat16.
 *
 * @param[out] c Pointer to the output matrix C (nv_bfloat16).
 * @param[in] a Pointer to the input matrix A (__nv_fp8_e4m3).
 * @param[in] b Pointer to the input matrix B (__nv_fp8_e4m3).
 * @param[in] bias Pointer to the bias vector (nv_bfloat16). Can be nullptr if no bias is applied.
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle used to manage the library context.
 * @param[in] workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param[in] workspace_size Size of the workspace in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B.
 * @param[in] mode Transpose mode for the operation (EMMTranspose).
 * @param[in] accumulate If true, accumulates the result into C (beta != 0). If false, overwrites C (beta = 0).
 * @param[in] stream The CUDA stream on which the operation will be executed.
 */
void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}

/**
 * @brief Performs matrix multiplication using cuBLASLt with mixed precision support.
 *
 * This function computes the matrix product C = A * B (or variations based on transpose mode),
 * optionally adding a bias vector and scaling factors. It acts as a wrapper around the
 * `matmul_cublaslt` implementation.
 *
 * @param[out] c Pointer to the output matrix C in nv_bfloat16 format.
 * @param[in] a Pointer to the input matrix A in __nv_fp8_e4m3 format.
 * @param[in] b Pointer to the input matrix B in __nv_fp8_e5m2 format.
 * @param[in] bias Pointer to the bias vector in nv_bfloat16 format (can be nullptr).
 * @param[in] scale_a Pointer to the scaling factor for matrix A (float).
 * @param[in] scale_b Pointer to the scaling factor for matrix B (float).
 * @param[in] handle The cuBLASLt handle to use for the operation.
 * @param[in] workspace Pointer to the device memory workspace required by cuBLASLt.
 * @param[in] workspace_size Size of the workspace in bytes.
 * @param[in] M Number of rows in matrix A and C.
 * @param[in] N Number of columns in matrix B and C.
 * @param[in] K Number of columns in A and rows in B (inner dimension).
 * @param[in] mode Transpose mode for the operation (e.g., Transpose, NoTranspose).
 * @param[in] accumulate If true, accumulates the result into the existing values of C (C += A * B).
 *                       If false, overwrites C (C = A * B).
 * @param[in] stream The CUDA stream to execute the kernel on.
 */
void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e5m2* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    matmul_cublaslt(c, a, b, bias, workspace, workspace_size, M, N, K, stream, handle, scale_a, scale_b, mode, accumulate);
}
