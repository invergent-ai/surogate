// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_KERNELS_MOE_MOE_COMMON_CUH
#define SUROGATE_SRC_KERNELS_MOE_MOE_COMMON_CUH

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "kernels/kernels.h"
#include "kernels/kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

template <typename T>
constexpr cudaDataType_t cublas_dtype() {
    if constexpr (std::is_same_v<T, float>)
        return CUDA_R_32F;
    else if constexpr (std::is_same_v<T, nv_bfloat16>)
        return CUDA_R_16BF;
    else if constexpr (std::is_same_v<T, half>)
        return CUDA_R_16F;
    else
        static_assert(!sizeof(T), "Unsupported type for cuBLAS");
}

/// One GEMM per expert, in cuBLAS column-major terms:
///   C[i] = alpha * op_a(A[i]) * op_b(B[i]) + beta * C[i],  op_a(A[i]) m[i] x k[i], op_b(B[i]) k[i] x n[i],
/// issued as a loop of cublasGemmEx on the handle's current stream (the callers set it).
///
/// Not cublasGemmGroupedBatchedEx: cuBLAS runs bf16, fp16 and fp32 groups on one kernel object shared by
/// the whole process, locks that object's mutex before the launch and unlocks it only after
/// cuLaunchKernel returns (cuBLAS 13.1.0.3: the setup at libcublas.so.13+0x576cc0 returns holding the
/// mutex at this+0x120, the run at +0x579eb0 unlocks it after the launch). A launch blocks while its
/// GPU's queue is full, and in multi-GPU training a queue stays full while its GPU waits in a collective
/// for another GPU. When that GPU's worker thread reaches any grouped GEMM it waits for the mutex, never
/// issues its part of the collective, and every GPU hangs (8-GPU bf16 MoE LoRA training, 2026-09-25).
/// A thread blocked in a cublasGemmEx launch holds nothing another worker waits for.
///
/// Each GEMM writes its own output block, so the result does not depend on scheduling.
template <typename T>
inline void moe_expert_gemms(cublasHandle_t handle,
                             cublasOperation_t transa,
                             cublasOperation_t transb,
                             const std::vector<int>& m,
                             const std::vector<int>& n,
                             const std::vector<int>& k,
                             float alpha,
                             const std::vector<const T*>& A,
                             const std::vector<int>& lda,
                             const std::vector<const T*>& B,
                             const std::vector<int>& ldb,
                             float beta,
                             const std::vector<T*>& C,
                             const std::vector<int>& ldc) {
    for (std::size_t i = 0; i < m.size(); ++i) {
        CUBLAS_CHECK(cublasGemmEx(handle,
                                  transa,
                                  transb,
                                  m[i],
                                  n[i],
                                  k[i],
                                  &alpha,
                                  A[i],
                                  cublas_dtype<T>(),
                                  lda[i],
                                  B[i],
                                  cublas_dtype<T>(),
                                  ldb[i],
                                  &beta,
                                  C[i],
                                  cublas_dtype<T>(),
                                  ldc[i],
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
    }
}

#endif  // SUROGATE_SRC_KERNELS_MOE_MOE_COMMON_CUH
