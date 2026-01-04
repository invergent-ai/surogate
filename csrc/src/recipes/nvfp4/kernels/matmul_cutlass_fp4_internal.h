// SPDX-License-Identifier: Apache-2.0
/**
 * @file matmul_cutlass_fp4_internal.h
 * @brief Internal declarations for architecture-specific FP4 GEMM kernels
 *
 * This header declares the architecture-specific kernel functions that are
 * implemented in separate .cu files for parallel compilation.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>

// ============================================================================
// SM103 (Blackwell B300) Kernel Declarations
// ============================================================================

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

void matmul_cutlass_fp4_sm103(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream);

void matmul_cutlass_fp4_sm103_alpha(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    const float* alpha_ptr,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream);

#endif  // CUTLASS_ARCH_MMA_SM103_SUPPORTED

// ============================================================================
// SM120/SM121 (Blackwell B200, RTX 50xx) Kernel Declarations
// ============================================================================

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

void matmul_cutlass_fp4_sm120(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream);

void matmul_cutlass_fp4_sm120_f32(
    float* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream);

void matmul_cutlass_fp4_sm120_alpha(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    const float* alpha_ptr,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream);

#endif  // CUTLASS_ARCH_MMA_SM120_SUPPORTED || CUTLASS_ARCH_MMA_SM121_SUPPORTED
