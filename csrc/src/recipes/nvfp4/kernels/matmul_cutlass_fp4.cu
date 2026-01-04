// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
/**
 * @file matmul_cutlass_fp4.cu
 * @brief CUTLASS-based FP4 GEMM dispatcher for Blackwell architectures
 *
 * This file provides the public API and runtime dispatch logic for FP4 GEMM kernels.
 * Architecture-specific kernel implementations are in separate files for parallel compilation:
 * - matmul_cutlass_fp4_sm100.cu: SM100 (B200) kernels
 * - matmul_cutlass_fp4_sm103.cu: SM103 (B300) kernels
 * - matmul_cutlass_fp4_sm120.cu: SM120/SM121 (RTX 50xx) kernels
 *
 * Architecture differences:
 * - SM100 (B200): Uses Sm100 arch with KernelTmaWarpSpecialized1SmMxf4Sm100 schedule, tile K=256
 * - SM103 (B300): Uses Sm103BlockScaledConfig with scale atom layout (8x4x4), tile K=768
 * - SM120/121 (RTX 50xx): Uses Sm1xxBlockScaledConfig with scale atom layout (32x4), tile K=128
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>

// Include CUTLASS headers only for the architecture support macros
#include "cutlass/arch/config.h"

// Internal declarations for architecture-specific functions
#include "matmul_cutlass_fp4_internal.h"

// Include kernels.h for public API declarations
#include "kernels/kernels.h"

// ============================================================================
// Public API Implementation
// ============================================================================

bool cutlass_supports_fp4() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int sm_version = props.major * 10 + props.minor;

    // FP4 requires Blackwell (SM100+)
    // SM90 (Hopper) does NOT have FP4 tensor core support
    return sm_version >= 100;
}

std::size_t cutlass_fp4_workspace_size(int M, int N, int K) {
    // Conservative workspace allocation for FP4 GEMM
    // Scale factor tensors and kernel workspace
    return 16 * 1024 * 1024;  // 16 MB
}

void matmul_cutlass_fp4(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int sm_version = props.major * 10 + props.minor;

    if (sm_version < 100) {
        throw std::runtime_error("CUTLASS FP4 GEMM requires Blackwell (SM100+). "
                                 "SM90 (Hopper) does not have FP4 tensor core support.");
    }

    // Dispatch to the appropriate kernel based on SM version
    // Each architecture has its own kernel with different tile sizes and schedules:
    // - SM100 (B200): tile K=256, KernelTmaWarpSpecialized1SmMxf4Sm100
    // - SM103 (B300): tile K=768, Sm103BlockScaledConfig
    // - SM120+ (RTX 50xx): tile K=128, Sm1xxBlockScaledConfig
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (sm_version == 100) {
        matmul_cutlass_fp4_sm100(d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
        return;
    }
#endif

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)
    if (sm_version == 103) {
        matmul_cutlass_fp4_sm103(d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
        return;
    }
#endif

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    if (sm_version >= 120) {
        matmul_cutlass_fp4_sm120(d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
        return;
    }
#endif

    throw std::runtime_error("CUTLASS FP4 GEMM not compiled for this architecture. "
                             "SM100 (B200), SM103 (B300), and SM120+ (RTX 50xx) supported. "
                             "Ensure CUDA_ARCHITECTURES includes 100a, 103a, 120a, or 121a.");
}

void matmul_cutlass_fp4_f32(
    float* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int sm_version = props.major * 10 + props.minor;

    if (sm_version < 100) {
        throw std::runtime_error("CUTLASS FP4 GEMM (FP32 out) requires Blackwell (SM100+). "
                                 "SM90 (Hopper) does not have FP4 tensor core support.");
    }

    // Dispatch to the appropriate kernel based on SM version
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (sm_version == 100) {
        matmul_cutlass_fp4_sm100_f32(d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
        return;
    }
#endif

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)
    if (sm_version == 103) {
        matmul_cutlass_fp4_sm103_f32(d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
        return;
    }
#endif

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    if (sm_version >= 120) {
        matmul_cutlass_fp4_sm120_f32(d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
        return;
    }
#endif

    throw std::runtime_error("CUTLASS FP4 GEMM (FP32 out) not compiled for this architecture. "
                             "SM100 (B200), SM103 (B300), and SM120+ (RTX 50xx) supported. "
                             "Ensure CUDA_ARCHITECTURES includes 100a, 103a, 120a, or 121a.");
}

void matmul_cutlass_fp4_alpha(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    const float* alpha_ptr,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int sm_version = props.major * 10 + props.minor;

    if (sm_version < 100) {
        throw std::runtime_error("CUTLASS FP4 GEMM (alpha-ptr) requires Blackwell (SM100+). "
                                 "SM90 (Hopper) does not have FP4 tensor core support.");
    }

    // Dispatch to the appropriate kernel based on SM version
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (sm_version == 100) {
        matmul_cutlass_fp4_sm100_alpha(d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
        return;
    }
#endif

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)
    if (sm_version == 103) {
        matmul_cutlass_fp4_sm103_alpha(d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
        return;
    }
#endif

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    if (sm_version >= 120) {
        matmul_cutlass_fp4_sm120_alpha(d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
        return;
    }
#endif

    throw std::runtime_error("CUTLASS FP4 GEMM (alpha-ptr) not compiled for this architecture. "
                             "SM100 (B200), SM103 (B300), and SM120+ (RTX 50xx) supported. "
                             "Ensure CUDA_ARCHITECTURES includes 100a, 103a, 120a, or 121a.");
}
