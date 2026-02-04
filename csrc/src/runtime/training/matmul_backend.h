// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_TRAINING_MATMUL_BACKEND_H
#define SUROGATE_SRC_TRAINING_MATMUL_BACKEND_H

#include <string>

/**
 * @brief Backend selection for matmul operations
 *
 * Controls which library is used for matrix multiplication:
 * - CUBLASLT: Default cuBLAS Lt backend, works on all GPUs
 * - CUTLASS: Use CUTLASS kernels (auto-selects SM90/SM120 variant based on GPU)
 * - AUTO: Automatically selects based on GPU architecture and data type
 */
enum class EMatmulBackend {
    CUBLASLT,    ///< Use cuBLAS Lt (default, universal support)
    CUTLASS,     ///< Use CUTLASS kernels (SM90 per-tensor or SM120 block-scaled)
    AUTO         ///< Auto-select based on GPU and dtype
};

inline std::string to_string(EMatmulBackend backend) {
    switch (backend) {
        case EMatmulBackend::CUBLASLT: return "cublaslt";
        case EMatmulBackend::CUTLASS: return "cutlass";
        case EMatmulBackend::AUTO: return "auto";
    }
    return "unknown";
}

#endif  // SUROGATE_SRC_TRAINING_MATMUL_BACKEND_H
