// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_KERNELS_MATMUL_PLANS_H
#define SUROGATE_SRC_KERNELS_MATMUL_PLANS_H

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

typedef struct cublasLtContext* cublasLtHandle_t;

class MatmulPlanCache {
public:
    explicit MatmulPlanCache(int device_id);
    ~MatmulPlanCache();

    MatmulPlanCache(const MatmulPlanCache&) = delete;
    MatmulPlanCache& operator=(const MatmulPlanCache&) = delete;
    MatmulPlanCache(MatmulPlanCache&&) noexcept;
    MatmulPlanCache& operator=(MatmulPlanCache&&) noexcept;

    void matmul(cublasLtHandle_t handle,
                void* d, const void* a, const void* b, const void* bias,
                std::byte* workspace, std::size_t workspace_size,
                int m, int n, int k, int ldc,
                int dtype_d, int dtype_a, int dtype_b, int dtype_bias,
                bool transA, bool transB,
                const float* scale_a, const float* scale_b,
                bool accumulate,
                cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // SUROGATE_SRC_KERNELS_MATMUL_PLANS_H

