// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "runtime/jit/jit_kernel.h"
#include "utilities/dtype.h"

class GlmMatmulKernels {
public:
    void load(const std::unordered_map<std::string, std::string>& manifests);
    bool is_ready() const;
    void matmul(void* out,
                const void* a,
                const void* b,
                ETensorDType da,
                ETensorDType db,
                ETensorDType dc,
                int m,
                int n,
                int k,
                int am,
                int ak,
                int bn,
                int bk,
                int oc,
                float alpha,
                float beta,
                cudaStream_t stream,
                const void* bias_bf16 = nullptr,
                const void* bias_fp32 = nullptr) const;
    void grouped(void* out,
                 const void* input,
                 const void* weights,
                 const int* offsets,
                 ETensorDType dtype,
                 int experts,
                 int rows,
                 int m,
                 int k,
                 float alpha,
                 float beta,
                 cudaStream_t stream) const;

private:
    std::unordered_map<std::string, JitKernel> mKernels;
};

// Native GEMM helpers and LoRA hooks run on the executor's worker thread.
// Scope the override to one GLM graph execution, including recomputation.
const GlmMatmulKernels* active_glm_matmul();
class ScopedGlmMatmul {
public:
    explicit ScopedGlmMatmul(const GlmMatmulKernels& kernels, bool enabled);
    ~ScopedGlmMatmul();
    ScopedGlmMatmul(const ScopedGlmMatmul&) = delete;
    ScopedGlmMatmul& operator=(const ScopedGlmMatmul&) = delete;

private:
    const GlmMatmulKernels* mPrevious;
};
