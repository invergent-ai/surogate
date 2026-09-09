// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#include "runtime/jit/glm_matmul_kernels.h"
#include <array>
#include <stdexcept>

namespace {
thread_local const GlmMatmulKernels* Current = nullptr;
constexpr std::array Names = {"glm_matmul_bf16_bf16_bf16",
                              "glm_matmul_bf16_bf16_fp32",
                              "glm_matmul_bf16_fp32_bf16",
                              "glm_matmul_bf16_fp32_fp32",
                              "glm_matmul_fp32_bf16_bf16",
                              "glm_matmul_fp32_bf16_fp32",
                              "glm_matmul_fp32_fp32_bf16",
                              "glm_matmul_fp32_fp32_fp32",
                              "glm_grouped_matmul_bf16",
                              "glm_grouped_matmul_fp32"};
const char* dtype_name(ETensorDType dtype) {
    if (dtype == ETensorDType::BF16) return "bf16";
    if (dtype == ETensorDType::FP32) return "fp32";
    throw std::runtime_error("GLM invariant matmul requires BF16 or FP32");
}
}  // namespace

const GlmMatmulKernels* active_glm_matmul() {
    return Current;
}
ScopedGlmMatmul::ScopedGlmMatmul(const GlmMatmulKernels& kernels, bool enabled)
    : mPrevious(Current) {
    Current = enabled && kernels.is_ready() ? &kernels : nullptr;
}
ScopedGlmMatmul::~ScopedGlmMatmul() {
    Current = mPrevious;
}

void GlmMatmulKernels::load(const std::unordered_map<std::string, std::string>& manifests) {
    for (const auto* name : Names)
        if (auto it = manifests.find(name); it != manifests.end())
            mKernels.insert_or_assign(name, JitKernel::load_manifest(it->second));
    if (!mKernels.empty() && !is_ready()) throw std::runtime_error("Incomplete GLM matmul manifests");
}

bool GlmMatmulKernels::is_ready() const {
    return mKernels.size() == Names.size();
}

void GlmMatmulKernels::matmul(void* out,
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
                              cudaStream_t stream) const {
    const auto name = std::string("glm_matmul_") + dtype_name(da) + "_" + dtype_name(db) + "_" + dtype_name(dc);
    const auto& kernel = mKernels.at(name);
    void* args[] = {&a, &b, &out, &m, &n, &k, &am, &ak, &bn, &bk, &oc, &alpha, &beta};
    kernel.launch_triton(dim3((n + 15) / 16, (m + 63) / 64), args, std::size(args), stream);
}

void GlmMatmulKernels::grouped(void* out,
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
                               cudaStream_t stream) const {
    const auto& kernel = mKernels.at(std::string("glm_grouped_matmul_") + dtype_name(dtype));
    if (experts > kernel.meta().const_int("NE", 0)) throw std::runtime_error("GLM matmul expert geometry mismatch");
    void* args[] = {&input, &weights, &out, &offsets, &m, &k, &experts, &alpha, &beta};
    kernel.launch_triton(dim3((rows + 15) / 16 + experts, (m + 63) / 64), args, std::size(args), stream);
}
