// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <unordered_map>
#include <vector>
#include "runtime/jit/jit_kernel.h"
#include "utilities/tensor.h"

/// Native launches of the vendored FLA KDA training pipeline. All temporary
/// storage belongs to the caller, so execution performs no allocation or host
/// synchronization and can be captured in a CUDA graph.
class KimiDeltaRuleKernels {
public:
    KimiDeltaRuleKernels() = default;
    KimiDeltaRuleKernels(const KimiDeltaRuleKernels&) = delete;
    KimiDeltaRuleKernels& operator=(const KimiDeltaRuleKernels&) = delete;
    KimiDeltaRuleKernels(KimiDeltaRuleKernels&&) = default;
    KimiDeltaRuleKernels& operator=(KimiDeltaRuleKernels&&) = default;
    void load(const std::unordered_map<std::string, std::string>& manifests);
    [[nodiscard]] bool is_ready() const;
    static std::size_t workspace_bytes(int B, int T, int H, int D, int num_docs, bool backward);

    /// Forward inputs: q/k/v (BF16), decay (FP32), beta (BF16).
    /// Backward prepends d_output and returns five FP32 gradients.
    /// cu_seqlens contains num_docs+1 INT32 offsets into the flattened B*T
    /// tokens, including padding tokens. nullptr means B independent rows.
    void run(bool backward,
             const std::vector<Tensor>& inputs,
             const std::vector<Tensor>& outputs,
             const std::int32_t* cu_seqlens,
             int num_docs,
             const Tensor& workspace,
             cudaStream_t stream) const;

private:
    std::unordered_map<std::string, JitKernel> mKernels;
};
