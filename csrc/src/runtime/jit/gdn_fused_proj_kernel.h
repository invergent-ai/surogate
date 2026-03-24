// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// JIT kernel manager for GDN fused projection (Qwen3.5 linear-attention decode).

#ifndef SUROGATE_SRC_RUNTIME_JIT_GDN_FUSED_PROJ_KERNEL_H
#define SUROGATE_SRC_RUNTIME_JIT_GDN_FUSED_PROJ_KERNEL_H

#include <optional>
#include <string>
#include <unordered_map>

#include "runtime/jit/jit_kernel.h"

/// Manages the JIT-compiled GDN fused projection kernel for inference decode.
class GdnFusedProjKernel {
public:
    /// Load kernel from manifests (called once in CompiledExecutor init).
    void load(const std::unordered_map<std::string, std::string>& manifests);

    /// Check if the contiguous variant is loaded and ready.
    [[nodiscard]] bool is_ready() const { return contiguous_.has_value(); }

    /// Launch the contiguous-layout split kernel.
    /// Args: mixed_qkv, z, b, a (outputs), mixed_qkvz, mixed_ba (inputs)
    void split_contiguous(dim3 grid, void** args, int num_args,
                          cudaStream_t stream) const;

private:
    std::optional<JitKernel> contiguous_;
};

#endif // SUROGATE_SRC_RUNTIME_JIT_GDN_FUSED_PROJ_KERNEL_H
