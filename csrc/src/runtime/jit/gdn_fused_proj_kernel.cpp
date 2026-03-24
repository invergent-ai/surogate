// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/jit/gdn_fused_proj_kernel.h"

#include <stdexcept>

void GdnFusedProjKernel::load(
    const std::unordered_map<std::string, std::string>& manifests) {
    auto it = manifests.find("gdn_fused_proj_contiguous");
    if (it != manifests.end()) {
        contiguous_ = JitKernel::load_manifest(it->second);
    }
}

void GdnFusedProjKernel::split_contiguous(
    dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    if (!contiguous_) {
        throw std::runtime_error(
            "GdnFusedProjKernel: gdn_fused_proj_contiguous not loaded");
    }
    contiguous_->launch_triton(grid, args, num_args, stream);
}
