// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/jit/gated_delta_rule_kernels.h"

#include <sstream>
#include <stdexcept>

static const char* KERNEL_NAMES[] = {
    "gdr_cumsum_fwd",
    "gdr_cumsum_rev",
    "gdr_kkt_fwd",
    "gdr_solve_tril",
    "gdr_wy_fwd",
    "gdr_fwd_h",
    "gdr_fwd_o",
    "gdr_bwd_dv_local",
    "gdr_bwd_dhu",
    "gdr_bwd_dqkwg",
    "gdr_bwd_wy",
};

void GatedDeltaRuleKernels::load_kernel(const std::unordered_map<std::string, std::string>& manifests,
                                        const std::string& name,
                                        std::optional<JitKernel>& target) {
    auto it = manifests.find(name);
    if (it != manifests.end()) {
        target = JitKernel::load_manifest(it->second);
    }
}

void GatedDeltaRuleKernels::load(const std::unordered_map<std::string, std::string>& manifests) {
    load_kernel(manifests, "gdr_cumsum_fwd", cumsum_fwd_);
    load_kernel(manifests, "gdr_cumsum_rev", cumsum_rev_);
    load_kernel(manifests, "gdr_kkt_fwd", kkt_fwd_);
    load_kernel(manifests, "gdr_solve_tril", solve_tril_);
    load_kernel(manifests, "gdr_wy_fwd", wy_fwd_);
    load_kernel(manifests, "gdr_fwd_h", fwd_h_);
    load_kernel(manifests, "gdr_fwd_o", fwd_o_);
    load_kernel(manifests, "gdr_bwd_dv_local", bwd_dv_local_);
    load_kernel(manifests, "gdr_bwd_dhu", bwd_dhu_);
    load_kernel(manifests, "gdr_bwd_dqkwg", bwd_dqkwg_);
    load_kernel(manifests, "gdr_bwd_wy", bwd_wy_);
    load_kernel(manifests, "gdr_l2norm_fwd_q", l2norm_fwd_q_);
    load_kernel(manifests, "gdr_l2norm_bwd_q", l2norm_bwd_q_);
}

bool GatedDeltaRuleKernels::is_ready() const {
    return cumsum_fwd_ && cumsum_rev_ && kkt_fwd_ && solve_tril_ && wy_fwd_ && fwd_h_ && fwd_o_ && bwd_dv_local_ &&
           bwd_dhu_ && bwd_dqkwg_ && bwd_wy_;
}

// --- Helper macro for launch methods ---
#define LAUNCH_KERNEL(field, pretty_name)                                              \
    if (!field) {                                                                      \
        throw std::runtime_error("GatedDeltaRuleKernels: " pretty_name " not loaded"); \
    }                                                                                  \
    field->launch_triton(grid, args, num_args, stream)

void GatedDeltaRuleKernels::cumsum_fwd(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(cumsum_fwd_, "gdr_cumsum_fwd");
}

void GatedDeltaRuleKernels::kkt_fwd(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(kkt_fwd_, "gdr_kkt_fwd");
}

void GatedDeltaRuleKernels::solve_tril(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(solve_tril_, "gdr_solve_tril");
}

void GatedDeltaRuleKernels::wy_fwd(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(wy_fwd_, "gdr_wy_fwd");
}

void GatedDeltaRuleKernels::fwd_h(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(fwd_h_, "gdr_fwd_h");
}

void GatedDeltaRuleKernels::fwd_o(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(fwd_o_, "gdr_fwd_o");
}

void GatedDeltaRuleKernels::cumsum_rev(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(cumsum_rev_, "gdr_cumsum_rev");
}

void GatedDeltaRuleKernels::bwd_dv_local(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(bwd_dv_local_, "gdr_bwd_dv_local");
}

void GatedDeltaRuleKernels::bwd_dhu(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(bwd_dhu_, "gdr_bwd_dhu");
}

void GatedDeltaRuleKernels::bwd_dqkwg(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(bwd_dqkwg_, "gdr_bwd_dqkwg");
}

void GatedDeltaRuleKernels::bwd_wy(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(bwd_wy_, "gdr_bwd_wy");
}

void GatedDeltaRuleKernels::l2norm_fwd_q(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(l2norm_fwd_q_, "gdr_l2norm_fwd_q");
}

void GatedDeltaRuleKernels::l2norm_bwd_q(dim3 grid, void** args, int num_args, cudaStream_t stream) const {
    LAUNCH_KERNEL(l2norm_bwd_q_, "gdr_l2norm_bwd_q");
}

#undef LAUNCH_KERNEL
