// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Kernel manager for the gated delta rule Triton kernels.
// Loads pre-compiled cubins from manifests and provides typed launch methods
// for each sub-kernel in the forward/backward pipeline.
//
// Usage:
//   GatedDeltaRuleKernels kernels;
//   kernels.load(options.JitKernelManifests);
//
//   // Forward pipeline
//   kernels.cumsum_fwd(grid, args, stream);
//   kernels.kkt_fwd(grid, args, stream);
//   ...

#ifndef SUROGATE_SRC_RUNTIME_JIT_GATED_DELTA_RULE_KERNELS_H
#define SUROGATE_SRC_RUNTIME_JIT_GATED_DELTA_RULE_KERNELS_H

#include <optional>
#include <string>
#include <unordered_map>

#include "runtime/jit/jit_kernel.h"

/// Manages all Triton sub-kernels for the gated delta rule layer.
///
/// The kernels are compiled AOT by Python (surogate/kernels/gated_delta_rule.py)
/// and loaded from JSON manifests at model init time. Each kernel is a separate
/// cubin with its own autotuned configuration.
///
/// Kernel naming convention (manifest keys):
///   gdr_cumsum_fwd     — per-chunk cumulative sum of g (forward)
///   gdr_cumsum_rev     — per-chunk reverse cumulative sum (for dg in backward)
///   gdr_kkt_fwd        — A = beta * K @ K^T * exp(g_diff), strict lower tri
///   gdr_solve_tril     — (I + A)^{-1} via block-recursive inversion
///   gdr_wy_fwd         — w = A_inv @ (k*beta*exp(g)), u = A_inv @ (v*beta)
///   gdr_fwd_h          — state recurrence (multi-block K up to 256)
///   gdr_fwd_o          — output: o = scale * (q @ h + causal_attn @ v_new)
///   gdr_bwd_dv_local   — local dv = A^T @ do
///   gdr_bwd_dhu        — backward state recurrence
///   gdr_bwd_dqkwg      — dq, dk, dw, dg gradients
///   gdr_bwd_wy         — backward through WY representation
///
class GatedDeltaRuleKernels {
public:
    /// Load all gated delta rule kernels from manifests.
    /// @param manifests  Maps kernel name -> manifest JSON path.
    ///                   Only entries with "gdr_" prefix are loaded.
    void load(const std::unordered_map<std::string, std::string>& manifests);

    /// Check if all required kernels are loaded and ready.
    [[nodiscard]] bool is_ready() const;

    // ======================================================================
    // Forward pipeline launches
    // ======================================================================

    /// L2 normalization forward (Q/K head vectors). Grid: (NT, B*H).
    void l2norm_fwd_q(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// L2 normalization backward (Q/K head vectors). Grid: (NT, B*H).
    void l2norm_bwd_q(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// Per-chunk cumulative sum of g. Grid: (NT, B*H).
    void cumsum_fwd(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// Scaled dot KKT. Grid: (NT, B*H).
    void kkt_fwd(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// Solve triangular (I+A)^{-1}. Grid: (NT, B*H).
    void solve_tril(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// WY forward (w, u computation). Grid: (NT, B*H).
    void wy_fwd(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// State recurrence forward. Grid: (cdiv(V,BV), B*H).
    void fwd_h(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// Output computation. Grid: (cdiv(V,BV), NT, B*H).
    void fwd_o(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    // ======================================================================
    // Backward pipeline launches
    // ======================================================================

    /// Reverse cumulative sum of dg. Grid: (NT, B*H).
    void cumsum_rev(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// Local dv computation. Grid: (NT, B*H).
    void bwd_dv_local(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// Backward state recurrence. Grid: (cdiv(V,BV), B*H).
    void bwd_dhu(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// Backward dq, dk, dw, dg. Grid: (NK, NT, B*H).
    void bwd_dqkwg(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// Backward through WY representation. Grid: (NT, B*H).
    void bwd_wy(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    // ======================================================================
    // Recurrent (decode / inference) kernel
    // ======================================================================

    /// Fused recurrent forward (single-token or short sequences). Grid: (cdiv(V,BV), B*H).
    void recurrent_fwd(dim3 grid, void** args, int num_args, cudaStream_t stream) const;

    /// Check if the recurrent kernel is loaded (optional — only needed for generation).
    [[nodiscard]] bool has_recurrent_fwd() const { return recurrent_fwd_.has_value(); }

private:
    /// Load a single kernel from manifests, if present.
    void load_kernel(const std::unordered_map<std::string, std::string>& manifests,
                     const std::string& name,
                     std::optional<JitKernel>& target);

    // Forward kernels
    std::optional<JitKernel> cumsum_fwd_;
    std::optional<JitKernel> kkt_fwd_;
    std::optional<JitKernel> solve_tril_;
    std::optional<JitKernel> wy_fwd_;
    std::optional<JitKernel> fwd_h_;
    std::optional<JitKernel> fwd_o_;

    // Backward kernels
    std::optional<JitKernel> cumsum_rev_;
    std::optional<JitKernel> bwd_dv_local_;
    std::optional<JitKernel> bwd_dhu_;
    std::optional<JitKernel> bwd_dqkwg_;
    std::optional<JitKernel> bwd_wy_;

    // L2 normalization kernels (used when use_qk_l2norm_in_kernel=true)
    std::optional<JitKernel> l2norm_fwd_q_;
    std::optional<JitKernel> l2norm_bwd_q_;

    // Recurrent forward kernel (single-token decode / inference)
    std::optional<JitKernel> recurrent_fwd_;
};

#endif // SUROGATE_SRC_RUNTIME_JIT_GATED_DELTA_RULE_KERNELS_H
