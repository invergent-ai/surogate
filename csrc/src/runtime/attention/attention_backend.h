// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Priority-based strategy registry for attention kernels (cuDNN SDPA,
// FlashAttention varlen, sliding-window custom, cuBLAS SDPA math).
// Each backend declares which shape / dtype / feature combinations it
// supports via ``supports()``; the registry picks the highest-priority
// supporting backend at dispatch time. Backends self-register on load.

#ifndef SUROGATE_RUNTIME_ATTENTION_BACKEND_H
#define SUROGATE_RUNTIME_ATTENTION_BACKEND_H

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

#include "utilities/dtype.h"
#include "utilities/tensor.h"

namespace dsl {

class DslRunState;  // forward decl — backends go through the params struct

/// Selection priorities for the built-in backends. Higher wins.
namespace attention_priority {
constexpr int kCuDNN = 100;
constexpr int kFlashVarlen = 90;
constexpr int kCustom = 80;
constexpr int kSDPA = 10;
}  // namespace attention_priority

/// All tensors + execution context needed to run a single
/// ``flash_attention`` / ``flash_attention_backward`` op. Callers fill in
/// the fields relevant to their direction (forward vs backward) and hand
/// the struct to a selected backend.
///
/// A single struct is used for both directions to avoid duplicating the
/// ~20 shared fields. Backend implementations consult only the fields
/// they need; fields that don't apply to the caller's direction (e.g.
/// ``d_qkv`` during forward) may be left default-constructed.
struct AttentionParams {
    // ---- Shape ---------------------------------------------------------
    int B = 0;
    int T = 0;
    int Hq = 0;
    int Hkv = 0;
    int Hs = 0;

    // ---- Behaviour -----------------------------------------------------
    /// Attention-window radius in tokens. ``0`` = full causal attention.
    /// Positive values enable sliding-window local attention.
    int window_size = 0;

    /// Custom softmax scale. ``0.0f`` means "kernel default" (1/sqrt(Hs)).
    float softmax_scale = 0.0f;

    bool causal = true;

    /// Forward uses BF16 kernels when ``dtype == BF16`` for the QKV/out
    /// tensors; FP32 backward paths auto-convert when needed.
    ETensorDType dtype = ETensorDType::BF16;

    // ---- Forward tensors ----------------------------------------------
    Tensor qkv;               ///< [B, T, Hq+2*Hkv, Hs]  input
    Tensor out;               ///< [B, T, Hq, Hs]        output
    Tensor lse;               ///< [B, Hq, T]  fp32      output (forward) / input (backward)
    Tensor* sinks = nullptr;  ///< [Hq] or null (GPT-OSS only)

    // ---- Backward tensors (unused during forward) ---------------------
    Tensor d_out;  ///< [B, T, Hq, Hs]        input
    Tensor d_qkv;  ///< [B, T, Hq+2*Hkv, Hs]  output

    // ---- Varlen / document masking ------------------------------------
    /// Cumulative sequence-length index into a packed (total_tokens, ...)
    /// tensor. ``nullptr`` disables the varlen path. Backends that do not
    /// support varlen must reject when this is non-null unless the shape
    /// happens to be a single dense document.
    const int32_t* cu_seqlens = nullptr;
    int num_docs = 0;
    int max_doc_seqlen = 0;
    int total_doc_tokens = 0;

    // ---- Execution context --------------------------------------------
    cudaStream_t stream = nullptr;
    cudnnHandle_t cudnn_handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;

    /// Pre-allocated cuDNN workspace. Empty (``Data == nullptr``) means
    /// the cuDNN backend will fail ``supports()``. Sized by
    /// ``workspace_bytes()`` during ``DslRunState`` allocation.
    Tensor cudnn_workspace;

    // ---- Runtime knobs -------------------------------------------------
    /// Deterministic varlen backward (opt-in via
    /// ``SUROGATE_FLASH_ATTN_VARLEN_BWD_DETERMINISTIC`` env var).
    bool deterministic_bwd = false;

    /// cuDNN backward batch chunking (``RuntimeOptions::AttBwdChunks``).
    /// ``1`` runs the whole batch at once; ``>1`` splits into
    /// ``B / chunks`` slices along the batch dim.
    int attn_bwd_chunks = 1;

    // ---- Temp allocator (backends that need scratch buffers) ----------
    /// Run state used to allocate short-lived scratch buffers
    /// (``dq_accum``, ``dsoftmax``, fp32 conversion temps, etc.). Backends
    /// must push every allocated tensor onto ``temps`` so the executor
    /// can release them at op end.
    DslRunState* run_state = nullptr;
    std::vector<Tensor>* temps = nullptr;

    /// Compute capability in the form ``major*10 + minor`` (e.g. SM120).
    /// Backends use it for hardware-gated decisions (FA3 requires SM90+,
    /// FP4/NVFP4 requires SM100+, etc.).
    int sm_version = 0;
};

/// Interface for an attention backend (cuDNN SDPA, FlashAttention varlen,
/// sliding-window custom, cuBLAS matmul fallback, …).
///
/// Backends are stateless beyond the information in ``AttentionParams``.
/// The registry caches backend instances and may call ``supports()`` /
/// ``workspace_bytes()`` on the same backend many times per op.
class AttentionBackend {
public:
    virtual ~AttentionBackend() = default;

    /// Short human-readable identifier (used in logs / error messages).
    virtual const char* name() const = 0;

    /// Selection priority — higher wins when multiple backends accept
    /// the same params. Typical values:
    ///   100  cuDNN (fastest when supported)
    ///    90  FlashAttention varlen (required for cu_seqlens)
    ///    80  Custom sliding-window kernel
    ///    10  SDPA (cuBLAS reference math, last-resort fallback)
    virtual int priority() const = 0;

    /// Returns ``true`` iff this backend can execute ``forward()`` /
    /// ``backward()`` on the given params. Must be cheap and side-effect
    /// free — the registry calls it during selection.
    virtual bool supports(const AttentionParams& p) const = 0;

    /// Scratch-workspace size in bytes needed to run on the given params.
    /// Called during ``DslRunState`` allocation to size the shared
    /// ``cudnn_workspace`` buffer. Backends that allocate temps via
    /// ``params.run_state->temp_alloc`` report 0 here — they size their
    /// own buffers per-op rather than sharing a persistent workspace.
    virtual size_t workspace_bytes(const AttentionParams& p) const {
        (void)p;
        return 0;
    }

    /// Run the forward attention op. Preconditions:
    ///   - ``supports(p)`` returned ``true`` for the same ``p``
    ///   - ``p.qkv`` / ``p.out`` / ``p.lse`` are resolved and of the
    ///     declared shapes.
    virtual void forward(AttentionParams& p) = 0;

    /// Run the backward attention op (d_qkv = ∂L/∂qkv).
    /// Same preconditions as ``forward()``; additionally ``p.d_out`` /
    /// ``p.out`` / ``p.lse`` / ``p.qkv`` and ``p.d_qkv`` are resolved.
    virtual void backward(AttentionParams& p) = 0;

    /// True if this backend's backward op produces per-rank-divergent
    /// gradients for GQA. When true and the executor is in EP with
    /// dp_size=1, the caller must ``all_reduce_avg`` ``d_qkv`` after
    /// backward to restore the shared value. FlashAttention varlen
    /// currently has this issue for GQA models.
    virtual bool gqa_backward_is_rank_divergent() const {
        return false;
    }
};

/// Process-wide registry of attention backends. Backends self-register
/// during static initialization via ``AttentionBackendRegistry::add()``.
///
/// Thread safety: ``add()`` is only called during static init (single
/// thread). ``select_*`` methods are const and safe to call from any
/// number of executor threads concurrently.
class AttentionBackendRegistry {
public:
    /// Returns the process-wide registry singleton.
    static AttentionBackendRegistry& instance();

    /// Register a backend. Call from a static initializer. Ownership is
    /// transferred to the registry; the backend outlives the process.
    void add(std::unique_ptr<AttentionBackend> backend);

    /// Select the highest-priority backend whose ``supports(p)`` is true.
    /// Throws ``std::runtime_error`` with a diagnostic if no backend
    /// matches (lists the params that made every backend reject).
    AttentionBackend& select(const AttentionParams& p) const;

    /// Workspace-size query for pre-allocation. Returns the max bytes
    /// reported by any backend that could match a shape with the given
    /// upper-bound dimensions. ``max_hs`` is the largest head size we
    /// might see across hybrid model layers.
    size_t max_workspace_bytes(int B,
                               int T,
                               int Hq,
                               int Hkv,
                               int max_hs,
                               cudnnHandle_t cudnn_handle,
                               cublasHandle_t cublas_handle) const;

    /// Read-only view of registered backends, in registration order.
    /// Primarily for tests / diagnostics.
    const std::vector<std::unique_ptr<AttentionBackend>>& backends() const {
        return mBackends;
    }

private:
    AttentionBackendRegistry() = default;
    AttentionBackendRegistry(const AttentionBackendRegistry&) = delete;
    AttentionBackendRegistry& operator=(const AttentionBackendRegistry&) = delete;

    std::vector<std::unique_ptr<AttentionBackend>> mBackends;
};

}  // namespace dsl

#endif  // SUROGATE_RUNTIME_ATTENTION_BACKEND_H
