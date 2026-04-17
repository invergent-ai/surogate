// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// cuDNN SDPA backend. Supports head_dim in [8, 128] divisible by 8,
// full causal attention, dense (non-packed) sequences, BF16 tensors,
// with a pre-allocated workspace. Fastest path when the shape fits.

#include <memory>
#include <stdexcept>
#include <vector>

#include "kernels/kernels.h"
#include "runtime/attention/attention_backend.h"
#include "runtime/dsl/dsl_run_state.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

// cuDNN SDPA backward internally rejects head_dim > 128. Gating here
// keeps us consistent with cuDNN's own constraint.
constexpr int kCuDNNMaxHeadDim = 128;

class CuDNNAttention final : public AttentionBackend {
public:
    const char* name() const override {
        return "cudnn";
    }

    int priority() const override {
        return attention_priority::kCuDNN;
    }

    bool supports(const AttentionParams& p) const override {
        if (p.Hs <= 0 || p.Hs > kCuDNNMaxHeadDim || (p.Hs % 8) != 0) {
            return false;
        }
        if (p.window_size > 0) {
            return false;
        }
        if (p.cu_seqlens != nullptr) {
            return false;
        }
        if (p.dtype != ETensorDType::BF16) {
            return false;
        }
        if (p.cudnn_handle == nullptr) {
            return false;
        }
        // Workspace presence is deliberately not checked: supports() is
        // also called during DslRunState sizing, before the workspace
        // exists. forward() / backward() assert via require_workspace().
        return true;
    }

    size_t workspace_bytes(const AttentionParams& p) const override {
        if (!supports(p)) {
            return 0;
        }
        // The workspace is shared by forward + backward. Backward is the
        // larger consumer, so sizing from the backward graph is safe for
        // both directions.
        return cudnn_get_workspace_size(p.B, p.T, p.Hq, p.Hkv, p.Hs, p.cudnn_handle);
    }

    void forward(AttentionParams& p) override {
        require_workspace(p);
        attention_forward_cudnn(p.out,
                                p.lse,
                                p.qkv,
                                p.cudnn_workspace,
                                p.cudnn_handle,
                                p.B,
                                p.T,
                                p.Hq,
                                p.Hkv,
                                p.Hs,
                                p.stream);
    }

    void backward(AttentionParams& p) override {
        require_workspace(p);

        const int chunks = p.attn_bwd_chunks;
        if (chunks < 1) {
            throw std::runtime_error("CuDNNAttention::backward: attn_bwd_chunks must be >= 1");
        }
        if (chunks == 1) {
            attention_backward_cudnn(p.d_qkv,
                                     p.lse,
                                     p.out,
                                     p.d_out,
                                     p.qkv,
                                     p.cudnn_workspace,
                                     p.cudnn_handle,
                                     p.B,
                                     p.T,
                                     p.Hq,
                                     p.Hkv,
                                     p.Hs,
                                     p.stream);
            return;
        }

        // Split the batch into ``chunks`` slices to fit within the shared
        // workspace when ``B`` is too large for cuDNN's peak buffer use.
        const long chunk_B = div_exact(static_cast<long>(p.B), static_cast<long>(chunks));
        for (int c = 0; c < chunks; ++c) {
            const long start = static_cast<long>(c) * chunk_B;
            const long end = start + chunk_B;

            Tensor d_out_chunk = slice(p.d_out, 0, start, end);
            Tensor out_chunk = slice(p.out, 0, start, end);
            Tensor lse_chunk = slice(p.lse, 0, start, end);
            Tensor qkv_chunk = slice(p.qkv, 0, start, end);
            Tensor d_qkv_chunk = slice(p.d_qkv, 0, start, end);

            attention_backward_cudnn(d_qkv_chunk,
                                     lse_chunk,
                                     out_chunk,
                                     d_out_chunk,
                                     qkv_chunk,
                                     p.cudnn_workspace,
                                     p.cudnn_handle,
                                     static_cast<int>(chunk_B),
                                     p.T,
                                     p.Hq,
                                     p.Hkv,
                                     p.Hs,
                                     p.stream);
        }
    }

private:
    static void require_workspace(const AttentionParams& p) {
        if (p.cudnn_workspace.Data == nullptr) {
            throw std::runtime_error("CuDNNAttention: selected backend but cudnn_workspace is unallocated. "
                                     "This indicates a workspace-sizing miss in DslRunState.");
        }
    }
};

// Self-register at load time.
struct CuDNNAttentionAutoRegister {
    CuDNNAttentionAutoRegister() {
        AttentionBackendRegistry::instance().add(std::make_unique<CuDNNAttention>());
    }
};
const CuDNNAttentionAutoRegister _cudnn_attention_auto_register;

}  // namespace
}  // namespace dsl
