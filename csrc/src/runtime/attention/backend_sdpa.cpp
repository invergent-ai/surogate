// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Reference SDPA math backend (PyTorch's ``math`` SDPA equivalent).
// Two cuBLAS matmuls around an explicit softmax. Last-resort fallback
// when the fused kernels can't take the shape — typically head_dim > 256
// with full causal attention.

#include <memory>

#include "kernels/kernels.h"
#include "runtime/attention/attention_backend.h"
#include "utilities/tensor.h"

namespace dsl {
namespace {

class SDPAAttention final : public AttentionBackend {
public:
    const char* name() const override {
        return "sdpa";
    }

    int priority() const override {
        return attention_priority::kSDPA;
    }

    bool supports(const AttentionParams& p) const override {
        if (p.Hs <= 0) {
            return false;
        }
        if (p.window_size > 0) {
            // Reference SDPA has no notion of local attention.
            return false;
        }
        if (p.cublas_handle == nullptr) {
            return false;
        }
        return true;
    }

    void forward(AttentionParams& p) override {
        attention_forward_matmul(p.out,
                                 p.lse,
                                 p.qkv,
                                 p.B,
                                 p.T,
                                 p.Hq,
                                 p.Hkv,
                                 p.Hs,
                                 p.cublas_handle,
                                 p.stream,
                                 p.softmax_scale);
    }

    void backward(AttentionParams& p) override {
        attention_backward_matmul(p.d_qkv,
                                  p.lse,
                                  p.out,
                                  p.d_out,
                                  p.qkv,
                                  p.B,
                                  p.T,
                                  p.Hq,
                                  p.Hkv,
                                  p.Hs,
                                  p.cublas_handle,
                                  p.stream,
                                  p.softmax_scale);
    }
};

struct SDPAAttentionAutoRegister {
    SDPAAttentionAutoRegister() {
        AttentionBackendRegistry::instance().add(std::make_unique<SDPAAttention>());
    }
};
const SDPAAttentionAutoRegister _sdpa_attention_auto_register;

}  // namespace
}  // namespace dsl
