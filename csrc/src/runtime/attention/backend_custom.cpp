// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Sliding-window attention for unpacked inputs (``window_size > 0 &&
// cu_seqlens == nullptr``). Forward runs on any dtype; backward kernel
// is FP32-only, so BF16 tensors round-trip through an FP32 scratch.

#include <memory>
#include <stdexcept>
#include <vector>

#include "kernels/kernels.h"
#include "runtime/attention/attention_backend.h"
#include "runtime/dsl/dsl_run_state.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"

namespace dsl {
namespace {

class CustomFlashAttention final : public AttentionBackend {
public:
    const char* name() const override {
        return "custom_sliding";
    }

    int priority() const override {
        return attention_priority::kCustom;
    }

    bool supports(const AttentionParams& p) const override {
        if (p.Hs <= 0) {
            return false;
        }
        if (p.window_size <= 0) {
            return false;
        }
        if (p.cu_seqlens != nullptr) {
            // Packed + sliding combines cleanly in FlashVarlen.
            return false;
        }
        if (p.run_state == nullptr || p.temps == nullptr) {
            return false;
        }
        return true;
    }

    void forward(AttentionParams& p) override {
        attention_forward_custom(p.out,
                                 p.lse,
                                 p.qkv,
                                 p.B,
                                 p.T,
                                 p.Hq,
                                 p.Hkv,
                                 p.Hs,
                                 p.window_size,
                                 p.stream,
                                 p.softmax_scale);
    }

    void backward(AttentionParams& p) override {
        DslRunState& rs = *p.run_state;
        std::vector<Tensor>& temps = *p.temps;

        if (p.out.DType == ETensorDType::FP32) {
            // FP32 inputs: call the kernel directly.
            attention_backward_custom(p.d_qkv,
                                      p.lse,
                                      p.out,
                                      p.d_out,
                                      p.qkv,
                                      p.B,
                                      p.T,
                                      p.Hq,
                                      p.Hkv,
                                      p.Hs,
                                      p.window_size,
                                      p.stream,
                                      p.softmax_scale);
            return;
        }

        if (p.out.DType != ETensorDType::BF16) {
            throw std::logic_error("CustomFlashAttention::backward: unsupported dtype (only BF16 / FP32)");
        }

        // BF16 round-trip via FP32 scratch. Custom backward kernel is FP32-only.
        auto shape_vec = [](const Tensor& t) {
            return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
        };

        Tensor out_f32 = rs.temp_alloc(ETensorDType::FP32, shape_vec(p.out), "flash_attention_out_f32");
        Tensor d_out_f32 = rs.temp_alloc(ETensorDType::FP32, shape_vec(p.d_out), "flash_attention_d_out_f32");
        Tensor qkv_f32 = rs.temp_alloc(ETensorDType::FP32, shape_vec(p.qkv), "flash_attention_qkv_f32");
        Tensor d_qkv_f32 = rs.temp_alloc(ETensorDType::FP32, shape_vec(p.d_qkv), "flash_attention_d_qkv_f32");
        temps.push_back(out_f32);
        temps.push_back(d_out_f32);
        temps.push_back(qkv_f32);
        temps.push_back(d_qkv_f32);

        convert_dtype(out_f32.get<float>(), p.out.get<nv_bfloat16>(), p.out.nelem(), p.stream);
        convert_dtype(d_out_f32.get<float>(), p.d_out.get<nv_bfloat16>(), p.d_out.nelem(), p.stream);
        convert_dtype(qkv_f32.get<float>(), p.qkv.get<nv_bfloat16>(), p.qkv.nelem(), p.stream);

        attention_backward_custom(d_qkv_f32,
                                  p.lse,
                                  out_f32,
                                  d_out_f32,
                                  qkv_f32,
                                  p.B,
                                  p.T,
                                  p.Hq,
                                  p.Hkv,
                                  p.Hs,
                                  p.window_size,
                                  p.stream,
                                  p.softmax_scale);

        convert_dtype(p.d_qkv.get<nv_bfloat16>(), d_qkv_f32.get<float>(), p.d_qkv.nelem(), p.stream);
    }
};

struct CustomFlashAttentionAutoRegister {
    CustomFlashAttentionAutoRegister() {
        AttentionBackendRegistry::instance().add(std::make_unique<CustomFlashAttention>());
    }
};
const CustomFlashAttentionAutoRegister _custom_attention_auto_register;

}  // namespace
}  // namespace dsl
