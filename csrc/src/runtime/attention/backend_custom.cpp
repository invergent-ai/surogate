// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Sliding-window attention. Dense sequences run directly through the
// in-tree kernel. Packed sequences are lowered to per-document dense
// calls so document boundaries are explicit and cannot leak.

#include <memory>
#include <stdexcept>
#include <vector>

#include "kernels/kernels.h"
#include "runtime/attention/attention_backend.h"
#include "runtime/attention/attention_kernels.h"
#include "runtime/dsl/dsl_run_state.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"

namespace dsl {
namespace {

struct PackedDocSegment {
    int batch_idx = 0;
    long row_start = 0;
    long global_start = 0;
    long length = 0;
};

std::vector<PackedDocSegment> collect_packed_doc_segments(const AttentionParams& p) {
    if (p.cu_seqlens_cpu == nullptr) {
        throw std::logic_error("CustomFlashAttention: missing host cu_seqlens for packed sliding attention");
    }
    if (p.num_docs <= 0) {
        throw std::logic_error("CustomFlashAttention: packed sliding attention requires num_docs > 0");
    }

    std::vector<PackedDocSegment> docs;
    docs.reserve(static_cast<std::size_t>(p.num_docs));

    long row_offset = 0;
    int batch_idx = 0;
    long covered = 0;
    for (int doc_idx = 0; doc_idx < p.num_docs; ++doc_idx) {
        const long start = static_cast<long>(p.cu_seqlens_cpu[doc_idx]);
        const long end = static_cast<long>(p.cu_seqlens_cpu[doc_idx + 1]);
        const long len = end - start;
        if (len <= 0) {
            throw std::logic_error("CustomFlashAttention: encountered empty packed document");
        }
        if (start != covered) {
            throw std::logic_error("CustomFlashAttention: cu_seqlens are not contiguous");
        }
        if (batch_idx >= p.B) {
            throw std::logic_error("CustomFlashAttention: packed documents exceed batch capacity");
        }
        if (row_offset + len > p.T) {
            throw std::logic_error("CustomFlashAttention: packed document crosses a batch row");
        }

        PackedDocSegment seg;
        seg.batch_idx = batch_idx;
        seg.row_start = row_offset;
        seg.global_start = start;
        seg.length = len;
        docs.push_back(seg);
        covered = end;
        row_offset += len;
        if (row_offset == p.T) {
            row_offset = 0;
            ++batch_idx;
        }
    }

    if (covered != static_cast<long>(p.total_doc_tokens) || covered != static_cast<long>(p.B) * p.T) {
        throw std::logic_error("CustomFlashAttention: packed documents do not cover the full dense batch");
    }
    if (batch_idx != p.B || row_offset != 0) {
        throw std::logic_error("CustomFlashAttention: packed documents leave a partial batch row");
    }
    return docs;
}

Tensor flatten_bt(const Tensor& src, long heads, long hs) {
    Tensor flat = src;
    flat.Rank = 3;
    flat.Sizes[0] = src.Sizes[0] * src.Sizes[1];
    flat.Sizes[1] = heads;
    flat.Sizes[2] = hs;
    flat.Sizes[3] = 1;
    flat.Sizes[4] = 1;
    return flat;
}

Tensor doc_slice_4d(const Tensor& flat, long global_start, long length, long heads, long hs) {
    Tensor doc = slice(flat, 0, global_start, global_start + length);
    doc.Rank = 4;
    doc.Sizes[0] = 1;
    doc.Sizes[1] = length;
    doc.Sizes[2] = heads;
    doc.Sizes[3] = hs;
    doc.Sizes[4] = 1;
    return doc;
}

void copy_lse_doc_to_dense(Tensor& dense_lse,
                           const Tensor& doc_lse,
                           int batch_idx,
                           long row_start,
                           int Hq,
                           int T,
                           cudaStream_t stream) {
    float* dst = dense_lse.get<float>() + static_cast<long>(batch_idx) * Hq * T + row_start;
    const float* src = doc_lse.get<float>();
    const std::size_t dst_pitch = static_cast<std::size_t>(T) * sizeof(float);
    const std::size_t src_pitch = static_cast<std::size_t>(doc_lse.Sizes[2]) * sizeof(float);
    const std::size_t width = static_cast<std::size_t>(doc_lse.Sizes[2]) * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DAsync(dst,
                                 dst_pitch,
                                 src,
                                 src_pitch,
                                 width,
                                 static_cast<std::size_t>(Hq),
                                 cudaMemcpyDeviceToDevice,
                                 stream));
}

void copy_lse_dense_to_doc(Tensor& doc_lse,
                           const Tensor& dense_lse,
                           int batch_idx,
                           long row_start,
                           int Hq,
                           int T,
                           cudaStream_t stream) {
    const float* src = dense_lse.get<float>() + static_cast<long>(batch_idx) * Hq * T + row_start;
    float* dst = doc_lse.get<float>();
    const std::size_t src_pitch = static_cast<std::size_t>(T) * sizeof(float);
    const std::size_t dst_pitch = static_cast<std::size_t>(doc_lse.Sizes[2]) * sizeof(float);
    const std::size_t width = static_cast<std::size_t>(doc_lse.Sizes[2]) * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DAsync(dst,
                                 dst_pitch,
                                 src,
                                 src_pitch,
                                 width,
                                 static_cast<std::size_t>(Hq),
                                 cudaMemcpyDeviceToDevice,
                                 stream));
}

void forward_packed_sliding_dense(AttentionParams& p, Tensor& out_tensor, Tensor& qkv_tensor) {
    DslRunState& rs = *p.run_state;
    std::vector<Tensor>& temps = *p.temps;
    const auto docs = collect_packed_doc_segments(p);

    Tensor qkv_flat = flatten_bt(qkv_tensor, p.Hq + 2 * p.Hkv, p.Hs);
    Tensor out_flat = flatten_bt(out_tensor, p.Hq, p.Hs);

    for (const PackedDocSegment& doc : docs) {
        Tensor qkv_doc = doc_slice_4d(qkv_flat, doc.global_start, doc.length, p.Hq + 2 * p.Hkv, p.Hs);
        Tensor out_doc = doc_slice_4d(out_flat, doc.global_start, doc.length, p.Hq, p.Hs);
        Tensor lse_doc = rs.temp_alloc(ETensorDType::FP32, {1, p.Hq, doc.length}, "packed_sliding_lse_doc");
        temps.push_back(lse_doc);

        attention_forward_custom(out_doc,
                                 lse_doc,
                                 qkv_doc,
                                 /*B=*/1,
                                 static_cast<int>(doc.length),
                                 p.Hq,
                                 p.Hkv,
                                 p.Hs,
                                 p.window_size,
                                 p.stream,
                                 p.softmax_scale);
        copy_lse_doc_to_dense(p.lse, lse_doc, doc.batch_idx, doc.row_start, p.Hq, p.T, p.stream);
    }
}

void backward_packed_sliding_fp32(AttentionParams& p,
                                  Tensor& d_qkv_tensor,
                                  const Tensor& lse_tensor,
                                  const Tensor& out_tensor,
                                  const Tensor& d_out_tensor,
                                  const Tensor& qkv_tensor) {
    DslRunState& rs = *p.run_state;
    std::vector<Tensor>& temps = *p.temps;
    const auto docs = collect_packed_doc_segments(p);

    Tensor qkv_flat = flatten_bt(qkv_tensor, p.Hq + 2 * p.Hkv, p.Hs);
    Tensor out_flat = flatten_bt(out_tensor, p.Hq, p.Hs);
    Tensor d_out_flat = flatten_bt(d_out_tensor, p.Hq, p.Hs);
    Tensor d_qkv_flat = flatten_bt(d_qkv_tensor, p.Hq + 2 * p.Hkv, p.Hs);

    for (const PackedDocSegment& doc : docs) {
        Tensor lse_doc = rs.temp_alloc(ETensorDType::FP32, {1, p.Hq, doc.length}, "packed_sliding_lse_doc");
        temps.push_back(lse_doc);
        copy_lse_dense_to_doc(lse_doc, lse_tensor, doc.batch_idx, doc.row_start, p.Hq, p.T, p.stream);

        Tensor qkv_doc = doc_slice_4d(qkv_flat, doc.global_start, doc.length, p.Hq + 2 * p.Hkv, p.Hs);
        Tensor out_doc = doc_slice_4d(out_flat, doc.global_start, doc.length, p.Hq, p.Hs);
        Tensor d_out_doc = doc_slice_4d(d_out_flat, doc.global_start, doc.length, p.Hq, p.Hs);
        Tensor d_qkv_doc = doc_slice_4d(d_qkv_flat, doc.global_start, doc.length, p.Hq + 2 * p.Hkv, p.Hs);

        attention_backward_custom(d_qkv_doc,
                                  lse_doc,
                                  out_doc,
                                  d_out_doc,
                                  qkv_doc,
                                  /*B=*/1,
                                  static_cast<int>(doc.length),
                                  p.Hq,
                                  p.Hkv,
                                  p.Hs,
                                  p.window_size,
                                  p.stream,
                                  p.softmax_scale);
    }
}

class CustomFlashAttention final : public AttentionBackend {
public:
    const char* name() const override {
        return "custom_sliding";
    }

    int priority() const override {
        return attention_priority::kCustom;
    }

    bool supports(const AttentionParams& p) const override {
        // Disabled pending a correctness fix. Sliding attention now routes
        // to flash-varlen's window_size_left path, which applies the same
        // local causal mask via FlashAttention varlen.
        (void)p;
        return false;
    }

    void forward(AttentionParams& p) override {
        if (p.cu_seqlens != nullptr) {
            forward_packed_sliding_dense(p, p.out, p.qkv);
            return;
        }

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

        if (p.cu_seqlens != nullptr) {
            if (p.out.DType == ETensorDType::FP32) {
                backward_packed_sliding_fp32(p, p.d_qkv, p.lse, p.out, p.d_out, p.qkv);
                return;
            }

            if (p.out.DType != ETensorDType::BF16) {
                throw std::logic_error(
                    "CustomFlashAttention::backward: unsupported packed-sliding dtype (only BF16 / FP32)");
            }

            auto shape_vec = [](const Tensor& t) {
                return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
            };

            Tensor out_f32 = rs.temp_alloc(ETensorDType::FP32, shape_vec(p.out), "packed_sliding_out_f32");
            Tensor d_out_f32 = rs.temp_alloc(ETensorDType::FP32, shape_vec(p.d_out), "packed_sliding_d_out_f32");
            Tensor qkv_f32 = rs.temp_alloc(ETensorDType::FP32, shape_vec(p.qkv), "packed_sliding_qkv_f32");
            Tensor d_qkv_f32 = rs.temp_alloc(ETensorDType::FP32, shape_vec(p.d_qkv), "packed_sliding_d_qkv_f32");
            temps.push_back(out_f32);
            temps.push_back(d_out_f32);
            temps.push_back(qkv_f32);
            temps.push_back(d_qkv_f32);

            convert_dtype(out_f32.get<float>(), p.out.get<nv_bfloat16>(), p.out.nelem(), p.stream);
            convert_dtype(d_out_f32.get<float>(), p.d_out.get<nv_bfloat16>(), p.d_out.nelem(), p.stream);
            convert_dtype(qkv_f32.get<float>(), p.qkv.get<nv_bfloat16>(), p.qkv.nelem(), p.stream);

            backward_packed_sliding_fp32(p, d_qkv_f32, p.lse, out_f32, d_out_f32, qkv_f32);

            convert_dtype(p.d_qkv.get<nv_bfloat16>(), d_qkv_f32.get<float>(), p.d_qkv.nelem(), p.stream);
            return;
        }

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
