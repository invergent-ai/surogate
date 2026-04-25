// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FlashAttention varlen backend. BF16, head_dim <= 256, with or
// without packed sequences, with or without sliding window. Synthesizes
// a dense cu_seqlens when the caller didn't supply one. Per-op scratch
// (dq_accum, dsoftmax, expanded dk/dv for GQA) comes from the executor's
// temp allocator — no persistent workspace.

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "kernels/kernels.h"
#include "runtime/attention/attention_backend.h"
#include "runtime/attention/attention_kernels.h"
#include "runtime/dsl/dsl_run_state.h"
#include "utilities/tensor.h"

namespace dsl {
namespace {

constexpr int kFlashMaxHeadDim = 256;

class FlashVarlenAttention final : public AttentionBackend {
public:
    const char* name() const override {
        return "flash_varlen";
    }

    int priority() const override {
        return attention_priority::kFlashVarlen;
    }

    bool gqa_backward_is_rank_divergent() const override {
        return true;
    }

    bool supports(const AttentionParams& p) const override {
        if (p.Hs <= 0 || p.Hs > kFlashMaxHeadDim) {
            return false;
        }
        if (p.dtype != ETensorDType::BF16) {
            return false;
        }
        if (p.run_state == nullptr || p.temps == nullptr) {
            // We need the run-state / temp list for dq_accum + dsoftmax
            // plus the synthesized cu_seqlens buffer.
            return false;
        }
        // FlashAttention-varlen natively supports sliding windows via its
        // window_size_left parameter — the kernel is ~4x faster than the
        // SDPA math path for sliding layers on Gemma4-sized shapes.
        // (Previous gate disabled this pending Gemma4 boundary-drift debug;
        // re-enabled after narrowing the drift to the packed-boundary
        // bookkeeping that is now handled by synthesize_dense_cu_seqlens.)
        return true;
    }

    void forward(AttentionParams& p) override {
        // supports() gated on p.dtype == BF16, which gives p.qkv.DType and
        // p.out.DType — the flash varlen kernels take raw bf16 pointers.
        synthesize_dense_cu_seqlens_if_needed(p);
        ensure_varlen_ranges(p);

        attention_forward_flash_varlen(p.out.get<nv_bfloat16>(),
                                       p.lse.get<float>(),
                                       p.qkv.get<nv_bfloat16>(),
                                       p.cu_seqlens,
                                       p.num_docs,
                                       p.max_doc_seqlen,
                                       p.total_doc_tokens,
                                       p.Hq,
                                       p.Hkv,
                                       p.Hs,
                                       p.stream,
                                       p.softmax_scale,
                                       std::max(p.window_size, 0));
    }

    void backward(AttentionParams& p) override {
        synthesize_dense_cu_seqlens_if_needed(p);
        ensure_varlen_ranges(p);

        DslRunState& rs = *p.run_state;
        std::vector<Tensor>& temps = *p.temps;

        // FlashAttention varlen backward rounds head_dim up to the next
        // 32-element (HS<=128) or 64-element (HS>128) multiple for its
        // internal accumulator layout.
        const int Hs_rounded = p.Hs <= 128 ? ((p.Hs + 31) / 32) * 32 : ((p.Hs + 63) / 64) * 64;
        const long padded_total = static_cast<long>(p.total_doc_tokens) + 128L * static_cast<long>(p.num_docs);
        const long dq_accum_stride = padded_total * static_cast<long>(p.Hq) * static_cast<long>(Hs_rounded);

        const int split_den = std::max(1, p.num_docs * p.Hq);
        const int dq_accum_splits =
            p.deterministic_bwd ? std::max(1, (rs.DeviceProp.multiProcessorCount + split_den - 1) / split_den) : 1;
        const long dq_accum_elems = dq_accum_stride * static_cast<long>(dq_accum_splits);
        const long dsoftmax_elems = static_cast<long>(p.Hq) * padded_total;

        Tensor dq_accum = rs.temp_alloc(ETensorDType::FP32, {dq_accum_elems}, "flash_attention_dq_accum");
        Tensor dsoftmax = rs.temp_alloc(ETensorDType::FP32, {dsoftmax_elems}, "flash_attention_dsoftmax");
        temps.push_back(dq_accum);
        temps.push_back(dsoftmax);

        // FlashAttention varlen backward uses both dsoftmax and dq_accum as
        // accumulation/work buffers. The executor temp allocator intentionally
        // reuses memory across ops, so these buffers must be cleared before
        // each launch even in the non-deterministic path.
        fill_zero(dsoftmax, p.stream);
        CUDA_CHECK(cudaMemsetAsync(dq_accum.Data, 0, static_cast<size_t>(dq_accum_elems) * sizeof(float), p.stream));

        // GQA expanded dk/dv: flash backward writes dK/dV at Hq head
        // positions, but the interleaved ``qkv`` buffer only has Hkv
        // K/V slots. Allocate separate (total_q, Hq, Hs) BF16 buffers
        // when Hq != Hkv so the kernel has somewhere to land the
        // pre-reduce gradients.
        nv_bfloat16* dk_exp_ptr = nullptr;
        nv_bfloat16* dv_exp_ptr = nullptr;
        if (p.Hq != p.Hkv) {
            const long exp_elems =
                static_cast<long>(p.total_doc_tokens) * static_cast<long>(p.Hq) * static_cast<long>(p.Hs);
            Tensor dk_expanded = rs.temp_alloc(ETensorDType::BF16, {exp_elems}, "flash_attention_dk_expanded");
            Tensor dv_expanded = rs.temp_alloc(ETensorDType::BF16, {exp_elems}, "flash_attention_dv_expanded");
            temps.push_back(dk_expanded);
            temps.push_back(dv_expanded);
            dk_exp_ptr = dk_expanded.get<nv_bfloat16>();
            dv_exp_ptr = dv_expanded.get<nv_bfloat16>();
            // Flash-varlen GQA backward leaves portions of these buffers
            // untouched before the Hq->Hkv reduce/scatter. Under EP with
            // dp_size=1, rank-local temp allocator history differs, so
            // stale BF16 contents here would become rank-specific K/V
            // noise. Zero them up front to avoid that.
            fill_zero(dk_expanded, p.stream);
            fill_zero(dv_expanded, p.stream);
        }

        // ``convert_dQ`` writes every Q element, but the MHA (Hq==Hkv)
        // path leaves the K/V sections untouched. For GQA, the final
        // reduce_scatter overwrites them anyway. Zeroing here is cheap
        // and removes both edge cases.
        fill_zero(p.d_qkv, p.stream);

        attention_backward_flash_varlen(p.d_qkv.get<nv_bfloat16>(),
                                        p.lse.get<float>(),
                                        p.out.get<nv_bfloat16>(),
                                        p.d_out.get<nv_bfloat16>(),
                                        p.qkv.get<nv_bfloat16>(),
                                        p.cu_seqlens,
                                        dq_accum.get<float>(),
                                        dsoftmax.get<float>(),
                                        dk_exp_ptr,
                                        dv_exp_ptr,
                                        p.num_docs,
                                        p.max_doc_seqlen,
                                        p.total_doc_tokens,
                                        p.Hq,
                                        p.Hkv,
                                        p.Hs,
                                        p.deterministic_bwd,
                                        p.stream,
                                        p.softmax_scale,
                                        std::max(p.window_size, 0));
    }

private:
    /// When the caller didn't provide cu_seqlens, build a dense one treating
    /// each batch row as a distinct document of length T. Stores the temp
    /// tensor in ``p.temps`` and updates ``p.cu_seqlens``/``num_docs``/
    /// ``max_doc_seqlen``/``total_doc_tokens`` in place.
    static void synthesize_dense_cu_seqlens_if_needed(AttentionParams& p) {
        if (p.cu_seqlens != nullptr) {
            return;
        }
        DslRunState& rs = *p.run_state;
        p.num_docs = p.B;
        p.max_doc_seqlen = p.T;
        p.total_doc_tokens = p.B * p.T;
        Tensor dense = rs.temp_alloc(ETensorDType::INT32, {static_cast<long>(p.num_docs + 1)}, "generated_cu_seqlens");
        p.temps->push_back(dense);
        fill_dense_cu_seqlens(dense.get<int32_t>(), p.num_docs, p.max_doc_seqlen, p.stream);
        p.cu_seqlens = dense.get<int32_t>();
    }

    static void ensure_varlen_ranges(const AttentionParams& p) {
        if (p.cu_seqlens == nullptr) {
            throw std::logic_error("FlashVarlenAttention: cu_seqlens is still null after synthesis");
        }
        if (p.num_docs <= 0 || p.max_doc_seqlen <= 0 || p.total_doc_tokens <= 0) {
            throw std::logic_error("FlashVarlenAttention: varlen document ranges are unset "
                                   "(num_docs / max_doc_seqlen / total_doc_tokens)");
        }
    }
};

struct FlashVarlenAttentionAutoRegister {
    FlashVarlenAttentionAutoRegister() {
        AttentionBackendRegistry::instance().add(std::make_unique<FlashVarlenAttention>());
    }
};
const FlashVarlenAttentionAutoRegister _flash_varlen_attention_auto_register;

}  // namespace
}  // namespace dsl
