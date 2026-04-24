// Capture-safe attention backend built on the ported PyTorch
// mem-efficient attention kernel (cutlass, BF16, SM80 ABI, GMEM variant
// with unbounded head_dim). Priority 95: above flash-varlen (90) but
// below cuDNN (100). It's selected when flash-varlen's Hs<=256 gate
// rejects and cu_seqlens is present (the packed-sequence case where
// SDPA's per-doc Python slicing is capture-unsafe).
//
// Forward only for this prototype. Backward pass is the next wiring
// step; SDPA stays registered until both directions land.

#include "runtime/attention/attention_backend.h"
#include "runtime/attention/mem_eff/mem_eff_dispatch.h"
#include "runtime/dsl/dsl_run_state.h"
#include "utilities/tensor.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace dsl {
namespace {

class MemEffAttention final : public AttentionBackend {
public:
    const char* name() const override {
        return "mem_eff";
    }

    int priority() const override {
        // Above flash-varlen (90) so we win for Hs>256 cases where
        // flash-varlen would reject. Below cuDNN (100).
        return 95;
    }

    bool supports(const AttentionParams& p) const override {
        auto reject = [&](const char* reason) {
            if (const char* dbg = std::getenv("SUROGATE_DEBUG_ATTN_SELECT")) {
                if (dbg[0] == '1') {
                    std::fprintf(stderr,
                                 "[mem_eff reject] %s (Hq=%d Hkv=%d Hs=%d varlen=%s)\n",
                                 reason,
                                 p.Hq,
                                 p.Hkv,
                                 p.Hs,
                                 p.cu_seqlens ? "yes" : "no");
                }
            }
            return false;
        };
        if (p.Hs <= 0) return reject("Hs<=0");
        if (p.dtype != ETensorDType::BF16) return reject("not bf16");
        // GQA (Hq != Hkv) is not supported by the underlying kernel
        // without a KV broadcast pre-step; skip for now and let
        // flash-varlen / SDPA handle those cases until we add broadcast.
        // Supports MHA (Hq==Hkv) and MQA (Hkv==1) via k_strideH=0.
        // True GQA with Hkv>1 && Hkv<Hq needs a head-group stride the
        // kernel doesn't expose; reject those cases for now.
        if (p.Hkv != p.Hq && p.Hkv != 1) return reject("GQA with Hkv>1 not supported");
        // Only claim the cu_seqlens case initially — this is where SDPA
        // is the worst fit (per-doc slicing). Dense non-varlen stays on
        // cuDNN / flash-varlen which are already capture-safe.
        if (p.cu_seqlens == nullptr) return reject("no cu_seqlens");
        if (p.run_state == nullptr || p.temps == nullptr) return reject("no run_state/temps");
        return true;
    }

    void forward(AttentionParams& p) override {
        DslRunState& rs = *p.run_state;
        std::vector<Tensor>& temps = *p.temps;

        // Compute layout from AttentionParams shape fields.
        const int B = p.B;
        const int T = p.T;
        const int Hq = p.Hq;
        const int Hkv = p.Hkv;
        const int Hs = p.Hs;
        const int HtotQKV = Hq + 2 * Hkv;

        // qkv is [B, T, Hq+2*Hkv, Hs] contiguous in BF16. Strides (elem).
        const int32_t q_strideM = HtotQKV * Hs;
        const int32_t q_strideH = Hs;
        const int64_t q_strideB = static_cast<int64_t>(T) * HtotQKV * Hs;

        auto* qkv_ptr = p.qkv.get<nv_bfloat16>();
        const void* q_ptr = qkv_ptr;
        const void* k_ptr = qkv_ptr + Hq * Hs;
        const void* v_ptr = qkv_ptr + (Hq + Hkv) * Hs;

        // Output is [B, T, Hq, Hs] contiguous BF16.
        void* out_ptr = p.out.get<nv_bfloat16>();
        const int32_t o_strideM = Hq * Hs;

        // GMEM variant needs a caller-provided FP32 accumulator.
        // Packed layout: sized by num_docs * max_doc_seqlen (over-
        // provisioned vs. total_doc_tokens but matches the kernel's
        // per-doc indexing). Pushed to temps so the executor frees it
        // at op end (capture-safe since temp_alloc uses the stack
        // arena, not cudaMalloc, once the arena is preallocated).
        const int max_seqlen = p.max_doc_seqlen;
        const int num_batches = p.num_docs;
        const long accum_elems = static_cast<long>(num_batches) * static_cast<long>(max_seqlen) *
                                 static_cast<long>(Hq) * static_cast<long>(Hs);
        Tensor accum = rs.temp_alloc(ETensorDType::FP32, {accum_elems}, "mem_eff_output_accum");
        temps.push_back(accum);

        // LSE scratch: the cutlass kernel writes [num_docs, Hq, lse_dim]
        // but our runtime's p.lse is [B, Hq, T]. Allocate a scratch LSE
        // sized to the kernel's layout, hand it to the kernel, and
        // scatter into p.lse via cu_seqlens-indexed positions so the
        // rest of the runtime sees values in the layout it expects.
        const int lse_dim = (max_seqlen + 7) / 8 * 8;
        const long lse_kernel_elems = static_cast<long>(num_batches) * Hq * lse_dim;
        Tensor lse_kernel_tensor = rs.temp_alloc(ETensorDType::FP32, {lse_kernel_elems}, "mem_eff_lse_scratch");
        temps.push_back(lse_kernel_tensor);
        float* lse_kernel_ptr = lse_kernel_tensor.get<float>();

        surogate::mem_eff::ForwardArgs args;
        args.q_ptr = q_ptr;
        args.k_ptr = k_ptr;
        args.v_ptr = v_ptr;
        args.out_ptr = out_ptr;
        args.lse_ptr = lse_kernel_ptr;
        args.output_accum_ptr = accum.get<float>();

        args.num_queries = max_seqlen;
        args.num_keys = max_seqlen;
        args.num_batches = num_batches;
        args.num_heads = Hq;
        args.head_dim_qk = Hs;
        args.head_dim_v = Hs;

        args.q_strideM = q_strideM;
        args.q_strideH = q_strideH;
        args.q_strideB = q_strideB;
        args.k_strideM = q_strideM;
        // MQA (Hkv==1, Hq>1): every Q head reads the single K/V head.
        // Setting strideH=0 makes the kernel's per-head pointer advance
        // (head_id * strideH) collapse to zero — exactly the MQA
        // semantics.
        args.k_strideH = (Hkv == 1 && Hq > 1) ? 0 : q_strideH;
        args.k_strideB = q_strideB;
        args.v_strideM = q_strideM;
        args.v_strideH = (Hkv == 1 && Hq > 1) ? 0 : q_strideH;
        args.v_strideB = q_strideB;
        args.o_strideM = o_strideM;

        args.seqstart_q_ptr = p.cu_seqlens;
        args.seqstart_k_ptr = p.cu_seqlens;
        args.seqlen_k_ptr = nullptr;

        args.causal = true;
        args.window_size = std::max(p.window_size, 0);
        args.softmax_scale = p.softmax_scale > 0.0f ? p.softmax_scale : (1.0f / std::sqrt(static_cast<float>(Hs)));
        args.stream = p.stream;

        surogate::mem_eff::forward_bf16_sm80(args);

        // Scatter scratch LSE into runtime's dense [B, Hq, T] layout.
        if (p.lse.Data) {
            surogate::mem_eff::lse_scatter_kernel_to_runtime(lse_kernel_ptr,
                                                             p.lse.get<float>(),
                                                             p.cu_seqlens,
                                                             num_batches,
                                                             Hq,
                                                             max_seqlen,
                                                             lse_dim,
                                                             T,
                                                             p.stream);
        }
    }

    void backward(AttentionParams& p) override {
        DslRunState& rs = *p.run_state;
        std::vector<Tensor>& temps = *p.temps;

        const int T = p.T;
        const int Hq = p.Hq;
        const int Hkv = p.Hkv;
        const int Hs = p.Hs;
        const int HtotQKV = Hq + 2 * Hkv;
        const int num_batches = p.num_docs;
        const int max_seqlen = p.max_doc_seqlen;

        // qkv and d_qkv share the interleaved [B, T, Hq+2*Hkv, Hs]
        // layout. Slice pointers for each of q/k/v and d_q/d_k/d_v.
        auto* qkv_ptr = p.qkv.get<nv_bfloat16>();
        auto* d_qkv_ptr = p.d_qkv.get<nv_bfloat16>();

        const int32_t q_strideM = HtotQKV * Hs;
        const int32_t q_strideH = Hs;
        const int64_t q_strideB = static_cast<int64_t>(T) * HtotQKV * Hs;

        // out and d_out are [B, T, Hq, Hs] contiguous.
        auto* out_ptr = p.out.get<nv_bfloat16>();
        auto* d_out_ptr = p.d_out.get<nv_bfloat16>();
        const int64_t o_strideM = Hq * Hs;
        const int64_t o_strideH = Hs;
        const int64_t o_strideB = static_cast<int64_t>(T) * Hq * Hs;

        // p.lse is in runtime layout [B, Hq, T] (saved from forward).
        // The kernel needs [num_docs, Hq, lse_dim] layout — gather from
        // the runtime layout into a scratch buffer before launch.
        if (p.lse.Data == nullptr) {
            throw std::runtime_error("mem_eff backend: backward requires saved LSE from forward");
        }
        const int lse_dim = (max_seqlen + 7) / 8 * 8;
        const long lse_kernel_elems = static_cast<long>(num_batches) * Hq * lse_dim;
        Tensor lse_kernel_tensor = rs.temp_alloc(ETensorDType::FP32, {lse_kernel_elems}, "mem_eff_bwd_lse_scratch");
        temps.push_back(lse_kernel_tensor);
        float* lse_ptr = lse_kernel_tensor.get<float>();
        surogate::mem_eff::lse_gather_runtime_to_kernel(lse_ptr,
                                                        p.lse.get<float>(),
                                                        p.cu_seqlens,
                                                        num_batches,
                                                        Hq,
                                                        max_seqlen,
                                                        lse_dim,
                                                        T,
                                                        p.stream);
        const int64_t lse_strideB = static_cast<int64_t>(Hq) * lse_dim;
        const int64_t lse_strideH = lse_dim;

        // Delta: [num_batches, Hq, lse_dim] fp32 (matches LSE layout —
        // the kernel reads delta_strideH alongside lse_strideH).
        Tensor delta =
            rs.temp_alloc(ETensorDType::FP32,
                          {static_cast<long>(num_batches), static_cast<long>(Hq), static_cast<long>(lse_dim)},
                          "mem_eff_bwd_delta");
        temps.push_back(delta);

        // Delta-compute kernel mirrors the kernel's LSE layout so the
        // kernel reads from the same [doc, head, q_in_doc] positions it
        // wrote during fwd. Since `out`/`d_out` are in runtime dense
        // [B, T, H, Hs] layout, we scatter-index the packed tokens via
        // cu_seqlens — but delta is per-query so for a prototype we
        // compute it into the kernel layout directly.
        //
        // compute_delta_bf16 is dense: it iterates over [num_batches,
        // num_heads, num_queries] with strides indexing into dense
        // out/d_out. That doesn't work for packed — instead, delta is
        // computed by the kernel when kKernelComputesDelta is true, OR
        // we need a packed-aware delta kernel. Fall back to launching
        // the delta kernel with num_batches=B (dense), num_queries=T,
        // and writing into a dense-layout delta, then gather to the
        // kernel layout via cu_seqlens like we did for LSE.
        Tensor delta_dense = rs.temp_alloc(ETensorDType::FP32,
                                           {static_cast<long>(p.B), static_cast<long>(Hq), static_cast<long>(T)},
                                           "mem_eff_bwd_delta_dense");
        temps.push_back(delta_dense);
        surogate::mem_eff::compute_delta_bf16(out_ptr,
                                              d_out_ptr,
                                              delta_dense.get<float>(),
                                              /*num_batches=*/p.B,
                                              Hq,
                                              /*num_queries=*/T,
                                              Hs,
                                              o_strideB,
                                              o_strideM,
                                              o_strideH,
                                              o_strideB,
                                              o_strideM,
                                              o_strideH,
                                              Hq * T,
                                              T,
                                              p.stream);
        // Gather delta_dense [B, Hq, T] into delta [num_docs, Hq, lse_dim].
        surogate::mem_eff::lse_gather_runtime_to_kernel(delta.get<float>(),
                                                        delta_dense.get<float>(),
                                                        p.cu_seqlens,
                                                        num_batches,
                                                        Hq,
                                                        max_seqlen,
                                                        lse_dim,
                                                        T,
                                                        p.stream);

        // Build BackwardArgs and probe workspace size via the kernel's
        // own helper. Allocate workspace, zero if required, then launch.
        const bool is_mqa = (Hkv == 1 && Hq > 1);

        // For MQA: write backward dK/dV into a separate [total, Hq, Hs]
        // scratch buffer, then reduce across Hq heads into d_qkv's
        // single Hkv slot. For MHA: write directly into d_qkv sections.
        // total_tokens = B*T is the PACKED layout size (d_qkv sits in
        // this space); num_docs*max_seqlen would over-count for docs
        // shorter than max.
        nv_bfloat16* d_k_target = nullptr;
        nv_bfloat16* d_v_target = nullptr;
        const long total_tokens = static_cast<long>(p.B) * T;
        if (is_mqa) {
            Tensor dk_partial = rs.temp_alloc(ETensorDType::BF16,
                                              {total_tokens, static_cast<long>(Hq), static_cast<long>(Hs)},
                                              "mem_eff_bwd_dk_partial");
            Tensor dv_partial = rs.temp_alloc(ETensorDType::BF16,
                                              {total_tokens, static_cast<long>(Hq), static_cast<long>(Hs)},
                                              "mem_eff_bwd_dv_partial");
            temps.push_back(dk_partial);
            temps.push_back(dv_partial);
            d_k_target = dk_partial.get<nv_bfloat16>();
            d_v_target = dv_partial.get<nv_bfloat16>();
        } else {
            d_k_target = d_qkv_ptr + Hq * Hs;
            d_v_target = d_qkv_ptr + (Hq + Hkv) * Hs;
        }

        surogate::mem_eff::BackwardArgs args;
        args.q_ptr = qkv_ptr;
        args.k_ptr = qkv_ptr + Hq * Hs;
        args.v_ptr = qkv_ptr + (Hq + Hkv) * Hs;
        args.out_ptr = out_ptr;
        args.d_out_ptr = d_out_ptr;
        args.lse_ptr = lse_ptr;
        args.d_q_ptr = d_qkv_ptr;
        args.d_k_ptr = d_k_target;
        args.d_v_ptr = d_v_target;
        args.delta_ptr = delta.get<float>();

        args.num_queries = max_seqlen;
        args.num_keys = max_seqlen;
        args.num_batches = num_batches;
        args.num_heads = Hq;
        args.head_dim_qk = Hs;
        args.head_dim_v = Hs;

        args.q_strideM = q_strideM;
        args.q_strideH = q_strideH;
        args.q_strideB = q_strideB;
        args.k_strideM = q_strideM;
        // MQA: broadcast K/V via strideH=0 (see forward).
        args.k_strideH = is_mqa ? 0 : q_strideH;
        args.k_strideB = q_strideB;
        args.v_strideM = q_strideM;
        args.v_strideH = is_mqa ? 0 : q_strideH;
        args.v_strideB = q_strideB;

        args.o_strideB = o_strideB;
        args.o_strideH = o_strideH;
        args.gO_strideB = o_strideB;
        args.gO_strideM = o_strideM;
        args.gO_strideH = o_strideH;

        args.lse_strideB = lse_strideB;
        args.lse_strideH = lse_strideH;
        args.delta_strideB = static_cast<int64_t>(Hq) * lse_dim;
        args.delta_strideH = lse_dim;

        if (is_mqa) {
            // dQ writes interleaved d_qkv, strideM=HtotQKV*Hs.
            // Kernel computes gQ_strideM = multiplier * num_heads *
            // head_dim. For HtotQKV=(Hq+2), we need
            // multiplier * Hq * Hs == HtotQKV * Hs. No integer
            // multiplier satisfies this for general Hq,Hkv — so we
            // configure multiplier=1, and the kernel will see
            // gQ_strideM = Hq * Hs. That's wrong for interleaved dQ.
            //
            // Fix: write dQ into a separate [total, Hq, Hs] scratch
            // buffer too, then scatter-copy into d_qkv's Q section
            // before mqa-reducing K/V.
            Tensor dq_partial = rs.temp_alloc(ETensorDType::BF16,
                                              {total_tokens, static_cast<long>(Hq), static_cast<long>(Hs)},
                                              "mem_eff_bwd_dq_partial");
            temps.push_back(dq_partial);
            args.d_q_ptr = dq_partial.get<nv_bfloat16>();
            args.gQKV_strideM_multiplier = 1;
            args.gQ_strideH = Hs;
            args.gK_strideH = Hs;
            args.gV_strideH = Hs;
            args.gQ_strideB = 0;  // varlen path ignores gQ_strideB when cu_seqlens is set
            args.gK_strideB = 0;
            args.gV_strideB = 0;
        } else {
            // MHA: Hkv==Hq so HtotQKV = 3*Hq. Set multiplier=3 so the
            // kernel's gQ_strideM = 3 * Hq * Hs matches the interleaved
            // layout.
            args.gQKV_strideM_multiplier = 3;
            args.gQ_strideH = q_strideH;
            args.gK_strideH = q_strideH;
            args.gV_strideH = q_strideH;
            args.gQ_strideB = q_strideB;
            args.gK_strideB = q_strideB;
            args.gV_strideB = q_strideB;
        }

        args.cu_seqlens_q_ptr = p.cu_seqlens;
        args.cu_seqlens_k_ptr = p.cu_seqlens;

        args.causal = true;
        args.window_size = std::max(p.window_size, 0);
        args.softmax_scale = p.softmax_scale > 0.0f ? p.softmax_scale : (1.0f / std::sqrt(static_cast<float>(Hs)));
        args.num_splits_key = 1;
        args.stream = p.stream;

        const std::size_t ws_bytes = surogate::mem_eff::backward_workspace_bytes(args);
        if (ws_bytes > 0) {
            const long ws_elems = static_cast<long>(ws_bytes / sizeof(float));
            Tensor workspace = rs.temp_alloc(ETensorDType::FP32, {ws_elems}, "mem_eff_bwd_workspace");
            temps.push_back(workspace);
            args.workspace_ptr = workspace.get<float>();
            if (surogate::mem_eff::backward_workspace_needs_zero(args)) {
                cudaMemsetAsync(args.workspace_ptr, 0, ws_bytes, p.stream);
            }
        }

        surogate::mem_eff::backward_bf16_sm80(args);

        if (is_mqa) {
            // The kernel wrote dQ into the [total, Hq, Hs] scratch and
            // dK/dV into [total, Hq, Hs] scratches. Now:
            //   1. Copy dQ scratch into d_qkv's Q section (interleaved).
            //   2. Reduce dK/dV scratches across Hq into d_qkv's K/V
            //      section (single Hkv=1 slot).
            auto* dq_partial = static_cast<const nv_bfloat16*>(args.d_q_ptr);
            auto* dk_partial = static_cast<const nv_bfloat16*>(args.d_k_ptr);
            auto* dv_partial = static_cast<const nv_bfloat16*>(args.d_v_ptr);

            // Scatter dQ: partial [total, Hq, Hs] → d_qkv[:, :Hq, :]
            //   partial_strideM = Hq * Hs, partial_strideH = Hs.
            //   out_strideM = (Hq + 2*Hkv) * Hs = HtotQKV * Hs.
            // We can reuse the mqa_reduce kernel as a "scatter copy" by
            // passing Hq=1 (identity) per head — but it's simpler to
            // just do a 2D memcpy. cudaMemcpy2DAsync handles stride-
            // mismatched copies cheaply.
            CUDA_CHECK(cudaMemcpy2DAsync(/*dst=*/d_qkv_ptr,
                                         /*dpitch=*/static_cast<std::size_t>(HtotQKV) * Hs * sizeof(nv_bfloat16),
                                         /*src=*/dq_partial,
                                         /*spitch=*/static_cast<std::size_t>(Hq) * Hs * sizeof(nv_bfloat16),
                                         /*width=*/static_cast<std::size_t>(Hq) * Hs * sizeof(nv_bfloat16),
                                         /*height=*/static_cast<std::size_t>(total_tokens),
                                         cudaMemcpyDeviceToDevice,
                                         p.stream));

            // Reduce dK/dV across Hq heads into d_qkv's K/V slot.
            nv_bfloat16* dk_out = d_qkv_ptr + Hq * Hs;
            nv_bfloat16* dv_out = d_qkv_ptr + (Hq + Hkv) * Hs;
            surogate::mem_eff::mqa_reduce_kv_bf16(dk_partial,
                                                  dv_partial,
                                                  dk_out,
                                                  dv_out,
                                                  static_cast<int>(total_tokens),
                                                  Hq,
                                                  Hs,
                                                  /*partial_strideM=*/Hq * Hs,
                                                  /*partial_strideH=*/Hs,
                                                  /*out_strideM=*/HtotQKV * Hs,
                                                  p.stream);
        }
    }
};

struct MemEffAttentionAutoRegister {
    MemEffAttentionAutoRegister() {
        AttentionBackendRegistry::instance().add(std::make_unique<MemEffAttention>());
    }
};
const MemEffAttentionAutoRegister _mem_eff_attention_auto_register;

}  // namespace
}  // namespace dsl
