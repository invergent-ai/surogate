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
        // Backward with MQA (Hkv!=Hq) needs a separate grad K/V buffer +
        // reduce, not yet wired. MHA backward works via the same
        // interleaved d_qkv layout as forward.
        const bool is_backward = (p.d_qkv.Data != nullptr || p.d_out.Data != nullptr);
        if (is_backward && p.Hkv != p.Hq) {
            return reject("MQA backward not yet implemented (need KV-grad reduce)");
        }
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

        surogate::mem_eff::ForwardArgs args;
        args.q_ptr = q_ptr;
        args.k_ptr = k_ptr;
        args.v_ptr = v_ptr;
        args.out_ptr = out_ptr;
        args.lse_ptr = p.lse.Data ? p.lse.get<float>() : nullptr;
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

        // LSE is [num_docs, Hq, max_seqlen] fp32 (saved by forward).
        float* lse_ptr = p.lse.Data ? p.lse.get<float>() : nullptr;
        if (lse_ptr == nullptr) {
            throw std::runtime_error("mem_eff backend: backward requires saved LSE from forward");
        }
        const int64_t lse_strideB = static_cast<int64_t>(Hq) * max_seqlen;
        const int64_t lse_strideH = max_seqlen;

        // Delta = sum(out * d_out, dim=-1). [num_batches, num_heads, num_queries].
        Tensor delta =
            rs.temp_alloc(ETensorDType::FP32,
                          {static_cast<long>(num_batches), static_cast<long>(Hq), static_cast<long>(max_seqlen)},
                          "mem_eff_bwd_delta");
        temps.push_back(delta);

        surogate::mem_eff::compute_delta_bf16(out_ptr,
                                              d_out_ptr,
                                              delta.get<float>(),
                                              num_batches,
                                              Hq,
                                              max_seqlen,
                                              Hs,
                                              o_strideB,
                                              o_strideM,
                                              o_strideH,
                                              o_strideB,
                                              o_strideM,
                                              o_strideH,
                                              Hq * max_seqlen,
                                              max_seqlen,
                                              p.stream);

        // Build BackwardArgs and probe workspace size via the kernel's
        // own helper. Allocate workspace, zero if required, then launch.
        surogate::mem_eff::BackwardArgs args;
        args.q_ptr = qkv_ptr;
        args.k_ptr = qkv_ptr + Hq * Hs;
        args.v_ptr = qkv_ptr + (Hq + Hkv) * Hs;
        args.out_ptr = out_ptr;
        args.d_out_ptr = d_out_ptr;
        args.lse_ptr = lse_ptr;
        args.d_q_ptr = d_qkv_ptr;
        args.d_k_ptr = d_qkv_ptr + Hq * Hs;
        args.d_v_ptr = d_qkv_ptr + (Hq + Hkv) * Hs;
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
        args.k_strideH = q_strideH;
        args.k_strideB = q_strideB;
        args.v_strideM = q_strideM;
        args.v_strideH = q_strideH;
        args.v_strideB = q_strideB;

        args.o_strideB = o_strideB;
        args.o_strideH = o_strideH;
        args.gO_strideB = o_strideB;
        args.gO_strideM = o_strideM;
        args.gO_strideH = o_strideH;

        args.lse_strideB = lse_strideB;
        args.lse_strideH = lse_strideH;
        args.delta_strideB = static_cast<int64_t>(Hq) * max_seqlen;
        args.delta_strideH = max_seqlen;

        args.gQKV_strideM_multiplier = 3;  // interleaved Q/K/V: gQ_strideM = 3*H*Hs (MHA: HtotQKV=3*Hq)
        args.gQ_strideH = q_strideH;
        args.gK_strideH = q_strideH;
        args.gV_strideH = q_strideH;
        args.gQ_strideB = q_strideB;
        args.gK_strideB = q_strideB;
        args.gV_strideB = q_strideB;

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
