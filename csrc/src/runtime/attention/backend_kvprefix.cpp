// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Chunked-sequence (KV-prefix) attention backend. Active only when
// AttentionParams::chunk_kv_len > 0: queries are the current chunk's T rows
// out of the interleaved qkv buffer; keys/values are the executor-owned
// contiguous per-layer caches holding the whole prefix including this chunk.
// Forward appends the chunk's (post-rope) K/V into the caches, then runs the
// FA2 varlen kernels cross-shape (cu_seqlens_q != cu_seqlens_k, bottom-right
// causal). Backward writes dQ into the chunk's dqkv, reduces the prefix-wide
// expanded dK/dV into FP32 accumulators, and emits the chunk's now-complete
// rows back into dqkv — exact as long as chunks run last-to-first.

#include <memory>
#include <stdexcept>

#include "runtime/attention/attention_backend.h"
#include "runtime/attention/attention_kernels.h"
#include "runtime/dsl/dsl_run_state.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

constexpr int kFlashMaxHeadDim = 256;

class KvPrefixAttention final : public AttentionBackend {
public:
    const char* name() const override {
        return "flash_kvprefix";
    }

    int priority() const override {
        return attention_priority::kKvPrefix;
    }

    bool gqa_backward_is_rank_divergent() const override {
        return true;  // same FA2 varlen backward kernels as flash_varlen
    }

    bool supports(const AttentionParams& p) const override {
        if (p.chunk_kv_len <= 0) {
            return false;
        }
        // Loud failures beat silent fallbacks here: chunked mode was
        // requested explicitly, so an unsupported combination is a config
        // error, not a reason to pick a dense backend.
        if (p.dtype != ETensorDType::BF16) {
            throw std::runtime_error("chunked sequence training requires BF16 attention");
        }
        if (p.B != 1) {
            throw std::runtime_error("chunked sequence training requires per-device batch size 1");
        }
        if (p.sinks != nullptr) {
            throw std::runtime_error("chunked sequence training does not support attention sinks");
        }
        if (p.Hs <= 0 || p.Hs > kFlashMaxHeadDim) {
            throw std::runtime_error("chunked sequence training: head size > 256 not supported");
        }
        if (!p.chunk_k_cache || !p.chunk_v_cache || !p.chunk_cu_q || !p.chunk_cu_k) {
            throw std::runtime_error("chunked sequence training: KV cache state not initialized");
        }
        if (p.run_state == nullptr || p.temps == nullptr) {
            return false;
        }
        return true;
    }

    void forward(AttentionParams& p) override {
        // Idempotent append: the re-forward before a chunk's backward
        // overwrites the same rows with identical values.
        // The cache pointers arrive window-relative; the append position is
        // global — write via the window-relative offset so the rows land at
        // their global positions.
        append_kv_to_cache(p.chunk_k_cache,
                           p.chunk_v_cache,
                           p.qkv.get<nv_bfloat16>(),
                           p.chunk_pos,
                           p.T,
                           p.Hq,
                           p.Hkv,
                           p.Hs,
                           p.stream);
        attention_forward_flash_kvprefix(p.out.get<nv_bfloat16>(),
                                         p.lse.get<float>(),
                                         p.qkv.get<nv_bfloat16>(),
                                         p.chunk_k_cache,
                                         p.chunk_v_cache,
                                         p.chunk_cu_q,
                                         p.chunk_cu_k,
                                         p.T,
                                         p.chunk_kv_len,
                                         p.Hq,
                                         p.Hkv,
                                         p.Hs,
                                         p.stream,
                                         p.softmax_scale,
                                         p.window_size,
                                         p.chunk_num_segs,
                                         p.chunk_max_q,
                                         p.chunk_max_k);
    }

    void backward(AttentionParams& p) override {
        if (!p.chunk_dk_accum || !p.chunk_dv_accum) {
            throw std::runtime_error("chunked sequence training: dKV accumulators not initialized");
        }
        auto& rs = *p.run_state;
        auto& temps = *p.temps;

        const int HS_rounded = p.Hs <= 128 ? ((p.Hs + 31) / 32) * 32 : ((p.Hs + 63) / 64) * 64;
        const long padded_q = static_cast<long>(p.T) + 128L * std::max(1, p.chunk_num_segs);
        Tensor dq_accum = rs.temp_alloc(
            ETensorDType::FP32, {padded_q * p.Hq * HS_rounded}, "kvprefix_dq_accum");
        Tensor dsoftmax = rs.temp_alloc(ETensorDType::FP32, {static_cast<long>(p.Hq) * padded_q}, "kvprefix_dsoftmax");
        Tensor dk_exp = rs.temp_alloc(
            ETensorDType::BF16, {static_cast<long>(p.chunk_kv_len) * p.Hq * p.Hs}, "kvprefix_dk_exp");
        Tensor dv_exp = rs.temp_alloc(
            ETensorDType::BF16, {static_cast<long>(p.chunk_kv_len) * p.Hq * p.Hs}, "kvprefix_dv_exp");
        temps.push_back(dq_accum);
        temps.push_back(dsoftmax);
        temps.push_back(dk_exp);
        temps.push_back(dv_exp);
        CUDA_CHECK(cudaMemsetAsync(dq_accum.Data, 0, dq_accum.bytes(), p.stream));
        CUDA_CHECK(cudaMemsetAsync(dsoftmax.Data, 0, dsoftmax.bytes(), p.stream));

        attention_backward_flash_kvprefix(p.d_qkv.get<nv_bfloat16>(),
                                          p.chunk_dk_accum,
                                          p.chunk_dv_accum,
                                          p.lse.get<float>(),
                                          p.out.get<nv_bfloat16>(),
                                          p.d_out.get<nv_bfloat16>(),
                                          p.qkv.get<nv_bfloat16>(),
                                          p.chunk_k_cache,
                                          p.chunk_v_cache,
                                          p.chunk_cu_q,
                                          p.chunk_cu_k,
                                          dq_accum.get<float>(),
                                          dsoftmax.get<float>(),
                                          dk_exp.get<nv_bfloat16>(),
                                          dv_exp.get<nv_bfloat16>(),
                                          p.chunk_pos,
                                          p.T,
                                          p.chunk_kv_len,
                                          p.Hq,
                                          p.Hkv,
                                          p.Hs,
                                          p.deterministic_bwd,
                                          p.stream,
                                          p.softmax_scale,
                                          p.window_size,
                                          p.chunk_num_segs,
                                          p.chunk_max_q,
                                          p.chunk_max_k);
    }
};

struct KvPrefixAttentionAutoRegister {
    KvPrefixAttentionAutoRegister() {
        AttentionBackendRegistry::instance().add(std::make_unique<KvPrefixAttention>());
    }
};
const KvPrefixAttentionAutoRegister _kvprefix_attention_auto_register;

}  // namespace
}  // namespace dsl
