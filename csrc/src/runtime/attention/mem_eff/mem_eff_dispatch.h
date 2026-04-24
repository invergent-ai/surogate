#pragma once

// Minimal dispatcher for the ported PyTorch mem-efficient attention
// forward kernel. Scope of this prototype:
//   * BF16 only.
//   * Forward only (backward ships in a follow-up commit).
//   * Packed (varlen) sequences via seqstart_q/k device pointers; dense
//     mode works too by passing ``cu_seqlens == nullptr``.
//   * Causal mask optional. Sliding window supported via window_size.
//
// Exposes a single entry point that takes raw device pointers plus shape
// metadata, picks the best-matching precompiled kernel based on
// head_dim, and launches it. The AttentionBackend wrapper (to be added
// under backend_mem_eff_varlen.cpp) will convert the runtime's
// AttentionParams into this form.

#include <cstdint>
#include <cuda_runtime.h>

namespace surogate {
namespace mem_eff {

struct ForwardArgs {
    // Packed layout: all Q/K/V tensors are in [total_tokens, num_heads, head_dim]
    // layout. For dense (non-varlen) inputs, total_tokens = B*T and the
    // caller is responsible for flattening before calling.
    const void* q_ptr = nullptr;  // bf16
    const void* k_ptr = nullptr;  // bf16
    const void* v_ptr = nullptr;  // bf16
    void* out_ptr = nullptr;      // bf16, same shape as Q
    float* lse_ptr = nullptr;     // [num_heads, total_tokens] fp32; may be null

    // Shape & layout
    int32_t num_heads_q = 0;
    int32_t num_heads_kv = 0;  // GQA: num_heads_q % num_heads_kv == 0
    int32_t head_dim_qk = 0;
    int32_t head_dim_v = 0;
    int32_t total_tokens = 0;  // sum over packed docs; = B*T for dense
    int32_t num_batches = 0;   // B for dense, num_docs for packed

    // Strides (elements, not bytes). For a contiguous [T, Hq, Hs] tensor:
    //   strideH = Hs, strideM = Hq*Hs, strideB = T*Hq*Hs (dense only;
    //   ignored when cu_seqlens != nullptr because packed layout has no
    //   per-batch stride — all docs share one [total_tokens, Hq, Hs]
    //   buffer).
    int32_t q_strideM = 0;
    int32_t q_strideH = 0;
    int64_t q_strideB = 0;
    int32_t k_strideM = 0;
    int32_t k_strideH = 0;
    int64_t k_strideB = 0;
    int32_t v_strideM = 0;
    int32_t v_strideH = 0;
    int64_t v_strideB = 0;
    int32_t o_strideM = 0;

    // Varlen (doc masking). When both non-null, the kernel runs in
    // packed mode: num_batches = number of docs; seqstart_q_ptr[i] /
    // seqstart_q_ptr[i+1] bound doc i's token range in the packed
    // [total_tokens, Hq, Hs] buffer. seqlen_k_ptr == nullptr means
    // self-attention (k range matches q range).
    const int32_t* seqstart_q_ptr = nullptr;
    const int32_t* seqstart_k_ptr = nullptr;
    const int32_t* seqlen_k_ptr = nullptr;

    // Mask
    bool causal = true;
    int32_t window_size = 0;  // >0 enables sliding window

    // Scale. ``<= 0`` defers to the kernel default (1/sqrt(head_dim_qk)).
    float softmax_scale = 0.0f;

    cudaStream_t stream = nullptr;
};

/// Returns true if a compiled kernel variant exists for the given
/// (head_dim_qk, head_dim_v). Used by the AttentionBackend supports() path.
bool forward_supported(int head_dim_qk, int head_dim_v);

/// Launch the forward kernel. Throws std::runtime_error if no kernel
/// variant matches the head-dim combo.
void forward_bf16_sm80(const ForwardArgs& args);

}  // namespace mem_eff
}  // namespace surogate
