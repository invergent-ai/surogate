#pragma once

// Minimal dispatcher for the ported PyTorch mem-efficient attention
// forward kernel. BF16 only, SM80 ABI (Ampere/Hopper/Blackwell).
//
// Layout: [B, M, num_heads, head_dim] for dense, or
//         [total_tokens, num_heads, head_dim] + cu_seqlens for packed.
// Strides are explicit so callers can use transposed or non-contiguous
// views — the kernel reads via strides, not via shape math.

#include <cstdint>
#include <cuda_runtime.h>

namespace surogate {
namespace mem_eff {

struct ForwardArgs {
    const void* q_ptr = nullptr;        // bf16
    const void* k_ptr = nullptr;        // bf16
    const void* v_ptr = nullptr;        // bf16
    void* out_ptr = nullptr;            // bf16
    float* lse_ptr = nullptr;           // fp32, optional
    float* output_accum_ptr = nullptr;  // fp32, REQUIRED for GMEM variant

    // Shape. `num_queries`/`num_keys` are max lengths (dense: seq_len;
    // packed: max over docs). `num_batches` = B for dense, num_docs for
    // packed.
    int32_t num_queries = 0;
    int32_t num_keys = 0;
    int32_t num_batches = 0;
    int32_t num_heads = 0;
    int32_t head_dim_qk = 0;
    int32_t head_dim_v = 0;

    // Strides (elements, not bytes). For PyTorch's [B, M, H, Hs] layout:
    //   strideB = M*H*Hs, strideM = H*Hs, strideH = Hs.
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

    // Packed-sequence (varlen) support. When both seqstart pointers
    // are non-null, the kernel treats q_ptr/k_ptr/v_ptr as
    // [total_tokens, H, Hs] and dispatches per-doc internally.
    const int32_t* seqstart_q_ptr = nullptr;
    const int32_t* seqstart_k_ptr = nullptr;
    const int32_t* seqlen_k_ptr = nullptr;  // optional: actual k-len per doc

    // Mask
    bool causal = true;
    int32_t window_size = 0;

    // Softmax scale. 0 or negative → kernel default.
    float softmax_scale = 0.0f;

    cudaStream_t stream = nullptr;
};

bool forward_supported(int head_dim_qk, int head_dim_v);
void forward_bf16_sm80(const ForwardArgs& args);

}  // namespace mem_eff
}  // namespace surogate
