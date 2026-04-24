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

// --- Backward -----------------------------------------------------------

struct BackwardArgs {
    // Saved-from-forward inputs
    const void* q_ptr = nullptr;      // bf16
    const void* k_ptr = nullptr;      // bf16
    const void* v_ptr = nullptr;      // bf16
    const void* out_ptr = nullptr;    // bf16 (saved forward output)
    const void* d_out_ptr = nullptr;  // bf16 (incoming gradient)
    const float* lse_ptr = nullptr;   // fp32 (saved forward logsumexp)

    // Outputs (gradients wrt q/k/v)
    void* d_q_ptr = nullptr;  // bf16
    void* d_k_ptr = nullptr;  // bf16
    void* d_v_ptr = nullptr;  // bf16

    // Scratch buffers caller must provide
    float* delta_ptr = nullptr;      // [num_batches, num_heads, num_queries] fp32
    float* workspace_ptr = nullptr;  // fp32; size via workspace_bytes()
    bool zero_workspace = false;     // set true iff workspace_bytes() > 0

    // Shape
    int32_t num_queries = 0;
    int32_t num_keys = 0;
    int32_t num_batches = 0;
    int32_t num_heads = 0;
    int32_t head_dim_qk = 0;
    int32_t head_dim_v = 0;

    // Strides for q/k/v/out/d_out (same layout as forward).
    int32_t q_strideM = 0;
    int32_t q_strideH = 0;
    int64_t q_strideB = 0;
    int32_t k_strideM = 0;
    int32_t k_strideH = 0;
    int64_t k_strideB = 0;
    int32_t v_strideM = 0;
    int32_t v_strideH = 0;
    int64_t v_strideB = 0;
    int64_t o_strideB = 0;
    int64_t o_strideH = 0;
    int64_t gO_strideM = 0;
    int64_t gO_strideH = 0;
    int64_t gO_strideB = 0;
    int64_t lse_strideB = 0;
    int64_t lse_strideH = 0;
    int64_t delta_strideB = 0;
    int64_t delta_strideH = 0;
    // Output-gradient strides. gQ_strideM, gK_strideM, gV_strideM are
    // computed by the kernel as gQKV_strideM_multiplier * num_heads * head_dim
    // (multiplier=3 for packed interleaved, 1 for separate buffers).
    int8_t gQKV_strideM_multiplier = 1;
    int64_t gQ_strideH = 0;
    int64_t gK_strideH = 0;
    int64_t gV_strideH = 0;
    int64_t gQ_strideB = 0;
    int64_t gK_strideB = 0;
    int64_t gV_strideB = 0;

    // Packed-sequence (varlen)
    const int32_t* cu_seqlens_q_ptr = nullptr;
    const int32_t* cu_seqlens_k_ptr = nullptr;

    // Mask
    bool causal = true;
    int32_t window_size = 0;
    float softmax_scale = 0.0f;

    int num_splits_key = 1;

    cudaStream_t stream = nullptr;
};

// Size (in bytes) of the workspace buffer the backward kernel needs.
// Call with the same shape params you'll pass to backward_bf16_sm80;
// compute + allocate + (if non-zero) zero-init the workspace before
// launching.
std::size_t backward_workspace_bytes(const BackwardArgs& args);
bool backward_workspace_needs_zero(const BackwardArgs& args);
void backward_bf16_sm80(const BackwardArgs& args);

// Precompute delta = sum(out * dout, dim=-1) in fp32. The backward
// kernel expects the delta tensor populated (kKernelComputesDelta is
// false in our instantiation).
void compute_delta_bf16(const void* out,
                        const void* dout,
                        float* delta,
                        int num_batches,
                        int num_heads,
                        int num_queries,
                        int head_dim_v,
                        long o_strideB,
                        long o_strideM,
                        long o_strideH,
                        long dO_strideB,
                        long dO_strideM,
                        long dO_strideH,
                        long delta_strideB,
                        long delta_strideH,
                        cudaStream_t stream);

}  // namespace mem_eff
}  // namespace surogate
