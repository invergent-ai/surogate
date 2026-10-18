// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Hashed n-gram memory ("transplant" arm, rl/ngram-memory-2026-10-18/DESIGN.md Appendix A3): the
// trainer side of the engine's single-stream `ngram_ple` layer prologue
// (csrc/src/serve/ops/ngram_ple/ngram_ple.cu). The table is frozen; the trained parameters are the
// key/value projections W_k, W_v [C, E] and the two gate gains g_k, g_q [C]. The dilated convolution
// is frozen at zero, which makes its branch silu(0) = 0, so it does not appear here.
//
// Per token t with residual r (bf16, the residual entering the memory's layer):
//   rows  : h-th hashed n-gram row of t (bit-exact with the engine's rows_kernel)
//   e     = concat_h IQ4_NL-decode(table[row]) rounded to bf16                       [E]
//   k, v  = W_k e, W_v e                                                              [C]
//   z     = < k * rsqrt(mean(k^2)+eps) * g_k , r * rsqrt(mean(r^2)+eps) * g_q > / sqrt(C)
//   g     = sigmoid(sign(z) * sqrt(max(|z|, 1e-6)))
//   u     = g * v   (rounded to the working dtype; 0 at the configured zero tokens)
// and the layer's residual becomes r + u. Contract: rl/ngram-transplant-2026-10-18/INTERFACE.md.

#ifndef SUROGATE_SRC_KERNELS_NGRAM_MEMORY_H
#define SUROGATE_SRC_KERNELS_NGRAM_MEMORY_H

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ngram_memory {

inline constexpr int kMaxHeads = 16;
inline constexpr int kMaxNgram = 3;
inline constexpr int kMaxZeroTokens = 4;
/// Saved per token by the gate forward: inv_k, inv_q, z, g.
inline constexpr int kTokenStats = 4;
/// Running statistics (FP32, accumulated by the gate forward when given):
/// [0] tokens, [1] sum g, [2] sum g^2, [3] #(g > 0.9), [4] #(g < 0.1), [5] sum |u|^2,
/// [6] sum |r|^2, [7] zeroed tokens.
inline constexpr int kGlobalStats = 8;

/// The engine's NgramPleHash (csrc/src/serve/api/ops/ngram_ple.h), same field meaning.
struct Hash {
    std::uint64_t multipliers[kMaxNgram] = {};
    std::int32_t head_offsets[kMaxHeads] = {};
    std::int32_t head_vocab_sizes[kMaxHeads] = {};
    std::int32_t ngram = 0;  ///< 2 or 3
    std::int32_t heads = 0;  ///< (ngram - 1) * heads_per_ngram
    std::int32_t eos_token = 0;
};

/// Token ids whose memory output is forced to zero (the engine's image/video pad rule).
struct ZeroTokens {
    std::int32_t ids[kMaxZeroTokens] = {-1, -1, -1, -1};
    std::int32_t count = 0;
};

/// rows[t * heads + h] for the B*T tokens of one (chunk of a) micro-batch.
/// `ids`, `positions` [B, T] (positions: plane 0 of the model's position ids; a conversation starts
/// where the position is 0, and predecessor k of a token is inside its conversation iff its position
/// is >= k). `carry` [B, 2] holds the two tokens before the chunk (oldest first; -1 = none) or is null.
/// A missing predecessor or an EOS predecessor cuts the context (it and every older one read as EOS).
void rows(int* rows, const int* ids, const int* positions, const int* carry, int B, int T, const Hash& hash,
          cudaStream_t stream);

/// emb[t, h*head_dim + i] = bf16_rn(decode(table[rows[t, h]])[i]) for IQ4_NL rows of `row_bytes`
/// (= head_dim / 32 * 18) bytes at a device-accessible `table` (device memory or mapped pinned host).
void gather_iq4nl(nv_bfloat16* emb, const int* rows, const std::uint8_t* table, int row_bytes, int head_dim,
                  int heads, int N, cudaStream_t stream);

/// out = bf16(float(a) + float(b)), element-wise (the residual the memory reads).
void residual_add(nv_bfloat16* out, const nv_bfloat16* a, const nv_bfloat16* b, long n, cudaStream_t stream);

/// Gate forward (T = float or nv_bfloat16): u [N, C], token stats [N, kTokenStats]; `global_stats`
/// (kGlobalStats floats) is accumulated when not null.
template <typename T>
void gate_forward(T* u, float* stats, float* global_stats, const T* r, const T* k, const T* v, const float* gk,
                  const float* gq, const int* ids, ZeroTokens zero, int N, int C, float eps, cudaStream_t stream);

/// Gate backward. `d_res` [N, C] holds dL/d(r + u) on entry and dL/dr on exit (the gate's query path
/// adds to it); dk, dv [N, C] receive dL/dk and dL/dv, w [N] the per-token weight of the gain gradients.
/// `abs_max` (may be null) is raised to the largest |d_res| written.
template <typename T>
void gate_backward(T* dk, T* dv, float* w, T* d_res, float* abs_max, const T* r, const T* k, const T* v,
                   const float* stats, const float* gk, const float* gq, const int* ids, ZeroTokens zero, int N,
                   int C, cudaStream_t stream);

/// Number of token partitions of the gain-gradient reduction (partials hold [parts, C] floats).
int gain_grad_parts(int N);

/// dgk[c] += gq[c] * S_c and dgq[c] += gk[c] * S_c with S_c = sum_t w_t r_tc k_tc (deterministic
/// two-stage reduction). G = float or nv_bfloat16 (the gradient dtype).
template <typename T, typename G>
void gain_grads(G* dgk, G* dgq, float* partials, const float* w, const T* r, const T* k, const float* gk,
                const float* gq, int N, int C, cudaStream_t stream);

/// The trainer's fused residual + RMSNorm with a third addend: residual = bf16(a + b + u) summed in
/// fp32, normed = bf16(rmsnorm(sum) * weight), rrms = 1/rms. Where u is 0 the arithmetic is the plain
/// fused kernel's (kernels/rmsnorm.cu) bit for bit, so a zero memory leaves every output unchanged.
void residual3_rmsnorm_forward(nv_bfloat16* residual, nv_bfloat16* normed, float* rrms, const nv_bfloat16* a,
                               const nv_bfloat16* b, const nv_bfloat16* u, const nv_bfloat16* weight, float eps,
                               int N, int C, cudaStream_t stream);

/// dst = float(src) for a BF16 (bf16 = true) or FP32 source.
void to_fp32(float* dst, const void* src, bool bf16, long n, cudaStream_t stream);

}  // namespace ngram_memory

#endif  // SUROGATE_SRC_KERNELS_NGRAM_MEMORY_H
