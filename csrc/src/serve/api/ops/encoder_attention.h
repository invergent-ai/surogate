#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

/**
 * Causal or bidirectional grouped-query attention over one sequence.
 *
 * An encoder has no KV cache and no decode step: every row attends every column
 * in a single pass, and the window -- when there is one -- is symmetric. For
 * `q_heads` query heads sharing `kv_heads` key/value heads, head `h` and query position
 * `i`:
 *
 *   s[i, j] = scale * sum over d of q[h, d, i] * k[d, j]        for admitted j
 *   p[i, .] = softmax over admitted j of s[i, .]
 *   out[h, d, i] = sum over admitted j of p[i, j] * v[d, j]
 *
 * `window` decides which j are admitted. `0` admits every j -- a global layer.
 * A positive W admits `abs(i - j) < W`, symmetric about the diagonal, which is
 * what a bidirectional model's local layer means and is **not** the causal
 * `i - j < W`. Note that transformers and llama.cpp disagree here: the former
 * uses W (`modeling_gemma3.py:491`), the latter W/2
 * (`LLAMA_SWA_TYPE_SYMMETRIC`). This op implements W, matching the reference
 * that produced the published checkpoint's own numbers.
 *
 * `q` is a contiguous BF16 tensor [q_heads*head_dim, tokens]; `k` and `v` are
 * contiguous BF16 [kv_heads*head_dim, tokens], shared within each query group. They are
 * taken separately rather than as one fused projection because everything
 * upstream already holds them that way: per-head QK norm and rope both need a
 * contiguous operand, which a fused [q|k|v, tokens] matrix does not give.
 * `out` is contiguous BF16 [q_heads*head_dim, tokens].
 *
 * With `causal=true`, keys after the query position are excluded, including in windowed layers.
 * `workspace` holds a query tile's scores and must be at least
 * `encoder_attention_workspace_bytes(q_heads, tokens)`. It is scratch: its
 * contents before and after the call mean nothing.
 *
 * Scores are accumulated and softmaxed in FP32; the probabilities are rounded to
 * BF16 for the second product. The oracle evaluates the expressions above in
 * FP64 from the represented inputs. The BF16 output is promoted and compared
 * directly with that result; output storage rounding belongs to the Op's
 * numerical criterion, not the oracle. Private kernel arithmetic is
 * implementation-defined. The Op writes all of out, writes workspace, and keeps
 * no persistent state.
 */
void encoder_attention(const Tensor& q, const Tensor& k, const Tensor& v, std::int32_t window,
                       float scale, Tensor& out, void* workspace, std::size_t workspace_bytes,
                       cudaStream_t stream, std::int32_t kv_heads = 1, bool causal = false);

/**
 * encoder_attention over a batch of sequences, all of them in one launch.
 *
 * The batch lies end to end: sequence `s` owns columns [offset_s, offset_s + length_s) of `q`,
 * `k`, `v` and `out`, and attends only within them, by the contract above with the sequence's own
 * positions. `segments` is a contiguous device I32 [batch, 2]: the `batch` offsets, then the
 * `batch` lengths, each length positive and the sequences disjoint and inside the tensors.
 * `longest` is the largest length; it sizes the grid. Columns no sequence owns are not written.
 *
 * A FlashAttention-2 kernel on the tensor cores: scores in FP32 with an online softmax, the
 * probabilities rounded to BF16 against each key tile's maximum, no workspace. The numerical
 * criterion is the one above.
 *
 * Returns false, launching nothing, outside its domain -- head dims 128 and 256, at most 65,535
 * sequences -- and the caller runs encoder_attention per sequence.
 * SUROGATE_SERVE_ENCODER_FLASH_ATTENTION=0 declines every call.
 */
[[nodiscard]] bool encoder_attention_batch(const Tensor& q, const Tensor& k, const Tensor& v,
                                           const Tensor& segments, std::int32_t longest,
                                           std::int32_t window, float scale, Tensor& out,
                                           cudaStream_t stream, std::int32_t kv_heads = 1,
                                           bool causal = false);

inline constexpr std::int32_t kEncoderAttentionQueryTile = 256;

/// Scratch bytes for one query tile; memory grows linearly in the sequence length.
std::size_t encoder_attention_workspace_bytes(std::int32_t q_heads, std::int32_t tokens);

/// Creates this device's cuBLASLt handle and workspace; call before stream capture.
void encoder_attention_prewarm();

} // namespace sinfer::ops
