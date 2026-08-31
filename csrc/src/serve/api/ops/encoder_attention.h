#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

/**
 * Non-causal grouped-query attention over one whole sequence.
 *
 * An encoder has no KV cache and no decode step: every row attends every column
 * in a single pass, and the window -- when there is one -- is symmetric. For
 * `q_heads` query heads sharing one key/value head, head `h` and query position
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
 * `qkv` is a contiguous BF16 tensor [q_heads*head_dim + 2*head_dim, tokens] --
 * the fused projection, with query heads first, then the key head, then the
 * value head. `out` is contiguous BF16 [q_heads*head_dim, tokens].
 *
 * `workspace` holds the score matrix and must be at least
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
void encoder_attention(const Tensor& qkv, std::int32_t q_heads, std::int32_t head_dim,
                       std::int32_t window, float scale, Tensor& out, void* workspace,
                       std::size_t workspace_bytes, cudaStream_t stream);

/// Scratch bytes `encoder_attention` needs for one sequence of `tokens`.
std::size_t encoder_attention_workspace_bytes(std::int32_t q_heads, std::int32_t tokens);

/// Creates this device's cuBLASLt handle and workspace; call before stream capture.
void encoder_attention_prewarm();

} // namespace sinfer::ops
