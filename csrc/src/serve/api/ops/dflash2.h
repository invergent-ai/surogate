#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops {

/**
 * Op: dflash_conv, DFlash 2's two-tap dynamic convolution over a draft block's rows.
 *
 * input and out are BF16 [H, W*B]: B lanes of W block rows each, a lane's rows adjacent and in
 * block order. dynamic is BF16 [4*G, W*B], one row's dynamic weights laid out
 * [stage][tap][group]; base is BF16 [H, 2, 2] ([stage][tap][channel] in memory). Channel c
 * belongs to group c / (H/G). For block row t of a lane, stage s and channel c:
 *
 *   w_tap   = base[s][tap][c] + dynamic[t][s][tap][c / (H/G)]          (fp32)
 *   out[t]  = input[t] * w_0 + (t > 0 ? input[t-1] * w_1 : 0) (+ residual[t])
 *
 * computed in fp32 and rounded to BF16 once. A lane's first row has no neighbour: the
 * convolution never reaches across lanes or into the context. Both taps take the weights of
 * the row being written. residual, when given, is BF16 [H, W*B] and may be `out` itself (the
 * stage that adds a sublayer's output to the residual stream in place); input must not overlap
 * out. H/G must be a multiple of 8.
 */
void dflash_conv(const Tensor& input, const Tensor& dynamic, const Tensor& base,
                 std::int32_t stage, std::int32_t width, const Tensor* residual, Tensor& out,
                 cudaStream_t stream);

/**
 * Op: dflash_select_top_k, the candidates of DFlash 2's selector.
 *
 * logits is BF16 [R, C]; for each column, ids (I32 [K, C]) and values (F32 [K, C]) receive the
 * K largest logits among rows [0, domain), ordered by value descending then row ascending. NaN
 * counts as -inf. A column with fewer than K rows pads with row 0 at -inf. 1 <= K <= 32.
 */
void dflash_select_top_k(const Tensor& logits, std::int32_t domain, Tensor& ids, Tensor& values,
                         cudaStream_t stream);

/**
 * Op: dflash_select_walk, DFlash 2's greedy walk through the selector's candidates.
 *
 * For each lane b of B and draft position p of P (column b*P + p of the candidate tensors):
 *
 *   edge[j]  = sum_d predecessor[prev][d] * hidden[d][b*P+p] * successor[ids[j][b*P+p]][d]
 *   draft    = ids[argmax_j (values[j][b*P+p] + edge[j])][b*P+p]   (first maximum wins)
 *
 * where prev is anchors[b] at p = 0 and the draft just chosen after that. ids (I32 [K, P*B])
 * hold token ids in [0, V); values are F32 [K, P*B]; hidden is BF16 [D, P*B]; the codebooks
 * are BF16 [D, V] (a token's D values contiguous). drafts is I32 [P*B]. Sums are fp32.
 * D is a multiple of 32 up to 1024; K is 1..32.
 */
void dflash_select_walk(const Tensor& anchors, const Tensor& ids, const Tensor& values,
                        const Tensor& hidden, const Tensor& predecessor, const Tensor& successor,
                        std::int32_t positions, Tensor& drafts, cudaStream_t stream);

} // namespace sinfer::ops
