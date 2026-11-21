#pragma once
// DSpark drafter ops (ENGINE-INKLING lane e2): Inkling-2-Mini's MTP drafter -- 3 dense block layers over the
// target's features plus the recurrent Markov head (rl/engine-inkling-2026-11-21/lanes/e2/DESIGN-dspark.md).
//
// Semantics: the functions of rl/engine-inkling-2026-11-21/ref/dspark_ref.py (rounding points marked `# R:`
// there), which port vLLM's Inkling2 DSpark (ref/src/vllm/models/inkling2/nvidia/dspark.py, ops/dspark.py,
// speculator.py). Golden vectors: rl/engine-inkling-2026-11-21/vectors/dspark_*/ (dspark_ref.py make_vectors).
// The GEMMs of the drafter are not here: they run on lane e1's invariant GEMM (inkling_gemm_inv).
//
// Layout convention (this engine's): the FIRST dimension is the fastest. "[2304, T]" is T columns of 2304
// contiguous values; a draft block of B lanes is [.., 8, B] (column j + 8 * b). All entry points take raw device
// pointers; BF16 means __nv_bfloat16 storage. Every kernel computes a column's output with one arithmetic,
// independent of the batch (DESIGN.md 11): no atomics, fixed reduction orders.
#include <cuda_runtime.h>

#include "api/ops/sampling.h"

#include <cstddef>
#include <cstdint>

namespace sinfer::ops {

struct DSparkGeometry {
    static constexpr std::int32_t kHidden      = 2304;
    static constexpr std::int32_t kQHeads      = 24;
    static constexpr std::int32_t kKvHeads     = 8;
    static constexpr std::int32_t kHeadDim     = 128;
    static constexpr std::int32_t kQRows       = kQHeads * kHeadDim;  // 3072
    static constexpr std::int32_t kKvRows      = kKvHeads * kHeadDim; // 1024
    static constexpr std::int32_t kQkvRows     = kQRows + 2 * kKvRows; // 5120: [q | k | v]
    static constexpr std::int32_t kIntermediate = 9216;
    static constexpr std::int32_t kLayers      = 3;
    static constexpr std::int32_t kBlock       = 8;   // anchor row + 7 mask rows
    static constexpr std::int32_t kMaxDrafts   = kBlock - 1;
    static constexpr std::int32_t kFeatures    = 4;   // target visits 5, 23, 41 and the final-norm output
    static constexpr std::int32_t kFeatureRows = kFeatures * kHidden; // 9216
    static constexpr std::int32_t kMixRows     = kLayers + 1;         // draft layers 0-2 + the Markov head
    static constexpr std::int32_t kWindow      = 1024;                // local draft layers 0 and 2
    /// The local layers' key range as a window over the block's query position anchor + 7:
    /// (anchor + 7) - key < 1032  <=>  key >= anchor - 1024 (block-anchored window, DESIGN-dspark.md 1).
    static constexpr std::int32_t kBlockWindow = kWindow + kBlock;
    static constexpr std::int32_t kRank        = 256;
    static constexpr std::int32_t kRnnRows     = 3 * kRank;                 // 768: [gate | cand | out]
    static constexpr std::int32_t kRnnStateCols = 2 * kRank;                // [state | w1[prev]]
    static constexpr std::int32_t kRnnContextCols = 2 * kHidden;            // [hidden | f_rnn]
    static constexpr std::int32_t kVocab       = 200058;
    static constexpr std::int32_t kVocabPadded = 201024;
    static constexpr float kEps       = 1e-6F;
    static constexpr float kSoftcap   = 30.0F;
    static constexpr float kRopeTheta = 1.0e6F;
    static constexpr float kScale     = 0.08838834764831845F; // 128^-0.5
    /// Vocabulary rows per CTA of dspark_markov_vocab (the partials' tile).
    static constexpr std::int32_t kVocabTile = 256;
};

/// Tiles of dspark_markov_vocab for a token domain (the partials' first extent).
[[nodiscard]] constexpr std::int32_t dspark_vocab_tiles(std::int32_t domain) {
    return (domain + DSparkGeometry::kVocabTile - 1) / DSparkGeometry::kVocabTile;
}

/// RoPE inverse frequencies as the reference builds them (fp32): 1 / theta^(2i / 128), i < 64.
void dspark_rope_inv_freq(float theta, float* host_out64);

/**
 * Feature mix + RMSNorm (`mix_rmsnorm`): for each column t and mix row r < rows,
 *   out[:, t, r] = bf16((m * rstd) * norms[:, r]),  m = (((x0 a0) + x1 a1) + x2 a2) + x3 a3  (fp32, unfused),
 *   x_f = features[f * 2304 + :, t], a = alpha[r * 4 + f], rstd = 1 / sqrt(sum m^2 / 2304 + eps).
 * features BF16 with `feature_stride` elements per column (>= 9216); alpha device FP32 [rows][4] (row r =
 * softmax(fuse_weight[r])); norms BF16 [2304, rows] contiguous; out BF16 [2304, columns, rows] contiguous.
 */
void dspark_feature_mix(const void* features, std::int64_t feature_stride, std::int32_t columns,
                        const float* alpha, const void* norms, std::int32_t rows, float eps, void* out,
                        cudaStream_t stream);

/**
 * Per-head RMSNorm, then (positions != null) neox RoPE: for head h of column t,
 *   y = bf16((x * rstd) * w), rstd = 1 / sqrt(sum x^2 / 128 + eps);
 *   out = bf16([y1 cos - y2 sin, y2 cos + y1 sin]), angle_i = fp32(pos_t) * inv_freq[i] (fp32, unfused).
 * x BF16 [128, heads, columns] with `x_stride` elements per column; out likewise with `out_stride` (may alias
 * x when the strides are equal); weight BF16 [128]; positions I32 [columns]; inv_freq device FP32 [64].
 */
void dspark_head_norm_rope(const void* x, std::int64_t x_stride, std::int32_t heads, std::int32_t columns,
                           const void* weight, const std::int32_t* positions, const float* inv_freq, float eps,
                           void* out, std::int64_t out_stride, cudaStream_t stream);

/**
 * The pre-norm residual step of a block layer (flashinfer fused_add_rmsnorm as the reference reads it):
 *   s = x + resid (fp32; x null: s = resid), resid_out = bf16(s), h = bf16((s * rstd) * w),
 *   rstd from the UNROUNDED s. All BF16 [2304, columns] contiguous; resid_out may alias resid.
 */
void dspark_add_rms_norm(const void* x, const void* resid, const void* weight, float eps, std::int32_t columns,
                         void* resid_out, void* h, cudaStream_t stream);

/// y[i, t] = bf16(g / (1 + exp(-g)) * u), g = gate_up[i, t], u = gate_up[9216 + i, t]; gate_up BF16
/// [18432, columns] ([gate_proj; up_proj] rows), y BF16 [9216, columns], both contiguous.
void dspark_silu_mul(const void* gate_up, std::int32_t columns, void* y, cudaStream_t stream);

/**
 * The draft block of B lanes: out[:, 0, b] = bf16((e * rstd) * embed_norm), e = embed row anchors[b]
 * (rstd = 1 / sqrt(sum e^2 / 2304 + eps)), out[:, j, b] = mask_embed for j = 1..7. embed BF16 rows of 2304
 * (`embed_rows` rows); out BF16 [2304, 8, B] contiguous.
 */
void dspark_embed_block(const std::int32_t* anchors, std::int32_t batch, const void* embed,
                        std::int32_t embed_rows, const void* embed_norm, const void* mask_embed, float eps,
                        void* out, cudaStream_t stream);

/// positions[j + 8 * b] = anchor_positions[b] + j, j < 8 (the draft block's positions; I32).
void dspark_block_positions(const std::int32_t* anchor_positions, std::int32_t batch, std::int32_t* positions,
                            cudaStream_t stream);

/// positions[j + 8 * b] = anchor_positions[b] + 7, j < 8: the block's query positions for lane c's draft attention
/// (every row of a lane sits at the block's last position, so all see one key set).
void dspark_block_query_positions(const std::int32_t* anchor_positions, std::int32_t batch, std::int32_t* positions,
                                  cudaStream_t stream);

/// One draft-KV plane of the paged pool: [page][kv head 8][64 tokens][128] like the target's planes; FP8 e4m3
/// unit scale (satfinite) or BF16. `block_tables` rows of `table_stride` I32 page ids.
struct DSparkPagedPlane {
    void* k_pages                     = nullptr;
    void* v_pages                     = nullptr;
    const std::int32_t* block_tables  = nullptr;
    std::int64_t table_stride         = 0;
    bool fp8                          = true;
    /// Positions >= limit are never written or read (the sequence capacity: a block near the end of the context).
    std::int32_t limit                = 0x7FFFFFFF;
};

/**
 * Draft-KV write: column n = j + width * b holds K and V of 8 heads x 128 at k + n * stride and v + n * stride
 * (BF16); it goes to position positions[n] of block-table row table_rows[b]. Columns with j >= counts[b] are
 * skipped (counts null: every column). FP8: e4m3(x) straight from the bf16 value, satfinite.
 */
void dspark_kv_write(const void* k, const void* v, std::int64_t stride, const std::int32_t* positions,
                     std::int32_t width, std::int32_t batch, const std::int32_t* counts,
                     const std::int32_t* table_rows, const DSparkPagedPlane& plane, cudaStream_t stream);

/**
 * Draft block attention (DSparkRef.attention; vLLM _rel_attention_kernel DRAFT mode without bias / gate /
 * sink): lane b's 8 query rows sit at position anchor_b + 7 and all attend over the keys [lo, anchor_b + 8) of
 * the plane, lo = window > 0 ? max(0, anchor_b + 8 - window) : 0 (window = kBlockWindow on layers 0 / 2, 0 on
 * the global layer 1); no mask among the 8 rows. The block's own K / V must already be in the plane at
 * [anchor_b, anchor_b + 8). Query head h reads KV head h / 3. Scores fp32, softmax with an online max, P
 * rounded to bf16 for P.V, the sum taken over the unrounded P; out = bf16(acc / sum).
 * q BF16 [128, 24, 8, B] with `q_stride` elements per column; out BF16 [128, 24, 8, B] contiguous.
 * NAIVE (correctness): one CTA per (kv head, lane); lane c's draft mode replaces it (BOARD 13:35 / c2 13:4x).
 */
void dspark_block_attention(const void* q, std::int64_t q_stride, const std::int32_t* anchor_positions,
                            const std::int32_t* table_rows, std::int32_t batch, std::int32_t window, float scale,
                            const DSparkPagedPlane& plane, void* out, cudaStream_t stream);

/**
 * f_rnn source: dst[:, lanes[b]] = src[:, counts[b] - 1, b] for lanes with counts[b] >= 1 (the mix row 3 of the
 * last committed column). src BF16 [2304, width, B] contiguous; dst BF16 [2304, lane_capacity]; lanes /
 * counts I32 [B] (counts null: width).
 */
void dspark_store_last_column(const void* src, std::int32_t width, std::int32_t batch,
                              const std::int32_t* counts, const std::int32_t* lanes, void* dst,
                              cudaStream_t stream);

/**
 * The Markov head's context input: x[:, j + k * b] = [hidden[:, 1 + j, b] ; f_rnn[:, lanes[b]]] for j < k.
 * hidden BF16 [2304, 8, B] (the block's final hidden), f_rnn BF16 [2304, lane_capacity] (lanes null: column b),
 * x BF16 [4608, k * B]. Then ctx = W[:, 512:5120] . x on inkling_gemm_inv (FP32 [768, k * B]); the bias is
 * added by dspark_markov_gate.
 */
void dspark_markov_pack(const void* hidden, const void* f_rnn, const std::int32_t* lanes, std::int32_t k,
                        std::int32_t batch, void* x, cudaStream_t stream);

/**
 * One Markov gate step j for B lanes (`_markov_gate_kernel`):
 *   raw = (ctx[:, j + k * b] + bias) + W_sp . [state_b ; w1[prev_b]]   (fp32; W_sp = rnn_proj[:, 0:512])
 *   g, c, o = bf16(raw[0:256]), bf16(raw[256:512]), bf16(raw[512:768])
 *   g = bf16(1 / (1 + exp(-g))); c = bf16(tanh(c)); state = bf16(bf16(g state) + bf16(bf16(1 - g) c));
 *   latent = bf16(tanh(o)).
 * ctx FP32 [768, k * B]; bias BF16 [768]; w_sp BF16 [768, 512] contiguous; w1 BF16 [Vpad, 256];
 * prev = prev_base[b * prev_stride] (step 0: the anchors, stride 1; step j: the drafts at j - 1, stride = their
 * first extent); state_in BF16 [256, B] (null: zeros, step 0) -> state_out BF16 [256, B] (a different buffer: 32 CTAs
 * per lane read the whole old state); latent BF16 [256, B]. One warp per gate index: rows i, 256 + i, 512 + i.
 */
void dspark_markov_gate(const float* ctx, std::int32_t j, std::int32_t k, std::int32_t batch, const void* bias,
                        const void* w_sp, const void* w1, const std::int32_t* prev_base, std::int32_t prev_stride,
                        const void* state_in, void* state_out, void* latent, cudaStream_t stream);

/**
 * One vocabulary pass of step j: for v < domain,
 *   bias_v = bf16(fold_w[v] . latent_b + fold_b[v]) (fp32), s = bf16(base[v, row, b] + bias_v),
 *   logit = bf16(30 tanh(s / 30));
 * per tile of kVocabTile rows the largest logit of each lane and its LOWEST id -> part_val / part_id
 * [tiles, B]. base BF16 [base_rows, 8, B] (row = 1 + j: the mask row); fold_w BF16 [>= domain, 256]; fold_b
 * BF16 [>= domain]; latent BF16 [256, B]. `ids` non-null: a pruned draft vocabulary -- base / fold_w / fold_b
 * rows are its tokens in ascending id order (gathered at load) and part_id holds ids[v] (ties keep the lowest
 * token id). Debug outputs (null in serving): bias_out / logits_out BF16 [domain, B].
 */
/// FP8 fold (fold_codes / fold_scales non-null: per-row e4m3 [domain rows, 256] + fp32 row scales; fold_w ignored):
/// bias_v = bf16(scale_v * (codes[v] . latent_b) + fold_b[v]) -- the M1 lever `fp8fold`, no acceptance loss.
/// COUPLED SAMPLED DRAFTS (agent-EN-e2c; `sampling` non-null): a lane whose config is a coupled untruncated sampled row
/// (SamplingConfig::speculative_coupled, temperature > 0, top_k 0, top_p 1) ranks by z = logit / T + Gumbel keyed by
/// sample_key(cfg, anchor_positions[b] + draft_index, kSamplePurposeDecode) -- the noise the coupled verify draws that
/// token with -- instead of the logit: a sampled draft d ~ softmax(logits / T) that equals the target's token whenever
/// the two Gumbel-max draws agree. Any draft is exact under coupled verification; only acceptance changes. Other lanes
/// (greedy, top-k / top-p, one-hot verification) keep the greedy draft.
void dspark_markov_vocab(const void* base, std::int32_t base_rows, std::int32_t row, std::int32_t batch,
                         const void* fold_w, const void* fold_b, const void* latent, std::int32_t domain,
                         const std::int32_t* ids, float* part_val, std::int32_t* part_id, void* bias_out,
                         void* logits_out, cudaStream_t stream, const void* fold_codes = nullptr,
                         const float* fold_scales = nullptr, const SamplingConfig* sampling = nullptr,
                         const std::int32_t* anchor_positions = nullptr, std::int32_t draft_index = 0);

/// drafts[j + stride * b] = the id of the largest partial (lowest id on ties) over the `tiles` partials of lane b
/// (stride = the drafts tensor's first extent, >= k).
void dspark_markov_pick(const float* part_val, const std::int32_t* part_id, std::int32_t tiles, std::int32_t batch,
                        std::int32_t j, std::int32_t stride, std::int32_t* drafts, cudaStream_t stream);

/**
 * The load-time fold (Inkling2MarkovHead.fold): fold_w[v, r] = bf16(sum_c unembed[v, c] hidden_out[c, r]),
 * fold_b[v] = bf16(sum_c unembed[v, c] hidden_out_bias[c]); fp64 products and sums in c order, one rounding.
 * unembed BF16 [rows, 2304]; hidden_out BF16 [2304, 256] (the Linear(256 -> 2304) weight); bias BF16 [2304];
 * fold_w BF16 [rows, 256]; fold_b BF16 [rows]. `ids` non-null: unembed row ids[v] (pruned vocabulary).
 */
void dspark_fold(const void* unembed, std::int32_t rows, const std::int32_t* ids, const void* hidden_out,
                 const void* hidden_out_bias, void* fold_w, void* fold_b, cudaStream_t stream);

/**
 * Load-time per-row e4m3 quantisation of a drafter linear (DSparkRef maybe_fp8, M1 lever `fp8w`):
 * scale_r = max(amax_r / 448, 1e-12) (fp32), codes[r, :] = e4m3(w[r, :] / scale_r) satfinite. w BF16 [n, k] row-major
 * (device memory or a mapped host bank), codes U8 [n, k], scales FP32 [n].
 */
void dspark_quantize_rows_e4m3(const void* w, std::int32_t n, std::int32_t k, void* codes, float* scales,
                               cudaStream_t stream);

/**
 * Width-invariant W8A16 GEMM for the drafter's linears: out[r, t] = bf16(scale_r * sum_k e4m3(codes[r, k]) x[k, t])
 * for t < tokens. ONE chain per row at every width: lane l of the row's warp sums the row's 16-code chunks l, l + 32,
 * ... in order (fp32 fma), a fixed shuffle tree adds the lanes, the row scale multiplies the sum, one bf16 rounding.
 * codes U8 [n, k], scales FP32 [n], x BF16 [k, tokens] (k contiguous), out BF16 with ldc elements per column;
 * k a multiple of 16, pointers 16-byte aligned.
 */
void dspark_gemm_w8(const void* codes, const float* scales, std::int32_t n, std::int32_t k, const void* x,
                    std::int32_t tokens, void* out, std::int32_t ldc, cudaStream_t stream);

} // namespace sinfer::ops
