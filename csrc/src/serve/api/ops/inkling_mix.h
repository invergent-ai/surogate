#pragma once

// Inkling-2-Mini hidden-stream ops: the block attention-residual mix, the loop
// message controller's two norms, the hidden-stream short convolution with its per-visit state, the
// embedding with its norm, and the softcapped fp32 LM head.
//
// Semantics: the python functions of the same names in the Python reference inkling_ref.py (not in this repository),
// which follow vLLM's Inkling2 sm_120 path (vllm/models/inkling2/nvidia/ops/{attn_res,loop,sconv,
// norm}.py). Rounding points: the target design notes, section 3. Every rsqrt is the non-FTZ
// `rsqrt.approx.f32` of `ops/norm.py`; tanh / exp are the accurate libm functions; products and sums the
// reference does not fuse are not fused.
//
// Layout convention (this engine's): the FIRST dimension is the fastest. A "plane" is [2304, T]: column t
// (one token's 2304 values, contiguous) starts `column_stride` elements after column t - 1, so one slot of
// the target's AttnRes stack (7 slots per column) and a packed [2304, T] buffer are both planes.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace sinfer::ops {

struct InklingMixGeometry {
    static constexpr std::int32_t kHidden      = 2304;
    static constexpr std::int32_t kConvTaps    = 4; ///< kernel width
    static constexpr std::int32_t kConvHistory = 3; ///< inputs kept per stream
    /// The visit's conv state channels: [K 1024 | V 1024 | attn-out 2304 | mlp-out 2304].
    static constexpr std::int32_t kStreamK      = 0;
    static constexpr std::int32_t kStreamV      = 1024;
    static constexpr std::int32_t kStreamAttn   = 2048;
    static constexpr std::int32_t kStreamMlp    = 4352;
    static constexpr std::int32_t kConvChannels = 6656;
    /// Completed blocks an AttnRes mix may read (the stack's sources before `last`).
    static constexpr std::int32_t kMaxStack = 5;
    static constexpr float kEps = 1e-6F;
};

/// A [2304, T] plane (see the header comment). BF16 or FP32. `data` may be written by the ops that
/// produce a plane; the ones that only read it take it const.
struct InklingPlane {
    void* data                 = nullptr;
    DType dtype                = DType::BF16;
    std::int64_t column_stride = InklingMixGeometry::kHidden; ///< elements
};

/// The sources of one AttnRes mix: `stack_count` BF16 planes (oldest block first), then `last`
/// (BF16 or FP32) plus, when `addend.data` is set, the BF16 `addend` (summed in fp32).
struct InklingAttnResSources {
    InklingPlane stack[InklingMixGeometry::kMaxStack];
    std::int32_t stack_count = 0;
    InklingPlane last;
    InklingPlane addend;
};

/**
 * Op inkling_attn_res (ref `inkling_attn_res`, vLLM `_attn_res_kernel` / `_attn_res_row_kernel`).
 *
 * For column t and query q (FP32, pre-scaled by 2304^-0.5): sources x_n in fp32;
 *   rstd_n = rsqrt(fma(sum x_n^2, 1/2304, 1e-6)), logit_n = (q . x_n) * rstd_n,
 *   p_n = exp(logit_n - max) , inv = 1 / sum p, mix = (sum_n fma(p_n * rstd_n, x_n)) * inv;
 * with `norm_weight`: out = bf16((mix * rsqrt(div_rn(sum mix^2, 2304) + norm_eps)) * w_q);
 * without: out = mix (FP32).
 * `queries` FP32 [2304, Q] contiguous (Q in 1..4); `norm_weight` BF16 [2304, Q] contiguous or null;
 * `out` contiguous [2304, columns, Q], BF16 with a norm, FP32 without. Graph-capturable.
 */
void inkling_attn_res(const InklingAttnResSources& sources, std::int32_t columns,
                      const Tensor& queries, const Tensor* norm_weight, float norm_eps, Tensor& out,
                      cudaStream_t stream);

/**
 * Latency-cut kernels: at decode / verify widths inkling_attn_res, inkling_blend_inputs, inkling_controller_tail and
 * inkling_short_conv run latency-cut kernels that compute the same bits as the reference kernels (invariance test
 * I8). They read their WEIGHTS (queries, norm weights, conv taps, gain / forget) before the PDL wait, so those must
 * not be written inside a step (they are written at load only). `on` = the reference kernels everywhere (tests and
 * A/B; also SUROGATE_SERVE_INKLING_MIX_REFERENCE=1 at start). Takes effect at the next launch / graph capture.
 */
void inkling_mix_use_reference_kernels(bool on) noexcept;

/**
 * Op inkling_embed (ref `inkling_embed`): out column t = bf16(rmsnorm_w(table row ids[t])), the
 * `_rms_norm_kernel` arithmetic: rstd = rsqrt(div_rn(sum x^2, 2304) + eps), y = (x * rstd) * w.
 * `ids` I32 [T] (each in [0, table_rows)), `table` BF16 rows of 2304, `norm_weight` BF16 [2304],
 * `out` a BF16 plane.
 */
void inkling_embed(const Tensor& ids, const void* table, std::int32_t table_rows,
                   const Tensor& norm_weight, float eps, const InklingPlane& out, cudaStream_t stream);

/**
 * vLLM `_rms_norm_kernel` with Q weight rows (the layer-0 attention input): out[:, t, q] =
 * bf16((x_t * rstd_t) * w_q). `x` a BF16 plane, `weight` BF16 [2304, Q] contiguous, `out` contiguous
 * BF16 [2304, columns, Q].
 */
void inkling_rms_norm(const InklingPlane& x, std::int32_t columns, const Tensor& weight, float eps,
                      Tensor& out, cudaStream_t stream);

/**
 * Loop controller, first half (vLLM `pair_rms_norm`): out[:, t, s] = [bf16(dest * rd) ; bf16(msg * rm)],
 * rd = rsqrt(div_rn(sum dest^2, 2304) + 1e-6) (weightless), likewise rm. `dest` FP32 [2304, T, S] or
 * [2304, T, 1] (broadcast over the S streams), `message` FP32 [2304, T, S], `out` BF16 [4608, T, S], all
 * contiguous.
 */
void inkling_pair_rms_norm(const Tensor& dest, const Tensor& message, Tensor& out,
                           cudaStream_t stream);

/**
 * A looped visit's controller inputs in ONE launch: inkling_attn_res(dest_sources, dest_queries) -> `dest` FP32
 * [2304, T, DQ], inkling_attn_res(bank, message_queries) -> `message` FP32 [2304, T, S], and
 * inkling_pair_rms_norm(dest, message) -> `pair` BF16 [4608, T, S], with the same device code in the same order:
 * bit-identical to the three separate calls. DQ is 1 (dest broadcast over the streams) or S; S in 1..3; `bank` is
 * one stack source + last, no addend.
 */
void inkling_blend_inputs(const InklingAttnResSources& dest_sources, const Tensor& dest_queries,
                          const InklingAttnResSources& bank, const Tensor& message_queries, std::int32_t columns,
                          Tensor& dest, Tensor& message, Tensor& pair, cudaStream_t stream);

/**
 * Loop controller, second half (vLLM `blend_rms_norm`, unfused):
 *   x = (1 - tanh(forget)) * dest + tanh(gain) * msg,  out = bf16((x * rsqrt(div_rn(sum x^2, 2304) +
 *   eps)) * w_s).
 * `gain_forget` FP32 [4608, T, S] (gain rows first), `norm_weight` BF16 [2304, S] or [2304, 1]
 * (broadcast), `dest` / `message` as inkling_pair_rms_norm, `out` BF16 [2304, T, S] contiguous.
 */
void inkling_blend_rms_norm(const Tensor& dest, const Tensor& message, const Tensor& gain_forget,
                            const Tensor& norm_weight, float eps, Tensor& out, cudaStream_t stream);

/**
 * The controller tail of a looped blend in ONE launch (decode / verify widths): h = bf16(the `down` split partials
 * added in chunk order) [inkling_gemm_inv's reduce], gain / forget = the invariant GEMM of h with the stream's
 * [4608, 64] weights [inkling_gemm_inv FP32], x = (1 - tanh f) dest + tanh(g) msg and out = bf16(x * rstd(x) * w)
 * [inkling_blend_rms_norm] -- bit-identical to those three launches. A thread-block cluster of 8 CTAs (288 channels
 * each) per 8 columns and stream; the norm's sum of squares goes through distributed shared memory in
 * blend_rms_norm's thread layout. `down_partials` = inkling_gemm_inv_partials of the 64 x 4608 `down` over the S
 * streams; `gain_forget_weight` = the first stream's [4608, 64] (the next streams follow contiguously).
 * inkling_controller_tail_supported: cluster launch available and 1 <= T <= 32 (wider rounds take the three ops:
 * each CTA reads 72 KB of weights, which only pays at small T); SUROGATE_SERVE_INKLING_NO_TAIL=1 turns it off.
 */
[[nodiscard]] bool inkling_controller_tail_supported(std::int32_t columns) noexcept;
void inkling_controller_tail(const float* down_partials, const void* gain_forget_weight, const Tensor& dest,
                             const Tensor& message, const Tensor& norm_weight, float eps, Tensor& out,
                             cudaStream_t stream);

/// One visit's conv state in the target's LinearAttentionStatePool: slot s, tap j (oldest first),
/// channel c at `base + s * slot_stride + j * tap_stride + c` (BF16, elements).
struct InklingConvStateView {
    void* base               = nullptr;
    std::int64_t slot_stride = 0;
    std::int64_t tap_stride  = InklingMixGeometry::kConvChannels;
};

/**
 * The rows of a conv call: B rows of W columns, column n = j + W * b of the inputs and outputs.
 * `positions` I32 [W, B] (taps at positions below 0 read zero), `valid_columns` I32 [B] or empty (all W
 * valid), `state_slots` I32 [B] (device). Columns past a row's valid count are computed from what the
 * input holds and leave the state alone.
 */
struct InklingConvRows {
    std::int32_t width = 0;
    std::int32_t batch = 0;
    Tensor positions;
    Tensor valid_columns;
    Tensor state_slots;
};

/**
 * Op inkling_short_conv (ref `inkling_short_conv`, vLLM `_tap_sum` static + `_sconv_kernel`) on the
 * `channels` channels of one stream starting at `channel_offset` of the state:
 *   acc = fma chain over taps k = 0..3 of bf16(x[t - 3 + k]) * bf16(w[c, k]) from 0,
 *   y = bf16((acc + x[t]) * 5^-0.5); with `addend`: out = bf16((residual + addend) + y) in fp32.
 * `x` a BF16 plane of `channels` (column stride >= channels), `weight` BF16 [4, channels] (the
 * artifact's [channels rows, 4 taps]), `residual` / `addend` / `out` BF16 planes (residual may be the
 * same memory as out: pointwise). Afterwards every row's slot holds the last 3 inputs after its last
 * valid column (in place). `capture_slot` >= 0 (B == 1 only) also writes the window that precedes
 * column `capture_column` to that slot (a rewrite checkpoint inside the segment).
 */
void inkling_short_conv(const InklingPlane& x, std::int32_t channels, const Tensor& weight,
                        const InklingConvRows& rows, const InklingConvStateView& state,
                        std::int32_t channel_offset, const InklingPlane* residual,
                        const InklingPlane* addend, const InklingPlane& out, cudaStream_t stream,
                        std::int32_t capture_column = -1, std::int32_t capture_slot = -1);

/**
 * The window preceding column `column` of a one-row segment (B == 1) -- state + x as inkling_short_conv
 * reads them -- written to slot `dst_slot` for channels [channel_offset, + channels). Used for the K / V
 * streams, whose convolution the attention prologue runs, at a rewrite checkpoint inside a segment.
 * Must run before the conv of that segment updates its slot.
 */
void inkling_conv_capture_window(const InklingPlane& x, std::int32_t channels,
                                 const InklingConvRows& rows, const InklingConvStateView& state,
                                 std::int32_t channel_offset, std::int32_t column,
                                 std::int32_t dst_slot, cudaStream_t stream);

/// The in-place window update alone (no conv): every row's slot := the last 3 inputs of [state | x columns]
/// after its last valid column, for channels [channel_offset, + channels). For the K / V streams, whose
/// convolution the attention prologue (or the naive one) computes from the old window first.
void inkling_conv_update_window(const InklingPlane& x, std::int32_t channels, const InklingConvRows& rows,
                                const InklingConvStateView& state, std::int32_t channel_offset,
                                cudaStream_t stream);

/// Copy channels [channel_offset, + channels) of slot `src` to slot `dst` (all 3 taps).
void inkling_conv_copy_slot(const InklingConvStateView& state, std::int32_t channel_offset,
                            std::int32_t channels, std::int32_t src, std::int32_t dst,
                            cudaStream_t stream);

/// Speculative-verify records (target design notes, section 2): per row b and visit v, the state before
/// the round (`old`: BF16 [6656, 3]) and the round's conv inputs (`inputs`: BF16 [6656, width_max]).
/// old at `old + ((b * visits + v) * 3) * 6656`, inputs at `inputs + ((b * visits + v) * width_max) *
/// 6656`.
struct InklingConvRecords {
    void* old                = nullptr;
    void* inputs             = nullptr;
    std::int32_t visits      = 0;
    std::int32_t width_max   = 0;
    std::int32_t batch_max   = 0;
};

/// The whole per-visit pool for the backup and the fold: visit v's state view = `base + v *
/// visit_stride` (elements), slots as InklingConvStateView.
struct InklingConvPool {
    void* base                = nullptr;
    std::int64_t visit_stride = 0;
    std::int64_t slot_stride  = 0;
    std::int32_t visits       = 0;
};

/// Before a verify round: records.old[b][v] = slot `slots[b]` of every visit (B rows, device I32 [B]).
void inkling_conv_backup(const InklingConvPool& pool, const Tensor& slots, std::int32_t batch,
                         const InklingConvRecords& records, cudaStream_t stream);

/// During a verify round: records.inputs[b][v][j] channels [channel_offset, + channels) = x column
/// j + W * b for j < W (the conv inputs of one stream of visit v).
void inkling_conv_record(const InklingPlane& x, std::int32_t channels, std::int32_t width,
                         std::int32_t batch, std::int32_t visit, std::int32_t channel_offset,
                         const InklingConvRecords& records, cudaStream_t stream);

struct InklingConvFoldRow {
    std::int32_t row       = 0; ///< the row of the verify round (records index b)
    std::int32_t slot      = 0; ///< the lane's current state slot
    std::int32_t committed = 0; ///< accepted columns (0: restore the state before the round)
};

/// At resolve: for every row and visit, slot = the last 3 of [old | inputs[0, committed)].
void inkling_conv_fold(const InklingConvPool& pool, std::span<const InklingConvFoldRow> rows,
                       const InklingConvRecords& records, cudaStream_t stream);

/**
 * out[0:n, t] = W[0:n, :] . x[:, t] in FP32 (bf16 operands, fp32 accumulate and output), `out` with
 * `ldc` FP32 elements per column. W BF16 [k, n] (k contiguous), x BF16 [k, tokens] (k contiguous).
 * n, k, ldc multiples of 8; pointers 16-byte aligned. cuBLASLt (plans cached per shape; call
 * inkling_gemm_f32_prepare before a capture for every shape a captured body uses).
 */
void inkling_gemm_bf16_f32(const void* weight, std::int32_t n, std::int32_t k, const void* x,
                           std::int32_t tokens, float* out, std::int32_t ldc, cudaStream_t stream);
void inkling_gemm_f32_prepare(std::int32_t n, std::int32_t k, std::int32_t tokens, std::int32_t ldc);

/// ---- Batch- and path-invariant GEMM (one invariant forward for engine and trainer) ----
///
/// out[0:n, t] = W[n, k] . x[:, t] for t < tokens, BF16 operands, FP32 accumulation, stored as `out_dtype`
/// (BF16: one rounding at the end; FP32: the accumulator). W is [n, k] with k contiguous, x is [k, tokens] with k
/// contiguous, out has `ldc` elements per token column.
///
/// Every output element is fixed by (n, k) alone, never by `tokens`, the batch, or the tile:
/// - the k range is cut into `inkling_gemm_split(n, k)` equal chunks: k / 256 chunks of 256 when n <= 64, k >= 1024 and
///   k % 256 == 0, else ONE chunk (in inkling2: the controller `down` 64 x 4608 = 18 x 256; every other projection,
///   and every problem of inkling_gemm_inv_group -- the dynamic taps 32 x 2304 included -- runs as one chunk);
/// - each chunk is ONE accumulator from zero running the m16n8k16 BF16 MMA over its k in increasing 16-wide steps
///   (k padded with zeros to a multiple of 64) -- the chain of ops/linear/bf16/bf16_invariant_gemm;
/// - the chunks' FP32 partials are added in chunk order (p0 + p1 + ... ) and the sum is stored.
/// The tile follows `tokens` for speed only; no tile changes a chain, no split-K follows the width.
///
/// `batch` runs `count` independent problems of one shape in one launch (the 3 streams of a blend): problem b reads
/// weight + b * weight_stride, x + b * x_stride (elements) and writes out + b * out_stride.
struct InklingGemmBatch {
    std::int32_t count        = 1;
    std::int64_t weight_stride = 0;
    std::int64_t x_stride      = 0;
    std::int64_t out_stride    = 0;
};
[[nodiscard]] std::int32_t inkling_gemm_split(std::int32_t n, std::int32_t k) noexcept;
/// Scratch for the chunks' partials (0 when the shape runs as one chunk).
[[nodiscard]] std::size_t inkling_gemm_inv_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                                           std::int32_t batch) noexcept;
/// k a positive multiple of 8, ldc >= n, weight and x 16-byte aligned, `workspace` at least
/// inkling_gemm_inv_workspace_bytes(n, k, tokens, batch.count) bytes (16-byte aligned) when the shape splits.
void inkling_gemm_inv(const void* weight, std::int32_t n, std::int32_t k, const void* x, std::int32_t tokens,
                      void* out, DType out_dtype, std::int32_t ldc, const InklingGemmBatch& batch, void* workspace,
                      std::size_t workspace_bytes, cudaStream_t stream);
/// Selects the narrow wo kernel (T <= 8, one problem, n % 16 == 0, k % 64 == 0; same bits as the generic kernel) for tests
/// and A/Bs. Default: off; SUROGATE_SERVE_INKLING_WO_NARROW=1 = the narrow kernel.
void inkling_gemm_inv_use_narrow(bool on) noexcept;

/// The first phase of a split shape alone: the chunks' FP32 partials, [splits][batch][tokens][n] (chunk-major), the
/// operand of inkling_controller_tail (which adds them in chunk order exactly as inkling_gemm_inv's reduce does).
void inkling_gemm_inv_partials(const void* weight, std::int32_t n, std::int32_t k, const void* x, std::int32_t tokens,
                               const InklingGemmBatch& batch, float* partials, std::size_t partials_bytes,
                               cudaStream_t stream);

/// Up to 6 problems of one k, token count and store type in ONE launch, each with its own weights, row count, input
/// and output (a global / looped visit's q-side / K / V row ranges and, on a global visit, the dynamic K / V taps).
/// Every problem runs as ONE chunk -- the chain of inkling_gemm_inv for an unsplit shape -- whatever
/// inkling_gemm_split says for its shape: a shape that a caller sends through a group must always take this entry.
struct InklingGemmProblem {
    const void* weight = nullptr; ///< [n, k]
    std::int32_t n     = 0;
    const void* x      = nullptr; ///< [k, tokens]
    void* out          = nullptr; ///< [ldc, tokens]
    std::int32_t ldc   = 0;
};
void inkling_gemm_inv_group(std::span<const InklingGemmProblem> problems, std::int32_t k, std::int32_t tokens,
                            DType out_dtype, cudaStream_t stream);

/// ---- The trainer's log-prob (the engine-op IR) ----
///
/// out[t] = log softmax(softcap(W_head . hidden[:, t]))[token_ids[t]] over the first `token_domain` rows, bit for bit
/// the value every log-prob the engine returns carries (completion token_logprobs and prompt_logprobs are
/// RawTokenScores.selected): the FP32 head of inkling_gemm_inv + inkling_softcap_f32 (Variant::lm_head), then the
/// kernel of ops::score_logprobs_device (the raw logits, temperature 1: a 512-thread max tree, per-thread expf sums,
/// a tree add, max + log(sum)). `hidden` BF16 [hidden_size, T]; `out` FP32 [T] (NaN for an id outside the domain).
[[nodiscard]] std::size_t inkling_head_logprob_workspace_bytes(std::int32_t vocab_rows, std::int32_t tokens) noexcept;
void inkling_head_logprob(const void* head_weight, std::int32_t vocab_rows, std::int32_t hidden_size,
                          std::int32_t token_domain, float softcap, const void* hidden, std::int32_t tokens,
                          const std::int32_t* token_ids, float* out, void* workspace, std::size_t workspace_bytes,
                          cudaStream_t stream);

/// One output slice of a LoRA-adapted projection (inkling_lora_delta): bank `bank` of the device bank table, its BF16
/// input plane `x` [k, T] (column stride `x_stride` elements) and its BF16 output plane `out` [rows, T] (column stride
/// `out_stride`); `rows` / k must be the bank's n / k (the binding guarantees it).
struct InklingLoraSlice {
    std::int32_t bank      = 0;
    std::int32_t rows      = 0;
    const void* x          = nullptr;
    std::int64_t x_stride  = 0;
    void* out              = nullptr;
    std::int64_t out_stride = 0;
};
inline constexpr int kInklingLoraMaxSlices = 6;

/// Bytes of the fp32 low-rank buffer of one inkling_lora_delta call: [tokens][slices][max_rank].
[[nodiscard]] std::size_t inkling_lora_delta_workspace_bytes(std::int32_t slices, std::int32_t max_rank,
                                                             std::int32_t tokens) noexcept;

/**
 * Attention LoRA (S10), ONE arithmetic at every width / batch: for every token t whose slot
 * s = token_slot[t] (or *uniform_slot) is >= 0 and every slice:
 *   low[r] = fma chain over k = 0..K-1 of A_s[r, k] * x[k, t] (fp32, from 0; 32 lane chains with a fixed xor-tree),
 *   delta[j] = fma chain over r = 0..R-1 of B_s[j, r] * low[r] (fp32, from 0),
 *   out[j, t] = bf16_rn(float(out[j, t]) + delta[j]) where delta[j] != 0 (else out stays: a zero adapter is bitwise
 *   the base). A negative slot leaves the token's rows untouched. `table` is the LoRA store's device bank table
 * (alpha / r folded into A by the store; R = the store's max_rank, shorter adapters zero-padded); per-token work only,
 * so the bits do not depend on T, the batch or the other tokens.
 */
void inkling_lora_delta(const void* table, const InklingLoraSlice* slices, std::int32_t slice_count,
                        std::int32_t max_rank, std::int32_t tokens, const std::int32_t* token_slot,
                        const std::int32_t* uniform_slot, void* workspace, std::size_t workspace_bytes,
                        cudaStream_t stream);

/// In place: x = (tanh(x / cap)) * cap in fp32 (vLLM LogitsProcessor soft_cap), FP32 [rows, T] with
/// `ldc` elements per column, first `rows` rows of each column.
void inkling_softcap_f32(float* logits, std::int32_t rows, std::int32_t tokens, std::int32_t ldc,
                         float cap, cudaStream_t stream);

} // namespace sinfer::ops
