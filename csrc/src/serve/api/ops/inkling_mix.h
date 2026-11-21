#pragma once

// Inkling-2-Mini hidden-stream ops (ENGINE-INKLING lane e1): the block attention-residual mix, the loop
// message controller's two norms, the hidden-stream short convolution with its per-visit state, the
// embedding with its norm, and the softcapped fp32 LM head.
//
// Semantics: the python functions of the same names in rl/engine-inkling-2026-11-21/ref/inkling_ref.py,
// which follow vLLM's Inkling2 sm_120 path (ref/src/vllm/models/inkling2/nvidia/ops/{attn_res,loop,sconv,
// norm}.py). Rounding points: lanes/e1/DESIGN-target.md section 3. Every rsqrt is the non-FTZ
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
 * Loop controller, second half (vLLM `blend_rms_norm`, unfused):
 *   x = (1 - tanh(forget)) * dest + tanh(gain) * msg,  out = bf16((x * rsqrt(div_rn(sum x^2, 2304) +
 *   eps)) * w_s).
 * `gain_forget` FP32 [4608, T, S] (gain rows first), `norm_weight` BF16 [2304, S] or [2304, 1]
 * (broadcast), `dest` / `message` as inkling_pair_rms_norm, `out` BF16 [2304, T, S] contiguous.
 */
void inkling_blend_rms_norm(const Tensor& dest, const Tensor& message, const Tensor& gain_forget,
                            const Tensor& norm_weight, float eps, Tensor& out, cudaStream_t stream);

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
 * streams, whose convolution lane c's prologue runs, at a rewrite checkpoint inside a segment.
 * Must run before the conv of that segment updates its slot.
 */
void inkling_conv_capture_window(const InklingPlane& x, std::int32_t channels,
                                 const InklingConvRows& rows, const InklingConvStateView& state,
                                 std::int32_t channel_offset, std::int32_t column,
                                 std::int32_t dst_slot, cudaStream_t stream);

/// The in-place window update alone (no conv): every row's slot := the last 3 inputs of [state | x columns]
/// after its last valid column, for channels [channel_offset, + channels). For the K / V streams, whose
/// convolution lane c's prologue (or the naive one) computes from the old window first.
void inkling_conv_update_window(const InklingPlane& x, std::int32_t channels, const InklingConvRows& rows,
                                const InklingConvStateView& state, std::int32_t channel_offset,
                                cudaStream_t stream);

/// Copy channels [channel_offset, + channels) of slot `src` to slot `dst` (all 3 taps).
void inkling_conv_copy_slot(const InklingConvStateView& state, std::int32_t channel_offset,
                            std::int32_t channels, std::int32_t src, std::int32_t dst,
                            cudaStream_t stream);

/// Speculative-verify records (lanes/e1/DESIGN-target.md 2): per row b and visit v, the state before
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

/// In place: x = (tanh(x / cap)) * cap in fp32 (vLLM LogitsProcessor soft_cap), FP32 [rows, T] with
/// `ldc` elements per column, first `rows` rows of each column.
void inkling_softcap_f32(float* logits, std::int32_t rows, std::int32_t tokens, std::int32_t ldc,
                         float cap, cudaStream_t stream);

} // namespace sinfer::ops
