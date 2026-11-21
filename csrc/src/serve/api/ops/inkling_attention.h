#pragma once

// Inkling-2-Mini attention (ENGINE-INKLING lane c): the fused prologue, the relative-bias attention
// (prefill / decode / verify), the gate-and-sink epilogue, the log-n query scale tau, the load-time
// relative-projection table, and (PROPOSED) the attention projections.
//
// Semantics: the python functions of the same names in rl/engine-inkling-2026-11-21/ref/inkling_ref.py,
// which follow vLLM's Inkling2 sm_120 path (ref/src/vllm/models/inkling2/nvidia/ops/{sconv,norm,
// attention,attention_sm90}.py). Rounding points and the reasons for every choice below:
// rl/engine-inkling-2026-11-21/lanes/c/DESIGN-attention.md (section numbers cited as "D<n>").
//
// STATUS: M1 header for the integrator's approval. Nothing declared here is implemented yet.
//
// Layout convention (this engine's): the FIRST dimension is the fastest. "[128, 24, N]" is N columns
// of 24 heads of 128 contiguous values. A batch of B rows of W columns each is N = W * B columns,
// column n = j + W * b. Geometry is fixed (Inkling-2-Mini, both local and global layers): every
// kernel is compiled for it and every entry point rejects other shapes.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace sinfer::ops {

/// Inkling-2-Mini attention geometry (DESIGN.md 1.3; ref/config.json `swa_*` equal to the global values).
struct InklingAttentionGeometry {
    static constexpr std::int32_t kQHeads    = 24;
    static constexpr std::int32_t kKvHeads   = 8;
    static constexpr std::int32_t kGroup     = kQHeads / kKvHeads; // 3
    static constexpr std::int32_t kHeadDim   = 128;
    static constexpr std::int32_t kRelDim    = 16;    // d_rel
    static constexpr std::int32_t kRelExtent = 1024;  // bias distances 0..1023 (global and local)
    static constexpr std::int32_t kWindow    = 1024;  // local layers: keys with pos_q - pos_k < 1024
    static constexpr std::int32_t kConvTaps  = 4;
    static constexpr std::int32_t kHidden    = 2304;
    static constexpr std::int32_t kAttnWidth = kQHeads * kHeadDim; // 3072, the gated output / wo_ud input

    /// Rows of the fused `attn/qkvr` projection (ARTIFACT.md 4.2): [wq 3072 | wr 384 | gate 3072 |
    /// sink 24 | wk 1024 | wv 1024]. A 3-input visit projects [kRowsQ, kRowsK) from x_q, the K rows
    /// from x_k and the V rows from x_v.
    static constexpr std::int32_t kRowsQ    = 0;
    static constexpr std::int32_t kRowsR    = 3072;
    static constexpr std::int32_t kRowsGate = 3456;
    static constexpr std::int32_t kRowsSink = 6528;
    static constexpr std::int32_t kRowsK    = 6552;
    static constexpr std::int32_t kRowsV    = 7576;
    static constexpr std::int32_t kRowsQkvr = 8600;

    static constexpr float kQkClip   = 22.627416997969522F; // qk_l2_clip_threshold (16 sqrt 2)
    static constexpr float kRelClip  = 8.0F;                // rel_l2_clip
    static constexpr float kNormEps  = 1e-6F;               // V RMS norm eps (rms_norm_eps)
    static constexpr float kClipEps  = 1e-6F;               // L2-clip eps
    static constexpr std::int32_t kTauFloor = 8192;         // log_scaling_n_floor
    static constexpr float kTauAlpha = 0.1F;                // log_scaling_alpha

    /// The decode / verify kernel takes rows of at most this many columns; wider rows run the prefill
    /// tile kernel (D5). DSpark verify rounds are 8 columns.
    static constexpr std::int32_t kMaxSmallColumns = 16;

    /// `inkling_rel_proj_table`: distance-major, zero-padded by kRelTablePad distances on each side.
    static constexpr std::int32_t kRelTablePad  = 128;
    static constexpr std::int32_t kRelTableRows = kRelExtent + 2 * kRelTablePad; // 1280
};

/// Local (sliding-window) or global (full) attention layer. Decides the window, the K/V convolution
/// (static, 1/sqrt(5) vs dynamic, 1/3) and whether q and r are scaled by tau. A local core layer in
/// loop passes 1-2 takes separate q / k / v inputs but stays Local.
enum class InklingLayerKind : std::uint8_t {
    Local,
    Global,
};

/**
 * The rows of one attention call (shared by the prologue and the attention of the same visit).
 *
 * `positions` is contiguous I32 [W, B]: the absolute position of column j of row b. A row's valid
 * columns hold consecutive positions p0, p0 + 1, ...; after the prologue the row's plane holds keys
 * [0, p_last + 1) (a local plane: at least the window before each query, D2). Columns past a row's valid
 * count must hold a position in [0, 1048576); they are never written or read and come out exact zero.
 * `valid_columns` is contiguous I32 [B] with every value in [0, W] (0: an idle padding row), or an
 * empty Tensor meaning every row has W valid columns. `kv_table_rows` is contiguous I32 [B]: the row of
 * the plane's block-table matrix each sequence is on.
 *
 * `max_visible_keys` is a host promise, the maximum over rows of p_last + 1. Only the prefill route
 * (W > kMaxSmallColumns) and the workspace query read it: a decode / verify launch plans from (B, W,
 * kind) alone, so a CUDA graph captured for a bucket replays at any context length up to the promise.
 */
struct InklingAttentionRows {
    Tensor positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    std::int32_t max_visible_keys = 0;
};

/**
 * The K / V short-convolution state of one visit (the attention's two of the visit's four streams).
 *
 * `states` is BF16 with logical shape [2048, 3, S]: channel c (K channels 0..1023, then V channels
 * 1024..2047), tap j (oldest first), slot s. Dimension 0 is contiguous; nb[1] and nb[2] are any
 * multiples of 16 bytes, so the K|V part of a larger per-visit slot (e1's 4-stream [6656, 3]) is a view.
 * `initial_slots` (I32 [B]) names the window each row starts from; taps at positions below 0 read as
 * zero whatever the slot holds. `output_slots` (I32 [B]) receives the window after the row's last valid
 * column, or with `per_column` (verify rounds) slot `output_slots[b] + j` receives the window after
 * column j for every valid j -- the engine's short_conv_snapshot convention. A row's initial slot may
 * alias any of its output slots; reservations of different rows must not overlap. Idle rows touch
 * nothing.
 */
struct InklingConvStateIO {
    Tensor states;
    Tensor initial_slots;
    Tensor output_slots;
    bool per_column = false;
};

/**
 * Prologue inputs: views into the attention projection's output (no copy), D3.
 *
 * q: BF16 [128, 24, N], dims 0-1 contiguous, any 16-byte column stride (rows [kRowsQ, kRowsR) of the
 * projection). r: BF16 [16, 24, N] likewise (rows [kRowsR, kRowsGate)). k, v: BF16 [1024, N], dim 0
 * contiguous, any 16-byte column stride. Global layers: k_dyn, v_dyn = BF16 [4, 8, N] (tap lag fastest,
 * then KV head) = weight_proj(x_k), weight_proj(x_v) as bf16 projection outputs; tau = FP32 [W, B] from
 * inkling_log_tau of the same positions. Local layers: k_dyn, v_dyn and tau empty.
 */
struct InklingPrologueInputs {
    Tensor q;
    Tensor r;
    Tensor k;
    Tensor v;
    Tensor k_dyn;
    Tensor v_dyn;
    Tensor tau;
};

/**
 * Per-layer prologue weights, as the artifact stores them (ARTIFACT.md 4.2), all BF16 contiguous.
 * q_scale: `attn/q_scale` [24 rows, 128] = [128, 24] here. r_scale: `attn/r_scale` = [16, 24].
 * Local: k_conv, v_conv = `attn/k_conv`, `attn/v_conv`, the artifact's [1024 rows, 4 taps] (a channel's
 * taps contiguous, oldest first) = [4, 1024] here. Global: `attn/k_conv_static`, `attn/v_conv_static`,
 * the artifact's [4 rows = lag 0 (current token) .. 3, 1024] = [1024, 4] here.
 */
struct InklingPrologueWeights {
    Tensor q_scale;
    Tensor r_scale;
    Tensor k_conv;
    Tensor v_conv;
};

/// Prologue outputs: q_out BF16 [128, 24, N] (clipped, tau-scaled) and r_out BF16 [16, 24, N]
/// (clipped, tau-scaled), both contiguous. Read by inkling_rel_attention of the same visit.
struct InklingPrologueOutputs {
    Tensor q;
    Tensor r;
};

/**
 * Op inkling_attention_prologue (OPS.md; ref `inkling_attention_prologue`, D1 P1-P8, D3).
 *
 * For every valid column: K and V = causal depthwise conv (Local: static, kernel 4, residual, 5^-0.5;
 * Global: dynamic taps static_w + dyn, residual, 1/3) of the bf16 projections over the row's history
 * (conv state + earlier columns), rounded to bf16; K L2-clipped to kQkClip, V RMS-normalised without a
 * weight; both written to `cache` at the column's position as unit-scale e4m3 (RNE, satfinite, straight
 * from fp32). q_out = bf16(l2clip(q * q_scale, kQkClip)), r_out = bf16(l2clip(r * r_scale, kRelClip));
 * Global: each further bf16(value * tau). Conv windows written per InklingConvStateIO.
 *
 * `cache` is the visit's plane (FP8 e4m3 K and V planes, no scale planes, head dim 128, 8 KV heads);
 * the op writes only the addressed slots. Launched with programmatic dependent launch: it triggers its
 * dependents at entry, so the attention that follows may prefetch the keys of earlier forwards.
 */
void inkling_attention_prologue(const InklingPrologueInputs& inputs,
                                const InklingPrologueWeights& weights, InklingLayerKind kind,
                                const InklingAttentionRows& rows, const InklingConvStateIO& conv,
                                const PagedKVBatchLayerView& cache, InklingPrologueOutputs& outputs,
                                cudaStream_t stream);

/**
 * Attention inputs of one visit. q, r: the prologue's outputs. gate: BF16 [3072, N], dim 0 contiguous,
 * any 16-byte column stride (projection rows [kRowsGate, kRowsSink)). sink: BF16 [24, N] likewise (rows
 * [kRowsSink, kRowsK)). rel_table: the layer's inkling_rel_proj_table.
 */
struct InklingAttentionInputs {
    Tensor q;
    Tensor r;
    Tensor gate;
    Tensor sink;
    Tensor rel_table;
};

/// Workspace for inkling_rel_attention / _out_lse at B rows of W columns (split partials, merge
/// counters); zero when the call plans a single split. Throws outside the supported domain.
[[nodiscard]] std::size_t inkling_rel_attention_workspace_bytes(InklingLayerKind kind,
                                                                std::int32_t width,
                                                                std::int32_t batch,
                                                                std::int32_t max_visible_keys);

/**
 * Op inkling_rel_attention, production route (ref `inkling_rel_attention` + `inkling_gate_sink`; D1
 * A1-A7, D4, D5).
 *
 * For valid column j of row b at position p, query head h (KV head h / 3), keys x of the row's plane:
 *   s(x)    = (q_h . K_x) / sqrt(128) + bias,  bias = bf16(r_h . rel_proj[:, p - x]) for 0 <= p - x < 1024,
 *             0 beyond (Global); visible x: x <= p, and p - x < 1024 (Local)
 *   o       = softmax_x(s) @ V,  lse = log sum_x exp s(x)
 *   out[:, h, column] = bf16( o * sigmoid(gate) * sigmoid(lse - sink[h]) )
 * K / V are the e4m3 codes widened exactly to bf16; q . K on bf16 tensor cores with fp32 accumulate.
 * `out` is contiguous BF16 [3072, N] (the wo_ud input); invalid columns and idle rows are exact zero.
 * Rows of W <= kMaxSmallColumns run the split-KV decode / verify kernel, wider rows the prefill tile
 * kernel. Reads the plane only; the prologue of the same visit must have written this round's keys.
 */
void inkling_rel_attention(const InklingAttentionInputs& inputs, InklingLayerKind kind,
                           const InklingAttentionRows& rows, const PagedKVBatchLayerView& cache,
                           WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * The same attention without the gate and sink (gate / sink unused): `out` FP32 [128, 24, N] and
 * `lse` FP32 [24, N] (natural log), contiguous -- the reference's MODE_OUT_LSE. For op tests against
 * the golden vectors and for debugging; not a production route.
 */
void inkling_rel_attention_out_lse(const InklingAttentionInputs& inputs, InklingLayerKind kind,
                                   const InklingAttentionRows& rows,
                                   const PagedKVBatchLayerView& cache, WorkspaceArena& workspace,
                                   Tensor& out, Tensor& lse, cudaStream_t stream);

/**
 * Op inkling_gate_sink (ref `inkling_gate_sink`, vLLM `_gate_sink_kernel`, D1 A7): result =
 * bf16((out * sigmoid(gate)) * sigmoid(lse - sink)), sigmoid(x) = rcp.approx(ex2.approx(-x log2 e) + 1).
 * out FP32 [128, 24, N] contiguous, lse FP32 [24, N] contiguous, gate BF16 [3072, N] and sink BF16
 * [24, N] with dim 0 contiguous and any column stride, result contiguous BF16 [3072, N].
 */
void inkling_gate_sink(const Tensor& out, const Tensor& lse, const Tensor& gate, const Tensor& sink,
                       Tensor& result, cudaStream_t stream);

/**
 * Op inkling_log_tau (ref `inkling_log_tau`, vLLM `compute_log_scaling_tau`, D1 P8):
 * tau = 1 + 0.1 * ln(max((pos + 1) * (1 / 8192), 1)) in fp32, product and sum rounded separately,
 * accurate logf. positions contiguous I32 (any shape), tau contiguous FP32 of the same element count.
 */
void inkling_log_tau(const Tensor& positions, Tensor& tau, cudaStream_t stream);

/**
 * Load-time transform of one layer's `attn/rel_proj` (BF16 [16 rows, 1024] = [1024, 16] here,
 * row j = rel dim j) into `table`, contiguous BF16 [16, kRelTableRows]: table[j, e] = rel_proj[j, d]
 * for distance d = e - kRelTablePad in [0, 1024), exact zero otherwise. One table per layer, shared by
 * the passes of a core layer. Replaces the reference's `shift_rel_proj`.
 */
void inkling_rel_proj_table(const Tensor& rel_proj, Tensor& table, cudaStream_t stream);

/**
 * Test and debug only: the dense relative logits rel[d, h, n] = bf16(r_out[:, h, n] . rel_proj[:, d])
 * for d in [0, 1024), FP32-accumulated on the same k16 MMA the attention kernels use -- the `rel`
 * tensor of lane a's PrologueOut. `rel` contiguous BF16 [1024, 24, N].
 */
void inkling_rel_logits(const Tensor& r_out, const Tensor& rel_table, Tensor& rel,
                        cudaStream_t stream);

// ---- PROPOSED (D6): the attention projections. Declared for review; not to be wired until the
// ---- integrator approves it and DESIGN 10.1 names the FP8 scheme.

/**
 * One projection of a visit: out[0:rows, n] = W[row_begin : row_begin + rows, :] . x[:, n], rows =
 * out.ne[0]. x: contiguous BF16 [K, N] (K = W.k). weight: the layer's `attn/qkvr` or `attn/output` in
 * BF16 (the oracle path) or the FP8 format 10.1 chooses, or a BF16 `attn/{k,v}_conv_proj`.
 * out: BF16 [rows, N], dim 0 contiguous, any 16-byte column stride (a slice of a larger output).
 */
struct InklingProjection {
    const Tensor* x       = nullptr;
    const Weight* weight  = nullptr;
    std::int32_t row_begin = 0;
    Tensor* out           = nullptr;
};

[[nodiscard]] std::size_t inkling_attn_linear_workspace_bytes(
    std::span<const InklingProjection> projections, std::int32_t max_columns);

/**
 * Op inkling_attn_linear (PROPOSED, D6): up to five projections in one launch where their formats
 * allow (a 3-input global visit: q-side, K, V, k_dyn, v_dyn). FP8: e4m3 codes x exact BF16 activations
 * (W8A16), fp32 accumulate, scales applied in fp32 (per row after the whole K; per 1x128 / 128x128 cell
 * per 128 of K), output rounded once to bf16. BF16 weights: the engine's existing linear_rows.
 */
void inkling_attn_linear(std::span<const InklingProjection> projections, WorkspaceArena* workspace,
                         cudaStream_t stream);

} // namespace sinfer::ops
