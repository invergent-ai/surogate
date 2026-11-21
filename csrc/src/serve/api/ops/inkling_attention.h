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
// STATUS: M1 header approved by agent-EN (7e7d4e1a); M2 revision aligns the rows / conv-state / call
// shape with lane e1's VisitRound contract (lanes/e1/DESIGN-target.md section 4 c): one call per visit,
// InPlace / Record conv modes on the visit's LinearAttentionStatePool slots, tau from positions inside.
//
// Layout convention (this engine's): the FIRST dimension is the fastest. "[128, 24, N]" is N columns
// of 24 heads of 128 contiguous values. A batch of B rows of W columns each is N = W * B columns,
// column n = j + W * b. Geometry is fixed (Inkling-2-Mini, both local and global layers): every
// kernel is compiled for it and every entry point rejects other shapes.

#include "api/ops/gqa_attention.h" // GqaExecutionEnvelope
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
 * Key groups of the invariant attention: the keys of every row are cut into groups whose boundaries depend only on the
 * absolute key position; a group is walked in 64-key blocks with one running softmax state, groups are folded in order
 * (inkling_rel_attention.cu header). The cut is part of the arithmetic: the engine and the trainer must use the SAME
 * value (it is a field of InklingAttentionRows, which both entry points take). Global / local group sizes:
 * P1 512 / 128; P2 2,048 (8,192 from key 65,536) / 256; P3 512 below key 65,536, 2,048 to 131,072, 8,192 beyond / 128;
 * P4 (measurement) P3 / 64; P5 (measurement) 256 below 65,536 then P3 / 128. Default = the process default: the
 * compiled kInklingDefaultKeyGroups = P3, FROZEN by lane h's price session (2026-11-21 14:1x: P3 mean -0.93 % vs the
 * non-invariant BASE, worst +1.1 % at 1x262k; P1 +5.4 %, P2 +10.8 % on their worst probes). P1 / P2 / P4 / P5 are
 * test / measurement options only. SUROGATE_INKLING_ATTN_GROUPS (p1 .. p5), if set when the process first uses these
 * ops, overrides the default (read once; a test override).
 */
enum class InklingKeyGroups : std::uint8_t {
    Default,
    P1,
    P2,
    P3,
    P4,
    P5,
};
inline constexpr InklingKeyGroups kInklingDefaultKeyGroups = InklingKeyGroups::P3;

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
 * `envelope` is the same host promise gqa_attention takes (lane e1's VisitBlock carries it); only its
 * `max_visible_keys` is read (the layer kind, not the envelope, decides the window), and only by the
 * prefill route (W > kMaxSmallColumns) and the workspace query: a decode / verify launch plans from
 * (B, W, kind) alone, so a CUDA graph captured for a bucket replays at any context length up to it.
 */
struct InklingAttentionRows {
    Tensor positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    GqaExecutionEnvelope envelope{};
    /// Caller's promise (default false): positions, valid_columns, kv_table_rows and the plane's block tables were
    /// complete before the step's FIRST kernel (host copies), and the plane's keys before each row's first position
    /// were written by earlier steps. The attention then reads them before its programmatic-dependency wait and asks
    /// L2 for its first old blocks, overlapping the prologue. Same bits either way.
    bool metadata_static = false;
    /// The key-group cut (see InklingKeyGroups); engine and trainer must pass the same value.
    InklingKeyGroups key_groups = InklingKeyGroups::Default;
    /// Optional persistent state (lane d's InklingMoeOptions::state pattern): device memory of
    /// inkling_attention_state_bytes() bytes, ZEROED ONCE by the caller when allocated, written by nothing else, never
    /// shared by two calls in flight (one per stream; calls on one stream are ordered). Not the transient workspace.
    /// With it the decode / verify split route's last-CTA fold takes its arrival counters from here and the last CTA
    /// returns each to zero before the grid ends (no per-call zeroing: neither the prologue's nor a zeroing launch; a
    /// captured graph replays from zeros). The counters pick only WHICH CTA folds, never the order of a sum: same bits
    /// with or without. Null = counters in the workspace, zeroed per call. Rows with B > 128 ignore it.
    std::int32_t* state = nullptr;
};

/// Bytes of InklingAttentionRows::state (arrival counters for up to 128 rows x 8 KV heads: 4 KiB).
[[nodiscard]] std::size_t inkling_attention_state_bytes();

/**
 * Process-wide tuning knobs of these ops: read ONCE from the environment when the process first uses them
 * (SUROGATE_INKLING_ATTN_GROUPS / _SPLITS / _CTAS / _PARTIAL_MB / _L2AHEAD / _PDL_TRIGGER / _VARIANT / _EARLY_META,
 * SUROGATE_INKLING_PREFILL_KS), never per call. Every knob changes the plan or the speed only, never the bits, except
 * `variant` (a numerics experiment; 17 = production) and `default_groups` (the cut for rows that pass Default).
 * Tests and benches may edit them between calls through inkling_attention_knobs(): not thread safe, never in serving.
 */
struct InklingAttentionKnobs {
    InklingKeyGroups default_groups = kInklingDefaultKeyGroups;
    int splits                      = 0;    // > 0: force this many workers (split route)
    int cta_target                  = 0;    // decode CTA target (0: 2 x SMs)
    int prefill_ks                  = 2;    // 2: 8-warp prefill kernel, 1: 4-warp
    int l2_ahead                    = -1;   // blocks asked of L2 ahead (-1: off; measured slower on every route)
    bool pdl_trigger                = true; // trigger dependents at kernel entry
    int variant                     = 17;   // numerics variant bits (production 17)
    bool early_meta                 = true; // honour InklingAttentionRows::metadata_static
    std::size_t partial_cap_bytes   = std::size_t{1} << 30; // split-route partials above this -> one worker
    // Prefill route (W > 16) only: split partials above this -> one worker (SUROGATE_INKLING_ATTN_PREFILL_PARTIAL_MB,
    // default 64). Bounds the workspace an 8,192-column chunk reserves at long context (agent-EN 15:4x: the 1 GiB cap
    // reserved 1,025 MiB at 262k); a wide chunk has CTAs from its columns, a narrow tail at long context runs unsplit.
    std::size_t prefill_partial_cap_bytes = std::size_t{64} << 20;
    // Decode / verify split route: the last worker CTA of a (row, KV head) folds its 3 x W rows (no fold_kernel launch)
    // when W x slots <= this (SUROGATE_INKLING_ATTN_FUSED_FOLD; 0 = fold_kernel always). Same bits either way.
    // Default 0: measured SLOWER than fold_kernel under graphs + PDL (perf18 16:0x: global 1x8k 18.2 -> 20.5 us, local
    // 6.0 -> 6.7, verify W8 19.6 -> 29.6; lanes/c/NOTES.md PERF LOG).
    int fused_fold_max = 0;
    // inkling_attn_linear, N <= 16: the cell-split narrow kernel (a row tile's cells over several warps, joined in cell
    // order) for 0 = no projection, 1 = wo_ud (K = 3,072) only, 2 = wo_ud and qkvr (SUROGATE_INKLING_LINEAR_SPLIT). Same
    // bits either way. Default 1 (perf18: wo N=1 9.4 -> 6.6 us, N=8 9.7 -> 7.2; qkvr N=1 13.4 -> 14.3: slower).
    int linear_split = 1;
    // inkling_attention (the fused call) on plain decode rows (W = 1, InPlace, no capture): the decode attention kernel
    // runs the prologue itself -- q / r per CTA, the K / V conv by the CTA owning the new key -- instead of a prologue
    // launch (SUROGATE_INKLING_ATTN_FUSED_PROLOGUE=1). Same bits either way. Default off: perf18 measured it ~1 us
    // SLOWER per call than the separate prologue (local 1x8k call 8.4 -> 9.4 us with the fused fold).
    bool fused_prologue = false;
};
[[nodiscard]] InklingAttentionKnobs& inkling_attention_knobs();

/// InPlace (prefill, decode, the narrow MTP round): each row's slot is read, then replaced by the
/// window after its last valid column. Record (speculative verify): the slot is only read; every valid
/// column's raw K / V input is written to `records`, and lane e1's fold commits the accepted prefix.
enum class InklingConvMode : std::uint8_t {
    InPlace,
    Record,
};

/**
 * The K / V short-convolution state of one visit (lane e1's LinearAttentionStatePool, DESIGN-target.md
 * section 2: per visit and slot BF16 [6656, 3], channels [K 0..1023 | V 1024..2047 | attn | mlp], oldest
 * tap first, values = the bf16 conv INPUTS).
 *
 * `states`: BF16 [C, 3, S] view of the visit's slots (C >= v_offset + 1024; dim 0 contiguous; nb[1],
 * nb[2] multiples of 16 bytes). K channel c is state channel k_offset + c, V channel c is v_offset + c.
 * `slots`: contiguous I32 [B], each row's slot. Taps at positions below 0 read as zero whatever the slot
 * holds. Record mode: `records` is BF16 [C, W_max, B] (dim 0 contiguous, W_max >= W): column j of row b
 * lands at (k_offset + c or v_offset + c, j, b); InPlace: `records` empty. `capture_column` >= 0 (InPlace,
 * B = 1: a single-segment prefill's rewrite checkpoint) also writes the window BEFORE that column (the
 * three inputs at columns capture_column - 3 .. - 1, from the slot where negative) to `capture_slot`.
 * Every state read of a row happens before any state write of that row, so a slot is safely read and
 * replaced by one call. Idle rows touch nothing.
 */
struct InklingConvStateArgs {
    Tensor states;
    Tensor slots;
    InklingConvMode mode = InklingConvMode::InPlace;
    Tensor records;
    std::int32_t capture_column = -1;
    std::int32_t capture_slot   = -1;
    std::int32_t k_offset       = 0;
    std::int32_t v_offset       = 1024;
};

/**
 * One visit's attention inputs: views into its projection outputs (no copy), D3. A 1-input visit reads
 * them all from one [8600, N] buffer; a 3-input visit from a [6552, N] query-side buffer (x_q) and two
 * [1024, N] buffers (x_k, x_v).
 *
 * q: BF16 [128, 24, N], dims 0-1 contiguous, column stride any multiple of 8 bytes (rows [kRowsQ, kRowsR)).
 * r: BF16 [16, 24, N] likewise (rows [kRowsR, kRowsGate)). gate: BF16 [3072, N] (rows [kRowsGate,
 * kRowsSink)), sink: BF16 [24, N] (rows [kRowsSink, kRowsK)). k, v: BF16 [1024, N]. Global layers only:
 * k_dyn, v_dyn = BF16 [4, 8, N] (tap lag fastest, then KV head) = bf16 weight_proj(x_k), weight_proj(x_v);
 * empty on local layers. tau is computed inside from the rows' positions (global layers).
 */
struct InklingProjections {
    Tensor q;
    Tensor r;
    Tensor gate;
    Tensor sink;
    Tensor k;
    Tensor v;
    Tensor k_dyn;
    Tensor v_dyn;
};

/**
 * Per-layer attention weights, as the artifact stores them (ARTIFACT.md 4.2), all BF16 contiguous.
 * q_scale: `attn/q_scale` [24 rows, 128] = [128, 24] here. r_scale: `attn/r_scale` = [16, 24].
 * Local: k_conv, v_conv = `attn/k_conv`, `attn/v_conv`, the artifact's [1024 rows, 4 taps] (a channel's
 * taps contiguous, oldest first) = [4, 1024] here. Global: `attn/k_conv_static`, `attn/v_conv_static`,
 * the artifact's [4 rows = lag 0 (current token) .. 3, 1024] = [1024, 4] here. rel_table: the layer's
 * inkling_rel_proj_table (built once at load from `attn/rel_proj`).
 */
struct InklingAttentionWeights {
    Tensor q_scale;
    Tensor r_scale;
    Tensor k_conv;
    Tensor v_conv;
    Tensor rel_table;
};

/**
 * THE call per visit (lane e1's contract, DESIGN-target.md 4 c): prologue + attention + gate and sink.
 * Writes the visit's K / V to `cache`, updates or records the conv state, and writes `out` = contiguous
 * BF16 [3072, N], the wo_ud input (invalid columns and idle rows exact zero). Forms as gqa_attention A1:
 * a decode / verify block (B rows of W <= 16 columns) or a prefill segment (B = 1, W = its length).
 * Transient q / r and split partials come from `workspace`
 * (inkling_attention_workspace_capacity_bytes).
 */
void inkling_attention(const InklingProjections& projections, const InklingAttentionWeights& weights,
                       InklingLayerKind kind, const InklingAttentionRows& rows,
                       const InklingConvStateArgs& conv, const PagedKVBatchLayerView& cache,
                       WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/// Workspace of inkling_attention for every W in [min_width, max_width] at B rows under `envelope`.
[[nodiscard]] std::size_t inkling_attention_workspace_capacity_bytes(InklingLayerKind kind,
                                                                     std::int32_t batch,
                                                                     std::int32_t min_width,
                                                                     std::int32_t max_width,
                                                                     GqaExecutionEnvelope envelope,
                                                                     InklingKeyGroups key_groups = InklingKeyGroups::Default);

// ---- The component ops (op tests against lane a's vectors; also callable on their own). ------------

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
 * Global: each further bf16(value * tau(position)). Conv state per InklingConvStateArgs. Reads q, r, k,
 * v, k_dyn, v_dyn of `projections` (gate / sink unused) and q_scale, r_scale, k_conv, v_conv of `weights`.
 *
 * `cache` is the visit's plane (FP8 e4m3 K and V planes, no scale planes, head dim 128, 8 KV heads);
 * the op writes only the addressed slots. Launched with programmatic dependent launch: it triggers its
 * dependents at entry, so the attention that follows may prefetch the keys of earlier forwards.
 */
void inkling_attention_prologue(const InklingProjections& projections,
                                const InklingAttentionWeights& weights, InklingLayerKind kind,
                                const InklingAttentionRows& rows, const InklingConvStateArgs& conv,
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

/// Workspace for inkling_rel_attention / _out_lse / _train at B rows of W columns: the split route's partials, zero when
/// the call plans one worker. Exactly (N = W * B):
///   slots   = local: (1023 >> log2(local group)) + 2 (9 at 128-key groups);  global: group_of(max_visible_keys - 1) + 1
///             (P3: max_visible_keys <= 65,536 -> ceil(keys / 512); then + ceil((keys - 65,536) / 2,048) up to 131,072,
///             then + ceil((keys - 131,072) / 8,192); 176 at 262,144)
///   bytes   = workers > 1 ? slots * N * 24 * (128 * 4 + 8) + 512 + (fused ? 32 * B + 256 : 0) : 0
///   fused   = W <= 16 (decode / verify) && workers > 1 && W * slots <= InklingAttentionKnobs::fused_fold_max (the last
///             worker CTA folds; its 8 arrival counters per row live here)
///   workers = decode / verify (W <= 16): clamp(floor(2 * SMs / (8 B)), 1, global 64 | local slots);
///             prefill: 1 when tiles * 24 * B >= 2 * SMs (tiles = ceil(W / 64)), else clamp(ceil(2 SMs / CTAs), 1, min(32, slots));
///             1 whenever bytes would exceed the partial cap: decode / verify InklingAttentionKnobs::partial_cap_bytes
///             (default 1 GiB), prefill min(that, prefill_partial_cap_bytes (default 64 MiB)). So a prefill call's
///             partials are <= 64 MiB (+ 512): inkling_attention_workspace_capacity_bytes(kind, 1, 1, 8192, 262k) =
///             max(qr(8192) = 56.6 MB, max_w qr(w) + partials(w)) <= ~65 MiB.
/// The plan (workers, cap) changes only the speed and this size, never the output bits. The call must use the same
/// envelope and key_groups as this query. Throws outside the supported domain.
[[nodiscard]] std::size_t inkling_rel_attention_workspace_bytes(InklingLayerKind kind,
                                                                std::int32_t width,
                                                                std::int32_t batch,
                                                                GqaExecutionEnvelope envelope,
                                                                InklingKeyGroups key_groups = InklingKeyGroups::Default);

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
 * Training entry (agent-EN 11:3x; lane g1b's engine-op IR): THE production attention -- same plan, kernels and per-row
 * arithmetic as inkling_rel_attention on every route -- writing in the same pass the gated output `out` (contiguous
 * BF16 [3072, N]), the pre-gate output `o` (contiguous FP32 [128, 24, N]) and the natural `lse` (contiguous FP32
 * [24, N]); invalid columns and idle rows are zero in all three. Bitwise: out == inkling_rel_attention's, o / lse ==
 * inkling_rel_attention_out_lse's on the same rows (test: invariance ... train entry).
 */
void inkling_rel_attention_train(const InklingAttentionInputs& inputs, InklingLayerKind kind,
                                 const InklingAttentionRows& rows, const PagedKVBatchLayerView& cache,
                                 WorkspaceArena& workspace, Tensor& out, Tensor& o, Tensor& lse, cudaStream_t stream);

/// What the fused training call exposes besides its gated output: the prologue's q_out (BF16 [128, 24, N]) and r_out
/// (BF16 [16, 24, N]), both contiguous and written by the prologue itself; tau (FP32 [N], global layers: the
/// prologue's own values; local layers: unused, may be empty); o and lse as inkling_rel_attention_train.
struct InklingAttentionTrainOutputs {
    Tensor q;
    Tensor r;
    Tensor tau;
    Tensor o;
    Tensor lse;
};

/// inkling_attention (prologue + attention + gate / sink) for the trainer: the same calls, the prologue's q / r / tau
/// kept, o and lse written by the same final pass as `out`. Needs no workspace for q / r (the caller's tensors).
void inkling_attention_train(const InklingProjections& projections, const InklingAttentionWeights& weights,
                             InklingLayerKind kind, const InklingAttentionRows& rows,
                             const InklingConvStateArgs& conv, const PagedKVBatchLayerView& cache,
                             WorkspaceArena& workspace, Tensor& out, InklingAttentionTrainOutputs& train,
                             cudaStream_t stream);

/**
 * DSpark draft block attention (agent-EN-e2b, 13:35; ref/dspark_ref.py DSparkRef.attention): q BF16 [128, 24, N]
 * contiguous (N = W * B; W = the 8 block columns of a lane), keys from the draft plane (paged unit-scale FP8 e4m3, the
 * block's own K / V already written there at [P_b, P_b + 8)). Every column of a lane passes the block's last position
 * P_b + 7 in rows.positions, so all its rows see one key set. window = 0: global; else keys with pos - key < window
 * (1,032 on draft layers 0 / 2 = keys >= P_b - 1,024). Scale 128^-0.5; no relative bias, gate, sink or tau. out = BF16
 * [128, 24, N] contiguous, o rounded once; invalid columns and idle rows zero. The same per-row arithmetic, key groups
 * (rows.key_groups) and fold as inkling_rel_attention: batch / width / split invariant.
 */
void inkling_draft_attention(const Tensor& q, std::int32_t window, const InklingAttentionRows& rows,
                             const PagedKVBatchLayerView& draft_plane, WorkspaceArena& workspace, Tensor& out,
                             cudaStream_t stream);
[[nodiscard]] std::size_t inkling_draft_attention_workspace_bytes(std::int32_t window, std::int32_t width,
                                                                  std::int32_t batch, GqaExecutionEnvelope envelope,
                                                                  InklingKeyGroups key_groups = InklingKeyGroups::Default);

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

// ---- The FP8 attention projections (D6; agent-EN 08:2x: lane c's, per-channel and 1x128 cells first). ------

/**
 * One projection of a visit: out[0:rows, n] = W[row_begin : row_begin + rows, :] . x[:, n], rows = out.ne[0].
 * x: contiguous, 16-byte aligned BF16 [K, N]. weight: an FP8 attention object as lane b binds it -- the block
 * route's Weight (QuantLayout::Fp8Block128, e4m3 codes [n][k], fp32 scale grid, cell scale_ne = (k, 1)
 * per-channel / (128, 1) 1x128 / (128, 128) 128x128). out: BF16 [rows, N], dim 0 contiguous, any column stride
 * (a slice of a larger output). BF16 weights (the oracle path) are not taken here: ops::linear / linear_rows.
 */
struct InklingProjection {
    const Tensor* x        = nullptr;
    const Weight* weight   = nullptr;
    std::int32_t row_begin = 0;
    Tensor* out            = nullptr;
};

/// Zero today: both kernels stage in shared memory only.
[[nodiscard]] std::size_t inkling_attn_linear_workspace_bytes(
    std::span<const InklingProjection> projections, std::int32_t max_columns);

/**
 * Op inkling_attn_linear (D6): one to four projections sharing K (2304 qkvr / 3072 wo_ud) and N in ONE launch
 * (a 3-input visit: the q-side rows from x_q, K from x_k, V from x_v). W8A16: codes widened exactly, BF16
 * activations as they are, fp32 accumulation; each scale cell's partial multiplied by its fp32 scale as the cell
 * ends (per-channel: once, after the whole row); output rounded once to bf16. N <= 8: a GEMV (decode, verify);
 * N > 8: a tensor-core GEMM. Projections of one GEMM launch share their cell kind (per-channel or not).
 */
void inkling_attn_linear(std::span<const InklingProjection> projections, WorkspaceArena* workspace,
                         cudaStream_t stream);

} // namespace sinfer::ops
