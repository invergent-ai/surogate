#pragma once

// Inkling-2-Mini (inkling2_loop) routed mixture: the five lane-d ops of OPS.md section 3
// (`inkling_route`, `inkling_quant_nvfp4_4o6`, `inkling_quant_mxfp8`, `inkling_softcap_swiglu`,
// `inkling_moe`) in one header. Design and numerics: rl/engine-inkling-2026-11-21/lanes/d/DESIGN-moe.md.
// The semantic definition of every op is the python function of the same name in
// rl/engine-inkling-2026-11-21/ref/inkling_ref.py, which follows the vLLM image's sm_120 kernels
// (ref/src/vllm/models/inkling2/nvidia/ops/{moe,nvfp4,mxfp8,activation,moe_sm120}.py).
//
// This op is deliberately NOT a `sparse_moe` geometry. The registered `sparse_moe` mixtures share a
// codec-generic kernel family whose arithmetic follows the round's width (static activation scales,
// a tactic per width); Inkling's experts are W4A4 / W8A8 with PER-TOKEN activation quantisation and
// a softcapped SwiGLU, and its reference fixes every rounding point. One arithmetic serves every
// width here: a token's bits do not depend on how many tokens share its round, on whether an adapter
// request shares it, or on the kernel family (decode / small-T / wide) that ran it.
//
// Tensor convention (as every serve op): `ne[0]` is the contiguous (feature) dimension, so a
// [features, tokens] tensor is row-major [tokens][features].

#include "core/arena.h"
#include "core/tensor.h"
#include "api/ops/lora.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops {

/// The one geometry this op serves (`ref/config.json` text_config). Fixed at compile time.
struct InklingMoeGeometry {
    static constexpr std::int32_t kHidden       = 2304;
    static constexpr std::int32_t kExperts      = 128;
    static constexpr std::int32_t kTopK         = 8;
    static constexpr std::int32_t kIntermediate = 1152;
    /// w13 rows per expert: gate and up INTERLEAVED (row 2j = gate j, row 2j + 1 = up j), the
    /// checkpoint's order and the artifact default (`moe.gate_up_rows = "interleaved"`).
    static constexpr std::int32_t kGateUpRows   = 2 * kIntermediate;
    static constexpr float kRouteScale          = 8.0F;
    static constexpr float kGateSoftcap         = 4.0F;
    static constexpr float kUpSoftcap           = 25.0F;
    /// The router's logits are the in-order fp32 sum of 1024-wide K-group partials
    /// (`ops/linear.py` `_k_group_tile`): groups [0,1024), [1024,2048), [2048,2304).
    static constexpr std::int32_t kRouterKGroup = 1024;
    /// The widest LoRA rank the fused adapter epilogues take (the RL recipe is r16).
    static constexpr std::int32_t kMaxAdapterRank = 64;
};

enum class InklingExpertFormat : std::uint8_t {
    /// Layers 0-19: W4A4. e2m1 codes, e4m3 block scale per 16, per-expert global amax / den.
    Nvfp4,
    /// Layers 20-23: W8A8. e4m3 codes, e8m0 scale per 32 along K, per ROW (`mxfp8-k32-v1`).
    Mxfp8,
};

/// One projection of all 128 experts, stacked by rows: expert e owns rows [e * N, (e + 1) * N),
/// N = 2304 for both projections (w13: kGateUpRows; w2: kHidden). K = 2304 (w13) or 1152 (w2).
/// Pointers are the artifact objects' planes as bound (ARTIFACT.md 4.2 / 4.4), never copies.
struct InklingExpertBank {
    /// Nvfp4: [E*N][K/2] bytes, two e2m1 per byte, LOW nibble = even column.
    /// Mxfp8: [E*N][K] e4m3fn bytes, row-major.
    const std::uint8_t* codes = nullptr;
    /// Nvfp4: e4m3 block scales in `blockscale-k16-m128x4-v1` over the stacked rows (128-row x
    ///        4-group tiles of 512 bytes; 2304 = 18 x 128, so no tile straddles two experts).
    /// Mxfp8: e8m0 exponents [E*N][K/32], row-major (the file's per-row scales).
    const std::uint8_t* scales = nullptr;
    /// Nvfp4 only: device FP32 [E], the RAW `*_weight_amax` (never pre-divided).
    const float* amax = nullptr;
    /// Nvfp4 only: device FP32 [1], the RAW `*_weight_outer_scale_denominator` (2688.0 today).
    const float* outer_den = nullptr;
};

/// One MoE layer as bound from the artifact (`text/layers/{L}/moe/*`).
struct InklingMoeWeights {
    InklingExpertFormat format = InklingExpertFormat::Nvfp4;
    /// BF16 [128][2304] (`moe/router`). Also the key the LoRA store's bank table is registered
    /// under for this layer (one weight set per layer; every pass of a core layer shares it).
    const void* router = nullptr;
    /// FP32 [128] (`moe/router_bias`): the selection bias (ranking only, never in the weights).
    const float* router_bias = nullptr;
    InklingExpertBank w13;
    InklingExpertBank w2;
};

/// The per-layer MoE payload lane e1's ModelView carries (`InklingMoePayload`, DESIGN-target 4).
using InklingMoePayload = InklingMoeWeights;

/// Routed-expert capture for routing replay (R3): one row per VISIT. With the runtime's capture
/// buffer (`io.routing_capture`, [served layers][columns][8]; Inkling's served layers are the 48
/// visits): ids = base + visit * columns * 8, capacity_tokens = columns, first_token = the call's
/// first column, token_stride = 8. Token t's ids (global
/// expert ids, selection order = descending sigmoid(logit) + bias, ties to the lower id, as
/// vLLM's `routed_expert_ids`) land at `ids + (first_token + t) * token_stride + [0, 8)`, for
/// t < live tokens and first_token + t < capacity_tokens. `token_stride` >= 8 lets the runtime
/// interleave visits ([token][visit][8]: stride 48 * 8) or keep one plane per visit (stride 8).
/// Written by the routing kernel itself (no extra launch); graph-capturable.
struct InklingMoeCapture {
    std::int32_t* ids            = nullptr;
    std::int32_t capacity_tokens = 0;
    std::int32_t first_token     = 0;
    std::int32_t token_stride    = InklingMoeGeometry::kTopK;
};

/// Expert LoRA (both formats; the deltas live on the BF16 sides of the quantised GEMMs).
///   gate_up[row, 2j]   += (B_gate[e] . (A_gate[e] . x))[j]       (fp32, before the bf16 round)
///   gate_up[row, 2j+1] += (B_up[e]   . (A_up[e]   . x))[j]
///   down[row, :]       += B_down[e] . (A_down[e] . y)            y = the bf16 SiTU output (gamma in)
/// `table` is the LoRA store's device bank table for `weights.router` in the `bind_lora_moe`
/// order: [0] router, [1] bias/none, then (gate, up, down) at [2 + 3e + {0,1,2}] (ports
/// 100 + 3e + {0,1,2}); each LoraBank has alpha/r folded into A (lora.h). A token whose slot is
/// negative is a base token: its rows get no delta and keep the base bits exactly.
struct InklingMoeAdapters {
    const LoraBank* table          = nullptr; ///< device; null = no adapter in this round
    const std::int32_t* token_slot = nullptr; ///< device I32 [T]; null with uniform_slot
    const std::int32_t* uniform_slot = nullptr; ///< device I32 [1]: one slot for the whole round
    std::int32_t max_rank          = 0;       ///< the store's padded rank (<= kMaxAdapterRank)
};

/// Test / debug taps: device buffers the call copies its intermediates into, by SLOT row
/// (token * 8 + slot; only routed slots are written). Null fields are skipped. Not for serving.
struct InklingMoeTaps {
    std::int32_t* ids     = nullptr; ///< I32 [T * 8]
    float* gammas         = nullptr; ///< FP32 [T * 8]
    std::uint8_t* y_codes = nullptr; ///< the re-quantised SiTU output: NVFP4 [T*8][576] | MXFP8 [T*8][1152]
    std::uint8_t* y_scales = nullptr; ///< its block scales, row-major: NVFP4 [T*8][72] | MXFP8 [T*8][36]
    float* y_amax         = nullptr; ///< NVFP4 only: FP32 [T * 8]
    void* down            = nullptr; ///< BF16 [T * 8][2304]: the down projection's rows
    void* gate_up         = nullptr; ///< BF16 [T * 8][2304]: the bf16 gate_up rows (interleaved), as the SiTU reads them
    void* y               = nullptr; ///< BF16 [T * 8][1152]: the SiTU output before re-quantisation
};

struct InklingMoeOptions {
    /// Device I32 [1]: the round's live token count (<= x's T). Tokens at or past it are padding:
    /// routed to no expert (they pull no expert bytes), output 0, not captured. Null = all T live.
    /// Lets one captured decode / verify graph serve fewer live tokens than its width.
    const std::int32_t* live_tokens = nullptr;
    const InklingMoeAdapters* adapters = nullptr;
    /// Explicit capture row for this call (the runtime passes the visit's row). When null, the
    /// thread's `ScopedSparseMoeCapture` target (api/ops/sparse_moe.h), if any, is used with
    /// token_stride 8 and first_token 0.
    const InklingMoeCapture* capture = nullptr;
    /// Test / debug only (see InklingMoeTaps).
    const InklingMoeTaps* taps = nullptr;
};

/// Transient workspace one `inkling_moe` call of up to `max_tokens` tokens needs (graph-stable,
/// caller-owned, no state across calls). `adapters` adds the LoRA low-rank buffers.
[[nodiscard]] std::size_t inkling_moe_workspace_bytes(InklingExpertFormat format,
                                                      std::int32_t max_tokens, bool adapters);

/**
 * out = MoE(x) for one layer visit (OPS.md `inkling_moe`; ref `inkling_moe(act_quant=True)`).
 *
 * x: BF16 [2304, T] contiguous (the visit's `mlp_in`); out: BF16 [2304, T] contiguous, written
 * (not accumulated): the fp32 combine of the token's 8 bf16 down rows in selection order,
 * rounded once to bf16. The router reads x too. T > 0; x, out, weights and workspace disjoint.
 *
 * Arithmetic (every rounding point of the reference, DESIGN-moe.md section 2):
 *   route: logits = (p0 + p1) + p2 over the 1024-wide K groups, each p a chain of bf16
 *          m16n8k16 MMAs from zero; key = sigmoid + bias; top-8; weights = renormalised
 *          sigmoids x 8 (the reference's log-space form and 8-term sum order).
 *   Nvfp4: x per token -> NVFP4 four-over-six (global amax / 1536); gate_up = bf16(acc *
 *          ((amax_x * amax_w13[e]) * rcp_rn(1536 * den13))) with acc the m16n8k64 block-scaled
 *          MMA chain from zero over K; y = bf16(((4 tanh(g/4) * sigmoid(g)) * 25 tanh(u/25)) *
 *          gamma) (the reference's PTX approximations); y per row -> NVFP4 four-over-six;
 *          down = bf16(acc * ((amax_y * amax_w2[e]) * rcp_rn(1536 * den2))).
 *   Mxfp8: x per 32-block -> e4m3 + e8m0 (round-up power-of-two scale); gate_up = bf16(acc),
 *          acc the bf16 m16n8k16 MMA chain over dequantised operands (weights = A, tokens = B,
 *          as the reference); y as above; y per 32-block -> MXFP8; down likewise.
 * Enqueued on `stream` without host synchronisation; graph-capturable for a fixed T.
 */
void inkling_moe(const Tensor& x, const InklingMoeWeights& weights, Tensor& out,
                 WorkspaceArena& workspace, cudaStream_t stream,
                 const InklingMoeOptions& options = {});

// ---- Sub-ops (OPS.md names). Each is the exact stage `inkling_moe` runs, exposed for the op
// ---- tests against the golden vectors and for the trainer lanes' cross-checks.

/// Router: x BF16 [2304, T] -> logits FP32 [128, T], ids I32 [8, T] (selection order),
/// weights FP32 [8, T] (x 8.0).
void inkling_route(const Tensor& x, const InklingMoeWeights& weights, Tensor& logits, Tensor& ids,
                   Tensor& route_weights, cudaStream_t stream);

/// Per-row NVFP4 "four-over-six" (`ops/nvfp4.py`): x BF16 [K, R] (K % 16 == 0) -> codes U8 [K/2, R]
/// (low nibble = even column), scales U8 (e4m3) [K/16, R] row-major, amax FP32 [R].
void inkling_quant_nvfp4_4o6(const Tensor& x, Tensor& codes, Tensor& scales, Tensor& amax,
                             cudaStream_t stream);

/// Per-32-block MXFP8 (`ops/mxfp8.py`): x BF16 [K, R] (K % 32 == 0) -> codes U8 (e4m3) [K, R],
/// exponents U8 (e8m0) [K/32, R].
void inkling_quant_mxfp8(const Tensor& x, Tensor& codes, Tensor& exponents, cudaStream_t stream);

/// Softcapped SwiGLU with the routing weight (`ops/activation.py` `_situ`): gate_up BF16
/// [2304, R] interleaved (column 2j = gate j, 2j + 1 = up j), gamma FP32 [R] -> y BF16 [1152, R].
void inkling_softcap_swiglu(const Tensor& gate_up, const Tensor& gamma, Tensor& y,
                            cudaStream_t stream);

} // namespace sinfer::ops
