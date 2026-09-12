#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/common/math.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <array>
#include <cstdint>

namespace sinfer::ops {

/**
 * The closed geometry of one sparse-MoE family instance. The kernels are compiled per
 * registered geometry; the wrapper derives the geometry from the weights and refuses any
 * combination that is not registered.
 */
/// How a mixture turns router logits into the weights a token's experts are summed with.
///
/// The engine served one of these for as long as it served one family. GLM-5.3 uses the other:
/// scores are a sigmoid rather than a softmax, selection is on the score plus a learned
/// per-expert bias, and the winners are renormalised among themselves and scaled. The bias
/// steers *which* experts are chosen without changing what they are worth once chosen, which
/// is why it is added for the ranking and dropped for the weight.
enum class SparseMoeGating : std::uint8_t {
    /// Rank on the logits, softmax over the winners. Qwen3.5/3.6, Qwen3.8-Flash-Next, Qwen3-MoE.
    SoftmaxTopK,
    /// Rank on sigmoid(logit) + bias, renormalise the winners' sigmoid scores, scale.
    SigmoidBiasTopK,
};

struct SparseMoeGeometry {
    std::int32_t hidden            = 0;
    std::int32_t experts           = 0;
    std::int32_t experts_per_token = 0;
    std::int32_t intermediate      = 0;
    /// How the router weights a token's experts. See `SparseMoeGating`.
    SparseMoeGating gating = SparseMoeGating::SoftmaxTopK;
    /// What the renormalised routed weights are multiplied by. One for every mixture whose
    /// router does not say otherwise; GLM-5.3 states 2.5.
    float routed_scale = 1.0F;
    /// Whether the always-on expert is gated, which is a separate question from whether there
    /// is one. Qwen's shared expert is weighted by a sigmoid the router carries as one extra
    /// row; GLM-5.3's is added with weight one and its router has exactly one row per expert.
    /// Coupling the two would read a 288-row router as 287 experts and a gate.
    bool shared_gated = true;
    /// The always-on expert's FFN width, or zero where there is no always-on expert.
    ///
    /// Not every mixture has one. Qwen3-30B-A3B, LFM2-MoE, GPT-OSS and Gemma 4 route every
    /// token entirely, and this op assumed the opposite so thoroughly that it read the routed
    /// experts' width off the shared expert's `down` projection -- so a checkpoint without one
    /// had no width at all, not merely a missing path. Zero here means the mixture is routed
    /// and nothing else: no extra router row, one fewer path per token, no shared weights.
    std::int32_t shared_intermediate = 0;
    /// The bound both halves of every expert's SwiGLU are clamped to before the product, or
    /// zero for the unclamped product. GLM-5.3 states 10 per layer (`swiglu_clamp_exp`); it is
    /// a training-time stability device, so a checkpoint trained under it produces activations
    /// that reach it and serving without the clamp is serving a different function.
    float swiglu_limit = 0.0F;
    /// Which gate the experts put their first half through. SiLU for every mixture this engine
    /// served before Gemma 4, whose experts are GELU-gated (`gelu_pytorch_tanh`) like the rest
    /// of that model. Part of the geometry rather than the weights because the kernels compile
    /// it away, and part of the *identity* because a checkpoint served through the other gate
    /// computes a different function and says nothing about it.
    GatedActivation activation = GatedActivation::Silu;
    /// Whether the router weights its winners by a learned per-expert scale after the
    /// renormalisation. Gemma 4 does and nothing else here does. Derived from the weights, like
    /// `gating` is from the router bias: the tensor's presence is the statement.
    bool per_expert_scaled = false;

    [[nodiscard]] constexpr bool has_shared() const noexcept { return shared_intermediate > 0; }
    /// The router carries one row per expert, plus the shared expert's gate where it has one.
    [[nodiscard]] constexpr std::int32_t router_rows() const noexcept {
        return experts + (has_shared() && shared_gated ? 1 : 0);
    }
    [[nodiscard]] constexpr std::int32_t expert_rows() const noexcept { return 2 * intermediate; }
    [[nodiscard]] constexpr std::int32_t shared_rows() const noexcept {
        return 2 * shared_intermediate;
    }
    [[nodiscard]] constexpr std::int32_t routed_gate_rows() const noexcept {
        return experts * expert_rows();
    }
    [[nodiscard]] constexpr std::int32_t routed_down_rows() const noexcept {
        return experts * hidden;
    }
    /// How many expert outputs a token's result sums: its selected experts, and the shared one
    /// where there is one.
    [[nodiscard]] constexpr std::int32_t paths() const noexcept {
        return experts_per_token + (has_shared() ? 1 : 0);
    }

    friend constexpr bool operator==(const SparseMoeGeometry&, const SparseMoeGeometry&) = default;
};

/// Qwen3.5/3.6 MoE (35B-A3B and the 4B/2B MTP heads): 256 experts, top-8, FFN 512, hidden 2048,
/// plus a shared expert of the same width.
inline constexpr SparseMoeGeometry kSparseMoeQwen36Geometry{
    2048, 256, 8, 512, SparseMoeGating::SoftmaxTopK, 1.0F, true, 512};
/// Qwen3.8-Flash-Next: 512 experts, top-10, FFN 640, hidden 2560, shared expert of the same
/// width.
inline constexpr SparseMoeGeometry kSparseMoeFlashNextGeometry{
    2560, 512, 10, 640, SparseMoeGating::SoftmaxTopK, 1.0F, true, 640};
/// Qwen3-30B-A3B: 128 experts, top-8, FFN 768, hidden 2048, and no shared expert at all -- the
/// first registered mixture that routes every token entirely.
inline constexpr SparseMoeGeometry kSparseMoeQwen3MoeGeometry{
    2048, 128, 8, 768, SparseMoeGating::SoftmaxTopK, 1.0F, true, 0};
/// Qwen3/Qwen3-VL-235B-A22B use wider routed-only experts with the same routing rule.
inline constexpr SparseMoeGeometry kSparseMoeQwen3Moe235BGeometry{
    4096, 128, 8, 1536, SparseMoeGating::SoftmaxTopK, 1.0F, true, 0};
/// GLM-5.3-Flash: 288 experts, top-8, FFN 2048 with an always-on expert of the same width, and
/// the sigmoid-plus-bias router its checkpoint declares (`expert_gating_func` 2,
/// `expert_weights_scale` 2.5).
inline constexpr SparseMoeGeometry kSparseMoeGlm53Geometry{
    4096, 288, 8, 2048, SparseMoeGating::SigmoidBiasTopK, 2.5F, /*shared_gated=*/false, 2048,
    /*swiglu_limit=*/10.0F};
/// Gemma 4 26B-A4B: 128 experts, top-8, FFN 704, hidden 2,816, no always-on expert, and the two
/// things no other registered mixture has -- **GELU-gated** experts, and a router that scales
/// its renormalised winners by a learned per-expert vector.
///
/// Its dense feed-forward is not here and does not belong here: Gemma 4 runs one on *every*
/// layer, beside the experts rather than instead of them, so it is the layer's own projection
/// and not a shared expert of this mixture. `shared_intermediate` stays zero, which is what
/// says every token is routed and nothing is always on.
inline constexpr SparseMoeGeometry kSparseMoeGemma4Geometry{
    2816, 128, 8, 704, SparseMoeGating::SoftmaxTopK, 1.0F, /*shared_gated=*/true,
    /*shared_intermediate=*/0, /*swiglu_limit=*/0.0F, GatedActivation::GeluTanh,
    /*per_expert_scaled=*/true};

/// LFM2-MoE expert banks. Checkpoint metadata selects their dimensions.
inline constexpr SparseMoeGeometry kSparseMoeLfm2Moe32Geometry{
    2048, 32, 4, 1792, SparseMoeGating::SigmoidBiasTopK, 1.0F};
inline constexpr SparseMoeGeometry kSparseMoeLfm2Moe64Geometry{
    2048, 64, 4, 1536, SparseMoeGating::SigmoidBiasTopK, 1.0F};

/// Every mixture this op serves. One list, so registering a geometry is one line here and one
/// kernel-body instantiation per route rather than a predicate repeated in five places.
inline constexpr std::array<SparseMoeGeometry, 8> kSparseMoeGeometries{
    kSparseMoeQwen36Geometry, kSparseMoeFlashNextGeometry, kSparseMoeQwen3MoeGeometry,
    kSparseMoeQwen3Moe235BGeometry,
    kSparseMoeGlm53Geometry, kSparseMoeGemma4Geometry,
    kSparseMoeLfm2Moe32Geometry, kSparseMoeLfm2Moe64Geometry};

struct SparseMoeWeights {
    Weight router_shared_gate;
    /// Device FP32 [experts]: the per-expert bias a `SigmoidBiasTopK` router ranks with.
    ///
    /// Its presence is what says which router this mixture has -- a softmax router has no such
    /// tensor and a sigmoid one cannot select without it -- so the geometry's gating is derived
    /// from it rather than stated twice. A router that silently ignored it would choose
    /// different experts from the ones the checkpoint was trained to.
    const float* router_bias = nullptr;
    /// What the renormalised routed weights are multiplied by. Cannot be read off any shape, so
    /// the caller states it; it must match the registered geometry's.
    float routed_scale = 1.0F;
    /// Whether the always-on expert is weighted by a router row or added with weight one. Also
    /// unreadable from the shapes -- a 288-row router is 288 experts ungated or 287 and a gate.
    bool shared_gated = true;
    /// The SwiGLU clamp the mixture was trained under, or zero for none. Unreadable from any
    /// shape, like the two above; it must match the registered geometry's.
    float swiglu_limit = 0.0F;
    /// Which gate the experts put their first half through. Unreadable from any shape, like
    /// the clamp and the routed scale above; it must match the registered geometry's, and a
    /// mismatch is refused rather than served as a different function.
    GatedActivation activation = GatedActivation::Silu;
    /// Device FP32 [experts]: the learned scale each expert's renormalised routing weight is
    /// multiplied by, or null where the mixture has none.
    ///
    /// Its presence is what says the mixture has one, exactly as `router_bias`'s says which
    /// gating it uses. Applied *after* the winners are renormalised, so it changes what an
    /// expert is worth without changing which experts are chosen -- the opposite of what
    /// `router_bias` does, and the reason the two cannot share a field.
    const float* per_expert_scale = nullptr;
    Weight routed_gate_up;
    Weight routed_down;
    Weight shared_gate_up;
    Weight shared_down;
    /// Routed experts selected per token; with the weight shapes this fixes the geometry.
    std::int32_t experts_per_token = 0;
    /// Optional per-layer device table I32 [experts] mapping an expert id to the row block the
    /// routed weights hold it in (an expert slot cache); null means the routed weights are the
    /// resident experts in id order.
    const std::int32_t* slot_of_expert = nullptr;
    /// NVFP4 only, and required there: the format's second level. NVFP4 pairs an e4m3 scale per
    /// 16 values with a global scale that the checkpoint stores per expert and per projection, so
    /// stacking the experts into one routed weight leaves a factor that `Weight`'s single
    /// `weight_scale_divisor` cannot express. A factor constant over an expert comes out of the
    /// dot product, so these are applied once to the finished dot rather than by the codec to
    /// every weight — the codecs stay identical across formats.
    ///
    /// Device F32, indexed by *row block* (the expert, or its slot when `slot_of_expert` is set),
    /// and multipliers: the dequantised weight is `code * block_scale * scale[block]`. The
    /// checkpoint's `weight_global_scale` divides, so a converter writes its reciprocal.
    ///   `routed_gate_up_scale` — [blocks][2], gate then up, matching the row halves of a block.
    ///   `routed_down_scale`    — [blocks].
    /// Null for every other format, where the codec carries the whole scale.
    const float* routed_gate_up_scale = nullptr;
    const float* routed_down_scale    = nullptr;
    /// The W4A4 runner's contract for the `routed-nvfp4` profile, all [blocks] FP32 and required
    /// together with the two above when both routed weights are NVFP4. In that profile the two
    /// `routed_gate_up_scale` entries of a block are equal and its row halves run [up; gate].
    ///   `routed_gate_up_act_scale` — the checkpoint's `input_global_scale`, the multiplier applied
    ///                                to the rows before their e2m1 rounding (6 * 448 / amax);
    ///   `routed_gate_up_alpha`     — 1 / (act scale * weight global scale), the fc1 epilogue
    ///                                alpha that undoes both global scales after the MMA;
    ///   `routed_down_act_scale`, `routed_down_alpha` — the same for down.
    const float* routed_gate_up_act_scale = nullptr;
    const float* routed_gate_up_alpha     = nullptr;
    const float* routed_down_act_scale    = nullptr;
    const float* routed_down_alpha        = nullptr;
};

/// The geometry implied by the weights (router rows, hidden, shared-down width, top-k).
[[nodiscard]] SparseMoeGeometry sparse_moe_geometry(const SparseMoeWeights& weights);

/// The narrowest round the routed-NVFP4 profile serves through the vendored TRT-LLM runner;
/// below it the round stays on our own kernels. Only a single-token round is below it: swept at
/// 2, 4, 8, 16 and 47 on the 35B, everything from 2 to 16 measured the same (16 users:
/// 1,413 / 1,410 / 1,413 / 1,408 decode tok/s) and 47 gave up 21 % because a 16-user decode
/// round is 16 columns wide and fell short of it. At one token the runner's permute-group-reduce
/// scaffolding is the whole round and costs about half the throughput, so that width stays ours.
/// `SUROGATE_SERVE_MOE_TRTLLM_MIN` overrides it, so the crossover can be re-measured.
inline constexpr std::int32_t kSparseMoeTrtllmMinTokens = 2;

/// The widest round `sparse_moe` hands the routed-NVFP4 runner in one call, and therefore the
/// width `sparse_moe_prepare` has to tune up to.
inline constexpr std::int32_t kSparseMoeTrtllmPrepareWidth = 4096;

/**
 * Tunes and caches the routed-NVFP4 runner's grouped-GEMM tactics for every round width up to
 * `max_tokens`, timing the candidates on `weights`. Must run before any stream capture that
 * contains a sparse_moe round of that profile (a captured round of an untuned width throws).
 * A no-op for every other profile and for widths already tuned; the choice persists on disk.
 */
void sparse_moe_prepare(const SparseMoeWeights& weights, std::int32_t max_tokens,
                        cudaStream_t stream);

enum class SparseMoeEpilogue : std::uint8_t {
    AddResidual,
};

/**
 * Returns the transient capacity required by SparseMoe for every T in the inclusive
 * [min_tokens,max_tokens] interval. The routed QTypes are the fixed implementation profile.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(const SparseMoeGeometry& geometry,
                                                              QType routed_gate_up,
                                                              QType routed_down,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Closed sparse-MoE Op over the registered geometries (kSparseMoeQwen36Geometry,
 * kSparseMoeFlashNextGeometry); the text below names the Qwen3.6 instance.
 *
 * For contiguous BF16 x [2048,T] and destination [2048,T] with T>0, 256 routed experts, top-8
 * selection, and one always-on shared expert, this Op owns router projection and selection,
 * selected routed and shared SwiGLU projections, down projections, their merge, and the
 * AddResidual epilogue independently for every token column. At an exact top-8 boundary tie the
 * lower expert id wins. destination is the only observable mutation: its incoming value is the
 * residual and its outgoing value is the BF16 sparse-MoE result plus that residual.
 *
 * The complete mathematical oracle starts from represented BF16 inputs, exact stored-weight
 * decode, and evaluates the logical formula naively in FP32/FP64. Scores, route weights, expert
 * activations, workspace representation, reduction association, and scale placement are private
 * execution choices rather than semantic rounding boundaries.
 *
 * The five weights have the exact registered shapes: BF16 router/shared gate [257,2048], routed
 * gate/up [256*1024,2048], routed down [256*2048,512], shared gate/up [1024,2048], and shared down
 * [2048,512]. Admitted codec profiles are Q4+Q5, Q4+Q6, and W8+W8 for the two routed banks; both
 * shared banks are W8. Expert e directly selects its stored row spans; no selected-weight gather
 * or repack occurs.
 *
 * Every positive T is supported.
 *
 * x, destination, all weight planes, and live workspace must be pairwise non-overlapping.
 * Execution is enqueued on stream without host synchronization. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Round hook: called once the routing of a round is final (decode after the top-k selection,
 * small-T after its selection, prefill after select/count of every token slice) with the
 * device I32 expert ids of that round and their FP32 router weights (`alpha`, same
 * [experts_per_token, tokens] layout) on `stream`, before any expert kernel of the round is
 * launched. An expert slot cache resolves and gathers here; the kernels then read the routed
 * weights through `SparseMoeWeights::slot_of_expert`, and a path whose expert the table maps
 * to -1 contributes nothing (the hook's owner computes it elsewhere). The callback runs on the
 * host at launch time (also during graph capture) and must only enqueue work on `stream`.
 */
struct SparseMoeRoundHook {
    /// `x` and `destination` are the round's own columns ([hidden, tokens] BF16 views: one
    /// token in the decode loop, the slice otherwise) so the hook can compute part of the
    /// round elsewhere and add it into the same destination.
    void (*resolve)(void* context, const Tensor& ids, const Tensor& alpha, const Tensor& x,
                    Tensor& destination, cudaStream_t stream) = nullptr;
    void* context                                             = nullptr;
    /// A bounded cache can resolve only this many tokens together on the narrow path.
    /// Zero leaves the normal schedule unchanged.
    std::int32_t max_resolve_tokens = 0;
    /// Wide W8 rounds retain their routing and grouped activations while fetching experts
    /// in ranges. The callback installs a slot table containing only [first, first+count).
    /// Every range completes both projections before the next range reuses its slots.
    std::int32_t expert_batch_size = 0;
    void (*resolve_experts)(void* context, const Tensor& ids, std::int32_t first,
                            std::int32_t count, cudaStream_t stream) = nullptr;
};

/// As above with a round hook; a hook with a null `resolve` is the plain call.
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream,
                const SparseMoeRoundHook& hook);

/**
 * As above, with the router reading `router_x` where the experts read `x`.
 *
 * Every mixture here but one projects both from the same tokens, and passing the same tensor
 * twice is exactly the call above. Gemma 4 does not: its router reads the post-attention
 * residual normalised with *no weight at all* and scaled by a learned per-channel vector, while
 * its experts read the same residual through `pre_feedforward_layernorm_2`. They are two
 * different projections of one token, and folding either norm into the other's weights would
 * either quantise a per-channel factor into a group-scaled bank or hide the router the
 * checkpoint states -- so the op takes both.
 *
 * `router_x` has `x`'s shape and dtype and must be disjoint from every other buffer here.
 */
void sparse_moe(const Tensor& x, const Tensor& router_x, const SparseMoeWeights& weights,
                SparseMoeEpilogue epilogue, Tensor& destination, WorkspaceArena& workspace,
                cudaStream_t stream, const SparseMoeRoundHook& hook);

} // namespace sinfer::ops
