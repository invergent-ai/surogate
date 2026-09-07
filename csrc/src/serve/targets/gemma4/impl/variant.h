#pragma once

#include "targets/gemma4/impl/config.h"
#include "targets/gemma4/impl/load/bindings.h"
#include <api/family/runtime.h>
#include <api/family/text_geometry.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::targets::gemma4::detail {

using GraphExecutionProfile = family::GraphExecutionProfile;

// Compile-time data and the closed execution leaves supplied to the family runtime. It owns
// no request state, execution phase, graph object, or schedule callback.
//
// Gemma 4 is a dense GQA stack whose every layer attends and whose attention output carries
// no gate. Inside the block it is Gemma 3's shape -- four sandwich norms, three unfused
// attention projections, a gated-GELU MLP with separate gate and up -- plus four things
// Gemma 3 does not have, all of them in the leaves below:
//
//   * the value is RMS-normalised with no weight, and on a global layer it *is* the key
//     projection's raw output (`attention_k_eq_v`);
//   * the block's whole output is multiplied by a per-layer scalar;
//   * the two kinds of layer attend at different head widths, so the geometry a leaf reads
//     depends on which layer it is running;
//   * the logits are squashed to +-30 before sampling.
//
// The linear-mixer and MTP leaves are declared because the runtime is a template over this
// interface, not because they can run: each one throws, naming why (see variant.cpp).
struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::AttentionProjectionPayload;
    using GdnProjectionWeights           = detail::GdnProjectionPayload;
    using PostMixerWeights               = detail::DensePostMixerPayload;
    using MtpAttentionProjectionWeights  = detail::MtpAttentionPayload;
    using MtpPostMixerWeights            = detail::DensePostMixerPayload;
    using VisionWeights                  = family::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    /// **One.** Gemma 4 sets `self.scaling = 1.0` and relies on its QK-norm to deliver
    /// unit-RMS queries and keys, so there is no `1/sqrt(head_dim)` divisor at all. That is
    /// also why a single constant serves both of this model's head widths, where a derived
    /// scale would have had to be per layer.
    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    /// Rows of the compacted proposal head. This target has no draft head at
    /// all, and the family reaches this constant only when
    /// `proposal_head == Optimized`, which the plan validator already ties to an
    /// enabled speculative backend -- which `bind_artifact` refuses here. The
    /// value names the only head this model has, so the unreachable branch would
    /// at least be sized rather than zero.
    static constexpr std::int32_t draft_head_rows              = TextConfig::output_rows;

    /// Read by the family runtime through `attention_output_gate<Variant>()`.
    /// The hybrid targets declare nothing and get the family default (true): the
    /// attention output is multiplied by sigmoid(gate). Gemma 3 writes no gate
    /// rows -- its projection is `[query][key][value]` and nothing else -- so the
    /// multiply is skipped rather than applied to whatever the unwritten gate
    /// plane last held.
    static constexpr bool attention_output_gate = false;

    /// **False, and this is where Gemma 4 parts company with Gemma 3.**
    ///
    /// The family defaults this to true and the Gemma 3 target relies on that default by
    /// saying nothing, because Gemma 3 stores its RMSNorm weights zero-centred and applies
    /// them as `1 + w`. Gemma 4 does not: `Gemma4RMSNorm` initialises its weight to *ones*
    /// and applies `normed * w`. The checkpoint settles it -- `input_layernorm` averages
    /// 6.62 and the final norm reaches 604, nothing centred on zero -- and the declaration
    /// carries no `unfold_unit_offset` on any norm to match.
    ///
    /// Inheriting the default here would apply `1 + w` to a weight that is already the full
    /// scale. Nothing would raise; the model would emit fluent text and mean none of it.
    static constexpr bool norm_unit_offset = false;

    // `attention_qk_norm` is deliberately ABSENT, and its absence is the decision: the
    // family default (apply it) is what Gemma 4 wants. It normalises each head of q and k
    // before rope, like Gemma 3 and Qwen3 and unlike Llama, and the artifact binds
    // `attention/query_norm` and `attention/key_norm` for every layer at that layer's own
    // head width. Llama's `= false` would skip the step and leave two bound tensors that
    // nothing reads.

    /// Parity probe: under SUROGATE_SERVE_DUMP_RESIDUAL the family loop's attention
    /// intermediates are written out, tagged and numbered by the order the layers run in.
    /// A no-op unless the variable is set.
    static void debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream);

    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     family::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    /// The complete Gemma attention tail: `residual += post_attention_norm(o_proj(attention))`.
    ///
    /// Gemma 4 sandwiches its attention block, so the residual takes
    /// `post_attention_layernorm(o_proj(attention))` rather than `o_proj(attention)`.
    /// Only the payload overload is declared: the family's plain `(attention, weight,
    /// residual, ...)` form cannot reach the norm weight, and declaring it as well
    /// would leave `ResidualHooks::attention_output` a fallback that silently drops
    /// the norm if its `requires` clause ever stopped matching.
    /// Argument order follows `ResidualHooks::attention_norm`, where the payload
    /// comes after the tensors it belongs to.
    static void attention_output_projection(const Tensor& attention, const Weight& weight,
                                            const FullAttentionProjectionWeights& weights,
                                            Tensor& residual, family::TextPhase phase,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights,
                                         Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_kv_projection(const Tensor& hidden,
                                  const MtpAttentionProjectionWeights& weights, Tensor& key,
                                  Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, family::TextPhase phase,
                                     WorkspaceArena& workspace, cudaStream_t stream);
    static void
    gdn_input_projection_snapshot(const Tensor& hidden, const GdnProjectionWeights& weights,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_slot,
                                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& output_gate, family::TextPhase phase,
                                  WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection_record(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
        Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
        family::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                      family::TextPhase phase, WorkspaceArena& workspace,
                                      cudaStream_t stream);
    /// The short-convolution mixer's input: RMSNorm the residual, then one projection that
    /// yields B, C and x stacked. Only a target whose `linear_mixer` is `ShortConv` runs it;
    /// the rest refuse, exactly as a dense target refuses the delta net's leaves.
    static void short_conv_projection(const Tensor& residual, const Tensor& norm_weight, float eps,
                                      const GdnProjectionWeights& weights, Tensor& bcx,
                                      family::TextPhase phase, WorkspaceArena& workspace,
                                      cudaStream_t stream);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           family::TextPhase phase, WorkspaceArena& workspace,
                           cudaStream_t stream);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);
    [[nodiscard]] static std::size_t
    mtp_attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_kv_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                               std::int32_t last);
    [[nodiscard]] static std::size_t
    mtp_q_gate_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                  family::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                         family::TextPhase phase,
                                                         std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_input_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                  family::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_record_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                   family::TextPhase phase, std::int32_t first,
                                                   std::int32_t last);
    [[nodiscard]] static std::size_t
    short_conv_projection_workspace_capacity_bytes(const family::TextGeometry& geometry,
                                                   WeightsProfile weights_profile,
                                                   family::TextPhase phase, std::int32_t first,
                                                   std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_norm_control_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
                                        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                            std::int32_t last);

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    ordinary_graph_profiles(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                          std::uint32_t batch_size);
};

} // namespace sinfer::targets::gemma4::detail
