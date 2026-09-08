#pragma once

#include "targets/gemma3/impl/config.h"
#include "targets/gemma3/impl/load/bindings.h"
#include <api/family/runtime.h>
#include <api/family/text_geometry.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::targets::gemma3_270m::detail {

using GraphExecutionProfile = family::GraphExecutionProfile;

// Compile-time data and the closed execution leaves supplied to the Qwen3.6 family runtime.
// It owns no request state, execution phase, graph object, or schedule callback.
//
// Gemma 3 is a dense GQA stack: every layer is full attention and the attention
// output carries no gate. Where it parts company with Llama and Qwen3 is inside
// the block -- four norms instead of two, three attention projections instead of
// one fused parent, and a gated-GELU MLP whose gate and up stay separate -- all
// of which live in the leaves below rather than in the shared loop. The
// linear-mixer and MTP leaves are declared because the runtime is a template
// over this interface, not because they can run: each one throws, naming why
// (see variant.cpp).
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

    /// 1/sqrt(256) = 0.0625. Read that off `query_pre_attn_scalar: 256`, not off
    /// `head_dim`: the two coincide at this size and stop coinciding at others --
    /// Gemma3-27B has head_dim 128 against a scalar of 168 -- so a target that
    /// derived the scale from the head dim would be silently wrong the first time
    /// a second size is added. `config.h` states the value; this names its source.
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

    /// Read by the family runtime through `attention_output_gate<Variant>()`.
    /// The hybrid targets declare nothing and get the family default (true): the
    /// attention output is multiplied by sigmoid(gate). Gemma 3 writes no gate
    /// rows -- its projection is `[query][key][value]` and nothing else -- so the
    /// multiply is skipped rather than applied to whatever the unwritten gate
    /// plane last held.
    static constexpr bool attention_output_gate = false;

    // Two switches are deliberately ABSENT here, and their absence is the whole
    // decision. Both default to true in the family
    // (`family/impl/runtime/residual_policy.h`), and true is what Gemma wants:
    //
    //   * `norm_unit_offset` -- Gemma stores RMSNorm weights zero-centred and
    //     applies them as `1 + w` (`Gemma3RMSNorm.forward`), and the artifact
    //     holds the unfolded `w` (`transform="unfold_unit_offset"` on every norm
    //     the declaration lists). Qwen3 and Llama store the scale directly and so
    //     declare `= false`; copying that line here would make every norm apply
    //     `w` instead of `1 + w`, which leaves a model that still emits fluent
    //     text and means none of it.
    //
    //   * `attention_qk_norm` -- Gemma 3 normalises each head of q and k before
    //     rope, like Qwen3 and unlike Llama. The artifact binds
    //     `attention/query_norm` and `attention/key_norm`, so the family's
    //     default (apply it) is correct; Llama's `= false` would skip the step
    //     and leave two bound tensors nothing reads.

    /// Parity probe: under SUROGATE_SERVE_DUMP_RESIDUAL the family loop's
    /// attention intermediates are written out, tagged and numbered by the order
    /// the layers run in. A no-op unless the variable is set.
    static void debug_probe(const char* tag, const Tensor& tensor, std::int32_t layer_count, cudaStream_t stream);

    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     family::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    /// The complete Gemma attention tail: `residual += post_attention_norm(o_proj(attention))`.
    ///
    /// Gemma sandwiches its attention block, so the residual takes
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

} // namespace sinfer::targets::gemma3_270m::detail
