#pragma once

#include "targets/glm5_next/impl/config.h"
#include <api/ops/gated_rmsnorm.h>
#include "targets/glm5_next/impl/load/bindings.h"
#include <api/family/runtime.h>
#include <api/family/text_geometry.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::targets::glm5_next::detail {

using GraphExecutionProfile = family::GraphExecutionProfile;

// Compile-time data and the closed execution leaves supplied to the family runtime. It owns no
// request state, execution phase, graph object, or schedule callback.
//
// GLM-5.3-Flash differs from every other target in this family at three places, and the leaves
// below are where each one lives. Its residual is four streams recombined by a doubly-stochastic
// matrix, so the two norm hooks collapse and the two output projections scatter. Its attention
// compresses the key and value to a 512-wide latent and expands it per head, with no rotary at
// all. Its non-attending layers run Kimi Delta Attention, whose forget gate is one value per key
// channel; the family's mixer dispatch handles that, and what is here is the projection that
// writes the wider gate. The MTP leaves are declared because the runtime is a template over this
// interface, not because they can run -- each one throws, naming why (see variant.cpp).
struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::LatentAttentionPayload;
    using GdnProjectionWeights           = detail::KdaProjectionPayload;
    using PostMixerWeights               = detail::FeedForwardPayload;
    using MtpAttentionProjectionWeights  = detail::MtpAttentionPayload;
    using MtpPostMixerWeights            = detail::FeedForwardPayload;
    using VisionWeights                  = family::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    /// Rows of the compacted proposal head. This target has no draft head at all; the value
    /// names the only head this model has, so the unreachable branch is at least sized.
    static constexpr std::int32_t draft_head_rows              = TextConfig::output_rows;

    /// The non-attending layers run Kimi Delta Attention: the delta net's recurrence with the
    /// forget gate per key channel. Read through `linear_mixer<Variant>()`.
    static constexpr family::LinearMixer linear_mixer = family::LinearMixer::KimiDelta;

    /// The latent attention writes no gate rows, so the multiply is skipped rather than applied
    /// to whatever the unwritten gate plane last held.
    static constexpr bool attention_output_gate = false;

    /// The attention normalises its two low ranks, not its heads: there is no `query_norm` or
    /// `key_norm` object to bind and no per-head normalisation to apply.
    static constexpr bool attention_qk_norm = false;

    /// NoPE. The checkpoint states `rope.dimension_count` 0 and carries no rotary split; the
    /// step is skipped rather than run at zero width.
    static constexpr bool applies_rotary = false;

    /// The delta recurrence gates its output with the logistic sigmoid, as Flash-Next does.
    static constexpr ops::GatedRmsGate gdn_output_gate = ops::GatedRmsGate::Sigmoid;

    /// Device memory reserved per decode lane for the ordinary graphs. The family's 12 MiB is
    /// sized for a residual as wide as the model; this one carries four streams of it, and the
    /// buffers a captured round bakes scale with that, so the measured requirement is 14 MiB.
    static constexpr std::size_t ordinary_graph_allowance_per_lane_bytes = 24ULL * 1024 * 1024;

    /// GLM-5.3 stores RMSNorm scales directly, not zero-centred: the scale is `w`, not `1 + w`.
    static constexpr bool norm_unit_offset = false;

    /// The residual is `hc_streams` copies of the model width. The two entry points and the
    /// final collapse are this target's, and the family reads them through `ResidualHooks`.
    static void embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                               WorkspaceArena& workspace, cudaStream_t stream);
    static void final_residual_mix(const ModelView& model, const Tensor& residual, Tensor& hidden,
                                   WorkspaceArena& workspace, cudaStream_t stream);
    /// Collapse the streams into the attention block's input, keeping the mixings its output
    /// projection scatters with.
    static void attention_norm(const Tensor& residual,
                               const FullAttentionProjectionWeights& weights, Tensor& hidden,
                               WorkspaceArena& workspace, cudaStream_t stream);
    /// The same for the feed-forward site.
    static void post_mixer_norm(const Tensor& residual, const PostMixerWeights& weights,
                                Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream);
    /// Device scratch the mixings are kept in between a site's collapse and its scatter.
    static void prewarm_device_scratch();


    /// Parity probe: under SUROGATE_SERVE_DUMP_RESIDUAL the family loop's intermediates are
    /// written out, tagged and numbered by the order the layers run in. A no-op otherwise.
    static void debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream);

    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     family::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    static void attention_output_projection(const Tensor& attention, const Weight& weight,
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
    /// Declared because the runtime is a template over this interface; this target's mixer is
    /// not a short convolution, so it throws.
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

} // namespace sinfer::targets::glm5_next::detail
