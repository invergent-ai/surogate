#pragma once

// Inkling-2-Mini's Variant: the compile-time data and leaves the family runtime is instantiated over
// (DESIGN-target.md 1). The family's layer loop never runs for this target: `visit_schedule` makes every
// TextContext loop hand its round to `run_visits`, which walks the 48 visits of `visits.h` over the
// AttnRes stack held in the residual (7 slots x 2304 per column). The GDN, MTP and per-layer leaves the
// runtime template names are declared because the template names them; each refuses (variant.cpp).

#include "targets/inkling2/impl/config.h"
#include "targets/inkling2/impl/model.h"
#include "family/impl/runtime/visit_round.h"

#include <api/family/runtime.h>
#include <api/family/text_geometry.h>
#include "core/linear_attention_state.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::targets::inkling2::detail {

using GraphExecutionProfile = family::GraphExecutionProfile;

struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::VisitPayload;
    using GdnProjectionWeights           = detail::NoGdnPayload;
    using PostMixerWeights               = detail::MoePayload;
    using MtpAttentionProjectionWeights  = detail::VisitPayload;
    using MtpPostMixerWeights            = detail::MoePayload;
    using VisionWeights                  = family::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    /// Plain RMSNorm weights (`y = x * rstd * w`), never `1 + w`.
    static constexpr bool norm_unit_offset = false;
    static constexpr bool applies_rotary   = false;

    // ---- the visit schedule (family/impl/runtime/visit_round.h) ----
    static constexpr bool visit_schedule = true;
    /// Logits stay FP32 after the softcap (agent-EN 08:2x; the reference's head_dtype float32).
    static constexpr bool logits_fp32 = true;
    /// `--kv-cache-dtype auto` resolves to FP8 e4m3 (unit scale) for this target.
    static constexpr bool kv_storage_fp8_default = true;

    /// Runs visits [round.first_visit, round.last_visit) over the round (DESIGN-target.md 1.2 / 1.4).
    static void run_visits(const ModelView& model, family::VisitRound& round);
    /// Workspace one `run_visits` call needs at the widest round of the phase.
    [[nodiscard]] static std::size_t visit_workspace_capacity_bytes(const family::TextGeometry& geometry,
                                                                    family::TextPhase phase,
                                                                    std::int32_t batch_size,
                                                                    std::int32_t max_width,
                                                                    std::int32_t max_columns,
                                                                    std::uint32_t max_visible_keys);
    /// The conv state: one "state layer" per visit, [K 1024 | V 1024 | attn 2304 | mlp 2304] x 3 taps.
    [[nodiscard]] static LinearAttentionStatePoolSpec visit_state_spec(const family::TextGeometry& geometry,
                                                                     std::int32_t slots);
    /// The KV plane windows (lane f1's hook): 1,024 for the 40 local visits, 0 for the 8 global ones.
    [[nodiscard]] static std::vector<std::int32_t> kv_layer_windows(const family::TextGeometry& geometry);

    // ---- residual hooks (ResidualHooks<Variant>) ----
    /// Slot 0 of the stack = embed_norm(embed[ids]); the other slots are written by the visits.
    static void embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                               WorkspaceArena& workspace, cudaStream_t stream);
    /// The final output mix: AttnRes over slots [0, 1, 2, 3] + slot 6 with the output query and the final norm.
    static void final_residual_mix(const ModelView& model, const Tensor& residual, Tensor& hidden,
                                   WorkspaceArena& workspace, cudaStream_t stream);
    /// logits FP32 [output_rows, T] = softcap30(hidden . unembed^T) in fp32 (`inkling_lm_head`).
    static void lm_head(const ModelView& model, const Tensor& hidden, Tensor& logits, cudaStream_t stream);
    static void prepare_lm_head(std::int32_t max_columns);

    // ---- closed leaves the runtime template names; each refuses (no family layer loop runs) ----
    static void attention_projection(const Tensor& hidden, const FullAttentionProjectionWeights& weights,
                                     Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                     family::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
    static void attention_output_projection(const Tensor& attention, const Weight& weight, Tensor& residual,
                                            family::TextPhase phase, WorkspaceArena& workspace,
                                            cudaStream_t stream);
    static void mtp_attention_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                         Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                  Tensor& key, Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_q_gate_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                      Tensor& query, Tensor& gate, WorkspaceArena& workspace,
                                      cudaStream_t stream);
    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights, Tensor& qkv,
                                     Tensor& output_gate, family::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    static void gdn_input_projection_snapshot(const Tensor& hidden, const GdnProjectionWeights& weights,
                                              const Tensor& conv_weight, Tensor& conv_states,
                                              const Tensor& valid_columns, const Tensor& initial_slot,
                                              const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                              Tensor& value, Tensor& output_gate, family::TextPhase phase,
                                              WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                            const Tensor& conv_weight, const Tensor& conv_states,
                                            const Tensor& valid_columns, const Tensor& initial_slots,
                                            Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                            Tensor& output_gate, family::TextPhase phase,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                      family::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
    static void short_conv_projection(const Tensor& residual, const Tensor& norm_weight, float eps,
                                      const GdnProjectionWeights& weights, Tensor& bcx, family::TextPhase phase,
                                      WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight, float eps,
                                            const GdnProjectionWeights& weights, Tensor& hidden, Tensor& g,
                                            Tensor& beta, WorkspaceArena& workspace, cudaStream_t stream);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           family::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights, Tensor& residual,
                               WorkspaceArena& workspace, cudaStream_t stream);

    [[nodiscard]] static std::size_t mtp_attention_projection_workspace_capacity_bytes(
        const family::TextGeometry& geometry, std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_kv_projection_workspace_capacity_bytes(
        const family::TextGeometry& geometry, std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_q_gate_projection_workspace_capacity_bytes(
        const family::TextGeometry& geometry, std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t attention_projection_workspace_capacity_bytes(
        const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t attention_output_projection_workspace_capacity_bytes(
        const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_workspace_capacity_bytes(
        const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
        std::int32_t batch_size, std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_record_workspace_capacity_bytes(
        const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
        std::int32_t batch_size, std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_output_projection_workspace_capacity_bytes(
        const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t short_conv_projection_workspace_capacity_bytes(
        const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_norm_control_projection_workspace_capacity_bytes(
        const family::TextGeometry& geometry, std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t post_mixer_workspace_capacity_bytes(
        const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(
        const family::TextGeometry& geometry, std::int32_t first, std::int32_t last);

    [[nodiscard]] static std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                                               std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window, std::uint32_t batch_size);
};

} // namespace sinfer::targets::inkling2::detail
