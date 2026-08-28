#pragma once

#include <api/targets/qwen4exp/package.h>
#include <api/targets/qwen3_6/runtime.h>

#include "api/ops/gated_rmsnorm.h"
#include "core/arena.h"
#include "core/ngram_ple_state.h"
#include "core/tensor.h"
#include "targets/qwen4exp/impl/config.h"
#include "targets/qwen4exp/impl/load/bindings.h"
#include "targets/qwen3_6/impl/runtime/prologue_columns.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen4exp::detail {

using GraphExecutionProfile = qwen3_6::GraphExecutionProfile;

// The closed leaves of the qwen3_6 family runtime for Qwen3.8-Flash-Next. The residual the
// family hands in is the four-stream hyper-connection residual; the norm hooks mix it into a
// block input and remember the inject gates, the output projections scatter the block output
// back, and the layer prologue adds the n-gram memory at its layer.
struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::AttentionProjectionPayload;
    using GdnProjectionWeights           = detail::GdnProjectionPayload;
    using PostMixerWeights               = detail::SparseMoePayload;
    using MtpAttentionProjectionWeights  = detail::AttentionProjectionPayload;
    using MtpPostMixerWeights            = detail::SparseMoePayload;
    using VisionWeights                  = qwen3_6::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    // The GDN output gate is a logistic sigmoid (`output_gate_type: sigmoid`), not SiLU.
    static constexpr ops::GatedRmsGate gdn_output_gate         = ops::GatedRmsGate::Sigmoid;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    static constexpr std::int32_t draft_head_rows              = 131072;

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    ordinary_graph_profiles(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                          std::uint32_t batch_size);

    // --- residual hooks (ResidualHooks<Variant> probes for these) ---
    static void embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                               WorkspaceArena& workspace, cudaStream_t stream);
    static void final_residual_mix(const ModelView& model, const Tensor& residual, Tensor& hidden,
                                   WorkspaceArena& workspace, cudaStream_t stream);
    static void attention_norm(const Tensor& residual, const FullAttentionProjectionWeights& weights,
                               Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream);
    static void post_mixer_norm(const Tensor& residual, const PostMixerWeights& weights,
                                Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream);
    /// Creates the device scratch the mix/combine pair shares; call before any graph capture.
    static void prewarm_device_scratch();
    /// Number of expert slots the cache on the current device should hold (0 disables); read
    /// when the cache is created in prewarm_device_scratch. SUROGATE_SERVE_EXPERT_SLOTS is the
    /// fallback when nothing was configured.
    static void configure_expert_slots(std::uint32_t slots);
    static constexpr bool has_layer_prologue = true;
    // 48 layers, a four-stream residual and the PLE nodes: the decode graphs measured
    // 15.7 MiB per lane (503 MB at 32 lanes) against the family's 12 MiB.
    static constexpr std::size_t ordinary_graph_allowance_per_lane_bytes = 20ULL * 1024ULL * 1024ULL;
    // Parity probe: dumps family-loop intermediates under SUROGATE_SERVE_DUMP_RESIDUAL.
    static void debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream);
    [[nodiscard]] static NgramPleStatePoolSpec ple_state_spec(std::int32_t slot_count);
    static void layer_prologue(const ModelView& model, int layer, Tensor& residual,
                               const qwen3_6::detail::PrologueColumns& columns,
                               NgramPleStatePool* ple_state, WorkspaceArena& workspace,
                               cudaStream_t stream);
    [[nodiscard]] static std::size_t layer_prologue_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last);

    // --- projections ---
    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    static void attention_output_projection(const Tensor& attention, const Weight& weight,
                                            Tensor& residual, qwen3_6::TextPhase phase,
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
                                     Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase phase,
                                     WorkspaceArena& workspace, cudaStream_t stream);
    static void
    gdn_input_projection_snapshot(const Tensor& hidden, const GdnProjectionWeights& weights,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_slot,
                                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& output_gate, qwen3_6::TextPhase phase,
                                  WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_input_projection_record(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
        Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
        qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
    static void gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                      qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                      cudaStream_t stream);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                           cudaStream_t stream);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);

    // --- workspace capacities ---
    [[nodiscard]] static std::size_t
    mtp_attention_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                                std::int32_t last);
    [[nodiscard]] static std::size_t
    mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase phase,
                                                         std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_record_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                   qwen3_6::TextPhase phase, std::int32_t first,
                                                   std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile, qwen3_6::TextPhase phase,
                                        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last);
};

} // namespace ninfer::targets::qwen4exp::detail
