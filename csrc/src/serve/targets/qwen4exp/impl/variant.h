#pragma once

#include <api/targets/qwen4exp/package.h>
#include <api/family/runtime.h>
#include <api/family/text_geometry.h>

#include "api/ops/gated_rmsnorm.h"
#include "core/arena.h"
#include "core/ngram_ple_state.h"
#include "core/tensor.h"
#include "targets/qwen4exp/impl/config.h"
#include "targets/qwen4exp/impl/load/bindings.h"
#include "family/impl/runtime/prologue_columns.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::targets::qwen4exp::detail {

using GraphExecutionProfile = family::GraphExecutionProfile;

// The closed leaves of the shared family runtime for Qwen3.8-Flash-Next. The residual the
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
    using VisionWeights                  = family::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    // The GDN output gate is a logistic sigmoid (`output_gate_type: sigmoid`), not SiLU.
    static constexpr ops::GatedRmsGate gdn_output_gate         = ops::GatedRmsGate::Sigmoid;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    // QSA indexer (phase 4): the cache carries one BF16 plane of this width per full-attention
    // layer, and the selection engages only past TextConfig::dense_exact_context.
    static constexpr std::int32_t indexer_head_dim             = TextConfig::indexer_head_dim;
    static constexpr std::int32_t indexer_heads                = TextConfig::indexer_heads;
    static constexpr std::int32_t indexer_block                = TextConfig::indexer_block;
    static constexpr std::int32_t indexer_top_k                = TextConfig::indexer_top_k;
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
    /// `runtime_floor_bytes`: what the runtime must be left after the pool (KV floor and
    /// headroom); an automatic pool never sizes itself into it.
    static void configure_expert_slots(std::uint32_t slots, std::size_t runtime_floor_bytes = 0);
    /// The registry's projection of what the runtime derives from the resident weights, stashed
    /// at plan time so the pool can leave it.
    static void configure_derived_reserve(std::size_t bytes);
    [[nodiscard]] static std::size_t derived_reserve();
    /// Fraction of a round's missing experts the host computes (SUROGATE_SERVE_CPU_MOE_SHARE is
    /// the fallback when nothing was configured).
    static void configure_cpu_moe_share(float share);
    /// Minimum round width (columns) for the split; 0 keeps the default (4).
    static void configure_cpu_moe_min_tokens(std::uint32_t tokens);
    /// Share of a prefill round's misses computed on the host (batched kernel) and the widest
    /// prefill round (the engine's prefill chunk), which sizes the host staging.
    static void configure_cpu_moe_prefill(float share, std::uint32_t prefill_chunk);
    /// Host pools per NUMA node (pipeline stages) instead of one over all cores.
    static void configure_cpu_pool_per_socket(bool per_socket);
    /// With `--cpu-moe-share auto`, times a PCIe gather and a host round of layer 0's experts
    /// (outside graph capture) and sets the share to host/(host+pcie). Called by
    /// create_program before the graphs are captured; a no-op otherwise.
    static void prepare_expert_split(const ModelView& model);
    static constexpr bool has_layer_prologue = true;
    // 48 layers, a four-stream residual, the PLE nodes and (with the CPU split) the host-round
    // nodes per layer: the decode graphs measure 15.7-21.4 MiB per lane at 16 lanes and 31.4 MiB
    // per lane at 32 (the wider lanes cost more) against the family's 12.
    static constexpr std::size_t ordinary_graph_allowance_per_lane_bytes = 48ULL * 1024ULL * 1024ULL;
    // Parity probe: dumps family-loop intermediates under SUROGATE_SERVE_DUMP_RESIDUAL.
    static void debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream);
    [[nodiscard]] static NgramPleStatePoolSpec ple_state_spec(std::int32_t slot_count);
    static void layer_prologue(const ModelView& model, int layer, Tensor& residual,
                               const family::detail::PrologueColumns& columns,
                               NgramPleStatePool* ple_state, WorkspaceArena& workspace,
                               cudaStream_t stream);
    [[nodiscard]] static std::size_t layer_prologue_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last);

    // --- projections ---
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

    // --- NextN draft head ---
    //
    // The head's block is a trunk full-attention block, so the family runs it with the trunk's
    // own mixer rather than a second program. What is the head's own is the fold on the way in
    // and the collapse on the way out, and those are these two.
    //
    // Declaring this tells the family to take that route: the fixed Qwen3.5-shaped draft tail
    // does not describe this architecture, whose residual is four streams wide.
    static constexpr bool mtp_block_is_trunk_layer = true;

    /// residual = eh_proj( concat( embedding_norm(embedding), hidden_norm(hidden) ) ), per
    /// stream. `embedding` is [hidden, T]; `hidden` and `residual` are the wide residual
    /// [residual, T]. eh_proj holds the checkpoint's fc_embedding and fc_hidden side by side,
    /// so one matmul over the pair is fc_embedding@e + fc_hidden@h. The streams stay distinct
    /// through it -- pooling them first is exactly what the residual exists to avoid.
    static void mtp_fold(const ModelView& model, const Tensor& embedding,
                         const Tensor& hidden, Tensor& residual, WorkspaceArena& workspace,
                         cudaStream_t stream);
    /// The head's own mixer collapses the streams into the width the LM head reads. It stands
    /// in for the output norm this architecture does not have.
    static void mtp_collapse(const ModelView& model, const Tensor& residual, Tensor& hidden,
                             WorkspaceArena& workspace, cudaStream_t stream);
    /// The head's block, for the family to run.
    [[nodiscard]] static const detail::FullAttentionWeights& mtp_block(const ModelView& model);
    [[nodiscard]] static std::size_t
    mtp_fold_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                      std::int32_t last);

    // --- workspace capacities ---
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
};

} // namespace sinfer::targets::qwen4exp::detail
