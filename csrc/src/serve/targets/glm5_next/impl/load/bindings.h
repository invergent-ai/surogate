#pragma once

#include <api/targets/glm5_next/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/text_geometry.h>
#include <api/family/vision.h>
#include <api/ops/manifold_hyper_connection.h>
#include <api/ops/sparse_moe.h>

#include "targets/glm5_next/impl/config.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "family/impl/load/host_bank.h"

#include <memory>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace sinfer::targets::glm5_next::detail {

[[nodiscard]] inline ops::SparseMoeGeometry moe_geometry(const family::TextGeometry& g) {
    return {.hidden = g.hidden, .experts = g.experts, .experts_per_token = g.experts_per_token,
            .intermediate = g.intermediate, .gating = ops::SparseMoeGating::SigmoidBiasTopK,
            .routed_scale = g.routed_scale, .shared_gated = false,
            .shared_intermediate = g.shared_intermediate, .swiglu_limit = g.swiglu_limit};
}

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

/// One site's manifold hyper-connection: the projection that produces all three mixings, its
/// biases and the three scales they are multiplied by. Every layer carries two of these, one
/// before the mixer and one before the feed-forward.
struct HyperConnectionPlan {
    WeightPlan mix;
    artifact::ObjectHandle base;
    artifact::ObjectHandle scale;
};

/// Multi-head latent attention, NoPE. The query passes through its own low rank and the
/// key/value through theirs; the two halves of the expansion are separate objects because the
/// checkpoint stores one of them transposed and only the other can be read where it lies.
struct LatentAttentionPlan {
    WeightPlan query_a;
    artifact::ObjectHandle query_a_norm;
    WeightPlan query_b;
    WeightPlan kv_a;
    artifact::ObjectHandle kv_a_norm;
    WeightPlan k_b;
    WeightPlan v_b;
    WeightPlan output;
};

/// Kimi Delta Attention. One fused q|k|v, three depthwise tap planes stacked into one, and the
/// two low-rank gates -- the forget gate, whose bias and per-head decay complete it, and the
/// output gate the gated RMSNorm reads.
struct KdaPlan {
    WeightPlan query_key_value;
    artifact::ObjectHandle convolution;
    WeightPlan decay_a;
    WeightPlan decay_b;
    artifact::ObjectHandle decay_bias;
    artifact::ObjectHandle a_log;
    WeightPlan beta;
    WeightPlan gate_a;
    WeightPlan gate_b;
    artifact::ObjectHandle norm;
    WeightPlan output;
};

/// The feed-forward. A layer is dense or the mixture, never both, and which it is is the
/// artifact's to say -- so the plan carries a flag rather than two vectors whose correspondence
/// to layer numbers would have to be reconstructed.
struct FeedForwardPlan {
    bool sparse = false;
    WeightPlan gate_up;   ///< dense: [2 * dense_intermediate, hidden]
    WeightPlan down;      ///< dense: [hidden, dense_intermediate]
    WeightPlan router;
    artifact::ObjectHandle router_bias;
    WeightPlan routed_gate_up;
    WeightPlan routed_down;
    WeightPlan shared_gate_up;
    WeightPlan shared_down;
};

struct TextLayerPlan {
    bool attends = false;
    /// Whether this pipeline stage runs this layer. A stage still *binds* every layer -- the
    /// artifact's shapes are validated whole, and every object it holds has to be consumed by
    /// the target that reads it -- but uploads only its own.
    bool resident = true;
    HyperConnectionPlan attention_hc;
    artifact::ObjectHandle input_norm;
    LatentAttentionPlan attention;
    KdaPlan kda;
    HyperConnectionPlan feed_forward_hc;
    artifact::ObjectHandle post_attention_norm;
    FeedForwardPlan feed_forward;
};

/// The NextN draft head: one latent-attention layer over the mixture, on a single-stream
/// residual of its own, between a fold that seeds it from the next token's embedding and the
/// trunk's normalised hidden and a norm that reads it out for the trunk's LM head. No
/// hyper-connection anywhere in it, which is what makes it the family's fixed draft tail rather
/// than a trunk block. The embedding table and the LM head are the trunk's.
struct MtpPlan {
    bool present  = false; ///< the artifact carries a head
    bool resident = false; ///< ...and this run wants it on this device: `--spec mtp`, on the
                           ///< stage that holds the head (a whole model, or the last stage)
    artifact::ObjectHandle embedding_norm; // [hidden]
    artifact::ObjectHandle hidden_norm;    // [hidden]
    WeightPlan input_projection;           // [hidden, 2 * hidden]
    artifact::ObjectHandle input_norm;
    LatentAttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    FeedForwardPlan feed_forward;
    artifact::ObjectHandle final_norm;     // `shared_head.norm`
};

struct BindingPlan {
    family::TextGeometry geometry{};
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;

    /// Whether this stage reads the embedding table. Every stage validates it; only the first
    /// uploads it. `finishes` is always true today -- see `bind_artifact` for why the head is
    /// carried everywhere.
    bool embeds   = true;
    bool finishes = true;

    WeightPlan token_embedding;
    std::vector<TextLayerPlan> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;

    MtpPlan mtp;

    /// Objects this stage puts in pinned host memory instead of on the card. Empty unless
    /// `--host-moe-layers` asked for it.
    family::HostBankPlan host_bank;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features, int stage_first = 0,
                               int stage_last = 0, std::uint32_t host_moe_layers = 0,
                               std::uint32_t gpu_layers = 0,
                               family::BankPlanes bank_planes = family::BankPlanes::Auto,
                               LoadProgress progress = {});

/// Which layers of this artifact attend, read from the objects it holds rather than from a
/// number beside them: a layer carrying a latent key/value projection attends, one carrying
/// convolution taps does not. The schedule is already in the file, and asking the file is what
/// keeps it from having to be declared twice and agree.
[[nodiscard]] family::TextGeometry declared_geometry_with_schedule(const artifact::Reader& reader);

/// The mixings a site's hyper-connection produces, carried with the weights that produce them.
struct HyperConnectionPayload {
    ops::ManifoldHyperConnectionWeights weights;
    /// Which model layer these belong to. Carried so a refusal can name the layer rather than
    /// the shape alone: on a pipeline stage the interesting question about an unbound weight is
    /// always "which layer, and does this stage run it".
    std::int32_t layer = -1;
    std::int32_t probe_layer_count = 0;
    std::int32_t streams = 0;
    std::int32_t sinkhorn_iterations = 0;
    float epsilon = 0.0F;
};

/// The attention site's payload: its hyper-connection and the latent projections.
struct LatentAttentionPayload {
    HyperConnectionPayload hc;
    /// The layer's `input_layernorm`, applied to the collapsed stream. It is the same tensor
    /// the family holds beside this payload; the collapse hook receives only the payload, so
    /// the norm it applies afterwards travels with it.
    Tensor norm;
    float rms_epsilon = 0.0F;
    Weight query_a;
    Tensor query_a_norm;
    Weight query_b;
    Weight kv_a;
    Tensor kv_a_norm;
    Weight k_b;
    Weight v_b;
};

/// The mixer site's payload, in the slot the family names for the linear mixer's projection.
struct KdaProjectionPayload {
    HyperConnectionPayload hc;
    float rms_epsilon = 0.0F;
    /// The bound the forget gate's logistic is scaled by, from `kda.gate_lower_bound`.
    float gate_lower_bound = 0.0F;
    Weight query_key_value;
    Weight decay_a;
    Weight decay_b;
    Tensor decay_bias;
    Tensor a_log;
    Weight beta;
    Weight gate_a;
    Weight gate_b;
};

/// The feed-forward site's payload: its hyper-connection, and either a dense SwiGLU or the
/// mixture. `sparse` says which; the other half is left empty rather than filled with zeros.
struct FeedForwardPayload {
    HyperConnectionPayload hc;
    /// The layer's `post_attention_layernorm`, for the same reason `LatentAttentionPayload`
    /// carries its own.
    Tensor norm;
    float rms_epsilon = 0.0F;
    bool sparse = false;
    Weight gate_up;
    Weight down;
    ops::SparseMoeWeights moe;
    /// The layer's index and the model's layer count (draft head included): what keys the
    /// expert cache's directory when the mixture's experts are in the host bank.
    std::int32_t layer  = -1;
    std::int32_t layers = 0;
    /// Host addresses of the routed expert objects when the bank holds them (`moe` carries the
    /// device-mapped aliases); null for a layer whose experts are on the card. The expert cache
    /// reads the bank's planes through these on the host instead of fetching them over PCIe.
    const std::byte* host_gate_up = nullptr;
    const std::byte* host_down    = nullptr;
    /// What the bank holds each banked half as (`family::BankPlanes`): Q4 or Q5 planes are a
    /// base pointer and a shape, readable by the expert cache alone; anything else is what the
    /// Weight says. Under the default each half is at the narrowest width that loses nothing --
    /// this file: gate/up Q4_K at four bits, down Q5_K at six, the three Q6_K down halves W8.
    family::BankPlanes gate_up_planes = family::BankPlanes::Native;
    family::BankPlanes down_planes    = family::BankPlanes::Native;
};

/// The draft head's attention: the trunk's latent projections without the hyper-connection,
/// because the head runs on a single-stream residual. The same member names as
/// `LatentAttentionPayload`, so one loader fills either.
struct MtpAttentionPayload {
    float rms_epsilon = 0.0F;
    Weight query_a;
    Tensor query_a_norm;
    Weight query_b;
    Weight kv_a;
    Tensor kv_a_norm;
    Weight k_b;
    Weight v_b;
};

using RuntimeModelView =
    family::ModelView<LatentAttentionPayload, KdaProjectionPayload, FeedForwardPayload,
                      MtpAttentionPayload, FeedForwardPayload, family::DFlashWeights>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using KdaWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    /// Pinned host memory for the experts this stage did not upload. Shared process-wide, so
    /// eight pipeline stages of one model pin one copy rather than eight.
    std::shared_ptr<family::HostBank> host_bank;
    family::FrontendResources frontend;
    RuntimeModelView runtime;
};

SINFER_TARGET_LOADED_MODEL_IMPL();

} // namespace sinfer::targets::glm5_next::detail
