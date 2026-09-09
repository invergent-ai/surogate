#pragma once

#include "family/impl/moe/banked_experts.h"

#include <api/targets/qwen3_5_moe/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/text_geometry.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/vision.h>

#include "targets/qwen3_5_moe/impl/config.h"
#include "artifact/binder.h"
#include "artifact/typed_binding.h"
#include "artifact/materializer.h"
#include "family/impl/load/host_bank.h"

#include <memory>
#include "core/tensor.h"
#include "api/ops/sparse_moe.h"

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

namespace sinfer::targets::qwen3_5_moe::detail {


struct MoePlan {
    artifact::ObjectHandle router_shared_gate;
    /// Bound by stored format, not by the profile's expectation: a GGUF serves its routed
    /// experts as its own K-quant superblocks, and reading those through the groupwise-int
    /// codec the profile names would decode every expert wrongly.
    artifact::LinearBinding routed_gate_up;
    artifact::LinearBinding routed_down;
    artifact::ObjectHandle shared_gate_up;
    artifact::ObjectHandle shared_down;
    /// NVFP4 only: the format's second level plus the W4A4 runner's activation scale and
    /// epilogue alpha, per expert and per projection (see `ops::SparseMoeWeights`). All unset for
    /// a groupwise-int artifact, which has no such objects.
    std::optional<artifact::ObjectHandle> routed_gate_up_scale;
    std::optional<artifact::ObjectHandle> routed_down_scale;
    std::optional<artifact::ObjectHandle> routed_gate_up_act_scale;
    std::optional<artifact::ObjectHandle> routed_gate_up_alpha;
    std::optional<artifact::ObjectHandle> routed_down_act_scale;
    std::optional<artifact::ObjectHandle> routed_down_alpha;
};

/// The attention input projection as one object per HF Linear. A quantized export stores
/// q/k/v that way, each with its own global scale, and there is no honest way to fuse two
/// of those; q and gate are the two halves of HF's q_proj, gathered per head.
struct SplitAttentionPlan {
    artifact::LinearBinding query;
    artifact::LinearBinding key;
    artifact::LinearBinding gate;
    artifact::LinearBinding value;
};

struct FullAttentionPlan {
    /// Exactly one of the two is set: the fused parent the groupwise-int artifacts carry, or
    /// the split the export stores. Each weight's format is whatever the artifact declares.
    std::optional<artifact::LinearBinding> query_key_gate_value;
    std::optional<SplitAttentionPlan> split;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::LinearBinding output;
};

struct SplitGdnInputPlan {
    artifact::LinearBinding query_key_value;
    artifact::LinearBinding z;
};

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle a_b_projection;
    /// As for attention: the fused qkv|z parent, or in_proj_qkv and in_proj_z as stored.
    std::optional<artifact::LinearBinding> query_key_value_z;
    std::optional<SplitGdnInputPlan> split;
    artifact::ObjectHandle norm;
    artifact::LinearBinding output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MoePlan moe;
};

struct MtpPlan {
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    MoePlan moe;
    artifact::ObjectHandle final_norm;
};

struct DFlashLayerPlan {
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle attention_output;
    artifact::ObjectHandle post_attention_norm;
    artifact::ObjectHandle gate_up;
    artifact::ObjectHandle down;
};

struct DFlashPlan {
    artifact::ObjectHandle feature_projection;
    artifact::ObjectHandle context_norm;
    std::vector<DFlashLayerPlan> layers;
    artifact::ObjectHandle final_norm;
};

struct BindingPlan {
    /// The dimensions bound against: the compiled config with the artifact's
    /// `geometry` member laid over it.
    family::TextGeometry geometry;
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;
    /// Which weight formats the artifact carries; decided by the identity, read by both the
    /// binder (which object formats to expect) and the loader (which Weights to build).
    WeightsProfile weights = WeightsProfile::GroupwiseInt;
    artifact::LinearBinding token_embedding;
    std::vector<TextLayerPlan> text_layers;
    artifact::ObjectHandle final_norm;
    artifact::LinearBinding output_head;
    artifact::LinearBinding draft_head; // format read from the artifact
    artifact::ObjectHandle draft_head_token_ids;
    MtpPlan mtp;
    bool has_mtp = false;
    family::VisionGeometry vision_geometry;
    family::VisionBackbonePlan vision_backbone;
    bool has_vision = false;
    family::VisionMergerInputPlan vision_merger_input;
    artifact::LinearBinding vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    family::VisionMergerNormPlan vision_merger_norm;
    DFlashPlan dflash;
    // Artifacts converted without the DFlash drafter checkpoint omit the
    // dflash/* objects; the DFlash backend requires has_dflash.
    bool has_dflash = false;

    /// Objects this stage put in pinned host memory instead of on the card.
    family::HostBankPlan host_bank;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

family::TextGeometry resolved_geometry(const artifact::Reader& reader);

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, family::StartupFeatures features,
                               WeightsProfile weights, std::uint32_t host_moe_layers = 0,
                               std::uint32_t gpu_layers = 0, LoadProgress progress = {});

struct SparseMoePayload {
    family::BankedExperts banked;
    ops::SparseMoeWeights op;
};

struct SplitAttentionWeights {
    Weight query;
    Weight key;
    Weight gate;
    Weight value;
};

struct AttentionProjectionPayload {
    std::int32_t head_dim = 0;
    /// The fused parent when `split` is empty; unused otherwise.
    Weight query_key_gate_value;
    std::optional<SplitAttentionWeights> split;
};

struct SplitGdnInputWeights {
    Weight query_key_value;
    Weight z;
};

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    Weight a_b_projection;
    /// The fused qkv|z parent when `split` is empty; unused otherwise.
    Weight query_key_value_z;
    std::optional<SplitGdnInputWeights> split;
};

using RuntimeModelView =
    family::ModelView<AttentionProjectionPayload, GdnProjectionPayload, SparseMoePayload,
                       AttentionProjectionPayload, SparseMoePayload,
                       family::DFlashWeights>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;
using DFlashWeights        = RuntimeModelView::DFlash;
using DFlashLayerWeights   = family::DFlashLayerWeights;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    /// Weights this stage kept in pinned host memory rather than on the card.
    std::shared_ptr<family::HostBank> host_bank;
    family::FrontendResources frontend;
    RuntimeModelView runtime;
};

SINFER_TARGET_LOADED_MODEL_IMPL();

} // namespace sinfer::targets::qwen3_5_moe::detail
