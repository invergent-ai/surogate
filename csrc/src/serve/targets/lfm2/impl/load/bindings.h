#pragma once

#include "family/impl/moe/banked_experts.h"

#include "family/impl/load/host_bank.h"

#include <api/targets/lfm2/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/text_geometry.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/vision.h>

#include "targets/lfm2/impl/config.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "api/ops/sparse_moe.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace sinfer::targets::lfm2::detail {

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

struct MlpPlan {
    bool sparse = false;
    WeightPlan gate_up;
    WeightPlan down;
    artifact::ObjectHandle router;
    artifact::ObjectHandle router_bias;
};

/// LFM2 attention is UNGATED: the fused projection carries `[query | key | value]` rows and
/// nothing else, where the hybrid family's carries `[query | key | output gate | value]`.
struct AttentionPlan {
    WeightPlan query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

/// The short-convolution mixer. One projection to B, C and x at once, the depthwise taps, and
/// one output projection -- and no norm between the convolution and the output, because the
/// gate C is applied inside the convolution rather than by a gated RMSNorm after it.
struct ConvPlan {
    WeightPlan in_projection;
    artifact::ObjectHandle convolution;
    WeightPlan out_projection;
};

/// A layer is one or the other. Which one is the artifact's to say, so the plan carries both
/// and a flag, rather than two vectors whose correspondence to layer numbers would have to be
/// reconstructed by whoever reads them.
struct TextLayerPlan {
    bool attends = false;
    artifact::ObjectHandle input_norm;
    AttentionPlan attention;
    ConvPlan convolution;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct BindingPlan {
    family::HostBankPlan host_bank;
    family::TextGeometry geometry = {};
    family::VisionGeometry vision_geometry;
    family::VisionBackbonePlan vision_backbone;
    artifact::ObjectHandle vision_post_norm_weight, vision_post_norm_bias;
    artifact::ObjectHandle projector_norm_weight, projector_norm_bias;
    artifact::LinearBinding projector_fc1, projector_fc2;
    artifact::ObjectHandle projector_fc1_bias, projector_fc2_bias;
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;

    WeightPlan token_embedding;
    std::vector<TextLayerPlan> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features);

/// Which layers of this artifact attend, read from the objects it holds rather than from a
/// number beside them: a layer that carries an attention output projection is an attention
/// layer, and one that carries convolution taps is not. The schedule is already in the file,
/// and asking the file is what keeps it from having to be declared twice and agree.
[[nodiscard]] family::TextGeometry declared_geometry_with_schedule(const artifact::Reader& reader);

struct DensePostMixerPayload {
    family::BankedExperts banked;
    Weight gate_up;
    Weight down;
    ops::SparseMoeWeights moe;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_value;
};

/// The linear mixer's projection, in the slot the family names for it. For this target it is
/// the short convolution's single input projection: [3 * hidden, hidden], holding B, C and x.
struct ShortConvProjectionPayload {
    Weight in_projection;
};

/// Declared so the family's `MtpWeights<...>` instantiates; never materialised, because
/// `bind_artifact` refuses speculation.
struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    family::ModelView<FusedAttentionProjectionPayload, ShortConvProjectionPayload,
                      DensePostMixerPayload, MtpAttentionPayload, DensePostMixerPayload,
                      family::DFlashWeights>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using ConvWeights          = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    std::shared_ptr<family::HostBank> host_bank;
    family::FrontendResources frontend;
    RuntimeModelView runtime;
};

SINFER_TARGET_LOADED_MODEL_IMPL();

} // namespace sinfer::targets::lfm2::detail
