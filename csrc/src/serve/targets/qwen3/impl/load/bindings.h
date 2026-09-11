#pragma once

#include "family/impl/load/qwen3_vl.h"

#include "family/impl/load/host_bank.h"

#include <api/targets/qwen3/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/text_geometry.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/vision.h>

#include "targets/qwen3/impl/config.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sinfer::targets::qwen3::detail {

// Qwen3 and Qwen3-VL use full attention at every text layer.
inline constexpr std::size_t kGdnLayers = 0;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

struct MlpPlan {
    WeightPlan gate_up;
    WeightPlan gate;
    WeightPlan up;
    bool separate = false;
    WeightPlan down;
};

/// Qwen3 attention is UNGATED: the fused projection carries
/// `[query | key | value]` rows and nothing else, where the hybrid family's
/// carries `[query | key | output gate | value]`. Naming the member for what it
/// holds is the whole guard against reading it at the family's stride.
struct AttentionPlan {
    WeightPlan query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct TextLayerPlan {
    bool resident = true;
    artifact::ObjectHandle input_norm;
    AttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct BindingPlan : family::Qwen3VlVisionPlan {
    family::HostBankPlan host_bank;
    /// The dimensions declared by the checkpoint.
    family::TextGeometry geometry = {};
    /// Text resources and, for Qwen3-VL, image/video processor settings.
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;

    WeightPlan token_embedding;
    /// One per layer, sized when the artifact is bound rather than by the type.
    std::vector<TextLayerPlan> text_layers;
    artifact::ObjectHandle final_norm;
    /// The converter resolves tied embeddings and stores the output head explicitly.
    WeightPlan output_head;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features);

struct DensePostMixerPayload {
    Weight gate_up;
    Weight gate;
    Weight up;
    Weight down;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_value;
};

/// The linear-mixer payload the shared ModelView still names. This target has no
/// linear layer, so no instance of it is ever constructed; it exists because the
/// family's `GdnWeights<Projection, PostMixer>` is a type, not an option.
struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
};

/// Likewise for the MTP head: declared so the family's `MtpWeights<...>`
/// instantiates, never materialised (`bind_artifact` refuses speculation).
struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    family::ModelView<FusedAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, family::DFlashWeights>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
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

} // namespace sinfer::targets::qwen3::detail
