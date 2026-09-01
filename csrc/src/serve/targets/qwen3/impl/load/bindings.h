#pragma once

#include <api/targets/qwen3/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sinfer::targets::qwen3::detail {

inline constexpr std::size_t kTextLayers          = 28;
inline constexpr std::size_t kFullAttentionLayers = 28;
// Every layer is full attention. The empty half of the family's split is not a
// placeholder: the shared ModelView is instantiated with it, and the shared
// runtime's GDN arrays and state pool are sized from it.
inline constexpr std::size_t kGdnLayers = 0;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

struct MlpPlan {
    WeightPlan gate_up;
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
    artifact::ObjectHandle input_norm;
    AttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct BindingPlan {
    /// Only four of the family plan's six slots are filled. Qwen3-0.6B publishes
    /// no image or video preprocessor config, and the loader refuses an artifact
    /// carrying an object no binder consumed -- so this target binds its own
    /// four rather than reusing `family::bind_frontend_resources`, which would
    /// demand two objects the artifact does not have.
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    /// Qwen3-0.6B sets `tie_word_embeddings`, but the converter stores the head
    /// as its own object rather than an alias, so this is an ordinary binding.
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
                       MtpAttentionPayload, DensePostMixerPayload, family::DFlashWeights<1>,
                       kFullAttentionLayers, kGdnLayers>;
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
    family::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace sinfer::targets::qwen3::detail
