#pragma once

#include <api/targets/llama/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/text_geometry.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/vision.h>

#include "targets/llama/impl/config.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sinfer::targets::llama::detail {

inline constexpr std::size_t kTextLayers          = 22;
inline constexpr std::size_t kFullAttentionLayers = 22;
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

/// Llama attention is UNGATED: the fused projection carries
/// `[query | key | value]` rows and nothing else, where the hybrid family's
/// carries `[query | key | output gate | value]`. Naming the member for what it
/// holds is the whole guard against reading it at the family's stride.
///
/// It is also UNNORMALISED per head: Llama has no `q_norm`/`k_norm` weights, so
/// unlike the Qwen3 plan this one has no handle for them. The artifact carries
/// no such object either, and the loader refuses an object no binder consumed --
/// so binding one here would reject every correct Llama artifact.
struct AttentionPlan {
    WeightPlan query_key_value;
    WeightPlan output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    AttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct BindingPlan {
    /// The dimensions bound against: the compiled config with the artifact's
    /// `geometry` member laid over it.
    family::TextGeometry geometry = family::TextGeometry::compiled<TextConfig>();
    /// Only four of the family plan's six slots are filled. TinyLlama-1.1B
    /// publishes no image or video preprocessor config, and the loader refuses an
    /// artifact carrying an object no binder consumed -- so this target binds its
    /// own four rather than reusing `family::bind_frontend_resources`, which
    /// would demand two objects the artifact does not have.
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    /// TinyLlama clears `tie_word_embeddings` and ships `lm_head.weight`, so the
    /// head is a genuinely distinct matrix rather than a view of the embedding.
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
                       MtpAttentionPayload, DensePostMixerPayload, family::DFlashWeights<1>>;
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

SINFER_TARGET_LOADED_MODEL_IMPL();

} // namespace sinfer::targets::llama::detail
