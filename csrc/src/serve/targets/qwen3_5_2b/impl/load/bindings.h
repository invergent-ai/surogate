#pragma once

#include <api/targets/qwen3_5_2b/package.h>
#include <api/targets/qwen3_6/frontend_resources.h>
#include <api/targets/qwen3_6/model_view.h>
#include <api/targets/qwen3_6/startup_features.h>
#include <api/targets/qwen3_6/vision.h>

#include "../config.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>

namespace ninfer::targets::qwen3_5_2b::detail {

inline constexpr std::size_t kTextLayers          = 24;
inline constexpr std::size_t kFullAttentionLayers = 6;
inline constexpr std::size_t kGdnLayers           = 18;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format          = artifact::NumericFormat::BF16;
    std::uint32_t weight_scale_divisor_bits = 0;
    std::uint32_t input_scale_divisor_bits  = 0;
};

struct MlpPlan {
    WeightPlan gate_up;
    WeightPlan down;
};

struct SplitAttentionProjectionPlan {
    WeightPlan query_key;
    WeightPlan gate_value;
};

struct FusedAttentionProjectionPlan {
    WeightPlan query_key_gate_value;
};

struct FullAttentionPlan {
    std::variant<SplitAttentionProjectionPlan, FusedAttentionProjectionPlan> projection;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct SplitGdnInputProjectionPlan {
    WeightPlan query_key;
    WeightPlan value_z;
};

struct FusedGdnInputProjectionPlan {
    WeightPlan query_key_value_z;
};

struct SplitGdnControlProjectionPlan {
    WeightPlan a_projection;
    WeightPlan b_projection;
};

struct FusedGdnControlProjectionPlan {
    WeightPlan a_b_projection;
};

using GdnControlProjectionPlan =
    std::variant<SplitGdnControlProjectionPlan, FusedGdnControlProjectionPlan>;

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    GdnControlProjectionPlan control_projection;
    std::variant<SplitGdnInputProjectionPlan, FusedGdnInputProjectionPlan> input_projection;
    artifact::ObjectHandle norm;
    WeightPlan output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
};

struct MtpPlan {
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
    artifact::ObjectHandle post_attention_norm;
    MlpPlan mlp;
    artifact::ObjectHandle final_norm;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;

    WeightPlan token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    WeightPlan output_head;
    artifact::ObjectHandle draft_head;
    artifact::ObjectHandle draft_head_token_ids;
    // surogate vendor patch (PATCHES.md #15): artifacts from MTP-less GGUF
    // exports omit the mtp/* objects; speculation requires has_mtp.
    bool has_mtp = false;
    MtpPlan mtp;

    // Typed on this checkpoint's own tower (24 layers of 1024), not the family
    // default: the plan's layer array is sized by the config.
    qwen3_6::VisionBackbonePlanFor<VisionConfig> vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features);

struct DensePostMixerPayload {
    Weight gate_up;
    Weight down;
};

struct SplitAttentionProjectionPayload {
    Weight query_key;
    Weight gate_value;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_gate_value;
};

using FullAttentionProjectionPayload =
    std::variant<SplitAttentionProjectionPayload, FusedAttentionProjectionPayload>;

struct SplitGdnInputProjectionPayload {
    Weight query_key;
    Weight value_z;
};

struct FusedGdnInputProjectionPayload {
    Weight query_key_value_z;
};

using GdnInputProjectionPayload =
    std::variant<SplitGdnInputProjectionPayload, FusedGdnInputProjectionPayload>;

struct SplitGdnControlProjectionPayload {
    Weight a_projection;
    Weight b_projection;
};

struct FusedGdnControlProjectionPayload {
    Weight a_b_projection;
};

using GdnControlProjectionPayload =
    std::variant<SplitGdnControlProjectionPayload, FusedGdnControlProjectionPayload>;

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    GdnControlProjectionPayload control_projection;
    GdnInputProjectionPayload input_projection;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, qwen3_6::DFlashWeights<6>,
                       kFullAttentionLayers, kGdnLayers, VisionConfig>;
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
    qwen3_6::FrontendResources frontend;
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

} // namespace ninfer::targets::qwen3_5_2b::detail
