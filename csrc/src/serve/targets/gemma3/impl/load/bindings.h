#pragma once

#include <api/targets/gemma3/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/text_geometry.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/vision.h>

#include "targets/gemma3/impl/config.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sinfer::targets::gemma3_270m::detail {

inline constexpr std::size_t kTextLayers          = 18;
inline constexpr std::size_t kFullAttentionLayers = 18;
// Every layer is full attention. The empty half of the family's split is not a
// placeholder: the shared ModelView is instantiated with it, and the shared
// runtime's GDN arrays and state pool are sized from it.
//
// "Full attention" here is the *mixer* axis, and it is easy to confuse with the
// other one: Gemma 3 alternates windowed against global attention on a schedule
// (`TextConfig::is_windowed_attention`). Both kinds are ordinary GQA layers
// storing identical objects, so the schedule never reaches this constant.
inline constexpr std::size_t kGdnLayers = 0;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

/// Gemma 3's MLP keeps gate and up separate, where the Qwen families and Llama
/// fuse them into one `mlp/gate_up`. The declaration says so
/// (`MLPConfig(fuse_gate_up=False)` on `surogate/dsl/blocks/gemma3.py`), and it
/// is not only a storage choice: two separate weights are two separate LoRA
/// bank keys, so `gate_proj` and `up_proj` adapters are servable here where the
/// fused targets have to refuse them.
///
/// `post_feedforward_norm` is the second half of the FFN sandwich: it normalises
/// the down projection's output *before* it reaches the residual, which is why
/// it travels with the MLP payload rather than sitting in the family's
/// `FullLayer` (whose one norm slot is already the pre-mixer norm).
struct MlpPlan {
    WeightPlan gate;
    WeightPlan up;
    WeightPlan down;
    artifact::ObjectHandle post_feedforward_norm;
};

/// Gemma 3 attention is UNGATED -- there are no output-gate rows anywhere -- and
/// UNFUSED: query, key and value are three artifact objects, where the Qwen
/// families and Llama store one `attention/query_key_value`. The declaration
/// gives the reason (`surogate/dsl/blocks/gemma3.py`): per-head QK norm and rope
/// both want their operand contiguous, and in a fused `[q|k|v, tokens]` matrix
/// the query rows of successive tokens are not adjacent. Three GEMMs are the
/// same FLOPs and compose from ops that already exist.
///
/// It is normalised per head, like Qwen3 and unlike Llama, so `query_norm` and
/// `key_norm` are bound and `Variant::attention_qk_norm` stays at the family
/// default (true).
///
/// `post_attention_norm` is Gemma's *sandwich* norm -- HF's
/// `post_attention_layernorm`, which normalises the attention block's output
/// before the residual add, not the MLP's input. It is NOT the family's
/// `FullLayer::post_attention_norm`, which is a pre-mixer norm and takes
/// Gemma's `pre_feedforward_norm` instead; see `bindings.cpp`.
struct AttentionPlan {
    WeightPlan query;
    WeightPlan key;
    WeightPlan value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
    artifact::ObjectHandle post_attention_norm;
};

/// Four norms per layer, against Llama's two. Gemma sandwiches each sub-block:
///
///   input_norm -> attention -> post_attention_norm -> (+= residual)
///   pre_feedforward_norm -> mlp -> post_feedforward_norm -> (+= residual)
///
/// The two named after the family's slots are the two the family already knows
/// how to apply; the other two ride in the payloads above.
struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    AttentionPlan attention;
    artifact::ObjectHandle pre_feedforward_norm;
    MlpPlan mlp;
};

struct BindingPlan {
    /// The dimensions bound against: the compiled config with the artifact's
    /// `geometry` member laid over it.
    family::TextGeometry geometry = family::TextGeometry::compiled<TextConfig>();
    /// Only four of the family plan's six slots are filled. gemma-3-270m-it
    /// publishes no image or video preprocessor config, and the loader refuses an
    /// artifact carrying an object no binder consumed -- so this target binds its
    /// own four rather than reusing `family::bind_frontend_resources`, which
    /// would demand two objects the artifact does not have.
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;

    WeightPlan token_embedding;
    /// One per layer, sized when the artifact is bound rather than by the type.
    std::vector<TextLayerPlan> text_layers;
    artifact::ObjectHandle final_norm;
    /// Gemma 3 ties its LM head to the embedding table -- the checkpoint ships no
    /// `lm_head.weight` at all, and `_build_gemma3_mappings` resolves `lm_head`
    /// to `embed_tokens.weight`. The converter stores the one table and names
    /// `text/output_head` a logical role on it, so on a tied artifact this holds
    /// `token_embedding`'s handle rather than a second binding; see
    /// `bindings.cpp`. It is an ordinary binding only for an artifact converted
    /// from an untied export, which stores a head of its own.
    WeightPlan output_head;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features);

struct DensePostMixerPayload {
    Weight gate;
    Weight up;
    Weight down;
    Tensor post_feedforward_norm;
};

struct AttentionProjectionPayload {
    Weight query;
    Weight key;
    Weight value;
    /// Gemma's sandwich norm for the attention output. Carried here because this
    /// payload is the only per-layer object the family hands to a Variant leaf on
    /// the attention side; see `Variant::attention_output_projection`.
    Tensor post_attention_norm;
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
    family::ModelView<AttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
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

} // namespace sinfer::targets::gemma3_270m::detail
