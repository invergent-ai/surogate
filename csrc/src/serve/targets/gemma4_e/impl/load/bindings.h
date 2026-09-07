#pragma once

#include <api/targets/gemma4_e/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/text_geometry.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/vision.h>

#include "targets/gemma4_e/impl/config.h"
#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sinfer::targets::gemma4_e::detail {

/// Every layer is full attention -- "full attention" being the *mixer* axis, on which a
/// windowed Gemma layer and a global one are both ordinary attention. The empty GDN half
/// of the family's split is not a placeholder: the shared ModelView is instantiated with
/// it and the shared runtime sizes its state pool from it.
///
/// Neither count is a constant here, unlike `gemma3_270m`: this target serves a 48-layer
/// 12B and a 60-layer 31B, so the layer count comes from the bound artifact.
inline constexpr std::size_t kGdnLayers = 0;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

/// Gemma 4 keeps gate and up separate, as Gemma 3 does and the Qwen families do not
/// (`MLPConfig(fuse_gate_up=False)` on `surogate/dsl/blocks/gemma4.py`). Two separate
/// weights are two separate LoRA bank keys, so `gate_proj` and `up_proj` adapters are
/// servable here where the fused targets refuse them.
///
/// `post_feedforward_norm` is the second half of the FFN sandwich: it normalises the down
/// projection's output *before* it reaches the residual, which is why it travels with the
/// MLP payload rather than in the family's `FullLayer`.
///
/// `layer_scalar` closes the block. Gemma 4 multiplies the layer's whole output by it
/// (`hidden_states *= self.layer_scalar`), and it is a real number rather than a
/// formality: across the 12B's 48 layers it runs from 0.0053 to 0.918 and is never 1.
/// It rides with the MLP payload because the MLP's residual add is the last thing the
/// block does before it applies.
struct MlpPlan {
    WeightPlan gate;
    WeightPlan up;
    WeightPlan down;
    artifact::ObjectHandle post_feedforward_norm;
    artifact::ObjectHandle layer_scalar;
};

/// The per-token, per-layer input this layer folds in after its feed-forward.
///
/// Five objects, and two of them are this layer's slice of a tensor the checkpoint holds
/// stacked for the whole model: `embedding` is a column range of the second embedding table
/// and `input_projection` a row range of the projection that mixes the hidden state into it.
/// The converter cuts both, because every op that reads a slice needs it contiguous and a
/// slice of the stacked form is strided for any prompt longer than one token.
struct PerLayerInputPlan {
    WeightPlan embedding;
    WeightPlan input_projection;
    WeightPlan gate;
    WeightPlan projection;
    artifact::ObjectHandle norm;
};

/// Gemma 4 attention is UNGATED and UNFUSED, as Gemma 3's is: query, key and value are
/// separate artifact objects. Two things are new here, and both are per-layer:
///
///  * **Two head geometries.** A windowed layer's heads are `head_dim` wide over
///    `kv_heads` key/value heads; a global layer's are `global_head_dim` wide over
///    `global_kv_heads`. `windowed` says which this layer is, and it is read off the
///    artifact rather than compiled -- see `bind_artifact`.
///
///  * **`k_eq_v`: a global layer may ship no value projection at all.** Its value is the
///    *key projection's raw output*, before the key norm and before rope, RMS-normalised
///    with no learnable scale (`v_norm = RMSNorm(..., with_scale=False)`). So `value` is
///    empty on those layers and the value path re-reads the key rows. The 12B ships
///    `v_proj` on 40 of its 48 layers; the 8 without are exactly the global ones.
///
/// `post_attention_norm` is Gemma's *sandwich* norm -- HF's `post_attention_layernorm`,
/// which normalises the attention block's output before the residual add, not the MLP's
/// input. It is NOT the family's `FullLayer::post_attention_norm`, which is a pre-mixer
/// norm and takes Gemma's `pre_feedforward_norm` instead; see `bindings.cpp`.
struct AttentionPlan {
    WeightPlan query;
    WeightPlan key;
    /// Empty where the layer reuses its key projection as its value.
    WeightPlan value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
    artifact::ObjectHandle post_attention_norm;
    /// Whether this layer attends through the window, which decides its head geometry,
    /// its rope base and whether it carries a value projection.
    bool windowed = true;
    /// True when the value is this layer's key projection, normalised without a weight.
    bool value_is_key = false;
    /// Whether this layer projects keys and values at all. The E-series ends in a run of
    /// layers that do not: they hold a query projection and attend over an earlier layer's
    /// planes. Read off the artifact -- such a layer carries no `attention/key` object.
    bool owns_kv = true;
};

/// Four norms per layer, against Llama's two. Gemma sandwiches each sub-block:
///
///   input_norm -> attention -> post_attention_norm -> (+= residual)
///   pre_feedforward_norm -> mlp -> post_feedforward_norm -> (+= residual)
///   *= layer_scalar
///
/// **None of them is stored zero-centred**, which is where Gemma 4 parts company with
/// Gemma 3: `Gemma4RMSNorm` initialises its weight to ones and applies `normed * w` where
/// Gemma 3 applies `normed * (1 + w)`. The declaration therefore carries no
/// `unfold_unit_offset` on any of them and `Variant::norm_unit_offset` is false.
struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    AttentionPlan attention;
    artifact::ObjectHandle pre_feedforward_norm;
    MlpPlan mlp;
    PerLayerInputPlan per_layer_input;
};

struct BindingPlan {
    /// The dimensions bound against: the compiled config with the artifact's `geometry`
    /// member laid over it. For this target that member also carries the *second* head
    /// geometry and the window schedule, because neither can be compiled for two sizes.
    family::TextGeometry geometry = family::TextGeometry::compiled<TextConfig>();
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;

    WeightPlan token_embedding;
    /// One per layer, sized when the artifact is bound rather than by the type.
    std::vector<TextLayerPlan> text_layers;
    artifact::ObjectHandle final_norm;
    /// Every published Gemma 4 ties its head to its embedding, and the converter stores
    /// the one table and names `text/output_head` a logical role on it -- so on a tied
    /// artifact this holds `token_embedding`'s handle rather than a second binding; see
    /// `bindings.cpp`. It is an ordinary binding only for an artifact converted from an
    /// untied export.
    WeightPlan output_head;
    /// The norm over each layer's per-layer input projection. One vector for the whole model,
    /// where the two tensors it normalises are stored cut per layer.
    artifact::ObjectHandle per_layer_projection_norm;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

/// Declare on `geometry` which layers attend through the window, read off the artifact's own
/// objects: each layer's query norm is as wide as its head, 256 windowed against 512 global.
///
/// Called from **two** places, and it has to be: `bind_artifact` needs it to bind each layer's
/// weights at that layer's width, and `Package::declared_geometry` needs it because the KV pool
/// is sized from *that* geometry, before any binding happens. They agreed only by accident
/// while the compiled schedule matched the artifact's -- and the compiled array is one size's.
/// A 60-layer 31B indexes a 48-entry array; a two-layer fixture reads a schedule that is not
/// its own. Either way one half sizes a layer's cache for a geometry the other half does not
/// run it at, and `gqa_attention` is what notices.
void declare_attention_schedule(const artifact::Reader& reader, family::TextGeometry& geometry);

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features);

/// What a layer needs to build and fold in its per-layer input, as materialised weights.
struct PerLayerInputWeights {
    Weight embedding;
    Weight input_projection;
    Weight gate;
    Weight projection;
    Tensor norm;
    /// Shared with every layer: the norm over the projection, before it is combined with the
    /// embedding.
    const Tensor* projection_norm = nullptr;
    std::int32_t width = 0;
};

struct DensePostMixerPayload {
    Weight gate;
    Weight up;
    Weight down;
    Tensor post_feedforward_norm;
    PerLayerInputWeights per_layer_input;
    /// The per-layer scalar the block's output is multiplied by, both as the artifact stores
    /// it and as the leaf uses it. `ops::scale` takes a host float, and this factor is one
    /// number per layer that never changes, so it is read back once when the model loads
    /// rather than dereferenced on the device every round.
    Tensor layer_scalar;
    float layer_scalar_value = 1.0F;
};

struct AttentionProjectionPayload {
    Weight query;
    Weight key;
    /// Empty on a layer whose value is its key projection; `value_is_key` says so rather
    /// than leaving a caller to infer it from an empty weight, which is the kind of
    /// inference that reads a null pointer once a bug puts one there for another reason.
    Weight value;
    /// Gemma's sandwich norm for the attention output. Carried here because this payload
    /// is the only per-layer object the family hands to a Variant leaf on the attention
    /// side; see `Variant::attention_output_projection`.
    Tensor post_attention_norm;
    bool windowed     = true;
    bool value_is_key = false;
    bool owns_kv      = true;
    /// This layer's head width. Carried because the value norm is *per head*, so the leaf has
    /// to view a `[kv_heads * head_dim, columns]` plane as `[head_dim, kv_heads * columns]`,
    /// and a static leaf has no runtime config to ask.
    std::int32_t head_dim = 0;
};

/// The linear-mixer payload the shared ModelView still names. This target has no linear
/// layer, so no instance of it is ever constructed; it exists because the family's
/// `GdnWeights<Projection, PostMixer>` is a type, not an option.
struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
};

/// Likewise for the MTP head: declared so the family's `MtpWeights<...>` instantiates,
/// never materialised (`bind_artifact` refuses speculation).
///
/// Gemma 4 does publish a draft head -- the GGUF releases ship an `mtp-*.gguf` beside
/// every size -- but the safetensors checkpoint this target converts from does not carry
/// one, so there is nothing to bind yet.
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
    /// The norm every layer's per-layer input projection passes through. One vector, held
    /// here and pointed at by each layer, because the two tensors around it are cut per layer
    /// and this one is not.
    Tensor per_layer_projection_norm;
};

SINFER_TARGET_LOADED_MODEL_IMPL();

} // namespace sinfer::targets::gemma4_e_e::detail
