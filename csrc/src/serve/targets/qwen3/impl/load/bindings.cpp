#include "targets/qwen3/impl/load/bindings.h"

#include "targets/qwen3/impl/config.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sinfer::targets::qwen3::detail {
namespace {

using artifact::NumericFormat;

/// Rows of the fused attention input projection: `[query | key | value]`.
/// The hybrid family's is `2 * query_size + 2 * kv_size` because it fuses an
/// output gate; Qwen3 has none, and this constant is the difference.
constexpr std::int32_t kAttentionInputRows = TextConfig::query_size + 2 * TextConfig::kv_size;
constexpr std::int32_t kMlpGateUpRows      = 2 * TextConfig::intermediate;

static_assert(kAttentionInputRows == 4096);
static_assert(TextConfig::query_projection_rows == TextConfig::query_size,
              "Qwen3 attention is ungated; a gated projection would be read at the wrong stride");

/// The four resources a text-only Qwen3 artifact carries. The family's binder
/// demands six; the two it adds are the image and video preprocessor configs,
/// which Qwen3-0.6B does not publish. The unfilled halves of the plan stay
/// default-constructed and `take_text_only_frontend_resources` leaves their
/// strings empty, which is what the frontend reads as "no pixel pipeline".



NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("qwen3: invalid weights profile");
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, kMlpGateUpRows, TextConfig::hidden);
    out.down = materialized_weight(materialized, plan.down, TextConfig::hidden,
                                   TextConfig::intermediate);
    return out;
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile, BindingPlan& out) {
    const NumericFormat weights = endpoint_format(weights_profile);
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(
            binder, prefix + "input_norm", NumericFormat::BF16, {TextConfig::hidden});
        target.attention.query_key_value =
            bind_weight(binder, prefix + "attention/query_key_value", weights,
                        {kAttentionInputRows, TextConfig::hidden});
        // Qwen3 normalises each head of q and k (`use_qk_norm`), over head_dim,
        // and carries no qkv bias (`attention_bias: false` in every released
        // Qwen3 config), which is why no bias object is bound here.
        target.attention.query_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
        target.attention.key_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
        target.attention.output = bind_weight(binder, prefix + "attention/output", weights,
                                              {TextConfig::hidden, TextConfig::query_size});
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
        target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", weights,
                                         {kMlpGateUpRows, TextConfig::hidden});
        target.mlp.down    = bind_weight(binder, prefix + "mlp/down", weights,
                                         {TextConfig::hidden, TextConfig::intermediate});
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    // The checkpoint's own dimensions, where it states them: absent members keep the
    // target's compiled value, so an artifact written before the member existed binds
    // exactly as it did.
    out.geometry = family::TextGeometry::declared<TextConfig>(binder.reader().geometry());
    out.frontend     = family::bind_text_only_frontend_resources(binder);
    out.features     = features;

    if (features.vision) {
        throw std::runtime_error("qwen3-0.6b is a text-only target: --vision is unsupported");
    }
    if (features.speculative_enabled()) {
        // Qwen3 ships no MTP block and the target declares no DFlash tower, so
        // there is nothing to draft with. Refusing here beats a missing-object
        // failure twenty objects later.
        throw std::runtime_error(
            "qwen3-0.6b carries no draft head (no MTP block, no DFlash tower); run without --spec");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {TextConfig::output_rows, TextConfig::hidden});
    bind_text_layers(binder, weights_profile, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {TextConfig::hidden});
    // `tie_word_embeddings` is a property of the checkpoint, not of the artifact:
    // the converter resolves it and stores the head as its own object.
    out.output_head = bind_weight(binder, "text/output_head", vocabulary_format,
                                  {TextConfig::output_rows, TextConfig::hidden});

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    // The layer storage is sized here, not by the type: the counts come from the
    // geometry these weights were bound against.
    runtime.geometry = plan.geometry;
    runtime.full_layers.resize(kFullAttentionLayers);
    runtime.gdn_layers.resize(kGdnLayers);
    frontend = family::take_text_only_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;

    runtime.token_embedding = materialized_weight(backing, plan.token_embedding,
                                                  TextConfig::output_rows, TextConfig::hidden);
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source  = plan.text_layers[layer];
        FullAttentionWeights& target = runtime.full_layers.at(layer);
        target.input_norm            = artifact::materialized_tensor(
            backing, source.input_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.projection = FusedAttentionProjectionPayload{
            .query_key_value = materialized_weight(backing, source.attention.query_key_value,
                                                   kAttentionInputRows, TextConfig::hidden),
        };
        target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                          NumericFormat::BF16,
                                                          {TextConfig::head_dim});
        target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                          NumericFormat::BF16,
                                                          {TextConfig::head_dim});
        target.output = materialized_weight(backing, source.attention.output, TextConfig::hidden,
                                            TextConfig::query_size);
        target.post_attention_norm = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        target.post_mixer = load_mlp(source.mlp, backing);
    }
    static_assert(kGdnLayers == 0, "a Qwen3 layer is never a linear mixer");

    runtime.final_norm  = artifact::materialized_tensor(backing, plan.final_norm,
                                                        NumericFormat::BF16, {TextConfig::hidden});
    runtime.output_head = materialized_weight(backing, plan.output_head, TextConfig::output_rows,
                                              TextConfig::hidden);
}

} // namespace sinfer::targets::qwen3::detail
