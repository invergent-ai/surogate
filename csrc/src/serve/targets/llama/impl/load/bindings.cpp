#include "targets/llama/impl/load/bindings.h"

#include "targets/llama/impl/config.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

namespace sinfer::targets::llama::detail {
namespace {

using artifact::NumericFormat;

/// Rows of the fused attention input projection: `[query | key | value]`.
/// The hybrid family's is `2 * query_size + 2 * kv_size` because it fuses an
/// output gate; Llama has none, and this constant is the difference.
[[nodiscard]] std::int32_t attention_input_rows(const family::TextGeometry& g) {
    return g.query_size() + 2 * g.kv_size();
}

[[nodiscard]] std::int32_t mlp_gate_up_rows(const family::TextGeometry& g) {
    return 2 * g.intermediate;
}

static_assert(TextConfig::query_projection_rows == TextConfig::query_size,
              "Llama attention is ungated; a gated projection would be read at the wrong stride");

/// The four resources a text-only Llama artifact carries. The family's binder
/// demands six; the two it adds are the image and video preprocessor configs,
/// which TinyLlama-1.1B does not publish. The unfilled halves of the plan stay
/// default-constructed and `take_text_only_frontend_resources` leaves their
/// strings empty, which is what the frontend reads as "no pixel pipeline".



NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("llama: invalid weights profile");
}

/// A matrix at whatever format the artifact declares. The profile's format is what a
/// converted checkpoint stores, but a GGUF served natively keeps its own K-quants, and the
/// kernels dispatch on the weight's qtype either way.
WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat,
                       std::initializer_list<std::uint64_t> shape) {
    if (shape.size() != 2) { throw std::logic_error("bind_weight: rank-two shape"); }
    const auto dims = std::vector<std::uint64_t>(shape);
    const artifact::LinearBinding binding =
        artifact::bind_linear(binder, name, static_cast<std::int32_t>(dims[0]),
                              static_cast<std::int32_t>(dims[1]));
    return WeightPlan{.object = binding.object, .format = binding.format};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized,
                               const family::TextGeometry& g) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, mlp_gate_up_rows(g), g.hidden);
    out.down = materialized_weight(materialized, plan.down, g.hidden,
                                   g.intermediate);
    return out;
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile, BindingPlan& out) {
    const family::TextGeometry& g = out.geometry;
    out.text_layers.resize(static_cast<std::size_t>(g.layers));
    const NumericFormat weights = endpoint_format(weights_profile);
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(g.layers); ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(
            binder, prefix + "input_norm", NumericFormat::BF16, {g.hidden});
        target.attention.query_key_value =
            bind_weight(binder, prefix + "attention/query_key_value", weights,
                        {attention_input_rows(g), g.hidden});
        // Llama normalises neither q nor k per head -- `LlamaAttention` ropes the
        // projection's own output -- so unlike Qwen3 there is no `query_norm` or
        // `key_norm` object to bind here, and `Variant::attention_qk_norm` is
        // false so the family runtime skips the step rather than reading a plane
        // nothing wrote. Llama also carries no qkv bias
        // (`attention_bias: false`), which is why no bias object is bound either.
        target.attention.output = bind_weight(binder, prefix + "attention/output", weights,
                                              {g.hidden, g.query_size()});
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {g.hidden});
        target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", weights,
                                         {mlp_gate_up_rows(g), g.hidden});
        target.mlp.down    = bind_weight(binder, prefix + "mlp/down", weights,
                                         {g.hidden, g.intermediate});
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
    const family::TextGeometry& g = out.geometry;
    out.frontend     = family::bind_text_only_frontend_resources(binder);
    out.features     = features;

    if (features.vision) {
        throw std::runtime_error("tinyllama-1.1b is a text-only target: --vision is unsupported");
    }
    if (features.speculative_enabled()) {
        // Llama ships no MTP block and the target declares no DFlash tower, so
        // there is nothing to draft with. Refusing here beats a missing-object
        // failure twenty objects later.
        throw std::runtime_error("tinyllama-1.1b carries no draft head (no MTP block, no DFlash "
                                 "tower); run without --spec");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {g.output_rows, g.hidden});
    bind_text_layers(binder, weights_profile, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {g.hidden});
    // TinyLlama unties the head: `lm_head.weight` is its own matrix in the
    // checkpoint, and the converter stores it as its own object.
    out.output_head = bind_weight(binder, "text/output_head", vocabulary_format,
                                  {g.output_rows, g.hidden});

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    // The layer storage is sized here, not by the type: the counts come from the
    // geometry these weights were bound against.
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;
    // Every layer of a dense decoder attends.
    runtime.full_layers.resize(static_cast<std::size_t>(g.layers));
    runtime.gdn_layers.resize(kGdnLayers);
    frontend = family::take_text_only_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;

    runtime.token_embedding = materialized_weight(backing, plan.token_embedding,
                                                  g.output_rows, g.hidden);
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(g.layers); ++layer) {
        const TextLayerPlan& source  = plan.text_layers[layer];
        FullAttentionWeights& target = runtime.full_layers.at(layer);
        target.input_norm            = artifact::materialized_tensor(
            backing, source.input_norm, NumericFormat::BF16, {g.hidden});
        target.projection = FusedAttentionProjectionPayload{
            .query_key_value = materialized_weight(backing, source.attention.query_key_value,
                                                   attention_input_rows(g), g.hidden),
        };
        // `target.query_norm` and `target.key_norm` stay default-constructed. The
        // family's FullAttentionWeights names them for the targets that have
        // them; this one does not, and the runtime only takes their address --
        // it never reads through it while `attention_qk_norm` is false.
        target.output = materialized_weight(backing, source.attention.output, g.hidden,
                                            g.query_size());
        target.post_attention_norm = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
        target.post_mixer = load_mlp(source.mlp, backing, g);
    }
    static_assert(kGdnLayers == 0, "a Llama layer is never a linear mixer");

    runtime.final_norm  = artifact::materialized_tensor(backing, plan.final_norm,
                                                        NumericFormat::BF16, {g.hidden});
    runtime.output_head = materialized_weight(backing, plan.output_head, g.output_rows,
                                              g.hidden);
}

} // namespace sinfer::targets::llama::detail
