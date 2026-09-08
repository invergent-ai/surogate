#include "targets/lfm2/impl/load/bindings.h"

#include "targets/lfm2/impl/config.h"

#include "artifact/reader.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::targets::lfm2::detail {
namespace {

using artifact::NumericFormat;

/// Rows of the fused attention input projection: `[query | key | value]`. Ungated, so there is
/// no output-gate block between key and value.
[[nodiscard]] std::int32_t attention_input_rows(const family::TextGeometry& g) {
    return g.query_size() + 2 * g.kv_size();
}

[[nodiscard]] std::int32_t mlp_gate_up_rows(const family::TextGeometry& g) {
    return 2 * g.intermediate;
}

/// The mixer's projection produces B, C and x at once, each as wide as the residual stream.
[[nodiscard]] std::int32_t conv_projection_rows(const family::TextGeometry& g) {
    return 3 * g.hidden;
}

static_assert(TextConfig::query_projection_rows == TextConfig::query_size,
              "LFM2 attention is ungated; a gated projection would be read at the wrong stride");

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("lfm2: invalid weights profile");
}

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
    out.down    = materialized_weight(materialized, plan.down, g.hidden, g.intermediate);
    return out;
}

[[nodiscard]] std::string layer_prefix(std::size_t layer) {
    return "text/layers/" + std::to_string(layer) + "/";
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile, BindingPlan& out) {
    const NumericFormat weights   = endpoint_format(weights_profile);
    const family::TextGeometry& g = out.geometry;
    out.text_layers.resize(static_cast<std::size_t>(g.layers));
    for (std::size_t layer = 0; layer < out.text_layers.size(); ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = layer_prefix(layer);
        target.attends           = g.layer_attends(static_cast<std::int32_t>(layer));
        // Both kinds carry these three: the mixer's input norm, the FFN's, and the FFN.
        target.input_norm = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                         NumericFormat::BF16, {g.hidden});
        if (target.attends) {
            target.attention.query_key_value =
                bind_weight(binder, prefix + "attention/query_key_value", weights,
                            {attention_input_rows(g), g.hidden});
            // LFM2 normalises each head of q and k over head_dim and carries no qkv bias, so
            // no bias object is bound here.
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {g.head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {g.head_dim});
            target.attention.output = bind_weight(binder, prefix + "attention/output", weights,
                                                  {g.hidden, g.query_size()});
        } else {
            target.convolution.in_projection =
                bind_weight(binder, prefix + "conv/in_proj", weights,
                            {conv_projection_rows(g), g.hidden});
            // Tap-major, [K, hidden] as the artifact writes it, which is [hidden, K] in the
            // engine's own layout -- the same repacking the linear-attention convolution needs.
            target.convolution.convolution = artifact::bind_device_tensor(
                binder, prefix + "conv/convolution", NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.gdn_conv_kernel),
                 static_cast<std::uint64_t>(g.hidden)});
            target.convolution.out_projection =
                bind_weight(binder, prefix + "conv/out_proj", weights, {g.hidden, g.hidden});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {g.hidden});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", weights, {mlp_gate_up_rows(g), g.hidden});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", weights, {g.hidden, g.intermediate});
    }
}

} // namespace

family::TextGeometry declared_geometry_with_schedule(const artifact::Reader& reader) {
    family::TextGeometry geometry = family::TextGeometry::resolved(reader.geometry(), reader.layer_types());
    return geometry;
}

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.geometry     = declared_geometry_with_schedule(binder.reader());
    out.frontend     = family::bind_text_only_frontend_resources(binder);
    out.features     = features;

    if (features.vision) {
        throw std::runtime_error("lfm2 is a text-only target: --vision is unsupported");
    }
    if (features.speculative_enabled()) {
        throw std::runtime_error(
            "lfm2 carries no draft head (no MTP block, no DFlash tower); run without --spec");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    const family::TextGeometry& g         = out.geometry;
    // LFM2 stores the embedding table in BF16 rather than at the endpoint format, which is what
    // `bind_linear` reads off the artifact anyway; the profile only says what to expect.
    out.token_embedding =
        bind_weight(binder, "text/token_embedding", vocabulary_format, {g.output_rows, g.hidden});
    bind_text_layers(binder, weights_profile, out);
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {g.hidden});
    out.output_head =
        bind_weight(binder, "text/output_head", vocabulary_format, {g.output_rows, g.hidden});

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;

    // Two inventories, sized from the schedule the artifact declared. A layer's index within
    // its own kind is its position among the layers of that kind, counted in order -- which is
    // exactly what the runtime's `full_idx` and `gdn_idx` compute for the same schedule.
    std::size_t attending = 0;
    for (const TextLayerPlan& source : plan.text_layers) { attending += source.attends ? 1U : 0U; }
    runtime.full_layers.resize(attending);
    runtime.gdn_layers.resize(plan.text_layers.size() - attending);
    frontend = family::take_text_only_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;

    runtime.token_embedding =
        materialized_weight(backing, plan.token_embedding, g.output_rows, g.hidden);

    std::size_t full_index = 0;
    std::size_t conv_index = 0;
    for (const TextLayerPlan& source : plan.text_layers) {
        const Tensor input_norm = artifact::materialized_tensor(backing, source.input_norm,
                                                                NumericFormat::BF16, {g.hidden});
        const Tensor post_norm  = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
        if (source.attends) {
            FullAttentionWeights& target = runtime.full_layers.at(full_index++);
            target.input_norm            = input_norm;
            target.projection            = FusedAttentionProjectionPayload{
                           .query_key_value = materialized_weight(backing, source.attention.query_key_value,
                                                       attention_input_rows(g), g.hidden),
            };
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {g.head_dim});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {g.head_dim});
            target.output =
                materialized_weight(backing, source.attention.output, g.hidden, g.query_size());
            target.post_attention_norm = post_norm;
            target.post_mixer          = load_mlp(source.mlp, backing, g);
        } else {
            ConvWeights& target = runtime.gdn_layers.at(conv_index++);
            target.input_norm   = input_norm;
            target.projection   = ShortConvProjectionPayload{
                  .in_projection = materialized_weight(backing, source.convolution.in_projection,
                                                       conv_projection_rows(g), g.hidden),
            };
            target.convolution = artifact::materialized_tensor(
                backing, source.convolution.convolution, NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.hidden),
                 static_cast<std::uint64_t>(g.gdn_conv_kernel)});
            // `norm` stays empty: the gate is applied inside the convolution, so there is no
            // gated RMSNorm between the mixer and its output projection.
            target.output              = materialized_weight(backing,
                                                             source.convolution.out_projection,
                                                             g.hidden, g.hidden);
            target.post_attention_norm = post_norm;
            target.post_mixer          = load_mlp(source.mlp, backing, g);
        }
    }

    runtime.final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {g.hidden});
    runtime.output_head = materialized_weight(backing, plan.output_head, g.output_rows, g.hidden);
}

} // namespace sinfer::targets::lfm2::detail
