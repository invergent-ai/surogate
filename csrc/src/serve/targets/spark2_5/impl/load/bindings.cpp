#include "targets/spark2_5/impl/load/bindings.h"

#include "targets/spark2_5/impl/config.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

namespace sinfer::targets::spark2_5::detail {
namespace {

using artifact::NumericFormat;

[[nodiscard]] std::int32_t attention_input_rows(const family::TextGeometry& g) {
    return g.query_size() + 2 * g.kv_size();
}

[[nodiscard]] std::int32_t mlp_gate_up_rows(const family::TextGeometry& g) {
    return 2 * g.intermediate;
}

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("spark2_5: invalid weights profile");
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
        target.resident = binder.contains_layer(static_cast<int>(layer));
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(
            binder, prefix + "input_norm", NumericFormat::BF16, {g.hidden});
        target.attention.query_key_value =
            bind_weight(binder, prefix + "attention/query_key_value", weights,
                        {attention_input_rows(g), g.hidden});
        target.attention.output_gate = bind_weight(binder, prefix + "attention/output_gate",
                                                   NumericFormat::BF16, {g.query_heads, g.hidden});
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
    out.geometry = family::TextGeometry::resolved(binder.reader().geometry(), binder.reader().layer_types());
    const family::TextGeometry& g = out.geometry;
    if (g.linear_layer_count() != 0 || g.sliding_window <= 0 || g.sliding_rotary_dim <= 0 ||
        g.sliding_rope_theta <= 0 || g.global_head_dim != 0 || g.residual_fp32 != 1) {
        throw std::invalid_argument("Spark requires attention-only layers and per-type rotary metadata");
    }
    out.frontend     = family::bind_text_only_frontend_resources(binder);
    out.features     = features;

    if (features.vision) {
        throw std::runtime_error("spark2_5 is a text-only target: --vision is unsupported");
    }
    if (features.speculative_enabled()) {
        throw std::runtime_error("spark2_5 carries no draft head (no MTP block, no DFlash "
                                 "tower); run without --spec");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {g.output_rows, g.hidden});
    bind_text_layers(binder, weights_profile, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {g.hidden});
    out.output_head = bind_weight(binder, "text/output_head", vocabulary_format,
                                  {g.output_rows, g.hidden});

    load_plan.materialization = binder.finish();
    out.host_bank = family::collect_host_bank(binder, load_plan.materialization);
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)),
      host_bank(plan.host_bank.objects.empty() ? nullptr : family::HostBank::shared(plan.host_bank)) {
    if (host_bank) { host_bank->attach(backing); }
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;
    runtime.full_layers.resize(static_cast<std::size_t>(g.layers));
    runtime.gdn_layers.resize(kGdnLayers);
    frontend = family::take_text_only_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;

    runtime.token_embedding = materialized_weight(backing, plan.token_embedding,
                                                  g.output_rows, g.hidden);
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(g.layers); ++layer) {
        const TextLayerPlan& source  = plan.text_layers[layer];
        if (!source.resident) { continue; }
        FullAttentionWeights& target = runtime.full_layers.at(layer);
        target.input_norm            = artifact::materialized_tensor(
            backing, source.input_norm, NumericFormat::BF16, {g.hidden});
        target.projection = FusedAttentionProjectionPayload{
            .query_key_value = materialized_weight(backing, source.attention.query_key_value,
                                                   attention_input_rows(g), g.hidden),
            .output_gate = materialized_weight(backing, source.attention.output_gate,
                                              g.query_heads, g.hidden),
        };
        target.output = materialized_weight(backing, source.attention.output, g.hidden,
                                            g.query_size());
        target.post_attention_norm = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
        target.post_mixer = load_mlp(source.mlp, backing, g);
    }
    static_assert(kGdnLayers == 0, "a Spark layer is never a linear mixer");

    runtime.final_norm  = artifact::materialized_tensor(backing, plan.final_norm,
                                                        NumericFormat::BF16, {g.hidden});
    runtime.output_head = materialized_weight(backing, plan.output_head, g.output_rows,
                                              g.hidden);
}

} // namespace sinfer::targets::spark2_5::detail
