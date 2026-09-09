#include "targets/qwen3/impl/load/bindings.h"

#include "targets/qwen3/impl/config.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

namespace sinfer::targets::qwen3::detail {
namespace {

using artifact::NumericFormat;

/// Rows of the fused attention input projection: `[query | key | value]`.
/// The hybrid family's is `2 * query_size + 2 * kv_size` because it fuses an
/// output gate; Qwen3 has none, and this is the difference.
[[nodiscard]] std::int32_t attention_input_rows(const family::TextGeometry& g) {
    return g.query_size() + 2 * g.kv_size();
}

[[nodiscard]] std::int32_t mlp_gate_up_rows(const family::TextGeometry& g) {
    return 2 * g.intermediate;
}

static_assert(TextConfig::query_projection_rows == TextConfig::query_size,
              "Qwen3 attention is ungated; a gated projection would be read at the wrong stride");

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("qwen3: invalid weights profile");
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
    if (plan.separate) {
        out.gate = materialized_weight(materialized, plan.gate, g.intermediate, g.hidden);
        out.up = materialized_weight(materialized, plan.up, g.intermediate, g.hidden);
    } else {
        out.gate_up = materialized_weight(materialized, plan.gate_up, mlp_gate_up_rows(g), g.hidden);
    }
    out.down    = materialized_weight(materialized, plan.down, g.hidden, g.intermediate);
    return out;
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile, BindingPlan& out) {
    const NumericFormat weights   = endpoint_format(weights_profile);
    const family::TextGeometry& g = out.geometry;
    // The shapes every tensor is checked against are the checkpoint's, so a differently
    // sized Qwen3 is bound by the same code: what makes this target a Qwen3 is the object
    // names and their arrangement, not how wide they are.
    out.text_layers.resize(static_cast<std::size_t>(g.layers));
    for (std::size_t layer = 0; layer < out.text_layers.size(); ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(
            binder, prefix + "input_norm", NumericFormat::BF16, {g.hidden});
        target.attention.query_key_value =
            bind_weight(binder, prefix + "attention/query_key_value", weights,
                        {attention_input_rows(g), g.hidden});
        // Qwen3 normalises each head of q and k (`use_qk_norm`), over head_dim,
        // and carries no qkv bias (`attention_bias: false` in every released
        // Qwen3 config), which is why no bias object is bound here.
        target.attention.query_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/query_norm", NumericFormat::BF16, {g.head_dim});
        target.attention.key_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/key_norm", NumericFormat::BF16, {g.head_dim});
        target.attention.output = bind_weight(binder, prefix + "attention/output", weights,
                                              {g.hidden, g.query_size()});
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {g.hidden});
        target.mlp.separate = binder.has(prefix + "mlp/gate");
        if (target.mlp.separate) {
            target.mlp.gate = bind_weight(binder, prefix + "mlp/gate", weights, {g.intermediate, g.hidden});
            target.mlp.up = bind_weight(binder, prefix + "mlp/up", weights, {g.intermediate, g.hidden});
        } else {
            target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", weights,
                                             {mlp_gate_up_rows(g), g.hidden});
        }
        target.mlp.down    = bind_weight(binder, prefix + "mlp/down", weights,
                                         {g.hidden, g.intermediate});
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    // Required dimensions and the layer schedule come from the artifact.
    out.geometry = family::TextGeometry::resolved(binder.reader().geometry(), binder.reader().layer_types());
    out.frontend     = family::bind_frontend_resources(binder);
    out.features     = features;

    const bool has_vision = binder.has("vision/patch_embedding");
    const bool vl = binder.reader().identity().architecture == "qwen3_vl";
    if (has_vision != vl || (features.vision && !has_vision)) {
        throw std::runtime_error("Qwen3-VL requires its vision encoder; Qwen3 is text-only");
    }
    if (vl) {
        out.vision_geometry = family::VisionGeometry::resolved(binder.reader().vision_geometry());
        const auto& v = out.vision_geometry;
        if (v.siglip2 || v.output_hidden != out.geometry.hidden ||
            v.deepstack_layers > out.geometry.layers || !out.geometry.mrope_temporal) {
            throw std::runtime_error("Qwen3-VL vision or MRoPE geometry disagrees with the text model");
        }
        const auto placement = features.vision ? artifact::TensorPlacement::Device
                                              : artifact::TensorPlacement::ValidateOnly;
        out.vision_backbone = family::bind_vision_backbone(binder, placement, v);
        out.vision_merger_input = family::bind_vision_merger_input(binder, placement, v);
        out.vision_merger_norm = family::bind_vision_merger_norm(binder, placement, v);
        out.vision_merger_output = artifact::bind_linear(
            binder, "vision/merger/fc2", v.output_hidden, v.merger_hidden(), placement);
        const auto tensor = [&](const std::string& name, int width) {
            return artifact::bind_tensor(binder, name, NumericFormat::BF16, {width}, placement);
        };
        out.vision_merger_output_bias = tensor("vision/merger/fc2_bias", v.output_hidden);
        for (int layer = 0; layer < v.layers; ++layer) {
            const auto prefix = "vision/layers/" + std::to_string(layer) + "/deepstack/";
            if (!binder.has(prefix + "fc1")) { continue; }
            out.deepstack.push_back(BindingPlan::DeepstackPlan{
                .layer = layer,
                .fc1 = artifact::bind_linear(binder, prefix + "fc1", v.merger_hidden(), v.merger_hidden(), placement),
                .fc2 = artifact::bind_linear(binder, prefix + "fc2", v.output_hidden, v.merger_hidden(), placement),
                .fc1_bias = tensor(prefix + "fc1_bias", v.merger_hidden()),
                .fc2_bias = tensor(prefix + "fc2_bias", v.output_hidden),
                .norm_weight = tensor(prefix + "norm/weight", v.merger_hidden()),
                .norm_bias = tensor(prefix + "norm/bias", v.merger_hidden()),
            });
        }
        if (out.deepstack.size() != static_cast<std::size_t>(v.deepstack_layers)) {
            throw std::runtime_error("Qwen3-VL deepstack objects disagree with the declared count");
        }
    }
    if (features.speculative_enabled()) {
        // Qwen3 ships no MTP block and the target declares no DFlash tower, so
        // there is nothing to draft with. Refusing here beats a missing-object
        // failure twenty objects later.
        throw std::runtime_error(
            "Qwen3 and Qwen3-VL carry no draft head; run without --spec");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    const family::TextGeometry& g         = out.geometry;
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {g.output_rows, g.hidden});
    bind_text_layers(binder, weights_profile, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {g.hidden});
    // `tie_word_embeddings` is a property of the checkpoint, not of the artifact:
    // the converter resolves it and stores the head as its own object.
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
    // The layer storage is sized here, not by the type: the counts come from the
    // geometry these weights were bound against.
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;
    runtime.full_layers.resize(static_cast<std::size_t>(g.layers));
    runtime.gdn_layers.resize(kGdnLayers);
    frontend = family::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    runtime.vision_geometry = plan.vision_geometry;
    if (plan.features.vision) {
        const auto& v = plan.vision_geometry;
        family::VisionWeights vision;
        vision.common = family::materialize_vision_common(backing, plan.vision_backbone,
            plan.vision_merger_input, plan.vision_merger_norm, v);
        const auto tensor = [&](artifact::ObjectHandle handle, int width) {
            return artifact::materialized_tensor(backing, handle, NumericFormat::BF16, {width});
        };
        vision.merger_fc2 = artifact::materialized_linear(backing, plan.vision_merger_output,
                                                         v.output_hidden, v.merger_hidden());
        vision.merger_fc2_bias = tensor(plan.vision_merger_output_bias, v.output_hidden);
        for (const auto& merger : plan.deepstack) {
            vision.deepstack.push_back(family::VisionWeights::DeepstackMerger{
                .layer = merger.layer,
                .fc1 = artifact::materialized_linear(backing, merger.fc1, v.merger_hidden(), v.merger_hidden()),
                .fc2 = artifact::materialized_linear(backing, merger.fc2, v.output_hidden, v.merger_hidden()),
                .fc1_bias = tensor(merger.fc1_bias, v.merger_hidden()),
                .fc2_bias = tensor(merger.fc2_bias, v.output_hidden),
                .norm_weight = tensor(merger.norm_weight, v.merger_hidden()),
                .norm_bias = tensor(merger.norm_bias, v.merger_hidden()),
            });
        }
        runtime.vision = std::move(vision);
    }

    runtime.token_embedding =
        materialized_weight(backing, plan.token_embedding, g.output_rows, g.hidden);
    for (std::size_t layer = 0; layer < plan.text_layers.size(); ++layer) {
        const TextLayerPlan& source  = plan.text_layers[layer];
        FullAttentionWeights& target = runtime.full_layers.at(layer);
        target.input_norm            = artifact::materialized_tensor(
            backing, source.input_norm, NumericFormat::BF16, {g.hidden});
        target.projection = FusedAttentionProjectionPayload{
            .query_key_value = materialized_weight(backing, source.attention.query_key_value,
                                                   attention_input_rows(g), g.hidden),
        };
        target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                          NumericFormat::BF16, {g.head_dim});
        target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                          NumericFormat::BF16, {g.head_dim});
        target.output = materialized_weight(backing, source.attention.output, g.hidden,
                                            g.query_size());
        target.post_attention_norm = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
        target.post_mixer = load_mlp(source.mlp, backing, g);
    }
    static_assert(kGdnLayers == 0, "a Qwen3 layer is never a linear mixer");

    runtime.final_norm  = artifact::materialized_tensor(backing, plan.final_norm,
                                                        NumericFormat::BF16, {g.hidden});
    runtime.output_head =
        materialized_weight(backing, plan.output_head, g.output_rows, g.hidden);
}

} // namespace sinfer::targets::qwen3::detail
