#include "targets/qwen3_moe/impl/load/bindings.h"

#include "targets/qwen3_moe/impl/config.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

namespace sinfer::targets::qwen3_moe::detail {
namespace {

using artifact::NumericFormat;

/// Rows of the fused attention input projection: `[query | key | value]`. The hybrid family's
/// is `2 * query_size + 2 * kv_size` because it fuses an output gate; this architecture has
/// none, and that is the difference.
[[nodiscard]] std::int32_t attention_input_rows(const family::TextGeometry& g) {
    return g.query_size() + 2 * g.kv_size();
}

/// Rows of the stacked routed experts. The declaration writes an expert-major `[E, 2M, C]`
/// parameter as rows, which is a contiguous reshape rather than a permutation.
[[nodiscard]] std::int32_t routed_gate_up_rows(const family::TextGeometry& g) {
    return TextConfig::experts * 2 * g.intermediate;
}

[[nodiscard]] std::int32_t routed_down_rows(const family::TextGeometry& g) {
    return TextConfig::experts * g.hidden;
}

static_assert(TextConfig::query_projection_rows == TextConfig::query_size,
              "this attention is ungated; a gated projection would be read at the wrong stride");

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("qwen3_moe: invalid weights profile");
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

SparseMoePayload load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& materialized,
                          const family::TextGeometry& g) {
    return SparseMoePayload{
        .op = {
            .router_shared_gate = artifact::materialized_weight(
                materialized, plan.router, NumericFormat::BF16, TextConfig::router_rows, g.hidden),
            // The *stored* format decides how these bytes are read. Passing the profile's
            // expectation instead decoded a GGUF's K-quant superblocks as the group-wise
            // row-split codec: the same byte count, an entirely different meaning, and every
            // routed expert silently wrong.
            .routed_gate_up = artifact::materialized_linear(materialized, plan.routed_gate_up,
                                                            routed_gate_up_rows(g), g.hidden),
            .routed_down    = artifact::materialized_linear(materialized, plan.routed_down,
                                                            routed_down_rows(g), g.intermediate),
            // No always-on expert: the two shared weights stay empty, and the op reads the
            // geometry -- not these -- to know that a token sums its selected experts and
            // nothing else.
            .shared_gate_up    = Weight{},
            .shared_down       = Weight{},
            .experts_per_token = TextConfig::experts_per_token,
        }};
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile,
                      std::uint32_t host_moe_layers, std::uint32_t gpu_layers,
                      BindingPlan& out) {
    const NumericFormat weights   = endpoint_format(weights_profile);
    const family::TextGeometry& g = out.geometry;
    // The shapes every tensor is checked against are the checkpoint's, so a differently sized
    // checkpoint is bound by the same code: what makes this target this architecture is the
    // object names and their arrangement, not how wide they are.
    out.text_layers.resize(static_cast<std::size_t>(g.layers));
    for (std::size_t layer = 0; layer < out.text_layers.size(); ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        // Past `gpu_layers` the whole layer is read from pinned host memory; below it, only the
        // experts, and only if asked. Both counts are whole-model.
        const artifact::ScopedPlacement placed(
            gpu_layers != 0 && layer >= gpu_layers ? artifact::TensorPlacement::HostBank
                                                   : artifact::TensorPlacement::Device);
        target.input_norm        = artifact::bind_device_tensor(
            binder, prefix + "input_norm", NumericFormat::BF16, {g.hidden});
        target.attention.query_key_value =
            bind_weight(binder, prefix + "attention/query_key_value", weights,
                        {attention_input_rows(g), g.hidden});
        // Each head of q and k is normalised over head_dim, and there is no qkv bias
        // (`attention_bias: false` in every released config), which is why no bias object is
        // bound here.
        target.attention.query_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/query_norm", NumericFormat::BF16, {g.head_dim});
        target.attention.key_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/key_norm", NumericFormat::BF16, {g.head_dim});
        target.attention.output = bind_weight(binder, prefix + "attention/output", weights,
                                              {g.hidden, g.query_size()});
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {g.hidden});
        target.moe.router = artifact::bind_device_tensor(
            binder, prefix + "moe/router", NumericFormat::BF16,
            {static_cast<std::uint64_t>(TextConfig::router_rows),
             static_cast<std::uint64_t>(g.hidden)});
        const artifact::ScopedPlacement experts(
            layer < host_moe_layers ? artifact::TensorPlacement::HostBank
                                    : artifact::ScopedPlacement::current());
        target.moe.routed_gate_up = artifact::bind_linear(
            binder, prefix + "moe/routed_gate_up", routed_gate_up_rows(g), g.hidden);
        target.moe.routed_down = artifact::bind_linear(
            binder, prefix + "moe/routed_down", routed_down_rows(g), g.intermediate);
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features, std::uint32_t host_moe_layers,
                               std::uint32_t gpu_layers, LoadProgress progress) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    // The checkpoint's own dimensions, where it states them: absent members keep the
    // target's compiled value, so an artifact written before the member existed binds
    // exactly as it did.
    out.geometry = family::TextGeometry::declared<TextConfig>(binder.reader().geometry());
    out.frontend     = family::bind_text_only_frontend_resources(binder);
    out.features     = features;

    if (features.vision) {
        throw std::runtime_error("qwen3_moe is a text-only target: --vision is unsupported");
    }
    if (features.speculative_enabled()) {
        // Qwen3-MoE ships no MTP block and the target declares no DFlash tower, so there is
        // nothing to draft with. Refusing here beats a missing-object failure twenty objects
        // later.
        throw std::runtime_error(
            "qwen3_moe carries no draft head (no MTP block, no DFlash tower); run without --spec");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    const family::TextGeometry& g         = out.geometry;
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {g.output_rows, g.hidden});
    bind_text_layers(binder, weights_profile, host_moe_layers, gpu_layers, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {g.hidden});
    // `tie_word_embeddings` is a property of the checkpoint, not of the artifact: the
    // converter resolves it and stores the head as its own object either way.
    out.output_head = bind_weight(binder, "text/output_head", vocabulary_format,
                                  {g.output_rows, g.hidden});

    load_plan.materialization = binder.finish();
    out.host_bank = family::collect_host_bank(binder, load_plan.materialization,
                                              std::move(progress));
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)),
      host_bank(plan.host_bank.objects.empty() ? nullptr
                                               : family::HostBank::shared(plan.host_bank)) {
    if (host_bank) { host_bank->attach(backing); }
    // The layer storage is sized here, not by the type: the counts come from the
    // geometry these weights were bound against.
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;
    runtime.full_layers.resize(static_cast<std::size_t>(g.layers));
    runtime.gdn_layers.resize(kGdnLayers);
    frontend = family::take_text_only_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;

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
        target.post_mixer = load_moe(source.moe, backing, g);
    }
    static_assert(kGdnLayers == 0, "every layer of this architecture attends");

    runtime.final_norm  = artifact::materialized_tensor(backing, plan.final_norm,
                                                        NumericFormat::BF16, {g.hidden});
    runtime.output_head =
        materialized_weight(backing, plan.output_head, g.output_rows, g.hidden);
}

} // namespace sinfer::targets::qwen3_moe::detail
