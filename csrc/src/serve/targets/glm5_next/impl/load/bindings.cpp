#include "targets/glm5_next/impl/load/bindings.h"

#include "targets/glm5_next/impl/config.h"

#include "artifact/reader.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::targets::glm5_next::detail {
namespace {

using artifact::NumericFormat;

/// The mixture's compiled geometry. Its shapes are a closed registry in the op, so the target
/// names the one entry it serves rather than deriving numbers the kernels cannot honour.
inline constexpr ops::SparseMoeGeometry kMoeGeometry = ops::kSparseMoeGlm53Geometry;

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("glm5_next: invalid weights profile");
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

[[nodiscard]] std::string layer_prefix(std::size_t layer) {
    return "text/layers/" + std::to_string(layer) + "/";
}

HyperConnectionPlan bind_hyper_connection(artifact::Binder& binder, const std::string& prefix,
                                          std::string_view site,
                                          const family::TextGeometry& g) {
    const std::string base = prefix + "hc/" + std::string(site);
    HyperConnectionPlan out;
    // BF16 rather than the endpoint format: these rows produce every mixing weight the residual
    // is recombined with, at both sites of every layer.
    out.mix   = bind_weight(binder, base + "_mix", NumericFormat::BF16,
                            {static_cast<std::uint64_t>(g.hyper_connection_mix_rows()),
                             static_cast<std::uint64_t>(g.residual)});
    out.base  = artifact::bind_device_tensor(
        binder, base + "_base", NumericFormat::FP32,
        {static_cast<std::uint64_t>(g.hyper_connection_mix_rows())});
    out.scale = artifact::bind_device_tensor(binder, base + "_scale", NumericFormat::FP32, {3});
    return out;
}

void bind_feed_forward(artifact::Binder& binder, const std::string& prefix, bool sparse,
                       NumericFormat weights, const family::TextGeometry& g,
                       FeedForwardPlan& out) {
    out.sparse = sparse;
    if (!sparse) {
        out.gate_up = bind_weight(binder, prefix + "mlp/gate_up", weights,
                                  {static_cast<std::uint64_t>(2 * g.dense_intermediate),
                                   static_cast<std::uint64_t>(g.hidden)});
        out.down    = bind_weight(binder, prefix + "mlp/down", weights,
                                  {static_cast<std::uint64_t>(g.hidden),
                                   static_cast<std::uint64_t>(g.dense_intermediate)});
        return;
    }
    const auto experts = static_cast<std::uint64_t>(kMoeGeometry.experts);
    // The router is BF16 and its bias FP32: a router that picks eight of 288 is the one place
    // in this model where a coarse width changes which experts run rather than by how much.
    out.router      = bind_weight(binder, prefix + "moe/router", NumericFormat::BF16,
                                  {static_cast<std::uint64_t>(kMoeGeometry.router_rows()),
                                   static_cast<std::uint64_t>(g.hidden)});
    out.router_bias = artifact::bind_device_tensor(binder, prefix + "moe/router_bias",
                                                   NumericFormat::FP32, {experts});
    out.routed_gate_up =
        bind_weight(binder, prefix + "moe/routed_gate_up", weights,
                    {experts * static_cast<std::uint64_t>(kMoeGeometry.expert_rows()),
                     static_cast<std::uint64_t>(g.hidden)});
    out.routed_down = bind_weight(binder, prefix + "moe/routed_down", weights,
                                  {experts * static_cast<std::uint64_t>(g.hidden),
                                   static_cast<std::uint64_t>(kMoeGeometry.intermediate)});
    out.shared_gate_up =
        bind_weight(binder, prefix + "moe/shared_gate_up", weights,
                    {static_cast<std::uint64_t>(kMoeGeometry.shared_rows()),
                     static_cast<std::uint64_t>(g.hidden)});
    out.shared_down = bind_weight(binder, prefix + "moe/shared_down", weights,
                                  {static_cast<std::uint64_t>(g.hidden),
                                   static_cast<std::uint64_t>(kMoeGeometry.shared_intermediate)});
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile, BindingPlan& out) {
    const NumericFormat weights   = endpoint_format(weights_profile);
    const family::TextGeometry& g = out.geometry;
    out.text_layers.resize(static_cast<std::size_t>(g.layers));
    for (std::size_t layer = 0; layer < out.text_layers.size(); ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = layer_prefix(layer);
        target.attends           = g.layer_attends(static_cast<std::int32_t>(layer));

        target.attention_hc = bind_hyper_connection(binder, prefix, "attn", g);
        target.input_norm   = artifact::bind_device_tensor(
            binder, prefix + "input_norm", NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        if (target.attends) {
            LatentAttentionPlan& mla = target.attention;
            mla.query_a      = bind_weight(binder, prefix + "mla/query_a", weights,
                                           {static_cast<std::uint64_t>(g.q_lora_rank),
                                            static_cast<std::uint64_t>(g.hidden)});
            mla.query_a_norm = artifact::bind_device_tensor(
                binder, prefix + "mla/query_a_norm", NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.q_lora_rank)});
            mla.query_b      = bind_weight(binder, prefix + "mla/query_b", weights,
                                           {static_cast<std::uint64_t>(g.query_size()),
                                            static_cast<std::uint64_t>(g.q_lora_rank)});
            mla.kv_a         = bind_weight(binder, prefix + "mla/kv_a", weights,
                                           {static_cast<std::uint64_t>(g.kv_lora_rank),
                                            static_cast<std::uint64_t>(g.hidden)});
            mla.kv_a_norm    = artifact::bind_device_tensor(
                binder, prefix + "mla/kv_a_norm", NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.kv_lora_rank)});
            // BF16: llama.cpp stores the key half of the expansion transposed, so it is the one
            // weight of this model that has to be materialised rather than read where it lies.
            mla.k_b    = bind_weight(binder, prefix + "mla/k_b", NumericFormat::BF16,
                                     {static_cast<std::uint64_t>(g.kv_size()),
                                      static_cast<std::uint64_t>(g.kv_lora_rank)});
            mla.v_b    = bind_weight(binder, prefix + "mla/v_b", weights,
                                     {static_cast<std::uint64_t>(g.kv_size()),
                                      static_cast<std::uint64_t>(g.kv_lora_rank)});
            mla.output = bind_weight(binder, prefix + "mla/output", weights,
                                     {static_cast<std::uint64_t>(g.hidden),
                                      static_cast<std::uint64_t>(g.query_size())});
        } else {
            KdaPlan& kda        = target.kda;
            kda.query_key_value = bind_weight(binder, prefix + "kda/query_key_value", weights,
                                              {static_cast<std::uint64_t>(g.convolution_dim()),
                                               static_cast<std::uint64_t>(g.hidden)});
            // Tap-major, [K, channels] as the artifact writes it, which is [channels, K] in the
            // engine's own layout.
            kda.convolution = artifact::bind_device_tensor(
                binder, prefix + "kda/convolution", NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.gdn_conv_kernel),
                 static_cast<std::uint64_t>(g.convolution_dim())});
            kda.decay_a    = bind_weight(binder, prefix + "kda/decay_a", weights,
                                         {static_cast<std::uint64_t>(g.kda_gate_rank),
                                          static_cast<std::uint64_t>(g.hidden)});
            kda.decay_b    = bind_weight(binder, prefix + "kda/decay_b", weights,
                                         {static_cast<std::uint64_t>(g.value_dim()),
                                          static_cast<std::uint64_t>(g.kda_gate_rank)});
            kda.decay_bias = artifact::bind_device_tensor(
                binder, prefix + "kda/decay_bias", NumericFormat::FP32,
                {static_cast<std::uint64_t>(g.value_dim())});
            kda.a_log      = artifact::bind_device_tensor(
                binder, prefix + "kda/a_log", NumericFormat::FP32,
                {static_cast<std::uint64_t>(g.gdn_value_heads)});
            kda.beta       = bind_weight(binder, prefix + "kda/beta", weights,
                                         {static_cast<std::uint64_t>(g.gdn_value_heads),
                                          static_cast<std::uint64_t>(g.hidden)});
            kda.gate_a     = bind_weight(binder, prefix + "kda/gate_a", weights,
                                         {static_cast<std::uint64_t>(g.kda_gate_rank),
                                          static_cast<std::uint64_t>(g.hidden)});
            kda.gate_b     = bind_weight(binder, prefix + "kda/gate_b", weights,
                                         {static_cast<std::uint64_t>(g.value_dim()),
                                          static_cast<std::uint64_t>(g.kda_gate_rank)});
            kda.norm       = artifact::bind_device_tensor(
                binder, prefix + "kda/norm", NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.gdn_value_head_dim)});
            kda.output     = bind_weight(binder, prefix + "kda/output", weights,
                                         {static_cast<std::uint64_t>(g.hidden),
                                          static_cast<std::uint64_t>(g.value_dim())});
        }
        target.feed_forward_hc     = bind_hyper_connection(binder, prefix, "ffn", g);
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        // Which layers are dense is the artifact's to say too, and it says it the same way:
        // a layer holding a router is a mixture layer.
        const bool sparse = binder.reader().find(prefix + "moe/router") != nullptr;
        bind_feed_forward(binder, prefix, sparse, weights, g, target.feed_forward);
    }
}

HyperConnectionPayload load_hyper_connection(const artifact::MaterializedArtifact& backing,
                                             const HyperConnectionPlan& plan,
                                             const family::TextGeometry& g) {
    HyperConnectionPayload out;
    out.weights.mix   = materialized_weight(backing, plan.mix, g.hyper_connection_mix_rows(),
                                            g.residual);
    out.weights.base  = artifact::materialized_tensor(
        backing, plan.base, NumericFormat::FP32,
        {static_cast<std::uint64_t>(g.hyper_connection_mix_rows())});
    out.weights.scale = artifact::materialized_tensor(backing, plan.scale, NumericFormat::FP32, {3});
    return out;
}

FeedForwardPayload load_feed_forward(const artifact::MaterializedArtifact& backing,
                                     const TextLayerPlan& source, const family::TextGeometry& g) {
    FeedForwardPayload out;
    out.hc     = load_hyper_connection(backing, source.feed_forward_hc, g);
    out.sparse = source.feed_forward.sparse;
    if (!out.sparse) {
        out.gate_up = materialized_weight(backing, source.feed_forward.gate_up,
                                          2 * g.dense_intermediate, g.hidden);
        out.down    = materialized_weight(backing, source.feed_forward.down, g.hidden,
                                          g.dense_intermediate);
        return out;
    }
    out.moe.router_shared_gate = materialized_weight(backing, source.feed_forward.router,
                                                     kMoeGeometry.router_rows(), g.hidden);
    out.moe.router_bias        = static_cast<const float*>(
        artifact::materialized_tensor(backing, source.feed_forward.router_bias,
                                      NumericFormat::FP32,
                                      {static_cast<std::uint64_t>(kMoeGeometry.experts)})
            .data);
    out.moe.routed_gate_up = materialized_weight(backing, source.feed_forward.routed_gate_up,
                                                 kMoeGeometry.routed_gate_rows(), g.hidden);
    out.moe.routed_down    = materialized_weight(backing, source.feed_forward.routed_down,
                                                 kMoeGeometry.routed_down_rows(),
                                                 kMoeGeometry.intermediate);
    out.moe.shared_gate_up = materialized_weight(backing, source.feed_forward.shared_gate_up,
                                                 kMoeGeometry.shared_rows(), g.hidden);
    out.moe.shared_down    = materialized_weight(backing, source.feed_forward.shared_down, g.hidden,
                                                 kMoeGeometry.shared_intermediate);
    out.moe.experts_per_token = kMoeGeometry.experts_per_token;
    out.moe.routed_scale      = kMoeGeometry.routed_scale;
    out.moe.shared_gated      = kMoeGeometry.shared_gated;
    out.moe.swiglu_limit      = kMoeGeometry.swiglu_limit;
    return out;
}

} // namespace

family::TextGeometry declared_geometry_with_schedule(const artifact::Reader& reader) {
    family::TextGeometry geometry = family::TextGeometry::declared<TextConfig>(reader.geometry());
    // Which layers attend is read off the objects, not off a number: a layer holding a latent
    // key/value projection attends. A checkpoint whose layer count the artifact declares
    // therefore brings its own schedule with it, and the two cannot disagree.
    for (std::int32_t layer = 0; layer < geometry.layers; ++layer) {
        const std::string name = "text/layers/" + std::to_string(layer) + "/mla/kv_a";
        if (reader.find(name) != nullptr) { geometry.declare_attention_layer(layer); }
    }
    if (!geometry.attention_schedule_declared) {
        throw std::runtime_error(
            "glm5_next: this artifact holds no latent-attention layer at all; every GLM-5.3 "
            "checkpoint attends at some of its layers, so the objects it carries do not "
            "describe this architecture");
    }
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
        throw std::runtime_error("glm5_next is a text-only target: --vision is unsupported");
    }
    if (features.speculative_enabled()) {
        throw std::runtime_error(
            "glm5_next carries no draft head: the checkpoint's NextN block is not bound, and "
            "Kimi Delta Attention has no replay-record form to verify a speculative round "
            "with. Run without --spec.");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    const family::TextGeometry& g         = out.geometry;
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {static_cast<std::uint64_t>(g.output_rows),
                                       static_cast<std::uint64_t>(g.hidden)});
    bind_text_layers(binder, weights_profile, out);
    out.final_norm      = artifact::bind_device_tensor(
        binder, "text/final_norm", NumericFormat::BF16,
        {static_cast<std::uint64_t>(g.hidden)});
    out.output_head     = bind_weight(binder, "text/output_head", vocabulary_format,
                                      {static_cast<std::uint64_t>(g.output_rows),
                                       static_cast<std::uint64_t>(g.hidden)});

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;

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
    std::size_t kda_index  = 0;
    for (const TextLayerPlan& source : plan.text_layers) {
        const Tensor input_norm = artifact::materialized_tensor(
            backing, source.input_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        const Tensor post_norm = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        if (source.attends) {
            FullAttentionWeights& target = runtime.full_layers.at(full_index++);
            target.input_norm            = input_norm;
            LatentAttentionPayload payload;
            payload.hc      = load_hyper_connection(backing, source.attention_hc, g);
            payload.query_a = materialized_weight(backing, source.attention.query_a, g.q_lora_rank,
                                                  g.hidden);
            payload.query_a_norm = artifact::materialized_tensor(
                backing, source.attention.query_a_norm, NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.q_lora_rank)});
            payload.query_b = materialized_weight(backing, source.attention.query_b, g.query_size(),
                                                  g.q_lora_rank);
            payload.kv_a    = materialized_weight(backing, source.attention.kv_a, g.kv_lora_rank,
                                                  g.hidden);
            payload.kv_a_norm = artifact::materialized_tensor(
                backing, source.attention.kv_a_norm, NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.kv_lora_rank)});
            payload.k_b = materialized_weight(backing, source.attention.k_b, g.kv_size(),
                                              g.kv_lora_rank);
            payload.v_b = materialized_weight(backing, source.attention.v_b, g.kv_size(),
                                              g.kv_lora_rank);
            target.projection = std::move(payload);
            // No per-head query or key norm: this attention normalises its two low ranks
            // instead, and those live in the projection payload.
            target.output = materialized_weight(backing, source.attention.output, g.hidden,
                                                g.query_size());
            target.post_attention_norm = post_norm;
            target.post_mixer          = load_feed_forward(backing, source, g);
        } else {
            KdaWeights& target = runtime.gdn_layers.at(kda_index++);
            target.input_norm  = input_norm;
            KdaProjectionPayload payload;
            payload.hc              = load_hyper_connection(backing, source.attention_hc, g);
            payload.query_key_value = materialized_weight(backing, source.kda.query_key_value,
                                                          g.convolution_dim(), g.hidden);
            payload.decay_a = materialized_weight(backing, source.kda.decay_a, g.kda_gate_rank,
                                                  g.hidden);
            payload.decay_b = materialized_weight(backing, source.kda.decay_b, g.value_dim(),
                                                  g.kda_gate_rank);
            payload.decay_bias = artifact::materialized_tensor(
                backing, source.kda.decay_bias, NumericFormat::FP32,
                {static_cast<std::uint64_t>(g.value_dim())});
            payload.a_log = artifact::materialized_tensor(
                backing, source.kda.a_log, NumericFormat::FP32,
                {static_cast<std::uint64_t>(g.gdn_value_heads)});
            payload.beta   = materialized_weight(backing, source.kda.beta, g.gdn_value_heads,
                                                 g.hidden);
            payload.gate_a = materialized_weight(backing, source.kda.gate_a, g.kda_gate_rank,
                                                 g.hidden);
            payload.gate_b = materialized_weight(backing, source.kda.gate_b, g.value_dim(),
                                                 g.kda_gate_rank);
            target.projection  = std::move(payload);
            target.convolution = artifact::materialized_tensor(
                backing, source.kda.convolution, NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.convolution_dim()),
                 static_cast<std::uint64_t>(g.gdn_conv_kernel)});
            target.norm   = artifact::materialized_tensor(
                backing, source.kda.norm, NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.gdn_value_head_dim)});
            target.output = materialized_weight(backing, source.kda.output, g.hidden,
                                                g.value_dim());
            target.post_attention_norm = post_norm;
            target.post_mixer          = load_feed_forward(backing, source, g);
        }
    }

    runtime.final_norm  = artifact::materialized_tensor(
        backing, plan.final_norm, NumericFormat::BF16,
        {static_cast<std::uint64_t>(g.hidden)});
    runtime.output_head = materialized_weight(backing, plan.output_head, g.output_rows, g.hidden);
}

} // namespace sinfer::targets::glm5_next::detail
