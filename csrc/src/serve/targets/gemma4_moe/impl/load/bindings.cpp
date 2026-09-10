#include "targets/gemma4_moe/impl/load/bindings.h"

#include "targets/gemma4_moe/impl/config.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
#include <string_view>

namespace sinfer::targets::gemma4_moe::detail {
namespace {

using artifact::NumericFormat;

static_assert(TextConfig::query_projection_rows == TextConfig::query_size,
              "Gemma 4 attention is ungated; a gated projection would be read at the wrong stride");

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("gemma4_moe: invalid weights profile");
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

Tensor materialized_norm(const artifact::MaterializedArtifact& materialized,
                         artifact::ObjectHandle handle, std::int32_t width) {
    return artifact::materialized_tensor(materialized, handle, NumericFormat::BF16, {width});
}

/// One BF16 element of device memory, as a host float.
///
/// A synchronous copy, which is what this wants to be: it runs once per layer at load, and
/// the value has to be in hand before the first round rather than at some later point on a
/// stream.
float read_device_bf16(const Tensor& tensor) {
    std::uint16_t bits = 0;
    const cudaError_t status =
        cudaMemcpy(&bits, tensor.data, sizeof(bits), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4_moe: could not read a layer scalar: ") +
                                 cudaGetErrorString(status));
    }
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16U;
    float value                 = 0.0F;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

std::int32_t routed_gate_up_rows(const family::TextGeometry& g) {
    return g.experts * 2 * g.intermediate;
}

std::int32_t routed_down_rows(const family::TextGeometry& g) {
    return g.experts * g.hidden;
}

MixturePostMixerPayload load_mlp(const MlpPlan& plan,
                                 const artifact::MaterializedArtifact& materialized,
                                 const family::TextGeometry& g) {
    const std::int32_t dense = dense_intermediate(g);
    MixturePostMixerPayload out;
    out.probe_layer_count = g.layers;
    out.rms_epsilon = g.rms_epsilon;
    out.gate = materialized_weight(materialized, plan.gate, dense, g.hidden);
    out.up   = materialized_weight(materialized, plan.up, dense, g.hidden);
    out.down = materialized_weight(materialized, plan.down, g.hidden, dense);
    out.post_feedforward_norm =
        materialized_norm(materialized, plan.post_feedforward_norm, g.hidden);
    out.post_feedforward_norm_dense =
        materialized_norm(materialized, plan.post_feedforward_norm_dense, g.hidden);
    out.pre_feedforward_norm_routed =
        materialized_norm(materialized, plan.pre_feedforward_norm_routed, g.hidden);
    out.post_feedforward_norm_routed =
        materialized_norm(materialized, plan.post_feedforward_norm_routed, g.hidden);
    out.router_scale = materialized_norm(materialized, plan.router_scale, g.hidden);
    out.op.router_shared_gate = artifact::materialized_weight(
        materialized, plan.router, NumericFormat::BF16, g.experts, g.hidden);
    // FP32 on the device, as the artifact stores it: this multiplies the routing weights, and
    // eight of 128 experts decide the layer's output.
    out.op.per_expert_scale = static_cast<const float*>(
        artifact::materialized_tensor(materialized, plan.per_expert_scale, NumericFormat::FP32,
                                      {static_cast<std::uint64_t>(g.experts)})
            .data);
    // The *stored* format decides how the expert bytes are read, not the profile's
    // expectation -- the trap the other routed targets record, and it costs nothing to avoid
    // here.
    out.op.routed_gate_up =
        artifact::materialized_linear(materialized, plan.routed_gate_up, routed_gate_up_rows(g),
                                      g.hidden);
    out.op.routed_down = artifact::materialized_linear(materialized, plan.routed_down,
                                                       routed_down_rows(g), g.intermediate);
    // No always-on expert. The dense branch beside the experts is the *layer's* projection,
    // not this mixture's shared one: it has no router row, it runs for every token, and the
    // op must not add it a second time.
    out.op.shared_gate_up    = Weight{};
    out.op.shared_down       = Weight{};
    out.op.experts_per_token = g.experts_per_token;
    out.op.activation        = ops::GatedActivation::GeluTanh;
    // One BF16 element, the way the checkpoint stores it, read back to the host once.
    out.layer_scalar       = materialized_norm(materialized, plan.layer_scalar, 1);
    out.layer_scalar_value = read_device_bf16(out.layer_scalar);
    return out;
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile,
                      std::uint32_t host_moe_layers, std::uint32_t gpu_layers, BindingPlan& out) {
    (void)gpu_layers;
    (void)host_moe_layers;
    family::TextGeometry& g     = out.geometry;
    const std::size_t layers    = static_cast<std::size_t>(g.layers);
    out.text_layers.resize(layers);
    const NumericFormat weights = endpoint_format(weights_profile);
    for (std::size_t layer = 0; layer < layers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        target.resident = binder.contains_layer(static_cast<int>(layer));
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        // The schedule is read from checkpoint metadata before binding.
        const bool windowed      = g.layer_is_windowed(static_cast<std::int32_t>(layer));

        const std::int32_t head_dim  = g.head_dim_for(windowed);
        const std::int32_t query_rows = g.query_size_for(windowed);
        const std::int32_t kv_rows    = g.kv_size_for(windowed);

        target.attention.windowed = windowed;
        // Bound in the converter's own order (inventory.py), so a diff of the two lists
        // reads straight down.
        target.input_norm = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                         NumericFormat::BF16, {g.hidden});
        target.attention.post_attention_norm =
            artifact::bind_device_tensor(binder, prefix + "post_attention_norm",
                                         NumericFormat::BF16, {g.hidden});
        target.pre_feedforward_norm =
            artifact::bind_device_tensor(binder, prefix + "pre_feedforward_norm",
                                         NumericFormat::BF16, {g.hidden});
        target.mlp.post_feedforward_norm =
            artifact::bind_device_tensor(binder, prefix + "post_feedforward_norm",
                                         NumericFormat::BF16, {g.hidden});
        target.attention.query = bind_weight(binder, prefix + "attention/query", weights,
                                             {static_cast<std::uint64_t>(query_rows),
                                              static_cast<std::uint64_t>(g.hidden)});
        target.attention.key   = bind_weight(binder, prefix + "attention/key", weights,
                                             {static_cast<std::uint64_t>(kv_rows),
                                              static_cast<std::uint64_t>(g.hidden)});
        // A `k_eq_v` global layer ships no value projection: its value is the key
        // projection's raw output, normalised without a weight. The artifact carries no
        // such object, so there is nothing to bind and the value path re-reads the key.
        target.attention.value_is_key = !windowed && g.attention_k_eq_v != 0;
        if (!target.attention.value_is_key) {
            target.attention.value = bind_weight(binder, prefix + "attention/value", weights,
                                                 {static_cast<std::uint64_t>(kv_rows),
                                                  static_cast<std::uint64_t>(g.hidden)});
        } else if (windowed) {
            // Only the global layers drop it. A windowed layer without a value projection
            // is an artifact this target cannot serve, and saying so here beats reading
            // the key rows twice and producing a model that is subtly wrong.
            throw std::runtime_error(
                "gemma4_moe: windowed layer " + std::to_string(layer) +
                " carries no attention/value; only a global layer may reuse its key");
        }
        // Per-head q/k norm, at this layer's head width. There is no v norm object: Gemma 4
        // normalises the value with no learnable scale at all, which is a kernel step
        // rather than a weight.
        target.attention.query_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/query_norm", NumericFormat::BF16, {head_dim});
        target.attention.key_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/key_norm", NumericFormat::BF16, {head_dim});
        target.attention.output = bind_weight(binder, prefix + "attention/output", weights,
                                              {static_cast<std::uint64_t>(g.hidden),
                                               static_cast<std::uint64_t>(query_rows)});
        const std::uint64_t dense = static_cast<std::uint64_t>(dense_intermediate(g));
        target.mlp.gate = bind_weight(binder, prefix + "mlp/gate", weights,
                                      {dense, static_cast<std::uint64_t>(g.hidden)});
        target.mlp.up   = bind_weight(binder, prefix + "mlp/up", weights,
                                      {dense, static_cast<std::uint64_t>(g.hidden)});
        target.mlp.down = bind_weight(binder, prefix + "mlp/down", weights,
                                      {static_cast<std::uint64_t>(g.hidden), dense});
        // The three norms the two branches meet under, then the routed branch itself. Bound in
        // the converter's own object order, so a diff of the two lists reads straight down.
        target.mlp.post_feedforward_norm_dense =
            artifact::bind_device_tensor(binder, prefix + "post_feedforward_norm_dense",
                                         NumericFormat::BF16, {g.hidden});
        target.mlp.pre_feedforward_norm_routed =
            artifact::bind_device_tensor(binder, prefix + "pre_feedforward_norm_routed",
                                         NumericFormat::BF16, {g.hidden});
        target.mlp.post_feedforward_norm_routed =
            artifact::bind_device_tensor(binder, prefix + "post_feedforward_norm_routed",
                                         NumericFormat::BF16, {g.hidden});
        target.mlp.router = artifact::bind_device_tensor(
            binder, prefix + "moe/router", NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.experts),
             static_cast<std::uint64_t>(g.hidden)});
        target.mlp.router_scale =
            artifact::bind_device_tensor(binder, prefix + "moe/router_scale",
                                         NumericFormat::BF16, {g.hidden});
        target.mlp.per_expert_scale = artifact::bind_device_tensor(
            binder, prefix + "moe/per_expert_scale", NumericFormat::FP32,
            {static_cast<std::uint64_t>(g.experts)});
        // The experts, and only the experts. The router and both scales stay resident: they
        // run for every token where an expert runs for one in sixteen, and together they are
        // 0.4 MB against the bank's 800.
        {
            target.mlp.routed_gate_up = artifact::bind_linear(
                binder, prefix + "moe/routed_gate_up", routed_gate_up_rows(g), g.hidden);
            target.mlp.routed_down = artifact::bind_linear(binder, prefix + "moe/routed_down",
                                                           routed_down_rows(g), g.intermediate);
        }
        target.mlp.layer_scalar = artifact::bind_device_tensor(
            binder, prefix + "layer_scalar", NumericFormat::BF16, {1});
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features, std::uint32_t host_moe_layers,
                               std::uint32_t gpu_layers, LoadProgress progress) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.geometry = family::TextGeometry::resolved_gemma4(
        binder.reader().geometry(), binder.reader().layer_types(), false, true);
    const family::TextGeometry& g = out.geometry;
    out.frontend = family::bind_text_only_frontend_resources(binder);
    out.features = features;

    if (features.vision) {
        // The 26B-A4B runs its vision through the same decoder stack rather than a tower, and
        // this target binds the text stack only.
        throw std::runtime_error(
            "gemma4_moe is a text-only target today: --vision is unsupported");
    }
    if (features.speculative_enabled()) {
        // Gemma 4 does publish a draft head -- every GGUF release ships an `mtp-*.gguf`
        // beside it -- but the safetensors checkpoint this target converts from carries
        // none, so the artifact has nothing to draft with.
        throw std::runtime_error(
            "gemma4_moe artifacts carry no draft head; run without --spec");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    // Stored unscaled. Gemma multiplies the looked-up row by sqrt(hidden) before the first
    // block, and the runtime holds that factor as the geometry's `embedding_scale`;
    // folding it in here would apply it twice.
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {static_cast<std::uint64_t>(g.output_rows),
                                       static_cast<std::uint64_t>(g.hidden)});
    bind_text_layers(binder, weights_profile, host_moe_layers, gpu_layers, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {g.hidden});
    // The head is the embedding table. Every published Gemma 4 ties them, so the converter
    // stores one table and names `text/output_head` a logical role on it (`ALIAS_SPECS` in
    // `surogate/serve/convert/gemma4_moe/inventory.py`). Binding it again would put a second
    // ~1 GB table on the device, and there is no second object to bind.
    //
    // `tie_word_embeddings` is a property of the checkpoint rather than of the
    // architecture, so an artifact converted from an untied export does store its own head
    // and this binds it where it is present.
    out.output_head = binder.has("text/output_head")
                          ? bind_weight(binder, "text/output_head", vocabulary_format,
                                        {static_cast<std::uint64_t>(g.output_rows),
                                         static_cast<std::uint64_t>(g.hidden)})
                          : out.token_embedding;

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
    // The layer storage is sized here, not by the type: the counts come from the geometry
    // these weights were bound against.
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;
    // Every layer of a dense decoder attends; windowed and global are both attention.
    runtime.full_layers.resize(static_cast<std::size_t>(g.layers));
    runtime.gdn_layers.resize(kGdnLayers);
    frontend = family::take_text_only_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;

    runtime.token_embedding =
        materialized_weight(backing, plan.token_embedding, g.output_rows, g.hidden);
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(g.layers); ++layer) {
        const TextLayerPlan& source  = plan.text_layers[layer];
        if (!source.resident) { continue; }
        FullAttentionWeights& target = runtime.full_layers.at(layer);
        const bool windowed          = source.attention.windowed;
        const std::int32_t head_dim  = g.head_dim_for(windowed);
        const std::int32_t query_rows = g.query_size_for(windowed);
        const std::int32_t kv_rows    = g.kv_size_for(windowed);

        target.input_norm = materialized_norm(backing, source.input_norm, g.hidden);
        target.projection = AttentionProjectionPayload{
            .query = materialized_weight(backing, source.attention.query, query_rows, g.hidden),
            .key   = materialized_weight(backing, source.attention.key, kv_rows, g.hidden),
            // Left empty where the value is the key: there is no object behind it, and
            // `value_is_key` is what the leaf reads rather than the emptiness.
            .value = source.attention.value_is_key
                         ? Weight{}
                         : materialized_weight(backing, source.attention.value, kv_rows,
                                               g.hidden),
            .post_attention_norm =
                materialized_norm(backing, source.attention.post_attention_norm, g.hidden),
            .windowed     = windowed,
            .value_is_key = source.attention.value_is_key,
            .head_dim     = head_dim,
            .rms_epsilon  = g.rms_epsilon,
        };
        target.query_norm = materialized_norm(backing, source.attention.query_norm, head_dim);
        target.key_norm   = materialized_norm(backing, source.attention.key_norm, head_dim);
        target.output =
            materialized_weight(backing, source.attention.output, g.hidden, query_rows);
        // The family's slot is named for where Llama's norm sits in the checkpoint; what
        // the runtime does with it is normalise the residual on the way into the
        // post-mixer. That is Gemma's `pre_feedforward_norm`, not its
        // `post_attention_layernorm` -- the latter normalises the attention block's
        // *output* and rides in the projection payload above. Swapping the two compiles,
        // loads, and produces a different model.
        target.post_attention_norm =
            materialized_norm(backing, source.pre_feedforward_norm, g.hidden);
        target.post_mixer = load_mlp(source.mlp, backing, g);
            target.post_mixer.banked = family::bind_banked_experts(
                target.post_mixer.op, host_bank.get(), source.mlp.routed_gate_up.object,
                source.mlp.routed_down.object, static_cast<std::int32_t>(layer), g.layers + g.mtp_layers);
    }
    static_assert(kGdnLayers == 0, "a Gemma 4 layer is never a linear mixer");

    runtime.final_norm = materialized_norm(backing, plan.final_norm, g.hidden);
    // Where the head is aliased, `plan.output_head` *is* `plan.token_embedding`, so this
    // reads back the one uploaded table rather than a second copy of it.
    runtime.output_head =
        materialized_weight(backing, plan.output_head, g.output_rows, g.hidden);
}

} // namespace sinfer::targets::gemma4_moe::detail
