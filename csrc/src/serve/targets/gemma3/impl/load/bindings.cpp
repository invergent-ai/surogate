#include "targets/gemma3/impl/load/bindings.h"

#include "targets/gemma3/impl/config.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sinfer::targets::gemma3_270m::detail {
namespace {

using artifact::NumericFormat;

static_assert(TextConfig::query_projection_rows == TextConfig::query_size,
              "Gemma 3 attention is ungated; a gated projection would be read at the wrong stride");
static_assert(TextConfig::query_size == 1024);
static_assert(TextConfig::kv_size == 256);

/// The four resources a text-only Gemma 3 artifact carries, matching
/// `RESOURCE_SPECS` in `surogate/serve/tools/convert/gemma3/inventory.py`. The
/// family's binder demands six; the two it adds are the image and video
/// preprocessor configs, which gemma-3-270m-it does not publish. The unfilled
/// halves of the plan stay default-constructed and
/// `take_text_only_frontend_resources` leaves their strings empty, which is what
/// the frontend reads as "no pixel pipeline".
///
/// The SentencePiece `tokenizer.model` beside them is deliberately not bound.
/// The frontend hands SentencePiece checkpoints to the project tokenizer, but it
/// still reads the scheme, the vocabulary and the added tokens out of
/// `tokenizer.json` -- exactly as the Llama target does with its own
/// SentencePiece release -- and an object no binder consumes is refused at load.
family::FrontendResourcePlan bind_text_only_frontend_resources(artifact::Binder& binder) {
    family::FrontendResourcePlan plan;
    plan.tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json");
    plan.tokenizer_config_json =
        artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json");
    plan.chat_template_jinja = artifact::bind_raw_resource(binder, "frontend/chat_template.jinja");
    plan.generation_config_json =
        artifact::bind_raw_resource(binder, "frontend/generation_config.json");
    return plan;
}

std::string take_resource_string(artifact::MaterializedArtifact& materialized,
                                 artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

family::FrontendResources
take_text_only_frontend_resources(artifact::MaterializedArtifact& materialized,
                                  const family::FrontendResourcePlan& plan) {
    family::FrontendResources out;
    out.tokenizer_json         = take_resource_string(materialized, plan.tokenizer_json);
    out.tokenizer_config_json  = take_resource_string(materialized, plan.tokenizer_config_json);
    out.chat_template_jinja    = take_resource_string(materialized, plan.chat_template_jinja);
    out.generation_config_json = take_resource_string(materialized, plan.generation_config_json);
    return out;
}

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("gemma3: invalid weights profile");
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

Tensor materialized_norm(const artifact::MaterializedArtifact& materialized,
                         artifact::ObjectHandle handle, std::int32_t width) {
    return artifact::materialized_tensor(materialized, handle, NumericFormat::BF16, {width});
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate = materialized_weight(materialized, plan.gate, TextConfig::intermediate,
                                   TextConfig::hidden);
    out.up   = materialized_weight(materialized, plan.up, TextConfig::intermediate,
                                   TextConfig::hidden);
    out.down = materialized_weight(materialized, plan.down, TextConfig::hidden,
                                   TextConfig::intermediate);
    out.post_feedforward_norm =
        materialized_norm(materialized, plan.post_feedforward_norm, TextConfig::hidden);
    return out;
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile, BindingPlan& out) {
    const NumericFormat weights = endpoint_format(weights_profile);
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        // Bound in the converter's own order (inventory.py), so a diff of the two
        // lists reads straight down.
        target.input_norm = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                         NumericFormat::BF16, {TextConfig::hidden});
        target.attention.post_attention_norm =
            artifact::bind_device_tensor(binder, prefix + "post_attention_norm",
                                         NumericFormat::BF16, {TextConfig::hidden});
        target.pre_feedforward_norm =
            artifact::bind_device_tensor(binder, prefix + "pre_feedforward_norm",
                                         NumericFormat::BF16, {TextConfig::hidden});
        target.mlp.post_feedforward_norm =
            artifact::bind_device_tensor(binder, prefix + "post_feedforward_norm",
                                         NumericFormat::BF16, {TextConfig::hidden});
        // Three matrices, not one fused parent: q is [1024, 640] and k and v are
        // [256, 640] each. Gemma 3 also carries no qkv bias (`attention_bias` is
        // false in every released config), which is why no bias object is bound.
        target.attention.query = bind_weight(binder, prefix + "attention/query", weights,
                                             {TextConfig::query_size, TextConfig::hidden});
        target.attention.key   = bind_weight(binder, prefix + "attention/key", weights,
                                             {TextConfig::kv_size, TextConfig::hidden});
        target.attention.value = bind_weight(binder, prefix + "attention/value", weights,
                                             {TextConfig::kv_size, TextConfig::hidden});
        // Per-head q/k norm over head_dim, as Qwen3 has and Llama does not. There
        // is no v norm -- that is Gemma 4's, and this checkpoint ships no such
        // tensor.
        target.attention.query_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
        target.attention.key_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
        target.attention.output = bind_weight(binder, prefix + "attention/output", weights,
                                              {TextConfig::hidden, TextConfig::query_size});
        target.mlp.gate = bind_weight(binder, prefix + "mlp/gate", weights,
                                      {TextConfig::intermediate, TextConfig::hidden});
        target.mlp.up   = bind_weight(binder, prefix + "mlp/up", weights,
                                      {TextConfig::intermediate, TextConfig::hidden});
        target.mlp.down = bind_weight(binder, prefix + "mlp/down", weights,
                                      {TextConfig::hidden, TextConfig::intermediate});
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = bind_text_only_frontend_resources(binder);
    out.features     = features;

    if (features.vision) {
        // Gemma 3 has vision-capable sizes; 270M is not one of them. The
        // checkpoint is `Gemma3ForCausalLM`, not `Gemma3ForConditionalGeneration`,
        // and the artifact carries no tower.
        throw std::runtime_error("gemma3-270m is a text-only target: --vision is unsupported");
    }
    if (features.speculative_enabled()) {
        // Gemma 3 ships no MTP block and the target declares no DFlash tower, so
        // there is nothing to draft with. Refusing here beats a missing-object
        // failure twenty objects later.
        throw std::runtime_error("gemma3-270m carries no draft head (no MTP block, no DFlash "
                                 "tower); run without --spec");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    // Stored unscaled. Gemma multiplies the looked-up row by sqrt(hidden) before
    // the first block; the family applies that from
    // `TextConfig::embedding_scale`, so folding it into the table here would
    // apply it twice.
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {TextConfig::output_rows, TextConfig::hidden});
    bind_text_layers(binder, weights_profile, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {TextConfig::hidden});
    // The head is the embedding table. Gemma 3 ties them and ships no
    // `lm_head.weight` at all, so the converter stores one table and names
    // `text/output_head` a logical role on it -- `ALIAS_SPECS` in
    // `surogate/serve/tools/convert/gemma3/inventory.py`, the same shape as
    // `mtp/token_embedding` on the qwen3_5_0_8b target. Binding it a second time
    // would put two ~168 MB tables on a device whose whole model is ~270 MB, and
    // there is no second object to bind: the artifact does not carry one.
    //
    // `tie_word_embeddings` is a property of the checkpoint rather than of the
    // architecture, so an artifact converted from an untied export does store its
    // own head, and this binds it where it is present. The probe is what lets one
    // binder read both artifact shapes; an object no binder consumes is refused at
    // load, so the two cannot be collapsed into an unconditional bind.
    out.output_head = binder.has("text/output_head")
                          ? bind_weight(binder, "text/output_head", vocabulary_format,
                                        {TextConfig::output_rows, TextConfig::hidden})
                          : out.token_embedding;

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = take_text_only_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;

    runtime.token_embedding = materialized_weight(backing, plan.token_embedding,
                                                  TextConfig::output_rows, TextConfig::hidden);
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source  = plan.text_layers[layer];
        FullAttentionWeights& target = runtime.full_layers.at(layer);
        target.input_norm = materialized_norm(backing, source.input_norm, TextConfig::hidden);
        target.projection = AttentionProjectionPayload{
            .query = materialized_weight(backing, source.attention.query, TextConfig::query_size,
                                         TextConfig::hidden),
            .key   = materialized_weight(backing, source.attention.key, TextConfig::kv_size,
                                         TextConfig::hidden),
            .value = materialized_weight(backing, source.attention.value, TextConfig::kv_size,
                                         TextConfig::hidden),
            .post_attention_norm = materialized_norm(
                backing, source.attention.post_attention_norm, TextConfig::hidden),
        };
        target.query_norm =
            materialized_norm(backing, source.attention.query_norm, TextConfig::head_dim);
        target.key_norm =
            materialized_norm(backing, source.attention.key_norm, TextConfig::head_dim);
        target.output = materialized_weight(backing, source.attention.output, TextConfig::hidden,
                                            TextConfig::query_size);
        // The family's slot is named for where Llama's norm sits in the
        // checkpoint; what the runtime does with it is normalise the residual on
        // the way into the post-mixer. That is Gemma's `pre_feedforward_norm`, not
        // its `post_attention_layernorm` -- the latter normalises the attention
        // block's *output* and rides in the projection payload above. Swapping
        // the two compiles, loads, and produces a different model.
        target.post_attention_norm = materialized_norm(backing, source.pre_feedforward_norm,
                                                       TextConfig::hidden);
        target.post_mixer          = load_mlp(source.mlp, backing);
    }
    static_assert(kGdnLayers == 0, "a Gemma 3 layer is never a linear mixer");

    runtime.final_norm = materialized_norm(backing, plan.final_norm, TextConfig::hidden);
    // Where the head is aliased, `plan.output_head` *is* `plan.token_embedding`,
    // so this reads back the one uploaded table rather than a second copy of it.
    runtime.output_head = materialized_weight(backing, plan.output_head, TextConfig::output_rows,
                                              TextConfig::hidden);
}

} // namespace sinfer::targets::gemma3_270m::detail
