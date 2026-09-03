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

/// The four resources a text-only Gemma 3 artifact carries, matching
/// `RESOURCE_SPECS` in `surogate/serve/convert/gemma3/inventory.py`. The
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
                               const artifact::MaterializedArtifact& materialized,
                               const family::TextGeometry& g) {
    DensePostMixerPayload out;
    out.gate = materialized_weight(materialized, plan.gate, g.intermediate,
                                   g.hidden);
    out.up   = materialized_weight(materialized, plan.up, g.intermediate,
                                   g.hidden);
    out.down = materialized_weight(materialized, plan.down, g.hidden,
                                   g.intermediate);
    out.post_feedforward_norm =
        materialized_norm(materialized, plan.post_feedforward_norm, g.hidden);
    return out;
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile, BindingPlan& out) {
    const family::TextGeometry& g = out.geometry;
    out.text_layers.resize(static_cast<std::size_t>(g.layers));
    const NumericFormat weights = endpoint_format(weights_profile);
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(g.layers); ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        // Bound in the converter's own order (inventory.py), so a diff of the two
        // lists reads straight down.
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
        // Three matrices, not one fused parent: q is [1024, 640] and k and v are
        // [256, 640] each. Gemma 3 also carries no qkv bias (`attention_bias` is
        // false in every released config), which is why no bias object is bound.
        target.attention.query = bind_weight(binder, prefix + "attention/query", weights,
                                             {g.query_size(), g.hidden});
        target.attention.key   = bind_weight(binder, prefix + "attention/key", weights,
                                             {g.kv_size(), g.hidden});
        target.attention.value = bind_weight(binder, prefix + "attention/value", weights,
                                             {g.kv_size(), g.hidden});
        // Per-head q/k norm over head_dim, as Qwen3 has and Llama does not. There
        // is no v norm -- that is Gemma 4's, and this checkpoint ships no such
        // tensor.
        target.attention.query_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/query_norm", NumericFormat::BF16, {g.head_dim});
        target.attention.key_norm = artifact::bind_device_tensor(
            binder, prefix + "attention/key_norm", NumericFormat::BF16, {g.head_dim});
        target.attention.output = bind_weight(binder, prefix + "attention/output", weights,
                                              {g.hidden, g.query_size()});
        target.mlp.gate = bind_weight(binder, prefix + "mlp/gate", weights,
                                      {g.intermediate, g.hidden});
        target.mlp.up   = bind_weight(binder, prefix + "mlp/up", weights,
                                      {g.intermediate, g.hidden});
        target.mlp.down = bind_weight(binder, prefix + "mlp/down", weights,
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
                                      {g.output_rows, g.hidden});
    bind_text_layers(binder, weights_profile, out);
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {g.hidden});
    // The head is the embedding table. Gemma 3 ties them and ships no
    // `lm_head.weight` at all, so the converter stores one table and names
    // `text/output_head` a logical role on it -- `ALIAS_SPECS` in
    // `surogate/serve/convert/gemma3/inventory.py`, the same shape as
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
                                        {g.output_rows, g.hidden})
                          : out.token_embedding;

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
        target.input_norm = materialized_norm(backing, source.input_norm, g.hidden);
        target.projection = AttentionProjectionPayload{
            .query = materialized_weight(backing, source.attention.query, g.query_size(),
                                         g.hidden),
            .key   = materialized_weight(backing, source.attention.key, g.kv_size(),
                                         g.hidden),
            .value = materialized_weight(backing, source.attention.value, g.kv_size(),
                                         g.hidden),
            .post_attention_norm = materialized_norm(
                backing, source.attention.post_attention_norm, g.hidden),
        };
        target.query_norm =
            materialized_norm(backing, source.attention.query_norm, g.head_dim);
        target.key_norm =
            materialized_norm(backing, source.attention.key_norm, g.head_dim);
        target.output = materialized_weight(backing, source.attention.output, g.hidden,
                                            g.query_size());
        // The family's slot is named for where Llama's norm sits in the
        // checkpoint; what the runtime does with it is normalise the residual on
        // the way into the post-mixer. That is Gemma's `pre_feedforward_norm`, not
        // its `post_attention_layernorm` -- the latter normalises the attention
        // block's *output* and rides in the projection payload above. Swapping
        // the two compiles, loads, and produces a different model.
        target.post_attention_norm = materialized_norm(backing, source.pre_feedforward_norm,
                                                       g.hidden);
        target.post_mixer          = load_mlp(source.mlp, backing, g);
    }
    static_assert(kGdnLayers == 0, "a Gemma 3 layer is never a linear mixer");

    runtime.final_norm = materialized_norm(backing, plan.final_norm, g.hidden);
    // Where the head is aliased, `plan.output_head` *is* `plan.token_embedding`,
    // so this reads back the one uploaded table rather than a second copy of it.
    runtime.output_head = materialized_weight(backing, plan.output_head, g.output_rows,
                                              g.hidden);
}

} // namespace sinfer::targets::gemma3_270m::detail
