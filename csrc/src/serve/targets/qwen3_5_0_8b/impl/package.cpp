#include <api/targets/qwen3_5_0_8b/package.h>
#include <api/targets/qwen3_6/frontend_resources.h>
#include <api/targets/qwen3_6/prepared_prompt.h>

#include <algorithm>

#include "api/ops/lora_store.h"
#include "artifact/reader.h"
#include "targets/qwen3_5_0_8b/impl/load/bindings.h"
#include "targets/qwen3_5_0_8b/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace sinfer::targets::qwen3_5_0_8b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace sinfer::targets::qwen3_5_0_8b::detail

namespace sinfer::targets::qwen3_5_0_8b {
namespace {

// General-task presets published with each exact model. Keep the registrations separate even
// while their values agree so an upstream model-specific change has one obvious owner.
constexpr ModelSamplingDefaults kQwen3_6Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

constexpr ModelSamplingDefaults kQwen3_8Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_6Defaults; }
    if (model == qwen3_8_model_id) { return kQwen3_8Defaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

std::uint32_t Package::maximum_context() noexcept { return detail::Variant::maximum_context; }

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::Qwen36GroupwiseInt;
    }
    if (identity.model_id == qwen3_8_model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::Qwen38GroupwiseInt;
    }
    if (identity.model_id == model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::Qwen36Nvfp4;
    }
    if (identity.model_id == qwen3_8_model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::Qwen38Nvfp4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(options))));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

namespace {

/// Binds decoded adapter payloads to this target's weights.
///
/// Keyed by the base weight's device pointer, because that is what the projection
/// hooks can see. `q_proj` adapts the fused attention projection's query output --
/// `attn_input_proj` scatters q/k/v into separate contiguous tensors, so the
/// delta needs no strided add -- and `o_proj` adapts the attention output.
///
/// Every payload must find a home. A module this target cannot place is refused
/// with its name: an adapter half-applied is a model that is neither the base nor
/// the fine-tune, and it would answer fluently either way.
void bind_lora(const detail::RuntimeModelView& runtime, const EngineOptions& options) {
    if (options.lora_payloads.empty()) { return; }
    ops::LoraStore& store = ops::lora_store_for_current_device();
    if (store.empty()) {
        // The widest round an adapter can see: one column per lane, times the
        // verify window when a draft is in flight.
        const std::uint32_t window = options.speculative.backend == SpeculativeBackend::None
                                         ? 1U
                                         : options.speculative.draft_tokens + 1U;
        // The widest round an adapter can see is a prefill chunk, not a decode
        // batch: sizing this for lanes alone left every prompt without a scratch,
        // so the prompt was built on the base model while the generated tokens
        // carried the delta -- an adapter that looked weak rather than unapplied.
        const std::uint32_t decode_columns =
            std::max<std::uint32_t>(options.max_concurrency, 1) * window;
        const std::uint32_t widest =
            std::max<std::uint32_t>(decode_columns, std::max<std::uint32_t>(options.prefill_chunk, 1));
        store.configure(static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_slots, 1)),
                        static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_max_rank, 1)),
                        static_cast<std::int32_t>(widest));
    }

    // Text layer index -> that layer's weights. Attention modules exist only on
    // the full-attention layers (every fourth from 3 in this family); the MLP
    // exists on every layer, so down_proj must route through GDN layers too -- a
    // real PEFT adapter carries it on all of them.
    const auto is_full = [](std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; };
    std::vector<const detail::FullAttentionWeights*> attention_by_layer(detail::TextConfig::layers,
                                                                        nullptr);
    std::vector<const detail::DensePostMixerPayload*> mlp_by_layer(detail::TextConfig::layers,
                                                                   nullptr);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < detail::TextConfig::layers; ++layer) {
        if (is_full(layer)) {
            const detail::FullAttentionWeights& full = runtime.full_layers.at(full_index++);
            attention_by_layer[layer]                = &full;
            mlp_by_layer[layer]                      = &full.post_mixer;
        } else {
            mlp_by_layer[layer] = &runtime.gdn_layers.at(gdn_index++).post_mixer;
        }
    }

    for (const auto& payload : options.lora_payloads) {
        if (payload.layer < 0 ||
            static_cast<std::size_t>(payload.layer) >= mlp_by_layer.size()) {
            throw std::invalid_argument("--lora-modules: layer " + std::to_string(payload.layer) +
                                        " is outside this model");
        }
        const detail::FullAttentionWeights* attention =
            attention_by_layer[static_cast<std::size_t>(payload.layer)];
        const Weight* base    = nullptr;
        std::int32_t port     = 0;
        // q, k and v leave one fused projection as separate contiguous tensors, so
        // they share a base weight and are told apart by the port. o and down have
        // a weight each.
        if (payload.module == "q_proj" || payload.module == "k_proj" ||
            payload.module == "v_proj" || payload.module == "o_proj") {
            if (attention == nullptr) {
                throw std::invalid_argument(
                    "--lora-modules: layer " + std::to_string(payload.layer) + " module '" +
                    payload.module +
                    "' -- this layer is linear attention, which has no self_attn projections; an "
                    "adapter naming them here was trained against a different architecture");
            }
            if (payload.module == "o_proj") {
                base = &attention->output;
                port = 3;
            } else {
                const auto* fused =
                    std::get_if<detail::FusedAttentionProjectionPayload>(&attention->projection);
                if (fused == nullptr) {
                    throw std::invalid_argument(
                        "--lora-modules: this artifact splits its attention projection, which the "
                        "attention adapter path does not bind");
                }
                base = &fused->query_key_gate_value;
                port = payload.module == "q_proj" ? 0 : (payload.module == "k_proj" ? 1 : 2);
            }
        } else if (payload.module == "down_proj") {
            base = &mlp_by_layer[static_cast<std::size_t>(payload.layer)]->down;
            port = 4;
        } else if (payload.module == "gate_proj" || payload.module == "up_proj") {
            // gate and up are one fused weight whose two halves are consumed by
            // SwiGLU inside the op, so neither half exists as a tensor a delta
            // could be added to. Adapting them needs the delta inside that op, not
            // after it; refusing is better than applying half an adapter.
            throw std::invalid_argument(
                "--lora-modules: '" + payload.module +
                "' is not applied yet -- gate and up are fused and consumed by SwiGLU inside the "
                "projection, so there is no intermediate tensor to add a delta to");
        } else {
            throw std::invalid_argument(
                "--lora-modules: module '" + payload.module +
                "' is not applied by this target (q_proj, k_proj, v_proj, o_proj, down_proj); an "
                "adapter only partly applied is neither the base model nor the fine-tune");
        }
        store.set_slot(base->qdata, port, payload.slot, payload.a, payload.b, payload.rank,
                       payload.in_dim, payload.out_dim, payload.scale);
    }
    ops::lora_set_active(true);
}

} // namespace

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    bind_lora(model.impl_->data.runtime, options);
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled = model.impl_->data.runtime.features.vision,
                                      .max_context    = options.max_context,
                                      .media_cache_bytes        = options.media_cache_bytes,
                                      .media_live_bytes         = options.media_live_bytes,
                                      .media_preprocess_threads = options.media_preprocess_threads,
                                      .chat_template_override = options.chat_template_override,
                                  });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::qwen3_5_0_8b
