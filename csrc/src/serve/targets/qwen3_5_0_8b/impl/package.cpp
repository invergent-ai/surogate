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

    // Text layer index -> the full-attention layer holding it, since only every
    // fourth layer is full attention in this family.
    // Mirrors is_full_layer in this target's bindings: every fourth layer from 3.
    const auto is_full = [](std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; };
    std::vector<const detail::FullAttentionWeights*> by_layer(detail::TextConfig::layers, nullptr);
    std::size_t full_index = 0;
    for (std::size_t layer = 0; layer < detail::TextConfig::layers; ++layer) {
        if (is_full(layer)) { by_layer[layer] = &runtime.full_layers.at(full_index++); }
    }

    for (const auto& payload : options.lora_payloads) {
        if (payload.layer < 0 || static_cast<std::size_t>(payload.layer) >= by_layer.size() ||
            by_layer[static_cast<std::size_t>(payload.layer)] == nullptr) {
            throw std::invalid_argument(
                "--lora-modules: layer " + std::to_string(payload.layer) + " module '" +
                payload.module + "' is not a full-attention layer of this model");
        }
        const detail::FullAttentionWeights& layer = *by_layer[static_cast<std::size_t>(payload.layer)];
        const Weight* base                = nullptr;
        if (payload.module == "q_proj") {
            const auto* fused = std::get_if<detail::FusedAttentionProjectionPayload>(&layer.projection);
            if (fused == nullptr) {
                throw std::invalid_argument(
                    "--lora-modules: this artifact splits its attention projection, which the "
                    "q_proj adapter path does not bind");
            }
            base = &fused->query_key_gate_value;
        } else if (payload.module == "o_proj") {
            base = &layer.output;
        } else {
            throw std::invalid_argument(
                "--lora-modules: module '" + payload.module +
                "' is not applied by this target (it applies q_proj and o_proj); an adapter "
                "that is only partly applied would be neither the base model nor the fine-tune");
        }
        store.add(base->qdata, payload.a, payload.b, payload.rank, payload.in_dim,
                  payload.out_dim, payload.scale);
        // Every width a captured decode graph can present, prewarmed here.
        //
        // The delta's two cuBLASLt GEMMs cache a plan per problem shape, and a plan
        // built *during* capture corrupts the graph -- measurably: a B=0 adapter,
        // whose delta is exactly zero, then changed the output and changed it
        // differently on each run. Preparing only T=1 was why LoRA had to run eager.
        // A captured decode round carries one column per lane, times the verify
        // window when a draft is in flight, so the widths are small and few.
        const std::uint32_t lanes = std::max<std::uint32_t>(options.max_concurrency, 1);
        const std::uint32_t window =
            options.speculative.backend == SpeculativeBackend::None
                ? 1U
                : options.speculative.draft_tokens + 1U;
        for (std::uint32_t tokens = 1; tokens <= lanes * window; ++tokens) {
            ops::lora_prepare(payload.out_dim, payload.in_dim, payload.rank,
                              static_cast<std::int32_t>(tokens));
        }
    }
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
