#include <api/targets/qwen3_6_35b_a3b/package.h>
#include "family/impl/lora_bind.h"
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"
#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace sinfer::targets::qwen3_6_35b_a3b::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::qwen3_6_35b_a3b::detail

namespace sinfer::targets::qwen3_6_35b_a3b {
namespace {

constexpr ModelSamplingDefaults kQwen3_6_35BA3BDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
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
    if (model == model_id) { return kQwen3_6_35BA3BDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

std::uint32_t Package::maximum_context() noexcept { return detail::Variant::maximum_context; }

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::GroupwiseInt;
    }
    if (identity.model_id == model_id && identity.weights_id == "routed-nvfp4") {
        return WeightsProfile::RoutedNvfp4;
    }
    if (identity.model_id == model_id && identity.weights_id == "compressed-tensors") {
        return WeightsProfile::CompressedTensors;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, family::startup_features(options), weights_profile)));
}

SINFER_TARGET_CONSTRUCT_LOADED_MODEL();

namespace {

void bind_lora(const detail::RuntimeModelView& runtime, const EngineOptions& options) {
    family::bind_lora_moe_hybrid<detail::TextConfig>(
        runtime, options, [](std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; });
}

} // namespace

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    bind_lora(model.impl_->data.runtime, options);
    return family::make_frontend(model.impl_->data.frontend,
                                  family::FrontendOptions{
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
    return family::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    // The routed-NVFP4 experts run on the vendored TRT-LLM runner, whose grouped GEMMs pick a
    // tactic per round width by measurement. That measurement launches and synchronises, so it
    // has to happen before the program captures its decode graphs; every layer shares the
    // geometry and the tactic, so tuning against one layer's weights tunes them all.
    if (model.impl_->weights_profile == WeightsProfile::RoutedNvfp4) {
        ops::sparse_moe_prepare(model.impl_->data.runtime.gdn_layers.at(0).post_mixer.op,
                                ops::kSparseMoeTrtllmPrepareWidth, device.stream);
    }
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::qwen3_6_35b_a3b
