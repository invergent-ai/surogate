#include <api/targets/qwen3_5/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include <algorithm>

#include "family/impl/lora_bind.h"
#include "artifact/reader.h"
#include "targets/qwen3_5/impl/load/bindings.h"
#include "targets/qwen3_5/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace sinfer::targets::qwen3_5::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::qwen3_5::detail

namespace sinfer::targets::qwen3_5 {
namespace {

// The general-task presets published with these checkpoints. Every model this target serves
// ships the same pair, so they are stated once; a checkpoint that later differs gets its own
// entry and a branch in sampling_defaults.
constexpr ModelSamplingDefaults kFamilyDefaults{
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
    if (std::find(model_ids.begin(), model_ids.end(), model) != model_ids.end()) {
        return kFamilyDefaults;
    }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

std::uint32_t Package::maximum_context() noexcept { return detail::Variant::maximum_context; }

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    const bool family_model =
        std::find(model_ids.begin(), model_ids.end(), identity.model_id) != model_ids.end();
    if (!family_model) {
        throw std::runtime_error("artifact identity '" + identity.model_id + "/" +
                                 identity.weights_id + "' is not supported by target '" +
                                 std::string(target_key) + "'");
    }
    if (identity.weights_id == "groupwise-int") { return WeightsProfile::GroupwiseInt; }
    if (identity.weights_id == "nvfp4-mixed") { return WeightsProfile::Nvfp4Uniform; }
    if (identity.weights_id == "nvfp4-all") { return WeightsProfile::Nvfp4All; }
    if (identity.weights_id == "fp8-block") { return WeightsProfile::Fp8Block; }
    if (identity.weights_id == "nvfp4") {
        // Asked before the family-wide rule: the 3.8 export leaves attention and GDN in FP8
        // and only the MLP in NVFP4, which is a different set of objects to bind.
        return identity.model_id == qwen3_8_model_id ? WeightsProfile::Nvfp4MlpOnly
                                                     : WeightsProfile::Nvfp4MixedBf16;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

std::string_view Package::target_key_for(std::string_view model) noexcept {
    if (model == "qwen3.6-27b") { return "qwen3_6"; }
    if (model == "qwen3.8-27b") { return "qwen3_8"; }
    return "qwen3_5";
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, family::startup_features(options))));
}

SINFER_TARGET_CONSTRUCT_LOADED_MODEL();

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
    family::bind_lora_hybrid<detail::TextConfig, detail::FusedAttentionProjectionPayload>(
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
                                                        WeightsProfile weights_profile,
                                                        const family::TextGeometry& geometry) {
    return family::make_sequence_planner<detail::Variant>(device, options, weights_profile,
                                                         geometry);
}

family::TextGeometry Package::declared_geometry(const artifact::Reader& reader) {
    return family::TextGeometry::declared<detail::TextConfig>(reader.geometry());
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::qwen3_5
