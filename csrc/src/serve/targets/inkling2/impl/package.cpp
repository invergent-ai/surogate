// Inkling-2-Mini package (lane e1): the registry's questions for the `inkling2` artifact. Bindings are lane
// b's (`impl/load/`); the served geometry, the model view and the visit schedule are this target's.

#include <api/targets/inkling2/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>
#include <api/family/startup_features.h>

#include "artifact/reader.h"
#include "targets/inkling2/impl/model.h"
#include "targets/inkling2/impl/variant.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace sinfer::targets::inkling2::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::inkling2::detail

namespace sinfer::targets::inkling2 {
namespace {

/// `generation_config.json` of the release states do_sample and temperature 1.0 only; the same preset for
/// both modes (Inkling's reasoning is effort-controlled, not a separate mode).
constexpr ModelSamplingDefaults kInklingDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 0,
                     .top_p             = 1.0F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 1.0F,
                     .top_k             = 0,
                     .top_p             = 1.0F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};

/// SUROGATE_SERVE_INKLING_ATTENTION = bf16 (default until DESIGN 10.1 names the scheme) | fp8-<scheme>.
load::AttentionChoice attention_choice() {
    load::AttentionChoice choice;
    const char* raw = std::getenv("SUROGATE_SERVE_INKLING_ATTENTION");
    if (raw == nullptr || std::string_view(raw).empty() || std::string_view(raw) == "bf16") { return choice; }
    const std::string_view value(raw);
    if (value.substr(0, 4) != "fp8-") {
        throw std::invalid_argument("SUROGATE_SERVE_INKLING_ATTENTION must be bf16 or fp8-<scheme>");
    }
    choice.fp8    = true;
    choice.scheme = std::string(value.substr(4));
    return choice;
}

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kInklingDefaults; }
    throw std::runtime_error("model '" + std::string(model) + "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

std::uint32_t Package::maximum_context() noexcept { return detail::Variant::maximum_context; }

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.architecture != target_key || identity.weights_id.rfind("release+attn-bf16", 0) != 0) {
        throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                                 "' is not supported by target '" + std::string(target_key) + "'");
    }
    return attention_choice().fp8 ? WeightsProfile::Fp8Attention : WeightsProfile::Bf16Attention;
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    if (options.pipeline_stage_first != 0 || options.pipeline_stage_last != 0) {
        // The visit split (DESIGN-target.md 1.4, DESIGN.md S12) is not wired yet.
        throw std::invalid_argument("inkling2: pipeline stages are not supported yet");
    }
    load::BindOptions bind;
    bind.attention = attention_choice();
    if (bind.attention.fp8 != (weights_profile == WeightsProfile::Fp8Attention)) {
        throw std::logic_error("inkling2: the weights profile and the attention choice disagree");
    }
    bind.draft = false; // DSpark: lane e2
    detail::ArtifactLoadPlan plan;
    plan.bindings.plan     = load::bind_artifact(binder, bind);
    plan.bindings.geometry = detail::visit_geometry(binder.reader());
    plan.bindings.features = family::startup_features(options);
    plan.materialization   = binder.finish();
    return LoadPlan(std::make_unique<LoadPlan::Impl>(weights_profile, std::move(plan)));
}

SINFER_TARGET_CONSTRUCT_LOADED_MODEL();

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    // The generic tokenizer frontend: token-id prompts and plain text until lane f2's native renderer is wired.
    return family::make_frontend(model.impl_->data.frontend,
                                 family::FrontendOptions{
                                     .vision_enabled           = false,
                                     .max_context              = options.max_context,
                                     .media_cache_bytes        = options.media_cache_bytes,
                                     .media_live_bytes         = options.media_live_bytes,
                                     .media_preprocess_threads = options.media_preprocess_threads,
                                     .chat_template_override   = options.chat_template_override,
                                     .registered_tokenizer     = false,
                                 });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                                                        WeightsProfile weights_profile,
                                                        const family::TextGeometry& geometry,
                                                        const family::VisionGeometry& vision_geometry) {
    return family::make_sequence_planner<detail::Variant>(device, options, weights_profile, geometry, vision_geometry);
}

family::TextGeometry Package::declared_geometry(const artifact::Reader& reader) {
    return detail::visit_geometry(reader);
}

std::unique_ptr<Package::Program> Package::create_program(const LoadedModel& model, SequencePlan&& plan,
                                                          DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    // Lane c's key-group size: P3 by default (agent-EN 12:5x, until lane h's price session decides); the
    // environment still overrides (p1 / p2 / p3).
    ::setenv("SUROGATE_INKLING_ATTN_GROUPS", "p3", 0);
    detail::Variant::prepare_lm_head(16);
    return family::create_program<detail::Variant>(model.impl_->data.runtime, model.impl_->weights_profile,
                                                   std::move(plan), device);
}

} // namespace sinfer::targets::inkling2
