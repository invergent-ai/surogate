#include <api/targets/qwen3_5_moe/package.h>
#include "family/impl/lora_bind.h"
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_5_moe/impl/load/bindings.h"
#include "targets/qwen3_5_moe/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace sinfer::targets::qwen3_5_moe::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::qwen3_5_moe::detail

namespace sinfer::targets::qwen3_5_moe {

namespace {
ops::SparseMoeGeometry offload_geometry(const family::TextGeometry& g) {
    ops::SparseMoeGeometry out{g.hidden, g.experts, g.experts_per_token, g.intermediate};
    out.shared_intermediate = g.shared_intermediate;
    return out;
}
} // namespace
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
    if (identity.architecture == target_key && identity.weights_id == "groupwise-int") {
        return WeightsProfile::GroupwiseInt;
    }
    if (identity.architecture == target_key && identity.weights_id == "routed-nvfp4") {
        return WeightsProfile::RoutedNvfp4;
    }
    if (identity.architecture == target_key && identity.weights_id == "compressed-tensors") {
        return WeightsProfile::CompressedTensors;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    binder.set_offload(options.resident_layer_limit(), options.host_moe_layers,
                       static_cast<std::uint32_t>(options.pipeline_stage_first));
    auto plan = detail::bind_artifact(binder, family::startup_features(options), weights_profile,
                              options.host_moe_layers, options.gpu_layers,
                              options.load_progress);
    const auto& g = plan.bindings.geometry;
    family::plan_banked_experts(binder, plan.bindings.host_bank, plan.materialization,
                                options, offload_geometry(g), g.layers + g.mtp_layers,
                                ops::LinearPolicy::AllowA4);
    return LoadPlan(std::make_unique<LoadPlan::Impl>(weights_profile, std::move(plan)));
}

SINFER_TARGET_CONSTRUCT_LOADED_MODEL();

namespace {

void bind_lora(const detail::RuntimeModelView& runtime, const EngineOptions& options) {
    family::bind_lora_moe_hybrid(
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
                                                        const family::TextGeometry& geometry,
                                                        const family::VisionGeometry& vision_geometry) {
    auto planner = family::make_sequence_planner<detail::Variant>(device, options, weights_profile,
                                                                  geometry, vision_geometry);
    family::configure_banked_experts(device, options, offload_geometry(geometry),
                                     geometry.layers + geometry.mtp_layers,
                                     planner.capacity_curve().minimum_device_reservation_bytes);
    return planner;
}

family::TextGeometry Package::declared_geometry(const artifact::Reader& reader) {
    return detail::resolved_geometry(reader);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    // The routed-NVFP4 experts run on the vendored TRT-LLM runner, whose grouped GEMMs pick a
    // tactic per round width by measurement. That measurement launches and synchronises, so it
    // has to happen before the program captures its decode graphs; every layer shares the
    // geometry and the tactic, so tuning against one layer's weights tunes them all.
    if (model.impl_->weights_profile == WeightsProfile::RoutedNvfp4) {
        const auto prepare = [&](const auto& layers) {
            for (const auto& layer : layers) {
                if (layer.post_mixer.op.routed_gate_up.qtype == QType::NVFP4) {
                    ops::sparse_moe_prepare(layer.post_mixer.op, ops::kSparseMoeTrtllmPrepareWidth, device.stream);
                    return;
                }
            }
        };
        prepare(model.impl_->data.runtime.gdn_layers);
        prepare(model.impl_->data.runtime.full_layers);
    }
    family::prepare_banked_experts(model.impl_->data.runtime);
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::qwen3_5_moe
