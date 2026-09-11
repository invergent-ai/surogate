#include <api/targets/spark2_5/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "artifact/reader.h"
#include "family/impl/lora_bind.h"
#include "targets/spark2_5/impl/load/bindings.h"
#include "targets/spark2_5/impl/variant.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sinfer::targets::spark2_5::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::spark2_5::detail

namespace sinfer::targets::spark2_5 {
namespace {

constexpr ModelSamplingDefaults kSparkDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 0,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 1.0F,
                     .top_k             = 0,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};


} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kSparkDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

std::uint32_t Package::maximum_context() noexcept { return detail::Variant::maximum_context; }

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.architecture == target_key && identity.weights_id == "groupwise-int") {
        return WeightsProfile::GroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    binder.set_layer_range(options.pipeline_stage_first, options.pipeline_stage_last);
    binder.set_offload(options.resident_layer_limit(), options.host_moe_layers,
                       static_cast<std::uint32_t>(options.pipeline_stage_first),
                       options.offload_vision, options.offload_embeddings, options.offload_output_head);
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, family::startup_features(options))));
}

SINFER_TARGET_CONSTRUCT_LOADED_MODEL();

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (options.lora_enable || !options.lora_payloads.empty()) {
        auto& store = ops::lora_store_for_current_device();
        family::configure_lora_store(store, options);
        const auto& runtime = model.impl_->data.runtime;
        const auto& g = runtime.geometry;
        for (std::size_t layer = 0; layer < runtime.full_layers.size(); ++layer) {
        if (runtime.full_layers[layer].input_norm.data == nullptr) { continue; }
            const auto& full = runtime.full_layers[layer];
            const auto* qkv = full.projection.query_key_value.qdata;
            const int total = g.query_size() + 2 * g.kv_size();
            store.register_module(layer, "q_k_v_proj", {qkv, family::kQueryPort, g.hidden,
                g.query_size(), total, 0});
            store.register_module(layer, "q_k_v_proj", {qkv, family::kKeyPort, g.hidden,
                g.kv_size(), total, g.query_size()});
            store.register_module(layer, "q_k_v_proj", {qkv, family::kValuePort, g.hidden,
                g.kv_size(), total, g.query_size() + g.kv_size()});
            store.register_module(layer, "g_proj", {full.projection.output_gate.qdata,
                family::kAttentionGatePort, g.hidden, g.query_heads});
            store.register_module(layer, "out_proj", {full.output.qdata, family::kOutputPort,
                g.query_size(), g.hidden});
            family::bind_lora_dense_mlp(store, layer, full.post_mixer, g.hidden, g.intermediate);
        }
        family::bind_lora_globals(store, runtime);
        family::finish_lora_bind(store, options);
    }
    return family::make_frontend(
        model.impl_->data.frontend,
        family::FrontendOptions{
            .vision_enabled = false,
            .max_context    = options.max_context,
            .media_cache_bytes        = options.media_cache_bytes,
            .media_live_bytes         = options.media_live_bytes,
            .media_preprocess_threads = options.media_preprocess_threads,
            .chat_template_override   = options.chat_template_override,
            .registered_tokenizer = false,
        });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile,
                                                        const family::TextGeometry& geometry,
                                                        const family::VisionGeometry& vision_geometry) {
    return family::make_sequence_planner<detail::Variant>(device, options, weights_profile,
                                                         geometry, vision_geometry);
}

family::TextGeometry Package::declared_geometry(const artifact::Reader& reader) {
    return family::TextGeometry::resolved(reader.geometry(), reader.layer_types());
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::spark2_5
