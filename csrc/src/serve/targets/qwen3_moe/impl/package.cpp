#include <api/targets/qwen3_moe/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "api/ops/lora.h"
#include "family/impl/lora_bind.h"
#include "api/ops/lora_store.h"
#include "family/impl/mlp_swiglu.h"
#include "artifact/reader.h"
#include "targets/qwen3_moe/impl/load/bindings.h"
#include "targets/qwen3_moe/impl/variant.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sinfer::targets::qwen3_moe::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::qwen3_moe_moe::detail

namespace sinfer::targets::qwen3_moe {

namespace {
ops::SparseMoeGeometry offload_geometry(const family::TextGeometry& g) {
    ops::SparseMoeGeometry out{g.hidden, g.experts, g.experts_per_token, g.intermediate};
    return out;
}
} // namespace
namespace {

// Qwen3-MoE's published general-task presets (model card, "Best Practices"): the same numbers
// the dense Qwen3 models ship, which is what the shared model card states for both.
constexpr ModelSamplingDefaults kQwen3MoeDefaults{
    .thinking     = {.temperature       = 0.6F,
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

/// Bind ordinary attention projections and each named routed expert independently.
void bind_lora(const detail::RuntimeModelView& runtime, const EngineOptions& options) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    using TextConfig      = detail::TextConfig;
    ops::LoraStore& store = ops::lora_store_for_current_device();
    if (store.empty()) {
        // The widest round an adapter can see. This target refuses speculation,
        // so a decode round is one column per lane.
        const std::uint32_t widest =
            std::max<std::uint32_t>(std::max<std::uint32_t>(decode_batch_capacity(options.max_concurrency), 1),
                                    std::max<std::uint32_t>(options.prefill_chunk, 1));
        store.configure(static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_slots, 1)),
                        static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_max_rank, 1)),
                        static_cast<std::int32_t>(widest));
    }

    using Binding = ops::LoraStore::ModuleBinding;
    const family::TextGeometry& g = runtime.geometry;
    for (std::size_t layer = 0; layer < runtime.full_layers.size(); ++layer) {
        if (runtime.full_layers[layer].input_norm.data == nullptr) { continue; }
        const auto index      = static_cast<std::int32_t>(layer);
        const auto& attention = runtime.full_layers.at(layer);
        const void* qkv       = attention.projection.query_key_value.qdata;
        store.register_module(
            index, "q_proj",
            Binding{qkv, family::kQueryPort, g.hidden, g.query_size()});
        store.register_module(
            index, "k_proj",
            Binding{qkv, family::kKeyPort, g.hidden, g.kv_size()});
        store.register_module(
            index, "v_proj",
            Binding{qkv, family::kValuePort, g.hidden, g.kv_size()});
        store.register_module(index, "o_proj",
                              Binding{attention.output.qdata, family::kOutputPort,
                                      g.query_size(),
                                      g.hidden});
        family::bind_lora_moe(store, index, attention.post_mixer.op);
    }
    family::bind_lora_globals(store, runtime);
    store.validate_payloads(options.lora_payloads);
    store.ensure_banks();

    for (const auto& payload : options.lora_payloads) {
        // A pipeline hands every stage the whole list; each applies the layers it
        // holds and leaves the rest to the stage that does.
        if (!store.covers_layer(payload.layer)) { continue; }
        store.set_payload(payload.slot, payload);
    }
    ops::lora_set_active(true);
}

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (std::find(model_ids.begin(), model_ids.end(), model) != model_ids.end()) { return kQwen3MoeDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

std::uint32_t Package::maximum_context() noexcept { return detail::Variant::maximum_context; }

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (accepts_architecture(identity.architecture) && identity.weights_id == "groupwise-int") {
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
    auto plan = detail::bind_artifact(binder, weights_profile, family::startup_features(options),
                              options.host_moe_layers, options.gpu_layers,
                              options.load_progress);
    const auto& g = plan.bindings.geometry;
    family::plan_banked_experts(binder, plan.bindings.host_bank, plan.materialization,
                                options, offload_geometry(g), g.layers + g.mtp_layers,
                                ops::LinearPolicy::AllowA4);
    return LoadPlan(std::make_unique<LoadPlan::Impl>(weights_profile, std::move(plan)));
}

SINFER_TARGET_CONSTRUCT_LOADED_MODEL();

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    bind_lora(model.impl_->data.runtime, options);
    return family::make_frontend(
        model.impl_->data.frontend,
        family::FrontendOptions{
            .vision_enabled = model.impl_->data.runtime.features.vision,
            .max_context    = options.max_context,
            .media_cache_bytes        = options.media_cache_bytes,
            .media_live_bytes         = options.media_live_bytes,
            .media_preprocess_threads = options.media_preprocess_threads,
            .chat_template_override   = options.chat_template_override,
            // Qwen3's tokenizer is its own 151,669-id domain with no vision tokens; the
            // family's registered-checkpoint assertions describe a different one entirely.
            .registered_tokenizer = false,
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
                                     planner.capacity_curve());
    return planner;
}

family::TextGeometry Package::declared_geometry(const artifact::Reader& reader) {
    return family::TextGeometry::resolved_moe(reader.geometry(), reader.layer_types());
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    family::prepare_banked_experts(model.impl_->data.runtime);
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::qwen3_moe
