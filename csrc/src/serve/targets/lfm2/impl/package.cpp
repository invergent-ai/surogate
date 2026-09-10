#include <api/targets/lfm2/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "api/ops/lora.h"
#include "family/impl/lora_bind.h"
#include "api/ops/lora_store.h"
#include "family/impl/mlp_swiglu.h"
#include "artifact/reader.h"
#include "targets/lfm2/impl/load/bindings.h"
#include "targets/lfm2/impl/variant.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sinfer::targets::lfm2::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::lfm2::detail

namespace sinfer::targets::lfm2 {

namespace {
ops::SparseMoeGeometry offload_geometry(const family::TextGeometry& g) {
    ops::SparseMoeGeometry out{g.hidden, g.experts, g.experts_per_token, g.intermediate};
    out.gating = ops::SparseMoeGating::SigmoidBiasTopK;
    out.routed_scale = g.routed_scale;
    return out;
}
} // namespace
namespace {

// LFM2's published defaults (model card and generation_config.json): temperature 0.3, top_p
// 0.95, repetition penalty 1.05. It has no thinking mode, so both rows are the same numbers --
// a reader asking for the thinking preset of a model that has none should get the model's
// preset, not a different one invented for the occasion.
constexpr SamplingPreset kLfm2Preset{.temperature       = 0.3F,
                                     .top_k             = 0,
                                     .top_p             = 0.95F,
                                     .min_p             = 0.0F,
                                     .presence_penalty  = 0.0F,
                                     .frequency_penalty = 0.0F};
constexpr ModelSamplingDefaults kLfm2Defaults{.thinking = kLfm2Preset,
                                              .non_thinking = kLfm2Preset};

/// Bind full attention, short-convolution projections, dense MLPs and routed experts.
void bind_lora(const detail::RuntimeModelView& runtime, const EngineOptions& options) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    using TextConfig      = detail::TextConfig;
    ops::LoraStore& store = ops::lora_store_for_current_device();
    if (store.empty()) {
        // The widest round an adapter can see. This target refuses speculation, so a decode
        // round is one column per lane.
        const std::uint32_t widest =
            std::max<std::uint32_t>(std::max<std::uint32_t>(decode_batch_capacity(options.max_concurrency), 1),
                                    std::max<std::uint32_t>(options.prefill_chunk, 1));
        store.configure(static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_slots, 1)),
                        static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_max_rank, 1)),
                        static_cast<std::int32_t>(widest));
    }

    using Binding                 = ops::LoraStore::ModuleBinding;
    const family::TextGeometry& g = runtime.geometry;
    // An adapter names its modules by *model* layer, so the walk is over the model's layers and
    // each one reaches into whichever inventory holds it. Binding by position within
    // `full_layers` would put layer 2's adapter on layer 0.
    std::size_t full_index = 0;
    std::size_t conv_index = 0;
    for (std::int32_t index = 0; index < g.layers; ++index) {
        const bool attends = g.layer_attends(index);
        if ((attends ? runtime.full_layers.at(full_index).input_norm.data
                     : runtime.gdn_layers.at(conv_index).input_norm.data) == nullptr) {
            if (attends) { ++full_index; } else { ++conv_index; }
            continue;
        }
        const detail::DensePostMixerPayload& mlp =
            attends ? runtime.full_layers.at(full_index).post_mixer
                    : runtime.gdn_layers.at(conv_index).post_mixer;
        if (attends) {
            const auto& attention = runtime.full_layers.at(full_index++);
            const void* qkv       = attention.projection.query_key_value.qdata;
            store.register_module(index, "q_proj",
                                  Binding{qkv, family::kQueryPort, g.hidden, g.query_size()});
            store.register_module(index, "k_proj",
                                  Binding{qkv, family::kKeyPort, g.hidden, g.kv_size()});
            store.register_module(index, "v_proj",
                                  Binding{qkv, family::kValuePort, g.hidden, g.kv_size()});
            store.register_module(index, "o_proj",
                                  Binding{attention.output.qdata, family::kOutputPort,
                                          g.query_size(), g.hidden});
            store.register_module(index, "out_proj", {attention.output.qdata,
                family::kOutputPort, g.query_size(), g.hidden});
        } else {
            const auto& conv = runtime.gdn_layers.at(conv_index++);
            store.register_module(index, "conv.in_proj", {conv.projection.in_projection.qdata,
                16, g.hidden, 3 * g.hidden});
            store.register_module(index, "conv.out_proj", {conv.output.qdata,
                family::kGdnOutputPort, g.hidden, g.hidden});
            for (const char* module : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
                store.register_layer_refusal(
                    index, module,
                    "this layer is a short-convolution mixer and has no attention projections");
            }
        }
        if (mlp.moe.routed_down.n > 0) {
            family::bind_lora_moe(store, index, mlp.moe);
        } else {
            family::bind_lora_dense_mlp(store, index, mlp, g.hidden, g.intermediate);
            store.register_module(index, "feed_forward.w1", {mlp.gate_up.qdata, family::kGatePort, g.hidden, g.intermediate});
            store.register_module(index, "feed_forward.w3", {mlp.gate_up.qdata, family::kUpPort, g.hidden, g.intermediate});
            store.register_module(index, "feed_forward.w2", {mlp.down.qdata, family::kDownPort, g.intermediate, g.hidden});
        }
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
    if (std::find(model_ids.begin(), model_ids.end(), model) != model_ids.end()) { return kLfm2Defaults; }
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
                       static_cast<std::uint32_t>(options.pipeline_stage_first));
    auto plan = detail::bind_artifact(binder, weights_profile, family::startup_features(options));
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
            // Each LFM checkpoint supplies its own vocabulary and optional image tokens.
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
                                     planner.capacity_curve().minimum_device_reservation_bytes);
    return planner;
}

family::TextGeometry Package::declared_geometry(const artifact::Reader& reader) {
    // Dimensions *and* schedule: which layers attend is this family's per-checkpoint choice, and
    // it is read off the objects the artifact holds. See `declared_geometry_with_schedule`.
    return detail::declared_geometry_with_schedule(reader);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    family::prepare_banked_experts(model.impl_->data.runtime);
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::lfm2
