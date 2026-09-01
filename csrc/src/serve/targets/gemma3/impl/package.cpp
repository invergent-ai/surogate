#include <api/targets/gemma3/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "api/ops/lora.h"
#include "api/ops/lora_store.h"
#include "artifact/reader.h"
#include "targets/gemma3/impl/load/bindings.h"
#include "targets/gemma3/impl/variant.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sinfer::targets::gemma3_270m::detail {

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

} // namespace sinfer::targets::gemma3_270m::detail

namespace sinfer::targets::gemma3_270m {
namespace {

// gemma-3-270m-it's published presets, as its `generation_config.json` states
// them. Gemma 3 has no thinking mode -- its chat template renders no reasoning
// turn -- so both entries are the same preset rather than two invented ones.
constexpr ModelSamplingDefaults kGemma3Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 64,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 1.0F,
                     .top_k             = 64,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};

/// The dense flavor of the family's adapter directory.
///
/// `bind_lora_hybrid` cannot serve this target: it reads a fused payload member
/// named `query_key_gate_value`, and reserves a branch for the linear layers
/// this model does not have. Nor is this the Llama/Qwen3 registration, where q,
/// k and v share one fused weight's pointer and are told apart by port: Gemma 3
/// fuses nothing, so every logical projection has a weight of its own. That has
/// one visible consequence -- `gate_proj` and `up_proj` are registered here
/// rather than refused, because there is a base matrix to add a delta to.
void bind_lora(const detail::RuntimeModelView& runtime, const EngineOptions& options) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    using TextConfig      = detail::TextConfig;
    ops::LoraStore& store = ops::lora_store_for_current_device();
    if (store.empty()) {
        // The widest round an adapter can see. This target refuses speculation,
        // so a decode round is one column per lane.
        const std::uint32_t widest =
            std::max<std::uint32_t>(std::max<std::uint32_t>(options.max_concurrency, 1),
                                    std::max<std::uint32_t>(options.prefill_chunk, 1));
        store.configure(static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_slots, 1)),
                        static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_max_rank, 1)),
                        static_cast<std::int32_t>(widest));
    }

    using Binding = ops::LoraStore::ModuleBinding;
    for (std::size_t layer = 0; layer < TextConfig::layers; ++layer) {
        const auto index      = static_cast<std::int32_t>(layer);
        const auto& attention = runtime.full_layers.at(layer);
        // Ports are the ones `variant.cpp` passes to `apply_lora`; with distinct
        // pointers the port is redundant as an identity, but the pair is the
        // store's key and the two halves have to agree.
        store.register_module(
            index, "q_proj",
            Binding{attention.projection.query.qdata, 0, TextConfig::hidden,
                    TextConfig::query_size});
        store.register_module(
            index, "k_proj",
            Binding{attention.projection.key.qdata, 1, TextConfig::hidden, TextConfig::kv_size});
        store.register_module(
            index, "v_proj",
            Binding{attention.projection.value.qdata, 2, TextConfig::hidden, TextConfig::kv_size});
        store.register_module(
            index, "o_proj",
            Binding{attention.output.qdata, 3, TextConfig::query_size, TextConfig::hidden});
        store.register_module(
            index, "down_proj",
            Binding{attention.post_mixer.down.qdata, 4, TextConfig::intermediate,
                    TextConfig::hidden});
        store.register_module(
            index, "gate_proj",
            Binding{attention.post_mixer.gate.qdata, 5, TextConfig::hidden,
                    TextConfig::intermediate});
        store.register_module(
            index, "up_proj",
            Binding{attention.post_mixer.up.qdata, 6, TextConfig::hidden,
                    TextConfig::intermediate});
    }
    store.ensure_banks();

    for (const auto& payload : options.lora_payloads) {
        store.set_module_slot(payload.layer, payload.module, payload.slot, payload.a, payload.b,
                              payload.rank, payload.in_dim, payload.out_dim, payload.scale);
    }
    ops::lora_set_active(true);
}

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kGemma3Defaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

std::uint32_t Package::maximum_context() noexcept { return detail::Variant::maximum_context; }

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::GroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, family::startup_features(options))));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    bind_lora(model.impl_->data.runtime, options);
    return family::make_frontend(
        model.impl_->data.frontend,
        family::FrontendOptions{
            .vision_enabled = false,
            .max_context    = options.max_context,
            .media_cache_bytes        = options.media_cache_bytes,
            .media_live_bytes         = options.media_live_bytes,
            .media_preprocess_threads = options.media_preprocess_threads,
            .chat_template_override   = options.chat_template_override,
            // Gemma's tokenizer is its own 262,144-id SentencePiece domain with
            // no Vision tokens; the family's registered-checkpoint assertions
            // describe a different tokenizer entirely.
            .registered_tokenizer = false,
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
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::gemma3_270m
