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
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    ops::LoraStore& store = ops::lora_store_for_current_device();
    if (store.empty()) {
        // The widest round an adapter can see: a prefill chunk, or every lane
        // times the verify window when a draft is in flight. Sizing this for
        // lanes alone left every prompt without a scratch, so the prompt was
        // built on the base model while the generated tokens carried the delta --
        // an adapter that looked weak rather than unapplied.
        const std::uint32_t window = options.speculative.backend == SpeculativeBackend::None
                                         ? 1U
                                         : options.speculative.draft_tokens + 1U;
        const std::uint32_t decode_columns =
            std::max<std::uint32_t>(options.max_concurrency, 1) * window;
        const std::uint32_t widest =
            std::max<std::uint32_t>(decode_columns, std::max<std::uint32_t>(options.prefill_chunk, 1));
        store.configure(static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_slots, 1)),
                        static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_max_rank, 1)),
                        static_cast<std::int32_t>(widest));
    }

    // The directory: where every adaptable module of every layer lives on this
    // model, registered once so loading is target-agnostic afterwards -- at
    // startup and from the runtime endpoints alike. q, k and v leave one fused
    // projection as separate contiguous tensors, so they share a bank key and are
    // told apart by the port; o and down have a weight each. Attention modules
    // exist only on the full-attention layers (every fourth from 3); the MLP is on
    // every layer.
    using Binding          = ops::LoraStore::ModuleBinding;
    const auto is_full     = [](std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; };
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < detail::TextConfig::layers; ++layer) {
        const auto index = static_cast<std::int32_t>(layer);
        const detail::DensePostMixerPayload* mlp = nullptr;
        if (is_full(layer)) {
            const detail::FullAttentionWeights& full = runtime.full_layers.at(full_index++);
            mlp                                      = &full.post_mixer;
            const auto* fused =
                std::get_if<detail::FusedAttentionProjectionPayload>(&full.projection);
            if (fused != nullptr) {
                const void* qkv = fused->query_key_gate_value.qdata;
                store.register_module(index, "q_proj",
                                      Binding{qkv, 0, detail::TextConfig::hidden,
                                              detail::TextConfig::query_heads *
                                                  detail::TextConfig::head_dim});
                store.register_module(index, "k_proj",
                                      Binding{qkv, 1, detail::TextConfig::hidden,
                                              detail::TextConfig::kv_heads *
                                                  detail::TextConfig::head_dim});
                store.register_module(index, "v_proj",
                                      Binding{qkv, 2, detail::TextConfig::hidden,
                                              detail::TextConfig::kv_heads *
                                                  detail::TextConfig::head_dim});
            } else {
                store.register_layer_refusal(
                    index, "q_proj",
                    "this artifact splits its attention projection, which the attention adapter "
                    "path does not bind");
            }
            store.register_module(index, "o_proj",
                                  Binding{full.output.qdata,
                                          3,
                                          detail::TextConfig::query_heads *
                                              detail::TextConfig::head_dim,
                                          detail::TextConfig::hidden});
        } else {
            mlp = &runtime.gdn_layers.at(gdn_index++).post_mixer;
            for (const char* module : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
                store.register_layer_refusal(
                    index, module,
                    "this layer is linear attention, which has no self_attn projections; an "
                    "adapter naming them here was trained against a different architecture");
            }
        }
        store.register_module(index, "down_proj",
                              Binding{mlp->down.qdata, 4, detail::TextConfig::intermediate,
                                      detail::TextConfig::hidden});
    }
    for (const char* module : {"gate_proj", "up_proj"}) {
        store.register_refusal(module,
                               "gate and up are fused and consumed by SwiGLU inside the "
                               "projection, so there is no intermediate tensor to add a delta to");
    }
    store.ensure_banks();

    for (const auto& payload : options.lora_payloads) {
        store.set_module_slot(payload.layer, payload.module, payload.slot, payload.a, payload.b,
                              payload.rank, payload.in_dim, payload.out_dim, payload.scale);
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
