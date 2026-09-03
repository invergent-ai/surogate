#include <api/targets/llama/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "api/ops/lora.h"
#include "api/ops/lora_store.h"
#include "artifact/reader.h"
#include "targets/llama/impl/load/bindings.h"
#include "targets/llama/impl/variant.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sinfer::targets::llama::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::llama::detail

namespace sinfer::targets::llama {
namespace {

// TinyLlama's published generation presets (model card, chat usage example).
// The model has no thinking mode -- its chat template renders no reasoning turn
// -- so both entries are the same preset rather than two invented ones.
constexpr ModelSamplingDefaults kTinyLlamaDefaults{
    .thinking     = {.temperature       = 0.7F,
                     .top_k             = 50,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 50,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
};

/// The dense flavor of the family's adapter directory.
///
/// `bind_lora_hybrid` cannot serve this target: it reads a fused payload member
/// named `query_key_gate_value`, and reserves a branch for the linear layers
/// this model does not have. The registrations below are the same contract --
/// q/k/v share the fused weight's pointer and are told apart by port, o and down
/// have a weight each -- over a stack where every layer is attention.
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
    const family::TextGeometry& g = runtime.geometry;
    for (std::size_t layer = 0; layer < runtime.full_layers.size(); ++layer) {
        const auto index      = static_cast<std::int32_t>(layer);
        const auto& attention = runtime.full_layers.at(layer);
        const void* qkv       = attention.projection.query_key_value.qdata;
        store.register_module(
            index, "q_proj",
            Binding{qkv, 0, g.hidden, TextConfig::query_heads * TextConfig::head_dim});
        store.register_module(
            index, "k_proj",
            Binding{qkv, 1, g.hidden, TextConfig::kv_heads * TextConfig::head_dim});
        store.register_module(
            index, "v_proj",
            Binding{qkv, 2, g.hidden, TextConfig::kv_heads * TextConfig::head_dim});
        store.register_module(index, "o_proj",
                              Binding{attention.output.qdata, 3,
                                      TextConfig::query_heads * TextConfig::head_dim,
                                      g.hidden});
        store.register_module(
            index, "down_proj",
            Binding{attention.post_mixer.down.qdata, 4, g.intermediate,
                    g.hidden});
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

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kTinyLlamaDefaults; }
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

SINFER_TARGET_CONSTRUCT_LOADED_MODEL();

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
            // Llama's tokenizer is its own 32,000-id SentencePiece domain with no
            // Vision tokens; the family's registered-checkpoint assertions
            // describe a different tokenizer entirely.
            .registered_tokenizer = false,
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

} // namespace sinfer::targets::llama
