#include <api/targets/gemma4_moe/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "api/ops/lora.h"
#include "family/impl/lora_bind.h"
#include "api/ops/lora_store.h"
#include "artifact/reader.h"
#include "targets/gemma4_moe/impl/load/bindings.h"
#include "targets/gemma4_moe/impl/variant.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sinfer::targets::gemma4_moe::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::gemma4_moe::detail

namespace sinfer::targets::gemma4_moe {

namespace {
ops::SparseMoeGeometry offload_geometry(const family::TextGeometry& g) {
    ops::SparseMoeGeometry out{g.hidden, g.experts, g.experts_per_token, g.intermediate};
    out.activation = ops::GatedActivation::GeluTanh;
    out.per_expert_scaled = true;
    return out;
}
} // namespace
namespace {

// Gemma 4's published presets, as `generation_config.json` states them. Gemma 4 has no
// thinking mode -- its chat template renders no reasoning turn -- so both entries are the
// same preset rather than two invented ones.
constexpr ModelSamplingDefaults kGemma4MoeDefaults{
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
/// `bind_lora_hybrid` cannot serve this target: it reads a fused payload member named
/// `query_key_gate_value`, and reserves a branch for the linear layers this model does not
/// have. Nor is this the Llama/Qwen3 registration, where q, k and v share one fused weight's
/// pointer and are told apart by port: Gemma 4 fuses nothing, so every logical projection
/// has a weight of its own. That has one visible consequence -- `gate_proj` and `up_proj`
/// are registered here rather than refused, because there is a base matrix to add a delta to.
///
/// Two Gemma 4 details show up here. The projections' shapes are **this layer's**, not the
/// model's, because the windowed and global layers attend at different head widths. And a
/// `k_eq_v` layer has no value matrix at all, so it registers no `v_proj`: there is no base
/// weight for an adapter to sit on, and registering the key's pointer under the value's name
/// would place a v_proj delta on the key projection.
void bind_lora(const detail::RuntimeModelView& runtime, const EngineOptions& options) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    ops::LoraStore& store = ops::lora_store_for_current_device();
    if (store.empty()) {
        // The widest round an adapter can see: a prefill chunk, or a decode round -- which
        // under DFlash verifies the drafts too, so it is a lane's draft window wide.
        const std::uint32_t window = options.speculative.backend == SpeculativeBackend::None
                                         ? 1U
                                         : options.speculative.draft_tokens + 1U;
        const std::uint32_t widest = std::max<std::uint32_t>(
            std::max<std::uint32_t>(
                decode_batch_capacity(options.max_concurrency, options.speculative.backend) *
                    window,
                1),
            std::max<std::uint32_t>(family::lora_prefill_columns(runtime.vision_geometry, options), 1));
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
        // Ports are the ones `variant.cpp` passes to `apply_lora`; with distinct
        // pointers the port is redundant as an identity, but the pair is the
        // store's key and the two halves have to agree.
        const bool windowed              = attention.projection.windowed;
        const std::int32_t query_rows    = g.query_size_for(windowed);
        const std::int32_t kv_rows       = g.kv_size_for(windowed);
        store.register_module(
            index, "q_proj",
            Binding{attention.projection.query.qdata, 0, g.hidden, query_rows});
        store.register_module(
            index, "k_proj",
            Binding{attention.projection.key.qdata, 1, g.hidden, kv_rows});
        if (!attention.projection.value_is_key) {
            store.register_module(
                index, "v_proj",
                Binding{attention.projection.value.qdata, 2, g.hidden, kv_rows});
        }
        store.register_module(
            index, "o_proj",
            Binding{attention.output.qdata, 3, query_rows, g.hidden});
        // The dense branch and routed experts have separate checkpoint modules.
        if (attention.post_mixer.fused_gate_up.qdata != nullptr) {
            // An NVFP4 export's dense gate and up are one fused matrix; an adapter's gate_proj
            // and up_proj deltas would each land on the wrong half of it.
            throw std::invalid_argument(
                "gemma4_moe: LoRA adapters are not served on an artifact whose dense "
                "feed-forward is NVFP4 (fused gate/up); merge the adapter before quantising");
        }
        const std::int32_t dense = detail::dense_intermediate(g);
        family::bind_lora_dense_mlp(store, index, attention.post_mixer, g.hidden, dense);
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
    if (model == model_id) { return kGemma4MoeDefaults; }
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
                                Package::linear_policy);
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
            .gemma_image_tokens = options.gemma_image_tokens,
            .chat_template_override   = options.chat_template_override,
            // Gemma's tokenizer is its own 262,144-id SentencePiece domain; the family's
            // registered-checkpoint assertions describe a different tokenizer entirely.
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
    return detail::resolved_geometry(reader);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    // The routed-NVFP4 experts run on the vendored TensorRT-LLM runner, which picks its grouped
    // GEMMs' tactics per round width by measurement. That launches and synchronises, so it has
    // to happen before the program captures a graph; every layer shares the geometry and so the
    // tactics, and tuning against one layer's weights tunes them all.
    if (model.impl_->weights_profile == WeightsProfile::RoutedNvfp4) {
        for (const auto& layer : model.impl_->data.runtime.full_layers) {
            if (layer.post_mixer.op.routed_gate_up.qtype == QType::NVFP4) {
                ops::sparse_moe_prepare(layer.post_mixer.op, ops::kSparseMoeTrtllmPrepareWidth,
                                        device.stream);
                break;
            }
        }
    }
    family::prepare_banked_experts(model.impl_->data.runtime);
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::gemma4_moe
