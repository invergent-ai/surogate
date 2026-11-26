// Inkling-2-Mini package: the registry's questions for the `inkling2` artifact. Bindings are the
// loader's (`impl/load/`); the served geometry, the model view and the visit schedule are this target's.

#include <api/targets/inkling2/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>
#include <api/family/startup_features.h>

#include "api/ops/lora_store.h"
#include "artifact/reader.h"
#include "family/impl/lora_bind.h"
#include "family/impl/lora_bind_inkling.h"
#include "targets/inkling2/impl/lora_attention.h"
#include "targets/inkling2/impl/decision/registry.h"
#include "targets/inkling2/impl/model.h"
#include "targets/inkling2/impl/variant.h"
#include "targets/inkling2/impl/visits.h"
#include "artifact/typed_binding.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <array>
#include <utility>
#include <vector>

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

/// S10, experts first (the kit's recipe trains expert w13 / w2): the MoE binding (family/impl/lora_bind_inkling.h)
/// per PHYSICAL layer, keyed on its router weight. Every pass of a core layer reads the same bank table, so one
/// adapter serves all its visits; adapters name physical layers (the trainer exports them that way).
void bind_lora(const detail::LoadedModelData& data, const EngineOptions& options) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    ops::LoraStore& store = ops::lora_store_for_current_device();
    family::configure_lora_store(store, options);
    // The attention delta kernel (ops::inkling_lora_delta) serves ranks 1..64 and reads the store's max_rank as its rank:
    // refuse a larger --max-lora-rank here, at startup, rather than at the first adapter round.
    if (store.max_rank() > 64) {
        throw std::invalid_argument("inkling2 attention LoRA supports --max-lora-rank up to 64");
    }
    for (std::size_t index = 0; index < data.weights.layers.size(); ++index) {
        const load::InklingLayerWeights& L = data.weights.layers[index];
        if (L.router.qdata == nullptr) { continue; } // not resident on this device
        ops::InklingMoeWeights key;
        key.router = L.router.qdata;
        family::bind_lora_inkling_moe(store, static_cast<int>(index), key,
                                      static_cast<std::int32_t>(options.lora_max_expert_rank));
        // Attention (vLLM parity): the six qkvr slices under their checkpoint names, the fused `attn.qkvr` twin, wo_ud.
        using Binding = ops::LoraStore::ModuleBinding;
        using C       = detail::TextConfig;
        if (L.qkvr.qdata == nullptr || L.output.qdata == nullptr) {
            throw std::logic_error("inkling2: attention LoRA needs the layer's qkvr / wo_ud weights bound");
        }
        std::vector<Binding> slices;
        for (int s = 0; s < 6; ++s) {
            const detail::AttnLoraSlice& sl = detail::kQkvrLoraSlices[s];
            const Binding b{L.qkvr.qdata, detail::kQkvrLoraPort + s, C::hidden, sl.rows};
            store.register_module(static_cast<int>(index), sl.name, b);
            Binding fused    = b;
            fused.source_out = C::qkvr_rows;
            fused.row_offset = sl.begin;
            store.register_module(static_cast<int>(index), detail::kQkvrLoraFusedName, fused);
            slices.push_back(b);
        }
        store.register_bank_table(L.qkvr.qdata, slices);
        const Binding wo{L.output.qdata, detail::kWoLoraPort, C::attn_width, C::hidden};
        store.register_module(static_cast<int>(index), detail::kWoLoraName, wo);
        store.register_bank_table(L.output.qdata, {wo});
    }
    family::finish_lora_bind(store, options);
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

std::size_t Package::extra_derived_residency_bytes(const EngineOptions& options) {
    if (options.speculative.backend != SpeculativeBackend::DFlash) { return 0; }
    // the drafter lives on the stage holding the last visit only (plan_load's draft_placement; M8b review)
    if (options.pipeline_stage_last != 0 && options.pipeline_stage_last < detail::TextConfig::layers) { return 0; }
    return detail::dspark_derived_device_bytes(detail::dspark_fp8_weights_requested(), detail::dspark_fp8_fold_requested());
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    load::BindOptions bind;
    // The 2-card visit split (2-card design): a pipeline stage runs visits [first, last) of the 48 and
    // keeps on its device the physical layers those visits use (a core layer is on every stage that runs one of its
    // passes); the other layers are validated only. The family hands the residual (the whole AttnRes stack, 7 x 2304
    // per column) across the boundary and keeps each stage's KV planes and conv states.
    if (options.pipeline_stage_first != 0 || options.pipeline_stage_last != 0) {
        const int first = options.pipeline_stage_first, last = options.pipeline_stage_last;
        if (first < 0 || last <= first || last > detail::TextConfig::layers) {
            throw std::invalid_argument("inkling2: a pipeline stage covers visits [first, last) of 0..48");
        }
        std::array<bool, detail::TextConfig::physical_layers> resident{};
        for (int v = first; v < last; ++v) { resident[static_cast<std::size_t>(detail::kVisits[v].layer)] = true; }
        const artifact::TensorPlacement in_force = artifact::ScopedPlacement::current();
        bind.layer_placement = [resident, in_force](std::int32_t layer) {
            return resident[static_cast<std::size_t>(layer)] ? in_force : artifact::TensorPlacement::ValidateOnly;
        };
        // DSpark: the drafter, its planes and its proposals live on the stage with the head (the last); the others
        // run the DFlash round's target half only (verify ids from the drafter stage; DESIGN-2card.md 5).
        if (last < detail::TextConfig::layers) { bind.draft_placement = artifact::TensorPlacement::ValidateOnly; }
    }
    bind.attention = attention_choice();
    bind.vision    = options.enable_vision;
    if (bind.attention.fp8 != (weights_profile == WeightsProfile::Fp8Attention)) {
        throw std::logic_error("inkling2: the weights profile and the attention choice disagree");
    }
    // The served drafter (model.h visit_geometry): the release's DSpark drafter, bound under `--speculative dflash` and
    // run through the DFlash runtime; or a DFlash checkpoint the converter added, bound below (DSpark's objects are
    // then validated only).
    const family::TextGeometry geometry = detail::visit_geometry(binder.reader());
    const bool speculative_dflash       = options.speculative.backend == SpeculativeBackend::DFlash;
    const bool checkpoint_drafter       = geometry.dflash.layers > 0 && !geometry.dflash.dspark;
    if (speculative_dflash && checkpoint_drafter &&
        (options.pipeline_stage_first != 0 || options.pipeline_stage_last != 0)) {
        throw std::invalid_argument("inkling2: a DFlash checkpoint drafter is served on one card only");
    }
    bind.draft  = speculative_dflash && !checkpoint_drafter;
    bind.dflash = speculative_dflash && checkpoint_drafter;
    // BF16 drafter weights by default since M9 (h2's A/B: the FP8 route is ~9.5 % slower at 8 streams); with
    // SUROGATE_SERVE_DSPARK_FP8_WEIGHTS=1 the BF16 dense linears sit in the host bank (attached in LoadedModelData,
    // released after derive) and are re-quantised at load. Not on a drafter-less pipeline stage.
    bind.draft_linears_host = bind.draft && detail::dspark_fp8_weights_requested() &&
                              bind.draft_placement != artifact::TensorPlacement::ValidateOnly;
    detail::ArtifactLoadPlan plan;
    plan.bindings.plan     = load::bind_artifact(binder, bind);
    plan.bindings.geometry = geometry;
    plan.bindings.features = family::startup_features(options);
    plan.materialization   = binder.finish();
    plan.bindings.host_bank = family::collect_host_bank(binder, plan.materialization);
    return LoadPlan(std::make_unique<LoadPlan::Impl>(weights_profile, std::move(plan)));
}

SINFER_TARGET_CONSTRUCT_LOADED_MODEL();

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    bind_lora(model.impl_->data, options);
    // Decision adapters (System 1.5): runners over this base, by name (api/targets/inkling2/decision.h). A decision
    // round is one whole prompt; SUROGATE_SERVE_INKLING_DECISION_MAX_TOKENS bounds it (default 16384).
    for (const auto& [name, path] : options.decision_adapters) {
        const char* raw = std::getenv("SUROGATE_SERVE_INKLING_DECISION_MAX_TOKENS");
        const std::int32_t max_tokens = raw != nullptr && *raw != '\0' ? std::atoi(raw) : 16384;
        detail::decision::install(name, model.impl_->data, path, max_tokens);
    }
    // The frontend renders Inkling prompts natively (inkling_render.h) and, with --vision, preprocesses their images
    // as vLLM's Inkling2ImageProcessor does (inkling_vision.h).
    return family::make_frontend(model.impl_->data.frontend,
                                 family::FrontendOptions{
                                     .vision_enabled           = model.impl_->data.runtime.features.vision,
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

std::vector<int> Package::pipeline_stage_bounds(int stage_count) {
    // 2 stages: cut after visit 30 (the end of core pass 1; 2-card 1M design, 3, counting KV, not only weights).
    // Card 0 = prelude + core passes 0-1 (layers 0-17, 5 global visits, embedding); card 1 = core pass 2 + coda + head
    // (+ DSpark's planes / drafter): layers 6-23, 3 global visits. At 1M both cards land near 22 GiB (design estimate).
    // SUROGATE_SERVE_INKLING_STAGE_CUT=<visit> overrides the cut (measurement only).
    if (stage_count != 2) { throw std::invalid_argument("inkling2: the visit split is defined for 2 stages"); }
    int cut = 30;
    if (const char* raw = std::getenv("SUROGATE_SERVE_INKLING_STAGE_CUT"); raw != nullptr && *raw != '\0') {
        cut = std::atoi(raw);
        if (cut <= 0 || cut >= detail::TextConfig::layers) {
            throw std::invalid_argument("SUROGATE_SERVE_INKLING_STAGE_CUT must be a visit in 1..47");
        }
    }
    return {0, cut, detail::TextConfig::layers};
}

family::TextGeometry Package::declared_geometry(const artifact::Reader& reader) {
    return detail::visit_geometry(reader);
}

std::unique_ptr<Package::Program> Package::create_program(const LoadedModel& model, SequencePlan&& plan,
                                                          DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    detail::Variant::prepare_lm_head(16);
    return family::create_program<detail::Variant>(model.impl_->data.runtime, model.impl_->weights_profile,
                                                   std::move(plan), device);
}

} // namespace sinfer::targets::inkling2
