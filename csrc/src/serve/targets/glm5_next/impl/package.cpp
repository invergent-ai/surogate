#include <api/targets/glm5_next/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "api/ops/lora.h"
#include "family/impl/lora_bind.h"
#include "api/ops/lora_store.h"
#include "family/impl/mlp_swiglu.h"
#include "artifact/reader.h"
#include "targets/glm5_next/impl/load/bindings.h"
#include "targets/glm5_next/impl/variant.h"
#include "targets/registry.h"
#include "family/impl/moe/expert_cache.h"
#include "api/ops/sparse_moe.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace sinfer::targets::glm5_next::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::glm5_next::detail

namespace sinfer::targets::glm5_next {
namespace {

// GLM-5.3-Flash's published defaults (`generation_config.json` and the GGUF's own
// `general.sampling.*`): temperature 1.0, top_p 0.95. Its chat template chooses a reasoning
// effort rather than switching a thinking mode on and off, so both rows are the same numbers --
// a reader asking for the thinking preset of a model that has no separate one should get the
// model's preset, not a different one invented for the occasion.
constexpr SamplingPreset kGlm5NextPreset{.temperature       = 1.0F,
                                         .top_k             = 0,
                                         .top_p             = 0.95F,
                                         .min_p             = 0.0F,
                                         .presence_penalty  = 0.0F,
                                         .frequency_penalty = 0.0F};
constexpr ModelSamplingDefaults kGlm5NextDefaults{.thinking     = kGlm5NextPreset,
                                                  .non_thinking = kGlm5NextPreset};

/// Bind the attention output, dense MLPs and each routed/shared expert.
/// MLA and KDA projection names outside this directory remain explicit refusals.
void bind_lora(const detail::RuntimeModelView& runtime, const EngineOptions& options) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    using TextConfig      = detail::TextConfig;
    ops::LoraStore& store = ops::lora_store_for_current_device();
    if (store.empty()) {
        // The widest round an adapter can see: a prefill chunk, or every lane verifying its
        // draft window plus the anchor.
        const std::uint32_t columns_per_lane = options.speculative.draft_tokens + 1;
        const std::uint32_t widest           = std::max<std::uint32_t>(
            std::max<std::uint32_t>(options.max_concurrency, 1) * columns_per_lane,
            std::max<std::uint32_t>(options.prefill_chunk, 1));
        store.configure(static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_slots, 1)),
                        static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_max_rank, 1)),
                        static_cast<std::int32_t>(widest));
    }

    using Binding                 = ops::LoraStore::ModuleBinding;
    const family::TextGeometry& g = runtime.geometry;
    // An adapter names its modules by *model* layer, so the walk is over the model's layers and
    // each one reaches into whichever inventory holds it.
    std::size_t full_index = 0;
    std::size_t kda_index  = 0;
    for (std::int32_t index = 0; index < g.layers; ++index) {
        const bool attends = g.layer_attends(index);
        const detail::FeedForwardPayload& mlp =
            attends ? runtime.full_layers.at(full_index).post_mixer
                    : runtime.gdn_layers.at(kda_index).post_mixer;
        if (attends) {
            const auto& attention = runtime.full_layers.at(full_index++);
            // The latent attention has no q, k or v projection to place an adapter on: the
            // query passes through its own low rank and the key and value through a shared
            // one, so an adapter trained on `q_proj` names a matrix this model does not have.
            for (const char* module : {"q_proj", "k_proj", "v_proj"}) {
                store.register_layer_refusal(
                    index, module,
                    "this layer's attention compresses its key and value to a latent and "
                    "expands them per head; it has no q/k/v projection to adapt");
            }
            store.register_module(index, "o_proj",
                                  Binding{attention.output.qdata, family::kOutputPort,
                                          g.query_heads * g.v_head_dim, g.hidden});
        } else {
            ++kda_index;
            for (const char* module : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
                store.register_layer_refusal(
                    index, module,
                    "this layer runs Kimi Delta Attention and has no attention projections");
            }
        }
        if (mlp.sparse) {
            family::bind_lora_moe(store, index, mlp.moe);
            continue;
        }
        store.register_module(
            index, "down_proj",
            Binding{mlp.down.qdata, family::kDownPort, g.dense_intermediate, g.hidden});
        // Gate and up share one fused parent, so they bind the way q/k/v do: one pointer, told
        // apart by port. A format whose halves cannot be projected on their own is refused
        // here, once, rather than throwing on every forward pass.
        const Weight& gate_up = mlp.gate_up;
        if (family::swiglu_halves_addressable(gate_up)) {
            store.register_module(
                index, "gate_proj",
                Binding{gate_up.qdata, family::kGatePort, g.hidden, g.dense_intermediate});
            store.register_module(
                index, "up_proj",
                Binding{gate_up.qdata, family::kUpPort, g.hidden, g.dense_intermediate});
        } else {
            for (const char* module : {"gate_proj", "up_proj"}) {
                store.register_layer_refusal(
                    index, module,
                    "this layer stores gate and up in a format whose halves are not "
                    "independently addressable, so the fused projection cannot be taken apart "
                    "to add their deltas");
            }
        }
    }
    store.validate_payloads(options.lora_payloads);
    store.ensure_banks();

    for (const auto& payload : options.lora_payloads) {
        // A pipeline hands every stage the whole list; each applies the layers it
        // holds and leaves the rest to the stage that does.
        if (!store.covers_layer(payload.layer)) { continue; }
        store.set_module_slot(payload.layer, payload.module, payload.slot, payload.a, payload.b,
                              payload.rank, payload.in_dim, payload.out_dim, payload.scale);
    }
    ops::lora_set_active(true);
}

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kGlm5NextDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

std::uint32_t Package::maximum_context() noexcept { return detail::Variant::maximum_context; }

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.architecture == target_key && identity.weights_id == "w8-mhc-v1") {
        return WeightsProfile::GroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

/// Whether this run keeps any mixture's experts in the host bank -- the only case the expert
/// cache has anything to hold. Sized from free device memory before the KV planner measures
/// it, a pool on a run with every expert resident would only take room from the cache.
bool banks_experts() {
    return family::ExpertCache::pool_floor() != 0;
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    binder.set_offload(options.resident_layer_limit(), options.host_moe_layers,
                       static_cast<std::uint32_t>(options.pipeline_stage_first));
    // The banked experts become planes as the bank fills. By default each object keeps the
    // narrowest planes that lose nothing: Q4G32AM where the file stores it 4-bit affine (this
    // file's gate and up experts, Q4_K), Q5G32AM where it is 5-bit affine (its down experts in
    // 40 layers, Q5_K), W8 where it is wider (the three Q6_K down halves). `--host-expert-bank
    // w8` keeps everything W8; `q4` requantises everything to four bits, the denser, faster,
    // lossy opt-in.
    const family::BankPlanes planes =
        options.host_expert_bank == EngineOptions::HostExpertBank::Q4   ? family::BankPlanes::Q4
        : options.host_expert_bank == EngineOptions::HostExpertBank::W8 ? family::BankPlanes::W8
                                                                         : family::BankPlanes::Auto;
    auto plan = detail::bind_artifact(binder, weights_profile, family::startup_features(options),
                                      options.pipeline_stage_first, options.pipeline_stage_last,
                                      options.host_moe_layers, options.gpu_layers, planes,
                                      options.load_progress);
    // What the pool will insist on, for a stage planner deciding how much to offload; zero
    // where nothing is banked, and set on every plan so a candidate never inherits the last.
    family::ExpertCache::configure_pool_floor(
        std::any_of(plan.bindings.host_bank.objects.begin(), plan.bindings.host_bank.objects.end(),
            [](const auto& object) { return object.name.ends_with("/routed_gate_up"); })
            ? family::ExpertCache::pool_floor_bytes(detail::moe_geometry(plan.bindings.geometry),
                                                    plan.bindings.geometry.layers + plan.bindings.geometry.mtp_layers,
                                                    options.expert_slots)
            : 0);
    if (banks_experts()) {
        std::fprintf(stderr, "glm5_next: host expert bank %s\n",
                     planes == family::BankPlanes::Q4
                         ? "Q4G32AM planes throughout (requantised while loading; this file's "
                           "down experts are Q5_K/Q6_K and lose precision here)"
                     : planes == family::BankPlanes::W8
                         ? "W8 planes throughout (decoded while loading)"
                         : "Q4G32AM planes for the 4-bit halves, Q5G32AM for the 5-bit ones, W8 "
                           "for the wider ones (repacked while loading; --host-expert-bank w8|q4 "
                           "forces one)");
    }
    if (banks_experts()) {
        // What the runtime will derive from the resident weights once they are on the device,
        // and the weights themselves, which at pool-sizing time are still in the artifact: an
        // automatic expert pool sizes itself before either is measured and has to leave both.
        family::ExpertCache::configure_derived_reserve(
            targets::projected_derived_residency_bytes(binder, plan.materialization,
                                                       Package::linear_policy) +
            static_cast<std::size_t>(plan.materialization.device_capacity_bytes));
        // ...and what the load holds only while it runs, which is the pool's other neighbour.
        family::ExpertCache::configure_load_staging(
            targets::projected_load_staging_bytes(binder, plan.materialization));
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(weights_profile, std::move(plan)));
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
            // GLM-5.3-Flash's tokenizer is its own 65,536-id domain with no vision tokens; the
            // family's registered-checkpoint assertions describe a different one entirely.
            .registered_tokenizer = false,
        });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile,
                                                        const family::TextGeometry& geometry,
                                                        const family::VisionGeometry& vision_geometry) {
    auto planner = family::make_sequence_planner<detail::Variant>(device, options,
                                                                  weights_profile, geometry, vision_geometry);
    // The stream mixings travel between a site's collapse and its scatter in a device buffer
    // this target owns, and every device that runs a layer needs its own before the first
    // forward -- a pipeline stage plans on the device it will run on, so this is where it is
    // reachable. The expert cache's pool is the same kind of thing, with one more constraint:
    // it is device memory the KV planner must not count as free, so it is created here, before
    // the engine measures free memory for `--kv-capacity auto`, and sized to leave the least
    // the planner says the runtime needs.
    {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(device.device));
        detail::Variant::prewarm_device_scratch(geometry);
        if (banks_experts() && !options.offload_planning_only) {
            // The runtime's floor and the load's staging are never resident together, so the
            // pool leaves room for the larger, on top of the weights.
            const std::size_t runtime_floor =
                family::ExpertCache::derived_reserve() +
                std::max(planner.capacity_curve().minimum_device_reservation_bytes +
                             options.kv_capacity.automatic_headroom_bytes,
                         family::ExpertCache::load_staging());
            if (std::getenv("SUROGATE_SERVE_PIPELINE_TRACE") != nullptr) {
                std::fprintf(stderr,
                             "pipeline-trace: device %d pool floor %.2f GiB = derived+weights "
                             "%.2f + max(KV minimum %.2f + headroom %.2f, load staging %.2f)\n",
                             device.device,
                             static_cast<double>(runtime_floor) / (1024.0 * 1024.0 * 1024.0),
                             static_cast<double>(family::ExpertCache::derived_reserve()) /
                                 (1024.0 * 1024.0 * 1024.0),
                             static_cast<double>(
                                 planner.capacity_curve().minimum_device_reservation_bytes) /
                                 (1024.0 * 1024.0 * 1024.0),
                             static_cast<double>(options.kv_capacity.automatic_headroom_bytes) /
                                 (1024.0 * 1024.0 * 1024.0),
                             static_cast<double>(family::ExpertCache::load_staging()) /
                                 (1024.0 * 1024.0 * 1024.0));
            }
            family::ExpertCache::configure(options, runtime_floor, geometry.experts);
            (void)family::ExpertCache::for_current_device(
                detail::moe_geometry(geometry), geometry.layers + geometry.mtp_layers);
        }
        CUDA_CHECK(cudaSetDevice(previous));
    }
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
    // The program captures its decode graphs at construction, so everything that would
    // allocate has to exist first. The latent expansion's key half is BF16, which routes
    // through cuBLASLt, and that route creates its handle and workspace per *device* on first
    // use -- so the current device has to be this stage's, not whichever one ran last.
    {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(device.device));
        ops::detail::bf16_cublaslt_prewarm();
        detail::Variant::prewarm_device_scratch(model.impl_->data.runtime.geometry);
        if (model.impl_->data.host_bank != nullptr) {
            // This stage banked experts (the bank exists exactly when it holds something).
            // `--cpu-moe-share auto`: the split's share is measured on the first banked
            // mixture layer, outside any capture; a no-op for a fixed share.
            const detail::RuntimeModelView& runtime = model.impl_->data.runtime;
            const detail::FeedForwardPayload* banked = nullptr;
            for (const auto& layer : runtime.gdn_layers) {
                if (layer.post_mixer.sparse && layer.post_mixer.host_gate_up != nullptr) {
                    banked = &layer.post_mixer;
                    break;
                }
            }
            for (const auto& layer : runtime.full_layers) {
                if (banked != nullptr) { break; }
                if (layer.post_mixer.sparse && layer.post_mixer.host_gate_up != nullptr) {
                    banked = &layer.post_mixer;
                }
            }
            family::ExpertCache& cache = family::ExpertCache::for_current_device(
                detail::moe_geometry(runtime.geometry), runtime.geometry.layers + runtime.geometry.mtp_layers);
            cache.prepare_split(banked == nullptr
                                    ? family::BankedMixture{}
                                    : family::BankedMixture{banked->layer, banked->layers,
                                                            &banked->moe, banked->gate_up_planes,
                                                            banked->down_planes,
                                                            banked->host_gate_up,
                                                            banked->host_down});
        }
        CUDA_CHECK(cudaSetDevice(previous));
    }
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::glm5_next
