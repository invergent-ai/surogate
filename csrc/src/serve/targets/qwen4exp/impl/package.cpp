#include "targets/registry.h"
#include <api/targets/qwen4exp/package.h>
#include "family/impl/lora_bind.h"
#include "family/impl/moe/expert_cache.h"
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "artifact/reader.h"
#include "ops/linear/bf16/bf16_cublaslt.h"
#include "targets/qwen4exp/impl/load/bindings.h"
#include "targets/qwen4exp/impl/variant.h"
#include "core/device.h"

#include <stdexcept>
#include <utility>

namespace sinfer::targets::qwen4exp::detail {

SINFER_TARGET_LOAD_PIMPL();

} // namespace sinfer::targets::qwen4exp::detail

namespace sinfer::targets::qwen4exp {
namespace {

// generation_config.json of Qwen3.8-Flash-Next (thinking and non-thinking presets as the
// Qwen3.5/3.6 family publishes them).
constexpr ModelSamplingDefaults kQwen38FlashNextDefaults{
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
    if (model == model_id) { return kQwen38FlashNextDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

std::uint32_t Package::maximum_context() noexcept { return detail::Variant::maximum_context; }

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "w8-hc-v1") {
        return WeightsProfile::W8HyperConnection;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    const family::StartupFeatures features = family::startup_features(options);
    if (features.vision) {
        throw std::runtime_error("qwen3.8-flash-next: vision is not served by this target");
    }
    if (features.dflash()) {
        throw std::runtime_error("qwen3.8-flash-next: DFlash is not served by this target");
    }
    if (features.mtp() && !binder.has("mtp/input_projection")) {
        throw std::runtime_error(
            "qwen3.8-flash-next: --spec mtp needs the NextN draft head, which this artifact does "
            "not carry; put the model's mtp-*.gguf beside the shards (or under MTP/) and convert "
            "again");
    }
    // The Q4 bank is the default. It halves the bytes every miss moves over PCIe and through
    // host DRAM against the W8 planes, and on the board's single-card shape that is 27.3
    // against 20.2 tok/s (2026-09-04, GGUF-native artifact, GPU 0). --host-expert-bank w8
    // keeps the 8-bit planes for a run that would rather spend the host RAM.
    const bool bank_q4 = options.host_expert_bank != EngineOptions::HostExpertBank::W8;
    if (bank_q4) {
        std::fprintf(stderr, "qwen4exp: host expert bank Q4G32AM (59 %% of the W8 bytes; "
                             "requantised while loading; --host-expert-bank w8 restores W8)\n");
    }
    auto plan = detail::bind_artifact(binder, features, options.pipeline_stage_first,
                                      options.pipeline_stage_last, bank_q4, options.load_progress);
    // What the runtime will derive from the resident weights once they are on the device: the
    // registry subtracts it from free memory before it resolves the KV capacity, so a pool that
    // sizes itself before that point has to leave it as well.
    // ...and the weights themselves, which at pool-sizing time are still in the artifact: the
    // pool is created before the engine measures free memory, and that measurement happens
    // before materialisation too.
    family::ExpertCache::configure_derived_reserve(
        targets::projected_derived_residency_bytes(binder, plan.materialization,
                                                       Package::linear_policy) +
        static_cast<std::size_t>(plan.materialization.device_capacity_bytes));
    // ...and what the load holds only while it runs, which is the pool's other neighbour.
    family::ExpertCache::configure_load_staging(
        targets::projected_load_staging_bytes(binder, plan.materialization));
    return LoadPlan(std::make_unique<LoadPlan::Impl>(weights_profile, std::move(plan)));
}

SINFER_TARGET_CONSTRUCT_LOADED_MODEL();

namespace {

void bind_lora(const detail::RuntimeModelView& runtime, const EngineOptions& options) {
    family::bind_lora_moe_hybrid<detail::TextConfig>(runtime, options, [](std::size_t layer) {
        return detail::TextConfig::is_full_attention(static_cast<int>(layer));
    });
}

} // namespace

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    bind_lora(model.impl_->data.runtime, options);
    return family::make_frontend(model.impl_->data.frontend,
                                  family::FrontendOptions{
                                      .vision_enabled = false,
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
                                                        const family::TextGeometry& geometry) {
    // The expert slot pool is device memory the KV planner must not count as free: create it
    // here, before the engine measures free memory for `--kv-capacity auto`. The planner is
    // built first because its capacity curve says the least the runtime must be left --
    // KV floor, round state, workspaces -- and a pool that sizes itself must leave that,
    // plus the automatic headroom, or the engine refuses to start once it asks for it.
    auto planner = family::make_sequence_planner<detail::Variant>(device, options, weights_profile,
                                                                 geometry);
    // The runtime's floor and the load's staging are never resident together, so the
    // pool leaves room for the larger, on top of the weights.
    const std::size_t runtime_floor =
        family::ExpertCache::derived_reserve() +
        std::max(planner.capacity_curve().minimum_device_reservation_bytes +
                     options.kv_capacity.automatic_headroom_bytes,
                 family::ExpertCache::load_staging());
    family::ExpertCache::configure(options, runtime_floor, detail::TextConfig::experts);
    {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(device.device));
        detail::Variant::prewarm_device_scratch();
        CUDA_CHECK(cudaSetDevice(previous));
    }
    return planner;
}

family::TextGeometry Package::declared_geometry(const artifact::Reader& reader) {
    return family::TextGeometry::declared<detail::TextConfig>(reader.geometry());
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    // The program captures its decode graphs at construction; the cuBLASLt route must own its
    // handle and workspace before any capture (plans themselves are host-side).
    ops::detail::bf16_cublaslt_prewarm();
    detail::Variant::prewarm_device_scratch();
    detail::Variant::prepare_expert_split(model.impl_->data.runtime);
    return family::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace sinfer::targets::qwen4exp
