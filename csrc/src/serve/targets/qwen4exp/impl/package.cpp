#include <api/targets/qwen4exp/package.h>
#include <api/targets/qwen3_6/frontend_resources.h>
#include <api/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "ops/linear/bf16/bf16_cublaslt.h"
#include "targets/qwen4exp/impl/load/bindings.h"
#include "targets/qwen4exp/impl/variant.h"
#include "core/device.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen4exp::detail {

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

} // namespace ninfer::targets::qwen4exp::detail

namespace ninfer::targets::qwen4exp {
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

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "w8-hc-v1") {
        return WeightsProfile::W8HyperConnection;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    const qwen3_6::StartupFeatures features = qwen3_6::startup_features(options);
    if (features.vision) {
        throw std::runtime_error("qwen3.8-flash-next: vision is not served by this target");
    }
    if (features.speculative_enabled()) {
        throw std::runtime_error(
            "qwen3.8-flash-next: speculative decoding (MTP/DFlash) is not served by this target");
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(weights_profile,
                                                     detail::bind_artifact(binder, features)));
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
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled = false,
                                      .max_context    = options.max_context,
                                      .media_cache_bytes        = options.media_cache_bytes,
                                      .media_live_bytes         = options.media_live_bytes,
                                      .media_preprocess_threads = options.media_preprocess_threads,
                                  });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    // The expert slot pool is device memory the KV planner must not count as free: create it
    // here, before the engine measures free memory for `--kv-capacity auto`.
    detail::Variant::configure_expert_slots(options.expert_slots);
    {
        // A pipeline stage whose pool holds (nearly) all of its layers' experts gains nothing
        // from the CPU split — every layer would still pay a host round trip — so the split
        // is off above 90 % residency unless the share was given explicitly.
        float share = options.cpu_moe_share;
        const bool staged = options.pipeline_stage_first != 0 || options.pipeline_stage_last != 0;
        if (staged && share < 0.0F && options.expert_slots > 0) {
            const int stage_layers = options.pipeline_stage_last - options.pipeline_stage_first;
            const std::uint64_t stage_experts = static_cast<std::uint64_t>(stage_layers) * detail::TextConfig::experts;
            if (static_cast<std::uint64_t>(options.expert_slots) * 10 >= stage_experts * 9) {
                share = 0.0F;
                std::fprintf(stderr, "qwen4exp: stage holds %u of %llu experts resident; CPU split off\n",
                             options.expert_slots, static_cast<unsigned long long>(stage_experts));
            }
        }
        detail::Variant::configure_cpu_moe_share(share);
    }
    detail::Variant::configure_cpu_moe_min_tokens(options.cpu_moe_min_tokens);
    detail::Variant::configure_cpu_moe_prefill(options.cpu_moe_prefill_share, options.prefill_chunk);
    detail::Variant::configure_cpu_pool_per_socket(options.cpu_moe_pool_per_socket);
    {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(device.device));
        detail::Variant::prewarm_device_scratch();
        CUDA_CHECK(cudaSetDevice(previous));
    }
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    // The program captures its decode graphs at construction; the cuBLASLt route must own its
    // handle and workspace before any capture (plans themselves are host-side).
    ops::detail::bf16_cublaslt_prewarm();
    detail::Variant::prewarm_device_scratch();
    detail::Variant::prepare_expert_split(model.impl_->data.runtime);
    return qwen3_6::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

} // namespace ninfer::targets::qwen4exp
