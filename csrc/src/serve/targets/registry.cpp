#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "ops/linear/w8a8/w8fp8_plane.h"
#include "runtime/engine/kv_capacity.h"
#include "targets/qwen4exp/impl/config.h"
#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_35b_a3b/impl/config.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    // max_context == 0 is the automatic request: the largest context the device's free memory
    // allows, resolved per target in construct_registered / construct_pipeline.

    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.kv_cache == KvCacheStorage::Fp8E4M3 &&
        options.speculative.backend == SpeculativeBackend::DFlash) {
        // DFlash commits its draft through kv_cache_append_prefix, which has no
        // e4m3 path. Refuse the pair at startup: the alternative is an
        // exception thrown mid-round once a draft first lands.
        throw std::invalid_argument(
            "--spec dflash needs a bf16 KV cache (pass --kv-cache-dtype bf16); its draft commit "
            "has no fp8 path");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1," +
                                    std::to_string(kMaximumConcurrency) + "]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

std::size_t subtract_saturating(std::size_t value, std::size_t amount) noexcept {
    return value > amount ? value - amount : 0;
}

// The FP8/FP4 and Marlin residencies derive a second, repacked copy of every
// W8 weight they adopt. Those copies are allocated lazily -- during graph
// warmup, or on first use when graphs are off -- which is after the KV cache
// has already been sized and committed. A sizing policy blind to them spends
// the whole card on KV and leaves the derivations to fail: with graphs on that
// surfaced as an out-of-memory during capture, and with --enforce-eager as
// silently corrupted output, because a failed derivation is reported by
// returning a null plane that callers still read through.
//
// Project the footprint from the resident weights rather than a flat constant.
// Only W8-format tensors are ever derived, so an artifact resident in NVFP4 or
// FP8 projects nothing and keeps the capacity it had, and a new architecture
// inherits the reserve from its weight formats without naming itself here.
std::size_t projected_derived_residency_bytes(const artifact::Binder& binder,
                                              const artifact::MaterializationPlan& plan) {
    if (!ops::detail::w8fp8_plane_enabled() && !ops::detail::marlin_plane_enabled()) { return 0; }
    std::uint64_t w8_bytes = 0;
    for (const artifact::DeviceMaterialization& object : plan.device_objects) {
        const auto& descriptor = binder.descriptor(object.object);
        const auto* tensor     = std::get_if<artifact::TensorDescriptor>(&descriptor);
        if (tensor != nullptr && tensor->format == artifact::NumericFormat::W8G32_F16S) {
            w8_bytes += object.bytes;
        }
    }
    // Measured on the 4B (4.80 GiB of W8 weights): 2,774 MiB of FP8/FP4 planes
    // plus 4,086 MiB of Marlin tiles, 1.40x the resident bytes. Adoption is
    // partial and profile-dependent, so round up rather than restate each
    // predicate here -- under-reserving corrupts output, over-reserving only
    // costs cache.
    return static_cast<std::size_t>(w8_bytes + w8_bytes / 2U);
}

// `max_context = 0` asks for the largest context the device can hold. The per-request ceiling
// sizes the KV floor (one sequence at full length must fit) and the block tables; the pool
// itself is then grown by `resolve_kv_capacity`. The byte stride per page does not depend on
// the ceiling, so one probe plan gives the slope, and the candidate is verified by planning at
// it — the ceiling also moves the workspace and graph reservations a little, so a candidate
// that does not fit is halved until it does.
inline constexpr std::uint32_t kAutoContextProbe = 2048;

template <class Target>
std::uint32_t resolve_automatic_context(DeviceContext& device, const EngineOptions& options,
                                        typename Target::WeightsProfile weights_profile,
                                        std::size_t budget_bytes) {
    const std::uint32_t native = Target::maximum_context();
    if (options.kv_capacity.mode == KvCapacityMode::Explicit) {
        // The operator fixed the pool: one sequence may use all of it, nothing more.
        const std::uint32_t fixed =
            std::min(native, std::max<std::uint32_t>(options.kv_capacity.explicit_tokens, 1));
        std::fprintf(stderr, "engine: max context %u tokens (the fixed KV pool, native %u)\n",
                     fixed, native);
        return fixed;
    }
    const std::size_t headroom = options.kv_capacity.automatic_headroom_bytes;
    const auto plan_at         = [&](std::uint32_t context) {
        EngineOptions probe   = options;
        probe.max_context     = context;
        probe.prefill_chunk   = std::min(options.prefill_chunk, context);
        return Target::make_sequence_planner(device, probe, weights_profile).capacity_curve();
    };

    const runtime::SequenceCapacityCurve probe = plan_at(kAutoContextProbe);
    std::uint32_t candidate                    = native;
    if (budget_bytes > headroom + probe.minimum_device_reservation_bytes &&
        probe.bytes_per_additional_main_page_group > 0) {
        const std::size_t spare =
            budget_bytes - headroom - probe.minimum_device_reservation_bytes;
        const std::uint64_t pages =
            static_cast<std::uint64_t>(probe.minimum_main_page_groups) +
            spare / probe.bytes_per_additional_main_page_group;
        candidate = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(native, pages * probe.main_page_tokens));
    }
    candidate = std::max(candidate, kAutoContextProbe);
    for (int attempt = 0; attempt < 8 && candidate > kAutoContextProbe; ++attempt) {
        const runtime::SequenceCapacityCurve curve = plan_at(candidate);
        if (curve.minimum_device_reservation_bytes + headroom <= budget_bytes) { break; }
        candidate = std::max(kAutoContextProbe, candidate / 2);
    }
    std::fprintf(stderr, "engine: max context auto-resolved to %u tokens (native %u)\n", candidate,
                 native);
    return candidate;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);

    artifact::Binder binder(reader);
    auto load_plan = Target::plan_load(binder, options, weights_profile);
    const std::size_t derived_residency_bytes =
        projected_derived_residency_bytes(binder, load_plan.materialization());
    const std::size_t preflight_runtime_bytes = subtract_saturating(
        runtime_bytes_after_planned_weights(load_plan.materialization().device_capacity_bytes),
        derived_residency_bytes);
    EngineOptions effective = options;
    if (effective.max_context == 0) {
        effective.max_context = resolve_automatic_context<Target>(device, options, weights_profile,
                                                                  preflight_runtime_bytes);
        effective.prefill_chunk = std::min(options.prefill_chunk, effective.max_context);
    }
    auto sequence_planner = Target::make_sequence_planner(device, effective, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    (void)runtime::resolve_kv_capacity(effective.kv_capacity, curve, preflight_runtime_bytes);

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();
    runtime::KvCapacityResolution capacity_resolution = runtime::resolve_kv_capacity(
        effective.kv_capacity, curve,
        subtract_saturating(current_free_device_bytes(), derived_residency_bytes));
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    auto loaded   = std::make_unique<Loaded>(std::move(model), effective);
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), device);
    device.synchronize();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    return ConstructedTarget{.active               = ActiveTarget(std::move(instance)),
                             .load                 = std::move(summary),
                             .sampling_defaults    = sampling_defaults,
                             .resolved_max_context = effective.max_context};
}

} // namespace

LoadedQwen3_5_0_8B::LoadedQwen3_5_0_8B(std::unique_ptr<Qwen3_5_0_8B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_5_0_8B::make_frontend(*model, options)) {}

LoadedQwen3_5_0_8B::~LoadedQwen3_5_0_8B() = default;

Qwen3_5_0_8BInstance::Qwen3_5_0_8BInstance(std::unique_ptr<LoadedQwen3_5_0_8B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_5_0_8B::SequencePlan sequence_plan,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_5_0_8B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_5_0_8BInstance::~Qwen3_5_0_8BInstance() = default;

LoadedQwen3_5_2B::LoadedQwen3_5_2B(std::unique_ptr<Qwen3_5_2B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_5_2B::make_frontend(*model, options)) {}

LoadedQwen3_5_2B::~LoadedQwen3_5_2B() = default;

Qwen3_5_2BInstance::Qwen3_5_2BInstance(std::unique_ptr<LoadedQwen3_5_2B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_5_2B::SequencePlan sequence_plan,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_5_2B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_5_2BInstance::~Qwen3_5_2BInstance() = default;

LoadedQwen3_5_4B::LoadedQwen3_5_4B(std::unique_ptr<Qwen3_5_4B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_5_4B::make_frontend(*model, options)) {}

LoadedQwen3_5_4B::~LoadedQwen3_5_4B() = default;

Qwen3_5_4BInstance::Qwen3_5_4BInstance(std::unique_ptr<LoadedQwen3_5_4B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_5_4B::SequencePlan sequence_plan,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_5_4B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_5_4BInstance::~Qwen3_5_4BInstance() = default;

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

LoadedQwen38FlashNext::LoadedQwen38FlashNext(
    std::unique_ptr<Qwen38FlashNext::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen38FlashNext::make_frontend(*model, options)) {}

LoadedQwen38FlashNext::~LoadedQwen38FlashNext() = default;

Qwen38FlashNextInstance::Qwen38FlashNextInstance(
    std::unique_ptr<LoadedQwen38FlashNext> stable_loaded, runtime::KvCapacityResolution resolution,
    Qwen38FlashNext::SequencePlan sequence_plan, DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen38FlashNext::create_program(*loaded->model, std::move(sequence_plan), device)) {}

Qwen38FlashNextInstance::~Qwen38FlashNextInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_5_0_8B::model_id) {
        return construct_registered<Qwen3_5_0_8B, LoadedQwen3_5_0_8B, Qwen3_5_0_8BInstance>(
            options, device, reader, load_start, Qwen3_5_0_8B::target_key);
    }
    if (identity.model_id == Qwen3_5_2B::model_id) {
        return construct_registered<Qwen3_5_2B, LoadedQwen3_5_2B, Qwen3_5_2BInstance>(
            options, device, reader, load_start, Qwen3_5_2B::target_key);
    }
    if (identity.model_id == Qwen3_5_4B::model_id) {
        return construct_registered<Qwen3_5_4B, LoadedQwen3_5_4B, Qwen3_5_4BInstance>(
            options, device, reader, load_start, Qwen3_5_4B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen38FlashNext::model_id) {
        return construct_registered<Qwen38FlashNext, LoadedQwen38FlashNext,
                                    Qwen38FlashNextInstance>(options, device, reader, load_start,
                                                             Qwen38FlashNext::target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

// A pipeline stage holds only its own layers, so its free memory differs from a whole-model
// run; the ceiling is resolved on the first stage and shared, which is what keeps every stage's
// block tables and KV floor identical.
template <class Target>
std::uint32_t resolve_automatic_context_for_pipeline(DeviceContext& device,
                                                     const EngineOptions& stage_options,
                                                     artifact::Reader& reader) {
    const auto weights_profile = Target::resolve_weights(reader.identity());
    artifact::Binder binder(reader);
    auto plan = Target::plan_load(binder, stage_options, weights_profile);
    const std::size_t budget = subtract_saturating(
        runtime_bytes_after_planned_weights(plan.materialization().device_capacity_bytes),
        projected_derived_residency_bytes(binder, plan.materialization()));
    return resolve_automatic_context<Target>(device, stage_options, weights_profile, budget);
}

template <class Target, class Loaded, class Instance, int Layers>
ConstructedTarget construct_pipeline(const EngineOptions& options, artifact::Reader& reader,
                                     Clock::time_point load_start, std::string_view target_key) {
    constexpr int layers  = Layers;
    const int stage_count = static_cast<int>(options.devices.size());
    if (stage_count > layers) { throw std::invalid_argument("more pipeline stages than layers"); }
    std::vector<std::unique_ptr<DeviceContext>> devices;
    std::vector<std::unique_ptr<Instance>> stages;
    LoadSummary summary;
    ModelSamplingDefaults sampling_defaults{};
    std::uint32_t resolved_kv      = 0;
    std::uint32_t resolved_context = 0;
    for (int s = 0; s < stage_count; ++s) {
        devices.push_back(std::make_unique<DeviceContext>(options.devices[static_cast<std::size_t>(s)]));
        EngineOptions stage_options             = options;
        stage_options.device                    = options.devices[static_cast<std::size_t>(s)];
        stage_options.pipeline_stage_first      = layers * s / stage_count;
        stage_options.pipeline_stage_last       = layers * (s + 1) / stage_count;
        stage_options.pipeline_import_pinned    = nullptr; // each stage owns its import buffer
        stage_options.cpu_moe_pool_per_socket   = std::getenv("SUROGATE_SERVE_CPU_MOE_POOL_SHARED") == nullptr;
        stage_options.pipeline_boundary_columns = options.prefill_chunk + options.max_concurrency + 128;
        if (s == 0 && stage_options.max_context == 0) {
            // Every stage must plan the same per-request ceiling: resolve it once, on the first
            // stage's device, and hand it to the rest.
            resolved_context        = resolve_automatic_context_for_pipeline<Target>(
                *devices.back(), stage_options, reader);
            stage_options.max_context = resolved_context;
            stage_options.prefill_chunk = std::min(options.prefill_chunk, resolved_context);
        }
        if (s > 0) {
            stage_options.kv_capacity = KvCapacityPolicy::explicit_capacity(resolved_kv);
            if (resolved_context != 0) {
                stage_options.max_context   = resolved_context; // stage 0 resolved it
                stage_options.prefill_chunk = std::min(options.prefill_chunk, resolved_context);
            }
        }
        ConstructedTarget stage = construct_registered<Target, Loaded, Instance>(
            stage_options, *devices.back(), reader, load_start, target_key);
        auto* instance = std::get<std::unique_ptr<Instance>>(stage.active).release();
        stages.emplace_back(instance);
        if (s == 0) {
            resolved_kv       = instance->kv_capacity_resolution.resolved_tokens;
            sampling_defaults = stage.sampling_defaults;
            summary           = std::move(stage.load);
        } else {
            summary.artifact_bytes_read += stage.load.artifact_bytes_read;
            summary.host_to_device_bytes += stage.load.host_to_device_bytes;
        }
        std::fprintf(stderr, "pipeline: stage %d on device %d, layers [%d, %d)\n", s, stage_options.device,
                     stage_options.pipeline_stage_first, stage_options.pipeline_stage_last);
    }
    summary.load_seconds = std::chrono::duration<double>(Clock::now() - load_start).count();
    auto pipeline = std::make_unique<runtime::PipelineInstance<Instance>>(std::move(devices), std::move(stages));
    return ConstructedTarget{.active               = ActiveTarget(std::move(pipeline)),
                             .load                 = std::move(summary),
                             .sampling_defaults    = sampling_defaults,
                             .resolved_max_context = resolved_context != 0 ? resolved_context
                                                                          : options.max_context};
}

ConstructedTarget construct_pipeline_target(const EngineOptions& options) {
    validate_options(options);
    if (options.devices.size() < 2) {
        throw std::invalid_argument("pipeline parallelism needs at least two devices");
    }
    const auto load_start = Clock::now();
    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen38FlashNext::model_id) {
        return construct_pipeline<Qwen38FlashNext, LoadedQwen38FlashNext, Qwen38FlashNextInstance,
                                  qwen4exp::detail::TextConfig::layers>(options, reader, load_start,
                                                                        Qwen38FlashNext::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::model_id || identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_pipeline<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance,
                                  qwen3_6_27b::detail::TextConfig::layers>(
            options, reader, load_start,
            identity.model_id == Qwen3_6_27B::model_id ? Qwen3_6_27B::target_key : Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_pipeline<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance,
                                  qwen3_6_35b_a3b::detail::TextConfig::layers>(options, reader, load_start,
                                                                               Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("pipeline parallelism is not wired for artifact '" + identity.model_id + "'");
}

} // namespace ninfer::targets
