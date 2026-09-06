#include <algorithm>
#include <optional>
#include <api/family/text_geometry.h>

#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/elastic_kv_region.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "ops/linear/w8a8/w8fp8_plane.h"
#include "runtime/engine/kv_capacity.h"
#include "targets/qwen4exp/impl/config.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <exception>
#include <thread>
#include <variant>
#include <utility>
#include <vector>

namespace sinfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".sinfer") {
        throw std::invalid_argument("SInfer accepts only .sinfer artifacts");
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
} // namespace

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

namespace {

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
                                        const family::TextGeometry& geometry,
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
        return Target::make_sequence_planner(device, probe, weights_profile, geometry)
            .capacity_curve();
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
    // The dimensions to plan and bind against: this target's compiled config with whatever
    // the artifact declares laid over it.
    const family::TextGeometry geometry            = Target::declared_geometry(reader);
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
        effective.max_context = resolve_automatic_context<Target>(
            device, options, weights_profile, geometry, preflight_runtime_bytes);
        effective.prefill_chunk = std::min(options.prefill_chunk, effective.max_context);
    }
    if (effective.elastic_kv_overcommit) { effective.elastic_kv = true; }
    auto sequence_planner =
        Target::make_sequence_planner(device, effective, weights_profile, geometry);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    // Overcommit: the physical cap is a guaranteed floor of one full-context request; every
    // page past it is entitled through the device gate at admission. An automatic policy
    // must not size the floor to what is free, or the first engine would guarantee itself
    // everything and the sharing would never start.
    const KvCapacityPolicy kv_policy =
        effective.elastic_kv_overcommit && effective.kv_capacity.mode == KvCapacityMode::Automatic
            ? KvCapacityPolicy::explicit_capacity(effective.max_context)
            : effective.kv_capacity;
    (void)runtime::resolve_kv_capacity(
        kv_policy, curve,
        subtract_saturating(preflight_runtime_bytes, elastic_kv_unmapped_commitment(device.device)));

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();
    // Elastic pools on this device have mapped only what they use so far; what they may still
    // map is not free for this engine's cap, or two engines would fill against each other.
    runtime::KvCapacityResolution capacity_resolution = runtime::resolve_kv_capacity(
        kv_policy, curve,
        subtract_saturating(
            subtract_saturating(current_free_device_bytes(), derived_residency_bytes),
            elastic_kv_unmapped_commitment(device.device)));
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    // The runtime reservation -- KV cache pages, round state, the workspaces -- is one large
    // allocation and the graphs are planned against it, so this is the last quiet stretch of a
    // load. Say what it is before it happens, for the same reason the expert pool does.
    const std::uint64_t reserved = sequence_plan.device_reservation_bytes();
    if (options.load_progress.callback) {
        options.load_progress.callback("runtime reservation", 0, reserved);
    }
    auto loaded   = std::make_unique<Loaded>(std::move(model), effective);
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), device);
    // Closed after the synchronise, not before it: the barrier drains the uploads and the
    // bank's page registration, which is several more seconds that belong to this phase
    // rather than to the silence after it.
    device.synchronize();
    if (options.load_progress.callback) {
        options.load_progress.callback("runtime reservation", reserved, reserved);
    }
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










ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    // Every package answers the same two questions -- do you serve this checkpoint, and what
    // do you call yourself when you do -- so adding a target is one line here.
    const auto dispatch = [&]<class Target, class Loaded, class Instance>(
                              std::optional<ConstructedTarget>& out) {
        if (out.has_value() || !family::package_serves<Target>(identity.model_id)) { return; }
        out = construct_registered<Target, Loaded, Instance>(
            options, device, reader, load_start,
            family::package_target_key_for<Target>(identity.model_id));
    };
    std::optional<ConstructedTarget> constructed;
    dispatch.template operator()<Gemma3, LoadedGemma3, Gemma3Instance>(constructed);
    dispatch.template operator()<Glm5Next, LoadedGlm5Next, Glm5NextInstance>(constructed);
    dispatch.template operator()<Lfm2, LoadedLfm2, Lfm2Instance>(constructed);
    dispatch.template operator()<Llama, LoadedLlama, LlamaInstance>(constructed);
    dispatch.template operator()<Qwen3Dense, LoadedQwen3Dense, Qwen3DenseInstance>(constructed);
    dispatch.template operator()<Qwen3Moe, LoadedQwen3Moe, Qwen3MoeInstance>(constructed);
    dispatch.template operator()<Qwen3_5, LoadedQwen3_5, Qwen3_5Instance>(constructed);
    dispatch.template operator()<Qwen3_5Moe, LoadedQwen3_5Moe, Qwen3_5MoeInstance>(constructed);
    dispatch.template operator()<Qwen38FlashNext, LoadedQwen38FlashNext,
                                 Qwen38FlashNextInstance>(constructed);
    if (constructed.has_value()) { return std::move(*constructed); }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

// A pipeline stage holds only its own layers, so its free memory differs from a whole-model
// run; the ceiling and the pool are shared across the stages, which is what keeps every stage's
// block tables and KV floor identical.
//
// They are resolved against the *tightest* stage, not the first. Every stage holds the same
// number of KV tokens but only its own layers' cache, and a split by layer count does not split
// the bytes evenly: GLM-5.3-Flash's first stage carries the three leading dense layers and the
// embedding where its last carries six mixture layers and the LM head -- 10.6 GiB against
// 26.1 GiB. Sizing the pool to what stage 0 has spare hands the heaviest stage one it cannot
// afford, and the refusal lands after every stage has already uploaded its weights.
//
// The preflight uploads nothing: planning reads the artifact's directory and asks the driver
// for free memory.
// Layers are not the same size. GLM-5.3-Flash's first three are dense (a 12,288-wide FFN) and
// its other forty-two carry a 288-expert mixture, so splitting 45 layers eight ways by *count*
// puts 10.6 GiB on the first card and 26.4 GiB on the last -- and since every stage must hold
// the same number of KV tokens, the pool the whole pipeline gets is what that last card can
// afford. Balancing the bytes instead is worth several GiB of pool on exactly the stage that
// decides it.
//
// What a layer weighs comes from the target itself: planning the range [0, k) for every k and
// differencing gives each layer's marginal bytes, with the per-stage constants (the embedding,
// the LM head) cancelling out. Planning reads the artifact's directory and uploads nothing.
// The boundaries then minimise the heaviest stage, which is the quantity that binds.
template <class Target>
std::vector<int> balanced_stage_bounds(const EngineOptions& options, artifact::Reader& reader,
                                       int layers, int stage_count) {
    const auto weights_profile = Target::resolve_weights(reader.identity());
    const auto planned_through = [&](int last) {
        EngineOptions probe        = options;
        probe.pipeline_stage_first = 0;
        probe.pipeline_stage_last  = last;
        artifact::Binder binder(reader);
        return Target::plan_load(binder, probe, weights_profile)
            .materialization()
            .device_capacity_bytes;
    };

    std::vector<std::uint64_t> marginal(static_cast<std::size_t>(layers), 0);
    std::uint64_t previous = planned_through(1);
    marginal[0]            = previous;
    for (int l = 1; l < layers; ++l) {
        const std::uint64_t through = planned_through(l + 1);
        // A layer cannot weigh less than nothing; if a target's planning is not monotone in the
        // range, fall back to an even share rather than inventing a negative.
        marginal[static_cast<std::size_t>(l)] = through > previous ? through - previous : 0;
        previous                              = through;
    }
    if (std::all_of(marginal.begin(), marginal.end(), [](std::uint64_t b) { return b == 0; })) {
        std::vector<int> even(static_cast<std::size_t>(stage_count) + 1, 0);
        for (int s = 0; s <= stage_count; ++s) {
            even[static_cast<std::size_t>(s)] = layers * s / stage_count;
        }
        return even;
    }

    // Minimise the heaviest stage: the smallest ceiling for which a left-to-right greedy fit
    // uses no more than `stage_count` stages, binary-searched over the byte range. Every stage
    // takes at least one layer, which the feasibility test enforces by construction.
    const std::uint64_t heaviest = *std::max_element(marginal.begin(), marginal.end());
    std::uint64_t total          = 0;
    for (const std::uint64_t bytes : marginal) { total += bytes; }
    const auto stages_needed = [&](std::uint64_t ceiling) {
        int used             = 1;
        std::uint64_t filled = 0;
        for (const std::uint64_t bytes : marginal) {
            if (filled + bytes > ceiling && filled > 0) {
                ++used;
                filled = 0;
            }
            filled += bytes;
        }
        return used;
    };
    std::uint64_t low  = heaviest;
    std::uint64_t high = total;
    while (low < high) {
        const std::uint64_t mid = low + (high - low) / 2;
        if (stages_needed(mid) <= stage_count) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    // Lay the layers out under that ceiling, leaving every remaining stage at least one layer.
    std::vector<int> bounds(static_cast<std::size_t>(stage_count) + 1, 0);
    int stage            = 0;
    std::uint64_t filled = 0;
    for (int l = 0; l < layers; ++l) {
        const int remaining_stages = stage_count - stage - 1;
        const int remaining_layers = layers - l;
        const bool must_close      = remaining_layers <= remaining_stages;
        if (stage + 1 < stage_count && filled > 0 &&
            (must_close || filled + marginal[static_cast<std::size_t>(l)] > low)) {
            ++stage;
            bounds[static_cast<std::size_t>(stage)] = l;
            filled                                  = 0;
        }
        filled += marginal[static_cast<std::size_t>(l)];
    }
    for (int s = stage + 1; s <= stage_count; ++s) {
        bounds[static_cast<std::size_t>(s)] = layers;
    }
    return bounds;
}

template <class Target>
struct StagePreflight {
    std::uint32_t max_context = 0;
    std::uint32_t kv_tokens   = 0;
};

template <class Target>
StagePreflight<Target>
preflight_pipeline(const EngineOptions& options, artifact::Reader& reader,
                   const std::vector<EngineOptions>& stage_options) {
    const auto weights_profile          = Target::resolve_weights(reader.identity());
    const family::TextGeometry geometry = Target::declared_geometry(reader);
    const auto count                    = stage_options.size();

    std::vector<std::size_t> budgets(count, 0);
    StagePreflight<Target> out{};
    for (std::size_t s = 0; s < count; ++s) {
        DeviceContext probe(stage_options[s].device);
        artifact::Binder binder(reader);
        auto plan  = Target::plan_load(binder, stage_options[s], weights_profile);
        budgets[s] = subtract_saturating(
            runtime_bytes_after_planned_weights(plan.materialization().device_capacity_bytes),
            projected_derived_residency_bytes(binder, plan.materialization()));
        if (options.max_context != 0) { continue; }
        const std::uint32_t resolved = resolve_automatic_context<Target>(
            probe, stage_options[s], weights_profile, geometry, budgets[s]);
        out.max_context = out.max_context == 0 ? resolved : std::min(out.max_context, resolved);
    }
    if (options.max_context != 0) { out.max_context = options.max_context; }

    // The ceiling is shared, so every stage plans the same curve shape; what differs is what a
    // token costs on each (its own layers) and what each has spare.
    for (std::size_t s = 0; s < count; ++s) {
        DeviceContext probe(stage_options[s].device);
        EngineOptions stage = stage_options[s];
        stage.max_context   = out.max_context;
        stage.prefill_chunk = std::min(options.prefill_chunk, out.max_context);
        if (stage.elastic_kv_overcommit) { stage.elastic_kv = true; }
        const runtime::SequenceCapacityCurve curve =
            Target::make_sequence_planner(probe, stage, weights_profile, geometry).capacity_curve();
        const KvCapacityPolicy policy =
            stage.elastic_kv_overcommit && stage.kv_capacity.mode == KvCapacityMode::Automatic
                ? KvCapacityPolicy::explicit_capacity(stage.max_context)
                : stage.kv_capacity;
        const std::uint32_t tokens =
            runtime::resolve_kv_capacity(
                policy, curve,
                subtract_saturating(budgets[s], elastic_kv_unmapped_commitment(probe.device)))
                .resolved_tokens;
        out.kv_tokens = out.kv_tokens == 0 ? tokens : std::min(out.kv_tokens, tokens);
    }
    return out;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_pipeline(const EngineOptions& options, artifact::Reader& reader,
                                     Clock::time_point load_start, std::string_view target_key,
                                     int layers) {
    const int stage_count = static_cast<int>(options.devices.size());
    if (stage_count > layers) { throw std::invalid_argument("more pipeline stages than layers"); }
    std::vector<std::unique_ptr<DeviceContext>> devices;
    std::vector<std::unique_ptr<Instance>> stages;
    LoadSummary summary;
    ModelSamplingDefaults sampling_defaults{};
    std::uint32_t resolved_kv      = 0;
    std::uint32_t resolved_context = 0;
    const std::vector<int> bounds =
        balanced_stage_bounds<Target>(options, reader, layers, stage_count);
    const auto stage_options_for = [&](int s) {
        EngineOptions stage_options             = options;
        stage_options.device                    = options.devices[static_cast<std::size_t>(s)];
        stage_options.pipeline_stage_first      = bounds[static_cast<std::size_t>(s)];
        stage_options.pipeline_stage_last       = bounds[static_cast<std::size_t>(s) + 1];
        stage_options.pipeline_import_pinned    = nullptr; // each stage owns its import buffer
        stage_options.cpu_moe_pool_per_socket   = std::getenv("SUROGATE_SERVE_CPU_MOE_POOL_SHARED") == nullptr;
        stage_options.pipeline_boundary_columns = options.prefill_chunk + options.max_concurrency + 128;
        // Zero until the preflight below has run, and then the shared values every stage --
        // stage 0 included -- is built with.
        if (resolved_kv != 0) {
            stage_options.kv_capacity   = KvCapacityPolicy::explicit_capacity(resolved_kv);
            stage_options.max_context   = resolved_context;
            stage_options.prefill_chunk = std::min(options.prefill_chunk, resolved_context);
        }
        return stage_options;
    };

    {
        std::vector<EngineOptions> probe_options;
        probe_options.reserve(static_cast<std::size_t>(stage_count));
        for (int s = 0; s < stage_count; ++s) { probe_options.push_back(stage_options_for(s)); }
        const StagePreflight<Target> shared =
            preflight_pipeline<Target>(options, reader, probe_options);
        resolved_context = shared.max_context;
        resolved_kv      = shared.kv_tokens;
        std::fprintf(stderr,
                     "pipeline: %d stages, max context %u tokens, KV pool %u tokens "
                     "(the tightest stage's)\n",
                     stage_count, resolved_context, resolved_kv);
    }

    // Stage 0 first, alone: the concurrent constructions below inherit whatever globals it
    // configures.
    devices.resize(static_cast<std::size_t>(stage_count));
    stages.resize(static_cast<std::size_t>(stage_count));
    {
        devices[0] = std::make_unique<DeviceContext>(options.devices[0]);
        const EngineOptions stage_options = stage_options_for(0);
        ConstructedTarget stage = construct_registered<Target, Loaded, Instance>(
            stage_options, *devices[0], reader, load_start, target_key);
        stages[0].reset(std::get<std::unique_ptr<Instance>>(stage.active).release());
        sampling_defaults = stage.sampling_defaults;
        summary           = std::move(stage.load);
        std::fprintf(stderr, "pipeline: stage 0 on device %d, layers [%d, %d)\n",
                     stage_options.device, stage_options.pipeline_stage_first,
                     stage_options.pipeline_stage_last);
    }

    // The remaining stages construct concurrently, one thread per card: their uploads, program
    // builds and graph captures touch disjoint devices, the artifact reader is stateless
    // (mmap + offset pread), and every configure_* global the constructors write is
    // mutex-guarded. A worker makes its own DeviceContext, whose constructor binds that
    // thread's current CUDA device. Startup drops from the sum of the stages to roughly
    // stage 0 plus the slowest remaining stage (~2 min -> ~35 s on 8 cards).
    // SUROGATE_SERVE_PIPELINE_SERIAL_CONSTRUCT=1 restores the one-at-a-time order.
    if (stage_count > 1) {
        const bool serial = std::getenv("SUROGATE_SERVE_PIPELINE_SERIAL_CONSTRUCT") != nullptr;
        std::vector<ConstructedTarget> results(static_cast<std::size_t>(stage_count));
        std::vector<std::exception_ptr> failures(static_cast<std::size_t>(stage_count));
        const auto construct_stage = [&](int s) {
            try {
                devices[static_cast<std::size_t>(s)] =
                    std::make_unique<DeviceContext>(options.devices[static_cast<std::size_t>(s)]);
                results[static_cast<std::size_t>(s)] = construct_registered<Target, Loaded, Instance>(
                    stage_options_for(s), *devices[static_cast<std::size_t>(s)], reader, load_start,
                    target_key);
            } catch (...) {
                failures[static_cast<std::size_t>(s)] = std::current_exception();
            }
        };
        if (serial) {
            for (int s = 1; s < stage_count; ++s) { construct_stage(s); }
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(stage_count) - 1);
            for (int s = 1; s < stage_count; ++s) { workers.emplace_back(construct_stage, s); }
            for (std::thread& worker : workers) { worker.join(); }
        }
        for (int s = 1; s < stage_count; ++s) {
            if (failures[static_cast<std::size_t>(s)]) {
                std::rethrow_exception(failures[static_cast<std::size_t>(s)]);
            }
            ConstructedTarget& stage = results[static_cast<std::size_t>(s)];
            stages[static_cast<std::size_t>(s)].reset(
                std::get<std::unique_ptr<Instance>>(stage.active).release());
            summary.artifact_bytes_read += stage.load.artifact_bytes_read;
            summary.host_to_device_bytes += stage.load.host_to_device_bytes;
            std::fprintf(stderr, "pipeline: stage %d on device %d, layers [%d, %d)\n", s,
                         options.devices[static_cast<std::size_t>(s)],
                         bounds[static_cast<std::size_t>(s)],
                         bounds[static_cast<std::size_t>(s) + 1]);
        }
        // The serial loop left the last stage's device current; keep that post-condition.
        CUDA_CHECK(cudaSetDevice(devices.back()->device));
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
    const auto dispatch = [&]<class Target, class Loaded, class Instance>(
                              std::optional<ConstructedTarget>& out) {
        if (out.has_value() || !family::package_serves<Target>(identity.model_id)) { return; }
        out = construct_pipeline<Target, Loaded, Instance>(
            options, reader, load_start,
            family::package_target_key_for<Target>(identity.model_id),
            Target::declared_geometry(reader).layers);
    };
    std::optional<ConstructedTarget> constructed;
    dispatch.template operator()<Glm5Next, LoadedGlm5Next, Glm5NextInstance>(constructed);
    dispatch.template operator()<Qwen38FlashNext, LoadedQwen38FlashNext,
                                 Qwen38FlashNextInstance>(constructed);
    dispatch.template operator()<Qwen3_5, LoadedQwen3_5, Qwen3_5Instance>(constructed);
    dispatch.template operator()<Qwen3_5Moe, LoadedQwen3_5Moe, Qwen3_5MoeInstance>(constructed);
    if (constructed.has_value()) { return std::move(*constructed); }
    throw std::runtime_error("pipeline parallelism is not wired for artifact '" + identity.model_id + "'");
}

} // namespace sinfer::targets
