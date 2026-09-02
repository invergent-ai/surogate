#include "core/device_footprint.h"
#include "core/sleep.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "ops/linear/w8a8/w4fp4_plane.h"
#include "ops/linear/w8a8/w8fp8_plane.h"
#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/program.h"
#include <cstdio>
#include <cstdlib>

#include "family/impl/runtime/schedule.h"
#include "api/ops/lora_store.h"
#include "api/ops/gdn_replay.h"
#include "api/ops/prepare_ragged_prefix.h"
#include "ops/linear/fp8/fp8_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
#include "api/ops/scatter.h"
#include "api/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {
namespace {

using Clock = std::chrono::steady_clock;

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

schedule::MtpGqaEnvelopes mtp_gqa_envelopes(std::uint32_t max_frontier, std::uint32_t k,
                                            std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    schedule::MtpGqaEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

schedule::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                           std::uint32_t k) {
    (void)min_frontier;
    return schedule::DFlashEnvelopes{
        .local  = {0, max_frontier},
        .full   = {0, max_frontier},
        .append = {0, k + 1},
    };
}

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label, cudaStream_t stream) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        // A profile switch rebuilds the executable from its definition. Patching the shared
        // executable in place (cudaGraphExecUpdate) was the residual corruption of
        // 2026-08-29: under rapid batch recomposition (a saturated 16-lane box with the burst
        // capped at 1) the in-place update misapplied parameters and 23 % of answers came out
        // truncated or with a wrong token; rebuilding on every switch scored 99/100 on the
        // same battery, at no measurable throughput cost (profile switches are rare against
        // the rounds between them). SUROGATE_SERVE_GRAPH_UPDATE_INPLACE=1 restores the old
        // path for bisection.
        static const bool update_in_place =
            std::getenv("SUROGATE_SERVE_GRAPH_UPDATE_INPLACE") != nullptr;
        if (update_in_place) {
            if (topology.executable.update(profile.definition)) {
                topology.executable.upload(stream); // refused update → fresh executable
            }
        } else {
            // Mirror the family's own initialisation exactly: a fresh executable must be
            // uploaded on the launch stream before its first launch (the upload commits its
            // device allocation; launching an un-uploaded chained executable produced garbage
            // egress on the first round after the switch).
            topology.executable.instantiate(profile.definition);
            topology.executable.upload(stream);
        }
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

template <class Prepare>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }

        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        DecodeGraphProfile& profile = family.profiles[profile_index];
        if (topology.installed_profile != profile_index) {
            topology.executable.update(profile.definition);
            topology.installed_profile = profile_index;
        }
        topology.executable.upload(device.stream);
        device.synchronize();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

                    DecodeGraphProfile& profile = family.profiles[i];
                    prepare(profile.min_execution_frontier, profile.batch_size);
                    device.synchronize();
                    topology.executable.launch(device.stream);
                    device.synchronize();
                    continue;
                }
                install_and_upload(topology, i);
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

} // namespace

void ProgramImplCore::configure_stage(const SequencePlanImpl& plan) {
    if (plan.pipeline_stage_first == 0 && plan.pipeline_stage_last == 0) { return; }
    constexpr int layers = static_cast<int>(TextConfig::layers);
    if (plan.pipeline_stage_first < 0 || plan.pipeline_stage_last > layers ||
        plan.pipeline_stage_first >= plan.pipeline_stage_last) {
        throw std::invalid_argument("pipeline stage layer range is invalid");
    }
    if (plan.pipeline_boundary_columns == 0) {
        throw std::invalid_argument("pipeline stage needs a boundary column capacity");
    }
    stage.first   = plan.pipeline_stage_first;
    stage.last    = plan.pipeline_stage_last;
    stage.columns = static_cast<std::int32_t>(plan.pipeline_boundary_columns);
    stage_boundary_bytes_ = static_cast<std::size_t>(residual_width<TextConfig>()) *
                            plan.pipeline_boundary_columns * sizeof(std::uint16_t);
    if (stage.first > 0) {
        // The stage owns its import buffer (the driver copies the previous stage's export
        // into it), so stages can run different micro-batches at the same time.
        if (plan.pipeline_import_pinned != nullptr) {
            stage.import_pinned = plan.pipeline_import_pinned;
        } else {
            CUDA_CHECK(cudaHostAlloc(&stage_import.data, stage_boundary_bytes_, cudaHostAllocPortable));
            stage.import_pinned = stage_import.data;
        }
    }
    if (stage.last < layers) {
        CUDA_CHECK(cudaHostAlloc(&stage_export.data, stage_boundary_bytes_, cudaHostAllocPortable));
        stage.export_pinned = stage_export.data;
    }
    std::fprintf(stderr, "pipeline stage: layers [%d, %d) of %d, boundary %u columns%s%s\n", stage.first,
                 stage.last, layers, plan.pipeline_boundary_columns, stage.first > 0 ? ", imports" : "",
                 stage.last < layers ? ", exports" : "");
}

ProgramImplCore::ProgramImplCore(const LoadedModelData& model_in, const SequencePlanImpl& plan,
                                 DeviceContext& device_in)
    : model(model_in), device(device_in), capacity(plan.capacity), kv_capacity(plan.kv_capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), speculative_backend(plan.speculative_backend),
      kv_dtype(plan.kv_dtype), kv_quant_group(plan.kv_quant_group),
      rewrite_checkpoints(plan.rewrite_checkpoints),
      proposal_head(plan.proposal_head), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}),
      round_host(sizeof(TokenId)),
      ordinary_host(
          plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(family::OrdinaryDecodeIngress) +
                                                     sizeof(family::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(family::MtpDecodeIngress) +
                                                          sizeof(family::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(plan.speculative_backend == SpeculativeBackend::DFlash
                      ? std::make_optional<PinnedHostBuffer>(sizeof(family::DFlashDecodeIngress) +
                                                             sizeof(family::DFlashDecodeEgress))
                      : std::nullopt) {
    if (model.weights_arena == nullptr) {
        throw std::invalid_argument("Qwen3.6 model view has no owning weight arena");
    }
    // Sleep mode: the workspace is per-round scratch, fully rewritten before any
    // read, so its pages can be dropped rather than backed up to host. The
    // persistent arena keeps the Offload default -- it holds KV values and
    // init-once fills (position cells, zeroed states, table rows) that must
    // come back byte-identical, and backing it up also keeps the prefix cache
    // warm across a sleep.
    sleep_tag_region(workspace_storage.base(), SleepTag::Discard);
    if (model.features != plan.features || model.mtp.has_value() != plan.features.mtp() ||
        model.dflash.has_value() != plan.features.dflash() ||
        model.optimized_proposal.has_value() != plan.features.optimized_proposal() ||
        model.vision.has_value() != plan.features.vision) {
        throw std::invalid_argument(
            "Qwen3.6 loaded weights do not match the frozen startup features");
    }
    if (model.mtp.has_value() && model.dflash.has_value()) {
        throw std::invalid_argument("MTP and DFlash model views are mutually exclusive");
    }
    if (model.dflash.has_value() && model.vision.has_value()) {
        throw std::invalid_argument("DFlash and Vision model views are mutually exclusive");
    }
    configure_stage(plan);
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    // SUROGATE_SERVE_ELASTIC_KV_RESERVE=N: granules kept mapped ahead of demand (0 maps on
    // demand only, every emptied granule going straight back) -- a bisection knob.
    const PagedKVElasticOptions elastic_kv{
        .device           = device.device,
        .fence_stream     = device.stream,
        .reserve_granules = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_ELASTIC_KV_RESERVE");
            return raw != nullptr && *raw != '\0' ? static_cast<std::uint32_t>(std::atoi(raw)) : 4U;
        }(),
    };
    decoder = std::make_unique<family::DecoderState>(backing, plan.persistent.decoder, &elastic_kv);
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None)) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) { dflash.emplace(backing, *plan.persistent.dflash); }
    if (dflash.has_value() != plan.features.dflash()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }

    io = family::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() != (speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden                  = plan.persistent.prefill_hidden.bind(backing);
    token_counts                    = plan.persistent.token_counts.bind(backing);
    sampling_config                 = plan.persistent.sampling_config.bind(backing);
    tail_hidden_store               = plan.persistent.tail_hidden.bind(backing);
    rewrite_checkpoint_hidden_store = plan.persistent.rewrite_checkpoint_hidden.bind(backing);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        SequenceState& sequence = sequences[lane];
        sequence.lane           = lane;
        sequence.tail_hidden    = tail_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.rewrite_checkpoint_hidden =
            rewrite_checkpoint_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    }

    set_device_i32(io.text_kv_table_row, 0);
    set_device_i32(io.backend_kv_table_row, 0);

    CUDA_CHECK(cudaMalloc(&chain_one_storage, sizeof(std::int32_t)));
    chain_one = Tensor(chain_one_storage, DType::I32, {1});
    set_device_i32(chain_one, 1);

    host_tokens = static_cast<TokenId*>(round_host.data());
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<family::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<family::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(family::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<family::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<family::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(family::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<family::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<family::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(family::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
    }
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    device.synchronize();
    // Both cuBLASLt routes build their handle and workspace on first use, and capture cannot
    // cudaMalloc. Build them here, while nothing is capturing (#85).
    ops::detail::nvfp4_cublaslt_prewarm();
    ops::detail::fp8_cublaslt_prewarm();
    prepare_graphs();
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

ProgramImplCore::~ProgramImplCore() noexcept {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
    if (chain_one_storage != nullptr) { (void)cudaFree(chain_one_storage); }
}

void ProgramImplCore::burst_egress_copy_host(void* user) noexcept {
    const auto* ctx = static_cast<const BurstEgressCopy*>(user);
    std::memcpy(ctx->destination, ctx->source,
                static_cast<std::size_t>(ctx->count) * sizeof(TokenId));
}

bool ProgramImplCore::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }
    const SequenceState& sequence = sequences[lane];
    const auto can_replace        = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                std::uint32_t new_pages) {
        return old_pages <= pool.entitled_pages() && new_pages <= pool.logical_page_capacity() &&
               new_pages <= pool.capacity_pages() - (pool.entitled_pages() - old_pages);
    };
    const std::uint32_t old_text = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, plan.impl_->text_kv_page_entitlement)) {
        return false;
    }
    const family::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, plan.impl_->backend_kv_page_entitlement);
}

bool ProgramImplCore::can_admit_lane_after_retained_eviction(
    std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }

    std::uint32_t reclaimable_text    = 0;
    std::uint32_t reclaimable_backend = 0;
    for (std::uint32_t other = 0; other < max_concurrency; ++other) {
        if (other == lane || !sequences[other].retained || !sequences[other].kv) { continue; }
        reclaimable_text += sequences[other].kv->text.page_entitlement();
        if (sequences[other].kv->backend) {
            reclaimable_backend += sequences[other].kv->backend->page_entitlement();
        }
    }

    const auto can_replace = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                std::uint32_t reclaimable_pages, std::uint32_t new_pages) {
        if (old_pages > pool.entitled_pages() ||
            reclaimable_pages > pool.entitled_pages() - old_pages ||
            new_pages > pool.logical_page_capacity()) {
            return false;
        }
        const std::uint32_t committed = pool.entitled_pages() - old_pages - reclaimable_pages;
        return new_pages <= pool.capacity_pages() - committed;
    };

    const SequenceState& sequence = sequences[lane];
    const std::uint32_t old_text  = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, reclaimable_text,
                     plan.impl_->text_kv_page_entitlement)) {
        return false;
    }

    const family::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, reclaimable_backend,
                       plan.impl_->backend_kv_page_entitlement);
}

runtime::AdmissionResources ProgramImplCore::admission_capacity() const noexcept {
    const family::PagedKVCache* backend = backend_kv_cache();
    return runtime::AdmissionResources{
        .active_lanes     = max_concurrency,
        .main_kv_pages    = decoder->text_kv.pool().capacity_pages(),
        .backend_kv_pages = backend != nullptr ? backend->pool().capacity_pages() : 0U,
    };
}

runtime::PrefillStepResult ProgramImplCore::start_prefill_lane(std::uint32_t lane,
                                                               PreparedPromptData&& prompt,
                                                               RequestPlan&& plan,
                                                               runtime::TransientRegion transient,
                                                               bool defer_first_chunk) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    RequestPlanImpl& request_plan = *plan.impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }
    // Fixed for the request's lifetime, so it is read once here and staged per
    // round rather than looked up on the hot path.
    request.lora_slot = request_plan.lora_slot;

    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != request_plan.summary.prompt_tokens ||
        (request_plan.vision.has_value() && !prompt.has_media())) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
    }
    if (prompt.identity.rewrite_checkpoint &&
        (prompt.identity.rewrite_checkpoint->frontier == 0 ||
         prompt.identity.rewrite_checkpoint->frontier > prompt_tokens)) {
        throw std::invalid_argument("prepared prompt has an invalid rewrite checkpoint");
    }
    const bool suffix_has_visual = std::any_of(
        prompt.token_types.begin() + static_cast<std::ptrdiff_t>(request_plan.reuse_base),
        prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
    if (suffix_has_visual != request_plan.vision.has_value()) {
        throw std::invalid_argument("request plan does not describe the prompt suffix modality");
    }
    if (request_plan.summary.transient_bytes != 0 &&
        (transient.data == nullptr || transient.size < request_plan.summary.transient_bytes ||
         transient.alignment < request_plan.summary.transient_alignment)) {
        throw std::invalid_argument("request transient region does not satisfy the plan");
    }
    if (request_plan.reuse != ReusePath::FullReset &&
        (!sequence.retained ||
         !family::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          request_plan.reuse_base, request_plan.lora_slot))) {
        throw std::logic_error("planned resident prefix is no longer reusable");
    }
    if (is_rewrite_checkpoint_restore(request_plan.reuse) &&
        (!sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.frontier != request_plan.reuse_base ||
         request_plan.reuse != restore_path(sequence.rewrite_checkpoint.kind))) {
        throw std::logic_error("planned rewrite checkpoint is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::KeepExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind != prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !family::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          sequence.rewrite_checkpoint.frontier,
                                          request_plan.lora_slot))) {
        throw std::logic_error("planned rewrite checkpoint retention is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::ReclassifyExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind == prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !family::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          sequence.rewrite_checkpoint.frontier,
                                          request_plan.lora_slot))) {
        throw std::logic_error("planned rewrite checkpoint reclassification is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::CaptureNew &&
        (!request_plan.rewrite_checkpoint_capture || !prompt.identity.rewrite_checkpoint ||
         request_plan.rewrite_checkpoint_capture->kind !=
             prompt.identity.rewrite_checkpoint->kind ||
         request_plan.rewrite_checkpoint_capture->frontier !=
             prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.rewrite_checkpoint_capture->frontier <= request_plan.reuse_base ||
         request_plan.rewrite_checkpoint_capture->frontier > prompt_tokens)) {
        throw std::logic_error("planned rewrite checkpoint capture is invalid");
    }
    // With rewrite checkpoints disabled the plan drops every checkpoint the
    // prompt describes: the pool has no slot to keep it in, and a later
    // rewrite of that turn simply replays its prefix.
    if (rewrite_checkpoints && request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::Drop &&
        prompt.identity.rewrite_checkpoint) {
        throw std::logic_error("planned rewrite checkpoint drop does not describe the prompt");
    }
    // surogate vendor patch (PATCHES.md #24): with opt-in deferral the
    // frontier legitimately lies AHEAD of the reuse base (the capture is
    // skipped, not already-held); the invariant relaxes to "the prompt
    // still describes a checkpoint".
    static const bool defer_capture_enabled = [] {
        const char* env = std::getenv("SUROGATE_SERVE_DEFER_REWRITE_CHECKPOINT");
        return env != nullptr && env[0] == '1';
    }();
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::DeferCapture &&
        (!prompt.identity.rewrite_checkpoint ||
         (!defer_capture_enabled &&
          (request_plan.reuse == ReusePath::FullReset ||
           prompt.identity.rewrite_checkpoint->frontier > request_plan.reuse_base)))) {
        throw std::logic_error("planned rewrite checkpoint deferral is invalid");
    }

    const auto started       = Clock::now();
    const std::uint32_t base = request_plan.reuse_base;
    const std::uint32_t initial_mtp_extent =
        speculative_backend == SpeculativeBackend::Mtp
            ? std::min({draft_window,
                        request_plan.summary.effective_output_tokens > 1
                            ? request_plan.summary.effective_output_tokens - 2
                            : 0U,
                        capacity - prompt_tokens > 0 ? capacity - prompt_tokens - 1 : 0U})
            : 0U;
    request.lifecycle = Lifecycle::Empty;
    sequence.retained = false;
    try {
        if (request_plan.reuse == ReusePath::FullReset) {
            sequence.kv.reset();
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
            reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                                request_plan.backend_kv_page_entitlement);
        } else if (request_plan.reuse == ReusePath::AppendAtFrontier) {
            static const bool reuse_trace = std::getenv("SUROGATE_SERVE_REUSE_TRACE") != nullptr;
            if (reuse_trace) {
                std::fprintf(stderr,
                             "reuse-trace: begin-append lane %u base %u prompt %u kv_valid %u\n",
                             sequence.lane, base, prompt_tokens, sequence.text_kv_valid);
            }
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
        } else if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash) {
                if (!dflash || !sequence.kv->backend || sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                }
                dflash->restore_rewrite_checkpoint(static_cast<std::int32_t>(sequence.lane),
                                                   device.stream);
                sequence.dflash_context_frontier = base;
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            decoder->copy_state_slot(
                LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency),
                LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                device.stream);
            if (base == prompt_tokens) { copy_tail(sequence, sequence.rewrite_checkpoint_hidden); }
            sequence.ledger.resize(base);
        } else {
            throw std::logic_error("request plan has an invalid prefix reuse path");
        }

        trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : speculative_backend == SpeculativeBackend::DFlash ? prompt_tokens
                                                                : 0U;
        materialize_sequence_kv(sequence, prompt_tokens, backend_materialized);
        install_sampling(sequence, request, request_plan.sampling);
        sequence.rope_delta = prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);

        // Prefill CUDA graphs (PATCHES.md #27): graph prompts compute their GDN/conv state in
        // the shared scratch slot so captured bodies stay lane-independent. The scratch is
        // owned per CHUNK, not per prompt: every graph chunk restores lane→scratch before it
        // runs and saves scratch→lane after (advance_prefill and the mixed round below).
        // Seeding once at begin was the cross-request leak of 2026-08-29: with two staged
        // prompts in flight, the other prompt's chunks ran the scratch between this prompt's
        // begin and its first chunk, so the chunk started from the other prompt's recurrent
        // state — the "capital of France in a primes answer" blends, and (through an eager
        // mixed continuation reading the lane slot instead) the invalid-UTF-8 crashes.
        const bool prefill_uses_graph = prefill_graphs.has_value() && !request_plan.vision &&
                                        !request_plan.prepare_mtp && !prompt.has_media() &&
                                        base < prompt_tokens;

        if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::Drop ||
            request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::CaptureNew) {
            sequence.rewrite_checkpoint = {};
        } else if (request_plan.rewrite_checkpoint_action ==
                   RewriteCheckpointAction::ReclassifyExisting) {
            sequence.rewrite_checkpoint.kind = prompt.identity.rewrite_checkpoint->kind;
        }
        request.timings            = {};
        request.pending            = {};
        sequence.mtp_draft_count   = 0;
        sequence.tail_hidden_valid = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        sequence.prefix_identity.assign(prompt, request_plan.lora_slot);

        if (speculative_backend == SpeculativeBackend::DFlash) {
            if (!dflash || !io.dflash_decode || !sequence.kv->backend) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
            *dflash_host_ingress                         = {};
            dflash_host_ingress->lanes[0]                = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->dflash_kv_table_rows[0] = sequence.kv->backend->bound_row();
            CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                       sizeof(family::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                       device.stream));
        }

        if (request_plan.vision) {
            std::vector<bool> used(prompt.media_payloads.size(), false);
            for (const VisionUseSpan& use : request_plan.vision->uses) {
                if (use.item_index >= used.size()) {
                    throw std::logic_error("Vision plan references a missing media payload");
                }
                used[use.item_index] = true;
            }
            for (std::size_t i = 0; i < used.size(); ++i) {
                if (!used[i]) { prompt.media_payloads[i].reset(); }
            }
        }
        if (prompt.has_media() && !request_plan.vision) { prompt.release_all_media_payloads(); }

        RequestControl::Prefill prefill{
            .prompt                     = std::move(prompt),
            .vision_plan                = std::move(request_plan.vision),
            .vision                     = nullptr,
            .transient                  = transient,
            .rewrite_checkpoint_capture = request_plan.rewrite_checkpoint_capture,
            .base                       = base,
            .cursor                     = base,
            .prompt_tokens              = prompt_tokens,
            .initial_mtp_extent         = initial_mtp_extent,
            .elapsed_seconds            = 0.0,
            .prepare_mtp                = request_plan.prepare_mtp,
            .use_graph                  = prefill_uses_graph,
            .reuse                      = request_plan.reuse,
            .mtp_bridge                 = request_plan.mtp_bridge,
        };
        request.prefill.emplace(std::move(prefill));
        auto& staged = *request.prefill;
        if (staged.vision_plan) {
            staged.vision = std::make_unique<schedule::VisionPrefillSession>(
                device, model, work, staged.prompt, *staged.vision_plan, staged.transient);
        }
        staged.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        // surogate vendor patch (PATCHES.md #23): segment timing probe.
        if (std::getenv("SUROGATE_SERVE_PREFILL_TIMING") != nullptr) {
            std::fprintf(stderr, "prefill-timing: staging %.3f ms\n",
                         staged.elapsed_seconds * 1e3);
        }
        request.lifecycle      = Lifecycle::Prefilling;
        // Deferred first chunk (PATCHES.md #30): leave the staged prefill to
        // the executor loop so it can ride a mixed round with the active
        // decode lanes. Ineligible shapes keep the classic behavior.
        if (defer_first_chunk && speculative_backend == SpeculativeBackend::None &&
            !staged.vision && !staged.prepare_mtp && staged.mtp_bridge == MtpBridgeMode::None &&
            staged.cursor < staged.prompt_tokens) {
            return runtime::PrefillStepResult{
                .summary = runtime::BeginSummary{.prompt_tokens        = staged.prompt_tokens,
                                                 .reused_prompt_tokens = staged.base,
                                                 .prefix_reuse_path    = staged.reuse},
                .processed_prompt_tokens = 0};
        }
        return advance_prefill(sequence, request);
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return advance_prefill(sequences[lane], requests[lane]);
}

void ProgramImplCore::replace_pending_tokens(std::span<const std::uint32_t> lanes,
                                             std::span<const TokenId> tokens) {
    if (lanes.size() != tokens.size()) {
        throw std::invalid_argument("pending token replacement has inconsistent membership");
    }
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        const std::uint32_t lane = lanes[i];
        if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
        RequestControl& request = requests[lane];
        SequenceState& sequence = sequences[lane];
        if ((request.pending.kind != PendingKind::Begin && request.pending.kind != PendingKind::Ordinary) ||
            request.pending.produced != 1 || sequence.ledger.empty()) {
            throw std::logic_error("pending token replacement needs one pending token on the lane");
        }
        sequence.ledger.back() = tokens[i];
    }
}

void ProgramImplCore::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("resolve_prefill_lane requires a pending prefill token");
    }
    resolve_non_speculative_pending(sequences[lane], requests[lane], 1, terminal);
}

void ProgramImplCore::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                            std::span<const std::uint32_t> accepted_tokens,
                                            std::span<const std::uint8_t> terminal,
                                            std::span<const std::uint8_t> cancelled) {
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (cancelled[row]) {
                clear_lane(sequences[lane], requests[lane]);
            } else {
                resolve_non_speculative_pending(sequences[lane], requests[lane],
                                                accepted_tokens[row], terminal[row] != 0);
            }
        }
        return;
    }

    if (!replay_records) {
        throw std::logic_error("speculative pending batch has no ReplaySSM records");
    }

    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative) {
            throw std::logic_error("speculative pending batch no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = sequences[lane];
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (speculative_backend == SpeculativeBackend::DFlash &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw std::logic_error("speculative pending row is not at its recorded base");
        }
        const std::uint32_t committed = cancelled[row] ? 0U : accepted_tokens[row];
        if ((cancelled[row] && accepted_tokens[row] != 0) ||
            (!cancelled[row] && (committed == 0 || committed > pending.produced ||
                                 (!terminal[row] && committed != pending.produced)))) {
            throw std::logic_error("speculative pending row has an invalid committed prefix");
        }
        fold_rows[row] = ops::GdnReplayFoldRow{
            .linear_state_slot = LinearStateSlots::current_state_slot(lane, max_concurrency),
            .commit_columns    = static_cast<std::int32_t>(committed),
        };
        const bool partial_terminal =
            !cancelled[row] && terminal[row] && committed < pending.produced;
        hidden_selectors[row] =
            static_cast<std::int32_t>(partial_terminal ? committed - 1U : pending.produced - 1U);
        needs_hidden_correction = needs_hidden_correction || partial_terminal;
    }

    const auto tail_started = Clock::now();
    try {
        ops::gdn_replay_fold(*replay_records, decoder->linear_attention.all_layers_view(),
                             std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                             device.stream);

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                family::MtpDecodeState& frame = *io.mtp_decode;
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else if (speculative_backend == SpeculativeBackend::DFlash && io.dflash_decode) {
                family::DFlashDecodeState& frame = *io.dflash_decode;
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, tail_hidden_store, device.stream);
        }

        if (speculative_backend == SpeculativeBackend::DFlash) {
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
            }
        }

        device.synchronize();
        work.reset();
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }

    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width = draft_window + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                clear_lane(sequence, request);
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            sequence.prefix_identity.append_generated(committed, sequence.rope_delta);
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row]) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                release_sequence_growth_entitlement(sequence);
                unbind_sequence_kv(sequence);
                sequence.retained = true;
                request.lifecycle = Lifecycle::Complete;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

void ProgramImplCore::abort_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

bool ProgramImplCore::has_retained_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency && sequences[lane].retained;
}

void ProgramImplCore::evict_retained_lane(std::uint32_t lane) noexcept {
    if (!has_retained_lane(lane)) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

GenerationTimings ProgramImplCore::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].timings : GenerationTimings{};
}

SpeculativeStats ProgramImplCore::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].speculative_stats : SpeculativeStats{};
}

void ProgramImplCore::clear_lane(SequenceState& sequence, RequestControl& request) noexcept {
    request.prefill.reset();
    sequence.kv.reset();
    request.lifecycle           = Lifecycle::Empty;
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.retained                = false;
    sequence.rewrite_checkpoint      = {};
    request.pending                  = {};
}

family::PagedKVCache* ProgramImplCore::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) { return &dflash->full; }
    return nullptr;
}

const family::PagedKVCache* ProgramImplCore::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) { return &dflash->full; }
    return nullptr;
}

std::uint32_t ProgramImplCore::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

void ProgramImplCore::reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                                          std::uint32_t backend_pages) {
    if (sequence.kv) { throw std::logic_error("sequence already owns a KV allocation bundle"); }
    if (text_pages == 0 || (backend_kv_cache() == nullptr) != (backend_pages == 0)) {
        throw std::invalid_argument("KV allocation entitlement does not match the active backend");
    }

    std::array<PagedKVReservation, 2> reservations{};
    std::size_t count     = 0;
    reservations[count++] = PagedKVReservation{
        .pool             = &decoder->text_kv.pool(),
        .page_entitlement = text_pages,
    };
    if (family::PagedKVCache* backend = backend_kv_cache(); backend != nullptr) {
        reservations[count++] = PagedKVReservation{
            .pool             = &backend->pool(),
            .page_entitlement = backend_pages,
        };
    }

    std::vector<PagedKVAllocation> allocations =
        reserve_paged_kv_bundle(std::span<const PagedKVReservation>(reservations.data(), count));
    SequenceKVBundle bundle;
    bundle.text = std::move(allocations[0]);
    if (count == 2) { bundle.backend.emplace(std::move(allocations[1])); }
    sequence.kv.emplace(std::move(bundle));
}

void ProgramImplCore::resize_sequence_kv_entitlement(SequenceState& sequence,
                                                     std::uint32_t text_pages,
                                                     std::uint32_t backend_pages) {
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    std::array<PagedKVResize, 2> changes{};
    std::size_t count = 0;
    changes[count++]  = PagedKVResize{
         .allocation       = &sequence.kv->text,
         .mapped_pages     = sequence.kv->text.mapped_page_count(),
         .page_entitlement = text_pages,
    };
    if (sequence.kv->backend) {
        changes[count++] = PagedKVResize{
            .allocation       = &*sequence.kv->backend,
            .mapped_pages     = sequence.kv->backend->mapped_page_count(),
            .page_entitlement = backend_pages,
        };
    }
    resize_paged_kv_bundle(std::span<PagedKVResize>(changes.data(), count));
}

void ProgramImplCore::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv || sequence.kv->text.bound_row() >= 0 ||
        (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0)) {
        throw std::logic_error("KV allocation bundle is unavailable or already bound");
    }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    sequence.kv->text.bind_row(row, device.stream);
    try {
        if (sequence.kv->backend) { sequence.kv->backend->bind_row(row, device.stream); }
        set_device_i32(io.text_kv_table_row, sequence.kv->text.bound_row());
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? sequence.kv->backend->bound_row() : 0);
    } catch (...) {
        if (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0) {
            sequence.kv->backend->unbind_row();
        }
        sequence.kv->text.unbind_row();
        throw;
    }
}

void ProgramImplCore::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    if (sequence.kv->backend) { sequence.kv->backend->unbind_row(); }
    sequence.kv->text.unbind_row();
}

void ProgramImplCore::materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                              std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    if (main_tokens > sequence.kv->text.mapped_token_capacity()) {
        sequence.kv->text.materialize_tokens(main_tokens, device.stream);
    }
    if (backend_tokens != 0 && backend_tokens > sequence.kv->backend->mapped_token_capacity()) {
        sequence.kv->backend->materialize_tokens(backend_tokens, device.stream);
    }
}

void ProgramImplCore::materialize_graph_chunk_window(SequenceState& sequence, std::uint32_t cursor,
                                                     std::uint32_t nominal) {
    // The window is rounded from the chunk's cursor, which prefix reuse and
    // rewrite-checkpoint restores place at arbitrary frontiers; the request's
    // entitlement covers this (PrefillGraphFamily::graph_prefill_reach), so the
    // mapping cannot fall outside it. Past capacity the graph declines to run
    // and the eager body writes only the real tokens, already mapped.
    const std::uint32_t window =
        cursor + static_cast<std::uint32_t>(PrefillGraphFamily::chunk_bucket_for(nominal));
    if (window > capacity) { return; }
    materialize_sequence_kv(sequence, window, 0);
}

void ProgramImplCore::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                       std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    sequence.kv->text.trim_tokens(main_tokens, device.stream);
    if (sequence.kv->backend) { sequence.kv->backend->trim_tokens(backend_tokens, device.stream); }
}

void ProgramImplCore::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    sequence.kv->text.cancel_unmapped_entitlement();
    if (sequence.kv->backend) { sequence.kv->backend->cancel_unmapped_entitlement(); }
}

family::PagedKVCacheView ProgramImplCore::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv) { throw std::logic_error("sequence has no KV allocation bundle"); }
    return decoder->text_kv.execution_view(sequence.kv->text);
}

family::PagedKVCacheView ProgramImplCore::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend) {
        throw std::logic_error("sequence has no MTP KV allocation");
    }
    return decoder->mtp_cache()->execution_view(*sequence.kv->backend);
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::ordered_reset(SequenceState& sequence) {
    decoder->reset_state_slot(
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), device.stream);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    SequenceState& sequence = sequences[0];

    std::vector<PagedKVAllocation> text_capture_allocations;
    std::vector<PagedKVAllocation> mtp_capture_allocations;
    std::vector<PagedKVAllocation> dflash_capture_allocations;
    const auto reserve_capture_rows = [&](family::PagedKVCache& cache,
                                          std::vector<PagedKVAllocation>& allocations,
                                          const char* label) {
        PagedKVPool& pool = cache.pool();
        if (pool.capacity_pages() < max_concurrency) {
            throw std::invalid_argument(std::string(label) +
                                        " cannot provide one Paged KV page per concurrent request");
        }
        allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            allocations.push_back(pool.reserve(1));
            PagedKVAllocation& allocation = allocations.back();
            allocation.bind_row(static_cast<std::int32_t>(row), device.stream);
            allocation.materialize_pages(1, device.stream);

            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            const std::int32_t page = allocation.page_ids().front();
            std::vector<std::int32_t> repeated(pool.logical_page_capacity(), page);
            Tensor table = pool.block_table_row(static_cast<std::int32_t>(row));
            CUDA_CHECK(cudaMemcpyAsync(table.data, repeated.data(), table.bytes(),
                                       cudaMemcpyHostToDevice, device.stream));
        }
    };
    reserve_capture_rows(decoder->text_kv, text_capture_allocations, "target KV cache");
    if (speculative_backend == SpeculativeBackend::Mtp) {
        reserve_capture_rows(*decoder->mtp_cache(), mtp_capture_allocations, "MTP KV cache");
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        reserve_capture_rows(dflash->full, dflash_capture_allocations, "DFlash Full KV cache");
    }
    device.synchronize();

    const DeviceFootprint footprint_before = sample_device_footprint();

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->position);
            controls.push_back(io.mtp->draft_tokens);
            controls.push_back(io.mtp->target_input_ids);
            controls.push_back(io.mtp->target_positions);
        }
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->produced_count); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto zero_capture_pages = [&](family::PagedKVCache& cache,
                                        const std::vector<PagedKVAllocation>& allocations,
                                        std::uint32_t batch_size) {
        std::vector<std::int32_t> pages;
        pages.reserve(batch_size);
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            pages.push_back(allocations[row].page_ids().front());
        }
        cache.pool().zero_pages(pages, device.stream);
    };
    const auto zero_cyclic_lane = [&](CyclicKVCache& cache, std::uint32_t lane) {
        for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = cache.layer_view(layer);
            const Tensor k                    = view.k.slice(3, static_cast<std::int32_t>(lane), 1);
            const Tensor v                    = view.v.slice(3, static_cast<std::int32_t>(lane), 1);
            CUDA_CHECK(cudaMemsetAsync(k.data, 0, k.bytes(), device.stream));
            CUDA_CHECK(cudaMemsetAsync(v.data, 0, v.bytes(), device.stream));
        }
    };

    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv, text_capture_allocations, batch_size);
        if (decoder->mtp_cache() != nullptr) {
            zero_capture_pages(*decoder->mtp_cache(), mtp_capture_allocations, batch_size);
        }
        if (dflash) { zero_capture_pages(dflash->full, dflash_capture_allocations, batch_size); }
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            decoder->reset_state_slot(
                LinearStateSlots::current_state_slot(row, max_concurrency), device.stream);
            if (dflash) {
                zero_cyclic_lane(dflash->local, row);
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
            }
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        if (io.dflash_decode) {
            *dflash_host_ingress       = {};
            *dflash_host_egress        = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                dflash_host_ingress->anchors[row] = 0;
                dflash_host_ingress->execution_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash frontier");
                dflash_host_ingress->context_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash context frontier");
                dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
                dflash_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                dflash_host_ingress->text_kv_table_rows[row]   = static_cast<std::int32_t>(row);
                dflash_host_ingress->dflash_kv_table_rows[row] = static_cast<std::int32_t>(row);
                dflash_host_ingress->lanes[row]                = static_cast<std::int32_t>(row);
                dflash_host_ingress->sampling[row]             = {};
            }
        }
        if (io.mtp_decode) {
            *mtp_host_ingress          = {};
            *mtp_host_egress           = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                mtp_host_ingress->anchors[row] = 0;
                mtp_host_ingress->base_frontiers[row] =
                    checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    checked_i32(capacity, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t step = 0; step < draft_window; ++step) {
                    mtp_host_ingress->current_drafts[row * draft_window + step] = 0;
                }
                for (std::uint32_t column = 0; column < width; ++column) {
                    mtp_host_ingress->target_rope_positions[row * width + column] =
                        checked_i32(frontier + std::min(column, extent),
                                    "graph representative MTP RoPE position");
                }
                mtp_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                mtp_host_ingress->mtp_kv_table_rows[row]  = static_cast<std::int32_t>(row);
                mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                mtp_host_ingress->rope_deltas[row]        = 0;
                mtp_host_ingress->sampling[row]           = {};
            }
        }
        if (io.ordinary) {
            *ordinary_host_ingress = {};
            *ordinary_host_egress  = {};
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                ordinary_host_ingress->tokens[row] = 0;
                ordinary_host_ingress->cache_positions[row] =
                    checked_i32(frontier, "graph representative ordinary position");
                ordinary_host_ingress->rope_positions[row] =
                    checked_i32(frontier, "graph representative ordinary RoPE position");
                ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                ordinary_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                ordinary_host_ingress->sampling[row]           = {};
                ordinary_host_ingress->lora_slots[row]         = -1;
            }
        }
    };
    const auto execution_core = [&] {
        return schedule::ExecutionCore{device,
                                       model,
                                       work,
                                       decoder->linear_attention,
                                       replay_records ? &*replay_records : nullptr,
                                       io,
                                       prefill_hidden,
                                       prefill_chunk,
                                       proposal_head, &decoder->ple, stage};
    };

    if (speculative_backend == SpeculativeBackend::None) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit = max_concurrency;
        schedule::OrdinaryBatchContext ordinary_state{execution_core(),      decoder->text_kv,
                                                      *io.ordinary,          *ordinary_host_ingress,
                                                      *ordinary_host_egress, tail_hidden_store,
                                                      chain_one};
        const GraphExecutionProfile code_warm = ordinary_profiles.front();
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        schedule::ordinary_decode_batch(ordinary_state, 1, {code_warm.min + 1, code_warm.max + 1},
                                        nullptr);
        device.synchronize();
        // Marlin band (PATCHES.md #33): derived planes must exist before the
        // captures, and only a band-sized round derives them (a capturing
        // stream may look one up but never derive). Warm one, then freeze the
        // shared scratch so the captures can bake its addresses.
        if (ordinary_batch_limit >= ops::detail::marlin_min_band_tokens()) {
            const std::uint32_t band =
                std::min<std::uint32_t>(ordinary_batch_limit,
                                        ops::detail::marlin_fixed_m());
            prepare_representative(code_warm.min, band);
            device.synchronize();
            schedule::ordinary_decode_batch(ordinary_state, static_cast<std::int32_t>(band),
                                            {code_warm.min + 1, code_warm.max + 1}, nullptr);
            device.synchronize();
            prepare_representative(code_warm.min, 1);
            device.synchronize();
        }

        // Warmup above has run every op once, so every weight that can adopt
        // Marlin residency already has. Close adoption HERE — before the first
        // capture, not merely before the last family — because a weight that
        // adopts between two captures leaves them disagreeing and the next exec
        // update fails with cudaErrorGraphExecUpdateFailure.
        ops::detail::marlin_fp8_close_adoption();
        ordinary_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit);
        for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
            for (const GraphExecutionProfile planned : ordinary_profiles) {
                ordinary_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * ordinary_batch_limit + (batch_size - 1U);
                const ops::GqaExecutionEnvelope envelope{planned.min + 1, planned.max + 1};
                schedule::capture_ordinary_decode_batch(ordinary_state,
                                                        static_cast<std::int32_t>(batch_size),
                                                        envelope, profile.definition);
            }
        }

        // Round chaining (PATCHES.md #32): the chained flavor of every
        // ordinary profile, captured over the same representative state. A pipeline stage
        // has no sampled tokens to chain (the head lives on the last stage), so it keeps
        // the family empty and always runs single rounds.
        if (!pipeline_stage()) {
        schedule::ordinary_decode_batch_chained(ordinary_state, 1,
                                                {code_warm.min + 1, code_warm.max + 1}, nullptr);
        device.synchronize();
        ordinary_chained_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit);
        for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
            for (const GraphExecutionProfile planned : ordinary_profiles) {
                ordinary_chained_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = ordinary_chained_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * ordinary_batch_limit + (batch_size - 1U);
                const ops::GqaExecutionEnvelope envelope{planned.min + 1, planned.max + 1};
                schedule::capture_ordinary_decode_batch_chained(
                    ordinary_state, static_cast<std::int32_t>(batch_size), envelope,
                    profile.definition);
            }
        }
        } // !pipeline_stage
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const auto planned_profiles = mtp_graph_profiles(capacity, draft_window);
        validate_graph_profiles(planned_profiles, capacity - 1, "MTP");
        schedule::MtpBatchContext mtp_state{
            execution_core(),  decoder->text_kv, *decoder->mtp_cache(), *io.mtp_decode,
            *mtp_host_ingress, *mtp_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = planned_profiles.front();
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        schedule::mtp_decode_batch(mtp_state, 1, draft_window,
                                   mtp_gqa_envelopes(code_warm.max, draft_window, capacity),
                                   nullptr);
        device.synchronize();

        mtp_graphs.profiles.reserve(planned_profiles.size() * max_concurrency);
        for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
            for (const GraphExecutionProfile planned : planned_profiles) {
                mtp_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = mtp_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * max_concurrency + (batch_size - 1U);
                schedule::capture_mtp_decode_batch(
                    mtp_state, static_cast<std::int32_t>(batch_size), draft_window,
                    mtp_gqa_envelopes(planned.max, draft_window, capacity), profile.definition);
            }
        }
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        const auto batch_one_profiles = dflash_graph_profiles(capacity, draft_window, 1);
        validate_graph_profiles(batch_one_profiles, capacity - 1, "DFlash");
        schedule::DFlashBatchContext dflash_state{
            execution_core(),     decoder->text_kv,    *dflash,          *io.dflash_decode,
            *dflash_host_ingress, *dflash_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = batch_one_profiles.front();
        const ops::GqaExecutionEnvelope code_warm_target{
            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   capacity, static_cast<std::uint64_t>(code_warm.max) + draft_window + 1ULL))};
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        schedule::dflash_decode_batch(dflash_state, 1, draft_window,
                                      dflash_envelopes(code_warm.min, code_warm.max, draft_window),
                                      code_warm_target, nullptr);
        device.synchronize();

        dflash_graphs.profiles.reserve(batch_one_profiles.size() * max_concurrency);
        for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
            const auto planned_profiles =
                batch_size == 1 ? batch_one_profiles
                                : dflash_graph_profiles(capacity, draft_window, batch_size);
            validate_graph_profiles(planned_profiles, capacity - 1, "DFlash");
            for (const GraphExecutionProfile planned : planned_profiles) {
                dflash_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = dflash_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * max_concurrency + (batch_size - 1U);
                const ops::GqaExecutionEnvelope target_envelope{
                    1,
                    static_cast<std::uint32_t>(std::min<std::uint64_t>(
                        capacity, static_cast<std::uint64_t>(planned.max) + draft_window + 1ULL))};

                schedule::capture_dflash_decode_batch(
                    dflash_state, static_cast<std::int32_t>(batch_size), draft_window,
                    dflash_envelopes(planned.min, planned.max, draft_window), target_envelope,
                    profile.definition);
            }
        }
    }

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative);
    }
    if (!ordinary_chained_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_chained_graphs, "ordinary chained", device,
                                 prepare_representative);
    }
    ops::detail::marlin_plane_freeze_scratch();
    if (speculative_backend == SpeculativeBackend::Mtp) {
        instantiate_graph_family(mtp_graphs, "MTP", device, prepare_representative);
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        instantiate_graph_family(dflash_graphs, "DFlash", device, prepare_representative);
    }

    ordered_reset(sequence);
    clear_stable_controls();
    for (Tensor& tensor : decoder->linear_attention.conv) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    for (Tensor& tensor : decoder->linear_attention.recurrent) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    if (dflash) {
        const auto zero_cyclic_cache = [&](CyclicKVCache& cache) {
            for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
                const CyclicKVCacheLayerView view = cache.layer_view(layer);
                CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), device.stream));
                CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), device.stream));
            }
        };
        zero_cyclic_cache(dflash->local);
        zero_cyclic_cache(dflash->rewrite_checkpoint_local);
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_features.data, 0,
                                   dflash->prefill_features.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_positions.data, 0,
                                   dflash->prefill_positions.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->pending_features.data, 0,
                                   dflash->pending_features.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();

    const DeviceFootprint footprint_after = sample_device_footprint();
    const DeviceFootprintDelta prepared    = device_footprint_delta(footprint_before,
                                                                    footprint_after);
    // Derived quant planes (fp8/fp4 registries) allocate lazily during the
    // pre-capture warmup decode; they carry their own VRAM guard and are not
    // graph memory, so exclude them. Both sides of this subtraction are the
    // bytes this process allocated -- the registries accumulate the sizes they
    // ask for, and the measurement above is per-process wherever the driver
    // will attribute it -- so a neighbouring process cannot move either one.
    const std::size_t plane_bytes =
        ops::detail::w8_derived_plane_bytes() + ops::detail::marlin_plane_bytes();
    const std::size_t consumed =
        prepared.bytes > plane_bytes ? prepared.bytes - plane_bytes : 0;
    graph_observed_bytes = consumed;
    if (consumed > graph_allowance_bytes) {
        // Refuse only on a figure that is this engine's own. Unattributed, the
        // number counts every process on the card: a second engine loading its
        // weights inside this window once aborted a startup with an allowance
        // error naming bytes this one had never allocated. Say so and continue
        // -- the headroom the pool reserves is what actually protects serving,
        // and it is checked against the device, not against this delta.
        if (!prepared.attributed) {
            std::fprintf(stderr,
                         "CUDA Graph preparation: device free fell by %zu bytes against a planned "
                         "allowance of %zu, but this figure counts every process on the device "
                         "(%s), so it is reported rather than enforced.\n",
                         consumed, graph_allowance_bytes, device_footprint_attribution_note());
            std::fflush(stderr);
        } else {
            throw std::runtime_error("CUDA Graph preparation consumed " + std::to_string(consumed) +
                                     " bytes, exceeding the planned allowance of " +
                                     std::to_string(graph_allowance_bytes) + " bytes");
        }
    }
    // surogate vendor patch (PATCHES.md #27): prefill CUDA graphs.
    static const bool prefill_graph_vetoed = [] {
        const char* env = std::getenv("SUROGATE_SERVE_PREFILL_GRAPH");
        return env != nullptr && env[0] == '0';
    }();
    if (!prefill_graph_vetoed && speculative_backend == SpeculativeBackend::None) {
        // The fp4 cutlass epilogue's device alpha scalar initializes lazily
        // with a synchronous copy; force it now so no captured body ever
        // triggers that init mid-capture (capture would be invalidated).
        (void)ops::detail::w4fp4_alpha_one();
        prefill_graphs.emplace(device, std::min(prefill_chunk, capacity), capacity,
                               LinearStateSlots::prefill_scratch_state_slot(max_concurrency));
        const auto effective_chunk =
            static_cast<std::int32_t>(std::min(prefill_chunk, capacity));

        // One EAGER warmup prefill chunk first: the decode warmup above only
        // derives quant planes for decode-routed weights, but capture must
        // find every plane the prefill body looks up already derived (the
        // registries correctly refuse to derive mid-capture and the dispatch
        // would bake the int8 fallback into the graph). Runs inside the
        // dummy-row window on the scratch state slot; the state/page zeroing
        // below cleans up after it.
        schedule::TextContext capture_card(device, model, work, {},
                                           decoder->linear_attention, io, prefill_hidden,
                                           prefill_chunk, 0, {}, &decoder->text_kv,
                                           decoder->mtp_cache());
        capture_card.set_ple_state(&decoder->ple);
        // A pipeline stage warms up and captures its own layer span only (the other
        // layers are not materialised on this device).
        capture_card.set_stage(stage);
        capture_card.set_linear_state_slots(
            prefill_graphs->scratch_state_slot(),
            rewrite_checkpoints ? LinearStateSlots::rewrite_checkpoint_state_slot(0, max_concurrency)
                                : kNoRewriteCheckpointSlot);
        capture_card.set_gdn_state_action(schedule::GdnStateAction::UpdateInPlace, nullptr);
        prepare_representative(1, 1);
        set_device_i32(io.text_kv_table_row, 0); // the bound dummy row
        // Capture the prefill graphs with the adapter kernels in them. A graph
        // records the launches it sees, so one captured while no round is
        // published carries no delta at all and every request replaying it is
        // served the base model -- while the decode graph, captured through a
        // path that does publish, applies one. That split is what made an
        // adapter's output vary with which graph a request happened to use.
        //
        // The slot the kernels read is a device cell, not a launch argument, so
        // capturing with the cell holding -1 records the work without binding it
        // to any adapter: each replay reads whatever the round wrote.
        struct LoraCaptureScope {
            bool held = false;
            LoraCaptureScope(std::int32_t columns, cudaStream_t stream) {
                if (!ops::lora_active()) { return; }
                ops::LoraRound round;
                round.uniform = true;
                round.scratch = ops::lora_store_for_current_device().scratch(columns);
                if (round.scratch.data == nullptr) { return; }
                ops::lora_store_for_current_device().write_uniform_slot(-1, stream);
                ops::lora_set_round(round);
                held = true;
            }
            ~LoraCaptureScope() {
                if (held) { ops::lora_clear_round(); }
            }
        } lora_capture(static_cast<std::int32_t>(effective_chunk), device.stream);
        {
            const std::vector<int> warm_ids(static_cast<std::size_t>(effective_chunk), 0);
            const schedule::PrefillChunkResult warm = capture_card.prefill_chunk(
                std::span<const int>(warm_ids), 0,
                static_cast<std::uint32_t>(effective_chunk), false);
            if (warm.processed_tokens == 0) {
                throw std::logic_error("prefill graph warmup chunk made no progress");
            }
            device.synchronize();
        }

        // Now precapture every bucket so first requests replay instead of
        // paying capture. The capture card mirrors the decode one — empty KV
        // view, base 0 — the captured body reaches KV only through the pool
        // plus device-side table rows.
        capture_card.set_prefill_graph_family(&*prefill_graphs);
        capture_card.precapture_prefill_graphs(effective_chunk);
        device.synchronize();
        work.reset();
    }

    for (PagedKVAllocation& allocation : dflash_capture_allocations) { allocation.unbind_row(); }
    dflash_capture_allocations.clear();
    for (PagedKVAllocation& allocation : mtp_capture_allocations) { allocation.unbind_row(); }
    mtp_capture_allocations.clear();
    for (PagedKVAllocation& allocation : text_capture_allocations) { allocation.unbind_row(); }
    text_capture_allocations.clear();

}

void ProgramImplCore::install_sampling(SequenceState& sequence, RequestControl& request,
                                       const ops::SamplingConfig& config) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({TextConfig::token_domain});
    CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImplCore::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImplCore::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                    std::span<const std::uint32_t> starts,
                                                    std::span<const std::uint32_t> counts) {
    if (speculative_backend != SpeculativeBackend::DFlash || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = sequences[lane];
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || !sequence.kv->backend || sequence.kv->text.bound_row() < 0 ||
            sequence.kv->backend->bound_row() < 0 || end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] = sequence.kv->backend->bound_row();
        dflash_host_ingress->lanes[row]                = static_cast<std::int32_t>(lane);
        materialize_sequence_kv(sequence, std::max(sequence.text_kv_valid, end), end);
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    family::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(family::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    Tensor lane_tensor   = frame.lanes.slice(0, 0, batch);
    Tensor device_starts = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends   = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows    = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions     = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {DFlashConfig::feature_rows,
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, lane_tensor, device_starts, device_ends,
                               features, positions, device_counts, device.stream);

    schedule::DFlashAppendContext state{{device, model, work, decoder->linear_attention,
                                         replay_records ? &*replay_records : nullptr, io,
                                         prefill_hidden, prefill_chunk, proposal_head, &decoder->ple},
                                        *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    schedule::dflash_append_context(state, features, positions, device_counts, lane_tensor,
                                    table_rows, {minimum_count, maximum_count});
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill(SequenceState& sequence,
                                                            RequestControl& request) {
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse};
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        schedule::PrefillContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, &decoder->ple},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            &sequence.rewrite_checkpoint_hidden,
            // Graph prompts run every chunk (graph or eager fallback alike) on
            // the shared scratch state slot (PATCHES.md #27).
            staged.use_graph
                ? LinearStateSlots::prefill_scratch_state_slot(max_concurrency)
                : LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
            rewrite_checkpoints
                ? LinearStateSlots::rewrite_checkpoint_state_slot(sequence.lane, max_concurrency)
                : kNoRewriteCheckpointSlot,
            staged.initial_mtp_extent,
            dflash_host_ingress,
            staged.use_graph && prefill_graphs.has_value() ? &*prefill_graphs : nullptr,
            requests[sequence.lane].lora_slot};

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden = is_rewrite_checkpoint_restore(staged.reuse)
                                                ? sequence.rewrite_checkpoint_hidden
                                                : sequence.tail_hidden;
            const schedule::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                schedule::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                bridge);
            } else {
                Tensor bridge_token = io.mtp->target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                schedule::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                 bridge.position, bridge.rope_position, false);
            }
            sequence.mtp_kv_valid = staged.base;
            staged.mtp_bridge     = MtpBridgeMode::None;
        }

        if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t nominal =
                std::min(prefill_chunk, staged.prompt_tokens - staged.cursor);
            const bool final_candidate = staged.cursor + nominal == staged.prompt_tokens;
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (speculative_backend == SpeculativeBackend::DFlash) {
                mark_workspace_usage(workspace_plan.dflash_context);
            }
            schedule::PrefillChunkResult result;
            if (staged.use_graph) {
                materialize_graph_chunk_window(sequence, staged.cursor, nominal);
                // Chunk-atomic scratch ownership: restore this prompt's state into the shared
                // scratch slot right before its chunk runs (the lane slot always holds the
                // prompt's current state — zeros after a reset, the resident state for
                // prefix-append, or the previous chunk's save).
                decoder->copy_state_slot(
                    LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                    LinearStateSlots::prefill_scratch_state_slot(max_concurrency), device.stream);
            }
            const std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier =
                staged.rewrite_checkpoint_capture
                    ? std::optional<std::uint32_t>(staged.rewrite_checkpoint_capture->frontier)
                    : std::nullopt;
            if (staged.vision) {
                mark_workspace_usage(workspace_plan.vision_encode);
                result = schedule::prefill_multimodal_chunk(
                    schedule_state, staged.prompt, *staged.vision, nominal,
                    rewrite_checkpoint_capture_frontier, final_candidate);
            } else {
                result = schedule::prefill_text_chunk(
                    schedule_state, std::span<const TokenId>(staged.prompt.token_ids), nominal,
                    rewrite_checkpoint_capture_frontier, final_candidate);
            }
            if (result.processed_tokens == 0 || result.processed_tokens > nominal) {
                throw std::logic_error("ordinary prefill chunk made invalid progress");
            }
            processed_prompt_tokens = result.processed_tokens;
            if (staged.vision) { staged.vision->release_encoded_media_payloads(); }
            if (std::getenv("SUROGATE_SERVE_PREFILL_TIMING") != nullptr) {
                std::fprintf(stderr, "prefill-timing: chunk nominal %u processed %u final %d\n",
                             nominal, result.processed_tokens, int(result.finalized));
            }
            {
                static const bool state_trace =
                    std::getenv("SUROGATE_SERVE_STATE_TRACE") != nullptr;
                if (state_trace) {
                    std::fprintf(stderr,
                                 "state-trace: run lane=%u cursor=%u nominal=%u graph=%d\n",
                                 sequence.lane, staged.cursor, nominal, int(staged.use_graph));
                }
            }
            if (staged.use_graph) {
                // Chunk-atomic scratch ownership: save the chunk's end state back to the lane
                // so the next chunk (this prompt's or another's) can restore its own.
                decoder->copy_state_slot(
                    LinearStateSlots::prefill_scratch_state_slot(max_concurrency),
                    LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                    device.stream);
            }
            staged.cursor += result.processed_tokens;
            sequence.text_kv_valid = staged.cursor;
            if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
            if (speculative_backend == SpeculativeBackend::DFlash) {
                sequence.dflash_context_frontier = staged.cursor;
            }

            if (!result.finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                // surogate vendor patch (PATCHES.md #23): segment timing probe.
                if (std::getenv("SUROGATE_SERVE_PREFILL_TIMING") != nullptr) {
                    std::fprintf(
                        stderr, "prefill-timing: slice %.3f ms (cursor %u/%u)\n",
                        std::chrono::duration<double>(Clock::now() - started).count() * 1e3,
                        staged.cursor, staged.prompt_tokens);
                }
                return runtime::PrefillStepResult{
                    .summary = summary, .processed_prompt_tokens = processed_prompt_tokens};
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            // (The chunk-atomic save above already parked the final state in the lane slot.)
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            schedule::sample_from_hidden(schedule_state, sequence.tail_hidden,
                                         checked_i32(staged.prompt_tokens, "sample position"),
                                         ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                schedule::mtp_bridge_and_propose(
                    schedule_state, io.token, sequence.tail_hidden,
                    checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                    bridge_rope, staged.initial_mtp_extent != 0);
                sequence.mtp_kv_valid = staged.prompt_tokens;
                staged.mtp_bridge     = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        std::array<TokenId, family::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        const auto pre_sync = Clock::now();
        device.synchronize();
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        // surogate vendor patch (PATCHES.md #23): segment timing probe.
        if (std::getenv("SUROGATE_SERVE_PREFILL_TIMING") != nullptr) {
            std::fprintf(stderr, "prefill-timing: compute+sync %.3f ms (sync tail %.3f ms)\n",
                         std::chrono::duration<double>(Clock::now() - started).count() * 1e3,
                         std::chrono::duration<double>(Clock::now() - pre_sync).count() * 1e3);
        }
        const double vision_seconds = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture =
            staged.rewrite_checkpoint_capture;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        sequence.mtp_drafts.begin());
        } else if (speculative_backend == SpeculativeBackend::DFlash &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        if (rewrite_checkpoint_capture) {
            const std::uint32_t frontier = rewrite_checkpoint_capture->frontier;
            if (frontier == 0 || frontier > prompt_tokens || sequence.text_kv_valid < frontier) {
                throw std::logic_error("rewrite checkpoint was not materialized by Text prefill");
            }
            if (speculative_backend == SpeculativeBackend::Mtp &&
                (!staged.prepare_mtp || sequence.mtp_kv_valid < frontier - 1)) {
                throw std::logic_error("rewrite checkpoint has no complete MTP prefix");
            }
            if (speculative_backend == SpeculativeBackend::DFlash &&
                (!dflash || !sequence.kv || !sequence.kv->backend ||
                 sequence.dflash_context_frontier < frontier)) {
                throw std::logic_error("rewrite checkpoint has no complete DFlash prefix");
            }
            sequence.rewrite_checkpoint = RewriteCheckpoint{
                .valid = true, .kind = rewrite_checkpoint_capture->kind, .frontier = frontier};
        }

        staged.prompt.release_all_media_payloads();

        request.prefill.reset();
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .processed_prompt_tokens = processed_prompt_tokens,
            .complete                = true,
        };
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(sequence, request);
        throw;
    }
}

// Launch half of the round lifecycle (runtime/contract/round_lifecycle.h):
// enqueue the round and its egress copies, record what consume will need, and
// return WITHOUT synchronizing.
runtime::RoundHandle
ProgramImplCore::launch_ordinary_round(std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            sequence.kv->text.bound_row() < 0 || sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start = Clock::now();
    try {
        // Round chaining (PATCHES.md #32): launch up to round_burst_limit
        // consecutive rounds; the chained flavor consumes the previous
        // round's sampled tokens device-side, per-round stream host
        // functions copy each egress out in order, and one synchronize
        // covers the whole burst.
        std::uint32_t burst = std::min(round_burst_limit, kChainBurstLimit);
        for (const runtime::RoundBudget& budget : budgets) {
            burst = std::min(burst, budget.generated_tokens_remaining);
        }
        burst = std::min(burst, capacity - maximum_frontier);
        if (burst == 0 || pipeline_stage()) { burst = 1; }

        DecodeGraphExecutable* executable = nullptr;
        DecodeGraphExecutable* chained    = nullptr;
        ops::GqaExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + burst};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch", device.stream);
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
            burst      = std::min(burst,
                                  profile.max_execution_frontier - maximum_frontier + 1);
            if (burst > 1) {
                DecodeGraphProfile& chained_profile = select_graph_profile(
                    ordinary_chained_graphs, static_cast<std::uint32_t>(lanes.size()),
                    maximum_frontier, "ordinary chained batch");
                chained =
                    &install_graph_profile(ordinary_chained_graphs, chained_profile, "ordinary chained batch", device.stream);
            }
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = sequences[lanes[row]];
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            ordinary_host_ingress->lanes[row]    = static_cast<std::int32_t>(sequence.lane);
            ordinary_host_ingress->sampling[row] = request.sampling_host;
            ordinary_host_ingress->lora_slots[row] = request.lora_slot;
            if (std::getenv("SUROGATE_SERVE_LORA_DEBUG") != nullptr) {
                std::fprintf(stderr, "lora-debug: row=%zu lane=%u slot=%d\n", row,
                             static_cast<unsigned>(sequence.lane), request.lora_slot);
            }
            materialize_sequence_kv(sequence, frontier + burst, 0);
        }

        schedule::OrdinaryBatchContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, &decoder->ple},
            decoder->text_kv,
            *io.ordinary,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            tail_hidden_store,
            chain_one};

        mark_workspace_usage(workspace_plan.ordinary_round);
        const std::int32_t rows_i32            = static_cast<std::int32_t>(lanes.size());
        family::OrdinaryDecodeState& ordinary = *io.ordinary;
        for (std::uint32_t round = 0; round < burst; ++round) {
            if (round == 0) {
                schedule::ordinary_decode_batch(schedule_state, rows_i32, envelope, executable);
                if (burst > 1) {
                    // The classic round-0 body carries no chain tail; advance
                    // the frame the same way the chained flavor does so round
                    // 1 consumes round 0's sampled tokens at the next
                    // position instead of replaying round 0.
                    Tensor tokens          = ordinary.tokens.slice(0, 0, rows_i32);
                    Tensor sampled         = ordinary.sampled_tokens.slice(0, 0, rows_i32);
                    Tensor cache_positions = ordinary.cache_positions.slice(0, 0, rows_i32);
                    Tensor rope_positions  = ordinary.rope_positions.slice(0, 0, rows_i32);
                    CUDA_CHECK(cudaMemcpyAsync(tokens.data, sampled.data,
                                               lanes.size() * sizeof(std::int32_t),
                                               cudaMemcpyDeviceToDevice, device.stream));
                    ops::offset_i32_positions(cache_positions, chain_one, cache_positions,
                                              device.stream);
                    ops::offset_i32_positions(rope_positions, chain_one, rope_positions,
                                              device.stream);
                }
            } else {
                schedule::ordinary_decode_batch_chained(schedule_state, rows_i32, envelope,
                                                        chained);
            }
            burst_copy_ctx[round] = BurstEgressCopy{
                .destination = burst_rounds.data() + round * kMaximumConcurrency,
                .source      = ordinary_host_egress->sampled_tokens.data(),
                .count       = static_cast<std::int32_t>(lanes.size())};
            CUDA_CHECK(cudaLaunchHostFunc(device.stream, &ProgramImplCore::burst_egress_copy_host,
                                          &burst_copy_ctx[round]));
        }
        in_flight_ = InFlightRound{.id    = ++in_flight_counter_,
                                   .rows  = static_cast<std::uint32_t>(lanes.size()),
                                   .burst = burst,
                                   .start = start};
        std::copy(lanes.begin(), lanes.end(), in_flight_.lanes.begin());
        return runtime::RoundHandle{.id = in_flight_.id, .rows = in_flight_.rows};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

// Consume half of the round lifecycle (runtime/contract/round_lifecycle.h).
// Everything here needs the device's answers, so this is the half that waits;
// the launch half deliberately does not, which is what lets a caller start the
// next round before this one is read back.
runtime::BatchedGeneratedRound
ProgramImplCore::consume_ordinary_round(runtime::RoundHandle handle) {
    if (!handle.valid() || handle.id != in_flight_.id) {
        throw std::logic_error("consuming a decode round that is not in flight");
    }
    const std::span<const std::uint32_t> lanes(in_flight_.lanes.data(), in_flight_.rows);
    const std::uint32_t burst = in_flight_.burst;
    const auto start          = in_flight_.start;
    try {
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = sequences[lanes[row]];
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            TokenId* row_tokens        = burst_tokens.data() + row * burst;
            for (std::uint32_t round = 0; round < burst; ++round) {
                row_tokens[round] = burst_rounds[round * kMaximumConcurrency + row];
            }
            validate_licensed_tokens(std::span<const TokenId>(row_tokens, burst));
            sequence.text_kv_valid     = base_E + burst;
            sequence.tail_hidden_valid = true;
            for (std::uint32_t round = 0; round < burst; ++round) {
                sequence.ledger.push_back(row_tokens[round]);
            }
            sequence.prefix_identity.append_generated(burst, sequence.rope_delta);
            burst_counts[row] = static_cast<std::int32_t>(burst);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = burst};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(burst_tokens.data(), lanes.size() * burst),
            .row_counts = std::span<const std::int32_t>(burst_counts.data(), lanes.size()),
            .row_stride = burst};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

// The synchronous form every caller used before the split: launch, then
// consume immediately. Kept so targets and tests that do not overlap read the
// same as they always did.
runtime::BatchedGeneratedRound
ProgramImplCore::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets) {
    return consume_ordinary_round(launch_ordinary_round(lanes, budgets));
}

std::string ProgramImplCore::last_mixed_round_description(std::size_t row) const {
    const LastMixedRound& last = last_mixed_round;
    std::string out = "last mixed round: band=" + std::to_string(last.band) +
                      " graph=" + (last.graph_hit ? "1" : "0") +
                      " max_frontier=" + std::to_string(last.maximum_frontier);
    if (row < last.row_frontiers.size()) {
        out += " row_frontier=" + std::to_string(last.row_frontiers[row]);
    }
    return out;
}

bool ProgramImplCore::mixed_round_supported(std::uint32_t prefill_lane) const noexcept {
    if (speculative_backend != SpeculativeBackend::None || prefill_lane >= max_concurrency) {
        return false;
    }
    const RequestControl& request = requests[prefill_lane];
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) { return false; }
    const RequestControl::Prefill& staged = *request.prefill;
    return !staged.vision && !staged.prepare_mtp && staged.mtp_bridge == MtpBridgeMode::None &&
           staged.cursor < staged.prompt_tokens && requests[prefill_lane].prefill->prompt.token_ids.size() != 0;
}

runtime::RoundHandle
ProgramImplCore::launch_mixed_round(std::span<const std::uint32_t> prefill_lanes,
                                       std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets) {
    // No decode lanes is allowed: the round is then a batched prefill step (pipeline stages
    // use it so a prompt's chunk is an asynchronous round like any other).
    if (speculative_backend != SpeculativeBackend::None || budgets.size() != lanes.size() ||
        prefill_lanes.empty() || prefill_lanes.size() > runtime::kMaximumMixedPrefills) {
        throw std::invalid_argument("mixed round requires plain decode lanes and prefill lanes");
    }
    for (std::size_t i = 0; i < prefill_lanes.size(); ++i) {
        const std::uint32_t lane = prefill_lanes[i];
        if (lane >= max_concurrency ||
            std::find(prefill_lanes.begin(), prefill_lanes.begin() + static_cast<std::ptrdiff_t>(i),
                      lane) != prefill_lanes.begin() + static_cast<std::ptrdiff_t>(i)) {
            throw std::invalid_argument("mixed round has an invalid or duplicate prefill lane");
        }
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
            throw std::logic_error("mixed round requires active prefill lanes");
        }
        const RequestControl::Prefill& entry = *request.prefill;
        if (entry.vision || entry.prepare_mtp || entry.cursor >= entry.prompt_tokens ||
            entry.mtp_bridge != MtpBridgeMode::None) {
            throw std::logic_error("mixed round does not support this staged prefill");
        }
    }
    // The first staged prompt owns the card (its KV view and cursor); every prompt's KV is
    // addressed per segment through the batch view and its own table row.
    const std::uint32_t prefill_lane = prefill_lanes.front();
    SequenceState& prefill_sequence  = sequences[prefill_lane];
    RequestControl& prefill_request  = requests[prefill_lane];
    RequestControl::Prefill& staged  = *prefill_request.prefill;

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || lane == prefill_lane ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("mixed round contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            sequence.kv->text.bound_row() < 0 || sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1) {
            throw std::logic_error("mixed round row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start        = Clock::now();
    const std::int32_t rows = static_cast<std::int32_t>(lanes.size());
    last_mixed_round.maximum_frontier = maximum_frontier;
    last_mixed_round.band             = -1;
    last_mixed_round.graph_hit        = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        last_mixed_round.row_frontiers[row] = sequences[lanes[row]].execution_frontier;
    }
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = sequences[lanes[row]];
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "mixed round position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "mixed round RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            ordinary_host_ingress->lanes[row]    = static_cast<std::int32_t>(sequence.lane);
            ordinary_host_ingress->sampling[row] = request.sampling_host;
            ordinary_host_ingress->lora_slots[row] = request.lora_slot;
            if (std::getenv("SUROGATE_SERVE_LORA_DEBUG") != nullptr) {
                std::fprintf(stderr, "lora-debug: row=%zu lane=%u slot=%d\n", row,
                             static_cast<unsigned>(sequence.lane), request.lora_slot);
            }
            materialize_sequence_kv(sequence, frontier + 1, 0);
        }
        // Mixed-round graphs (PATCHES.md #30) pad the decode batch to its
        // bucket by duplicating row 0: a pad column recomputes that lane's
        // own update, so every state/KV write lands as identical bytes.
        const std::int32_t batch_bucket = PrefillGraphFamily::batch_bucket_for(rows);
        if (batch_bucket < rows) {
            throw std::logic_error("mixed round bucket is smaller than the decode row count");
        }
        for (std::int32_t row = rows; row < batch_bucket; ++row) {
            const std::size_t pad                          = static_cast<std::size_t>(row);
            ordinary_host_ingress->tokens[pad]             = ordinary_host_ingress->tokens[0];
            ordinary_host_ingress->cache_positions[pad]    = ordinary_host_ingress->cache_positions[0];
            ordinary_host_ingress->rope_positions[pad]     = ordinary_host_ingress->rope_positions[0];
            ordinary_host_ingress->text_kv_table_rows[pad] = ordinary_host_ingress->text_kv_table_rows[0];
            ordinary_host_ingress->lanes[pad]              = ordinary_host_ingress->lanes[0];
            ordinary_host_ingress->sampling[pad]           = ordinary_host_ingress->sampling[0];
            ordinary_host_ingress->lora_slots[pad]         = ordinary_host_ingress->lora_slots[0];
        }
        family::OrdinaryDecodeState& ordinary = *io.ordinary;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, ordinary_host_ingress,
                                   sizeof(family::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   device.stream));

        // The workspace plan covers prefill_chunk columns; the decode batch
        // rides in the same window, so the chunk shrinks by the batch size.
        const std::uint32_t mixed_chunk_cap =
            prefill_chunk > static_cast<std::uint32_t>(rows)
                ? prefill_chunk - static_cast<std::uint32_t>(rows)
                : 1U;
        // Every staged prompt draws from the same window: give each one its remaining
        // tokens in turn until the window is spent, so a round packs as many short prompts
        // as fit and still chunks a long one exactly as before (#80).
        std::array<std::uint32_t, runtime::kMaximumMixedPrefills> nominals{};
        std::uint32_t window_left = mixed_chunk_cap;
        std::size_t staged_count  = 0;
        for (std::size_t i = 0; i < prefill_lanes.size() && window_left > 0; ++i) {
            const RequestControl::Prefill& entry = *requests[prefill_lanes[i]].prefill;
            const std::uint32_t want = entry.prompt_tokens - entry.cursor;
            nominals[i]              = std::min(window_left, want);
            window_left -= nominals[i];
            ++staged_count;
        }
        if (staged_count == 0 || nominals[0] == 0) {
            throw std::logic_error("mixed round staged no prefill tokens");
        }
        // How many prompt tokens this round consumes must not depend on whether a CUDA graph
        // is available: the pipeline runs the same round on every stage, each with its own
        // graph family, and a stage that replays the graph would otherwise advance a prompt by
        // the graph's 128-rounded chunk while a stage that fell back to the eager body advanced
        // it by the unrounded nominal. The cursors then diverge and the lane's later stages read
        // a KV frontier the earlier ones never wrote — coherent output for a different context.
        // So decide the plan first, from state every stage shares, and let only the *mechanism*
        // vary.
        static const bool kNoMixedGraph = std::getenv("SUROGATE_SERVE_NO_MIXED_GRAPH") != nullptr;
        const std::uint32_t graph_cap =
            prefill_chunk > static_cast<std::uint32_t>(batch_bucket)
                ? ((prefill_chunk - static_cast<std::uint32_t>(batch_bucket)) / 128U) * 128U
                : 0U;
        const std::uint32_t graph_nominal =
            std::min(graph_cap, staged.prompt_tokens - staged.cursor);
        const bool graph_planned = !kNoMixedGraph && staged_count == 1 && staged.use_graph &&
                                   prefill_graphs.has_value() && batch_bucket == rows &&
                                   graph_nominal > 0;
        if (graph_planned) {
            nominals[0]  = graph_nominal; // the graph's chunk, eager fallback included
            staged_count = 1;
        }
        const std::uint32_t nominal = nominals[0];
        const bool final_candidate = staged.cursor + nominal == staged.prompt_tokens;
        mark_workspace_usage(workspace_plan.text_prefill);
        mark_workspace_usage(workspace_plan.ordinary_round);

        schedule::TextContext card(device, model, work, text_kv_view(prefill_sequence),
                                   decoder->linear_attention, io, prefill_hidden, prefill_chunk,
                                   staged.cursor, {}, &decoder->text_kv, decoder->mtp_cache());
        card.set_ple_state(&decoder->ple);
        card.set_stage(stage);
        card.set_sampling(static_cast<const ops::SamplingConfig*>(
            sampling_config.slice(1, static_cast<std::int32_t>(prefill_sequence.lane), 1).data));
        // Graph prompts run every chunk (graph or eager fallback alike) on
        // the shared scratch state slot (PATCHES.md #27) so mixed rounds and
        // classic chunks of one prompt stay on a single state stream.
        card.set_linear_state_slots(
            staged.use_graph
                ? LinearStateSlots::prefill_scratch_state_slot(max_concurrency)
                : LinearStateSlots::current_state_slot(prefill_sequence.lane, max_concurrency),
            rewrite_checkpoints
                ? LinearStateSlots::rewrite_checkpoint_state_slot(prefill_sequence.lane,
                                                                  max_concurrency)
                : kNoRewriteCheckpointSlot);
        card.set_gdn_state_action(schedule::GdnStateAction::UpdateInPlace, nullptr);
        if (staged.use_graph && prefill_graphs.has_value()) {
            card.set_prefill_graph_family(&*prefill_graphs);
        }
        set_device_i32(io.text_kv_table_row, prefill_sequence.kv->text.bound_row());

        schedule::TextContext::MixedDecodeSlice slice;
        slice.ids                = ordinary.tokens.slice(0, 0, rows);
        slice.cache_positions    = ordinary.cache_positions.slice(0, 0, rows);
        slice.rope_positions     = ordinary.rope_positions.slice(0, 0, rows);
        slice.kv_table_rows      = ordinary.text_kv_table_rows.slice(0, 0, rows);
        slice.linear_state_slots = ordinary.lanes.slice(0, 0, rows);
        slice.envelope           = {maximum_frontier + 1, maximum_frontier + 1};
        slice.hidden             = ordinary.hidden.slice(1, 0, rows);
        slice.logits             = ordinary.logits.slice(1, 0, rows);

        (void)final_candidate;
        schedule::PrefillChunkResult chunk{};
        bool graph_hit = false;
        // Mixed-round graphs are ON. Two causes of the corruption they used to
        // produce have been found and fixed: a pad column racing a live lane on
        // its GDN state slot (#50, fixed by capturing per exact decode width),
        // and an unbanded attention envelope (#55, below).
        // SUROGATE_SERVE_NO_MIXED_GRAPH=1 forces the eager path for bisecting.
        if (graph_planned) {
            // The graph ladder rounds the chunk up to a 128 bucket, so the
            // nominal leaves room for both the rounding and the batch bucket
            // inside the prefill_chunk workspace window (graph_nominal above).
            if (graph_nominal > 0) {
                materialize_graph_chunk_window(prefill_sequence, staged.cursor, graph_nominal);
            }
            if (graph_nominal > 0) {
                schedule::TextContext::MixedDecodeSlice bucket_slice;
                bucket_slice.ids                = ordinary.tokens.slice(0, 0, batch_bucket);
                bucket_slice.cache_positions    = ordinary.cache_positions.slice(0, 0, batch_bucket);
                bucket_slice.rope_positions     = ordinary.rope_positions.slice(0, 0, batch_bucket);
                bucket_slice.kv_table_rows      = ordinary.text_kv_table_rows.slice(0, 0, batch_bucket);
                bucket_slice.linear_state_slots = ordinary.lanes.slice(0, 0, batch_bucket);
                // Band the decode envelope exactly as the ordinary decode graphs
                // do, and key the capture on that band, so a replay can never
                // see a frontier outside the window it was captured for. The
                // graph previously baked {1, kv_capacity} here — the one
                // structural difference between it and that proven path.
                // A round without decode rows (a pipeline stage prefilling a
                // staged prompt asynchronously) has no decode band: the decode
                // blocks of the body are skipped for batch 0, so the envelope
                // and band are placeholders and the key carries bucket 0.
                DecodeGraphProfile* mixed_profile =
                    rows > 0 ? &select_graph_profile(ordinary_graphs,
                                                     static_cast<std::uint32_t>(rows),
                                                     maximum_frontier, "mixed round")
                             : nullptr;
                bucket_slice.envelope = mixed_profile
                                            ? ops::GqaExecutionEnvelope{mixed_profile->min_execution_frontier + 1,
                                                                 mixed_profile->max_execution_frontier + 1}
                                            : ops::GqaExecutionEnvelope{1, 1};
                bucket_slice.hidden             = ordinary.hidden.slice(1, 0, batch_bucket);
                bucket_slice.logits             = ordinary.logits.slice(1, 0, batch_bucket);
                // The band is part of the key. This used to pass
                // mixed_profile.topology_class, which graph_profiles_through
                // never sets, so it was zero for every band and the key never
                // carried the band at all: the first mixed graph captured for a
                // (chunk, batch) pair — captured while every lane was young,
                // with the low band's max_visible baked into its attention grid
                // — replayed for the rest of the process, truncating attention
                // for every lane that later crossed the band edge. That was the
                // decode-heavy corruption: victims were always lanes past 512,
                // always in mixed rounds, and runs whose lanes never crossed a
                // band (long prompts, short generations, a moved edge) were
                // clean. The profile's index identifies the band exactly.
                const auto mixed_band =
                    mixed_profile ? static_cast<std::int32_t>(mixed_profile -
                                                              ordinary_graphs.profiles.data())
                                  : 0;
                last_mixed_round.band = mixed_band;
                // Chunk-atomic scratch ownership (see advance_prefill): the captured mixed
                // body runs the prompt's columns on the shared scratch slot.
                decoder->copy_state_slot(
                    LinearStateSlots::current_state_slot(prefill_sequence.lane, max_concurrency),
                    LinearStateSlots::prefill_scratch_state_slot(max_concurrency), device.stream);
                if (card.try_mixed_graph_chunk(
                        std::span<const TokenId>(staged.prompt.token_ids), staged.cursor,
                        graph_nominal, bucket_slice, batch_bucket, mixed_band)) {
                    graph_hit                  = true;
                    last_mixed_round.graph_hit = true;
                    chunk     = schedule::PrefillChunkResult{.processed_tokens = graph_nominal,
                                                             .finalized        = false};
                }
            }
        }
        if (!graph_hit) {
            std::array<schedule::TextContext::MixedPrefillSegment, runtime::kMaximumMixedPrefills>
                segments{};
            for (std::size_t i = 0; i < staged_count; ++i) {
                SequenceState& sequence              = sequences[prefill_lanes[i]];
                const RequestControl::Prefill& entry = *requests[prefill_lanes[i]].prefill;
                // Each prompt's chunk must own the KV it is about to write.
                materialize_sequence_kv(sequence, entry.cursor + nominals[i], 0);
                segments[i] = schedule::TextContext::MixedPrefillSegment{
                    .ids = std::span<const TokenId>(entry.prompt.token_ids)
                               .subspan(entry.cursor, nominals[i]),
                    .kv_base      = static_cast<std::int32_t>(entry.cursor),
                    .kv_table_row = sequence.kv->text.bound_row(),
                    .state_slot   = static_cast<std::int32_t>(
                        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency)),
                    .finalize = false,
                };
            }
            chunk = card.mixed_chunk_multi(
                std::span<const schedule::TextContext::MixedPrefillSegment>(segments.data(),
                                                                            staged_count),
                slice, schedule::TextContext::MixedPrefillFinalize{});
        }

        if (rows > 0) {
            Tensor sampled         = ordinary.sampled_tokens.slice(0, 0, rows);
            Tensor cache_positions = ordinary.cache_positions.slice(0, 0, rows);
            Tensor lanes_tensor    = ordinary.lanes.slice(0, 0, rows);
            ops::scatter(slice.hidden, lanes_tensor, tail_hidden_store, device.stream);
            ops::sample(slice.logits, sampled, TextConfig::token_domain, ordinary.sampling,
                        cache_positions, ops::kSamplePurposeDecode, work, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(ordinary_host_egress, ordinary.egress.data,
                                       sizeof(family::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                       device.stream));
        }
        // The round is enqueued; consume_mixed_round synchronises and commits it.
        mixed_in_flight_.valid        = true;
        mixed_in_flight_.id           = ++mixed_in_flight_counter_;
        mixed_in_flight_.start        = start;
        mixed_in_flight_.rows         = static_cast<std::uint32_t>(lanes.size());
        std::copy(lanes.begin(), lanes.end(), mixed_in_flight_.lanes.begin());
        mixed_in_flight_.prefill_lane_count = static_cast<std::uint32_t>(prefill_lanes.size());
        std::copy(prefill_lanes.begin(), prefill_lanes.end(), mixed_in_flight_.prefill_lanes.begin());
        mixed_in_flight_.staged_count = staged_count;
        mixed_in_flight_.graph_hit    = graph_hit;
        mixed_in_flight_.chunk        = chunk;
        mixed_in_flight_.nominals     = nominals;
        return runtime::RoundHandle{.id = mixed_in_flight_.id, .rows = mixed_in_flight_.rows};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(prefill_sequence, prefill_request);
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::MixedRoundResult ProgramImplCore::consume_mixed_round(runtime::RoundHandle handle) {
    if (!handle.valid() || !mixed_in_flight_.valid || handle.id != mixed_in_flight_.id) {
        throw std::logic_error("consuming a mixed round that is not in flight");
    }
    MixedInFlight& flight = mixed_in_flight_;
    flight.valid          = false;
    const std::span<const std::uint32_t> lanes(flight.lanes.data(), flight.rows);
    const std::span<const std::uint32_t> prefill_lanes(flight.prefill_lanes.data(), flight.prefill_lane_count);
    const auto start                                  = flight.start;
    const std::size_t staged_count                    = flight.staged_count;
    const bool graph_hit                              = flight.graph_hit;
    const schedule::PrefillChunkResult chunk          = flight.chunk;
    const std::array<std::uint32_t, runtime::kMaximumMixedPrefills> nominals = flight.nominals;
    const std::uint32_t prefill_lane = prefill_lanes.front();
    SequenceState& prefill_sequence  = sequences[prefill_lane];
    RequestControl& prefill_request  = requests[prefill_lane];
    try {
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = sequences[lanes[row]];
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid     = base_E + 1;
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }

        runtime::MixedRoundResult result;
        result.round = runtime::BatchedGeneratedRound{
            .tokens = std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(),
                                               lanes.size())};
        // Each staged prompt advances by its own chunk; the graph path processes exactly the
        // first one, so its count matches what the forward consumed.
        // The plan (which prompts, how many tokens each) was fixed before the forward, so a
        // graph replay and the eager body advance the cursors identically.
        std::uint32_t column = 0;
        result.prefill_count = staged_count;
        for (std::size_t i = 0; i < result.prefill_count; ++i) {
            const std::uint32_t lane_id          = prefill_lanes[i];
            SequenceState& sequence              = sequences[lane_id];
            RequestControl::Prefill& entry       = *requests[lane_id].prefill;
            const std::uint32_t processed        = nominals[i];
            const runtime::BeginSummary summary{.prompt_tokens        = entry.prompt_tokens,
                                                .reused_prompt_tokens = entry.base,
                                                .prefix_reuse_path    = entry.reuse};
            entry.cursor += processed;
            sequence.text_kv_valid = entry.cursor;
            result.prefills[i]     = runtime::PrefillStepResult{
                    .summary = summary, .processed_prompt_tokens = processed};
            if (entry.use_graph && graph_hit) {
                // Chunk-atomic scratch ownership: park the chunk's end state in the lane slot
                // whether or not the prompt is finished — another prompt's chunk may run the
                // scratch before this one's next chunk.
                decoder->copy_state_slot(
                    LinearStateSlots::prefill_scratch_state_slot(max_concurrency),
                    LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
                    device.stream);
            }
            if (entry.cursor == entry.prompt_tokens) {
                // The next ordinary advance takes the zero-suffix path off the tail hidden
                // (reusing the whole finalize machinery). Each prompt's last column sits at
                // the end of its own segment.
                copy_tail(sequence,
                          prefill_hidden.slice(
                              1, static_cast<std::int32_t>(column + processed) - 1, 1));
                sequence.tail_hidden_valid = true;
            }
            column += processed;
        }
        return result;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane(prefill_sequence, prefill_request);
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::MixedRoundResult
ProgramImplCore::advance_prefill_mixed(std::span<const std::uint32_t> prefill_lanes,
                                       std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets) {
    return consume_mixed_round(launch_mixed_round(prefill_lanes, lanes, budgets));
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                                  std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            sequence.kv->text.bound_row() < 0 || sequence.kv->backend->bound_row() < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto started = Clock::now();
    try {
        DecodeGraphExecutable* executable = nullptr;
        schedule::MtpGqaEnvelopes envelopes =
            mtp_gqa_envelopes(maximum_frontier, draft_window, capacity);
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch");
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch", device.stream);
            envelopes  = mtp_gqa_envelopes(profile.max_execution_frontier, draft_window, capacity);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = sequences[lanes[row]];
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const std::uint32_t extent =
                std::min({sequence.mtp_draft_count, draft_window, max_by_budget,
                          capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            mtp_host_ingress->mtp_kv_table_rows[row]  = sequence.kv->backend->bound_row();
            mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(sequence.lane);
            mtp_host_ingress->rope_deltas[row]        = sequence.rope_delta;
            mtp_host_ingress->sampling[row]           = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1,
                                    std::min(capacity, frontier + extent + draft_window));
        }

        schedule::MtpBatchContext schedule_state{{device, model, work, decoder->linear_attention,
                                                  replay_records ? &*replay_records : nullptr, io,
                                                  prefill_hidden, prefill_chunk, proposal_head, &decoder->ple},
                                                 decoder->text_kv,
                                                 *decoder->mtp_cache(),
                                                 *io.mtp_decode,
                                                 *mtp_host_ingress,
                                                 *mtp_host_egress,
                                                 tail_hidden_store};

        mark_workspace_usage(workspace_plan.mtp_round);
        schedule::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                   draft_window, envelopes, executable);
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                     std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::DFlash || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const std::uint32_t width           = draft_window + 1U;
    std::uint32_t maximum_frontier      = 0;
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            sequence.kv->text.bound_row() < 0 || sequence.kv->backend->bound_row() < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        const std::uint32_t extent =
            std::min({draft_window, max_by_budget, capacity - sequence.execution_frontier - 1U});
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequence.execution_frontier + extent + 1U);
    }

    const auto started = Clock::now();
    try {
        DecodeGraphExecutable* executable   = nullptr;
        schedule::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, draft_window);
        ops::GqaExecutionEnvelope target_envelope{1, maximum_target_tokens};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(dflash_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "DFlash batch");
            executable      = &install_graph_profile(dflash_graphs, profile, "DFlash batch", device.stream);
            envelopes       = dflash_envelopes(profile.min_execution_frontier,
                                               profile.max_execution_frontier, draft_window);
            target_envelope = {
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(profile.max_execution_frontier) +
                                     draft_window + 1ULL))};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = sequences[lanes[row]];
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t extent =
                std::min({draft_window, max_by_budget, capacity - frontier - 1U});
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            dflash_host_ingress->proposal_extents[row]     = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            dflash_host_ingress->text_kv_table_rows[row]   = sequence.kv->text.bound_row();
            dflash_host_ingress->dflash_kv_table_rows[row] = sequence.kv->backend->bound_row();
            dflash_host_ingress->lanes[row]    = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->sampling[row] = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1U, frontier);
        }

        schedule::DFlashBatchContext schedule_state{{device, model, work, decoder->linear_attention,
                                                     replay_records ? &*replay_records : nullptr,
                                                     io, prefill_hidden, prefill_chunk,
                                                     proposal_head, &decoder->ple},
                                                    decoder->text_kv,
                                                    *dflash,
                                                    *io.dflash_decode,
                                                    *dflash_host_ingress,
                                                    *dflash_host_egress,
                                                    tail_hidden_store};

        mark_workspace_usage(workspace_plan.dflash_round);
        schedule::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                      draft_window, envelopes, target_envelope, executable);
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = sequences[lanes[row]];
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                                 .kind          = PendingKind::Speculative,
                                 .base_E        = base_E,
                                 .base_S        = base_S,
                                 .prompt_tokens = 0,
                                 .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) { return decode_mtp_batch(lanes, budgets); }
    return decode_dflash_batch(lanes, budgets);
}

void ProgramImplCore::resolve_non_speculative_pending(SequenceState& sequence,
                                                      RequestControl& request,
                                                      std::uint32_t accepted_tokens,
                                                      bool terminal) {
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    const std::uint32_t produced = request.pending.produced;
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        produced == 0 || accepted_tokens == 0 || accepted_tokens > produced ||
        (!terminal && accepted_tokens != produced) ||
        (request.pending.kind == PendingKind::Begin && produced != 1)) {
        throw std::logic_error("non-speculative pending round must commit a licensed prefix");
    }

    bool retain_prefix = true;
    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        if (accepted_tokens < produced) {
            // Chained-burst truncation (PATCHES.md #32, terminal-only): trim
            // the ledger and identity to the licensed prefix. The lane's
            // resident GDN state ran the full burst, so the prefix must not
            // be retained for reuse.
            sequence.ledger.resize(request.pending.base_S + accepted_tokens);
            sequence.prefix_identity.truncate(request.pending.base_S + accepted_tokens);
            sequence.text_kv_valid =
                std::min(sequence.text_kv_valid, request.pending.base_E + accepted_tokens);
            retain_prefix = false;
        }
        sequence.execution_frontier = request.pending.base_E + accepted_tokens;
        sequence.ledger_frontier    = request.pending.base_S + accepted_tokens;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) {
        sequence.mtp_draft_count = 0;
        release_sequence_growth_entitlement(sequence);
        unbind_sequence_kv(sequence);
        sequence.retained = retain_prefix;
        static const bool reuse_trace = std::getenv("SUROGATE_SERVE_REUSE_TRACE") != nullptr;
        if (reuse_trace && retain_prefix) {
            std::fprintf(stderr, "reuse-trace: retain lane %u frontier %u ledger %zu\n",
                         sequence.lane, sequence.execution_frontier, sequence.ledger.size());
        }
    }
    request.lifecycle = terminal ? Lifecycle::Complete : Lifecycle::Active;
    request.pending   = {};
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device      = device.device;
    out.max_context = capacity;
    out.kv_capacity = kv_capacity;
    out.kv_cache = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16 : KvCacheStorage::Int8Group64;
    DeviceArena& weights = *model.weights_arena;
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), work.used(), work.peak_used()};
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.cuda_graph_observed_bytes    = graph_observed_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    return out;
}

PagedKVOccupancy ProgramImplCore::kv_occupancy() const noexcept {
    return decoder->text_kv.pool().occupancy();
}

void ProgramImplCore::kv_settle() noexcept {
    if (ElasticKvRegion* region = decoder->text_kv.pool().elastic_region()) {
        try {
            region->wait_idle();
        } catch (...) {}
    }
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    model.weights_arena->reset_peak();
    persistent.reset_peak();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
