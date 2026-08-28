#include "targets/qwen4exp/impl/variant.h"

#include "api/ops/causal_conv1d_silu.h"
#include "api/ops/embedding.h"
#include "api/ops/gdn_gating.h"
#include "api/ops/expert_slot_cache.h"
#include "api/ops/cpu_expert_compute.h"
#include "api/ops/hyper_connection.h"
#include "api/ops/linear.h"
#include "api/ops/ngram_ple.h"
#include "api/ops/scatter.h"
#include "api/ops/sparse_moe.h"
#include "core/device.h"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include <algorithm>
#include <array>
#include <deque>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen4exp::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen4exp_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen4exp::detail {
namespace {

constexpr std::int32_t kStreams  = TextConfig::hc_count;
constexpr std::int32_t kHidden   = TextConfig::hidden;
constexpr std::int32_t kResidual = TextConfig::hc_width;
constexpr std::int32_t kLowRank  = TextConfig::hc_low_rank;
constexpr float kEps             = TextConfig::rms_epsilon;
constexpr auto kPolicy           = ops::LinearPolicy::A16Only;

// The mix hook of a block computes the inject gates its output projection scatters with. The
// family plans the hooks' scratch as transient, so the gates cannot live in the arena between
// the two calls; they use a small device buffer created before any graph capture, and travel
// through a thread-local handle rather than a family-visible parameter.
constexpr std::size_t kInjectScratchBytes = 1u << 20; // [4 streams, T] FP32 up to T = 65536
thread_local Tensor t_inject;
// A host-computed partial of the block being executed: consumed by the next combine.
struct PendingPartial {
    const float* device_alias = nullptr;
    cudaEvent_t join          = nullptr;
};
thread_local PendingPartial t_partial;

struct InjectScratch {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

// Expert slot cache (phase 2): one pool per device, enabled by SUROGATE_SERVE_EXPERT_SLOTS
// (slot count; 0 or unset keeps the zero-copy path). The pool/directory/miss list are device
// memory owned here; the per-layer host banks come from the layer's W8 host Weights.
struct ExpertSlotCache {
    struct Layer {
        ExpertSlotCache* owner = nullptr;
        std::int32_t index     = -1;
        ops::ExpertHostBank bank;
        ops::CpuExpertBank cpu_bank; // host addresses of the same planes
        std::int32_t round_tokens = 0; // set before the host function of a round is enqueued
        // Round bookkeeping for the CPU split: a round may reach the hook in several slices
        // (prefill and mixed rounds); the split decision is per round and each slice is staged
        // at its column offset.
        std::int32_t round_total  = 0;
        std::int32_t round_offset = 0;
        std::int32_t round_slice  = 0; // ordinal of the next slice within the round
        bool round_split          = false;
    };
    // Pinned mirrors of the device job list, one per slice ordinal: the copies run on the
    // main stream, so slice k+1's copy may land while slice k's host function still reads
    // its jobs — each slice therefore owns a mirror.
    struct JobMirror {
        void* block                = nullptr;
        std::int32_t* tokens       = nullptr;
        std::int32_t* experts      = nullptr;
        float* weights             = nullptr;
        long long* count           = nullptr;
    };
    static constexpr int kJobMirrors = 8;
    std::array<JobMirror, kJobMirrors> mirrors{};
    // Host-function contexts. A hook runs at graph capture and its context pointer is baked
    // into the host-function node, so a context must stay valid and unchanged for every
    // replay: one address-stable context per distinct (layer, offset, tokens), shared by every
    // graph and eager round with that slice shape.
    struct SliceContext {
        Layer* entry             = nullptr;
        std::int32_t offset      = 0;
        std::int32_t tokens      = 0;
        std::int32_t ordinal     = 0;
        const JobMirror* mirror  = nullptr;
    };
    std::deque<SliceContext> slice_contexts;
    SliceContext& slice_context(Layer& entry, std::int32_t offset, std::int32_t tokens, std::int32_t ordinal) {
        for (SliceContext& c : slice_contexts) {
            if (c.entry == &entry && c.offset == offset && c.tokens == tokens && c.ordinal == ordinal) { return c; }
        }
        slice_contexts.push_back(SliceContext{&entry, offset, tokens, ordinal, &mirrors[static_cast<std::size_t>(ordinal)]});
        return slice_contexts.back();
    }
    bool enabled = false;
    std::int32_t slots = 0;
    void* pool_memory      = nullptr;
    void* directory_memory = nullptr;
    void* miss_memory      = nullptr;
    ops::ExpertSlotPool pool;
    ops::ExpertSlotDirectory directory;
    ops::ExpertMissList misses;
    std::vector<Layer> layers;

    Layer& layer(const SparseMoePayload& weights) {
        if (weights.layer < 0 || weights.layer >= static_cast<std::int32_t>(layers.size())) {
            throw std::logic_error("qwen4exp: MoE payload has no layer index for the slot cache");
        }
        Layer& entry = layers[static_cast<std::size_t>(weights.layer)];
        if (entry.owner == nullptr) {
            entry.owner = this;
            entry.index = weights.layer;
            entry.bank  = ops::expert_host_bank(ops::kSparseMoeFlashNextGeometry,
                                                weights.op.routed_gate_up, weights.op.routed_down);
            if (weights.host_gate_up != nullptr && weights.host_down != nullptr) {
                // Plane offsets are the same in the host and device views of the object.
                const auto gate_scale_offset = static_cast<const std::byte*>(weights.op.routed_gate_up.scales) -
                                               static_cast<const std::byte*>(weights.op.routed_gate_up.qdata);
                const auto down_scale_offset = static_cast<const std::byte*>(weights.op.routed_down.scales) -
                                               static_cast<const std::byte*>(weights.op.routed_down.qdata);
                entry.cpu_bank.gate_up_codes  = weights.host_gate_up;
                entry.cpu_bank.gate_up_scales = weights.host_gate_up + gate_scale_offset;
                entry.cpu_bank.down_codes     = weights.host_down;
                entry.cpu_bank.down_scales    = weights.host_down + down_scale_offset;
            }
        }
        return entry;
    }

    // --- CPU expert split (SUROGATE_SERVE_CPU_MOE_SHARE=<fraction of misses>) ---
    std::uint32_t cpu_share_q16 = 0;
    std::int32_t cpu_max_tokens = 0;
    // Prefill rounds (wider than cpu_max_tokens) use their own share; 0 keeps the full gather.
    std::uint32_t cpu_prefill_share_q16  = 0;
    std::int32_t cpu_prefill_max_tokens  = 0;
    bool auto_share             = false;
    bool share_measured         = false;
    std::unique_ptr<ops::CpuExpertPool> cpu_pool;
    cudaStream_t cpu_stream = nullptr; // side stream: the host round overlaps the GPU experts
    cudaEvent_t fork_event   = nullptr;
    cudaEvent_t join_event   = nullptr;
    cudaEvent_t copied_event = nullptr; // the slice's staging copies are done (the job list may be reused)
    std::int32_t stage_tokens = 0;      // columns the host staging holds
    void* cpu_jobs_memory = nullptr;
    ops::ExpertCpuJobList cpu_jobs;
    // Pinned host staging: activations, jobs, count, and the FP32 partial the GPU adds back.
    std::uint16_t* x_host   = nullptr;
    float* out_host         = nullptr;
    void* out_device_alias  = nullptr;
    void* jobs_host_block           = nullptr; // pinned mirror of the device job list (same carve)
    std::size_t jobs_block_bytes    = 0;
    std::int32_t* jobs_tokens_host  = nullptr;
    std::int32_t* jobs_experts_host = nullptr;
    float* jobs_weights_host        = nullptr;
    long long* jobs_count_host      = nullptr;
    std::int32_t cpu_min_tokens     = 4; // below this the host round-trip costs more than it saves
    std::vector<ops::CpuExpertJob> job_scratch;

    bool cpu_split_enabled() const { return cpu_pool != nullptr; }
    // The share for a round of `tokens` columns in total (0 = no split): decode-sized rounds
    // use the decode share, wider ones the prefill share, and nothing wider than the staging.
    std::uint32_t share_for(std::int32_t tokens) const {
        if (tokens > stage_tokens) { return 0U; }
        if (tokens <= cpu_max_tokens) { return cpu_share_q16; }
        return tokens <= cpu_prefill_max_tokens ? cpu_prefill_share_q16 : 0U;
    }
    // Called by post_mixer before the MoE op: fixes the round's split decision and share.
    void begin_round(Layer& entry, std::int32_t tokens) {
        entry.round_total  = tokens;
        entry.round_offset = 0;
        entry.round_slice  = 0;
        entry.round_split  = cpu_split_enabled() && share_for(tokens) > 0 && tokens >= cpu_min_tokens &&
                             entry.cpu_bank.gate_up_codes != nullptr;
    }

    static void run_cpu_round(void* context) {
        auto* slice            = static_cast<SliceContext*>(context);
        Layer* entry           = slice->entry;
        ExpertSlotCache& cache = *entry->owner;
        const std::int32_t hidden = ops::kSparseMoeFlashNextGeometry.hidden;
        const std::int32_t tokens = slice->tokens;
        const std::size_t column0 = static_cast<std::size_t>(slice->offset) * hidden;
        const JobMirror& mirror   = *slice->mirror;
        const long long count     = std::min<long long>(*mirror.count, cache.cpu_jobs.capacity);
        std::fill_n(cache.out_host + column0, static_cast<std::size_t>(hidden) * tokens, 0.0F);
        if (count <= 0) { return; }
        cache.job_scratch.resize(static_cast<std::size_t>(count));
        for (long long i = 0; i < count; ++i) {
            cache.job_scratch[static_cast<std::size_t>(i)] = {mirror.tokens[i], mirror.experts[i], mirror.weights[i]};
        }
        ops::CpuExpertRound round{cache.x_host + column0, cache.out_host + column0, tokens, cache.job_scratch};
        cache.cpu_pool->run(entry->cpu_bank, round);
    }

    // Stages one slice of the round (columns [round_offset, round_offset + tokens) of the
    // block) and forks its host round onto the side stream; the block's combine joins the
    // last slice's event and adds the round-wide partial (v2 overlap).
    void cpu_round(Layer& entry, const Tensor& x, Tensor& /*destination*/, cudaStream_t stream) {
        const std::int32_t hidden = ops::kSparseMoeFlashNextGeometry.hidden;
        const std::int32_t tokens = static_cast<std::int32_t>(x.numel() / hidden);
        const std::int32_t offset = entry.round_offset;
        if (offset + tokens > stage_tokens) {
            throw std::logic_error("qwen4exp: CPU split round wider than its staging");
        }
        entry.round_tokens = tokens;
        const std::int32_t ordinal = entry.round_slice++;
        if (ordinal >= kJobMirrors) {
            throw std::logic_error("qwen4exp: CPU split round has more slices than job mirrors");
        }
        SliceContext& slice = slice_context(entry, offset, tokens, ordinal);
        const std::size_t column0 = static_cast<std::size_t>(offset) * hidden;
        // The staging copies run on the main stream: they are small, and stream order then
        // guarantees the next slice's resolve cannot rewrite the job list before it is copied
        // — without making the main stream wait behind the side stream (which would queue it
        // behind the previous layer's host round and serialise host and GPU).
        CUDA_CHECK(cudaMemcpyAsync(x_host + column0, x.data,
                                   static_cast<std::size_t>(hidden) * tokens * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(slice.mirror->block, cpu_jobs_memory, jobs_block_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        // Fork only the host function onto the side stream so it overlaps the GPU experts.
        CUDA_CHECK(cudaEventRecord(fork_event, stream));
        CUDA_CHECK(cudaStreamWaitEvent(cpu_stream, fork_event, 0));
        CUDA_CHECK(cudaLaunchHostFunc(cpu_stream, &ExpertSlotCache::run_cpu_round, &slice));
        CUDA_CHECK(cudaEventRecord(join_event, cpu_stream));
        entry.round_offset = offset + tokens;
        t_partial = PendingPartial{static_cast<const float*>(out_device_alias), join_event};
    }

    static void resolve_round(void* context, const Tensor& ids, const Tensor& alpha,
                              const Tensor& x, Tensor& destination, cudaStream_t stream) {
        auto* entry = static_cast<Layer*>(context);
        ExpertSlotCache& cache = *entry->owner;
        const std::int32_t tokens =
            static_cast<std::int32_t>(x.numel() / ops::kSparseMoeFlashNextGeometry.hidden);
        // The split decision is per round (begin_round); every slice of the round follows it.
        const bool split          = entry->round_split && x.data != nullptr && destination.data != nullptr;
        const std::uint32_t share = split ? cache.share_for(entry->round_total) : 0U;
        if (split) {
            ops::expert_slot_resolve(ids, alpha, entry->index, cache.directory, cache.misses,
                                     &cache.cpu_jobs, share, stream);
        } else {
            ops::expert_slot_resolve(ids, entry->index, cache.directory, cache.misses, stream);
        }
        ops::expert_slot_gather(entry->bank, cache.misses, cache.pool, stream);
        if (split) { cache.cpu_round(*entry, x, destination, stream); }
        if (cache.stats_every > 0) { cache.record_stats(ids, stream); }
    }

    // Diagnostic hit-rate readout (SUROGATE_SERVE_EXPERT_STATS=<rounds>): every `stats_every`
    // rounds the miss count is read back synchronously, so it perturbs throughput and is not
    // for measurements. A "round" here is one layer's resolve.
    std::int64_t stats_every  = 0;
    std::int64_t stats_rounds = 0;
    std::int64_t stats_misses = 0;
    std::int64_t stats_ids    = 0;
    void record_stats(const Tensor& ids, cudaStream_t stream) {
        // Hooks run once at capture time and never during graph replays, and a synchronous
        // readout is illegal on a capturing stream: the readout is an eager-mode
        // (--no-cuda-graph) diagnostic only.
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            return;
        }
        ++stats_rounds;
        stats_ids += ids.numel();
        if (stats_rounds % stats_every != 0) { return; }
        long long misses = 0;
        CUDA_CHECK(cudaMemcpyAsync(&misses, cache_count_ptr(), sizeof(misses), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        stats_misses += misses * stats_every; // sampled: one readout stands for the window
        std::fprintf(stderr,
                     "qwen4exp: expert cache rounds=%lld sampled misses/round=%lld ids/round=%.1f "
                     "(cumulative sampled miss share %.1f%%)\n",
                     static_cast<long long>(stats_rounds), misses,
                     static_cast<double>(stats_ids) / static_cast<double>(stats_rounds),
                     100.0 * static_cast<double>(stats_misses) / static_cast<double>(stats_ids));
    }
    const void* cache_count_ptr() const { return misses.count.data; }
};

std::mutex& expert_slot_mutex() {
    static std::mutex mutex;
    return mutex;
}
std::unordered_map<int, std::uint32_t>& configured_expert_slots() {
    static std::unordered_map<int, std::uint32_t> configured;
    return configured;
}
std::unordered_map<int, float>& configured_cpu_share() {
    static std::unordered_map<int, float> configured;
    return configured;
}
std::unordered_map<int, std::pair<float, std::uint32_t>>& configured_cpu_prefill() {
    static std::unordered_map<int, std::pair<float, std::uint32_t>> map;
    return map;
}
std::unordered_map<int, std::uint32_t>& configured_cpu_min_tokens() {
    static std::unordered_map<int, std::uint32_t> configured;
    return configured;
}

ExpertSlotCache& expert_slot_cache_for_current_device() {
    static std::unordered_map<int, ExpertSlotCache> registry;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(expert_slot_mutex());
    auto it = registry.find(device);
    if (it != registry.end()) { return it->second; }
    ExpertSlotCache& cache = registry[device];
    // --expert-slots N wins when set; otherwise the environment knob; 0 keeps zero-copy.
    long requested = 0;
    if (auto configured = configured_expert_slots().find(device);
        configured != configured_expert_slots().end() && configured->second > 0) {
        requested = static_cast<long>(configured->second);
    } else if (const char* raw = std::getenv("SUROGATE_SERVE_EXPERT_SLOTS");
               raw != nullptr && *raw != '\0') {
        requested = std::strtol(raw, nullptr, 10);
    }
    if (requested <= 0) { return cache; }
    const auto geometry = ops::kSparseMoeFlashNextGeometry;
    // A round can touch every expert of a layer, and resolve must never leave a routed expert
    // unmapped, so the pool holds at least one layer's worth of experts.
    cache.slots         = std::max(static_cast<std::int32_t>(requested), geometry.experts);
    const std::size_t pool_bytes = ops::expert_slot_pool_bytes(geometry, cache.slots);
    const std::size_t dir_bytes =
        ops::expert_slot_directory_bytes(TextConfig::layers, geometry.experts, cache.slots);
    // A round can miss at most one whole layer's expert set.
    const std::size_t miss_bytes = ops::expert_miss_list_bytes(geometry.experts);
    CUDA_CHECK(cudaMalloc(&cache.pool_memory, pool_bytes));
    CUDA_CHECK(cudaMalloc(&cache.directory_memory, dir_bytes));
    CUDA_CHECK(cudaMalloc(&cache.miss_memory, miss_bytes));
    cache.pool      = ops::create_expert_slot_pool(geometry, cache.slots, cache.pool_memory);
    cache.directory = ops::create_expert_slot_directory(TextConfig::layers, geometry.experts,
                                                        cache.slots, cache.directory_memory,
                                                        nullptr);
    cache.misses    = ops::create_expert_miss_list(geometry.experts, cache.miss_memory);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    cache.layers.resize(static_cast<std::size_t>(TextConfig::layers));
    cache.enabled = true;
    if (const char* stats = std::getenv("SUROGATE_SERVE_EXPERT_STATS"); stats != nullptr && *stats != '\0') {
        cache.stats_every = std::strtol(stats, nullptr, 10);
    }
    double fraction = 0.0;
    bool auto_share = false;
    if (auto configured = configured_cpu_share().find(device);
        configured != configured_cpu_share().end() && configured->second != 0.0F) {
        if (configured->second < 0.0F) {
            auto_share = true;
            fraction   = 0.7; // placeholder until prepare_expert_split measures the rates
        } else {
            fraction = static_cast<double>(configured->second);
        }
    } else if (const char* share = std::getenv("SUROGATE_SERVE_CPU_MOE_SHARE");
               share != nullptr && *share != '\0') {
        if (std::string(share) == "auto") {
            auto_share = true;
            fraction   = 0.7;
        } else {
            fraction = std::strtod(share, nullptr);
        }
    }
    // Prefill share: -1 (unset) → 0.5 when the split is on (measured optimum at concurrency;
    // 0.7 is better for a single user), 0 turns the prefill split off.
    double prefill_fraction        = -1.0;
    std::uint32_t prefill_chunk    = 0;
    if (auto configured = configured_cpu_prefill().find(device); configured != configured_cpu_prefill().end()) {
        prefill_fraction = static_cast<double>(configured->second.first);
        prefill_chunk    = configured->second.second;
    }
    if (const char* share = std::getenv("SUROGATE_SERVE_CPU_MOE_PREFILL_SHARE"); share != nullptr && *share != '\0') {
        prefill_fraction = std::strtod(share, nullptr);
        if (prefill_chunk == 0) { prefill_chunk = 2048; }
    }
    if (prefill_fraction < 0.0) { prefill_fraction = fraction > 0.0 ? 0.5 : 0.0; }
    if (prefill_chunk == 0) { prefill_chunk = 2048; }
    {
        if (fraction > 0.0 || prefill_fraction > 0.0) {
            cache.cpu_share_q16 = static_cast<std::uint32_t>(std::min(1.0, std::max(0.0, fraction)) * 65536.0);
            cache.cpu_max_tokens = 64; // decode and small-T rounds use cpu_share; wider rounds are prefill
            if (prefill_fraction > 0.0 && prefill_chunk > 0) {
                cache.cpu_prefill_share_q16 = static_cast<std::uint32_t>(std::min(1.0, prefill_fraction) * 65536.0);
                cache.cpu_prefill_max_tokens = static_cast<std::int32_t>(prefill_chunk);
            }
            const std::int32_t hidden = geometry.hidden;
            // Staging covers the widest round: a prefill chunk plus the decode lanes of a mixed
            // round (or just the decode lanes without a prefill share).
            const std::int32_t stage_tokens = cache.cpu_prefill_max_tokens > 0
                                                  ? cache.cpu_prefill_max_tokens + 256
                                                  : cache.cpu_max_tokens;
            cache.stage_tokens              = stage_tokens;
            // Jobs per slice: a slice is at most a prefill chunk (or the decode lanes) wide.
            const std::int32_t capacity = std::max(cache.cpu_max_tokens, cache.cpu_prefill_max_tokens) *
                                          geometry.experts_per_token;
            CUDA_CHECK(cudaMalloc(&cache.cpu_jobs_memory, ops::expert_cpu_job_list_bytes(capacity)));
            cache.cpu_jobs = ops::create_expert_cpu_job_list(capacity, cache.cpu_jobs_memory);
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&cache.x_host),
                                     static_cast<std::size_t>(hidden) * stage_tokens * sizeof(std::uint16_t),
                                     cudaHostAllocPortable));
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&cache.out_host),
                                     static_cast<std::size_t>(hidden) * stage_tokens * sizeof(float),
                                     cudaHostAllocMapped | cudaHostAllocPortable));
            CUDA_CHECK(cudaHostGetDevicePointer(&cache.out_device_alias, cache.out_host, 0));
            cache.jobs_block_bytes = ops::expert_cpu_job_list_bytes(capacity);
            for (auto& m : cache.mirrors) {
                CUDA_CHECK(cudaHostAlloc(&m.block, cache.jobs_block_bytes, cudaHostAllocPortable));
                // Same carve as the device list, so field offsets match after the block copy.
                const ops::ExpertCpuJobList view = ops::create_expert_cpu_job_list(capacity, m.block);
                m.tokens  = static_cast<std::int32_t*>(view.tokens.data);
                m.experts = static_cast<std::int32_t*>(view.experts.data);
                m.weights = static_cast<float*>(view.weights.data);
                m.count   = static_cast<long long*>(view.count.data);
            }
            cache.jobs_host_block   = cache.mirrors[0].block;
            cache.jobs_tokens_host  = cache.mirrors[0].tokens;
            cache.jobs_experts_host = cache.mirrors[0].experts;
            cache.jobs_weights_host = cache.mirrors[0].weights;
            cache.jobs_count_host   = cache.mirrors[0].count;
            if (auto configured = configured_cpu_min_tokens().find(device);
                configured != configured_cpu_min_tokens().end() && configured->second > 0) {
                cache.cpu_min_tokens = static_cast<std::int32_t>(configured->second);
            } else if (const char* min = std::getenv("SUROGATE_SERVE_CPU_MOE_MIN_TOKENS");
                       min != nullptr && *min != '\0') {
                cache.cpu_min_tokens = static_cast<std::int32_t>(std::strtol(min, nullptr, 10));
            }
            ops::CpuExpertPoolOptions pool_options;
            if (const char* threads = std::getenv("SUROGATE_SERVE_CPU_MOE_THREADS"); threads != nullptr && *threads != '\0') {
                pool_options.threads = static_cast<std::uint32_t>(std::strtoul(threads, nullptr, 10));
            }
            cache.cpu_pool = std::make_unique<ops::CpuExpertPool>(geometry, pool_options);
            CUDA_CHECK(cudaStreamCreateWithFlags(&cache.cpu_stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&cache.fork_event, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&cache.join_event, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&cache.copied_event, cudaEventDisableTiming));
            cache.auto_share = auto_share;
            std::fprintf(stderr, "qwen4exp: CPU expert split enabled: %.0f%% of decode misses, %.0f%% of prefill misses (rounds up to %d columns) on %u host threads%s\n",
                         100.0 * fraction, 100.0 * prefill_fraction, cache.cpu_prefill_max_tokens,
                         cache.cpu_pool->threads(), auto_share ? " (auto: measured at startup)" : "");
        }
    }
    std::fprintf(stderr, "qwen4exp: expert slot cache enabled: %d slots (%.1f GiB pool)\n",
                 cache.slots, static_cast<double>(pool_bytes) / (1024.0 * 1024.0 * 1024.0));
    return cache;
}

InjectScratch& inject_scratch_for_current_device() {
    static std::mutex mutex;
    static std::unordered_map<int, InjectScratch> registry;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(mutex);
    return registry[device];
}

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier, const std::vector<std::uint32_t>& ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::size_t round_up(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// Debug parity dumps: SUROGATE_SERVE_DUMP_RESIDUAL=<dir> writes the residual streams before
// every layer and the final mixed hidden state of the first forwards with at most 64 columns,
// as raw BF16 with a 16-byte header {magic, rows, columns, forward}. Synchronises the stream.
struct ResidualDump {
    std::string dir;
    int forward = 0;
    int limit   = 0;
};

ResidualDump& residual_dump() {
    static ResidualDump dump = [] {
        ResidualDump out;
        if (const char* raw = std::getenv("SUROGATE_SERVE_DUMP_RESIDUAL"); raw != nullptr && *raw) {
            out.dir   = raw;
            out.limit = 2;
        }
        return out;
    }();
    return dump;
}

void dump_tensor(const Tensor& tensor, const std::string& name, int forward, cudaStream_t stream) {
    const std::size_t bytes = tensor.bytes();
    std::vector<std::byte> host(bytes);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tensor.data, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::string path = residual_dump().dir + "/" + name + ".bin";
    FILE* file             = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { return; }
    const std::int32_t header[4] = {0x52455344, tensor.ne[0], tensor.ne[1], forward};
    std::fwrite(header, sizeof(header), 1, file);
    std::fwrite(host.data(), 1, bytes, file);
    std::fclose(file);
}

// Layer 0's prologue marks the start of a forward; the layer index is kept for the
// per-block intermediate dumps of the first two layers.
int g_dump_layer = -1;
int g_dump_block = 0; // mixes seen in the current layer: 0 = mixer block, 1 = MLP block

void maybe_dump_layer(int layer, const Tensor& residual, cudaStream_t stream) {
    ResidualDump& dump = residual_dump();
    if (dump.dir.empty() || residual.ne[1] > 64) { return; }
    if (layer == 0) { dump.forward += 1; }
    g_dump_layer = layer;
    g_dump_block = 0;
    if (dump.forward > dump.limit) { return; }
    dump_tensor(residual, "f" + std::to_string(dump.forward) + "_layer" + std::to_string(layer),
                dump.forward, stream);
}

void maybe_dump_block(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    ResidualDump& dump = residual_dump();
    if (dump.dir.empty() || tensor.ne[1] > 64 || dump.forward > dump.limit || g_dump_layer < 0 ||
        g_dump_layer > 1) {
        return;
    }
    dump_tensor(tensor,
                "f" + std::to_string(dump.forward) + "_L" + std::to_string(g_dump_layer) +
                    (g_dump_block == 0 ? "_mixer_" : "_mlp_") + tag,
                dump.forward, stream);
}

void maybe_dump_final(const Tensor& hidden, cudaStream_t stream) {
    ResidualDump& dump = residual_dump();
    if (dump.dir.empty() || hidden.ne[1] > 64 || dump.forward > dump.limit) { return; }
    dump_tensor(hidden, "f" + std::to_string(dump.forward) + "_final", dump.forward, stream);
}

std::size_t plane_bytes(std::int32_t rows, std::int32_t tokens, DType dtype) {
    return round_up(static_cast<std::size_t>(rows) * static_cast<std::size_t>(tokens) *
                    dtype_size(dtype));
}

std::size_t mix_capacity(std::int32_t first, std::int32_t last) {
    return ops::hyper_connection_mix_workspace_capacity_bytes(kStreams, kHidden, kLowRank, first,
                                                              last);
}

std::size_t w8_capacity(std::int32_t rows, std::int32_t columns, std::int32_t first,
                        std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, rows, columns, kPolicy, first,
                                                last);
}

// Mixes the residual streams into `hidden` and keeps the inject gates for the combine.
void mix_into(const Tensor& residual, const ops::HyperConnectionWeights& weights, Tensor& hidden,
              WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    const InjectScratch& scratch = inject_scratch_for_current_device();
    const std::size_t needed =
        static_cast<std::size_t>(kStreams) * static_cast<std::size_t>(tokens) * sizeof(float);
    if (scratch.data == nullptr || needed > scratch.bytes) {
        throw std::logic_error("qwen4exp: inject scratch is missing or too small for this forward");
    }
    Tensor inject(scratch.data, DType::FP32, {kStreams, tokens});
    ops::hyper_connection_mix(residual, weights, kStreams, kEps, hidden, &inject, workspace,
                              stream);
    t_inject = inject;
    maybe_dump_block("mixed", hidden, stream);
    maybe_dump_block("inject", inject, stream);
}

// Scatters a block output into the residual streams with the gates the mix kept.
void combine_into(const Tensor& block_output, Tensor& residual, cudaStream_t stream) {
    if (t_inject.data == nullptr || t_inject.ne[1] != residual.ne[1]) {
        throw std::logic_error("qwen4exp: combine without a matching mix");
    }
    maybe_dump_block("blockout", block_output, stream);
    if (t_partial.device_alias != nullptr) {
        // Join the host round (side stream) before the combine reads its partial.
        CUDA_CHECK(cudaStreamWaitEvent(stream, t_partial.join, 0));
        ops::hyper_connection_combine(block_output, t_partial.device_alias, t_inject, residual, stream);
        t_partial = PendingPartial{};
    } else {
        ops::hyper_connection_combine(block_output, t_inject, residual, stream);
    }
    maybe_dump_block("combined", residual, stream);
    g_dump_block += 1;
    t_inject = Tensor{};
}

Tensor rows_of(const Tensor& fused, std::int32_t begin, std::int32_t count, WorkspaceArena& work,
               cudaStream_t stream) {
    Tensor out = work.alloc(DType::BF16, {count, fused.ne[1]});
    ops::extract_bf16_columns(fused, begin, out, stream);
    return out;
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return graph_profiles_through(capacity - 1, {127, 511, 2047});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t, std::uint32_t) {
    return {};
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

// --- residual hooks ---------------------------------------------------------------------------

void Variant::embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    // Transient: the broadcast consumes the embedding before anything else runs.
    auto scope                = workspace.scope();
    const std::int32_t tokens = residual.ne[1];
    Tensor embedded           = workspace.alloc(DType::BF16, {kHidden, tokens});
    ops::embedding(ids, model.token_embedding, embedded, stream);
    ops::broadcast_streams(embedded, kStreams, residual, stream);
}

void Variant::configure_expert_slots(std::uint32_t slots) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(expert_slot_mutex());
    configured_expert_slots()[device] = slots;
}

void Variant::prepare_expert_split(const ModelView& model) {
    ExpertSlotCache& cache = expert_slot_cache_for_current_device();
    if (!cache.enabled || !cache.cpu_split_enabled() || !cache.auto_share || cache.share_measured) {
        if (cache.auto_share && !cache.share_measured) {
            std::fprintf(stderr, "qwen4exp: CPU split auto share not measured (cache %d, pool %d)\n",
                         int(cache.enabled), int(cache.cpu_split_enabled()));
        }
        return;
    }
    // Layer 0's routed experts: the first MoE layer's payload.
    const SparseMoePayload* payload = nullptr;
    for (const auto& gdn : model.gdn_layers) {
        if (gdn.post_mixer.layer >= 0) { payload = &gdn.post_mixer; break; }
    }
    if (payload == nullptr || payload->host_gate_up == nullptr) {
        std::fprintf(stderr, "qwen4exp: CPU split auto share not measured (no MoE layer with a host bank)\n");
        return;
    }
    ExpertSlotCache::Layer& layer = cache.layer(*payload);
    const auto geometry           = ops::kSparseMoeFlashNextGeometry;
    constexpr int kExpertsTimed   = 64;
    constexpr int kRepeats        = 4;
    cudaStream_t stream           = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // PCIe: gather kExpertsTimed experts into the first slots (the pool is otherwise empty).
    std::vector<std::int32_t> slots(kExpertsTimed), experts(kExpertsTimed);
    for (int i = 0; i < kExpertsTimed; ++i) { slots[i] = i; experts[i] = i; }
    const long long count = kExpertsTimed;
    CUDA_CHECK(cudaMemcpy(cache.misses.slots.data, slots.data(), slots.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache.misses.experts.data, experts.data(), experts.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cache.misses.count.data, &count, sizeof(count), cudaMemcpyHostToDevice));
    ops::expert_slot_gather(layer.bank, cache.misses, cache.pool, stream); // warm
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int r = 0; r < kRepeats; ++r) { ops::expert_slot_gather(layer.bank, cache.misses, cache.pool, stream); }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float gather_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&gather_ms, t0, t1));
    const double expert_bytes = static_cast<double>(layer.bank.gate_up_codes_bytes_per_expert + layer.bank.gate_up_scales_bytes_per_expert +
                                                    layer.bank.down_codes_bytes_per_expert + layer.bank.down_scales_bytes_per_expert);
    const double pcie_gbs = expert_bytes * kExpertsTimed * kRepeats / (gather_ms * 1e-3) / 1e9;
    // Host: the same experts as jobs over 8 tokens of zeros-free activations (the round's x is
    // arbitrary for timing; use the staged buffer as is).
    const int tokens = 8;
    cache.job_scratch.resize(static_cast<std::size_t>(kExpertsTimed));
    for (int i = 0; i < kExpertsTimed; ++i) { cache.job_scratch[static_cast<std::size_t>(i)] = {i % tokens, i, 0.1F}; }
    std::fill_n(cache.x_host, static_cast<std::size_t>(geometry.hidden) * tokens, static_cast<std::uint16_t>(0x3F80)); // 1.0 in BF16
    ops::CpuExpertRound round{cache.x_host, cache.out_host, tokens, cache.job_scratch};
    std::fill_n(cache.out_host, static_cast<std::size_t>(geometry.hidden) * tokens, 0.0F);
    cache.cpu_pool->run(layer.cpu_bank, round); // warm
    const auto h0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRepeats; ++r) { cache.cpu_pool->run(layer.cpu_bank, round); }
    const double host_s   = std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count();
    const double host_gbs = expert_bytes * kExpertsTimed * kRepeats / host_s / 1e9;
    double share = host_gbs / (host_gbs + pcie_gbs);
    share        = std::min(0.9, std::max(0.3, share));
    cache.cpu_share_q16  = static_cast<std::uint32_t>(share * 65536.0);
    cache.share_measured = true;
    // Leave the pool directory clean for the real rounds.
    ops::expert_slot_directory_reset(cache.directory, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::fprintf(stderr, "qwen4exp: CPU split auto share: host %.0f GB/s, PCIe gather %.0f GB/s -> %.0f%% of misses on the host\n",
                 host_gbs, pcie_gbs, 100.0 * share);
}

void Variant::configure_cpu_moe_prefill(float share, std::uint32_t prefill_chunk) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    configured_cpu_prefill()[device] = {share, prefill_chunk};
}

void Variant::configure_cpu_moe_min_tokens(std::uint32_t tokens) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(expert_slot_mutex());
    configured_cpu_min_tokens()[device] = tokens;
}

void Variant::configure_cpu_moe_share(float share) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(expert_slot_mutex());
    configured_cpu_share()[device] = share;
}

void Variant::prewarm_device_scratch() {
    InjectScratch& scratch = inject_scratch_for_current_device();
    if (scratch.data == nullptr) {
        CUDA_CHECK(cudaMalloc(&scratch.data, kInjectScratchBytes));
        scratch.bytes = kInjectScratchBytes;
    }
    expert_slot_cache_for_current_device();
}

void Variant::final_residual_mix(const ModelView& model, const Tensor& residual, Tensor& hidden,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    ops::hyper_connection_mix(residual, model.output_mix, kStreams, kEps, hidden, nullptr,
                              workspace, stream);
    maybe_dump_final(hidden, stream);
}

void Variant::attention_norm(const Tensor& residual, const FullAttentionProjectionWeights& weights,
                             Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream) {
    mix_into(residual, weights.mix, hidden, workspace, stream);
}

void Variant::post_mixer_norm(const Tensor& residual, const PostMixerWeights& weights,
                              Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream) {
    mix_into(residual, weights.mix, hidden, workspace, stream);
}

void Variant::layer_prologue(const ModelView& model, int layer, Tensor& residual,
                             const qwen3_6::detail::PrologueColumns& columns,
                             NgramPleStatePool* ple_state, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    maybe_dump_layer(layer, residual, stream);
    if (layer != model.ple.layer) { return; }
    if (ple_state == nullptr || ple_state->empty()) {
        throw std::logic_error("qwen4exp: the PLE layer needs its state pool");
    }
    ops::NgramPleColumns ple_columns{columns.ids, columns.segment_begin, columns.slots,
                                     columns.segment_last};
    ops::NgramPleState state{ple_state->history, ple_state->conv_state};
    maybe_dump_block("ple_in", residual, stream);
    ops::ngram_ple_forward(residual, ple_columns, model.ple.hash, model.ple.table, model.ple.op,
                           state, kStreams, TextConfig::ple_conv_kernel,
                           TextConfig::ple_conv_dilation, kEps, workspace, stream);
    maybe_dump_block("ple_out", residual, stream);
}

void Variant::debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    ResidualDump& dump = residual_dump();
    if (dump.dir.empty() || dump.forward > dump.limit || g_dump_layer < 0 || g_dump_layer > 1) {
        return;
    }
    // Flatten to [ne0, rest] so the dump header describes the payload.
    const std::int64_t rest = tensor.numel() / tensor.ne[0];
    if (rest > 8192) { return; }
    Tensor flat = tensor.view({tensor.ne[0], static_cast<std::int32_t>(rest)});
    dump_tensor(flat,
                "f" + std::to_string(dump.forward) + "_L" + std::to_string(g_dump_layer) +
                    (g_dump_block == 0 ? "_mixer_" : "_mlp_") + tag,
                dump.forward, stream);
}

NgramPleStatePoolSpec Variant::ple_state_spec(std::int32_t slot_count) {
    return NgramPleStatePoolSpec{
        .history_tokens = TextConfig::ple_ngram - 1,
        .conv_history   = TextConfig::ple_conv_history,
        .channels       = kResidual,
        .slot_count     = slot_count,
        .eos_token      = TextConfig::eos_token,
    };
}

std::size_t Variant::layer_prologue_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    // The PLE forward and the transient embedding never overlap; reserve the larger.
    return std::max(ops::ngram_ple_workspace_capacity_bytes(kStreams, kHidden,
                                                            TextConfig::ple_embed,
                                                            TextConfig::ple_heads, first, last),
                    plane_bytes(kHidden, last, DType::BF16));
}

// --- projections -------------------------------------------------------------------------------

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                = workspace.scope();
    const std::int32_t tokens = hidden.ne[1];
    Tensor fused = workspace.alloc(DType::BF16, {TextConfig::query_projection_rows, tokens});
    ops::linear(hidden, weights.query_key_gate_value, fused, kPolicy, workspace, stream);
    // Row order from the converter: q | k | gate | v.
    ops::extract_bf16_columns(fused, 0, query, stream);
    ops::extract_bf16_columns(fused, TextConfig::query_size, key, stream);
    ops::extract_bf16_columns(fused, TextConfig::query_size + TextConfig::kv_size, gate, stream);
    ops::extract_bf16_columns(fused, 2 * TextConfig::query_size + TextConfig::kv_size, value,
                              stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    Tensor output  = workspace.alloc(DType::BF16, {kHidden, attention.ne[1]});
    ops::linear(attention, weight, output, kPolicy, workspace, stream);
    combine_into(output, residual, stream);
}

void Variant::mtp_attention_projection(const Tensor&, const MtpAttentionProjectionWeights&,
                                       Tensor&, Tensor&, Tensor&, Tensor&, WorkspaceArena&,
                                       cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

void Variant::mtp_kv_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                Tensor&, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

void Variant::mtp_q_gate_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                    Tensor&, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                = workspace.scope();
    const std::int32_t tokens = static_cast<std::int32_t>(hidden.ne[1] * hidden.ne[2]);
    Tensor flat_hidden        = hidden.view({kHidden, tokens});
    Tensor fused = workspace.alloc(DType::BF16, {TextConfig::gdn_projection_rows, tokens});
    ops::linear(flat_hidden, weights.query_key_value_z, fused, kPolicy, workspace, stream);
    maybe_dump_block("gdn_fused", fused, stream);
    Tensor qkv_flat  = qkv.view({TextConfig::convolution_dim, tokens});
    Tensor gate_flat = output_gate.view({TextConfig::value_dim, tokens});
    ops::extract_bf16_columns(fused, 0, qkv_flat, stream);
    ops::extract_bf16_columns(fused, TextConfig::convolution_dim, gate_flat, stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase phase, WorkspaceArena& workspace,
    cudaStream_t stream) {
    auto scope               = workspace.scope();
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t tokens = width * batch;
    Tensor projected = workspace.alloc(DType::BF16, {TextConfig::convolution_dim, width, batch});
    gdn_input_projection(hidden, weights, projected, output_gate, phase, workspace, stream);
    Tensor convolved = workspace.alloc(DType::BF16, {TextConfig::convolution_dim, width, batch});
    ops::causal_conv1d_silu_snapshot(projected, conv_weight, conv_states, valid_columns,
                                     initial_slot, snapshot_base_slot, convolved, stream);
    Tensor convolved_flat = convolved.view({TextConfig::convolution_dim, tokens});
    maybe_dump_block("gdn_conv", convolved_flat, stream);
    Tensor query_flat     = query.view({TextConfig::key_dim, tokens});
    Tensor key_flat       = key.view({TextConfig::key_dim, tokens});
    Tensor value_flat     = value.view({TextConfig::value_dim, tokens});
    ops::extract_bf16_columns(convolved_flat, 0, query_flat, stream);
    ops::extract_bf16_columns(convolved_flat, TextConfig::key_dim, key_flat, stream);
    ops::extract_bf16_columns(convolved_flat, 2 * TextConfig::key_dim, value_flat, stream);
}

void Variant::gdn_input_projection_record(const Tensor&, const GdnProjectionWeights&,
                                          const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                          Tensor&, qwen3_6::TextPhase, WorkspaceArena&,
                                          cudaStream_t) {
    throw std::logic_error("qwen4exp: speculative replay records are not served");
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    auto scope    = workspace.scope();
    Tensor output = workspace.alloc(DType::BF16, {kHidden, hidden.ne[1]});
    maybe_dump_block("gdn_final", hidden, stream);
    ops::linear(hidden, weight, output, kPolicy, workspace, stream);
    maybe_dump_block("gdn_out", output, stream);
    combine_into(output, residual, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor&, float,
                                          const GdnProjectionWeights& weights, Tensor& hidden,
                                          Tensor& g, Tensor& beta, WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    mix_into(residual, weights.mix, hidden, workspace, stream);
    auto scope                = workspace.scope();
    const std::int32_t tokens = hidden.ne[1];
    const std::int32_t heads  = TextConfig::gdn_value_heads;
    Tensor ab                 = workspace.alloc(DType::BF16, {2 * heads, tokens});
    ops::detail::bf16_cublaslt_gemm(weights.a_b_projection, hidden, ab, stream);
    Tensor a = rows_of(ab, 0, heads, workspace, stream);
    Tensor b = rows_of(ab, heads, heads, workspace, stream);
    ops::gdn_gating(a, b, weights.a_log, weights.dt_bias, g, beta, stream);
    maybe_dump_block("gdn_ab", ab, stream);
    maybe_dump_block("gdn_g", g, stream);
    maybe_dump_block("gdn_beta", beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                = workspace.scope();
    const std::int32_t tokens = hidden.ne[1];
    // The MoE op adds into its destination; a zeroed plane turns that into a plain store.
    Tensor output = workspace.alloc(DType::BF16, {kHidden, tokens});
    CUDA_CHECK(cudaMemsetAsync(output.data, 0, output.bytes(), stream));
    const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
        ops::kSparseMoeFlashNextGeometry, weights.op.routed_gate_up.qtype,
        weights.op.routed_down.qtype, tokens, tokens));
    WorkspaceArena leaf(storage);
    ExpertSlotCache& cache = expert_slot_cache_for_current_device();
    if (cache.enabled) {
        // Routed experts come from the device slot pool: the round hook resolves the routing
        // against the directory and gathers the misses from the host bank before the expert
        // kernels run; the kernels read the pool through the layer's slot table.
        ExpertSlotCache::Layer& layer = cache.layer(weights);
        const ops::SparseMoeWeights pooled =
            ops::expert_slot_weights(cache.pool, cache.directory, weights.layer, weights.op);
        cache.begin_round(layer, tokens);
        ops::SparseMoeRoundHook hook{&ExpertSlotCache::resolve_round, &layer};
        ops::sparse_moe(hidden, pooled, ops::SparseMoeEpilogue::AddResidual, output, leaf, stream,
                        hook);
    } else {
        ops::sparse_moe(hidden, weights.op, ops::SparseMoeEpilogue::AddResidual, output, leaf,
                        stream);
    }
    combine_into(output, residual, stream);
}

void Variant::mtp_post_mixer(const Tensor&, const MtpPostMixerWeights&, Tensor&, WorkspaceArena&,
                             cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

// --- workspace capacities ----------------------------------------------------------------------

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    // The mix hook's planes live in the same mixer scope as the projection.
    return mix_capacity(first, last) +
           plane_bytes(TextConfig::query_projection_rows, last, DType::BF16) +
           w8_capacity(TextConfig::query_projection_rows, kHidden, first, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                          qwen3_6::TextPhase,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    return plane_bytes(kHidden, last, DType::BF16) +
           w8_capacity(kHidden, TextConfig::query_size, first, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    return plane_bytes(TextConfig::gdn_projection_rows, last, DType::BF16) +
           w8_capacity(TextConfig::gdn_projection_rows, kHidden, first, last);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width) {
    const std::int32_t tokens = batch_size * max_width;
    return 2 * plane_bytes(TextConfig::convolution_dim, tokens, DType::BF16) +
           gdn_input_projection_workspace_capacity_bytes(profile, phase, batch_size * min_width,
                                                         tokens);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(WeightsProfile,
                                                                          qwen3_6::TextPhase,
                                                                          std::int32_t,
                                                                          std::int32_t,
                                                                          std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    return plane_bytes(kHidden, last, DType::BF16) +
           w8_capacity(kHidden, TextConfig::value_dim, first, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    const std::int32_t heads = TextConfig::gdn_value_heads;
    return mix_capacity(first, last) + plane_bytes(2 * heads, last, DType::BF16) +
           2 * plane_bytes(heads, last, DType::BF16);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile, qwen3_6::TextPhase,
                                                         std::int32_t first, std::int32_t last) {
    return mix_capacity(first, last) + plane_bytes(kHidden, last, DType::BF16) +
           round_up(ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeFlashNextGeometry,
                                                             QType::W8G32_F16S, QType::W8G32_F16S,
                                                             first, last));
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

} // namespace ninfer::targets::qwen4exp::detail
