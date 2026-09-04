#include "targets/qwen4exp/impl/variant.h"
#include "core/numa.h"

#include "family/impl/lora_hook.h"
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
#include <thread>
#include <cctype>
#include <array>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::qwen4exp::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS qwen4exp_runtime
#include "family/impl/runtime/instantiate.h"

namespace sinfer::targets::qwen4exp::detail {
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
    int device                = -1; // the device whose buffers this partial points at
};
thread_local PendingPartial t_partial;
// The inject gates and the partial are handed from the mixer to the combine through this
// thread, while the buffers they name belong to a device. One thread drives every stage of a
// pipeline, so a hand-off that crosses a stage boundary would add one device's expert output
// into another's residual. SUROGATE_SERVE_CPU_MOE_VERIFY=1 checks that it never does.
thread_local int t_inject_device = -1;

[[nodiscard]] inline bool cpu_moe_verify() {
    static const bool on = std::getenv("SUROGATE_SERVE_CPU_MOE_VERIFY") != nullptr;
    return on;
}

inline void check_device_handoff(const char* what, int produced_on) {
    if (!cpu_moe_verify() || produced_on < 0) { return; }
    int current = -1;
    cudaGetDevice(&current);
    if (current != produced_on) {
        std::fprintf(stderr,
                     "qwen4exp: %s was produced on device %d and is being consumed on device "
                     "%d\n",
                     what, produced_on, current);
    }
}

struct InjectScratch {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

// Expert slot cache (phase 2): one pool per device, enabled by SUROGATE_SERVE_EXPERT_SLOTS
// (slot count; 0 or unset keeps the zero-copy path). The pool/directory/miss list are device
// memory owned here; the per-layer host banks come from the layer's W8 host Weights.
// How long the GPU actually waits for the host round (SUROGATE_SERVE_CPU_MOE_JOIN_PROBE=1).
//
// The CPU split forks the host experts onto a side stream and joins them here, and the design
// assumes the join is free -- that the host round finishes inside the GPU's own expert work.
// Whether that holds decides where single-user time goes: if the GPU waits, the round is paced
// by the host and the lever is host round latency, not the PCIe gather. Nothing measured it.
//
// A pair of events straddles the wait, and the pair is read one round later, when it has long
// since completed -- so the probe never adds a synchronise of its own. Two pairs alternate so
// a round can be in flight while the previous one is read. Eager only: under graph capture an
// event record is a node and elapsed time between nodes is not a quantity you can ask for.
struct JoinProbe {
    // A ring rather than a pair. With two buffers a pair whose events were not yet complete
    // got overwritten on its next turn, so only the joins that finished fastest were ever
    // measured -- a probe for stalls that silently dropped the stalls. The ring is deep enough
    // that the oldest entry has long completed, and it is drained with a blocking wait rather
    // than a query, so every join is counted.
    static constexpr int kRing = 16;
    bool enabled     = false;
    bool initialised = false;
    int slot         = 0;
    bool pending[kRing]{};
    cudaEvent_t before[kRing]{};
    cudaEvent_t after[kRing]{};
    double waited_ms = 0.0;
    float worst_ms     = 0.0F;
    bool open          = false;
    std::int64_t joins = 0;
    std::int64_t combines = 0;
    std::int64_t reported = 0;

    void ensure() {
        if (initialised) { return; }
        initialised = true;
        for (int i = 0; i < kRing; ++i) {
            CUDA_CHECK(cudaEventCreate(&before[i]));
            CUDA_CHECK(cudaEventCreate(&after[i]));
        }
    }

    /// Reporting for the gather instance: same cadence, its own wording.
    void tick_gather() {
        if (!enabled) { return; }
        ++combines;
        if (combines < reported + 512) { return; }
        reported = combines;
        if (joins == 0) {
            std::fprintf(stderr, "qwen4exp: miss-gather never ran in %lld layers\n",
                         static_cast<long long>(combines));
            return;
        }
        std::fprintf(stderr,
                     "qwen4exp: miss-gather occupied %.3f ms over %lld layers "
                     "(mean %.3f ms/layer, worst %.3f)\n",
                     waited_ms, static_cast<long long>(joins),
                     waited_ms / static_cast<double>(joins), static_cast<double>(worst_ms));
    }

    /// Called on every combine, join or not: the report lives here rather than in `wrap`
    /// because a run where the split never engages takes no joins at all, and that is the
    /// case the probe most needs to be able to say out loud.
    void tick() {
        if (!enabled) { return; }
        ++combines;
        if (combines < reported + 512) { return; }
        reported = combines;
        if (joins > 0) {
            std::fprintf(stderr,
                         "qwen4exp: host-round join waited %.3f ms over %lld joins of %lld "
                         "combines (mean %.3f ms/join, worst %.3f)\n",
                         waited_ms, static_cast<long long>(joins),
                         static_cast<long long>(combines),
                         waited_ms / static_cast<double>(joins),
                         static_cast<double>(worst_ms));
        } else {
            std::fprintf(stderr,
                         "qwen4exp: host-round join never taken in %lld combines -- the CPU "
                         "split did not engage (rounds below --cpu-moe-min-tokens carry no host "
                         "partial, so every miss of such a round crosses PCIe)\n",
                         static_cast<long long>(combines));
        }
    }

    /// Opens a timed span on `stream`, reclaiming the ring slot it reuses.
    void begin(cudaStream_t stream) {
        if (!enabled) { return; }
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            return;
        }
        ensure();
        if (pending[slot]) {
            CUDA_CHECK(cudaEventSynchronize(after[slot]));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, before[slot], after[slot]));
            waited_ms += static_cast<double>(ms);
            if (ms > worst_ms) { worst_ms = ms; }
            ++joins;
            pending[slot] = false;
        }
        CUDA_CHECK(cudaEventRecord(before[slot], stream));
        open = true;
    }

    /// Closes the span opened by `begin`.
    void end(cudaStream_t stream) {
        if (!enabled || !open) { return; }
        open = false;
        CUDA_CHECK(cudaEventRecord(after[slot], stream));
        pending[slot] = true;
        slot          = (slot + 1) % kRing;
    }

    /// Drains whichever pair has completed, then straddles this join with the other.
    void wrap(cudaStream_t stream, cudaEvent_t join) {
        if (!enabled) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, join, 0));
            return;
        }
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, join, 0));
            return;
        }
        ensure();
        // Reclaim the slot we are about to reuse: it is kRing joins old, so the wait is
        // nominally free, and blocking rather than querying keeps the sample unbiased.
        if (pending[slot]) {
            CUDA_CHECK(cudaEventSynchronize(after[slot]));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, before[slot], after[slot]));
            waited_ms += static_cast<double>(ms);
            if (ms > worst_ms) { worst_ms = ms; }
            ++joins;
            pending[slot] = false;
        }
        CUDA_CHECK(cudaEventRecord(before[slot], stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, join, 0));
        CUDA_CHECK(cudaEventRecord(after[slot], stream));
        pending[slot] = true;
        slot          = (slot + 1) % kRing;
    }
};

/// How long the PCIe miss-gather occupies the round. The bytes-over-bandwidth estimate says
/// it is a large share of single-user decode; this measures it, because whether overlapping
/// the gather with the hit-path compute is worth the surgery depends on the real number.
JoinProbe& gather_probe() {
    static JoinProbe probe = [] {
        JoinProbe p;
        const char* on = std::getenv("SUROGATE_SERVE_GATHER_PROBE");
        p.enabled      = on != nullptr && *on != '\0' && *on != '0';
        return p;
    }();
    return probe;
}

JoinProbe& join_probe() {
    static JoinProbe probe = [] {
        JoinProbe p;
        const char* on = std::getenv("SUROGATE_SERVE_CPU_MOE_JOIN_PROBE");
        p.enabled      = on != nullptr && *on != '\0' && *on != '0';
        return p;
    }();
    return probe;
}


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
        // Issue order of this slice. Not a race detector: the host runs far ahead of the
        // stream, so a large gap between this and `staging_generation` is normal — the
        // overwriting copy is only *enqueued*, and the main stream cannot reach it until the
        // combine's wait on `join_event` has retired this callback.
        std::uint64_t staged_generation = 0;
    };
    std::uint64_t staging_generation = 0;
    std::deque<SliceContext> slice_contexts;
    SliceContext& slice_context(Layer& entry, std::int32_t offset, std::int32_t tokens, std::int32_t ordinal) {
        for (SliceContext& c : slice_contexts) {
            if (c.entry == &entry && c.offset == offset && c.tokens == tokens && c.ordinal == ordinal) { return c; }
        }
        slice_contexts.push_back(SliceContext{&entry, offset, tokens, ordinal, &mirrors[static_cast<std::size_t>(ordinal)]});
        return slice_contexts.back();
    }
    bool enabled = false;
    std::int32_t slots     = 0;
    std::int32_t scan_ring = 0; // trailing slots reserved for prefill scans (0 = plain LRU)
    void* pool_memory      = nullptr;
    void* directory_memory = nullptr;
    void* miss_memory      = nullptr;
    void* stats_memory     = nullptr;
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
            const auto geometry = ops::kSparseMoeFlashNextGeometry;
            if (weights.host_bank_q4) {
                entry.bank = ops::expert_host_bank_q4(geometry, weights.op.routed_gate_up.qdata,
                                                      weights.op.routed_down.qdata);
                if (weights.host_gate_up != nullptr && weights.host_down != nullptr) {
                    const ops::Q4BankPlanes gate = ops::q4_bank_planes(
                        static_cast<std::int64_t>(geometry.experts) * geometry.expert_rows(),
                        geometry.hidden);
                    const ops::Q4BankPlanes down = ops::q4_bank_planes(
                        static_cast<std::int64_t>(geometry.experts) * geometry.hidden,
                        geometry.intermediate);
                    entry.cpu_bank.format         = ops::ExpertBankFormat::Q4G32AM;
                    entry.cpu_bank.gate_up_codes  = weights.host_gate_up;
                    entry.cpu_bank.gate_up_scales = weights.host_gate_up + gate.scales_offset;
                    entry.cpu_bank.gate_up_mins   = weights.host_gate_up + gate.mins_offset;
                    entry.cpu_bank.down_codes     = weights.host_down;
                    entry.cpu_bank.down_scales    = weights.host_down + down.scales_offset;
                    entry.cpu_bank.down_mins      = weights.host_down + down.mins_offset;
                }
                return entry;
            }
            entry.bank  = ops::expert_host_bank(geometry, weights.op.routed_gate_up,
                                                weights.op.routed_down);
            if (entry.bank.format == ops::ExpertBankFormat::GgmlBlocks) {
                // The blocks are the bank: the CPU path decodes a row at a time with the same
                // codec the gather uses, so the host reads exactly the bytes the GGUF holds and
                // the artifact still stores no second copy of the experts.
                entry.cpu_bank.format       = ops::ExpertBankFormat::GgmlBlocks;
                entry.cpu_bank.gate_up_ggml = entry.bank.gate_up_ggml;
                entry.cpu_bank.down_ggml    = entry.bank.down_ggml;
                entry.cpu_bank.gate_up_codes = weights.host_gate_up != nullptr
                                                   ? weights.host_gate_up
                                                   : entry.bank.gate_up_codes;
                entry.cpu_bank.down_codes = weights.host_down != nullptr
                                                ? weights.host_down
                                                : entry.bank.down_codes;
                return entry;
            }
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
    bool prefill_share_default  = false; // prefill share not given: follows the measured decode share
    std::shared_ptr<ops::CpuExpertPool> cpu_pool; // one per process: pipeline stages share the host cores
    cudaStream_t cpu_stream = nullptr; // side stream: the host round overlaps the GPU experts
    cudaStream_t fake_stream = nullptr; // SUROGATE_SERVE_CPU_MOE_FAKE_WAIT: timing-only fork/join
    cudaEvent_t fake_fork = nullptr, fake_join = nullptr;
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

    static void fake_wait_round(void* context) {
        const int ms = *static_cast<const int*>(context);
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
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
        if (cache.stagecheck) { cache.stagecheck_round(*entry, slice, column0, tokens); }
        std::fill_n(cache.out_host + column0, static_cast<std::size_t>(hidden) * tokens, 0.0F);
        if (count <= 0) { return; }
        cache.job_scratch.resize(static_cast<std::size_t>(count));
        // SUROGATE_SERVE_CPU_MOE_VERIFY=1: the mirror this callback reads must describe *this*
        // slice. A round of T columns routes at most T * experts_per_token paths, and every job
        // names a column of the slice and an expert of the layer — so a mirror overwritten by a
        // later slice, or read before its copy landed, shows up here instead of as a wrong token
        // several layers later. Off by default: it is a per-job check on the host's critical path.
        static const bool verify = std::getenv("SUROGATE_SERVE_CPU_MOE_VERIFY") != nullptr;
        if (verify) {
            const auto geometry = ops::kSparseMoeFlashNextGeometry;
            const long long ceiling =
                static_cast<long long>(tokens) * geometry.experts_per_token;
            long long bad = count > ceiling ? -1 : 0;
            for (long long i = 0; bad == 0 && i < count; ++i) {
                if (mirror.tokens[i] < 0 || mirror.tokens[i] >= tokens ||
                    mirror.experts[i] < 0 || mirror.experts[i] >= geometry.experts) {
                    bad = i + 1;
                }
            }
            if (bad != 0) {
                std::fprintf(stderr,
                             "qwen4exp: CPU split job list is inconsistent (layer %d, slice "
                             "offset %d, tokens %d, ordinal %d, count %lld, ceiling %lld, %s)\n",
                             entry->index, slice->offset, tokens, slice->ordinal, count, ceiling,
                             bad < 0 ? "count over ceiling" : "job out of range");
            }
        }
        for (long long i = 0; i < count; ++i) {
            cache.job_scratch[static_cast<std::size_t>(i)] = {mirror.tokens[i], mirror.experts[i], mirror.weights[i]};
        }
        ops::CpuExpertRound round{cache.x_host + column0, cache.out_host + column0, tokens, cache.job_scratch};
        cache.cpu_pool->run(entry->cpu_bank, round);
        // SUROGATE_SERVE_CPU_MOE_SELFCHECK=1: run the pool a second time and compare (a race
        // in the pool shows as a run-to-run difference far above accumulation-order noise),
        // then recompute two of the round's tokens on this thread with the reference job
        // path and compare those columns (a deterministic error in the pool shows there).
        // Off by default: it doubles the host round.
        static const bool selfcheck = std::getenv("SUROGATE_SERVE_CPU_MOE_SELFCHECK") != nullptr;
        if (selfcheck) { cache.selfcheck_round(*entry, slice, round); }
    }

    std::int64_t selfcheck_rounds     = 0;
    std::int64_t selfcheck_violations = 0;
    // SUROGATE_SERVE_CPU_MOE_STAGECHECK=1: a second copy of every slice's activations is taken on
    // the side stream at the fork (x_check) and compared with the main-stream staging in the
    // host round; after each combine the device's view of the host partial is copied back
    // (out_check) and compared with what the host wrote, in the next host round.
    const bool stagecheck = std::getenv("SUROGATE_SERVE_CPU_MOE_STAGECHECK") != nullptr;
    std::uint16_t* x_check    = nullptr;
    float* out_check          = nullptr;
    std::int32_t out_check_tokens = -1; // tokens the last combine covered; -1 = nothing pending
    std::int64_t stagecheck_rounds = 0, stagecheck_x_violations = 0, stagecheck_out_violations = 0;
    void stagecheck_round(Layer& entry, const SliceContext* slice, std::size_t column0, std::int32_t tokens) {
        const std::int32_t hidden = ops::kSparseMoeFlashNextGeometry.hidden;
        ++stagecheck_rounds;
        // The previous combine's device-side view of the partial vs what the host wrote.
        if (out_check_tokens >= 0 && slice->offset == 0) {
            const std::size_t n = static_cast<std::size_t>(hidden) * out_check_tokens;
            std::size_t bad = 0, first = n;
            for (std::size_t i = 0; i < n; ++i) {
                if (out_check[i] != out_host[i]) { if (first == n) { first = i; } ++bad; }
            }
            if (bad != 0) {
                ++stagecheck_out_violations;
                std::fprintf(stderr,
                             "qwen4exp: stage check: the device read a host partial that differs from what "
                             "the host wrote (%zu of %zu values, first at token %zu row %zu: device %.4g host "
                             "%.4g; layer %d, %d tokens; violation %lld of %lld rounds)\n",
                             bad, n, first / static_cast<std::size_t>(hidden), first % static_cast<std::size_t>(hidden),
                             static_cast<double>(out_check[first]), static_cast<double>(out_host[first]),
                             entry.index, out_check_tokens, static_cast<long long>(stagecheck_out_violations),
                             static_cast<long long>(stagecheck_rounds));
            }
            out_check_tokens = -1;
        }
        // The side-stream copy of the activations vs the main-stream staging.
        const std::size_t n = static_cast<std::size_t>(hidden) * tokens;
        std::size_t bad = 0, first = n;
        for (std::size_t i = 0; i < n; ++i) {
            if (x_check[column0 + i] != x_host[column0 + i]) { if (first == n) { first = i; } ++bad; }
        }
        if (bad != 0) {
            ++stagecheck_x_violations;
            std::fprintf(stderr,
                         "qwen4exp: stage check: the activations staged for the host differ between the "
                         "main-stream copy and the side-stream copy (%zu of %zu values, first at token %zu; "
                         "layer %d, slice offset %d, %d tokens; violation %lld of %lld rounds)\n",
                         bad, n, first / static_cast<std::size_t>(hidden), entry.index, slice->offset, tokens,
                         static_cast<long long>(stagecheck_x_violations), static_cast<long long>(stagecheck_rounds));
        }
        if (stagecheck_rounds % 2000 == 0) {
            std::fprintf(stderr, "qwen4exp: stage check: %lld rounds, %lld activation and %lld partial violations\n",
                         static_cast<long long>(stagecheck_rounds), static_cast<long long>(stagecheck_x_violations),
                         static_cast<long long>(stagecheck_out_violations));
        }
    }

    void selfcheck_round(Layer& entry, const SliceContext* slice, const ops::CpuExpertRound& round) {
        const auto geometry       = ops::kSparseMoeFlashNextGeometry;
        const std::int32_t hidden = geometry.hidden;
        const std::size_t elements = static_cast<std::size_t>(hidden) * round.tokens;
        ++selfcheck_rounds;
        std::vector<float> first(round.out, round.out + elements);
        std::fill_n(round.out, elements, 0.0F);
        cpu_pool->run(entry.cpu_bank, round);
        float scale = 1e-6F;
        for (const float v : first) { scale = std::max(scale, std::fabs(v)); }
        float worst_rerun = 0.0F;
        std::int32_t worst_rerun_token = -1;
        for (std::size_t i = 0; i < elements; ++i) {
            const float d = std::fabs(first[i] - round.out[i]);
            if (d > worst_rerun) {
                worst_rerun       = d;
                worst_rerun_token = static_cast<std::int32_t>(i / static_cast<std::size_t>(hidden));
            }
        }
        // Reference: every token of a narrow round on one round in 8, sixteen tokens of a wide
        // round on one round in 32 (the single-thread job path is slow, and this runs on the
        // host's critical path). A per-job error at the percent level needs this coverage.
        const bool narrow          = round.tokens <= 32;
        const bool reference_round = narrow ? selfcheck_rounds % 8 == 0 : selfcheck_rounds % 32 == 0;
        std::vector<std::byte> scratch_bytes(reference_round ? ops::cpu_expert_scratch_bytes(geometry) + 64 : 0);
        auto* scratch = reinterpret_cast<std::byte*>(
            (reinterpret_cast<std::uintptr_t>(scratch_bytes.data()) + 63) & ~std::uintptr_t{63});
        std::vector<float> reference(static_cast<std::size_t>(hidden));
        std::vector<char> token_checked(static_cast<std::size_t>(round.tokens), 0);
        float worst_ref              = 0.0F;
        std::int32_t worst_ref_token = -1;
        int checked_count            = 0;
        const int check_limit        = narrow ? round.tokens : 16;
        for (const ops::CpuExpertJob& job : round.jobs) {
            if (!reference_round || checked_count >= check_limit) { break; }
            if (token_checked[static_cast<std::size_t>(job.token)] != 0) { continue; }
            token_checked[static_cast<std::size_t>(job.token)] = 1;
            ++checked_count;
            std::fill(reference.begin(), reference.end(), 0.0F);
            for (const ops::CpuExpertJob& other : round.jobs) {
                if (other.token != job.token) { continue; }
                ops::cpu_expert_compute_job(geometry, entry.cpu_bank, other,
                                            round.x + static_cast<std::size_t>(job.token) * hidden,
                                            reference.data(), scratch);
            }
            const float* column = round.out + static_cast<std::size_t>(job.token) * hidden;
            for (std::int32_t r = 0; r < hidden; ++r) {
                const float d = std::fabs(reference[static_cast<std::size_t>(r)] - column[r]);
                if (d > worst_ref) {
                    worst_ref       = d;
                    worst_ref_token = job.token;
                }
            }
        }
        const float tolerance = 5e-3F * scale;
        if (worst_rerun > tolerance || worst_ref > tolerance) {
            ++selfcheck_violations;
            std::fprintf(stderr,
                         "qwen4exp: host expert self-check violation (layer %d, slice offset %d, "
                         "tokens %d, jobs %zu): rerun differs by %.3g at token %d, reference "
                         "differs by %.3g at token %d, scale %.3g (violation %lld of %lld rounds)\n",
                         entry.index, slice->offset, round.tokens, round.jobs.size(),
                         static_cast<double>(worst_rerun), worst_rerun_token,
                         static_cast<double>(worst_ref), worst_ref_token,
                         static_cast<double>(scale), static_cast<long long>(selfcheck_violations),
                         static_cast<long long>(selfcheck_rounds));
        } else if (selfcheck_rounds % 500 == 0) {
            std::fprintf(stderr, "qwen4exp: host expert self-check: %lld rounds, %lld violations\n",
                         static_cast<long long>(selfcheck_rounds),
                         static_cast<long long>(selfcheck_violations));
        }
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
        slice.staged_generation = ++staging_generation;
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
        if (stagecheck && x_check != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(x_check + column0, x.data,
                                       static_cast<std::size_t>(hidden) * tokens * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, cpu_stream));
        }
        CUDA_CHECK(cudaLaunchHostFunc(cpu_stream, &ExpertSlotCache::run_cpu_round, &slice));
        CUDA_CHECK(cudaEventRecord(join_event, cpu_stream));
        entry.round_offset = offset + tokens;
        int partial_device = -1;
        CUDA_CHECK(cudaGetDevice(&partial_device));
        t_partial =
            PendingPartial{static_cast<const float*>(out_device_alias), join_event, partial_device};
    }

    // SUROGATE_SERVE_CPU_MOE_SHADOW=<n>: every n-th split round is recomputed entirely on the
    // GPU (this hook resolves without a host share, so every miss is gathered) into a shadow
    // plane, and the split's GPU part plus the host partial is compared with it per token.
    std::int64_t shadow_rounds = 0, shadow_checked = 0, shadow_violations = 0;
    static void resolve_round_shadow(void* context, const Tensor& ids, const Tensor& /*alpha*/,
                                     const Tensor& /*x*/, Tensor& /*destination*/,
                                     cudaStream_t stream) {
        auto* entry            = static_cast<Layer*>(context);
        ExpertSlotCache& cache = *entry->owner;
        const bool scan        = entry->round_total > 32;
        ops::expert_slot_resolve(ids, entry->index, cache.directory, cache.misses, stream, scan);
        ops::expert_slot_gather(entry->bank, cache.misses, cache.pool, stream);
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
        // Wide rounds are prompt scans: their misses go to the directory's scan ring so the
        // decode working set stays resident (design/INFERENCE.md, scan resistance).
        const bool scan = entry->round_total > 32;
        if (split) {
            ops::expert_slot_resolve(ids, alpha, entry->index, cache.directory, cache.misses,
                                     &cache.cpu_jobs, share,
                                     ops::kSparseMoeFlashNextGeometry.experts_per_token, stream,
                                     scan);
        } else {
            ops::expert_slot_resolve(ids, entry->index, cache.directory, cache.misses, stream,
                                     scan);
        }
        gather_probe().begin(stream);
        ops::expert_slot_gather(entry->bank, cache.misses, cache.pool, stream);
        gather_probe().end(stream);
        gather_probe().tick_gather();
        if (split) { cache.cpu_round(*entry, x, destination, stream); }
        // SUROGATE_SERVE_CPU_MOE_FAKE_WAIT=<ms>: with the split off, still fork a host function
        // that sleeps for <ms> and make the combine wait for it — the host split's stream
        // timing without its data. Tells a data fault in the host path from a latent race
        // elsewhere that the split's idle gaps expose.
        static const int fake_wait_ms = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_CPU_MOE_FAKE_WAIT");
            return raw != nullptr && *raw != '\0' ? std::atoi(raw) : 0;
        }();
        if (!split && fake_wait_ms > 0) {
            if (cache.fake_stream == nullptr) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&cache.fake_stream, cudaStreamNonBlocking));
                CUDA_CHECK(cudaEventCreateWithFlags(&cache.fake_fork, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&cache.fake_join, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(cache.fake_fork, stream));
            CUDA_CHECK(cudaStreamWaitEvent(cache.fake_stream, cache.fake_fork, 0));
            CUDA_CHECK(cudaLaunchHostFunc(cache.fake_stream, &ExpertSlotCache::fake_wait_round,
                                          const_cast<int*>(&fake_wait_ms)));
            CUDA_CHECK(cudaEventRecord(cache.fake_join, cache.fake_stream));
            CUDA_CHECK(cudaStreamWaitEvent(stream, cache.fake_join, 0));
        }
        if (cache.stats_every > 0) { cache.record_stats(ids, stream); }
        if (cache.directory_verify) { cache.verify_directory(ids, entry->index, stream); }
    }

    // SUROGATE_SERVE_SLOT_DIRECTORY_VERIFY=1: after every resolve, read the directory back and
    // check it is a bijection (slot_of_expert and expert_of_slot agree both ways) and that every
    // expert this layer routed to is either resident or stamped for the host this round. A
    // synchronous eager-mode diagnostic; a violation is printed once per layer-round.
    const bool directory_verify = std::getenv("SUROGATE_SERVE_SLOT_DIRECTORY_VERIFY") != nullptr;
    std::int64_t directory_violations = 0;
    void verify_directory(const Tensor& ids, int layer, cudaStream_t stream) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            return;
        }
        const std::int32_t experts = directory.experts;
        const std::int32_t layers  = directory.layers;
        const auto slots           = static_cast<std::int32_t>(directory.expert_of_slot.ne[0]);
        std::vector<int> table(static_cast<std::size_t>(layers) * experts);
        std::vector<int> owners(static_cast<std::size_t>(slots));
        std::vector<unsigned> cpu_stamp(static_cast<std::size_t>(experts));
        std::vector<int> routed(static_cast<std::size_t>(ids.numel()));
        unsigned round = 0;
        CUDA_CHECK(cudaMemcpyAsync(table.data(), directory.slot_of_expert.data,
                                   table.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(owners.data(), directory.expert_of_slot.data,
                                   owners.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(cpu_stamp.data(), directory.cpu_round.data,
                                   cpu_stamp.size() * sizeof(unsigned), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(routed.data(), ids.data, routed.size() * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&round, directory.round.data, sizeof(round),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::int64_t bad_forward = 0, bad_backward = 0, unserved = 0;
        for (std::size_t f = 0; f < table.size(); ++f) {
            const int slot = table[f];
            if (slot < 0) { continue; }
            if (slot >= slots || owners[static_cast<std::size_t>(slot)] != static_cast<int>(f)) {
                ++bad_forward;
            }
        }
        for (std::int32_t s = 0; s < slots; ++s) {
            const int flat = owners[static_cast<std::size_t>(s)];
            if (flat < 0) { continue; }
            if (flat >= static_cast<int>(table.size()) || table[static_cast<std::size_t>(flat)] != s) {
                ++bad_backward;
            }
        }
        for (const int expert : routed) {
            if (expert < 0 || expert >= experts) { continue; }
            const std::size_t flat = static_cast<std::size_t>(layer) * experts + expert;
            if (table[flat] < 0 && cpu_stamp[static_cast<std::size_t>(expert)] != round) {
                ++unserved;
            }
        }
        if (bad_forward != 0 || bad_backward != 0 || unserved != 0) {
            ++directory_violations;
            std::fprintf(stderr,
                         "qwen4exp: slot directory violation at layer %d round %u: %lld forward, "
                         "%lld backward, %lld routed experts neither resident nor on the host "
                         "(violation %lld)\n",
                         layer, round, static_cast<long long>(bad_forward),
                         static_cast<long long>(bad_backward), static_cast<long long>(unserved),
                         static_cast<long long>(directory_violations));
        }
    }

    // Hit-rate readout (SUROGATE_SERVE_EXPERT_STATS=<rounds>): the counters are accumulated
    // on the device by the resolve kernel and read back exactly. A "round" is one layer's
    // resolve.
    //
    // What this replaced got three things wrong, and all three flattered the cache. It divided
    // distinct experts allocated a slot by routed paths *including duplicates* -- different
    // populations, and the wider the round the more the duplicates diluted it. With a CPU split
    // it counted PCIe gathers rather than misses, because a miss handed to the host never
    // reaches the miss list. And it extrapolated: one sampled round stood for the whole window.
    //
    // The counters are the kernel's own, so a captured decode keeps counting through replays.
    // The readout still runs on the host, so it prints on eager rounds; under CUDA graphs the
    // totals accumulate and appear at the next uncaptured resolve. Nothing is lost, only
    // deferred -- `--enforce-eager` reports continuously.
    std::int64_t stats_every  = 0;
    std::int64_t stats_rounds = 0;
    std::array<long long, ops::kExpertSlotStatCount> stats_last{};
    void record_stats(const Tensor&, cudaStream_t stream) {
        if (stats_every <= 0 || directory.stats.data == nullptr) { return; }
        // A synchronous readout is illegal on a capturing stream; the device counters keep
        // accumulating regardless and the next eager round prints the full total.
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            return;
        }
        if (++stats_rounds % stats_every != 0) { return; }
        std::array<long long, ops::kExpertSlotStatCount> counters{};
        CUDA_CHECK(cudaMemcpyAsync(counters.data(), directory.stats.data, sizeof(counters),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Windowed as well as cumulative: a prompt's wide rounds and a single-user decode's
        // one-token rounds have completely different miss shapes, and a cumulative-only figure
        // buries the decode steady state under the prefill that preceded it.
        const auto report = [](const char* label, const std::array<long long, ops::kExpertSlotStatCount>& c) {
            const double rounds    = static_cast<double>(c[ops::kExpertSlotStatRounds]);
            const double lookups   = static_cast<double>(c[ops::kExpertSlotStatLookups]);
            const double distinct  = static_cast<double>(c[ops::kExpertSlotStatDistinct]);
            const double resident  = static_cast<double>(c[ops::kExpertSlotStatResident]);
            const double gathered  = static_cast<double>(c[ops::kExpertSlotStatGathered]);
            const double host_side = static_cast<double>(c[ops::kExpertSlotStatHostRouted]);
            if (rounds <= 0.0 || distinct <= 0.0) { return; }
            // Every rate is over distinct experts asked for, the population the cache answers.
            // `PCIe` is what crossed the bus; `host` is what the CPU split absorbed; together
            // they are the misses.
            std::fprintf(stderr,
                         "qwen4exp: expert cache %s rounds=%lld distinct/round=%.1f (of %.1f "
                         "paths) hit %.1f%% miss %.1f%% = PCIe %.1f%% + host %.1f%%\n",
                         label, static_cast<long long>(c[ops::kExpertSlotStatRounds]),
                         distinct / rounds, lookups / rounds, 100.0 * resident / distinct,
                         100.0 * (gathered + host_side) / distinct, 100.0 * gathered / distinct,
                         100.0 * host_side / distinct);
        };
        std::array<long long, ops::kExpertSlotStatCount> window{};
        for (int i = 0; i < ops::kExpertSlotStatCount; ++i) {
            window[i] = counters[i] - stats_last[i];
        }
        report("window", window);
        report("total ", counters);
        stats_last = counters;
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
bool& configured_cpu_pool_per_socket() {
    static bool per_socket = false;
    return per_socket;
}

// The NUMA node of a CUDA device (its PCI function's `numa_node`), -1 when unknown.
int device_numa_node(int device) {
    char bus_id[32] = {};
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) != cudaSuccess) { return -1; }
    std::string path = "/sys/bus/pci/devices/";
    for (char* c = bus_id; *c != '\0'; ++c) { *c = static_cast<char>(std::tolower(static_cast<unsigned char>(*c))); }
    path += bus_id;
    path += "/numa_node";
    FILE* file = std::fopen(path.c_str(), "r");
    if (file == nullptr) { return -1; }
    int node = -1;
    if (std::fscanf(file, "%d", &node) != 1) { node = -1; }
    std::fclose(file);
    return node;
}

// The physical cores of a NUMA node: the first half of its cpulist on SMT-2 parts (the
// usual numbering lists the siblings after all physical cores).
std::vector<int> node_physical_cpus(int node) {
    std::vector<int> cpus;
    const std::string path = "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";
    FILE* file             = std::fopen(path.c_str(), "r");
    if (file == nullptr) { return cpus; }
    char buffer[512] = {};
    if (std::fgets(buffer, sizeof(buffer), file) != nullptr) {
        const char* p = buffer;
        while (*p != '\0' && *p != '\n') {
            char* end   = nullptr;
            const long a = std::strtol(p, &end, 10);
            long b       = a;
            p            = end;
            if (*p == '-') { b = std::strtol(p + 1, &end, 10); p = end; }
            for (long c = a; c <= b; ++c) { cpus.push_back(static_cast<int>(c)); }
            if (*p == ',') { ++p; }
        }
    }
    std::fclose(file);
    const unsigned threads_per_core = std::thread::hardware_concurrency() / 2 > 0 ? 2 : 1;
    if (threads_per_core == 2 && cpus.size() >= 2) { cpus.resize(cpus.size() / 2); }
    return cpus;
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
    const auto geometry = ops::kSparseMoeFlashNextGeometry;
    if (requested <= 0) {
        // No slot count asked for. One layer's experts is the floor the resolve needs, and it
        // is also what the pool used to settle on -- which left most of a card idle and cost
        // more than half the decode rate, because every other expert then crossed PCIe on the
        // token that wanted it. Take what is free instead, keeping a margin for the KV cache
        // and the round's workspaces, and stop at the whole model.
        // Half of what is free, not all of it: this runs before the KV cache and the round's
        // workspaces are allocated, so the reading is an overstatement of what the pool may
        // take. Half leaves those their room and still lands near the slot count the
        // single-card measurements use.
        const std::size_t per_slot = ops::expert_slot_pool_bytes(geometry, 1);
        const std::size_t free     = device_free_bytes(device);
        const std::size_t budget   = free / 2;
        const auto whole_model =
            static_cast<long>(TextConfig::layers) * static_cast<long>(geometry.experts);
        requested = per_slot == 0 ? 0 : static_cast<long>(budget / per_slot);
        requested = std::min(requested, whole_model);
        if (requested < geometry.experts) { return cache; }
    }
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
    // Scan resistance: reserve one expert-set of trailing slots for prefill scans when the
    // pool cannot hold every expert of its layers but is comfortably bigger than one scan.
    // Below that, plain LRU (a ring would eat half a tiny pool for no stable set to protect).
    const auto stage_experts =
        static_cast<std::int64_t>(TextConfig::layers) * geometry.experts; // whole-model bound
    cache.scan_ring = (static_cast<std::int64_t>(cache.slots) < stage_experts &&
                       cache.slots >= geometry.experts * 3 / 2 &&
                       std::getenv("SUROGATE_SERVE_NO_SCAN_RING") == nullptr)
                          ? geometry.experts
                          : 0;
    cache.directory = ops::create_expert_slot_directory(TextConfig::layers, geometry.experts,
                                                        cache.slots, cache.scan_ring,
                                                        cache.directory_memory,
                                                        nullptr);
    cache.misses    = ops::create_expert_miss_list(geometry.experts, cache.miss_memory);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    cache.layers.resize(static_cast<std::size_t>(TextConfig::layers));
    cache.enabled = true;
    if (const char* stats = std::getenv("SUROGATE_SERVE_EXPERT_STATS"); stats != nullptr && *stats != '\0') {
        cache.stats_every = std::strtol(stats, nullptr, 10);
        if (cache.stats_every > 0) {
            // The counters live on the device and the resolve kernel increments them, so a
            // captured decode keeps counting through every replay; the host only ever reads.
            const std::size_t bytes = sizeof(long long) * ops::kExpertSlotStatCount;
            CUDA_CHECK(cudaMalloc(&cache.stats_memory, bytes));
            CUDA_CHECK(cudaMemset(cache.stats_memory, 0, bytes));
            cache.directory.stats =
                Tensor(cache.stats_memory, DType::I64, {ops::kExpertSlotStatCount});
        }
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
    // Prefill share: -1 (unset) → 0. The prefill split is a measured loss at every chunk width
    // on this host (2026-08-30, 28k prompt, x16 card): pure GPU reads 1,632 / 2,428 / 3,275 /
    // 3,944 prompt tok/s at chunks 1,024 / 2,048 / 4,096 / 8,192, while the auto split (45 %)
    // read 484 at 1,024 and forcing 0.5-1.0 read 151-168 at any width — the host GEMM runs at
    // ~16 % of VNNI peak and every layer's combine waits on its host tail, so the split caps the
    // round at the CPU's pace. The decode split is a separate decision and keeps its own share:
    // at one token per lane the gather is misses-only and the host genuinely relieves it. An
    // explicit --cpu-moe-prefill-share still turns the prefill split on for re-measurement.
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
    const bool prefill_default = prefill_fraction < 0.0;
    if (prefill_default) { prefill_fraction = 0.0; }
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
            // The staging buffers are what this device DMAs through every round, so they go on
            // its own node rather than across both (core/numa.h). The expert bank, which every
            // core reads, is interleaved instead.
            {
                int placement_device = 0;
                CUDA_CHECK(cudaGetDevice(&placement_device));
                const ScopedMemoryPolicy placement = ScopedMemoryPolicy::for_device(placement_device);
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&cache.x_host),
                                         static_cast<std::size_t>(hidden) * stage_tokens * sizeof(std::uint16_t),
                                         cudaHostAllocPortable));
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&cache.out_host),
                                         static_cast<std::size_t>(hidden) * stage_tokens * sizeof(float),
                                         cudaHostAllocMapped | cudaHostAllocPortable));
            }
            CUDA_CHECK(cudaHostGetDevicePointer(&cache.out_device_alias, cache.out_host, 0));
            if (cache.stagecheck) {
                // Allocated here, not lazily: the first host round may run under graph capture.
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&cache.x_check),
                                         static_cast<std::size_t>(hidden) * stage_tokens * sizeof(std::uint16_t),
                                         cudaHostAllocPortable));
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&cache.out_check),
                                         static_cast<std::size_t>(hidden) * stage_tokens * sizeof(float),
                                         cudaHostAllocPortable));
            }
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
            {
                // One pool per process, or — for pipeline stages — one per NUMA node pinned to
                // that node's physical cores, so stages on different sockets run their host
                // rounds concurrently on their own cores and memory.
                static std::mutex pool_mutex;
                static std::unordered_map<int, std::weak_ptr<ops::CpuExpertPool>> pools; // node → pool (-1 = shared)
                int node = -1;
                if (configured_cpu_pool_per_socket()) {
                    node = device_numa_node(device);
                    if (node >= 0) {
                        pool_options.cpus = node_physical_cpus(node);
                        if (pool_options.cpus.empty()) { node = -1; }
                    }
                }
                std::lock_guard<std::mutex> lock(pool_mutex);
                cache.cpu_pool = pools[node].lock();
                if (cache.cpu_pool == nullptr) {
                    cache.cpu_pool = std::make_shared<ops::CpuExpertPool>(geometry, pool_options);
                    pools[node]    = cache.cpu_pool;
                    if (node >= 0) {
                        std::fprintf(stderr, "qwen4exp: host expert pool for NUMA node %d: %zu threads\n", node,
                                     pool_options.cpus.size());
                    }
                }
            }
            CUDA_CHECK(cudaStreamCreateWithFlags(&cache.cpu_stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&cache.fork_event, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&cache.join_event, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&cache.copied_event, cudaEventDisableTiming));
            cache.auto_share            = auto_share;
            cache.prefill_share_default = prefill_default;
            std::fprintf(stderr, "qwen4exp: CPU expert split enabled: %.0f%% of decode misses, %.0f%% of prefill misses (rounds up to %d columns) on %u host threads%s\n",
                         100.0 * fraction, 100.0 * prefill_fraction, cache.cpu_prefill_max_tokens,
                         cache.cpu_pool->threads(), auto_share ? " (auto: measured at startup)" : "");
        }
    }
    std::fprintf(stderr,
                 "qwen4exp: expert slot cache enabled: %d slots (%.1f GiB pool), scan ring %d\n",
                 cache.slots, static_cast<double>(pool_bytes) / (1024.0 * 1024.0 * 1024.0),
                 cache.scan_ring);
    if (const std::string numa = numa_policy_description(); !numa.empty()) {
        std::fprintf(stderr, "qwen4exp: %s\n", numa.c_str());
    }
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
    CUDA_CHECK(cudaGetDevice(&t_inject_device));
    maybe_dump_block("mixed", hidden, stream);
    maybe_dump_block("inject", inject, stream);
}

// Scatters a block output into the residual streams with the gates the mix kept.
void combine_into(const Tensor& block_output, Tensor& residual, cudaStream_t stream) {
    if (t_inject.data == nullptr || t_inject.ne[1] != residual.ne[1]) {
        throw std::logic_error("qwen4exp: combine without a matching mix");
    }
    join_probe().tick();
    maybe_dump_block("blockout", block_output, stream);
    if (t_partial.device_alias != nullptr) {
        check_device_handoff("a host expert partial", t_partial.device);
        // Join the host round (side stream) before the combine reads its partial.
        join_probe().wrap(stream, t_partial.join);
        ops::hyper_connection_combine(block_output, t_partial.device_alias, t_inject, residual, stream);
        {
            ExpertSlotCache& cache = expert_slot_cache_for_current_device();
            if (cache.stagecheck && cache.out_check != nullptr) {
                const std::int32_t tokens = block_output.ne[1];
                CUDA_CHECK(cudaMemcpyAsync(cache.out_check, cache.out_device_alias,
                                           static_cast<std::size_t>(ops::kSparseMoeFlashNextGeometry.hidden) *
                                               tokens * sizeof(float),
                                           cudaMemcpyDefault, stream));
                cache.out_check_tokens = tokens;
            }
        }
        t_partial = PendingPartial{};
    } else {
        ops::hyper_connection_combine(block_output, t_inject, residual, stream);
    }
    maybe_dump_block("combined", residual, stream);
    g_dump_block += 1;
    check_device_handoff("the inject gates", t_inject_device);
    t_inject_device = -1;
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
    // The prefill share optimum sits below the decode one and tracks host strength (measured
    // 2026-08-28: 0.5 at 32 host threads where the decode share measures ~0.8, 0.3 at 16
    // threads): share − 0.3, clamped to [0.2, 0.7], unless given explicitly.
    if (cache.prefill_share_default && cache.cpu_prefill_share_q16 > 0) {
        const double prefill = std::min(0.7, std::max(0.2, share - 0.3));
        cache.cpu_prefill_share_q16 = static_cast<std::uint32_t>(prefill * 65536.0);
        std::fprintf(stderr, "qwen4exp: CPU split auto prefill share: %.0f%%\n", 100.0 * prefill);
    }
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
    // Same lock as the other configure_* writes: pipeline stages configure concurrently.
    std::lock_guard<std::mutex> lock(expert_slot_mutex());
    configured_cpu_prefill()[device] = {share, prefill_chunk};
}

void Variant::configure_cpu_pool_per_socket(bool per_socket) {
    std::lock_guard<std::mutex> lock(expert_slot_mutex());
    configured_cpu_pool_per_socket() = per_socket;
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
                             const family::detail::PrologueColumns& columns,
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
                                   Tensor& gate, Tensor& key, Tensor& value, family::TextPhase,
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
    family::apply_lora_qkv(weights.query_key_gate_value, hidden, query, key, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    Tensor output  = workspace.alloc(DType::BF16, {kHidden, attention.ne[1]});
    ops::linear(attention, weight, output, kPolicy, workspace, stream);
    // Before the hyper-connection combine: the delta belongs to o_proj's output,
    // and the combine is what distributes it across the residual streams.
    family::apply_lora(weight, 3, attention, output, stream);
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
                                   Tensor& qkv, Tensor& output_gate, family::TextPhase,
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
    Tensor& output_gate, family::TextPhase phase, WorkspaceArena& workspace,
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
                                          Tensor&, family::TextPhase, WorkspaceArena&,
                                          cudaStream_t) {
    throw std::logic_error("qwen4exp: speculative replay records are not served");
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    family::TextPhase, WorkspaceArena& workspace,
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
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
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
        static const int shadow_every = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_CPU_MOE_SHADOW");
            return raw != nullptr && *raw != '\0' ? std::atoi(raw) : 0;
        }();
        if (shadow_every > 0 && layer.round_split && t_partial.device_alias != nullptr) {
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            const bool capturing = cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
                                   capture != cudaStreamCaptureStatusNone;
            if (!capturing && ++cache.shadow_rounds % shadow_every == 0) {
                Tensor shadow = workspace.alloc(DType::BF16, {kHidden, tokens});
                CUDA_CHECK(cudaMemsetAsync(shadow.data, 0, shadow.bytes(), stream));
                const DeviceSpan storage2 = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
                    ops::kSparseMoeFlashNextGeometry, weights.op.routed_gate_up.qtype,
                    weights.op.routed_down.qtype, tokens, tokens));
                WorkspaceArena leaf2(storage2);
                ops::SparseMoeRoundHook shadow_hook{&ExpertSlotCache::resolve_round_shadow, &layer};
                ops::sparse_moe(hidden, pooled, ops::SparseMoeEpilogue::AddResidual, shadow, leaf2,
                                stream, shadow_hook);
                CUDA_CHECK(cudaStreamWaitEvent(stream, t_partial.join, 0));
                const std::size_t n = static_cast<std::size_t>(kHidden) * tokens;
                std::vector<std::uint16_t> part(n), full(n);
                CUDA_CHECK(cudaMemcpyAsync(part.data(), output.data, n * sizeof(std::uint16_t),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(full.data(), shadow.data, n * sizeof(std::uint16_t),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                const float* partial = cache.out_host;
                auto bf16 = [](std::uint16_t bits) {
                    std::uint32_t w = static_cast<std::uint32_t>(bits) << 16;
                    float f;
                    std::memcpy(&f, &w, sizeof(f));
                    return f;
                };
                float scale = 1e-6F, worst = 0.0F;
                std::int32_t worst_token = -1;
                std::int64_t bad_tokens = 0;
                for (std::int32_t t = 0; t < tokens; ++t) {
                    float token_worst = 0.0F;
                    for (std::int32_t d = 0; d < kHidden; ++d) {
                        const std::size_t i = static_cast<std::size_t>(t) * kHidden + d;
                        const float reference = bf16(full[i]);
                        const float value     = bf16(part[i]) + partial[i];
                        scale                 = std::max(scale, std::fabs(reference));
                        token_worst           = std::max(token_worst, std::fabs(value - reference));
                    }
                    if (token_worst > worst) { worst = token_worst; worst_token = t; }
                    if (token_worst > 0.05F) { ++bad_tokens; }
                }
                ++cache.shadow_checked;
                if (worst > 0.02F * scale) {
                    ++cache.shadow_violations;
                    // The worst token, decomposed: its GPU part, host partial, full result, and
                    // the magnitude of its activation (host activations are int8 per group).
                    float part_max = 0.0F, partial_max = 0.0F, full_max = 0.0F, x_max = 0.0F;
                    std::int32_t x_argmax = -1;
                    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(kHidden));
                    CUDA_CHECK(cudaMemcpy(x_bits.data(),
                                          static_cast<const std::uint16_t*>(hidden.data) +
                                              static_cast<std::size_t>(worst_token) * kHidden,
                                          x_bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
                    for (std::int32_t d = 0; d < kHidden; ++d) {
                        const std::size_t i = static_cast<std::size_t>(worst_token) * kHidden + d;
                        part_max    = std::max(part_max, std::fabs(bf16(part[i])));
                        partial_max = std::max(partial_max, std::fabs(partial[i]));
                        full_max    = std::max(full_max, std::fabs(bf16(full[i])));
                        const float xv = std::fabs(bf16(x_bits[static_cast<std::size_t>(d)]));
                        if (xv > x_max) { x_max = xv; x_argmax = d; }
                    }
                    std::int64_t host_jobs = 0;
                    for (const ops::CpuExpertJob& job : cache.job_scratch) {
                        if (job.token == worst_token) { ++host_jobs; }
                    }
                    std::fprintf(stderr,
                                 "qwen4exp: host split shadow mismatch (layer %d, %d tokens): worst "
                                 "|split - full| %.4g at token %d, scale %.3g, %lld tokens over 0.05 "
                                 "(violation %lld of %lld checked); token %d: max|gpu part| %.4g, "
                                 "max|host partial| %.4g, max|full| %.4g, max|x| %.4g at dim %d, %lld "
                                 "host jobs in the last slice\n",
                                 weights.layer, tokens, static_cast<double>(worst), worst_token,
                                 static_cast<double>(scale), static_cast<long long>(bad_tokens),
                                 static_cast<long long>(cache.shadow_violations),
                                 static_cast<long long>(cache.shadow_checked), worst_token,
                                 static_cast<double>(part_max), static_cast<double>(partial_max),
                                 static_cast<double>(full_max), static_cast<double>(x_max), x_argmax,
                                 static_cast<long long>(host_jobs));
                } else if (cache.shadow_checked % 200 == 0) {
                    std::fprintf(stderr, "qwen4exp: host split shadow: %lld rounds checked, %lld mismatches\n",
                                 static_cast<long long>(cache.shadow_checked),
                                 static_cast<long long>(cache.shadow_violations));
                }
            }
        }
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

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    // The mix hook's planes live in the same mixer scope as the projection.
    return mix_capacity(first, last) +
           plane_bytes(TextConfig::query_projection_rows, last, DType::BF16) +
           w8_capacity(TextConfig::query_projection_rows, kHidden, first, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile,
                                                                          family::TextPhase,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    return plane_bytes(kHidden, last, DType::BF16) +
           w8_capacity(kHidden, TextConfig::query_size, first, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    return plane_bytes(TextConfig::gdn_projection_rows, last, DType::BF16) +
           w8_capacity(TextConfig::gdn_projection_rows, kHidden, first, last);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile profile, family::TextPhase phase, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width) {
    const std::int32_t tokens = batch_size * max_width;
    return 2 * plane_bytes(TextConfig::convolution_dim, tokens, DType::BF16) +
           gdn_input_projection_workspace_capacity_bytes(geometry, profile, phase, batch_size * min_width,
                                                         tokens);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile,
                                                                          family::TextPhase,
                                                                          std::int32_t,
                                                                          std::int32_t,
                                                                          std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile,
                                                                    family::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    return plane_bytes(kHidden, last, DType::BF16) +
           w8_capacity(kHidden, TextConfig::value_dim, first, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                          std::int32_t last) {
    const std::int32_t heads = TextConfig::gdn_value_heads;
    return mix_capacity(first, last) + plane_bytes(2 * heads, last, DType::BF16) +
           2 * plane_bytes(heads, last, DType::BF16);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile, family::TextPhase,
                                                         std::int32_t first, std::int32_t last) {
    return mix_capacity(first, last) + plane_bytes(kHidden, last, DType::BF16) +
           round_up(ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeFlashNextGeometry,
                                                             QType::W8G32_F16S, QType::W8G32_F16S,
                                                             first, last));
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t, std::int32_t) {
    return 0;
}

} // namespace sinfer::targets::qwen4exp::detail
