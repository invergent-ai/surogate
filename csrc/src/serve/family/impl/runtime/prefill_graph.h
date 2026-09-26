#pragma once

// surogate serve — PrefillGraphFamily (PATCHES.md #27).
//
// Bucket-captured prefill chunk bodies. One CUDA graph per padded chunk
// length ("bucket": multiples of 128 up to the effective prefill chunk, plus
// the chunk itself), captured lazily on first use under a live request and
// replayed for every later chunk that rounds to the same bucket. Per-replay
// parameters travel through a pinned PrefillGraphIngress {base, valid} that
// the captured body memcpys into a device mirror (the OrdinaryDecodeIngress
// idiom), and through pinned token-id staging copied into the arena ids
// tensor whose address is replay-stable (deterministic recipe sequence after
// work_.reset()).
//
// The family owns every resource a captured body bakes that is not already
// persistent: the pinned staging, the device ingress mirror, the iota vector
// the in-graph position fill offsets, and a device rope-position buffer (the
// text workspace plan sizes rope_axes = 0, so the graph body must not take
// rope positions from the arena).
//
// Every bucket the startup precapture takes is pinned. Graphs captured later,
// under live traffic (mixed rounds, keyed by exact decode width, chunk bucket
// and band), share a byte budget: a new shape is captured only while the budget
// and the device both have room for it, after evicting graphs that have gone
// idle, and otherwise its round runs the eager body, which is numerically
// authoritative. Without the budget those graphs accumulated for the life of
// the process, one per shape ever seen, until a capture found the device full
// (#228).

#include "core/decode_graph.h"
#include "ops/linear/bf16/bf16_cublaslt.h"
#include "ops/linear/fp8/fp8_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
#include "api/types.h"
#include "core/device.h"
#include "core/device_footprint.h"
#include "core/elastic_kv_region.h"
#include "core/tensor.h"

#include "api/ops/position.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <compare>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>

namespace sinfer::family::detail {

struct PrefillGraphIngress {
    std::int32_t base        = 0; // absolute position of the chunk's first token
    std::int32_t valid       = 0; // real token count in the padded window
    std::int32_t batch_valid = 0; // mixed rounds: real decode lanes in the batch bucket
};

class PrefillGraphFamily {
public:
    using Clock = std::chrono::steady_clock;

    // Graphs captured after startup may hold this much together (SUROGATE_SERVE_PREFILL_GRAPH_BUDGET_MIB).
    // Half the headroom the KV cache leaves free; the rest is for workspace growth.
    static constexpr std::size_t kDefaultLazyBudgetBytes = 512ULL << 20;
    // A capture after startup must leave this much of the device free, counting what the
    // KV cache is entitled to but has not mapped yet.
    static constexpr std::size_t kLazyCaptureFloorBytes = 256ULL << 20;
    // A graph unused this long may be evicted for a new shape. Evicting sooner turns a
    // working set wider than the budget into a capture per round, and a capture costs
    // more than the eager round it replaces.
    static constexpr std::chrono::seconds kEvictionIdle{30};

    PrefillGraphFamily(DeviceContext& device, std::uint32_t effective_prefill_chunk,
                       std::uint32_t kv_capacity, std::int32_t scratch_state_slot,
                       std::size_t lazy_budget_bytes = kDefaultLazyBudgetBytes)
        : device_(device), prefill_chunk_(static_cast<std::int32_t>(effective_prefill_chunk)),
          kv_capacity_(kv_capacity), scratch_state_slot_(scratch_state_slot),
          lazy_budget_bytes_(lazy_budget_bytes) {
        if (prefill_chunk_ <= 0) {
            throw std::invalid_argument("prefill graph family requires a positive chunk");
        }
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ids_staging_),
                                 static_cast<std::size_t>(prefill_chunk_) * sizeof(std::int32_t),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ingress_staging_),
                                 sizeof(PrefillGraphIngress), cudaHostAllocDefault));
        *ingress_staging_ = {};

        CUDA_CHECK(cudaMalloc(&ingress_device_storage_, 3 * sizeof(std::int32_t)));
        CUDA_CHECK(
            cudaMalloc(&iota_storage_,
                       static_cast<std::size_t>(prefill_chunk_) * sizeof(std::int32_t)));
        CUDA_CHECK(
            cudaMalloc(&rope_positions_storage_,
                       static_cast<std::size_t>(prefill_chunk_) * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMemsetAsync(ingress_device_storage_, 0, 3 * sizeof(std::int32_t),
                                   device_.stream));
        Tensor iota = iota_window(prefill_chunk_);
        ops::fill_i32_positions(iota, 0, device_.stream);
    }

    ~PrefillGraphFamily() {
        buckets_.clear();
        if (rope_positions_storage_ != nullptr) { cudaFree(rope_positions_storage_); }
        if (iota_storage_ != nullptr) { cudaFree(iota_storage_); }
        if (ingress_device_storage_ != nullptr) { cudaFree(ingress_device_storage_); }
        if (ingress_staging_ != nullptr) { cudaFreeHost(ingress_staging_); }
        if (ids_staging_ != nullptr) { cudaFreeHost(ids_staging_); }
    }

    PrefillGraphFamily(const PrefillGraphFamily&)            = delete;
    PrefillGraphFamily& operator=(const PrefillGraphFamily&) = delete;

    [[nodiscard]] std::int32_t bucket_for(std::int32_t tokens) const noexcept {
        const std::int32_t rounded = ((tokens + 127) / 128) * 128;
        return rounded < prefill_chunk_ ? rounded : prefill_chunk_;
    }

    [[nodiscard]] std::uint32_t kv_capacity() const noexcept { return kv_capacity_; }
    [[nodiscard]] std::int32_t scratch_state_slot() const noexcept { return scratch_state_slot_; }
    [[nodiscard]] std::size_t graph_bytes() const noexcept { return graph_bytes_; }
    [[nodiscard]] std::size_t lazy_graph_bytes() const noexcept { return lazy_bytes_; }
    [[nodiscard]] std::uint64_t evictions() const noexcept { return evictions_; }
    [[nodiscard]] std::uint64_t refusals() const noexcept { return refusals_; }
    [[nodiscard]] std::size_t graph_count() const noexcept { return buckets_.size(); }

    // Tests: what the family reads as the device's free bytes (default
    // device_budget_free_bytes), and how long a graph must sit unused before it
    // may be evicted (default kEvictionIdle).
    void set_free_bytes_probe(std::size_t (*probe)()) noexcept { free_bytes_probe_ = probe; }
    void set_eviction_idle(Clock::duration idle) noexcept { eviction_idle_ = idle; }

    // The startup precapture is done: pin what it captured and budget everything after.
    void finish_startup() noexcept {
        for (auto& [key, entry] : buckets_) { entry.pinned = true; }
        lazy_bytes_ = 0;
        startup_    = false;
    }

    // Staging the host fills before every replay (and before capture).
    [[nodiscard]] std::int32_t* ids_staging() noexcept { return ids_staging_; }
    [[nodiscard]] PrefillGraphIngress* ingress_staging() noexcept { return ingress_staging_; }

    // Device tensors the captured body bakes.
    [[nodiscard]] Tensor ingress_device() const {
        return Tensor(ingress_device_storage_, DType::I32, {3});
    }
    [[nodiscard]] Tensor iota_window(std::int32_t tokens) const {
        return Tensor(iota_storage_, DType::I32, {tokens});
    }
    [[nodiscard]] Tensor rope_positions_window(std::int32_t tokens) const {
        return Tensor(rope_positions_storage_, DType::I32, {tokens});
    }

    // The chunk rounding the graph ladder applies, exposed so callers can map
    // KV for the window the graph actually writes (PATCHES.md #55).
    [[nodiscard]] static std::int32_t chunk_bucket_for(std::uint32_t length) noexcept {
        return static_cast<std::int32_t>(((length + 127U) / 128U) * 128U);
    }
    // The furthest token a graph chunk of this prompt can write, from any cursor.
    // A chunk starting at c writes [c, c + chunk_bucket_for(prompt - c)), and
    // c + roundup128(prompt - c) <= prompt + 127 for every c in [0, prompt]. This
    // is what a request must be entitled to, since chunk starts are not aligned:
    // prefix reuse and rewrite-checkpoint restores begin at arbitrary frontiers.
    [[nodiscard]] static std::uint32_t graph_prefill_reach(std::uint32_t prompt_tokens) noexcept {
        return prompt_tokens + 127U;
    }

    struct Key {
        bool mixed;
        std::int32_t chunk, batch, band;
        auto operator<=>(const Key&) const = default;
    };

    [[nodiscard]] static Key prefill_key(std::int32_t chunk_bucket) noexcept {
        return {false, chunk_bucket, 0, 0};
    }

    [[nodiscard]] static Key mixed_key(std::int32_t chunk_bucket,
                                        std::int32_t batch_bucket,
                                        std::int32_t band = 0) noexcept {
        // The captured chunk width and decode envelope are independent. Keep
        // their full values and the graph kind rather than overlapping bit fields.
        // `band` is the ordinary profile's index, not its shared topology class.
        return {true, chunk_bucket, batch_bucket, band};
    }
    // Round the decode batch up to the next multiple of 8, never past the
    // concurrency ceiling. The bucket must be >= the real row count: a bucket
    // below it would run a graph with fewer decode columns than the round has
    // rows and the uncovered rows would read stale egress (a silent
    // wrong-token bug, which is what a fixed 32 cap produced once the ceiling
    // rose above 32).
    [[nodiscard]] static std::int32_t batch_bucket_for(std::int32_t batch) noexcept {
        // Exact, not rounded (PATCHES.md #50). Rounding meant padding the
        // decode batch, and a pad column shares its ingress row with a live
        // lane — including that lane's GDN state slot, which the conv and
        // recurrent snapshots update read-modify-write. Two columns doing RMW
        // on one slot inside a single launch is a race; it corrupted about one
        // round in a hundred thousand and surfaced as a bad token far away.
        // Exact widths remove padding entirely. Capture is on demand, so only
        // the widths that actually occur cost a graph; if one stops fitting the
        // engine stops rather than degrading, and --enforce-eager is the way to
        // ask for a graph-free run.
        const auto ceiling = static_cast<std::int32_t>(kMaximumBatchColumns);
        return batch > ceiling ? ceiling : batch;
    }

    DecodeGraphExecutable* ensure(std::int32_t bucket, const std::function<void()>& body) {
        return ensure(prefill_key(bucket), body);
    }

    /**
     * The executable for `key`, capturing it now via `body` if this is the key's
     * first use. Returns nullptr when a graph captured after startup does not fit
     * the budget or the device: the caller runs the eager body.
     */
    DecodeGraphExecutable* ensure(Key key, const std::function<void()>& body) {
        const Clock::time_point now = Clock::now();
        auto found                  = buckets_.find(key);
        if (found != buckets_.end()) {
            found->second.last_use = now;
            return &found->second.executable;
        }
        if (!startup_ && !admit_lazy_capture(key, now)) { return nullptr; }
        if (log_enabled()) {
            std::fprintf(stderr, "prefill-graph: capturing %s chunk %d batch %d band %d\n",
                         key.mixed ? "mixed" : "prefill", key.chunk, key.batch, key.band);
        }
        try {
            // A prefill bucket is captured lazily, under a live request, on whichever thread
            // and device the round is running -- not on the one that built the program. Every
            // cuBLASLt plane creates its handle and workspace on first use, and creating them
            // allocates, which a capture forbids. Prewarming here is what makes "the first
            // request on a device captures" a legal thing to do; the program constructor's
            // prewarm only covers the device and context it was built in.
            ops::detail::bf16_cublaslt_prewarm();
            ops::detail::fp8_cublaslt_prewarm();
            ops::detail::nvfp4_cublaslt_prewarm();

            // Sampled after the prewarm, whose workspaces are not this graph's. KV the
            // elastic cache maps or unmaps meanwhile is not either.
            const DeviceFootprint footprint_before = sample_device_footprint();
            const std::size_t kv_before            = elastic_kv_mapped_bytes(device_.device);

            DecodeGraphDefinition definition;
            definition.capture(device_.stream, body);
            DecodeGraphExecutable executable;
            executable.instantiate(definition);
            executable.upload(device_.stream);
            device_.synchronize();

            // Per-process where the driver will attribute it, so a neighbouring
            // engine's allocations are not counted as this family's graphs.
            const std::size_t measured =
                device_footprint_delta(footprint_before, sample_device_footprint()).bytes;
            const std::size_t kv_after = elastic_kv_mapped_bytes(device_.device);
            const std::size_t counted  = measured + (kv_before > kv_after ? kv_before - kv_after : 0);
            const std::size_t excluded = kv_after > kv_before ? kv_after - kv_before : 0;
            const std::size_t bytes    = counted > excluded ? counted - excluded : 0;
            graph_bytes_ += bytes;
            largest_capture_bytes_ = std::max(largest_capture_bytes_, bytes);

            auto emplaced = buckets_.emplace(
                key, Entry{std::move(executable), bytes, now, startup_});
            if (!startup_) {
                lazy_bytes_ += bytes;
                // Over only when the graph came out larger than the estimate admitted.
                while (lazy_bytes_ > lazy_budget_bytes_ && evict_idle(now, &emplaced.first->first)) {}
            }
            if (log_enabled()) {
                std::fprintf(stderr,
                             "prefill-graph: captured %s chunk %d batch %d band %d: %.1f MiB "
                             "(%zu graphs, %.1f MiB, %.1f of %.1f MiB after startup)\n",
                             key.mixed ? "mixed" : "prefill", key.chunk, key.batch, key.band,
                             mib(bytes), buckets_.size(), mib(graph_bytes_), mib(lazy_bytes_),
                             mib(lazy_budget_bytes_));
            }
            return &emplaced.first->second.executable;
        } catch (const std::exception& error) {
            // Do not degrade to the eager body. A capture that runs out of
            // memory leaves the device with no margin, and the other lazy
            // allocators on this path (Marlin scratch, the derived weight
            // planes) fail the same way a moment later without saying so —
            // the observed sequence was a fallback that "worked" and then an
            // illegal access one round later. Serving eagerly is a decision
            // for the operator to make up front with --enforce-eager, so
            // report the shortfall and stop while the state is still sound.
            // The eager fallback that is safe happens before this point:
            // admit_lazy_capture refuses a capture while the margin is still
            // there, so reaching here means something outside that accounting
            // took the memory.
            std::fprintf(stderr,
                         "prefill-graph: capture of the %s graph for chunk %d batch %d band %d "
                         "failed: %s\n"
                         "Prefill graphs held %.1f MiB in %zu graphs (%.1f MiB of a %.1f MiB "
                         "budget captured after startup; the largest took %.1f MiB) with "
                         "%.1f MiB of the device free.\n"
                         "The device has no room left for CUDA Graphs at this configuration. "
                         "Re-run with --enforce-eager to serve without them, or lower "
                         "--max-num-seqs / --kv-capacity / SUROGATE_SERVE_PREFILL_GRAPH_BUDGET_MIB "
                         "to leave room.\n",
                         key.mixed ? "mixed" : "prefill", key.chunk, key.batch, key.band,
                         error.what(), mib(graph_bytes_), buckets_.size(), mib(lazy_bytes_),
                         mib(lazy_budget_bytes_), mib(largest_capture_bytes_),
                         mib(device_budget_free_bytes(device_.device)));
            std::fflush(stderr);
            std::_Exit(EXIT_FAILURE);
        }
    }

private:
    struct Entry {
        DecodeGraphExecutable executable;
        std::size_t bytes = 0;
        Clock::time_point last_use{};
        bool pinned = false;
    };

    [[nodiscard]] static bool log_enabled() noexcept {
        static const bool enabled = std::getenv("SUROGATE_SERVE_PREFILL_GRAPH_LOG") != nullptr;
        return enabled;
    }
    [[nodiscard]] static double mib(std::size_t bytes) noexcept {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    }

    // What the device can still give a graph: free memory (under --gpu-memory-limit-mib,
    // the limit's) less what the elastic KV cache is entitled to map and has not.
    [[nodiscard]] std::size_t device_available_bytes() const noexcept {
        const std::size_t free_bytes = free_bytes_probe_ != nullptr
                                           ? free_bytes_probe_()
                                           : device_budget_free_bytes(device_.device);
        const std::size_t committed  = elastic_kv_unmapped_commitment(device_.device);
        return free_bytes > committed ? free_bytes - committed : 0;
    }

    // Evicts the least recently used graph captured after startup that has been idle
    // for eviction_idle_, other than `keep`. False when there is none.
    bool evict_idle(Clock::time_point now, const Key* keep = nullptr) {
        auto victim = buckets_.end();
        for (auto it = buckets_.begin(); it != buckets_.end(); ++it) {
            const Entry& entry = it->second;
            if (entry.pinned || (keep != nullptr && it->first == *keep) ||
                now - entry.last_use < eviction_idle_) {
                continue;
            }
            if (victim == buckets_.end() || entry.last_use < victim->second.last_use) {
                victim = it;
            }
        }
        if (victim == buckets_.end()) { return false; }
        const std::size_t bytes = victim->second.bytes;
        if (log_enabled()) {
            std::fprintf(stderr, "prefill-graph: evicting %s chunk %d batch %d band %d (%.1f MiB)\n",
                         victim->first.mixed ? "mixed" : "prefill", victim->first.chunk,
                         victim->first.batch, victim->first.band, mib(bytes));
        }
        // An executable still in flight is freed when its launch completes.
        buckets_.erase(victim);
        lazy_bytes_  -= std::min(lazy_bytes_, bytes);
        graph_bytes_ -= std::min(graph_bytes_, bytes);
        ++evictions_;
        return true;
    }

    // Whether a graph for `key` may be captured now, evicting idle graphs to make room.
    // A graph is as large as the largest one captured so far, as far as anyone can tell
    // before capturing it; one that turns out larger is trimmed back after.
    bool admit_lazy_capture(const Key& key, Clock::time_point now) {
        const std::size_t estimate = largest_capture_bytes_;
        bool evicted               = false;
        while (lazy_bytes_ + estimate > lazy_budget_bytes_ && evict_idle(now)) { evicted = true; }
        bool fits = lazy_budget_bytes_ > 0 && lazy_bytes_ + estimate <= lazy_budget_bytes_;
        if (fits) {
            // Evicted executables return their memory once the stream has finished with them.
            if (evicted) { device_.synchronize(); }
            while (device_available_bytes() < estimate + kLazyCaptureFloorBytes) {
                if (!evict_idle(now)) {
                    fits = false;
                    break;
                }
                device_.synchronize();
            }
        }
        if (fits) { return true; }
        ++refusals_;
        if (log_enabled() || !refusal_reported_) {
            refusal_reported_ = true;
            std::fprintf(stderr,
                         "prefill-graph: not capturing the %s graph for chunk %d batch %d band %d: "
                         "graphs captured after startup hold %.1f of a %.1f MiB budget, the next "
                         "needs about %.1f MiB, and %.1f MiB of the device is free. Rounds of shapes "
                         "without a graph run eagerly until older graphs go idle; "
                         "SUROGATE_SERVE_PREFILL_GRAPH_BUDGET_MIB sets the budget.\n",
                         key.mixed ? "mixed" : "prefill", key.chunk, key.batch, key.band,
                         mib(lazy_bytes_), mib(lazy_budget_bytes_), mib(estimate),
                         mib(device_available_bytes()));
            std::fflush(stderr);
        }
        return false;
    }

    DeviceContext& device_;
    std::int32_t prefill_chunk_;
    std::uint32_t kv_capacity_;
    std::int32_t scratch_state_slot_;

    std::int32_t* ids_staging_          = nullptr;
    PrefillGraphIngress* ingress_staging_ = nullptr;
    void* ingress_device_storage_       = nullptr;
    void* iota_storage_                 = nullptr;
    void* rope_positions_storage_       = nullptr;

    std::map<Key, Entry> buckets_;
    std::size_t graph_bytes_           = 0;
    std::size_t lazy_budget_bytes_     = 0;
    std::size_t lazy_bytes_            = 0;
    std::size_t largest_capture_bytes_ = 0;
    std::uint64_t evictions_           = 0;
    std::uint64_t refusals_            = 0;
    bool refusal_reported_             = false;
    bool startup_                      = true;
    Clock::duration eviction_idle_     = kEvictionIdle;
    std::size_t (*free_bytes_probe_)() = nullptr;
};

} // namespace sinfer::family::detail
