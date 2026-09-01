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
// Capture failures poison the family permanently (dead()): the eager body is
// always available and numerically authoritative.

#include "core/decode_graph.h"
#include "api/types.h"
#include "core/device.h"
#include "core/tensor.h"

#include "api/ops/position.h"

#include <cuda_runtime.h>

#include <cstdint>
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
    PrefillGraphFamily(DeviceContext& device, std::uint32_t effective_prefill_chunk,
                       std::uint32_t kv_capacity, std::int32_t scratch_state_slot)
        : device_(device), prefill_chunk_(static_cast<std::int32_t>(effective_prefill_chunk)),
          kv_capacity_(kv_capacity), scratch_state_slot_(scratch_state_slot) {
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

    /**
     * The executable for `bucket`, capturing it now via `body` if this is the
     * bucket's first use. Returns nullptr when the family is dead or capture
     * fails (the caller runs the eager body; failure poisons the family).
     */
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

    [[nodiscard]] static std::int32_t mixed_key(std::int32_t chunk_bucket,
                                                std::int32_t batch_bucket,
                                                std::int32_t band) noexcept {
        // The frontier band belongs in the key: the captured body bakes the
        // band's envelope into its decode attention grid, so a replay under a
        // different band truncates attention for every lane past the captured
        // max. `band` is the ordinary profile's index — not its topology class,
        // which is zero for every profile and once left the band out of the key.
        return (band << 20) | mixed_key(chunk_bucket, batch_bucket);
    }

    [[nodiscard]] static std::int32_t mixed_key(std::int32_t chunk_bucket,
                                                std::int32_t batch_bucket) noexcept {
        return (chunk_bucket << 8) | batch_bucket;
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
        const auto ceiling = static_cast<std::int32_t>(kMaximumConcurrency);
        return batch > ceiling ? ceiling : batch;
    }

    DecodeGraphExecutable* ensure(std::int32_t bucket, const std::function<void()>& body) {
        auto found = buckets_.find(bucket);
        if (found != buckets_.end()) { return &found->second; }
        if (std::getenv("SUROGATE_SERVE_PREFILL_GRAPH_LOG") != nullptr) {
            std::fprintf(stderr, "prefill-graph: capturing bucket %d (chunk %d)\n", bucket,
                         prefill_chunk_);
        }
        try {
            std::size_t free_before = 0;
            std::size_t total       = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_before, &total));

            DecodeGraphDefinition definition;
            definition.capture(device_.stream, body);
            DecodeGraphExecutable executable;
            executable.instantiate(definition);
            executable.upload(device_.stream);
            device_.synchronize();

            std::size_t free_after = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
            graph_bytes_ += free_before > free_after ? free_before - free_after : 0;

            auto emplaced = buckets_.emplace(bucket, std::move(executable));
            return &emplaced.first->second;
        } catch (const std::exception& error) {
            // Do not degrade to the eager body. A capture that runs out of
            // memory leaves the device with no margin, and the other lazy
            // allocators on this path (Marlin scratch, the derived weight
            // planes) fail the same way a moment later without saying so —
            // the observed sequence was a fallback that "worked" and then an
            // illegal access one round later. Serving eagerly is a decision
            // for the operator to make up front with --enforce-eager, so
            // report the shortfall and stop while the state is still sound.
            std::fprintf(stderr,
                         "prefill-graph: capture for bucket %d failed: %s\n"
                         "The device has no room left for CUDA Graphs at this configuration. "
                         "Re-run with --enforce-eager to serve without them, or lower "
                         "--max-num-seqs / --kv-capacity to leave room.\n",
                         bucket, error.what());
            std::fflush(stderr);
            std::_Exit(EXIT_FAILURE);
        }
    }

private:
    DeviceContext& device_;
    std::int32_t prefill_chunk_;
    std::uint32_t kv_capacity_;
    std::int32_t scratch_state_slot_;

    std::int32_t* ids_staging_          = nullptr;
    PrefillGraphIngress* ingress_staging_ = nullptr;
    void* ingress_device_storage_       = nullptr;
    void* iota_storage_                 = nullptr;
    void* rope_positions_storage_       = nullptr;

    std::map<std::int32_t, DecodeGraphExecutable> buckets_;
    std::size_t graph_bytes_ = 0;
};

} // namespace sinfer::family::detail
