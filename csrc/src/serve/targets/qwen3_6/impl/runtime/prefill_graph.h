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

namespace ninfer::targets::qwen3_6::detail {

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

    [[nodiscard]] bool dead() const noexcept { return dead_; }
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
        // the widths that actually occur cost a graph, and the family's budget
        // guard falls back to eager if they stop fitting.
        const auto ceiling = static_cast<std::int32_t>(kMaximumConcurrency);
        return batch > ceiling ? ceiling : batch;
    }

    DecodeGraphExecutable* ensure(std::int32_t bucket, const std::function<void()>& body) {
        if (dead_) { return nullptr; }
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
            std::fprintf(stderr,
                         "prefill-graph: capture for bucket %d failed (%s); the eager prefill "
                         "body serves all further chunks\n",
                         bucket, error.what());
            dead_ = true;
            buckets_.clear();
            (void)cudaGetLastError();
            try {
                device_.synchronize();
            } catch (...) {}
            (void)cudaGetLastError();
            return nullptr;
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
    bool dead_               = false;
};

} // namespace ninfer::targets::qwen3_6::detail
