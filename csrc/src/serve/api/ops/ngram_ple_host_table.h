#pragma once

// The n-gram table read in place from files on disk: the release's own shards (BF16, or E4M3
// with an optional per-tensor scale), never converted and never copied into memory of the
// engine's own. The files are mapped read-only, so the page cache is the one copy every process on
// the host shares, and the kernel may reclaim it. Each forward's rows are gathered on the host into
// a bounded pinned staging buffer and copied to the device:
//
//   1. the GPU hashes every column's n-grams (as for a resident table) and writes the row ids into
//      a pinned, device-mapped buffer;
//   2. a CUDA host function on the same stream copies those rows out of the mappings into the
//      pinned staging buffer (several threads for a wide round);
//   3. one host-to-device copy puts them where the n-gram forward reads its embedding.
//
// Host functions can be captured, so captured decode graphs, prefill graph buckets and eager
// rounds all take this path. The staging buffers are per device and allocated by `prepare`, which
// must run before any graph capture (Variant::prewarm_device_scratch does).

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sinfer::ops {

/// How a stored row's values are encoded.
enum class NgramPleRowType : std::uint8_t {
    Bf16, ///< `head_dim` BF16 values
    E4m3, ///< `head_dim` FP8 E4M3FN values, each times the table's scale
};

/// One shard: `rows` consecutive table rows starting at byte `offset` of `path`.
struct NgramPleShard {
    std::string path;
    std::uint64_t offset = 0;
    std::int64_t rows    = 0;
};

class NgramPleHostTable {
public:
    /// Maps the shards' files read-only (each file once, MADV_RANDOM). Shards are in table order;
    /// every row is `head_dim` values of `type`. Throws when a file is shorter than its shards.
    /// Unless SUROGATE_SERVE_NGRAM_TABLE_POPULATE=0, helper threads then populate the shards'
    /// pages in the background (MADV_POPULATE_READ): the file is read into the page cache once (an
    /// already cached file costs only the page tables) and later gathers take no page faults. The
    /// pages stay reclaimable; nothing is locked.
    NgramPleHostTable(std::vector<NgramPleShard> shards, std::int32_t head_dim, NgramPleRowType type,
                      float scale);
    ~NgramPleHostTable();
    NgramPleHostTable(const NgramPleHostTable&)            = delete;
    NgramPleHostTable& operator=(const NgramPleHostTable&) = delete;

    [[nodiscard]] std::int64_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::int32_t head_dim() const noexcept { return head_dim_; }
    /// Stored bytes per row (2 * head_dim for BF16, head_dim for E4M3).
    [[nodiscard]] std::int32_t row_bytes() const noexcept { return row_bytes_; }
    [[nodiscard]] NgramPleRowType type() const noexcept { return type_; }
    [[nodiscard]] float scale() const noexcept { return scale_; }
    [[nodiscard]] std::size_t shards() const noexcept { return shard_base_.size(); }

    /// The current device's pinned buffers: the row ids the GPU writes (`ids`, mapped at
    /// `ids_device`) and the gathered rows (`rows`), for up to `capacity` rows a forward.
    struct Staging {
        std::int32_t* ids        = nullptr;
        std::int32_t* ids_device = nullptr;
        std::uint8_t* rows       = nullptr;
        std::int64_t capacity    = 0;
    };
    /// Allocates the current device's staging for `max_rows` rows (columns x heads) unless it
    /// already holds that many. Not during graph capture.
    const Staging& prepare(std::int64_t max_rows);
    /// The current device's staging; throws when `prepare` has not run for it.
    [[nodiscard]] const Staging& staging() const;

    /// Enqueues on `stream` the copy of the `count` rows whose ids are in staging().ids into
    /// staging().rows. Inside a capture the job is kept for every replay of the graph.
    void enqueue_gather(std::int64_t count, cudaStream_t stream) const;
    /// Copies `count` rows by id into `out` (count * row_bytes() bytes) on the calling thread and,
    /// for wide requests, helper threads. An id outside the table gives a zero row and is counted
    /// (`bad_ids`).
    void gather(const std::int32_t* ids, std::int64_t count, std::uint8_t* out) const;
    [[nodiscard]] std::uint64_t bad_ids() const noexcept;
    /// SUROGATE_SERVE_NGRAM_TABLE_STATS=1: the host gathers' timing, by request size, also printed
    /// to stderr every 30 s while gathers run and when the table goes.
    [[nodiscard]] std::string timing_summary() const;

    /// The table qwen4exp models of this process read (SUROGATE_SERVE_NGRAM_TABLE), if one was
    /// opened; `prewarm` prepares its staging on the current device.
    static std::shared_ptr<NgramPleHostTable> active();
    static void set_active(std::shared_ptr<NgramPleHostTable> table);

private:
    struct Mapping {
        void* base        = nullptr;
        std::size_t bytes = 0;
        int fd            = -1; // kept for the populate's reads
    };
    struct Job;

    std::vector<Mapping> mappings_;
    std::vector<const std::uint8_t*> shard_base_;
    std::vector<std::int64_t> shard_first_; // first table row of each shard, then the total
    std::int64_t uniform_rows_ = 0;         // rows per shard when every shard but the last has as many
    std::int64_t rows_         = 0;
    std::int32_t head_dim_     = 0;
    std::int32_t row_bytes_    = 0;
    NgramPleRowType type_      = NgramPleRowType::Bf16;
    float scale_               = 1.0F;

    mutable std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<Staging>> staging_;
    std::vector<std::unique_ptr<Staging>> retired_; // smaller stagings queued work may still read
    mutable std::vector<std::unique_ptr<Job>> captured_jobs_;
    mutable std::atomic<std::uint64_t> bad_ids_{0};
    struct Timing {
        std::atomic<std::uint64_t> gathers{0}, rows{0}, nanoseconds{0}, max_nanoseconds{0};
    };
    // up to 16 rows (one decode lane), 256 (16 lanes), 4,096, wider (prefill)
    mutable Timing timing_[4];
    mutable std::atomic<std::int64_t> last_print_{0};
    std::vector<std::thread> populate_;
    std::atomic<bool> stop_populate_{false};
    void populate();
    void record(std::int64_t count, std::uint64_t nanoseconds) const;
};

} // namespace sinfer::ops
