#include "api/ops/ngram_ple_host_table.h"

#include "core/device.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <sstream>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>

namespace sinfer::ops {

namespace {

std::mutex g_active_mutex;
std::shared_ptr<NgramPleHostTable> g_active;

// Rows a single thread copies before a gather is split over helpers: a decode round's few hundred
// rows stay on the host function's own thread.
constexpr std::int64_t kRowsPerThread = 4096;
constexpr unsigned kMaxThreads        = 16;

} // namespace

// SUROGATE_SERVE_NGRAM_TABLE_DUMP=<dir>: the first SUROGATE_SERVE_NGRAM_TABLE_DUMP_GATHERS (4)
// host gathers write gather-<n>.bin = {int64 count, int32 row_bytes, int32 0, int32 ids[count],
// rows[count * row_bytes]}, for checking the staged rows against the files (debugging aid).
void maybe_dump_gather(const std::int32_t* ids, std::int64_t count, std::int32_t row_bytes,
                       const std::uint8_t* rows) {
    static const char* dir = std::getenv("SUROGATE_SERVE_NGRAM_TABLE_DUMP");
    if (dir == nullptr || *dir == '\0') { return; }
    static const long limit = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_NGRAM_TABLE_DUMP_GATHERS");
        return raw != nullptr && *raw ? std::atol(raw) : 4L;
    }();
    static std::atomic<long> written{0};
    const long n = written.fetch_add(1);
    if (n >= limit) { return; }
    const std::string path = std::string(dir) + "/gather-" + std::to_string(n) + ".bin";
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { return; }
    const std::int32_t header[2] = {row_bytes, 0};
    std::fwrite(&count, sizeof(count), 1, file);
    std::fwrite(header, sizeof(header), 1, file);
    std::fwrite(ids, sizeof(std::int32_t), static_cast<std::size_t>(count), file);
    std::fwrite(rows, static_cast<std::size_t>(row_bytes), static_cast<std::size_t>(count), file);
    std::fclose(file);
}

struct NgramPleHostTable::Job {
    const NgramPleHostTable* table = nullptr;
    const Staging* staging         = nullptr;
    std::int64_t count             = 0;
    bool persistent                = false;
};

NgramPleHostTable::NgramPleHostTable(std::vector<NgramPleShard> shards, std::int32_t head_dim,
                                     NgramPleRowType type, float scale)
    : head_dim_(head_dim), type_(type), scale_(scale) {
    if (shards.empty() || head_dim <= 0) {
        throw std::invalid_argument("n-gram host table: no shards or no row width");
    }
    if (!(scale > 0.0F)) { throw std::invalid_argument("n-gram host table: the scale must be positive"); }
    row_bytes_ = type == NgramPleRowType::Bf16 ? 2 * head_dim : head_dim;
    std::map<std::string, std::size_t> by_path;
    for (const NgramPleShard& shard : shards) {
        if (shard.rows <= 0) { throw std::invalid_argument("n-gram host table: an empty shard in " + shard.path); }
        auto found = by_path.find(shard.path);
        if (found == by_path.end()) {
            const int fd = ::open(shard.path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                throw std::runtime_error("n-gram host table: cannot open " + shard.path + ": " + std::strerror(errno));
            }
            struct stat status {};
            if (::fstat(fd, &status) != 0) {
                const int error = errno;
                ::close(fd);
                throw std::runtime_error("n-gram host table: cannot stat " + shard.path + ": " + std::strerror(error));
            }
            const auto bytes = static_cast<std::size_t>(status.st_size);
            void* base       = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
            const int error  = errno;
            if (base == MAP_FAILED) {
                ::close(fd);
                throw std::runtime_error("n-gram host table: cannot map " + shard.path + ": " + std::strerror(error));
            }
            // Rows are read one at a time from anywhere in the file: read-ahead would only pull
            // pages no lookup asked for.
            (void)::madvise(base, bytes, MADV_RANDOM);
            mappings_.push_back({base, bytes, fd});
            found = by_path.emplace(shard.path, mappings_.size() - 1).first;
        }
        const Mapping& mapping = mappings_[found->second];
        const std::uint64_t end =
            shard.offset + static_cast<std::uint64_t>(shard.rows) * static_cast<std::uint64_t>(row_bytes_);
        if (end > mapping.bytes) {
            throw std::runtime_error("n-gram host table: " + shard.path + " is shorter than its shard at " +
                                     std::to_string(shard.offset));
        }
        shard_base_.push_back(static_cast<const std::uint8_t*>(mapping.base) + shard.offset);
        shard_first_.push_back(rows_);
        rows_ += shard.rows;
    }
    shard_first_.push_back(rows_);
    uniform_rows_ = shards.front().rows;
    for (std::size_t i = 0; i + 1 < shards.size(); ++i) {
        if (shards[i].rows != uniform_rows_) {
            uniform_rows_ = 0;
            break;
        }
    }
    const char* raw = std::getenv("SUROGATE_SERVE_NGRAM_TABLE_POPULATE");
    if (raw == nullptr || std::string(raw) != "0") { populate(); }
}

void NgramPleHostTable::populate() {
    // Ranges of whole pages covering every shard, in 64 MiB pieces the helpers take in turn.
    const auto page = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
    struct Piece {
        void* begin;
        std::size_t bytes;
        int fd;
        off_t offset; // of `begin` in the file
    };
    auto pieces = std::make_shared<std::vector<Piece>>();
    std::size_t total = 0;
    for (std::size_t s = 0; s < shard_base_.size(); ++s) {
        const auto first = reinterpret_cast<std::uintptr_t>(shard_base_[s]) / page * page;
        const auto last  = reinterpret_cast<std::uintptr_t>(shard_base_[s]) +
                          static_cast<std::uintptr_t>(shard_first_[s + 1] - shard_first_[s]) * row_bytes_;
        const Mapping* owner = nullptr;
        for (const Mapping& m : mappings_) {
            const auto base = reinterpret_cast<std::uintptr_t>(m.base);
            if (first >= base && first < base + m.bytes) { owner = &m; }
        }
        if (owner == nullptr) { continue; }
        const auto base = reinterpret_cast<std::uintptr_t>(owner->base);
        for (std::uintptr_t at = first; at < last; at += (64U << 20)) {
            const std::size_t bytes = std::min<std::uintptr_t>(64U << 20, (last - at + page - 1) / page * page);
            pieces->push_back({reinterpret_cast<void*>(at), bytes, owner->fd, static_cast<off_t>(at - base)});
            total += bytes;
        }
    }
    auto next     = std::make_shared<std::atomic<std::size_t>>(0);
    auto done     = std::make_shared<std::atomic<unsigned>>(0);
    auto started  = std::chrono::steady_clock::now();
    const unsigned helpers = std::max(1U, std::min(16U, std::thread::hardware_concurrency() / 4));
    for (unsigned h = 0; h < helpers; ++h) {
        populate_.emplace_back([this, pieces, next, done, started, helpers, total, page] {
            std::vector<std::uint8_t> scratch(4U << 20);
            for (std::size_t i = next->fetch_add(1); i < pieces->size() && !stop_populate_.load(); i = next->fetch_add(1)) {
                const Piece& piece = (*pieces)[i];
                // Read the piece into the page cache in large requests first (the mapping is
                // MADV_RANDOM, so faulting it in would go page by page), then map what was read.
                for (std::size_t done_bytes = 0; done_bytes < piece.bytes && !stop_populate_.load();) {
                    const ssize_t got = ::pread(piece.fd, scratch.data(), std::min(scratch.size(), piece.bytes - done_bytes),
                                                piece.offset + static_cast<off_t>(done_bytes));
                    if (got <= 0) { break; }
                    done_bytes += static_cast<std::size_t>(got);
                }
#ifdef MADV_POPULATE_READ
                if (::madvise(piece.begin, piece.bytes, MADV_POPULATE_READ) == 0) { continue; }
#endif
                // Older kernels: touch every page.
                volatile std::uint8_t sink = 0;
                for (std::size_t at = 0; at < piece.bytes && !stop_populate_.load(); at += page) {
                    sink = static_cast<const volatile std::uint8_t*>(piece.begin)[at];
                }
                (void)sink;
            }
            if (done->fetch_add(1) + 1 == helpers && !stop_populate_.load() && total >= (1ULL << 30)) {
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                std::fprintf(stderr, "n-gram table: %.1f GB of rows in the page cache and mapped (%.1f s)\n",
                             static_cast<double>(total) / 1e9, seconds);
            }
        });
    }
}

NgramPleHostTable::~NgramPleHostTable() {
    stop_populate_.store(true);
    for (std::thread& helper : populate_) {
        if (helper.joinable()) { helper.join(); }
    }
    const auto release = [](const std::unique_ptr<Staging>& staging) {
        if (!staging) { return; }
        if (staging->ids != nullptr) { (void)cudaFreeHost(staging->ids); }
        if (staging->rows != nullptr) { (void)cudaFreeHost(staging->rows); }
    };
    for (auto& [device, staging] : staging_) {
        (void)device;
        release(staging);
    }
    for (const auto& staging : retired_) { release(staging); }
    if (timing_[0].gathers.load() + timing_[1].gathers.load() + timing_[2].gathers.load() +
            timing_[3].gathers.load() != 0) {
        std::fprintf(stderr, "n-gram table: host gathers %s\n", timing_summary().c_str());
    }
    for (const Mapping& mapping : mappings_) {
        (void)::munmap(mapping.base, mapping.bytes);
        if (mapping.fd >= 0) { ::close(mapping.fd); }
    }
}

const NgramPleHostTable::Staging& NgramPleHostTable::prepare(std::int64_t max_rows) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(mutex_);
    auto& slot = staging_[device];
    if (slot && slot->capacity >= max_rows) { return *slot; }
    auto fresh = std::make_unique<Staging>();
    fresh->capacity = max_rows;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&fresh->ids),
                             static_cast<std::size_t>(max_rows) * sizeof(std::int32_t),
                             cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&fresh->ids_device), fresh->ids, 0));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&fresh->rows),
                             static_cast<std::size_t>(max_rows) * static_cast<std::size_t>(row_bytes_),
                             cudaHostAllocPortable));
    // A smaller staging may still be read by queued work or a captured graph: kept until the table goes.
    if (slot) { retired_.push_back(std::move(slot)); }
    slot = std::move(fresh);
    return *slot;
}

const NgramPleHostTable::Staging& NgramPleHostTable::staging() const {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto found = staging_.find(device);
    if (found == staging_.end() || !found->second) {
        throw std::logic_error("n-gram host table: no staging on device " + std::to_string(device) +
                               " (prepare it before the first forward)");
    }
    return *found->second;
}

void NgramPleHostTable::gather(const std::int32_t* ids, std::int64_t count, std::uint8_t* out) const {
    const auto copy = [&](std::int64_t begin, std::int64_t end) {
        std::uint64_t bad = 0;
        for (std::int64_t i = begin; i < end; ++i) {
            const std::int64_t id = ids[i];
            std::uint8_t* to      = out + i * row_bytes_;
            if (id < 0 || id >= rows_) {
                std::memset(to, 0, static_cast<std::size_t>(row_bytes_));
                ++bad;
                continue;
            }
            std::size_t shard = 0;
            if (uniform_rows_ > 0) {
                shard = std::min(static_cast<std::size_t>(id / uniform_rows_), shard_base_.size() - 1);
            } else {
                shard = static_cast<std::size_t>(
                    std::upper_bound(shard_first_.begin(), shard_first_.end(), id) - shard_first_.begin() - 1);
            }
            const std::int64_t local = id - shard_first_[shard];
            std::memcpy(to, shard_base_[shard] + local * row_bytes_, static_cast<std::size_t>(row_bytes_));
        }
        if (bad != 0) { bad_ids_.fetch_add(bad, std::memory_order_relaxed); }
    };
    const unsigned threads = static_cast<unsigned>(std::min<std::int64_t>(
        kMaxThreads, std::max<std::int64_t>(1, count / kRowsPerThread)));
    if (threads <= 1) {
        copy(0, count);
        return;
    }
    const std::int64_t chunk = (count + threads - 1) / threads;
    std::vector<std::jthread> helpers;
    helpers.reserve(threads - 1);
    for (unsigned t = 1; t < threads; ++t) {
        const std::int64_t begin = static_cast<std::int64_t>(t) * chunk;
        if (begin >= count) { break; }
        helpers.emplace_back([&copy, begin, end = std::min(count, begin + chunk)] { copy(begin, end); });
    }
    copy(0, std::min(count, chunk));
}

std::uint64_t NgramPleHostTable::bad_ids() const noexcept { return bad_ids_.load(std::memory_order_relaxed); }

void NgramPleHostTable::record(std::int64_t count, std::uint64_t nanoseconds) const {
    static const bool enabled = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_NGRAM_TABLE_STATS");
        return raw != nullptr && *raw != '\0' && std::string(raw) != "0";
    }();
    if (!enabled) { return; }
    Timing& bucket = timing_[count <= 16 ? 0 : count <= 256 ? 1 : count <= 4096 ? 2 : 3];
    bucket.gathers.fetch_add(1, std::memory_order_relaxed);
    bucket.rows.fetch_add(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
    bucket.nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
    std::uint64_t seen = bucket.max_nanoseconds.load(std::memory_order_relaxed);
    while (nanoseconds > seen && !bucket.max_nanoseconds.compare_exchange_weak(seen, nanoseconds)) {}
    const std::int64_t now  = std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::steady_clock::now().time_since_epoch()).count();
    std::int64_t last = last_print_.load(std::memory_order_relaxed);
    static const std::int64_t interval = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_NGRAM_TABLE_STATS");
        const long seconds = raw != nullptr ? std::atol(raw) : 0;
        return static_cast<std::int64_t>(seconds > 1 ? seconds : 30);
    }();
    if (now - last >= interval && last_print_.compare_exchange_strong(last, now)) {
        std::fprintf(stderr, "n-gram table: host gathers %s\n", timing_summary().c_str());
    }
}

std::string NgramPleHostTable::timing_summary() const {
    static const char* names[4] = {"<=16 rows", "17-256 rows", "257-4096 rows", ">4096 rows"};
    std::ostringstream out;
    for (int b = 0; b < 4; ++b) {
        const std::uint64_t n = timing_[b].gathers.load(), rows = timing_[b].rows.load(), ns = timing_[b].nanoseconds.load();
        if (n == 0) { continue; }
        out << names[b] << ": " << n << " gathers, " << rows / n << " rows each, mean "
            << static_cast<double>(ns) / static_cast<double>(n) / 1e3 << " us, max "
            << static_cast<double>(timing_[b].max_nanoseconds.load()) / 1e3 << " us, "
            << static_cast<double>(rows) * static_cast<double>(row_bytes_) / std::max<double>(1.0, static_cast<double>(ns)) << " GB/s; ";
    }
    return out.str();
}

void NgramPleHostTable::enqueue_gather(std::int64_t count, cudaStream_t stream) const {
    const Staging& buffers = staging();
    if (count > buffers.capacity) {
        throw std::runtime_error("n-gram host table: a forward of " + std::to_string(count) +
                                 " rows exceeds the staging's " + std::to_string(buffers.capacity) +
                                 " (SUROGATE_SERVE_NGRAM_TABLE_COLUMNS)");
    }
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capturing));
    auto job                    = std::make_unique<Job>(Job{this, &buffers, count, capturing != cudaStreamCaptureStatusNone});
    Job* raw                    = job.get();
    const cudaHostFn_t function = [](void* user) {
        auto* work = static_cast<Job*>(user);
        const auto started = std::chrono::steady_clock::now();
        work->table->gather(work->staging->ids, work->count, work->staging->rows);
        work->table->record(work->count, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now() - started).count()));
        maybe_dump_gather(work->staging->ids, work->count, work->table->row_bytes(), work->staging->rows);
        if (!work->persistent) { delete work; }
    };
    if (raw->persistent) {
        const std::lock_guard<std::mutex> lock(mutex_);
        captured_jobs_.push_back(std::move(job));
    } else {
        (void)job.release();
    }
    const cudaError_t status = cudaLaunchHostFunc(stream, function, raw);
    if (status != cudaSuccess && !raw->persistent) { delete raw; }
    CUDA_CHECK(status);
}

std::shared_ptr<NgramPleHostTable> NgramPleHostTable::active() {
    const std::lock_guard<std::mutex> lock(g_active_mutex);
    return g_active;
}

void NgramPleHostTable::set_active(std::shared_ptr<NgramPleHostTable> table) {
    const std::lock_guard<std::mutex> lock(g_active_mutex);
    g_active = std::move(table);
}

} // namespace sinfer::ops
