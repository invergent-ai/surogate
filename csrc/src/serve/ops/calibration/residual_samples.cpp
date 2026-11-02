// Residual samples at layer-group boundaries (api/ops/calibration.h), for the block influence
// 1 - cos(in, out) of each group (RECIPE.md section 9.3).
//
// A forward's sampled columns are equally spaced (every Nth over all forwards, in the order they
// start), so each boundary's copy is one strided cudaMemcpy2DAsync of whole columns into pinned
// memory, followed by an event: nothing waits for it. Once the event has fired (seen at a later
// call, or waited for by a flush), the host turns BF16 or FP32 into FP16 (saturating, and
// counting what saturated) and appends one record, in the order the copies were enqueued:
//
//   RecordHeader (64 bytes) | int32 columns[count] | int32 token_ids[count] | fp16 [count][rows]
//
// Several forwards can be open at once (a pipeline's stages run different ones): each keeps the
// boundary it expects next, and a call at boundary b goes to the oldest open forward expecting b.
// Each record also carries the token ids its own boundary read, and a record whose ids are not its
// forward's is flagged -- what a forward abandoned midway would cause for the ones after it.
// `residual-samples.json` beside the file says how to read it and how much it holds.

#include "api/ops/calibration.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sinfer::ops {
namespace {

struct RecordHeader {
    char magic[8];          // "RSAMPv1\0"
    std::uint64_t forward;  // ordinal of the forward among those this file samples
    std::int32_t boundary;  // 0 .. boundaries-1
    std::int32_t boundaries;
    std::int32_t layer;     // whose input this is; the trunk's layer count for the final residual
    std::int32_t rows;      // values per column (a hyper-connected residual: streams x hidden)
    std::int32_t count;     // sampled columns
    std::int32_t device;
    std::int64_t column_base; // the forward's first column in the global column count
    std::int32_t flags;       // kIdsDiffer: this boundary's token ids are not its forward's
    std::int32_t reserved[3];
};
/// The record's token ids (as read at its own boundary) differ from those its forward started
/// with: it was matched to the wrong forward, which only a forward abandoned midway can cause.
constexpr std::int32_t kIdsDiffer = 1;
static_assert(sizeof(RecordHeader) == 64);

struct Forward {
    std::uint64_t index      = 0;
    std::int64_t base        = 0;
    std::int32_t columns     = 0;
    std::int32_t first       = 0;
    std::int32_t count       = 0;
    std::int32_t next        = 0;
    /// The sampled columns' token ids, filled when boundary 0's copy lands (-1 without ids).
    std::shared_ptr<std::vector<std::int32_t>> token_ids;
};

struct PinnedBuffer {
    void* data           = nullptr;
    std::size_t capacity = 0;
};

/// One boundary's copy in flight: the device writes the pinned buffers in stream order, and the
/// record is written once `done` has fired, so no forward ever waits for its own samples.
struct Pending {
    cudaEvent_t done = nullptr;
    RecordHeader header{};
    std::int32_t first      = 0;
    std::size_t value_bytes = 2;
    std::size_t bytes       = 0;
    PinnedBuffer values;
    PinnedBuffer ids; // the sampled columns' token ids as this boundary read them (when it has ids)
    bool fills_token_ids = false; // boundary 0: they are the forward's
    std::shared_ptr<std::vector<std::int32_t>> token_ids;
};

/// More open forwards than any pipeline keeps in flight; past it the oldest is taken as lost.
constexpr std::size_t kMaxOpenForwards = 256;
/// Copies in flight beyond this many bytes make the next sample wait for the oldest to land.
constexpr std::size_t kMaxPendingBytes = std::size_t{1} << 30;

struct Sampler {
    std::mutex mutex;
    ResidualSampleOptions options;
    std::atomic<bool> enabled{false};
    std::FILE* file = nullptr;
    std::list<Forward> open; // oldest first (erasing one keeps the others' iterators)
    std::deque<Pending> pending; // copies in flight, in the order they were enqueued
    std::size_t pending_bytes = 0;
    std::vector<PinnedBuffer> spare; // pinned buffers of written records, reused
    std::int64_t columns_seen   = 0; // columns of every forward started (the sampling grid)
    std::int64_t reserved       = 0; // sampled columns promised to started forwards (the cap)
    std::int64_t sampled        = 0; // sampled columns of complete forwards
    std::uint64_t forwards      = 0;
    std::uint64_t complete      = 0;
    std::uint64_t sampled_forwards = 0;
    std::uint64_t dropped       = 0;
    std::uint64_t unmatched     = 0;
    std::uint64_t records       = 0;
    std::uint64_t ids_differ    = 0;
    std::uint64_t saturated     = 0;
    std::int32_t rows           = 0;
    std::int32_t boundaries     = 0;
    std::vector<std::uint16_t> half;
};

Sampler& sampler() {
    static Sampler instance;
    return instance;
}

thread_local int quiet_depth = 0; // ResidualSampleQuiet guards alive on this thread

/// FP16 bits of `value`, rounded to nearest even by the compiler's own conversion; a finite value
/// beyond FP16's range saturates to +-65504 (and is counted) rather than becoming infinite.
std::uint16_t to_half(float value, std::uint64_t& saturated) {
    if (std::isfinite(value) && std::fabs(value) > 65504.0F) {
        ++saturated;
        value = std::copysign(65504.0F, value);
    }
    const _Float16 half = static_cast<_Float16>(value);
    std::uint16_t bits  = 0;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float from_bf16(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0F;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

std::string summary_json(const Sampler& s) {
    std::ostringstream out;
    out << "{\n  \"format\": \"surogate-residual-samples-v1\",\n"
        << "  \"file\": \"residual-samples.bin\",\n"
        << "  \"model\": \"" << s.options.model << "\",\n"
        << "  \"every\": " << s.options.every << ",\n  \"max_tokens\": " << s.options.max_tokens << ",\n"
        << "  \"rows\": " << s.rows << ",\n  \"boundaries\": " << s.boundaries << ",\n"
        << "  \"dtype\": \"float16\",\n"
        << "  \"sampled_columns\": " << s.sampled << ",\n  \"columns_seen\": " << s.columns_seen << ",\n"
        << "  \"forwards\": " << s.forwards << ",\n  \"complete_forwards\": " << s.complete << ",\n"
        << "  \"sampled_forwards\": " << s.sampled_forwards << ",\n"
        << "  \"open_forwards\": " << s.open.size() << ",\n"
        << "  \"dropped_forwards\": " << s.dropped << ",\n  \"unmatched_calls\": " << s.unmatched << ",\n"
        << "  \"records\": " << s.records << ",\n  \"records_ids_differ\": " << s.ids_differ << ",\n"
        << "  \"saturated_values\": " << s.saturated << ",\n"
        << "  \"record_layout\": \"64-byte header {char magic[8]='RSAMPv1'; u64 forward; i32 boundary, boundaries, "
           "layer, rows, count, device; i64 column_base; i32 flags (1: token ids differ from the forward's); i32 reserved[3]}, then i32 columns[count] (within the "
           "forward), i32 token_ids[count], f16 values[count][rows] (column-major: one column's rows contiguous)\",\n"
        << "  \"semantics\": [\n"
        << "    \"boundary b < boundaries-1 is the residual entering layer `layer` (a layer-group start: layer 0 and "
           "every layer after a full-attention layer); boundary boundaries-1 is the final residual the output mixer "
           "reads, i.e. the last group's output\",\n"
        << "    \"a hyper-connected residual stores its streams stacked: rows = streams * hidden, stream s at rows "
           "[s*hidden, (s+1)*hidden)\",\n"
        << "    \"records flagged 1 (records_ids_differ) belong to another forward than their header says: drop their "
           "forward\",\n"
        << "    \"use only forwards that have a record for every boundary; sampled_forwards of the complete_forwards "
           "hold sampled columns (a forward between two grid points holds none), dropped_forwards fell behind a "
           "younger forward at some boundary and are incomplete\",\n"
        << "    \"columns are every Nth over all forwards in the order they started (column_base + columns[i]); "
           "sampling stops at max_tokens\",\n"
        << "    \"only eagerly executed forwards are sampled (CUDA-graph replays run no host code)\"\n  ]\n}\n";
    return out.str();
}

void write_summary(const Sampler& s) {
    std::ofstream(std::filesystem::path(s.options.directory) / "residual-samples.json",
                  std::ios::trunc) << summary_json(s);
}

/// A pinned buffer of at least `bytes`: the smallest spare one that fits, else a new one (rounded
/// to 1 MiB, so the few record sizes a run has share buffers).
PinnedBuffer take_buffer(Sampler& s, std::size_t bytes) {
    auto best = s.spare.end();
    for (auto it = s.spare.begin(); it != s.spare.end(); ++it) {
        if (it->capacity >= bytes && (best == s.spare.end() || it->capacity < best->capacity)) { best = it; }
    }
    if (best != s.spare.end()) {
        const PinnedBuffer out = *best;
        s.spare.erase(best);
        return out;
    }
    constexpr std::size_t kMiB = std::size_t{1} << 20;
    PinnedBuffer out;
    out.capacity = (std::max(bytes, kMiB) + kMiB - 1) / kMiB * kMiB;
    CUDA_CHECK(cudaMallocHost(&out.data, out.capacity));
    return out;
}

void write_record(Sampler& s, Pending& p) {
    const auto count = static_cast<std::size_t>(p.header.count);
    const std::int32_t* ids = p.token_ids->data();
    if (p.ids.data != nullptr) {
        const auto* observed = static_cast<const std::int32_t*>(p.ids.data);
        if (p.fills_token_ids) {
            std::memcpy(p.token_ids->data(), observed, count * sizeof(std::int32_t));
        } else if (std::find(p.token_ids->begin(), p.token_ids->end(), -1) == p.token_ids->end() &&
                   !std::equal(observed, observed + count, p.token_ids->begin())) {
            p.header.flags |= kIdsDiffer;
            ++s.ids_differ;
        }
        ids = observed;
    }
    const std::size_t values = static_cast<std::size_t>(p.header.rows) * count;
    s.half.resize(values);
    if (p.value_bytes == 2) {
        const auto* in = static_cast<const std::uint16_t*>(p.values.data);
        for (std::size_t i = 0; i < values; ++i) { s.half[i] = to_half(from_bf16(in[i]), s.saturated); }
    } else {
        const auto* in = static_cast<const float*>(p.values.data);
        for (std::size_t i = 0; i < values; ++i) { s.half[i] = to_half(in[i], s.saturated); }
    }
    std::vector<std::int32_t> column_ids(count);
    for (std::size_t i = 0; i < count; ++i) {
        column_ids[i] = p.first + static_cast<std::int32_t>(i) * s.options.every;
    }
    std::fwrite(&p.header, sizeof(p.header), 1, s.file);
    std::fwrite(column_ids.data(), sizeof(std::int32_t), count, s.file);
    std::fwrite(ids, sizeof(std::int32_t), count, s.file);
    std::fwrite(s.half.data(), sizeof(std::uint16_t), values, s.file);
    ++s.records;
}

enum class Drain : std::uint8_t {
    Landed,   // the copies that have already landed, without waiting
    UnderCap, // and wait for the oldest while more than kMaxPendingBytes are in flight
    All,      // wait for every one
};

/// Writes (or, without `write`, discards) the copies in flight, oldest first, as `mode` says. In
/// order, so a forward's boundary-0 record, which fills its token ids, precedes its others.
void drain(Sampler& s, Drain mode, bool write = true) {
    while (!s.pending.empty()) {
        Pending& p = s.pending.front();
        if (mode == Drain::All || (mode == Drain::UnderCap && s.pending_bytes > kMaxPendingBytes)) {
            CUDA_CHECK(cudaEventSynchronize(p.done));
        } else {
            const cudaError_t state = cudaEventQuery(p.done);
            if (state == cudaErrorNotReady) { break; }
            CUDA_CHECK(state);
        }
        if (write) { write_record(s, p); }
        CUDA_CHECK(cudaEventDestroy(p.done));
        s.spare.push_back(p.values);
        if (p.ids.data != nullptr) { s.spare.push_back(p.ids); }
        s.pending_bytes -= p.bytes;
        s.pending.pop_front();
    }
}

} // namespace

void configure_residual_samples(const ResidualSampleOptions& options) {
    Sampler& s = sampler();
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (s.file != nullptr) { throw std::logic_error("residual samples: configure once, before any model loads"); }
    if (options.directory.empty()) { return; }
    if (options.every <= 0 || options.max_tokens <= 0) {
        throw std::invalid_argument("residual samples: every and max_tokens must be positive");
    }
    std::filesystem::create_directories(options.directory);
    const auto path = std::filesystem::path(options.directory) / "residual-samples.bin";
    s.file = std::fopen(path.c_str(), "wb");
    if (s.file == nullptr) { throw std::runtime_error("residual samples: cannot write " + path.string()); }
    s.options = options;
    write_summary(s);
    s.enabled.store(true, std::memory_order_relaxed);
}

bool residual_samples_enabled() noexcept { return sampler().enabled.load(std::memory_order_relaxed); }

ResidualSampleQuiet::ResidualSampleQuiet() noexcept { ++quiet_depth; }
ResidualSampleQuiet::~ResidualSampleQuiet() { --quiet_depth; }

void residual_sample(std::int32_t boundary, std::int32_t boundaries, std::int32_t layer,
                     const Tensor& residual, const Tensor& ids, cudaStream_t stream) {
    Sampler& s = sampler();
    if (!s.enabled.load(std::memory_order_relaxed) || quiet_depth > 0) { return; }
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capturing));
    if (capturing != cudaStreamCaptureStatusNone) { return; }
    if (residual.dtype != DType::BF16 && residual.dtype != DType::FP32) {
        throw std::invalid_argument("residual samples: the residual must be BF16 or FP32");
    }
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (s.file == nullptr) { return; }
    drain(s, Drain::Landed);
    const std::int32_t columns = residual.ne[1];
    const std::int32_t rows    = residual.ne[0];
    std::list<Forward>::iterator at;
    if (boundary == 0) {
        if (s.open.size() >= kMaxOpenForwards) {
            s.open.pop_front();
            ++s.dropped;
        }
        Forward next;
        next.index   = s.forwards++;
        next.base    = s.columns_seen;
        next.columns = columns;
        s.columns_seen += columns;
        const auto every = static_cast<std::int64_t>(s.options.every);
        next.first = static_cast<std::int32_t>((every - next.base % every) % every);
        const std::int64_t left = std::max<std::int64_t>(0, s.options.max_tokens - s.reserved);
        next.count = next.first < columns
                         ? static_cast<std::int32_t>(std::min<std::int64_t>(
                               (columns - 1 - next.first) / every + 1, left))
                         : 0;
        s.reserved += next.count;
        next.token_ids = std::make_shared<std::vector<std::int32_t>>(static_cast<std::size_t>(next.count), -1);
        s.open.push_back(std::move(next));
        at = std::prev(s.open.end());
        if (s.rows == 0) { s.rows = rows; s.boundaries = boundaries; }
    } else {
        // The oldest open forward expecting this boundary. Every boundary sees the forwards in the
        // order they started, so an older one that has not reached it yet never will: it is lost
        // (and so is one of another width, which no forward expecting this call can be).
        at = s.open.begin();
        while (at != s.open.end() && at->next != boundary) { ++at; }
        if (at == s.open.end()) {
            ++s.unmatched;
            return; // started before the sampler was reset, or never seen at boundary 0
        }
        for (auto older = s.open.begin(); older != at;) {
            if (older->next < boundary) {
                older = s.open.erase(older);
                ++s.dropped;
            } else {
                ++older;
            }
        }
        if (at->columns != columns) {
            s.open.erase(at);
            ++s.dropped;
            ++s.unmatched;
            return;
        }
    }
    Forward& f = *at;
    if (f.count > 0) {
        const std::size_t value_bytes  = residual.dtype == DType::BF16 ? 2 : 4;
        const std::size_t column_bytes = static_cast<std::size_t>(rows) * value_bytes;
        const auto count               = static_cast<std::size_t>(f.count);
        const auto stride              = static_cast<std::size_t>(s.options.every);
        Pending p;
        p.first       = f.first;
        p.value_bytes = value_bytes;
        p.token_ids   = f.token_ids;
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        std::memcpy(p.header.magic, "RSAMPv1", 8);
        p.header.forward     = f.index;
        p.header.boundary    = boundary;
        p.header.boundaries  = boundaries;
        p.header.layer       = layer;
        p.header.rows        = rows;
        p.header.count       = f.count;
        p.header.device      = device;
        p.header.column_base = f.base;
        if (ids.data != nullptr && ids.numel() >= columns) {
            p.fills_token_ids = boundary == 0;
            p.ids = take_buffer(s, count * sizeof(std::int32_t));
            CUDA_CHECK(cudaMemcpy2DAsync(p.ids.data, sizeof(std::int32_t),
                                         static_cast<const std::int32_t*>(ids.data) + f.first,
                                         stride * sizeof(std::int32_t), sizeof(std::int32_t), count,
                                         cudaMemcpyDeviceToHost, stream));
            p.bytes += count * sizeof(std::int32_t);
        }
        p.values = take_buffer(s, column_bytes * count);
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(f.first) * residual.nb[1];
        CUDA_CHECK(cudaMemcpy2DAsync(p.values.data, column_bytes, source, stride * residual.nb[1],
                                     column_bytes, count, cudaMemcpyDeviceToHost, stream));
        p.bytes += column_bytes * count;
        CUDA_CHECK(cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(p.done, stream));
        s.pending_bytes += p.bytes;
        s.pending.push_back(std::move(p));
        drain(s, Drain::UnderCap);
    }
    f.next = boundary + 1;
    if (boundary == boundaries - 1) {
        s.sampled += f.count;
        ++s.complete;
        if (f.count > 0) { ++s.sampled_forwards; }
        s.open.erase(at);
        if (s.complete % 64 == 0) {
            std::fflush(s.file);
            write_summary(s);
        }
    }
}

std::string residual_samples_flush() {
    Sampler& s = sampler();
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (s.file == nullptr) { throw std::runtime_error("residual samples are off (start the engine with --residual-samples-out DIR)"); }
    drain(s, Drain::All);
    std::fflush(s.file);
    write_summary(s);
    return summary_json(s);
}

std::string residual_samples_reset() {
    Sampler& s = sampler();
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (s.file == nullptr) { throw std::runtime_error("residual samples are off (start the engine with --residual-samples-out DIR)"); }
    drain(s, Drain::All, /*write=*/false);
    const auto path = std::filesystem::path(s.options.directory) / "residual-samples.bin";
    std::fclose(s.file);
    s.file = std::fopen(path.c_str(), "wb");
    if (s.file == nullptr) {
        s.enabled.store(false, std::memory_order_relaxed);
        throw std::runtime_error("residual samples: cannot rewrite " + path.string());
    }
    s.open.clear();
    s.columns_seen = s.reserved = s.sampled = 0;
    s.forwards = s.complete = s.sampled_forwards = s.dropped = s.unmatched = s.records = s.ids_differ = s.saturated = 0;
    write_summary(s);
    return summary_json(s);
}

} // namespace sinfer::ops
