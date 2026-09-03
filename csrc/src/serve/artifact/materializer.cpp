#include "artifact/materializer.h"

#include "ops/linear/ggml/ggml_repack.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sinfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 4;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

class Slot {
public:
    explicit Slot(std::size_t bytes) : buffer(bytes) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            CUDA_CHECK(cudaEventSynchronize(event));
            pending = false;
        }
    }

    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending      = false;
};

struct CopyRange {
    /// Which file the offsets are in: zero is the artifact, one and up an external file it
    /// serves weights from. Ranges are ordered and coalesced within a source, never across one.
    std::uint32_t source       = 0;
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
    /// Offset into the transform scratch, for a range whose object is rearranged rather than
    /// copied; `destination` is resolved from it once the scratch is allocated.
    std::uint64_t staged_at = 0;
    bool staged             = false;
};

struct PendingTransform {
    PayloadTransform transform = PayloadTransform::None;
    std::uint64_t source       = 0; // offset into the scratch
    std::byte* destination     = nullptr;
    std::uint64_t bytes        = 0;
    std::uint64_t stored       = 0;
    std::int32_t rows          = 0;
    std::int32_t columns       = 0;
};

struct ReadSpan {
    std::uint32_t source = 0;
    std::uint64_t begin  = 0;
    std::uint64_t end    = 0;
};

} // namespace

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device;
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() {
    if (!device_arena_) { throw ArtifactError("artifact has no device tensor backing"); }
    return *device_arena_;
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress) {
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
    out.stats_.device_capacity_bytes = capacity;
    out.stats_.tensor_count          = plan.device_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }

    std::vector<CopyRange> ranges;
    ranges.reserve(plan.device_objects.size());
    std::vector<PendingTransform> transforms;
    std::vector<std::size_t> staged_ranges;
    std::uint64_t staged_bytes = 0;
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    std::uint64_t total          = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        const auto& descriptor = reader.objects().at(placement.object.index);
        const auto object_runs = reader.runs(descriptor);
        DeviceSpan storage =
            out.device_arena_->alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                           static_cast<std::size_t>(placement.alignment));
        const auto actual_offset =
            static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                       static_cast<std::byte*>(out.device_arena_->base()));
        std::uint64_t run_bytes = 0;
        for (const PayloadRun& run : object_runs) {
            run_bytes = checked_add(run_bytes, run.bytes, "artifact tensor run bytes overflow u64");
        }
        const auto* tensor = std::get_if<TensorDescriptor>(&descriptor);
        const auto transform =
            tensor != nullptr ? tensor->transform : PayloadTransform::None;
        if (actual_offset != placement.offset ||
            (transform == PayloadTransform::None && run_bytes != placement.bytes)) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device = storage.data;
        // A transformed object's runs are its source bytes, which land in scratch and are
        // rearranged into `storage` once every read has finished.
        std::byte* cursor_out = static_cast<std::byte*>(storage.data);
        if (transform != PayloadTransform::None) {
            transforms.push_back(PendingTransform{
                .transform   = transform,
                .source      = staged_bytes,
                .destination = cursor_out,
                .bytes       = run_bytes,
                .stored      = placement.bytes,
                .rows        = static_cast<std::int32_t>(tensor->shape.at(0)),
                .columns     = static_cast<std::int32_t>(tensor->shape.at(1)),
            });
            cursor_out    = nullptr; // filled in below, once scratch exists
            staged_bytes  = checked_add(staged_bytes, run_bytes, "artifact staging overflow u64");
            staged_ranges.push_back(ranges.size());
        }
        // An object gathered from several runs of a GGUF lands as several copies into one
        // allocation, in the order the runs are declared.
        std::uint64_t within = 0;
        for (const PayloadRun& run : object_runs) {
            ranges.push_back(CopyRange{
                .source       = run.source,
                .source_begin = run.offset,
                .source_end   = checked_add(run.offset, run.bytes,
                                            "artifact tensor source range overflows u64"),
                .destination  = cursor_out == nullptr ? nullptr : cursor_out + within,
                .staged_at    = cursor_out == nullptr
                                    ? transforms.back().source + within
                                    : 0,
                .staged       = cursor_out == nullptr,
            });
            within += run.bytes;
        }
        total = checked_add(total, transform == PayloadTransform::None ? placement.bytes : run_bytes,
                            "artifact tensor byte count overflows u64");
    }
    if (ranges.empty()) { throw ArtifactError("materialization plan has no device tensors"); }
    // Scratch for the objects that are rearranged rather than copied. It is transient: the reads
    // land here, the rearranging writes their real allocations, and it is freed before serving.
    std::unique_ptr<DeviceArena> staging;
    if (staged_bytes != 0) {
        staging = std::make_unique<DeviceArena>(static_cast<std::size_t>(staged_bytes));
        auto* base = static_cast<std::byte*>(
            staging->alloc_bytes(static_cast<std::size_t>(staged_bytes), 256).data);
        for (CopyRange& range : ranges) {
            if (range.staged) { range.destination = base + range.staged_at; }
        }
        for (PendingTransform& pending : transforms) { pending.source += 0; }
        for (std::size_t i = 0; i < transforms.size(); ++i) {
            transforms[i].source = reinterpret_cast<std::uint64_t>(base + transforms[i].source);
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source == b.source ? a.source_begin < b.source_begin : a.source < b.source;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        // Two objects may legitimately read the same bytes of a GGUF -- a tied embedding and
        // output head, say -- so overlap is only an error inside the artifact's own payload,
        // whose objects the directory already requires to be disjoint and ordered.
        if (ranges[i].source == ranges[i - 1].source && ranges[i].source == 0 &&
            ranges[i].source_begin < ranges[i - 1].source_end) {
            throw ArtifactError("materialization source ranges overlap");
        }
    }

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || read_spans.back().source != range.source ||
            begin > align_up(read_spans.back().end, alignment,
                             "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{range.source, begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::size_t slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
    const std::size_t slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_read_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes));
    }
    out.stats_.peak_staging_bytes = static_cast<std::uint64_t>(slot_bytes) * slot_count;

    std::size_t next_slot  = 0;
    std::size_t next_range = 0;
    const auto start       = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
            Slot& slot = *slots[next_slot++ % slot_count];
            slot.wait();

            const std::uint64_t remaining = span.end - source;
            const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            auto destination =
                std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
            const std::size_t bytes_read = reader.read_direct(span.source, source, destination);
            const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
            if (bytes_read < required) {
                throw ArtifactError("direct artifact read ended before the planned tensor range");
            }
            out.stats_.file_bytes =
                checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
            const std::uint64_t chunk_end =
                checked_add(source, bytes_read, "artifact direct I/O result overflows u64");

            // Offsets only order within a file, so both walks are gated on the source too:
            // an external file's small offsets must not match ranges left in the artifact's.
            while (next_range < ranges.size() &&
                   (ranges[next_range].source < span.source ||
                    (ranges[next_range].source == span.source &&
                     ranges[next_range].source_end <= source))) {
                ++next_range;
            }
            std::size_t range_index = next_range;
            while (range_index < ranges.size() && ranges[range_index].source == span.source &&
                   ranges[range_index].source_begin < chunk_end) {
                const CopyRange& range         = ranges[range_index];
                const std::uint64_t copy_begin = std::max(source, range.source_begin);
                const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                if (copy_begin < copy_end) {
                    const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                    CUDA_CHECK(cudaMemcpyAsync(
                        range.destination +
                            static_cast<std::size_t>(copy_begin - range.source_begin),
                        static_cast<std::byte*>(slot.buffer.data()) +
                            static_cast<std::size_t>(copy_begin - source),
                        amount, cudaMemcpyHostToDevice, device.load_stream));
                    copied =
                        checked_add(copied, amount, "artifact copied byte count overflows u64");
                }
                if (range.source_end <= chunk_end) {
                    ++range_index;
                } else {
                    break;
                }
            }
            next_range = range_index;
            CUDA_CHECK(cudaEventRecord(slot.event, device.load_stream));
            slot.pending = true;

            if (progress != nullptr && progress->callback && copied != last_published &&
                copied < total) {
                last_published = copied;
                progress->callback("weights", copied, total);
            }
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    if (copied != total || next_range != ranges.size()) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    for (const PendingTransform& pending : transforms) {
        switch (pending.transform) {
        case PayloadTransform::Q8ToW8RowSplit:
            ops::detail::ggml::q8_0_to_w8_rowsplit_launch(
                reinterpret_cast<const void*>(pending.source), pending.destination, pending.rows,
                pending.columns, static_cast<std::size_t>(pending.stored), device.load_stream);
            break;
        case PayloadTransform::None:
            throw ArtifactError("a pending transform must name one");
        }
    }
    if (!transforms.empty()) { CUDA_CHECK(cudaStreamSynchronize(device.load_stream)); }
    staging.reset();
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

} // namespace sinfer::artifact
