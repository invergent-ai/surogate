#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "artifact/reader.h"

#include "ops/linear/ggml/ggml_repack.h"

#include <array>
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
#include <unordered_map>

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
    /// For a range whose object is rearranged rather than copied: its offset within that
    /// object's own staged block, and which transform owns it. `destination` is resolved from
    /// the pair when the wave holding that transform allocates its scratch.
    std::uint64_t staged_at   = 0;
    bool staged               = false;
    std::size_t transform_index = 0;
};

struct PendingTransform {
    PayloadTransform transform = PayloadTransform::None;
    /// The object's column permutation, if it has one; uploaded with the scratch.
    std::vector<std::int32_t> host_map;
    const std::int32_t* group_map = nullptr;
    std::uint64_t source       = 0; // resolved to a scratch address when its wave runs
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

void MaterializedArtifact::attach_host_object(ObjectHandle handle, const void* device_pointer) {
    if (handle.index >= objects_.size()) {
        throw ArtifactError("host bank attachment names an object outside the artifact");
    }
    if (objects_[handle.index].device != nullptr) {
        throw ArtifactError("host bank attachment would replace a device-resident object");
    }
    objects_[handle.index].device = const_cast<void*>(device_pointer);
    std::size_t offset = 0;
    for (auto& segment : objects_[handle.index].segments) {
        segment.qdata = static_cast<const std::byte*>(device_pointer) + offset;
        offset += segment.bytes;
    }
}

std::span<const WeightSegment> MaterializedArtifact::segments(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].segments;
}

std::span<const std::int32_t> MaterializedArtifact::input_group_map(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    const auto& object = objects_[handle.index];
    return {object.input_group_map, object.input_groups};
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
                                 DeviceContext& device, LoadProgress* progress,
                                 std::span<const BorrowedTensor> borrowed) {
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    std::unordered_map<std::string, const BorrowedTensor*> bindings;
    for (const auto& tensor : borrowed) {
        if (!bindings.emplace(tensor.name, &tensor).second) {
            throw ArtifactError("duplicate borrowed tensor: " + tensor.name);
        }
    }
    if (!borrowed.empty()) {
        if (!plan.bank_objects.empty() || bindings.size() != plan.device_objects.size()) {
            throw ArtifactError("borrowed weights must cover every device tensor without host offload");
        }
        for (const auto& placement : plan.device_objects) {
            const auto& descriptor = std::get<TensorDescriptor>(reader.objects().at(placement.object.index));
            const auto found = bindings.find(descriptor.name);
            if (found == bindings.end()) { throw ArtifactError("missing borrowed tensor: " + descriptor.name); }
            const auto& tensor = *found->second;
            const auto format = tensor.dtype == SharedWeightDType::FP32 ? NumericFormat::FP32 : NumericFormat::BF16;
            if (descriptor.format != format ||
                descriptor.layout != StorageLayout::ContiguousLeV1 || tensor.shape != descriptor.shape ||
                tensor.bytes != placement.bytes || tensor.device != device.device || tensor.data == nullptr) {
                throw ArtifactError("borrowed tensor must match contiguous dtype, shape, bytes and device: " + descriptor.name);
            }
            cudaPointerAttributes attributes{};
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, tensor.data));
            if (attributes.type != cudaMemoryTypeDevice || attributes.device != device.device) {
                throw ArtifactError("borrowed tensor is not on the engine's CUDA device: " + descriptor.name);
            }
            out.objects_.at(placement.object.index).device = tensor.data;
        }
        // A non-owning empty arena preserves the model's accounting interface.
        out.device_arena_ = std::make_unique<DeviceArena>(DeviceSpan{});
    } else {
        out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
        out.stats_.device_capacity_bytes = capacity;
    }
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

    if (!borrowed.empty()) { return out; }

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
        if (tensor != nullptr && !tensor->segments.empty()) {
            // Typed row runs land back to back; each segment starts where the previous one's
            // bytes end.
            auto& segments     = out.objects_.at(placement.object.index).segments;
            std::uint64_t at   = 0;
            std::int32_t row   = 0;
            const auto columns = tensor->shape.at(1);
            for (const TensorSegment& segment : tensor->segments) {
                const std::array<std::uint64_t, 2> segment_shape = {segment.rows, columns};
                const auto bytes = tensor_encoded_size(tensor->layout, segment.format, segment_shape);
                segments.push_back(WeightSegment{
                    .row_begin = row,
                    .rows      = static_cast<std::int32_t>(segment.rows),
                    .qtype     = qtype_for(segment.format),
                    .qdata     = static_cast<const std::byte*>(storage.data) + at,
                    .bytes     = bytes,
                });
                at = checked_add(at, bytes, "segment offset");
                row += static_cast<std::int32_t>(segment.rows);
            }
        }
        // A transformed object's runs are its source bytes, which land in scratch and are
        // rearranged into `storage` once every read has finished.
        std::byte* cursor_out = static_cast<std::byte*>(storage.data);
        if (transform != PayloadTransform::None) {
            transforms.push_back(PendingTransform{
                .transform   = transform,
                .host_map    = tensor->group_map,
                .source      = 0, // its wave's offset, then that wave's address
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
                .staged_at    = cursor_out == nullptr ? within : 0,
                .staged       = cursor_out == nullptr,
                .transform_index = transforms.empty() ? 0 : transforms.size() - 1,
            });
            within += run.bytes;
        }
        total = checked_add(total, transform == PayloadTransform::None ? placement.bytes : run_bytes,
                            "artifact tensor byte count overflows u64");
    }
    // Host tensors retain the same typed row metadata as resident tensors. Pointers are
    // filled by attach_host_object once their pinned allocation exists.
    for (const auto& placement : plan.bank_objects) {
        const auto& tensor = std::get<TensorDescriptor>(reader.objects().at(placement.object.index));
        auto& segments = out.objects_.at(placement.object.index).segments;
        std::int32_t row = 0;
        for (const auto& segment : tensor.segments) {
            const std::array<std::uint64_t, 2> shape = {segment.rows, tensor.shape.at(1)};
            segments.push_back(WeightSegment{
                .row_begin = row, .rows = static_cast<std::int32_t>(segment.rows),
                .qtype = qtype_for(segment.format), .qdata = nullptr,
                .bytes = tensor_encoded_size(tensor.layout, segment.format, shape),
            });
            row += static_cast<std::int32_t>(segment.rows);
        }
    }
    if (ranges.empty()) { throw ArtifactError("materialization plan has no device tensors"); }
    // Column group maps the ops apply to activations: a ggml-blocks object read in the file's
    // column order carries one without a transform. They live as long as the weights do.
    {
        std::vector<std::int32_t> flat;
        std::vector<std::pair<std::size_t, std::size_t>> spans; // object index, offset
        const auto collect = [&](const auto& placements) {
            for (const auto& placement : placements) {
                const auto* tensor = std::get_if<TensorDescriptor>(&reader.objects().at(placement.object.index));
                if (tensor == nullptr || tensor->group_map.empty() || tensor->transform != PayloadTransform::None) {
                    continue;
                }
                spans.emplace_back(placement.object.index, flat.size());
                flat.insert(flat.end(), tensor->group_map.begin(), tensor->group_map.end());
            }
        };
        collect(plan.device_objects);
        collect(plan.bank_objects);
        if (!flat.empty()) {
            out.input_maps_ = std::make_unique<DeviceArena>(flat.size() * sizeof(std::int32_t));
            auto* maps      = static_cast<std::int32_t*>(
                out.input_maps_->alloc_bytes(flat.size() * sizeof(std::int32_t), 256).data);
            CUDA_CHECK(cudaMemcpyAsync(maps, flat.data(), flat.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, device.load_stream));
            CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
            for (const auto& [index, offset] : spans) {
                const auto* tensor = std::get_if<TensorDescriptor>(&reader.objects().at(index));
                out.objects_.at(index).input_group_map = maps + offset;
                out.objects_.at(index).input_groups    = tensor->group_map.size();
            }
        }
    }
    std::unique_ptr<DeviceArena> group_maps;
    if (staged_bytes != 0) {
        // The column permutations, gathered into one small upload: one entry per 32 columns,
        // shared by every row of an object, so a whole model's worth is a few kilobytes. These
        // live for the whole load rather than per wave, being tiny.
        std::vector<std::int32_t> flat;
        for (const PendingTransform& pending : transforms) {
            flat.insert(flat.end(), pending.host_map.begin(), pending.host_map.end());
        }
        if (!flat.empty()) {
            group_maps = std::make_unique<DeviceArena>(flat.size() * sizeof(std::int32_t));
            auto* maps = static_cast<std::int32_t*>(
                group_maps->alloc_bytes(flat.size() * sizeof(std::int32_t), 256).data);
            CUDA_CHECK(cudaMemcpyAsync(maps, flat.data(), flat.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, device.load_stream));
            std::size_t cursor = 0;
            for (PendingTransform& pending : transforms) {
                if (pending.host_map.empty()) { continue; }
                pending.group_map = maps + cursor;
                cursor += pending.host_map.size();
            }
        }
    }
    // The load runs in passes: the objects copied straight to their allocations first, then the
    // rearranged ones in waves that each hold at most `kLoadStagingCapBytes` of scratch (plus
    // whatever single object is larger than that on its own). Every pass reads its own ranges in
    // file order, so the bytes read are the same as one pass would read; what changes is that the
    // scratch is a wave's worth rather than the whole model's -- ~9 GiB on a 200 GB checkpoint --
    // which is memory anything created before the load, the expert slot pool above all, gets to
    // keep. `projected_load_staging_bytes` projects the same figure.
    struct Pass {
        std::vector<CopyRange> ranges;
        std::vector<std::size_t> transforms; // empty for the pass that copies straight through
        std::uint64_t staging_bytes = 0;
    };
    std::vector<Pass> passes;
    {
        std::uint64_t largest = 0;
        for (const PendingTransform& pending : transforms) {
            largest = std::max(largest, pending.bytes);
        }
        const std::uint64_t cap = std::max<std::uint64_t>(largest, kLoadStagingCapBytes);
        std::vector<std::uint64_t> offset_of(transforms.size(), 0);
        std::vector<std::size_t> pass_of(transforms.size(), 0);
        Pass direct;
        passes.push_back(std::move(direct)); // pass 0 is always the straight copies
        for (std::size_t t = 0; t < transforms.size(); ++t) {
            const std::uint64_t need =
                align_up(transforms[t].bytes, 256, "artifact staging overflow u64");
            if (passes.size() == 1 || passes.back().staging_bytes + need > cap) {
                passes.push_back(Pass{});
            }
            offset_of[t]         = passes.back().staging_bytes;
            transforms[t].source = passes.back().staging_bytes;
            pass_of[t]   = passes.size() - 1;
            passes.back().transforms.push_back(t);
            passes.back().staging_bytes += need;
        }
        for (const CopyRange& range : ranges) {
            if (!range.staged) {
                passes.front().ranges.push_back(range);
                continue;
            }
            CopyRange copy = range;
            copy.staged_at += offset_of[range.transform_index];
            passes[pass_of[range.transform_index]].ranges.push_back(copy);
        }
        if (passes.front().ranges.empty()) { passes.erase(passes.begin()); }
    }
    for (Pass& pass : passes) {
        std::sort(pass.ranges.begin(), pass.ranges.end(), [](const CopyRange& a, const CopyRange& b) {
            return a.source == b.source ? a.source_begin < b.source_begin : a.source < b.source;
        });
        for (std::size_t i = 1; i < pass.ranges.size(); ++i) {
            // Two objects may legitimately read the same bytes of a GGUF -- a tied embedding and
            // output head, say -- so overlap is only an error inside the artifact's own payload,
            // whose objects the directory already requires to be disjoint and ordered.
            if (pass.ranges[i].source == pass.ranges[i - 1].source && pass.ranges[i].source == 0 &&
                pass.ranges[i].source_begin < pass.ranges[i - 1].source_end) {
                throw ArtifactError("materialization source ranges overlap");
            }
        }
    }

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    // The read slots are host pinned buffers and are reused by every pass; size them from the
    // whole load rather than from one pass, so a small final wave still reads in big requests.
    std::uint64_t aligned_read_bytes = 0;
    for (const Pass& pass : passes) {
        std::vector<ReadSpan> spans;
        for (const CopyRange& range : pass.ranges) {
            const std::uint64_t begin = align_down(range.source_begin, alignment);
            if (spans.empty() || spans.back().source != range.source ||
                begin > align_up(spans.back().end, alignment,
                                 "artifact direct I/O span overflows u64")) {
                spans.push_back(ReadSpan{range.source, begin, range.source_end});
            } else {
                spans.back().end = std::max(spans.back().end, range.source_end);
            }
        }
        for (const ReadSpan& span : spans) {
            aligned_read_bytes = checked_add(
                aligned_read_bytes,
                align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
                "artifact direct I/O byte count overflows u64");
        }
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
    std::uint64_t peak_staging = 0;
    for (const Pass& pass : passes) { peak_staging = std::max(peak_staging, pass.staging_bytes); }
    out.stats_.peak_staging_bytes =
        static_cast<std::uint64_t>(slot_bytes) * slot_count + peak_staging;

    const auto start_time = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    /// One pass: read every range it owns, in file order, through the slot ring.
    const auto read_pass = [&](const std::vector<CopyRange>& pass_ranges) {
        std::vector<ReadSpan> read_spans;
        read_spans.reserve(pass_ranges.size());
        for (const CopyRange& range : pass_ranges) {
            const std::uint64_t begin = align_down(range.source_begin, alignment);
            if (read_spans.empty() || read_spans.back().source != range.source ||
                begin > align_up(read_spans.back().end, alignment,
                                 "artifact direct I/O span overflows u64")) {
                read_spans.push_back(ReadSpan{range.source, begin, range.source_end});
            } else {
                read_spans.back().end = std::max(read_spans.back().end, range.source_end);
            }
        }
        std::size_t next_slot  = 0;
        std::size_t next_range = 0;
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
                out.stats_.file_bytes = checked_add(out.stats_.file_bytes, bytes_read,
                                                    "artifact read bytes overflow u64");
                const std::uint64_t chunk_end =
                    checked_add(source, bytes_read, "artifact direct I/O result overflows u64");

                // Offsets only order within a file, so both walks are gated on the source too:
                // an external file's small offsets must not match ranges left in the artifact's.
                while (next_range < pass_ranges.size() &&
                       (pass_ranges[next_range].source < span.source ||
                        (pass_ranges[next_range].source == span.source &&
                         pass_ranges[next_range].source_end <= source))) {
                    ++next_range;
                }
                std::size_t range_index = next_range;
                std::size_t resume      = pass_ranges.size();
                while (range_index < pass_ranges.size() &&
                       pass_ranges[range_index].source == span.source &&
                       pass_ranges[range_index].source_begin < chunk_end) {
                    const CopyRange& range         = pass_ranges[range_index];
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
                    // Every range overlapping this chunk takes its slice, including ones that
                    // reach past it: two objects may read the same source bytes -- a tied
                    // embedding and output head do -- and stopping at the first unfinished range
                    // would leave the second one's earlier chunks uncopied.
                    if (range.source_end > chunk_end && resume == pass_ranges.size()) {
                        resume = range_index;
                    }
                    ++range_index;
                }
                // Resume at the first range this chunk did not finish. Ranges after it that the
                // chunk did complete are skipped by the guard above on the next pass, because
                // their source_end is behind the next chunk's start.
                next_range = std::min(resume, range_index);
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
        if (next_range != pass_ranges.size()) {
            throw ArtifactError("direct materialization did not consume every range of a pass (" +
                                std::to_string(next_range) + " of " +
                                std::to_string(pass_ranges.size()) + ")");
        }
    };

    for (Pass& pass : passes) {
        // The wave's scratch, if it has one: the reads land here and the rearranging kernels
        // write the real allocations, after which it goes back before the next wave asks.
        std::unique_ptr<DeviceArena> staging;
        if (pass.staging_bytes != 0) {
            staging = std::make_unique<DeviceArena>(static_cast<std::size_t>(pass.staging_bytes));
            auto* base = static_cast<std::byte*>(
                staging->alloc_bytes(static_cast<std::size_t>(pass.staging_bytes), 256).data);
            for (CopyRange& range : pass.ranges) { range.destination = base + range.staged_at; }
            for (const std::size_t t : pass.transforms) {
                transforms[t].source = reinterpret_cast<std::uint64_t>(base) +
                                       transforms[t].source; // source holds the wave offset
            }
        }
        read_pass(pass.ranges);
        for (const std::size_t t : pass.transforms) {
            const PendingTransform& pending = transforms[t];
            switch (pending.transform) {
            case PayloadTransform::Q8ToW8RowSplit:
                ops::detail::ggml::q8_0_to_w8_rowsplit_launch(
                    reinterpret_cast<const void*>(pending.source), pending.destination,
                    pending.rows, pending.columns, static_cast<std::size_t>(pending.stored),
                    pending.group_map, device.load_stream);
                break;
            case PayloadTransform::None:
                throw ArtifactError("a pending transform must name one");
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
        staging.reset();
    }
    group_maps.reset();
    if (copied != total) {
        throw ArtifactError("direct materialization did not cover every tensor byte (copied " +
                            std::to_string(copied) + " of " + std::to_string(total) + ")");
    }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

} // namespace sinfer::artifact
