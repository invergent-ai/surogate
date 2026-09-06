#include "family/impl/load/host_bank.h"
#include "core/numa.h"

#include "api/ops/cpu_expert_compute.h"
#include "api/ops/expert_slot_cache.h"
#include "ops/linear/ggml/ggml_host_decode.h"
#include "core/device.h"

#include <string>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <future>
#include <thread>

#include <sys/mman.h>
#if defined(__linux__) && __has_include(<numa.h>) && __has_include(<numaif.h>)
#define SINFER_HOST_BANK_NUMA 1
#include <numa.h>
#include <numaif.h>
#endif

namespace sinfer::family {

namespace {

/// Bytes an object occupies once assembled, however it is stored.
std::size_t object_bytes(const HostObjectPlan& plan) {
    if (plan.q4_rows != 0) {
        // Requantised on the way in: the object is the Q4 planes, whatever it came from.
        return ops::q4_bank_planes(plan.q4_rows, plan.q4_k).total_bytes;
    }
    if (plan.decode_rows != 0) {
        // Decoded on the way in: the object is the W8 planes, not the blocks it came from.
        return static_cast<std::size_t>(plan.decode_rows) * plan.decode_k +
               static_cast<std::size_t>(plan.decode_rows) * (plan.decode_k / 32) * 2;
    }
    if (!plan.parts.empty()) {
        std::size_t total = 0;
        for (const auto& part : plan.parts) { total += part.size(); }
        return total;
    }
    return plan.payload.size();
}

} // namespace


std::size_t HostBankPlan::total_bytes() const noexcept {
    std::size_t total = 0;
    for (const auto& object : objects) { total += object_bytes(object); }
    return total;
}

HostBank::HostBank(const HostBankPlan& plan) {
    objects_.reserve(plan.objects.size());
    // The bank is filled object by object, and each one is a whole layer's routed experts, so
    // the byte count moves in steps a reader can see. Reported before the first copy as well,
    // so the phase appears the moment it starts rather than when its first layer lands.
    const std::uint64_t planned = plan.total_bytes();
    const auto report = [&](std::uint64_t done) {
        if (plan.progress.callback) { plan.progress.callback("expert bank", done, planned); }
    };
    report(0);
    for (const auto& source : plan.objects) {
        const bool q4 = source.q4_rows > 0;
        const ops::Q4BankPlanes q4_planes =
            q4 ? ops::q4_bank_planes(source.q4_rows, source.q4_k) : ops::Q4BankPlanes{};
        HostObject object;
        object.bytes = q4 ? q4_planes.total_bytes : object_bytes(source);
        object.name  = source.name;
        if (object.bytes == 0 || (source.payload.empty() && source.parts.empty())) {
            throw std::invalid_argument("host bank object " + source.name + " is empty");
        }
        // Every host expert thread reads every expert, so the bank belongs across the nodes
        // rather than on one of them (core/numa.h). The policy has to be in place before the
        // pages exist: pinned pages cannot be moved afterwards.
        //
        // Allocation is mmap -> mbind(interleave) -> first-touch from many threads ->
        // cudaHostRegister. cudaHostAlloc pins at 1.8 GB/s and does not parallelise (measured:
        // 4 concurrent allocations take exactly as long as one — the kernel serialises them);
        // faulting the pages from 16 threads and registering the populated region runs at
        // ~10 GB/s. mbind is used rather than the caller's set_mempolicy scope because the
        // touch workers' faults, not this thread's, place the pages. cudaHostAlloc stays as
        // the fallback when any step refuses.
        object.host = nullptr;
        {
            void* mem = ::mmap(nullptr, object.bytes, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mem != MAP_FAILED) {
#if defined(SINFER_HOST_BANK_NUMA)
                if (numa_available() >= 0 && numa_num_configured_nodes() > 1) {
                    (void)::mbind(mem, object.bytes, MPOL_INTERLEAVE, numa_all_nodes_ptr->maskp,
                                  numa_all_nodes_ptr->size + 1, 0);
                }
#endif
                {
                    const std::size_t touch_workers = std::max<std::size_t>(
                        16, std::thread::hardware_concurrency() / 2);
                    std::vector<std::thread> touchers;
                    touchers.reserve(touch_workers);
                    const std::size_t chunk =
                        (object.bytes + touch_workers - 1) / touch_workers;
                    for (std::size_t w = 0; w < touch_workers; ++w) {
                        const std::size_t begin = w * chunk;
                        if (begin >= object.bytes) { break; }
                        const std::size_t count = std::min(chunk, object.bytes - begin);
                        touchers.emplace_back([mem, begin, count] {
                            std::memset(static_cast<std::byte*>(mem) + begin, 0, count);
                        });
                    }
                    for (auto& toucher : touchers) { toucher.join(); }
                }
                if (cudaHostRegister(mem, object.bytes,
                                     cudaHostRegisterMapped | cudaHostRegisterPortable) ==
                    cudaSuccess) {
                    object.host       = mem;
                    object.registered = true;
                } else {
                    (void)cudaGetLastError();
                    (void)::munmap(mem, object.bytes);
                }
            }
        }
        if (object.host == nullptr) {
            const ScopedMemoryPolicy placement = ScopedMemoryPolicy::interleaved();
            CUDA_CHECK(cudaHostAlloc(&object.host, object.bytes,
                                     cudaHostAllocMapped | cudaHostAllocPortable));
        }
        void* device = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&device, object.host, 0));
        object.device = device;
        // The artifact mapping is read-ahead-free (it serves random object reads), so a
        // straight copy would fault it in one page at a time; ask for sequential readahead over
        // the whole object first, then copy it with several threads.
        {
            const auto address = reinterpret_cast<std::uintptr_t>(source.payload.data());
            const std::uintptr_t page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
            const std::uintptr_t start = address & ~(page - 1);
            const std::size_t length =
                static_cast<std::size_t>(address + source.payload.size() - start);
            (void)madvise(reinterpret_cast<void*>(start), length, MADV_SEQUENTIAL);
            (void)madvise(reinterpret_cast<void*>(start), length, MADV_WILLNEED);
        }
        std::vector<std::thread> threads;
        if (source.decode_rows != 0) {
            // Decode while copying. The blocks arrive as stretches of the GGUF in row order,
            // each a whole number of rows; a worker owns a contiguous row range and walks the
            // stretches to find its rows. The planes it writes are the ones a converted
            // artifact stores, so nothing downstream learns the bank was decoded here.
            const std::int64_t rows   = source.decode_rows;
            const std::int32_t k      = source.decode_k;
            const std::int64_t row_in = ops::ggml_row_bytes(source.decode_type, k);
            if (row_in <= 0) {
                throw std::invalid_argument("host bank object " + source.name +
                                            " is not a block format this build decodes");
            }
            std::vector<std::span<const std::byte>> stretches = source.parts;
            if (stretches.empty()) { stretches.push_back(source.payload); }
            // Row r begins at byte r * row_in of the concatenation; a stretch that is not a
            // whole number of rows would let a row straddle two, which no source produces.
            std::vector<std::int64_t> first_row;
            first_row.reserve(stretches.size() + 1);
            std::int64_t running = 0;
            for (const auto& part : stretches) {
                if (part.size() % static_cast<std::size_t>(row_in) != 0) {
                    throw std::logic_error("host bank object " + source.name +
                                           " has a run that is not whole rows");
                }
                first_row.push_back(running);
                running += static_cast<std::int64_t>(part.size() / static_cast<std::size_t>(row_in));
            }
            first_row.push_back(running);
            if (running != rows) {
                throw std::logic_error("host bank object " + source.name +
                                       " has " + std::to_string(running) + " rows, expected " +
                                       std::to_string(rows));
            }
            // Straight to W8 planes, or -- when the bank is the Q4 one -- through a row of W8
            // scratch into the packed nibble planes, so a GGUF's blocks reach Q4G32AM by the
            // same two steps a converted artifact's W8 does, one row at a time.
            auto* codes  = reinterpret_cast<std::int8_t*>(object.host);
            auto* scales = reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(object.host) +
                                                            static_cast<std::size_t>(rows) * k);
            auto* q4_dst        = static_cast<std::byte*>(object.host);
            auto* q4_dst_codes  = reinterpret_cast<std::uint8_t*>(q4_dst);
            auto* q4_dst_scales = reinterpret_cast<std::uint16_t*>(q4_dst + q4_planes.scales_offset);
            auto* q4_dst_mins   = reinterpret_cast<std::uint16_t*>(q4_dst + q4_planes.mins_offset);
            const std::size_t workers = std::max<std::size_t>(16, std::thread::hardware_concurrency());
            const std::int64_t chunk  = (rows + static_cast<std::int64_t>(workers) - 1) /
                                       static_cast<std::int64_t>(workers);
            for (std::size_t w = 0; w < workers; ++w) {
                const std::int64_t begin = static_cast<std::int64_t>(w) * chunk;
                if (begin >= rows) { break; }
                const std::int64_t end = std::min(rows, begin + chunk);
                threads.emplace_back([=, stretches, first_row, &source] {
                    std::vector<std::int8_t> row_codes(q4 ? static_cast<std::size_t>(k) : 0);
                    std::vector<std::uint16_t> row_scales(q4 ? static_cast<std::size_t>(k / 32) : 0);
                    std::size_t part = 0;
                    while (part + 1 < first_row.size() && first_row[part + 1] <= begin) { ++part; }
                    for (std::int64_t r = begin; r < end; ++r) {
                        while (first_row[part + 1] <= r) { ++part; }
                        const std::byte* blocks =
                            stretches[part].data() +
                            static_cast<std::size_t>(r - first_row[part]) * static_cast<std::size_t>(row_in);
                        std::int8_t* out_codes    = q4 ? row_codes.data() : codes + r * k;
                        std::uint16_t* out_scales = q4 ? row_scales.data() : scales + r * (k / 32);
                        if (!ops::ggml_decode_row_w8(source.decode_type, blocks, k, out_codes,
                                                     out_scales)) {
                            throw std::runtime_error("host bank object " + source.name +
                                                     " failed to decode a row");
                        }
                        if (q4) {
                            const std::int64_t group0 = r * (k / 32);
                            ops::requantise_w8_expert_groups_to_q4(
                                out_codes, out_scales, k / 32, q4_dst_codes + group0 * 16,
                                q4_dst_scales + group0, q4_dst_mins + group0);
                        }
                    }
                });
            }
        } else if (q4) {
            // Requantise while copying: every worker owns a contiguous group range of the
            // parallel (row, k-group) order, reading the W8 codes and scales planes and
            // writing the packed nibbles plus the FP16 scale/min planes.
            const auto* src_codes = reinterpret_cast<const std::int8_t*>(source.payload.data());
            const auto* src_scales = reinterpret_cast<const std::uint16_t*>(
                source.payload.data() + source.q4_w8_scale_offset);
            auto* dst        = static_cast<std::byte*>(object.host);
            auto* dst_codes  = reinterpret_cast<std::uint8_t*>(dst);
            auto* dst_scales = reinterpret_cast<std::uint16_t*>(dst + q4_planes.scales_offset);
            auto* dst_mins   = reinterpret_cast<std::uint16_t*>(dst + q4_planes.mins_offset);
            const auto groups = static_cast<std::int64_t>(q4_planes.groups);
            const std::size_t workers =
                std::max<std::size_t>(16, std::thread::hardware_concurrency() / 2);
            const std::int64_t chunk = (groups + static_cast<std::int64_t>(workers) - 1) /
                                       static_cast<std::int64_t>(workers);
            for (std::size_t w = 0; w < workers; ++w) {
                const std::int64_t begin = static_cast<std::int64_t>(w) * chunk;
                if (begin >= groups) { break; }
                const std::int64_t count = std::min(chunk, groups - begin);
                threads.emplace_back([=] {
                    ops::requantise_w8_expert_groups_to_q4(
                        src_codes + begin * 32, src_scales + begin, count, dst_codes + begin * 16,
                        dst_scales + begin, dst_mins + begin);
                });
            }
        } else if (!source.parts.empty()) {
            // Read in place: the bytes are stretches of the GGUF, so the copy walks them in
            // order. One worker per part keeps the same threaded fill; the parts of a fused
            // expert are large and few, not scattered singletons.
            std::size_t offset = 0;
            for (const std::span<const std::byte>& part : source.parts) {
                threads.emplace_back([dst = static_cast<std::byte*>(object.host) + offset, part] {
                    std::memcpy(dst, part.data(), part.size());
                });
                offset += part.size();
                if (threads.size() >= 32) {
                    for (auto& thread : threads) { thread.join(); }
                    threads.clear();
                }
            }
        } else {
            const std::size_t workers = 16;
            const std::size_t chunk   = (object.bytes + workers - 1) / workers;
            for (std::size_t w = 0; w < workers; ++w) {
                const std::size_t begin = w * chunk;
                if (begin >= object.bytes) { break; }
                const std::size_t count = std::min(chunk, object.bytes - begin);
                threads.emplace_back([&, begin, count] {
                    std::memcpy(static_cast<std::byte*>(object.host) + begin,
                                source.payload.data() + begin, count);
                });
            }
        }
        for (auto& thread : threads) { thread.join(); }
        total_bytes_ += object.bytes;
        objects_.emplace_back(source.handle.index, object);
        report(total_bytes_);
    }
    report(planned);
}

HostBank::~HostBank() {
    for (auto& [index, object] : objects_) {
        if (object.host == nullptr) { continue; }
        if (object.registered) {
            (void)cudaHostUnregister(object.host);
            (void)::munmap(object.host, object.bytes);
        } else {
            (void)cudaFreeHost(object.host);
        }
    }
}

const HostObject& HostBank::object(artifact::ObjectHandle handle) const {
    for (const auto& [index, object] : objects_) {
        if (index == handle.index) { return object; }
    }
    throw std::out_of_range("host bank has no object for this handle");
}


std::shared_ptr<HostBank> HostBank::shared(const HostBankPlan& plan) {
    // The mutex guards only the map. Construction — pinning tens of GB and requantising the
    // planes — happens outside it, through a per-key future: callers with the same key share
    // one build, callers with different keys (the pipeline's stages, one bank per layer range)
    // build concurrently. Holding the lock across construction serialised the stages' bank
    // builds and was most of the 8-card startup.
    static std::mutex mutex;
    static std::unordered_map<std::string, std::weak_ptr<HostBank>> banks;
    static std::unordered_map<std::string, std::shared_future<std::shared_ptr<HostBank>>> building;
    std::string key;
    for (const auto& source : plan.objects) {
        key += source.name;
        key += ':';
        key += std::to_string(source.payload.size());
        if (source.q4_rows > 0) { key += ":q4"; }
        key += ';';
    }
    std::shared_future<std::shared_ptr<HostBank>> pending;
    std::promise<std::shared_ptr<HostBank>> promise;
    bool builder = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (auto found = banks.find(key); found != banks.end()) {
            if (auto live = found->second.lock()) { return live; }
        }
        if (auto found = building.find(key); found != building.end()) {
            pending = found->second;
        } else {
            pending       = promise.get_future().share();
            building[key] = pending;
            builder       = true;
        }
    }
    if (!builder) {
        auto bank = pending.get();
        if (bank != nullptr) { return bank; }
        // The builder failed; fall through and try to build it ourselves.
        return HostBank::shared(plan);
    }
    std::shared_ptr<HostBank> bank;
    try {
        bank = std::make_shared<HostBank>(plan);
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            building.erase(key);
        }
        promise.set_value(nullptr); // waiters retry rather than inherit our exception
        throw;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        banks[key] = bank;
        building.erase(key);
    }
    promise.set_value(bank);
    return bank;
}

// -------------------------------------------------------------------------------------------
// Binding an object into the bank instead of onto the device
// -------------------------------------------------------------------------------------------

HostObjectPlan host_plan(artifact::Binder& binder, artifact::ObjectHandle handle,
                         const std::string& name) {
    const auto runs = binder.runs(handle);
    HostObjectPlan plan{handle, {}, name};
    if (runs.size() == 1) {
        plan.payload = binder.payload(handle).data;
        return plan;
    }
    plan.parts.reserve(runs.size());
    for (const artifact::PayloadRun& run : runs) { plan.parts.push_back(binder.run_span(run)); }
    return plan;
}

artifact::ObjectHandle host_tensor(artifact::Binder& binder, HostBankPlan& bank,
                                   const std::string& name, artifact::NumericFormat format,
                                   std::initializer_list<std::uint64_t> shape) {
    const artifact::ObjectHandle handle =
        artifact::bind_tensor(binder, name, format, shape, artifact::TensorPlacement::ValidateOnly);
    bank.objects.push_back(host_plan(binder, handle, name));
    return handle;
}

artifact::LinearBinding host_linear(artifact::Binder& binder, HostBankPlan& bank,
                                    const std::string& name, std::int32_t rows,
                                    std::int32_t columns) {
    const artifact::LinearBinding binding =
        artifact::bind_linear(binder, name, rows, columns, artifact::TensorPlacement::ValidateOnly);
    bank.objects.push_back(host_plan(binder, binding.object, name));
    return binding;
}

// -------------------------------------------------------------------------------------------
// Reading the bank from a kernel
// -------------------------------------------------------------------------------------------

Weight host_ggml_weight(const HostObject& object, artifact::NumericFormat format, std::int32_t rows,
                        std::int32_t columns) {
    const auto values = static_cast<std::int32_t>(artifact::ggml_block_values(format));
    if (columns % values != 0) {
        throw std::logic_error("host bank object " + object.name +
                               " has a width that is not a whole number of blocks");
    }
    const auto* bytes = static_cast<const std::byte*>(object.device);
    Weight out{};
    out.payload       = bytes;
    out.payload_bytes = object.bytes;
    out.qtype         = artifact::qtype_for(format);
    out.layout        = QuantLayout::GgmlBlocks;
    out.qdata         = bytes;
    // A GGML block carries its own scale, so there is no separate plane and no padding: the
    // stored width is the logical width.
    out.scales          = nullptr;
    out.group_size      = static_cast<std::uint32_t>(values);
    out.group           = values;
    out.scale_dtype     = DType::FP16;
    out.ndim            = 2;
    out.n               = rows;
    out.k               = columns;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.shape[2]        = 1;
    out.shape[3]        = 1;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.padded_shape[2] = 1;
    out.padded_shape[3] = 1;
    return out;
}

Weight host_w8_weight(const HostObject& object, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::RowSplitGeometry geometry =
        artifact::row_split_geometry(artifact::NumericFormat::W8G32_F16S, shape);
    if (geometry.encoded_bytes != object.bytes) {
        throw std::logic_error("host bank object " + object.name + " has an unexpected size");
    }
    const auto* bytes = static_cast<const std::byte*>(object.device);
    Weight out{};
    out.payload          = bytes;
    out.payload_bytes    = geometry.encoded_bytes;
    out.high_plane_bytes = geometry.high_plane_bytes;
    out.qtype            = QType::W8G32_F16S;
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    out.qdata            = bytes;
    out.qhigh            = nullptr;
    out.scales           = bytes + geometry.scale_plane_offset;
    out.n                = rows;
    out.k                = columns;
    out.group            = static_cast<std::int32_t>(geometry.group_size);
    out.scale_dtype      = DType::FP16;
    out.ndim             = 2;
    out.shape[0]         = rows;
    out.shape[1]         = columns;
    out.padded_shape[0]  = rows;
    out.padded_shape[1]  = static_cast<std::int32_t>(geometry.padded_columns);
    return out;
}

} // namespace sinfer::family
