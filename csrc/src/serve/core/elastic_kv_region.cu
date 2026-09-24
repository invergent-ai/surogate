#include "core/elastic_kv_region.h"

#include "core/device.h"
#include "core/device_footprint.h"
#include "core/device_memory_error.h"
#include "core/engine_context.h"
#include "core/sleep.h"

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sinfer {
namespace {

void driver_check(CUresult result, const char* what) {
    if (result == CUDA_SUCCESS) { return; }
    const char* text = nullptr;
    cuGetErrorString(result, &text);
    throw std::runtime_error(std::string(what) + " failed: " +
                             (text != nullptr ? text : "unknown CUDA driver error"));
}

CUmemAllocationProp allocation_prop(int device) {
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id         = device;
    return prop;
}

// The process-wide commitment ledger: each region's guaranteed cap, its entitlement and what
// it has mapped, all in bytes. Outstanding is what it may still map that is spoken for.
struct Commitment {
    int device           = 0;
    std::size_t cap      = 0;
    std::size_t entitled = 0;
    std::size_t mapped   = 0;
    ElasticKvRegion* region = nullptr;

    [[nodiscard]] std::size_t outstanding() const noexcept {
        const std::size_t spoken_for = std::max(cap, entitled);
        return spoken_for > mapped ? spoken_for - mapped : 0;
    }
};
std::mutex& ledger_mutex() {
    static std::mutex instance;
    return instance;
}
std::map<const void*, Commitment>& ledger() {
    static std::map<const void*, Commitment> instance;
    return instance;
}
// Pressure: the last time the gate refused a region on each device.
using PressureClock = std::chrono::steady_clock;
constexpr auto kPressureHold = std::chrono::seconds(2);
std::map<int, PressureClock::time_point>& pressure_marks() {
    static std::map<int, PressureClock::time_point> instance;
    return instance;
}
// Caller holds the ledger mutex.
std::size_t outstanding_on(int device, const void* except) noexcept {
    std::size_t bytes = 0;
    for (const auto& [key, entry] : ledger()) {
        if (entry.device == device && key != except) { bytes += entry.outstanding(); }
    }
    return bytes;
}
// Free memory as this process may use it: under --gpu-memory-limit-mib, its remaining budget.
std::size_t region_device_free_bytes(int device) noexcept { return device_budget_free_bytes(device); }

} // namespace

std::size_t elastic_kv_unmapped_commitment(int device) noexcept {
    const std::lock_guard<std::mutex> lock(ledger_mutex());
    return outstanding_on(device, nullptr);
}

std::size_t elastic_kv_mapped_bytes(int device) noexcept {
    const std::lock_guard<std::mutex> lock(ledger_mutex());
    std::size_t bytes = 0;
    for (const auto& [key, entry] : ledger()) {
        if (entry.device == device) { bytes += entry.mapped; }
    }
    return bytes;
}

bool elastic_kv_device_pressure(int device) noexcept {
    const std::lock_guard<std::mutex> lock(ledger_mutex());
    const auto it = pressure_marks().find(device);
    return it != pressure_marks().end() && PressureClock::now() - it->second < kPressureHold;
}

struct ElasticKvRegion::Impl {
    struct Granule {
        bool mapped              = false;
        std::uint32_t used_pages = 0;
        std::uint64_t generation = 0; ///< bumped on every acquire; a stale unmap job skips
        std::vector<CUmemGenericAllocationHandle> handles; ///< one per plane while mapped
        std::vector<void*> backup;                         ///< pinned host, one per plane, while asleep
    };
    struct Job {
        enum class Kind { Trim, Unmap } kind = Kind::Trim;
        std::uint32_t granule                = 0;
        std::uint64_t generation             = 0;
        cudaEvent_t fence                    = nullptr;
        bool forced                          = false; ///< a release: past the reserve threshold
    };

    ElasticKvRegionSpec spec;
    CUdeviceptr va            = 0;
    std::size_t granule_bytes = 0;
    std::uint32_t granules    = 0;
    std::vector<Granule> state;
    std::size_t mapped = 0; ///< bytes currently mapped
    bool asleep        = false;

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<Job> jobs;
    bool stopping = false;
    bool busy     = false;
    bool release_requested = false; ///< another region on the device is short
    std::thread worker;

    // ---- device mapping (caller holds mutex) ----

    /// Fault injection for tests (SUROGATE_SERVE_FAULT_KV_OOM=N): the N-th on-demand map in
    /// this process fails once as if the device were out of memory. Not for production use.
    static bool fault_injected_oom() {
        static const long target = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_FAULT_KV_OOM");
            return raw == nullptr ? 0L : std::strtol(raw, nullptr, 10);
        }();
        static std::atomic<long> maps{0};
        const bool inject = target > 0 && maps.fetch_add(1) + 1 == target;
        if (inject) {
            std::fprintf(stderr, "elastic kv: injected out-of-memory (SUROGATE_SERVE_FAULT_KV_OOM=%ld)\n", target);
        }
        return inject;
    }

    /// Give back up to `at_most` mapped granules no page uses (the reserve, and granules waiting
    /// for their unmap job), highest first: the pool hands out low page ids first, so those are
    /// the ones demand reaches last. The engine's stream is drained first, so nothing still in
    /// flight can touch them; inside a capture that is impossible, and nothing is released.
    /// Returns the granules released. Executor thread only, with the mutex held.
    std::uint32_t reclaim_free_granules(std::uint32_t at_most = ~std::uint32_t{0}) {
        // A null fence stream is the legacy stream, which release_page fences on.
        cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(spec.fence_stream, &capturing) != cudaSuccess ||
            capturing != cudaStreamCaptureStatusNone) {
            (void)cudaGetLastError();
            return 0;
        }
        if (cudaStreamSynchronize(spec.fence_stream) != cudaSuccess) { return 0; }
        std::uint32_t released = 0;
        for (std::uint32_t g = granules; g-- > 0 && released < at_most;) {
            if (state[g].mapped && state[g].used_pages == 0) {
                unmap_granule(g);
                ++released;
            }
        }
        return released;
    }

    std::uint32_t mapped_granule_count() const noexcept {
        std::uint32_t count = 0;
        for (const Granule& granule : state) { count += granule.mapped ? 1U : 0U; }
        return count;
    }

    /// Granules the region may hold mapped at once: its committed cap, or under overcommit the
    /// larger entitlement the device gate granted. Automatic KV sizing plans the device around
    /// that many bytes and leaves the rest to the engine's other allocations, so the reserve and
    /// the emptied granules kept for reuse have to fit inside it too. Mapping past it is what
    /// filled the device in the 2026-09-24 fuzz run: 951 pages in use, 1,072 mapped, a cap of
    /// 1,010, and the next map failed.
    std::uint32_t mapped_granule_limit() const noexcept {
        std::size_t bytes = 0;
        {
            const std::lock_guard<std::mutex> ledger_lock(ledger_mutex());
            const auto it = ledger().find(this);
            if (it == ledger().end()) { return granules; }
            bytes = std::max(it->second.cap, it->second.entitled);
        }
        if (granule_bytes == 0) { return granules; }
        return static_cast<std::uint32_t>(
            std::min<std::size_t>(granules, (bytes + granule_bytes - 1) / granule_bytes));
    }

    std::size_t device_free_bytes() const noexcept {
        return spec.free_bytes_probe != nullptr ? spec.free_bytes_probe()
                                                : region_device_free_bytes(spec.device);
    }

    /// Before an on-demand map at the limit: trade a granule no page uses for the one demand
    /// needs. With none to trade -- every mapped granule holds a page, and the pool's lowest free
    /// page lies in an unmapped one -- the map goes past the plan only while the device keeps
    /// the headroom free; otherwise it is refused as out of memory, which the executor recovers
    /// from, instead of taking memory a workspace or a graph capture was promised.
    void make_room_for_demand() {
        const std::uint32_t limit = mapped_granule_limit();
        const std::uint32_t count = mapped_granule_count();
        if (count < limit) { return; }
        if (reclaim_free_granules(count - limit + 1) != 0 && mapped_granule_count() < limit) {
            return;
        }
        const std::size_t free_bytes = device_free_bytes();
        if (free_bytes >= spec.headroom_bytes + granule_bytes) { return; }
        throw DeviceOutOfMemory(
            "elastic kv: the KV cache is at its planned size (" + std::to_string(count) +
            " granules mapped, " + std::to_string((count * granule_bytes) >> 20) +
            " MiB), every mapped granule holds pages, and mapping another would leave less than " +
            std::to_string(spec.headroom_bytes >> 20) + " MiB free on the device (" +
            std::to_string(free_bytes >> 20) + " MiB free)");
    }

    /// Map granule `g` on every plane. `on_demand` is the executor's own call for pages it is
    /// handing out: when the device is out of memory it first reclaims the free granules and
    /// tries once more, and a failure that remains is a `DeviceOutOfMemory` the executor can
    /// recover from. The worker's reserve refill never reclaims (it would only trade one free
    /// granule for another).
    void map_granule(std::uint32_t g, bool on_demand = false) {
        Granule& granule = state[g];
        if (granule.mapped) { return; }
        if (on_demand) { make_room_for_demand(); }
        const CUmemAllocationProp prop = allocation_prop(spec.device);
        CUmemAccessDesc access         = {};
        access.location                = prop.location;
        access.flags                   = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        granule.handles.assign(spec.planes.size(), {});
        for (std::size_t p = 0; p < spec.planes.size(); ++p) {
            const std::size_t span = spec.planes[p].page_bytes * spec.granule_pages;
            const CUdeviceptr at   = va + spec.planes[p].offset + span * g;
            CUmemGenericAllocationHandle handle{};
            const bool injected = on_demand && p == 0 && fault_injected_oom();
            // An injected failure stands for a card that stays full: no reclaim, no retry.
            CUresult created = injected ? CUDA_ERROR_OUT_OF_MEMORY : cuMemCreate(&handle, span, &prop, 0);
            if (created == CUDA_ERROR_OUT_OF_MEMORY && on_demand && !injected) {
                const std::uint32_t released = reclaim_free_granules();
                if (released != 0) {
                    std::fprintf(stderr,
                                 "elastic kv: out of device memory mapping granule %u; released "
                                 "%u free granules (%zu MiB) and retrying\n",
                                 g, released, (released * granule_bytes) >> 20);
                    created = cuMemCreate(&handle, span, &prop, 0);
                }
            }
            if (created != CUDA_SUCCESS) {
                for (std::size_t q = 0; q < p; ++q) { unmap_plane(g, q); }
                granule.handles.clear();
                if (created == CUDA_ERROR_OUT_OF_MEMORY) {
                    std::size_t free_bytes = 0, total_bytes = 0;
                    (void)cudaMemGetInfo(&free_bytes, &total_bytes);
                    throw DeviceOutOfMemory("cuMemCreate failed: out of memory mapping " +
                                            std::to_string(span >> 20) + " MiB of KV cache (" +
                                            std::to_string(free_bytes >> 20) +
                                            " MiB free on the device)");
                }
                driver_check(created, "cuMemCreate");
            }
            CUresult mapped_result = cuMemMap(at, span, 0, handle, 0);
            if (mapped_result == CUDA_SUCCESS) {
                mapped_result = cuMemSetAccess(at, span, &access, 1);
                if (mapped_result != CUDA_SUCCESS) { (void)cuMemUnmap(at, span); }
            }
            if (mapped_result != CUDA_SUCCESS) {
                (void)cuMemRelease(handle);
                for (std::size_t q = 0; q < p; ++q) { unmap_plane(g, q); }
                granule.handles.clear();
                if (mapped_result == CUDA_ERROR_OUT_OF_MEMORY) {
                    throw DeviceOutOfMemory("cuMemMap failed: out of memory mapping KV cache");
                }
                driver_check(mapped_result, "cuMemMap");
            }
            granule.handles[p] = handle;
        }
        granule.mapped = true;
        mapped += granule_bytes;
        const std::lock_guard<std::mutex> ledger_lock(ledger_mutex());
        ledger()[this].mapped = mapped;
    }

    void unmap_plane(std::uint32_t g, std::size_t p) {
        const std::size_t span = spec.planes[p].page_bytes * spec.granule_pages;
        const CUdeviceptr at   = va + spec.planes[p].offset + span * g;
        (void)cuMemUnmap(at, span);
        (void)cuMemRelease(state[g].handles[p]);
    }

    void unmap_granule(std::uint32_t g) {
        Granule& granule = state[g];
        if (!granule.mapped) { return; }
        for (std::size_t p = 0; p < spec.planes.size(); ++p) { unmap_plane(g, p); }
        granule.handles.clear();
        granule.mapped = false;
        mapped -= granule_bytes;
        const std::lock_guard<std::mutex> ledger_lock(ledger_mutex());
        ledger()[this].mapped = mapped;
    }

    // Trim only well above the reserve: a granule freed by one request and needed by the next
    // must not go through an unmap and a map (96 driver calls on the 27B) in between.
    std::uint32_t trim_threshold() const noexcept {
        if (spec.reserve_granules == 0) { return 0; } // no reserve: every empty granule goes back
        if (elastic_kv_device_pressure(spec.device)) { return 0; } // the device is short: no reserve
        return std::max<std::uint32_t>(2 * spec.reserve_granules, spec.reserve_granules + 2);
    }

    std::uint32_t mapped_free_count() const noexcept {
        std::uint32_t count = 0;
        for (const Granule& granule : state) {
            if (granule.mapped && granule.used_pages == 0) { ++count; }
        }
        return count;
    }

    // ---- worker ----

    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            cv.wait(lock, [&] { return stopping || !jobs.empty(); });
            if (stopping) { return; }
            Job job = jobs.front();
            jobs.pop_front();
            busy = true;
            if (job.kind == Job::Kind::Trim) {
                // Map the lowest unmapped granules until the reserve is met. The pool hands out
                // low page ids first, so these are the granules demand reaches next.
                if (!asleep && !elastic_kv_device_pressure(spec.device)) {
                    // Never past the limit: a reserve granule is still device memory.
                    const std::uint32_t limit = mapped_granule_limit();
                    for (std::uint32_t g = 0; g < granules &&
                                              mapped_free_count() < spec.reserve_granules &&
                                              mapped_granule_count() < limit;
                         ++g) {
                        if (!state[g].mapped) {
                            try {
                                map_granule(g);
                            } catch (const std::exception& error) {
                                // Out of VRAM ahead of demand is not an error yet; the executor
                                // maps on demand and reports the failure where it matters.
                                std::fprintf(stderr, "elastic kv: reserve map skipped: %s\n",
                                             error.what());
                                break;
                            }
                        }
                    }
                }
            } else {
                // The fence was recorded when the granule emptied; wait for it off the lock so a
                // busy stream never stalls an acquire, then re-check under the lock.
                lock.unlock();
                if (job.fence != nullptr) {
                    (void)cudaEventSynchronize(job.fence);
                    (void)cudaEventDestroy(job.fence);
                }
                lock.lock();
                Granule& granule = state[job.granule];
                if (!asleep && granule.mapped && granule.used_pages == 0 &&
                    granule.generation == job.generation &&
                    (job.forced || mapped_free_count() > trim_threshold())) {
                    unmap_granule(job.granule);
                }
            }
            busy = false;
            cv.notify_all();
        }
    }

    void post(Job job) {
        jobs.push_back(job);
        cv.notify_all();
    }
};

ElasticKvRegion::ElasticKvRegion(ElasticKvRegionSpec spec) : impl_(std::make_unique<Impl>()) {
    Impl& impl = *impl_;
    impl.spec  = std::move(spec);
    if (impl.spec.bytes == 0 || impl.spec.page_count == 0 || impl.spec.granule_pages == 0 ||
        impl.spec.planes.empty()) {
        throw std::invalid_argument("elastic kv region: empty specification");
    }
    CUDA_CHECK(cudaSetDevice(impl.spec.device));
    CUDA_CHECK(cudaFree(nullptr)); // the driver API needs an initialized context
    const CUmemAllocationProp prop = allocation_prop(impl.spec.device);
    std::size_t granularity        = 0;
    driver_check(
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");
    impl.granules = (impl.spec.page_count + impl.spec.granule_pages - 1) / impl.spec.granule_pages;
    for (const ElasticKvPlane& plane : impl.spec.planes) {
        const std::size_t span = plane.page_bytes * impl.spec.granule_pages;
        if (plane.page_bytes == 0 || plane.offset % granularity != 0 || span % granularity != 0 ||
            plane.offset + span * impl.granules > impl.spec.bytes) {
            throw std::invalid_argument(
                "elastic kv region: a plane is not aligned to the mapping granularity (" +
                std::to_string(granularity) + " bytes) or overruns the span");
        }
        impl.granule_bytes += span;
    }
    if (impl.spec.bytes % granularity != 0) {
        throw std::invalid_argument("elastic kv region: span is not a granularity multiple");
    }
    {
        // Commit the cap against what is free once every other region's unmapped remainder is
        // set aside. The same refusal an arena pool gets at cudaMalloc time, without paying it.
        const std::uint32_t cap_pages = impl.spec.cap_pages != 0 ? impl.spec.cap_pages : impl.spec.page_count;
        const std::size_t cap_granules = (cap_pages + impl.spec.granule_pages - 1) / impl.spec.granule_pages;
        const std::size_t cap_bytes    = cap_granules * impl.granule_bytes;
        const std::size_t free_bytes =
            impl.spec.free_bytes_probe != nullptr ? impl.spec.free_bytes_probe() : region_device_free_bytes(impl.spec.device);
        const std::size_t spoken_for = elastic_kv_unmapped_commitment(impl.spec.device);
        if (cap_bytes > free_bytes - std::min(spoken_for, free_bytes)) {
            throw std::runtime_error(
                "elastic kv region: a cap of " + std::to_string(cap_bytes >> 20) +
                " MiB does not fit beside " + std::to_string(spoken_for >> 20) +
                " MiB already committed to other engines with " + std::to_string(free_bytes >> 20) +
                " MiB free");
        }
        const std::lock_guard<std::mutex> lock(ledger_mutex());
        ledger()[impl_.get()] = Commitment{impl.spec.device, cap_bytes, 0, 0, this};
    }
    driver_check(cuMemAddressReserve(&impl.va, impl.spec.bytes, granularity, 0, 0),
                 "cuMemAddressReserve");
    impl.state.assign(impl.granules, Impl::Granule{});
    impl.worker = std::thread([this] { impl_->run(); });
    {
        const std::lock_guard<std::mutex> lock(impl.mutex);
        impl.post(Impl::Job{Impl::Job::Kind::Trim});
    }
    // Sleep mode sees the region through hooks rather than as a fixed span: only the
    // granules holding pages are backed up, and the footprint is what is mapped.
    sleep_register_sparse(this, SparseRegionHooks{
                                    .owner        = ops::current_ops_owner(),
                                    .device       = impl.spec.device,
                                    .sleep_fn     = [this] { return sleep(); },
                                    .wake_fn      = [this] { return wake(); },
                                    .mapped_bytes = [this] { return mapped_bytes(); },
                                });
}

ElasticKvRegion::~ElasticKvRegion() {
    sleep_unregister_sparse(this);
    Impl& impl = *impl_;
    {
        const std::lock_guard<std::mutex> lock(ledger_mutex());
        ledger().erase(impl_.get());
    }
    {
        const std::lock_guard<std::mutex> lock(impl.mutex);
        impl.stopping = true;
        impl.cv.notify_all();
    }
    if (impl.worker.joinable()) { impl.worker.join(); }
    for (Impl::Job& job : impl.jobs) {
        if (job.fence != nullptr) { (void)cudaEventDestroy(job.fence); }
    }
    for (std::uint32_t g = 0; g < impl.granules; ++g) {
        impl.unmap_granule(g);
        for (void* backup : impl.state[g].backup) {
            if (backup != nullptr) { (void)cudaFreeHost(backup); }
        }
    }
    if (impl.va != 0) { (void)cuMemAddressFree(impl.va, impl.spec.bytes); }
}

void* ElasticKvRegion::base() const noexcept { return reinterpret_cast<void*>(impl_->va); }
std::size_t ElasticKvRegion::bytes() const noexcept { return impl_->spec.bytes; }
std::uint32_t ElasticKvRegion::granule_count() const noexcept { return impl_->granules; }
std::uint32_t ElasticKvRegion::granule_pages() const noexcept { return impl_->spec.granule_pages; }
std::size_t ElasticKvRegion::granule_bytes() const noexcept { return impl_->granule_bytes; }

void ElasticKvRegion::acquire_page(std::int32_t page) {
    Impl& impl = *impl_;
    const ScopedDevice selected(impl.spec.device);
    if (page < 0 || static_cast<std::uint32_t>(page) >= impl.spec.page_count) {
        throw std::out_of_range("elastic kv region: page out of range");
    }
    const std::uint32_t g = static_cast<std::uint32_t>(page) / impl.spec.granule_pages;
    const std::lock_guard<std::mutex> lock(impl.mutex);
    if (impl.asleep) { throw std::logic_error("elastic kv region: acquire while asleep"); }
    Impl::Granule& granule = impl.state[g];
    ++granule.generation;
    if (!granule.mapped) { impl.map_granule(g, /*on_demand=*/true); } // demand outran the reserve
    ++granule.used_pages;
    if (granule.used_pages == 1) { impl.post(Impl::Job{Impl::Job::Kind::Trim}); } // top the reserve up
}

void ElasticKvRegion::release_page(std::int32_t page) noexcept try {
    Impl& impl = *impl_;
    const ScopedDevice selected(impl.spec.device);
    if (page < 0 || static_cast<std::uint32_t>(page) >= impl.spec.page_count) { return; }
    const std::uint32_t g = static_cast<std::uint32_t>(page) / impl.spec.granule_pages;
    const std::lock_guard<std::mutex> lock(impl.mutex);
    Impl::Granule& granule = impl.state[g];
    if (granule.used_pages == 0) { return; }
    if (--granule.used_pages != 0) { return; }
    // Empty. The fence is recorded on the engine's stream now, at the round boundary that freed
    // the page; nothing captured is in progress here, but a capture in flight would swallow the
    // record into the graph, so leave the granule mapped in that case and let a later release
    // retire it.
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if (impl.spec.fence_stream != nullptr &&
        (cudaStreamIsCapturing(impl.spec.fence_stream, &capturing) != cudaSuccess ||
         capturing != cudaStreamCaptureStatusNone)) {
        return;
    }
    cudaEvent_t fence = nullptr;
    if (cudaEventCreateWithFlags(&fence, cudaEventDisableTiming) != cudaSuccess) { return; }
    if (cudaEventRecord(fence, impl.spec.fence_stream) != cudaSuccess) {
        (void)cudaEventDestroy(fence);
        return;
    }
    impl.post(Impl::Job{Impl::Job::Kind::Unmap, g, granule.generation, fence});
} catch (...) {
    // Retirement is best effort, including when the CUDA device is unavailable.
}

std::size_t ElasticKvRegion::mapped_bytes() const noexcept {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->mapped;
}

std::uint32_t ElasticKvRegion::granule_limit() const noexcept { return impl_->mapped_granule_limit(); }

std::vector<std::uint8_t> ElasticKvRegion::mapped_granule_mask() const {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<std::uint8_t> mask(impl_->granules, 0);
    for (std::uint32_t g = 0; g < impl_->granules; ++g) { mask[g] = impl_->state[g].mapped ? 1 : 0; }
    return mask;
}

std::uint32_t ElasticKvRegion::mapped_granules() const noexcept {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    std::uint32_t count = 0;
    for (const Impl::Granule& granule : impl_->state) { count += granule.mapped ? 1U : 0U; }
    return count;
}

bool ElasticKvRegion::try_entitle(std::uint32_t pages) noexcept {
    Impl& impl = *impl_;
    const std::size_t granules = (static_cast<std::size_t>(pages) + impl.spec.granule_pages - 1) /
                                 impl.spec.granule_pages;
    const std::size_t bytes = granules * impl.granule_bytes;
    std::vector<ElasticKvRegion*> to_release;
    {
        const std::lock_guard<std::mutex> lock(ledger_mutex());
        Commitment& mine = ledger()[impl_.get()];
        // Within the guarantee, or not growing: nothing to ask.
        if (bytes <= mine.cap || bytes <= mine.entitled) {
            mine.entitled = bytes;
            return true;
        }
        if (!impl.spec.overcommit) { return false; }
        // Beyond the guarantee: every region's outstanding bytes, this growth on top of what is
        // already mapped here, and two granules of slack for the reserve maps in flight must
        // fit what the device has free right now.
        const std::size_t others = outstanding_on(impl.spec.device, impl_.get());
        const std::size_t growth = bytes > mine.mapped ? bytes - mine.mapped : 0;
        // Two granules cover the reserve maps in flight; the spec's headroom is what the
        // engine keeps for graph captures and workspace growth.
        const std::size_t slack = std::max(2 * impl.granule_bytes, impl.spec.headroom_bytes);
        const std::size_t free_bytes =
            impl.spec.free_bytes_probe != nullptr ? impl.spec.free_bytes_probe() : region_device_free_bytes(impl.spec.device);
        if (others + growth + slack <= free_bytes) {
            mine.entitled = bytes;
            return true;
        }
        pressure_marks()[impl.spec.device] = PressureClock::now();
        for (const auto& [key, entry] : ledger()) {
            if (entry.device == impl.spec.device && key != impl_.get() && entry.region != nullptr) {
                to_release.push_back(entry.region);
            }
        }
    }
    // Off the ledger lock: each region takes its own mutex to post the job.
    for (ElasticKvRegion* region : to_release) { region->release_reserve(); }
    return false;
}

void ElasticKvRegion::set_entitled_pages(std::uint32_t pages) noexcept {
    Impl& impl = *impl_;
    const std::size_t granules = (static_cast<std::size_t>(pages) + impl.spec.granule_pages - 1) /
                                 impl.spec.granule_pages;
    const std::lock_guard<std::mutex> lock(ledger_mutex());
    ledger()[impl_.get()].entitled = granules * impl.granule_bytes;
}

void ElasticKvRegion::release_reserve() noexcept {
    Impl& impl = *impl_;
    const std::lock_guard<std::mutex> lock(impl.mutex);
    impl.release_requested = true;
}

void ElasticKvRegion::flush_reserve_release() noexcept try {
    Impl& impl = *impl_;
    const ScopedDevice selected(impl.spec.device);
    const std::lock_guard<std::mutex> lock(impl.mutex);
    if (!impl.release_requested || impl.asleep || impl.stopping) { return; }
    // Only the engine's own thread may fence its stream: a record from anywhere else lands
    // inside whatever that thread is capturing and invalidates the capture. Same rule as
    // release_page: while a capture is in flight, leave it for the next boundary.
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if (impl.spec.fence_stream != nullptr &&
        (cudaStreamIsCapturing(impl.spec.fence_stream, &capturing) != cudaSuccess ||
         capturing != cudaStreamCaptureStatusNone)) {
        return;
    }
    impl.release_requested = false;
    for (std::uint32_t g = 0; g < impl.granules; ++g) {
        Impl::Granule& granule = impl.state[g];
        if (!granule.mapped || granule.used_pages != 0) { continue; }
        cudaEvent_t fence = nullptr;
        if (impl.spec.fence_stream != nullptr) {
            if (cudaEventCreateWithFlags(&fence, cudaEventDisableTiming) != cudaSuccess) { continue; }
            if (cudaEventRecord(fence, impl.spec.fence_stream) != cudaSuccess) {
                (void)cudaEventDestroy(fence);
                continue;
            }
        }
        impl.post(Impl::Job{Impl::Job::Kind::Unmap, g, granule.generation, fence, /*forced=*/true});
    }
} catch (...) {
    // Keep the region mapped if its device cannot be bound for retirement.
}

void ElasticKvRegion::wait_idle() {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->cv.wait(lock, [&] { return impl_->jobs.empty() && !impl_->busy; });
}

std::size_t ElasticKvRegion::sleep() {
    wait_idle();
    Impl& impl = *impl_;
    const std::lock_guard<std::mutex> lock(impl.mutex);
    if (impl.asleep) { return 0; }
    CUDA_CHECK(cudaSetDevice(impl.spec.device));
    std::size_t released = 0;
    for (std::uint32_t g = 0; g < impl.granules; ++g) {
        Impl::Granule& granule = impl.state[g];
        if (!granule.mapped) { continue; }
        if (granule.used_pages != 0) {
            granule.backup.assign(impl.spec.planes.size(), nullptr);
            for (std::size_t p = 0; p < impl.spec.planes.size(); ++p) {
                const std::size_t span = impl.spec.planes[p].page_bytes * impl.spec.granule_pages;
                const auto at = reinterpret_cast<const void*>(impl.va + impl.spec.planes[p].offset +
                                                              span * g);
                CUDA_CHECK(cudaMallocHost(&granule.backup[p], span));
                CUDA_CHECK(cudaMemcpy(granule.backup[p], at, span, cudaMemcpyDeviceToHost));
            }
        }
        impl.unmap_granule(g);
        released += impl.granule_bytes;
    }
    impl.asleep = true;
    return released;
}

std::size_t ElasticKvRegion::wake() {
    Impl& impl = *impl_;
    std::size_t mapped = 0;
    {
        const std::lock_guard<std::mutex> lock(impl.mutex);
        if (!impl.asleep) { return 0; }
        CUDA_CHECK(cudaSetDevice(impl.spec.device));
        for (std::uint32_t g = 0; g < impl.granules; ++g) {
            Impl::Granule& granule = impl.state[g];
            if (granule.used_pages == 0) {
                // Emptied while asleep (its requests were failed or cancelled meanwhile): nothing
                // to restore, and the pinned backup is no longer anyone's.
                for (void* backup : granule.backup) {
                    if (backup != nullptr) { (void)cudaFreeHost(backup); }
                }
                granule.backup.clear();
                continue;
            }
            if (granule.backup.empty()) {
                if (!granule.mapped) { throw std::logic_error("live sleeping granule has no backup"); }
                continue; // completed by an earlier wake attempt
            }
            if (granule.backup.size() != impl.spec.planes.size()) {
                throw std::logic_error("sleep backup does not cover every cache plane");
            }
            const bool already_mapped = granule.mapped;
            impl.map_granule(g); // throws on OOM; granules already woken stay woken for a retry
            if (!already_mapped) { mapped += impl.granule_bytes; }
            for (std::size_t p = 0; p < impl.spec.planes.size(); ++p) {
                const std::size_t span = impl.spec.planes[p].page_bytes * impl.spec.granule_pages;
                const auto at = reinterpret_cast<void*>(impl.va + impl.spec.planes[p].offset + span * g);
                if (granule.backup[p] != nullptr) {
                    const auto copied = cudaMemcpy(at, granule.backup[p], span, cudaMemcpyHostToDevice);
                    if (copied != cudaSuccess) {
                        throw std::runtime_error(std::string("elastic KV restore failed: ") + cudaGetErrorString(copied));
                    }
                    (void)cudaFreeHost(granule.backup[p]);
                    granule.backup[p] = nullptr;
                }
            }
            granule.backup.clear();
        }
        impl.asleep = false;
        impl.post(Impl::Job{Impl::Job::Kind::Trim});
    }
    return mapped;
}

} // namespace sinfer
