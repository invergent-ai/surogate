#include "core/elastic_kv_region.h"

#include "core/device.h"
#include "core/engine_context.h"
#include "core/sleep.h"

#include <cuda.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

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

// The process-wide commitment ledger: each region's cap in bytes and what it has mapped.
struct Commitment {
    int device       = 0;
    std::size_t cap  = 0;
    std::size_t mapped = 0;
};
std::mutex& ledger_mutex() {
    static std::mutex instance;
    return instance;
}
std::map<const void*, Commitment>& ledger() {
    static std::map<const void*, Commitment> instance;
    return instance;
}

} // namespace

std::size_t elastic_kv_unmapped_commitment(int device) noexcept {
    const std::lock_guard<std::mutex> lock(ledger_mutex());
    std::size_t bytes = 0;
    for (const auto& [key, entry] : ledger()) {
        if (entry.device == device && entry.cap > entry.mapped) { bytes += entry.cap - entry.mapped; }
    }
    return bytes;
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
    std::thread worker;

    // ---- device mapping (caller holds mutex) ----

    void map_granule(std::uint32_t g) {
        Granule& granule = state[g];
        if (granule.mapped) { return; }
        const CUmemAllocationProp prop = allocation_prop(spec.device);
        CUmemAccessDesc access         = {};
        access.location                = prop.location;
        access.flags                   = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        granule.handles.assign(spec.planes.size(), {});
        for (std::size_t p = 0; p < spec.planes.size(); ++p) {
            const std::size_t span = spec.planes[p].page_bytes * spec.granule_pages;
            const CUdeviceptr at   = va + spec.planes[p].offset + span * g;
            CUmemGenericAllocationHandle handle{};
            driver_check(cuMemCreate(&handle, span, &prop, 0), "cuMemCreate");
            CUresult mapped_result = cuMemMap(at, span, 0, handle, 0);
            if (mapped_result == CUDA_SUCCESS) {
                mapped_result = cuMemSetAccess(at, span, &access, 1);
                if (mapped_result != CUDA_SUCCESS) { (void)cuMemUnmap(at, span); }
            }
            if (mapped_result != CUDA_SUCCESS) {
                (void)cuMemRelease(handle);
                for (std::size_t q = 0; q < p; ++q) { unmap_plane(g, q); }
                granule.handles.clear();
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
                if (!asleep) {
                    for (std::uint32_t g = 0;
                         g < granules && mapped_free_count() < spec.reserve_granules; ++g) {
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
                    mapped_free_count() > trim_threshold()) {
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
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const std::size_t spoken_for = elastic_kv_unmapped_commitment(impl.spec.device);
        if (cap_bytes > free_bytes - std::min(spoken_for, free_bytes)) {
            throw std::runtime_error(
                "elastic kv region: a cap of " + std::to_string(cap_bytes >> 20) +
                " MiB does not fit beside " + std::to_string(spoken_for >> 20) +
                " MiB already committed to other engines with " + std::to_string(free_bytes >> 20) +
                " MiB free");
        }
        const std::lock_guard<std::mutex> lock(ledger_mutex());
        ledger()[impl_.get()] = Commitment{impl.spec.device, cap_bytes, 0};
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
    if (page < 0 || static_cast<std::uint32_t>(page) >= impl.spec.page_count) {
        throw std::out_of_range("elastic kv region: page out of range");
    }
    const std::uint32_t g = static_cast<std::uint32_t>(page) / impl.spec.granule_pages;
    const std::lock_guard<std::mutex> lock(impl.mutex);
    if (impl.asleep) { throw std::logic_error("elastic kv region: acquire while asleep"); }
    Impl::Granule& granule = impl.state[g];
    ++granule.generation;
    if (!granule.mapped) { impl.map_granule(g); } // demand outran the reserve: map here and now
    ++granule.used_pages;
    if (granule.used_pages == 1) { impl.post(Impl::Job{Impl::Job::Kind::Trim}); } // top the reserve up
}

void ElasticKvRegion::release_page(std::int32_t page) noexcept {
    Impl& impl = *impl_;
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
}

std::size_t ElasticKvRegion::mapped_bytes() const noexcept {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->mapped;
}

std::uint32_t ElasticKvRegion::mapped_granules() const noexcept {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    std::uint32_t count = 0;
    for (const Impl::Granule& granule : impl_->state) { count += granule.mapped ? 1U : 0U; }
    return count;
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
            if (granule.used_pages == 0) { continue; }
            impl.map_granule(g); // throws on OOM; granules already woken stay woken for a retry
            for (std::size_t p = 0; p < impl.spec.planes.size(); ++p) {
                const std::size_t span = impl.spec.planes[p].page_bytes * impl.spec.granule_pages;
                const auto at = reinterpret_cast<void*>(impl.va + impl.spec.planes[p].offset + span * g);
                if (granule.backup[p] != nullptr) {
                    CUDA_CHECK(cudaMemcpy(at, granule.backup[p], span, cudaMemcpyHostToDevice));
                    (void)cudaFreeHost(granule.backup[p]);
                    granule.backup[p] = nullptr;
                }
            }
            granule.backup.clear();
            mapped += impl.granule_bytes;
        }
        impl.asleep = false;
        impl.post(Impl::Job{Impl::Job::Kind::Trim});
    }
    return mapped;
}

} // namespace sinfer
