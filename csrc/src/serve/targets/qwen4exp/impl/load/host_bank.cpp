#include "targets/qwen4exp/impl/load/host_bank.h"
#include "core/numa.h"

#include "api/ops/cpu_expert_compute.h"
#include "api/ops/expert_slot_cache.h"
#include "core/device.h"

#include <string>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
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

namespace sinfer::targets::qwen4exp::detail {

namespace {

/// Bytes an object occupies once assembled, however it is stored.
std::size_t object_bytes(const HostObjectPlan& plan) {
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
        if (q4) {
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
    }
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

} // namespace sinfer::targets::qwen4exp::detail
