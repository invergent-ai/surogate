#include "targets/qwen4exp/impl/load/host_bank.h"

#include "core/device.h"

#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace ninfer::targets::qwen4exp::detail {

std::size_t HostBankPlan::total_bytes() const noexcept {
    std::size_t total = 0;
    for (const auto& object : objects) { total += object.payload.size(); }
    return total;
}

HostBank::HostBank(const HostBankPlan& plan) {
    objects_.reserve(plan.objects.size());
    for (const auto& source : plan.objects) {
        HostObject object;
        object.bytes = source.payload.size();
        object.name  = source.name;
        if (object.bytes == 0) {
            throw std::invalid_argument("host bank object " + source.name + " is empty");
        }
        CUDA_CHECK(cudaHostAlloc(&object.host, object.bytes, cudaHostAllocMapped | cudaHostAllocPortable));
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
            const std::size_t length = static_cast<std::size_t>(address + object.bytes - start);
            (void)madvise(reinterpret_cast<void*>(start), length, MADV_SEQUENTIAL);
            (void)madvise(reinterpret_cast<void*>(start), length, MADV_WILLNEED);
        }
        const std::size_t workers = 16;
        const std::size_t chunk   = (object.bytes + workers - 1) / workers;
        std::vector<std::thread> threads;
        for (std::size_t w = 0; w < workers; ++w) {
            const std::size_t begin = w * chunk;
            if (begin >= object.bytes) { break; }
            const std::size_t count = std::min(chunk, object.bytes - begin);
            threads.emplace_back([&, begin, count] {
                std::memcpy(static_cast<std::byte*>(object.host) + begin,
                            source.payload.data() + begin, count);
            });
        }
        for (auto& thread : threads) { thread.join(); }
        total_bytes_ += object.bytes;
        objects_.emplace_back(source.handle.index, object);
    }
}

HostBank::~HostBank() {
    for (auto& [index, object] : objects_) {
        if (object.host != nullptr) { (void)cudaFreeHost(object.host); }
    }
}

const HostObject& HostBank::object(artifact::ObjectHandle handle) const {
    for (const auto& [index, object] : objects_) {
        if (index == handle.index) { return object; }
    }
    throw std::out_of_range("host bank has no object for this handle");
}

} // namespace ninfer::targets::qwen4exp::detail
