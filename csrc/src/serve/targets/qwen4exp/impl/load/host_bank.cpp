#include "targets/qwen4exp/impl/load/host_bank.h"

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
        const bool q4 = source.q4_rows > 0;
        const ops::Q4BankPlanes q4_planes =
            q4 ? ops::q4_bank_planes(source.q4_rows, source.q4_k) : ops::Q4BankPlanes{};
        HostObject object;
        object.bytes = q4 ? q4_planes.total_bytes : source.payload.size();
        object.name  = source.name;
        if (object.bytes == 0 || source.payload.empty()) {
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
        if (object.host != nullptr) { (void)cudaFreeHost(object.host); }
    }
}

const HostObject& HostBank::object(artifact::ObjectHandle handle) const {
    for (const auto& [index, object] : objects_) {
        if (index == handle.index) { return object; }
    }
    throw std::out_of_range("host bank has no object for this handle");
}


std::shared_ptr<HostBank> HostBank::shared(const HostBankPlan& plan) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::weak_ptr<HostBank>> banks;
    std::string key;
    for (const auto& source : plan.objects) {
        key += source.name;
        key += ':';
        key += std::to_string(source.payload.size());
        if (source.q4_rows > 0) { key += ":q4"; }
        key += ';';
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (auto found = banks.find(key); found != banks.end()) {
        if (auto live = found->second.lock()) { return live; }
    }
    auto bank  = std::make_shared<HostBank>(plan);
    banks[key] = bank;
    return bank;
}

} // namespace ninfer::targets::qwen4exp::detail
