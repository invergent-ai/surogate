// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// OffloadManager: Group-based CPU/GPU tensor offloading implementation.

#include "runtime/qlora/offload_manager.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>

#include <fmt/format.h>

#include "utilities/utils.h"

namespace qlora {

namespace {

/// Concrete implementation of the OffloadManager interface.
///
/// Manages groups of QuantizedTensors, supporting:
/// - On-demand loading from CPU to GPU
/// - LRU eviction when GPU memory budget is exceeded
/// - Async prefetching on a separate stream with event-based synchronization
class OffloadManagerImpl final : public OffloadManager {
public:
    OffloadManagerImpl(const OffloadConfig& config,
                       const std::shared_ptr<TensorAllocator>& allocator)
        : mConfig(config)
        , mAllocator(allocator)
        , mCurrentStep(0)
    {
        if (mConfig.enable_prefetch) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&mPrefetchStream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&mPrefetchEvent, cudaEventDisableTiming));
        }
    }

    ~OffloadManagerImpl() noexcept override {
        if (mPrefetchStream) {
            cudaStreamDestroy(mPrefetchStream);
        }
        if (mPrefetchEvent) {
            cudaEventDestroy(mPrefetchEvent);
        }
        // Free GPU shadow buffers for all groups
        for (auto& [gid, group] : mGroups) {
            free_gpu_buffers(group);
        }
    }

    // =========================================================================
    // OffloadManager interface
    // =========================================================================

    void register_tensor(
        QuantizedTensor* tensor,
        int group_id,
        const std::string& name) override
    {
        auto& group = get_or_create_group(group_id);

        TensorEntry entry;
        entry.tensor = tensor;
        entry.name = name;

        // Track the total bytes for this tensor
        entry.cpu_data_bytes = tensor->data.is_null() ? 0 : tensor->data.bytes();
        entry.cpu_scales_bytes = tensor->scales.is_null() ? 0 : tensor->scales.bytes();
        entry.cpu_meta_bytes = tensor->meta.is_null() ? 0 : tensor->meta.bytes();
        entry.cpu_meta2_bytes = tensor->meta2.is_null() ? 0 : tensor->meta2.bytes();

        group.total_bytes += entry.cpu_data_bytes + entry.cpu_scales_bytes
                           + entry.cpu_meta_bytes + entry.cpu_meta2_bytes;
        group.entries.push_back(std::move(entry));

        // If offloading is disabled (max_resident == 0 means unlimited), mark as resident
        if (mConfig.max_resident_groups == 0) {
            group.state = GroupState::RESIDENT;
        }
    }

    bool load_group(int group_id, cudaStream_t stream) override {
        auto it = mGroups.find(group_id);
        if (it == mGroups.end()) {
            return false;
        }
        auto& group = it->second;

        // Update LRU timestamp
        group.last_access_step = mCurrentStep;

        if (group.state == GroupState::RESIDENT) {
            return true;  // Already loaded
        }

        if (group.state == GroupState::LOADING) {
            // Prefetch in progress - wait for it to complete
            if (mPrefetchEvent) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, mPrefetchEvent, 0));
            }
            group.state = GroupState::RESIDENT;
            mNumResident++;
            group.load_count++;
            return true;
        }

        // Evict LRU group(s) if needed
        evict_if_needed(group_id);

        // Transfer from CPU to GPU
        allocate_gpu_buffers(group);
        transfer_to_gpu(group, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        group.state = GroupState::RESIDENT;
        mNumResident++;
        group.load_count++;

        return true;
    }

    void unload_group(int group_id, cudaStream_t stream) override {
        auto it = mGroups.find(group_id);
        if (it == mGroups.end()) {
            return;
        }
        auto& group = it->second;

        if (group.state == GroupState::UNLOADED) {
            return;
        }

        if (group.state == GroupState::LOADING) {
            // Wait for prefetch to finish before unloading
            if (mPrefetchStream) {
                CUDA_CHECK(cudaStreamSynchronize(mPrefetchStream));
            }
        }

        // Transfer data back to CPU buffers
        transfer_to_cpu(group, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Free GPU shadow buffers
        free_gpu_buffers(group);

        group.state = GroupState::UNLOADED;
        if (mNumResident > 0) {
            mNumResident--;
        }
    }

    void prefetch_group(int group_id, cudaStream_t /*stream*/) override {
        if (!mConfig.enable_prefetch || !mPrefetchStream) {
            return;
        }

        auto it = mGroups.find(group_id);
        if (it == mGroups.end()) {
            return;
        }
        auto& group = it->second;

        if (group.state == GroupState::RESIDENT || group.state == GroupState::LOADING) {
            return;  // Already loaded or being loaded
        }

        // Evict if needed before starting prefetch
        evict_if_needed(group_id);

        // Allocate GPU buffers and start async transfer on prefetch stream
        allocate_gpu_buffers(group);
        transfer_to_gpu(group, mPrefetchStream);

        // Record event so load_group() can wait on it
        CUDA_CHECK(cudaEventRecord(mPrefetchEvent, mPrefetchStream));

        group.state = GroupState::LOADING;
    }

    void new_step() override {
        mCurrentStep++;
    }

    [[nodiscard]] bool is_resident(int group_id) const override {
        auto it = mGroups.find(group_id);
        return it != mGroups.end() && it->second.state == GroupState::RESIDENT;
    }

    [[nodiscard]] GroupState get_state(int group_id) const override {
        auto it = mGroups.find(group_id);
        return it != mGroups.end() ? it->second.state : GroupState::UNLOADED;
    }

    [[nodiscard]] GroupStats get_stats(int group_id) const override {
        auto it = mGroups.find(group_id);
        if (it == mGroups.end()) {
            return {};
        }
        const auto& g = it->second;
        GroupStats stats;
        stats.group_id = group_id;
        stats.state = g.state;
        stats.num_tensors = static_cast<int>(g.entries.size());
        stats.total_bytes = g.total_bytes;
        stats.last_access_step = g.last_access_step;
        stats.load_count = g.load_count;
        return stats;
    }

    [[nodiscard]] int num_groups() const override {
        return static_cast<int>(mGroups.size());
    }

    [[nodiscard]] int num_resident() const override {
        return mNumResident;
    }

    [[nodiscard]] size_t gpu_memory_used() const override {
        size_t total = 0;
        for (const auto& [gid, group] : mGroups) {
            if (group.state == GroupState::RESIDENT || group.state == GroupState::LOADING) {
                total += group.total_bytes;
            }
        }
        return total;
    }

    [[nodiscard]] size_t cpu_memory_used() const override {
        size_t total = 0;
        for (const auto& [gid, group] : mGroups) {
            if (group.state == GroupState::UNLOADED) {
                total += group.total_bytes;
            }
        }
        return total;
    }

    void set_max_resident_groups(int max_groups) override {
        mConfig.max_resident_groups = max_groups;
    }

    [[nodiscard]] int max_resident_groups() const override {
        return mConfig.max_resident_groups;
    }

    [[nodiscard]] size_t max_group_bytes() const override {
        size_t max_bytes = 0;
        for (const auto& [gid, group] : mGroups) {
            max_bytes = std::max(max_bytes, group.total_bytes);
        }
        return max_bytes;
    }

private:
    // =========================================================================
    // Internal data structures
    // =========================================================================

    /// Per-tensor tracking within a group.
    struct TensorEntry {
        QuantizedTensor* tensor = nullptr;
        std::string name;

        // Sizes of CPU-side buffers (for allocation tracking)
        size_t cpu_data_bytes = 0;
        size_t cpu_scales_bytes = 0;
        size_t cpu_meta_bytes = 0;
        size_t cpu_meta2_bytes = 0;

        // CPU-side buffer pointers (set when unloaded from GPU)
        // These hold copies of the quantized data on the CPU.
        std::byte* cpu_data = nullptr;
        std::byte* cpu_scales = nullptr;
        std::byte* cpu_meta = nullptr;
        std::byte* cpu_meta2 = nullptr;

        // GPU-side shadow buffer pointers (set when loaded to GPU)
        std::byte* gpu_data = nullptr;
        std::byte* gpu_scales = nullptr;
        std::byte* gpu_meta = nullptr;
        std::byte* gpu_meta2 = nullptr;
    };

    /// Per-group tracking.
    struct Group {
        std::vector<TensorEntry> entries;
        GroupState state = GroupState::UNLOADED;
        size_t total_bytes = 0;
        int64_t last_access_step = 0;
        int load_count = 0;
    };

    // =========================================================================
    // Internal helpers
    // =========================================================================

    Group& get_or_create_group(int group_id) {
        auto it = mGroups.find(group_id);
        if (it == mGroups.end()) {
            it = mGroups.emplace(group_id, Group{}).first;
        }
        return it->second;
    }

    /// Find the LRU resident group (excluding the specified group).
    int find_lru_group(int exclude_group_id) const {
        int lru_id = -1;
        int64_t oldest_step = std::numeric_limits<int64_t>::max();

        for (const auto& [gid, group] : mGroups) {
            if (gid == exclude_group_id) continue;
            if (group.state != GroupState::RESIDENT) continue;

            if (group.last_access_step < oldest_step) {
                oldest_step = group.last_access_step;
                lru_id = gid;
            }
        }
        return lru_id;
    }

    /// Evict enough groups to make room for one more, if needed.
    void evict_if_needed(int incoming_group_id) {
        if (mConfig.max_resident_groups == 0) {
            return;  // Unlimited
        }

        while (mNumResident >= mConfig.max_resident_groups) {
            int lru_id = find_lru_group(incoming_group_id);
            if (lru_id < 0) {
                break;  // Nothing to evict
            }

            // Create a synchronous stream for eviction
            cudaStream_t evict_stream = nullptr;
            CUDA_CHECK(cudaStreamCreateWithFlags(&evict_stream, cudaStreamNonBlocking));
            unload_group(lru_id, evict_stream);
            CUDA_CHECK(cudaStreamDestroy(evict_stream));
        }
    }

    /// Allocate GPU shadow buffers for a group's tensors.
    void allocate_gpu_buffers(Group& group) {
        for (auto& entry : group.entries) {
            if (!entry.tensor) continue;

            // Save current (CPU) pointers
            entry.cpu_data = entry.tensor->data.Data;
            entry.cpu_scales = entry.tensor->scales.Data;
            entry.cpu_meta = entry.tensor->meta.Data;
            entry.cpu_meta2 = entry.tensor->meta2.Data;

            // Allocate GPU buffers
            if (entry.cpu_data_bytes > 0) {
                CUDA_CHECK(cudaMalloc(&entry.gpu_data, entry.cpu_data_bytes));
            }
            if (entry.cpu_scales_bytes > 0) {
                CUDA_CHECK(cudaMalloc(&entry.gpu_scales, entry.cpu_scales_bytes));
            }
            if (entry.cpu_meta_bytes > 0) {
                CUDA_CHECK(cudaMalloc(&entry.gpu_meta, entry.cpu_meta_bytes));
            }
            if (entry.cpu_meta2_bytes > 0) {
                CUDA_CHECK(cudaMalloc(&entry.gpu_meta2, entry.cpu_meta2_bytes));
            }
        }
    }

    /// Free GPU shadow buffers for a group.
    void free_gpu_buffers(Group& group) {
        for (auto& entry : group.entries) {
            if (entry.gpu_data) { cudaFree(entry.gpu_data); entry.gpu_data = nullptr; }
            if (entry.gpu_scales) { cudaFree(entry.gpu_scales); entry.gpu_scales = nullptr; }
            if (entry.gpu_meta) { cudaFree(entry.gpu_meta); entry.gpu_meta = nullptr; }
            if (entry.gpu_meta2) { cudaFree(entry.gpu_meta2); entry.gpu_meta2 = nullptr; }
        }
    }

    /// Transfer a group's tensors from CPU to GPU and update tensor pointers.
    void transfer_to_gpu(Group& group, cudaStream_t stream) {
        for (auto& entry : group.entries) {
            if (!entry.tensor) continue;

            if (entry.gpu_data && entry.cpu_data && entry.cpu_data_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.gpu_data, entry.cpu_data,
                                            entry.cpu_data_bytes,
                                            cudaMemcpyHostToDevice, stream));
                entry.tensor->data.Data = entry.gpu_data;
                entry.tensor->data.Device = mConfig.device_id;
            }
            if (entry.gpu_scales && entry.cpu_scales && entry.cpu_scales_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.gpu_scales, entry.cpu_scales,
                                            entry.cpu_scales_bytes,
                                            cudaMemcpyHostToDevice, stream));
                entry.tensor->scales.Data = entry.gpu_scales;
                entry.tensor->scales.Device = mConfig.device_id;
            }
            if (entry.gpu_meta && entry.cpu_meta && entry.cpu_meta_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.gpu_meta, entry.cpu_meta,
                                            entry.cpu_meta_bytes,
                                            cudaMemcpyHostToDevice, stream));
                entry.tensor->meta.Data = entry.gpu_meta;
                entry.tensor->meta.Device = mConfig.device_id;
            }
            if (entry.gpu_meta2 && entry.cpu_meta2 && entry.cpu_meta2_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.gpu_meta2, entry.cpu_meta2,
                                            entry.cpu_meta2_bytes,
                                            cudaMemcpyHostToDevice, stream));
                entry.tensor->meta2.Data = entry.gpu_meta2;
                entry.tensor->meta2.Device = mConfig.device_id;
            }
        }
    }

    /// Transfer a group's tensors from GPU back to CPU and restore pointers.
    void transfer_to_cpu(Group& group, cudaStream_t stream) {
        for (auto& entry : group.entries) {
            if (!entry.tensor) continue;

            if (entry.gpu_data && entry.cpu_data && entry.cpu_data_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.cpu_data, entry.gpu_data,
                                            entry.cpu_data_bytes,
                                            cudaMemcpyDeviceToHost, stream));
                entry.tensor->data.Data = entry.cpu_data;
                entry.tensor->data.Device = -1;
            }
            if (entry.gpu_scales && entry.cpu_scales && entry.cpu_scales_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.cpu_scales, entry.gpu_scales,
                                            entry.cpu_scales_bytes,
                                            cudaMemcpyDeviceToHost, stream));
                entry.tensor->scales.Data = entry.cpu_scales;
                entry.tensor->scales.Device = -1;
            }
            if (entry.gpu_meta && entry.cpu_meta && entry.cpu_meta_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.cpu_meta, entry.gpu_meta,
                                            entry.cpu_meta_bytes,
                                            cudaMemcpyDeviceToHost, stream));
                entry.tensor->meta.Data = entry.cpu_meta;
                entry.tensor->meta.Device = -1;
            }
            if (entry.gpu_meta2 && entry.cpu_meta2 && entry.cpu_meta2_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.cpu_meta2, entry.gpu_meta2,
                                            entry.cpu_meta2_bytes,
                                            cudaMemcpyDeviceToHost, stream));
                entry.tensor->meta2.Data = entry.cpu_meta2;
                entry.tensor->meta2.Device = -1;
            }
        }
    }

    // =========================================================================
    // Members
    // =========================================================================

    OffloadConfig mConfig;
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unordered_map<int, Group> mGroups;

    int64_t mCurrentStep = 0;
    int mNumResident = 0;

    // Prefetch infrastructure
    cudaStream_t mPrefetchStream = nullptr;
    cudaEvent_t mPrefetchEvent = nullptr;
};

}  // anonymous namespace

// =============================================================================
// Factory function
// =============================================================================

std::unique_ptr<OffloadManager> create_offload_manager(
    const OffloadConfig& config,
    const std::shared_ptr<TensorAllocator>& allocator) {
    return std::make_unique<OffloadManagerImpl>(config, allocator);
}

}  // namespace qlora
