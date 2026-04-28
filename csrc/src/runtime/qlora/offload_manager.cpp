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

bool is_device_pointer(const std::byte* ptr) {
    if (!ptr) {
        return false;
    }
    cudaPointerAttributes attrs{};
    const cudaError_t st = cudaPointerGetAttributes(&attrs, ptr);
    if (st != cudaSuccess) {
        // Pageable host memory often returns invalid-value here; treat as host.
        (void)cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    return attrs.type == cudaMemoryTypeDevice;
#else
    return attrs.memoryType == cudaMemoryTypeDevice;
#endif
}

/// Concrete implementation of the OffloadManager interface.
///
/// Manages groups of QuantizedTensors, supporting:
/// - On-demand loading from CPU to GPU
/// - LRU eviction when GPU memory budget is exceeded
/// - Async prefetching on a separate stream with event-based synchronization
class OffloadManagerImpl final : public OffloadManager {
public:
    OffloadManagerImpl(const OffloadConfig& config, const std::shared_ptr<TensorAllocator>& allocator)
        : mConfig(config),
          mAllocator(allocator),
          mCurrentStep(0) {
        if (mConfig.enable_prefetch) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&mPrefetchStream, cudaStreamNonBlocking));
        }
    }

    ~OffloadManagerImpl() noexcept override {
        if (mPrefetchStream) {
            cudaStreamDestroy(mPrefetchStream);
        }
        // Free GPU shadow buffers for all groups
        for (auto& [gid, group] : mGroups) {
            if (group.prefetch_done) {
                cudaEventDestroy(group.prefetch_done);
                group.prefetch_done = nullptr;
            }
            free_gpu_buffers(group);
            free_owned_host_buffers(group);
        }
    }

    // =========================================================================
    // OffloadManager interface
    // =========================================================================

    void register_tensor(QuantizedTensor* tensor, int group_id, const std::string& name) override {
        auto& group = get_or_create_group(group_id);

        TensorEntry entry;
        entry.tensor = tensor;
        entry.name = name;

        // Track the total bytes for this tensor
        entry.cpu_data_bytes = tensor->data.is_null() ? 0 : tensor->data.bytes();
        entry.cpu_scales_bytes = tensor->scales.is_null() ? 0 : tensor->scales.bytes();
        entry.cpu_meta_bytes = tensor->meta.is_null() ? 0 : tensor->meta.bytes();
        entry.cpu_meta2_bytes = tensor->meta2.is_null() ? 0 : tensor->meta2.bytes();
        entry.cpu_data = tensor->data.Data;
        entry.cpu_scales = tensor->scales.Data;
        entry.cpu_meta = tensor->meta.Data;
        entry.cpu_meta2 = tensor->meta2.Data;

        // Offload path requires a stable host copy. If the source tensor is
        // device-backed (e.g. external import), stage one host copy now and
        // keep it as canonical CPU storage for future load/unload cycles.
        if (mConfig.max_resident_groups > 0) {
            ensure_host_backing(*tensor, entry);
        }

        group.total_bytes +=
            entry.cpu_data_bytes + entry.cpu_scales_bytes + entry.cpu_meta_bytes + entry.cpu_meta2_bytes;
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

        touch_group(group);

        if (group.state == GroupState::RESIDENT) {
            return true;  // Already loaded
        }

        if (group.state == GroupState::LOADING) {
            // Prefetch in progress - wait for it to complete
            if (group.prefetch_done) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, group.prefetch_done, 0));
            }
            group.state = GroupState::RESIDENT;
            mNumResident++;
            group.load_count++;
            return true;
        }

        // Evict LRU group(s) if needed on the caller stream so eviction is
        // ordered after any in-flight compute that may still read those
        // buffers. Using a separate stream can race and free/live-migrate
        // quantized storage while kernels are still running.
        evict_if_needed(group_id, stream);

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
            if (group.prefetch_done) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, group.prefetch_done, 0));
            }
        }

        if (mImmutableTensors) {
            restore_cpu_pointers(group);
        } else {
            // Transfer mutable data back to CPU buffers.
            transfer_to_cpu(group, stream);
        }
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

        touch_group(group);

        if (group.state == GroupState::RESIDENT || group.state == GroupState::LOADING) {
            return;  // Already loaded or being loaded
        }

        // Do not evict on the prefetch stream. Cross-stream eviction can race
        // with compute kernels using resident groups.
        if (mConfig.max_resident_groups > 0) {
            int loading = 0;
            for (const auto& [gid, g] : mGroups) {
                (void)gid;
                if (g.state == GroupState::LOADING) {
                    loading++;
                }
            }
            if ((mNumResident + loading) >= mConfig.max_resident_groups) {
                return;
            }
        }

        // Allocate GPU buffers and start async transfer on prefetch stream
        allocate_gpu_buffers(group);
        transfer_to_gpu(group, mPrefetchStream);

        // Record event so load_group() can wait on it
        if (!group.prefetch_done) {
            CUDA_CHECK(cudaEventCreateWithFlags(&group.prefetch_done, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(group.prefetch_done, mPrefetchStream));

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

    void set_immutable(bool immutable) override {
        mImmutableTensors = immutable;
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

        // Host buffers allocated/staged by OffloadManager itself.
        bool owns_cpu_data = false;
        bool owns_cpu_scales = false;
        bool owns_cpu_meta = false;
        bool owns_cpu_meta2 = false;
    };

    /// Per-group tracking.
    struct Group {
        std::vector<TensorEntry> entries;
        GroupState state = GroupState::UNLOADED;
        size_t total_bytes = 0;
        int64_t last_access_step = 0;
        int load_count = 0;
        cudaEvent_t prefetch_done = nullptr;
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

    void touch_group(Group& group) {
        group.last_access_step = ++mAccessCounter;
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
    void evict_if_needed(int incoming_group_id, cudaStream_t stream) {
        if (mConfig.max_resident_groups == 0) {
            return;  // Unlimited
        }

        while (mNumResident >= mConfig.max_resident_groups) {
            int lru_id = find_lru_group(incoming_group_id);
            if (lru_id < 0) {
                break;  // Nothing to evict
            }

            unload_group(lru_id, stream);
        }
    }

    /// Allocate GPU shadow buffers for a group's tensors.
    void allocate_gpu_buffers(Group& group) {
        for (auto& entry : group.entries) {
            if (!entry.tensor) continue;

            // Canonical CPU pointers are established at registration (and staged
            // if needed). Do not overwrite them here from potentially transient
            // device pointers.
            if (!entry.cpu_data) entry.cpu_data = entry.tensor->data.Data;
            if (!entry.cpu_scales) entry.cpu_scales = entry.tensor->scales.Data;
            if (!entry.cpu_meta) entry.cpu_meta = entry.tensor->meta.Data;
            if (!entry.cpu_meta2) entry.cpu_meta2 = entry.tensor->meta2.Data;

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
            if (entry.gpu_data) {
                cudaFree(entry.gpu_data);
                entry.gpu_data = nullptr;
            }
            if (entry.gpu_scales) {
                cudaFree(entry.gpu_scales);
                entry.gpu_scales = nullptr;
            }
            if (entry.gpu_meta) {
                cudaFree(entry.gpu_meta);
                entry.gpu_meta = nullptr;
            }
            if (entry.gpu_meta2) {
                cudaFree(entry.gpu_meta2);
                entry.gpu_meta2 = nullptr;
            }
        }
    }

    /// Transfer a group's tensors from CPU to GPU and update tensor pointers.
    void transfer_to_gpu(Group& group, cudaStream_t stream) {
        for (auto& entry : group.entries) {
            if (!entry.tensor) continue;

            if (entry.gpu_data && entry.cpu_data && entry.cpu_data_bytes > 0) {
                CUDA_CHECK(
                    cudaMemcpyAsync(entry.gpu_data, entry.cpu_data, entry.cpu_data_bytes, cudaMemcpyDefault, stream));
                entry.tensor->data.Data = entry.gpu_data;
                entry.tensor->data.Device = mConfig.device_id;
            }
            if (entry.gpu_scales && entry.cpu_scales && entry.cpu_scales_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.gpu_scales,
                                           entry.cpu_scales,
                                           entry.cpu_scales_bytes,
                                           cudaMemcpyDefault,
                                           stream));
                entry.tensor->scales.Data = entry.gpu_scales;
                entry.tensor->scales.Device = mConfig.device_id;
            }
            if (entry.gpu_meta && entry.cpu_meta && entry.cpu_meta_bytes > 0) {
                CUDA_CHECK(
                    cudaMemcpyAsync(entry.gpu_meta, entry.cpu_meta, entry.cpu_meta_bytes, cudaMemcpyDefault, stream));
                entry.tensor->meta.Data = entry.gpu_meta;
                entry.tensor->meta.Device = mConfig.device_id;
            }
            if (entry.gpu_meta2 && entry.cpu_meta2 && entry.cpu_meta2_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.gpu_meta2,
                                           entry.cpu_meta2,
                                           entry.cpu_meta2_bytes,
                                           cudaMemcpyDefault,
                                           stream));
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
                CUDA_CHECK(
                    cudaMemcpyAsync(entry.cpu_data, entry.gpu_data, entry.cpu_data_bytes, cudaMemcpyDefault, stream));
                entry.tensor->data.Data = entry.cpu_data;
                entry.tensor->data.Device = -1;
            }
            if (entry.gpu_scales && entry.cpu_scales && entry.cpu_scales_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.cpu_scales,
                                           entry.gpu_scales,
                                           entry.cpu_scales_bytes,
                                           cudaMemcpyDefault,
                                           stream));
                entry.tensor->scales.Data = entry.cpu_scales;
                entry.tensor->scales.Device = -1;
            }
            if (entry.gpu_meta && entry.cpu_meta && entry.cpu_meta_bytes > 0) {
                CUDA_CHECK(
                    cudaMemcpyAsync(entry.cpu_meta, entry.gpu_meta, entry.cpu_meta_bytes, cudaMemcpyDefault, stream));
                entry.tensor->meta.Data = entry.cpu_meta;
                entry.tensor->meta.Device = -1;
            }
            if (entry.gpu_meta2 && entry.cpu_meta2 && entry.cpu_meta2_bytes > 0) {
                CUDA_CHECK(cudaMemcpyAsync(entry.cpu_meta2,
                                           entry.gpu_meta2,
                                           entry.cpu_meta2_bytes,
                                           cudaMemcpyDefault,
                                           stream));
                entry.tensor->meta2.Data = entry.cpu_meta2;
                entry.tensor->meta2.Device = -1;
            }
        }
    }

    /// Restore tensor descriptors to canonical CPU storage without copying data.
    /// This is valid only for immutable quantized weights: GPU shadows are read-only
    /// replicas of CPU-backed quantized storage, so eviction can drop them directly.
    void restore_cpu_pointers(Group& group) {
        for (auto& entry : group.entries) {
            if (!entry.tensor) continue;

            if (entry.cpu_data && entry.cpu_data_bytes > 0) {
                entry.tensor->data.Data = entry.cpu_data;
                entry.tensor->data.Device = -1;
            }
            if (entry.cpu_scales && entry.cpu_scales_bytes > 0) {
                entry.tensor->scales.Data = entry.cpu_scales;
                entry.tensor->scales.Device = -1;
            }
            if (entry.cpu_meta && entry.cpu_meta_bytes > 0) {
                entry.tensor->meta.Data = entry.cpu_meta;
                entry.tensor->meta.Device = -1;
            }
            if (entry.cpu_meta2 && entry.cpu_meta2_bytes > 0) {
                entry.tensor->meta2.Data = entry.cpu_meta2;
                entry.tensor->meta2.Device = -1;
            }
        }
    }

    std::byte* alloc_host_buffer(size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }
        if (mConfig.use_pinned_memory) {
            void* ptr = nullptr;
            CUDA_CHECK(cudaMallocHost(&ptr, bytes));
            return static_cast<std::byte*>(ptr);
        }
        return new std::byte[bytes];
    }

    void free_host_buffer(std::byte* ptr) noexcept {
        if (!ptr) return;
        if (mConfig.use_pinned_memory) {
            cudaFreeHost(ptr);
        } else {
            delete[] ptr;
        }
    }

    void maybe_stage_one_component(std::byte*& cpu_ptr, size_t bytes, bool& owns_cpu_ptr, Tensor& field) {
        if (!cpu_ptr || bytes == 0 || !is_device_pointer(cpu_ptr)) {
            return;
        }
        std::byte* host_ptr = alloc_host_buffer(bytes);
        CUDA_CHECK(cudaMemcpy(host_ptr, cpu_ptr, bytes, cudaMemcpyDeviceToHost));
        cpu_ptr = host_ptr;
        owns_cpu_ptr = true;
        field.Data = host_ptr;
        field.Device = -1;
    }

    void ensure_host_backing(QuantizedTensor& tensor, TensorEntry& entry) {
        maybe_stage_one_component(entry.cpu_data, entry.cpu_data_bytes, entry.owns_cpu_data, tensor.data);
        maybe_stage_one_component(entry.cpu_scales, entry.cpu_scales_bytes, entry.owns_cpu_scales, tensor.scales);
        maybe_stage_one_component(entry.cpu_meta, entry.cpu_meta_bytes, entry.owns_cpu_meta, tensor.meta);
        maybe_stage_one_component(entry.cpu_meta2, entry.cpu_meta2_bytes, entry.owns_cpu_meta2, tensor.meta2);
    }

    void free_owned_host_buffers(Group& group) noexcept {
        for (auto& entry : group.entries) {
            if (entry.owns_cpu_data) {
                free_host_buffer(entry.cpu_data);
                entry.cpu_data = nullptr;
                entry.owns_cpu_data = false;
            }
            if (entry.owns_cpu_scales) {
                free_host_buffer(entry.cpu_scales);
                entry.cpu_scales = nullptr;
                entry.owns_cpu_scales = false;
            }
            if (entry.owns_cpu_meta) {
                free_host_buffer(entry.cpu_meta);
                entry.cpu_meta = nullptr;
                entry.owns_cpu_meta = false;
            }
            if (entry.owns_cpu_meta2) {
                free_host_buffer(entry.cpu_meta2);
                entry.cpu_meta2 = nullptr;
                entry.owns_cpu_meta2 = false;
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
    int64_t mAccessCounter = 0;
    int mNumResident = 0;
    bool mImmutableTensors = false;

    // Prefetch infrastructure
    cudaStream_t mPrefetchStream = nullptr;
};

}  // anonymous namespace

// =============================================================================
// Factory function
// =============================================================================

std::unique_ptr<OffloadManager> create_offload_manager(const OffloadConfig& config,
                                                       const std::shared_ptr<TensorAllocator>& allocator) {
    return std::make_unique<OffloadManagerImpl>(config, allocator);
}

}  // namespace qlora
