// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILITIES_STACK_H
#define SUROGATE_SRC_UTILITIES_STACK_H

#include <cstddef>
#include <vector>
#include "utilities/tensor.h"

class DeviceMemoryStack {
public:
    DeviceMemoryStack() = default;
    DeviceMemoryStack(std::byte* memory, std::size_t amount, int device_id);

    std::byte* allocate(std::size_t amount, const char* name="<unnamed>");
    Tensor allocate(ETensorDType dtype, const std::vector<long>& shape, const char* name="<unnamed>");

    void free(std::byte* ptr);
    void free(Tensor& tensor);

    std::size_t unused_capacity() const;
    std::size_t bytes_used() const;
    std::size_t max_utilization() const;
    int device_id() const;
    bool owns(const std::byte* ptr) const;
    bool is_live(const std::byte* ptr) const;

    // Checkpoint/restore for CUDA graph compatibility.
    // Save the current stack position and restore it later to ensure
    // temp_alloc returns the same addresses when graphs are replayed.
    struct Checkpoint {
        std::byte* top;
        std::size_t alloc_count;
    };
    Checkpoint checkpoint() const;
    void restore(const Checkpoint& cp);

    struct sAllocRecord {
        std::byte* Pointer;
        std::size_t Amount;
        const char* Name;
    };
    using AllocationList = std::vector<sAllocRecord>;

    const AllocationList& get_high_mark() const { return mHighMark; }
    void set_high_mark(const AllocationList& list) { mHighMark = list; }

    std::vector<std::pair<std::string, long>> get_allocation_stats() const;

private:
    int mDeviceID;
    std::byte* mBackingMemory;
    std::byte* mTop;
    std::size_t mCapacity;

    void _track_max();

    AllocationList mAlloc;

    std::size_t mMaxUtilization = 0;
    std::vector<sAllocRecord> mHighMark;
};

// Debug: watch a specific stack memory range and log overlapping allocations.
void set_stack_watch_range(const std::byte* ptr, std::size_t bytes, const char* tag = nullptr);
void clear_stack_watch_range();

#endif //SUROGATE_SRC_UTILITIES_STACK_H
