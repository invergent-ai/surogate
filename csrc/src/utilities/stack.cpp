// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "stack.h"
#include "utilities/utils.h"

/**
 * @brief Construct a stack allocator over a pre-allocated device memory region.
 *
 * The stack grows upwards from @p memory and supports LIFO frees.
 *
 * @param memory    Base pointer to the backing memory region.
 * @param amount    Total capacity of the backing region in bytes.
 * @param device_id Device identifier associated with the memory (used for Tensor views).
 */
DeviceMemoryStack::DeviceMemoryStack(std::byte* memory, std::size_t amount, int device_id) :
    mBackingMemory(memory), mTop(memory), mDeviceID(device_id), mCapacity(amount) {

}

/**
 * @brief Allocate raw bytes from the stack with fixed alignment.
 *
 * Allocation is rounded up to a 4096-byte boundary. The returned pointer is the
 * start of the allocated region. Must be freed in strict LIFO order via free(ptr).
 *
 * @param amount Number of bytes requested.
 * @param name   Human-readable label used for high-water-mark/statistics tracking.
 * @return Pointer to the beginning of the allocated region within the backing memory.
 *
 * @throws std::bad_alloc   If the allocation would exceed the backing capacity.
 */
std::byte* DeviceMemoryStack::allocate(std::size_t amount, const char* name) {
    constexpr size_t alignment = 4096;
    std::size_t aligned_amount = div_ceil(amount, alignment) * alignment;
    std::byte* new_top = mTop + aligned_amount;
    if(new_top > mBackingMemory + mCapacity) {
        std::size_t used = mTop - mBackingMemory;
        fprintf(stderr, "[Stack OOM] Failed to allocate '%s': requested=%zu MB, used=%zu MB, capacity=%zu MB\n",
                name, aligned_amount / (1024*1024), used / (1024*1024), mCapacity / (1024*1024));
        // Print recent allocations
        fprintf(stderr, "[Stack OOM] Recent allocations:\n");
        size_t start = mAlloc.size() > 10 ? mAlloc.size() - 10 : 0;
        for (size_t i = start; i < mAlloc.size(); ++i) {
            fprintf(stderr, "  - %s: %zu MB\n", mAlloc[i].Name, mAlloc[i].Amount / (1024*1024));
        }
        throw std::bad_alloc();
    }

    mAlloc.emplace_back(mTop, aligned_amount, name);
    mTop = new_top;
    _track_max();
    return mAlloc.back().Pointer;
}

/**
 * @brief Allocate a Tensor view backed by stack-allocated memory.
 *
 * Computes the total byte size as: product(shape) * sizeof(dtype), allocates that
 * many bytes from the stack, and returns a Tensor referencing the allocated region.
 *
 * @param dtype Element type for the resulting Tensor.
 * @param shape Tensor shape (dimensions). The product of all entries determines element count.
 * @param name  Human-readable label used for high-water-mark/statistics tracking.
 * @return Tensor that references stack-allocated device memory.
 *
 * @throws std::bad_alloc If the underlying byte allocation would exceed capacity.
 */
Tensor DeviceMemoryStack::allocate(ETensorDType dtype, const std::vector<long>& shape, const char* name) {
    std::size_t total = std::accumulate(std::begin(shape), std::end(shape), (long)get_dtype_size(dtype), std::multiplies<>());
    return Tensor::from_pointer(allocate(total, name), mDeviceID, dtype, shape);
}

/**
 * @brief Free the most recent raw allocation (LIFO).
 *
 * Only the last allocation may be freed. Passing any other pointer is a logic error.
 *
 * @param ptr Pointer previously returned by allocate(amount, name).
 *
 * @throws std::logic_error If there are no allocations to free.
 * @throws std::logic_error If @p ptr is not the most-recent allocation.
 */
void DeviceMemoryStack::free(std::byte* ptr) {
    if(mAlloc.empty()) {
        throw std::logic_error("DeviceMemoryStack::free_left called with empty allocation list");
    }
    if(mAlloc.back().Pointer != ptr) {
        throw std::logic_error("DeviceMemoryStack::free_left called with wrong pointer");
    }
    mTop = mAlloc.back().Pointer;
    mAlloc.pop_back();
}

/**
 * @brief Return allocation stats for the maximum-utilization (high-water) point.
 *
 * The returned vector contains (name, bytes) pairs for allocations present at the
 * time of peak utilization.
 *
 * @return Vector of (allocation name, allocation size in bytes).
 */
std::vector<std::pair<std::string, long>> DeviceMemoryStack::get_allocation_stats() const {
    std::vector<std::pair<std::string, long>> result;
    for (auto& [ptr, amount, name]: get_high_mark()) {
        result.emplace_back(name, amount);
    }
    return result;
}

/**
 * @brief Update high-water-mark bookkeeping after an allocation.
 *
 * If current utilization exceeds the previous maximum, records the new maximum
 * and snapshots the current allocation list.
 */
void DeviceMemoryStack::_track_max() {
    if(bytes_used() > mMaxUtilization) {
        mMaxUtilization = bytes_used();
        mHighMark = mAlloc;
    }
}

/**
 * @brief Bytes remaining in the backing region.
 *
 * @return Number of unused bytes (capacity - used).
 */
std::size_t DeviceMemoryStack::unused_capacity() const {
    return mCapacity - (mTop - mBackingMemory);
}

/**
 * @brief Bytes currently allocated (in use).
 *
 * @return Number of bytes currently used by allocations.
 */
std::size_t DeviceMemoryStack::bytes_used() const {
    return mCapacity - unused_capacity();
}

/**
 * @brief Maximum number of bytes ever simultaneously allocated.
 *
 * @return Peak utilization in bytes since construction.
 */
std::size_t DeviceMemoryStack::max_utilization() const {
    return mMaxUtilization;
}

/**
 * @brief Free the most recent allocation associated with a Tensor (LIFO).
 *
 * Frees @p tensor.Data and sets it to nullptr. The tensor being freed must correspond
 * to the most recent allocation; otherwise this is a logic error (same rules as free(ptr)).
 *
 * @param tensor Tensor whose Data pointer was obtained from this stack.
 *
 * @throws std::logic_error If the allocation order is violated or there is nothing to free.
 */
void DeviceMemoryStack::free(Tensor& tensor) {
    free(tensor.Data);
    tensor.Data = nullptr;
}

/**
 * @brief Get the device identifier associated with this stack.
 *
 * @return Device id passed at construction time.
 */
int DeviceMemoryStack::device_id() const {
    return mDeviceID;
}

/**
 * @brief Save the current stack position for later restoration.
 *
 * Used for CUDA graph compatibility: save the stack state before graph capture,
 * then restore before each graph replay to ensure temp_alloc returns the same
 * memory addresses that were captured in the graph.
 *
 * @return Checkpoint containing the current stack position.
 */
DeviceMemoryStack::Checkpoint DeviceMemoryStack::checkpoint() const {
    return Checkpoint{mTop, mAlloc.size()};
}

/**
 * @brief Restore the stack to a previously saved checkpoint.
 *
 * Resets the stack top pointer and allocation list to the saved state.
 * This ensures subsequent allocations return the same addresses as they did
 * after the checkpoint was taken.
 *
 * @param cp Checkpoint previously obtained from checkpoint().
 *
 * @throws std::logic_error If the checkpoint is invalid (top pointer out of range).
 */
void DeviceMemoryStack::restore(const Checkpoint& cp) {
    if (cp.top < mBackingMemory || cp.top > mBackingMemory + mCapacity) {
        throw std::logic_error("DeviceMemoryStack::restore: invalid checkpoint");
    }
    mTop = cp.top;
    mAlloc.resize(cp.alloc_count);
}
