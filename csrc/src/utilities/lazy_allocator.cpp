// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_allocator.h"
#include "utilities/tensor.h"
#include "utilities/allocator.h"
#include "utilities/stack.h"

/**
 * @brief Register a tensor for deferred allocation.
 *
 * The tensor metadata (rank, dtype, shape) is set immediately, but no backing
 * storage is assigned yet (`target->Data` is set to nullptr). The tensor pointer
 * is recorded internally and will receive a slice of a single contiguous backing
 * allocation once commit() is called.
 *
 * @param target Pointer to the tensor to configure and later assign storage to.
 *               Must remain valid until the next call to commit() and must not
 *               be committed twice without re-registering.
 * @param dtype  Element data type of the tensor to be allocated.
 * @param shape  Tensor shape (sizes per dimension). Its length defines Rank.
 */
void LazyAllocator::allocate(Tensor* target, ETensorDType dtype, const std::vector<long>& shape) {
    target->Data = nullptr;
    target->Rank = shape.size();
    target->DType = dtype;
    std::copy(shape.begin(), shape.end(), target->Sizes.begin());
    mTargets.push_back(target);
}

/**
 * @brief Allocate one contiguous backing tensor and assign slices to all registered targets.
 *
 * Computes the required total byte size by rounding each target's byte size up
 * to a fixed page size, allocates a single BYTE tensor of that total size using
 * the provided allocator, then assigns each target a disjoint page-aligned slice
 * within the backing allocation. Clears the internal target list afterwards.
 *
 * @param storage Allocator used to create the backing tensor.
 * @param type    Allocation type passed through to the allocator (e.g. host/device/managed).
 * @param name    Debug/diagnostic name for the backing allocation (may be null depending on allocator).
 * @return        The backing BYTE tensor owning the storage for all committed targets.
 *                The returned tensor must outlive all target tensors using its memory.
 */
Tensor LazyAllocator::commit(TensorAllocator& storage, EAllocationType type, const char* name) {
    std::size_t total_size = 0;
    constexpr std::size_t page_size = 4096;
    for (auto& target : mTargets) {
        std::size_t tgt_size = div_ceil(target->bytes(), page_size) * page_size;
        total_size += tgt_size;
    }

    Tensor backing = storage.allocate(ETensorDType::BYTE, name, type, {(long)total_size});
    auto* ptr = backing.get<std::byte>();
    for (auto& target : mTargets) {
        target->Data = ptr;
        target->Device = backing.Device;
        ptr += div_ceil(target->bytes(), page_size) * page_size;
    }

    mTargets.clear();

    return backing;
}

/**
 * @brief Allocate one contiguous backing tensor from a stack allocator and assign slices to targets.
 *
 * Same behavior as the TensorAllocator overload: sizes are rounded up to a fixed
 * page size, a single BYTE backing tensor is allocated, targets are assigned
 * page-aligned slices, and the internal target list is cleared.
 *
 * @param storage Stack-style allocator used to obtain the backing tensor.
 * @param name    Debug/diagnostic name for the backing allocation.
 * @return        The backing BYTE tensor owning the storage for all committed targets.
 *                The returned tensor must outlive all target tensors using its memory.
 */
Tensor LazyAllocator::commit(DeviceMemoryStack& storage, const char* name) {
    std::size_t total_size = 0;
    constexpr std::size_t page_size = 4096;
    for (auto& target : mTargets) {
        std::size_t tgt_size = div_ceil(target->bytes(), page_size) * page_size;
        total_size += tgt_size;
    }

    Tensor backing = storage.allocate(ETensorDType::BYTE, {(long)total_size}, name);
    auto* ptr = backing.get<std::byte>();
    for (auto& target : mTargets) {
        target->Data = ptr;
        target->Device = backing.Device;
        ptr += div_ceil(target->bytes(), page_size) * page_size;
    }

    mTargets.clear();

    return backing;
}
