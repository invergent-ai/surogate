// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILITIES_LAZY_ALLOCATOR_H
#define SUROGATE_SRC_UTILITIES_LAZY_ALLOCATOR_H

#include <vector>

class Tensor;
class TensorAllocator;
class DeviceMemoryStack;
enum class EAllocationType : int;
enum class ETensorDType : int;

//! \brief Allows allocating tensors lazily from an underlying eager allocator
//! \details The allocate function registers a tensor for allocation, but does not reserve any memory yet.
//! Only after calling commit, the memory is reserved and the tensor is ready to be used. All allocations
//! done via allocate are coalesced into a single memory allocation from the backing storage.
class LazyAllocator {
public:
    //! Registers a tensor for lazy allocation. Sets the metadata of `target` (rank, dtype, sizes) but leaves
    //! the data pointer null. Actual memory allocation happens during `commit()`.
    //! Note that `target` will be modified by `commit()`, so it must remain valid.
    void allocate(Tensor* target, ETensorDType dtype, const std::vector<long>& shape);

    //! Allocates memory for all registered tensors from `storage`, using `type` and `name` to request a sufficiently
    //! large area of memory from the underlying `TensorAllocator`.
    //! Returns the backing tensor that must be kept alive for the lifetime of all allocated tensors.
    Tensor commit(TensorAllocator& storage, EAllocationType type, const char* name);

    //! Allocates memory for all registered tensors from `storage`, using `name` to request a sufficiently
    //! large area of memory from the underlying `DeviceMemoryStack`.
    //! Returns the backing tensor that must be kept alive for the lifetime of all allocated tensors.
    Tensor commit(DeviceMemoryStack& storage, const char* name);

private:
    std::vector<Tensor*> mTargets;
};

#endif  //SUROGATE_SRC_UTILITIES_LAZY_ALLOCATOR_H
