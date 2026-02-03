// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "allocator.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_map>

#include <fmt/core.h>

#include "utilities/gpu_info.h"

TensorAllocator::TensorAllocator(TensorAllocator&&) noexcept = default;
TensorAllocator& TensorAllocator::operator=(TensorAllocator&&) noexcept = default;

struct sTotalAllocations {
    long ON_DEVICE = 0;
    long MANAGED = 0;
    long PINNED = 0;
    long ON_HOST = 0;
    long WRITE_CMB = 0;

    /**
     * @brief Access the byte counter for a given allocation kind.
     *
     * @param kind Allocation type to access.
     * @return Reference to the counter corresponding to @p kind.
     * @throws std::logic_error If @p kind is not a recognized allocation type.
     */
    long& operator[](EAllocationType kind)
    {
        switch (kind) {
            case EAllocationType::ON_DEVICE: return ON_DEVICE;
            case EAllocationType::MANAGED: return MANAGED;
            case EAllocationType::PINNED: return PINNED;
            case EAllocationType::WRITE_CMB: return WRITE_CMB;
            case EAllocationType::ON_HOST: return ON_HOST;
            default: throw std::logic_error("Unknown allocation type");
        }
    }
};

struct TensorAllocator::sAllocStats
{
    std::string Context = "";
    std::unordered_map<std::string, sTotalAllocations> TensorStats;
    std::unordered_map<std::string, sTotalAllocations> ContextStats;
};

/**
 * @brief Allocate raw storage for a tensor on the requested memory kind and wrap it in a Tensor object.
 *
 * The returned Tensor owns a newly allocated memory region and records the current CUDA device id
 * for device/managed allocations. Host allocations use did = -1.
 *
 * @tparam Container Container type providing .size() and iteration over dimension sizes (e.g., std::vector<long>).
 * @param dtype Element type of the tensor.
 * @param kind Allocation kind (device, managed, pinned, write-combined, or host).
 * @param shape Tensor shape (rank = shape.size()).
 * @return A Tensor describing the allocated storage and metadata.
 * @throws std::runtime_error If tensor rank exceeds MAX_TENSOR_DIM.
 * @throws cuda_error On CUDA allocation failures (e.g., out-of-memory).
 */
template<typename Container>
Tensor allocate_tensor(ETensorDType dtype, EAllocationType kind, const Container& shape)
{
    if(shape.size() > MAX_TENSOR_DIM) {
        throw std::runtime_error("Tensor rank too large");
    }

    int did;
    CUDA_CHECK(cudaGetDevice(&did));

    int rank = narrow<int>(shape.size());
    std::size_t total = std::accumulate(std::begin(shape), std::end(shape), 1l, std::multiplies<>());
    std::byte* ptr;
    if(kind == EAllocationType::ON_DEVICE) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), total * get_dtype_size(dtype)));
    } else if(kind == EAllocationType::MANAGED) {
        CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&ptr), total * get_dtype_size(dtype)));
    } else if(kind == EAllocationType::PINNED) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ptr), total * get_dtype_size(dtype), cudaHostAllocMapped));
        did = -1;
    }  else if(kind == EAllocationType::WRITE_CMB) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ptr), total * get_dtype_size(dtype), cudaHostAllocWriteCombined | cudaHostAllocMapped));
        did = -1;
    } else {
        ptr = new std::byte[total * get_dtype_size(dtype)];
        did = -1;
    }
    std::array<long, MAX_TENSOR_DIM> sizes{};
    std::copy(shape.begin(), shape.end(), sizes.begin());
    std::fill(sizes.begin() + shape.size(), sizes.end(), 1);

    return Tensor{dtype, sizes, ptr, nullptr, rank, did};
}

/**
 * @brief Update allocation statistics for a named entry.
 *
 * @param target Map of name -> allocation totals to update.
 * @param name Name/key of the tensor/context whose counters should be updated.
 * @param kind Allocation kind to increment.
 * @param bytes Number of bytes to add to the counter.
 */
void record_stats(std::unordered_map<std::string, sTotalAllocations>& target, std::string name, EAllocationType kind, long bytes) {
    target[name][kind] += narrow<long>(bytes);
}

/**
 * @brief Allocate a tensor with explicit allocation kind and initializer_list shape.
 *
 * @param dtype Tensor element type.
 * @param name Logical name used for stats and error reporting.
 * @param kind Allocation kind (device/managed/pinned/write-combined/host).
 * @param shape Tensor dimensions.
 * @return Allocated Tensor.
 * @throws std::runtime_error On CUDA OOM (with enriched message) or rank/other allocation errors.
 */
Tensor TensorAllocator::allocate(ETensorDType dtype, const char* name, EAllocationType kind, const std::initializer_list<long>& shape) {
    return allocate_impl(dtype, name, kind, shape);
}

/**
 * @brief Allocate a tensor with explicit allocation kind and vector shape.
 *
 * @param dtype Tensor element type.
 * @param name Logical name used for stats and error reporting.
 * @param kind Allocation kind (device/managed/pinned/write-combined/host).
 * @param shape Tensor dimensions.
 * @return Allocated Tensor.
 * @throws std::runtime_error On CUDA OOM (with enriched message) or rank/other allocation errors.
 */
Tensor TensorAllocator::allocate(ETensorDType dtype, const char* name, EAllocationType kind, const std::vector<long>& shape) {
    return allocate_impl(dtype, name, kind, shape);
}

/**
 * @brief Allocate a device tensor (default kind) with initializer_list shape.
 *
 * @param dtype Tensor element type.
 * @param name Logical name used for stats and error reporting.
 * @param shape Tensor dimensions.
 * @return Allocated Tensor on device memory.
 */
Tensor TensorAllocator::allocate(ETensorDType dtype, const char* name, const std::initializer_list<long>& shape) {
    return allocate_impl(dtype, name, EAllocationType::ON_DEVICE, shape);
}

/**
 * @brief Allocate a device tensor (default kind) with vector shape.
 *
 * @param dtype Tensor element type.
 * @param name Logical name used for stats and error reporting.
 * @param shape Tensor dimensions.
 * @return Allocated Tensor on device memory.
 */
Tensor TensorAllocator::allocate(ETensorDType dtype, const char* name, const std::vector<long>& shape) {
    return allocate_impl(dtype, name, EAllocationType::ON_DEVICE, shape);
}

/**
 * @brief Allocate a shard (partition) of a tensor along the first dimension.
 *
 * The first dimension is divided exactly by @p num_shards; each shard gets shape[0]/num_shards rows.
 *
 * @param dtype Tensor element type.
 * @param shard_idx Index of this shard in [0, num_shards).
 * @param num_shards Total number of shards.
 * @param name Logical name used for stats and error reporting.
 * @param shape Full (unsharded) tensor shape; must have at least one dimension.
 * @param kind Allocation kind for the shard storage.
 * @return TensorShard containing the allocated shard tensor and sharding metadata.
 * @throws std::runtime_error / std::logic_error If div_exact fails (shape not divisible) or allocation fails.
 */
TensorShard TensorAllocator::allocate_shard(ETensorDType dtype, int shard_idx, int num_shards, const char* name, const std::vector<long>& shape,  EAllocationType kind) {
    std::vector<long> shard_shape(shape);
    shard_shape[0] = div_exact(shape[0], (long)num_shards);
    return TensorShard(allocate(dtype, name, kind, shard_shape), shard_idx, num_shards, shape);
}

/**
 * @brief Implementation helper for allocate() overloads.
 *
 * Allocates storage via allocate_tensor(), records pointer for later cleanup, updates per-tensor and per-context
 * statistics, and triggers the optional allocation callback.
 *
 * @tparam Container Container type providing .size() and iteration over dimension sizes.
 * @param dtype Tensor element type.
 * @param name Logical name used for stats and error reporting.
 * @param kind Allocation kind.
 * @param shape Tensor dimensions.
 * @return Allocated Tensor.
 * @throws std::runtime_error On CUDA OOM with detailed guidance; rethrows other cuda_error exceptions.
 */
template<typename Container>
Tensor TensorAllocator::allocate_impl(ETensorDType dtype, const char* name, EAllocationType kind, const Container& shape) {
    try {
        Tensor allocated = allocate_tensor(dtype, kind, shape);
        const char* safe_name = name ? name : "<unnamed>";
        m_Pointers.emplace_back(sAllocationData{kind, allocated.Data, narrow<long>(allocated.bytes()), safe_name, m_Stats->Context});
        record_stats(m_Stats->TensorStats, name, kind, allocated.bytes());
        if (!m_Stats->Context.empty()){
            record_stats(m_Stats->ContextStats, m_Stats->Context, kind, allocated.bytes());
        }
        if(mCallback) {
            mCallback(m_Stats->Context, name, kind, allocated.bytes());
        }
        return allocated;
    } catch (const cuda_error& error) {
        if(error.code == cudaErrorMemoryAllocation) {
            print_stats();
            std::string shape_str = "[";
            for(auto s: shape) {
                shape_str += std::to_string(s) + ", ";
            }
            shape_str.pop_back();
            shape_str.pop_back();
            shape_str.push_back(']');
            std::string message = fmt::format(
                "Cuda OOM when allocating tensor {} of shape {} with dtype {} in context {}. "
                "Try reducing batch/seq length or offloading optimizer state "
                "with --offload-optimizer (and optionally --offload-gradients).",
                name, shape_str, dtype_to_str(dtype), m_Stats->Context);
            throw std::runtime_error(message);
        }
        throw;
    }
}

/**
 * @brief Construct a TensorAllocator with empty allocation statistics.
 */
TensorAllocator::TensorAllocator() : m_Stats(std::make_unique<sAllocStats>()) {
}

/**
 * @brief Destructor; frees all tracked allocations.
 *
 * Synchronizes the device first, then frees each pointer using the appropriate CUDA/free/delete routine.
 * Never throws: on CUDA errors during cleanup it prints a warning and continues best-effort cleanup.
 */
TensorAllocator::~TensorAllocator() noexcept {
    // Destructors must not throw; CUDA_CHECK throws on error.
    // Best-effort cleanup: report CUDA errors but never terminate the process here,
    // otherwise we can mask the original exception that triggered stack unwinding.
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        fprintf(stderr,
                "WARNING: Cuda error in TensorAllocator::~TensorAllocator() during cudaDeviceSynchronize(): %s\n",
                cudaGetErrorString(sync_status));
        fflush(stderr);
        // Clear sticky error state so frees have a chance to proceed.
        (void)cudaGetLastError();
    }

    int did = 0;
    const cudaError_t dev_status = cudaGetDevice(&did);
    if (dev_status != cudaSuccess) {
        fprintf(stderr,
                "WARNING: Cuda error in TensorAllocator::~TensorAllocator() during cudaGetDevice(): %s\n",
                cudaGetErrorString(dev_status));
        fflush(stderr);
        did = -1;
        (void)cudaGetLastError();
    }
    for (auto& ptr: m_Pointers) {
        const char* kind_str;
        switch (ptr.Kind) {
            case EAllocationType::ON_DEVICE: kind_str = "device"; break;
            case EAllocationType::MANAGED: kind_str = "managed"; break;
            case EAllocationType::PINNED: kind_str = "pinned"; break;
            case EAllocationType::WRITE_CMB: kind_str = "write-combined"; break;
            case EAllocationType::ON_HOST: kind_str = "host"; break;
            default: kind_str = "unknown"; break;
        }

        switch (ptr.Kind) {
            case EAllocationType::ON_DEVICE:
            case EAllocationType::MANAGED: {
                const cudaError_t st = cudaFree(ptr.Pointer);
                if (st != cudaSuccess) {
                    fprintf(stderr,
                            "WARNING: Cuda error on device %d when freeing allocation %p [%s of size %ld]: %s\n",
                            did, ptr.Pointer, kind_str, ptr.Size, cudaGetErrorString(st));
                    fflush(stderr);
                    (void)cudaGetLastError();
                }
                break;
            }
            case EAllocationType::WRITE_CMB:
            case EAllocationType::PINNED: {
                const cudaError_t st = cudaFreeHost(ptr.Pointer);
                if (st != cudaSuccess) {
                    fprintf(stderr,
                            "WARNING: Cuda error on device %d when freeing host allocation %p [%s of size %ld]: %s\n",
                            did, ptr.Pointer, kind_str, ptr.Size, cudaGetErrorString(st));
                    fflush(stderr);
                    (void)cudaGetLastError();
                }
                break;
            }
            case EAllocationType::ON_HOST:
                delete[] ptr.Pointer;
                break;
        }
    }
}

/**
 * @brief Memory category with name and optimization hints.
 */
struct MemoryCategory {
    const char* name;
    const char* hints;
};

/**
 * @brief Categorize a tensor name into a memory category for grouped reporting.
 *
 * Categories help users understand which --offload-* flags to use.
 *
 * @param name Tensor name.
 * @return Category with name and optimization hints.
 */
static MemoryCategory categorize_tensor(const std::string& name) {
    // Optimizer state (momentum/variance)
    if (name.find("opt_m_") == 0 || name.find("opt_v_") == 0) {
        return {"optimizer", "enable 'offload-optimizer'"};
    }
    // Gradients (d_ prefix)
    if (name.find("d_") == 0) {
        return {"gradients", "enable 'shard-gradients' and 'offload-gradients'"};
    }
    // FP8/quantization buffers
    if (name.find("fp8_") == 0 || name.find("fp4_") == 0 || name.find("quant") != std::string::npos ||
        name.find("_q") == name.size() - 2 || name.find("_scales") != std::string::npos) {
        return {"quants", "enable 'offload-quants' and 'persistent-quants'"};
    }
    // Workspace buffers
    if (name.find("_ws") != std::string::npos || name.find("workspace") != std::string::npos) {
        return {"workspace", "(not offloadable)"};
    }
    // Activations
    if (name.find("act_") == 0 || name.find("cache") != std::string::npos ||
        name.find("residual") != std::string::npos || name.find("logits") != std::string::npos) {
        return {"activations", "enable 'recompute'"};
    }
    // Model weights
    return {"weights", "enable 'offload-master'"};
}

/**
 * @brief Print a summary of large device allocations tracked per tensor name.
 *
 * Groups allocations by category (weights, gradients, optimizer, quants, workspace)
 * to help users understand which --offload-* flags can reduce memory usage.
 */
void TensorAllocator::print_stats() const {
    // Group allocations by category
    struct CategoryData {
        std::vector<std::pair<std::string, long>> tensors;
        long total = 0;
        const char* hints = "";
    };
    std::unordered_map<std::string, CategoryData> categories;

    for (auto& [name, amount]: m_Stats->TensorStats) {
        if (amount.ON_DEVICE < total_allocation() / 1024) continue;  // skip tiny tensors
        auto cat = categorize_tensor(name);
        auto& data = categories[cat.name];
        data.tensors.emplace_back(name, amount.ON_DEVICE);
        data.total += amount.ON_DEVICE;
        data.hints = cat.hints;
    }

    // Sort categories by total size (descending)
    std::vector<std::pair<std::string, CategoryData*>> sorted_categories;
    for (auto& [name, data] : categories) {
        sorted_categories.emplace_back(name, &data);
    }
    std::sort(sorted_categories.begin(), sorted_categories.end(),
              [](const auto& a, const auto& b) { return a.second->total > b.second->total; });

    // Print each category
    for (const auto& [category, data] : sorted_categories) {
        std::cerr << "\n=== " << category << " (" << data->total / 1024 / 1024 << " MiB) [" << data->hints << "] ===\n";

        // Sort tensors within category by size (descending)
        std::sort(data->tensors.begin(), data->tensors.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& [name, bytes] : data->tensors) {
            if (bytes >= 1024 * 1024 * 20) {
                std::cerr << "  " << name << ": " << bytes / 1024 / 1024 << " MiB\n";
            } else {
                std::cerr << "  " << name << ": " << bytes / 1024 << " KiB\n";
            }
        }
    }
    std::cerr << "\n";
}

/**
 * @brief Compute total bytes allocated across all kinds tracked by this allocator.
 *
 * @return Total allocated bytes (sum of all tracked pointer sizes).
 */
std::size_t TensorAllocator::total_allocation() const {
    std::size_t total = 0;
    for(const auto& ptr: m_Pointers) {
        total += ptr.Size;
    }
    return total;
}

/**
 * @brief Compute total bytes allocated for a specific allocation kind.
 *
 * @param kind Allocation kind to filter by.
 * @return Total allocated bytes for @p kind.
 */
std::size_t TensorAllocator::total_allocation(EAllocationType kind) const {
    std::size_t total = 0;
    for(const auto& ptr: m_Pointers) {
        if(ptr.Kind == kind) {
            total += ptr.Size;
        }
    }
    return total;
}

/**
 * @brief Set the current allocation context name.
 *
 * The context name is used to attribute subsequent allocations to a logical "segment" for reporting.
 *
 * @param ctx New context name.
 */
void TensorAllocator::set_context(const std::string& ctx) {
    m_Stats->Context = ctx;
}

/**
 * @brief Get the current allocation context name.
 *
 * @return Reference to the current context string.
 */
const std::string& TensorAllocator::get_context() const {
    return m_Stats->Context;
}

/**
 * @brief RAII helper that sets an allocator context for the lifetime of the monitor.
 *
 * On construction, saves the previous context and sets the allocator context to @p name.
 *
 * @param name Context name to set while the monitor is alive.
 * @param alloc Allocator whose context should be managed.
 */
TensorAllocator::AllocationMonitor::AllocationMonitor(const std::string& name, TensorAllocator* alloc) :
    mName(name), mAllocator(alloc), mParent(alloc->get_context()) {
    alloc->set_context(mName);
}

/**
 * @brief Move-construct an AllocationMonitor.
 *
 * Transfers ownership of the active scope to the new object and disables the moved-from monitor.
 *
 * @param other Monitor to move from.
 */
TensorAllocator::AllocationMonitor::AllocationMonitor(AllocationMonitor&& other) noexcept
    : mName(std::move(other.mName)),
      mParent(std::move(other.mParent)),
      mAllocator(other.mAllocator),
      mActive(other.mActive) {
    other.mAllocator = nullptr;
    other.mActive = false;
}

/**
 * @brief Move-assign an AllocationMonitor.
 *
 * If the current monitor is active, it performs best-effort restoration of its parent context before taking
 * over @p other. The moved-from monitor is disabled.
 *
 * @param other Monitor to move from.
 * @return *this.
 */
TensorAllocator::AllocationMonitor& TensorAllocator::AllocationMonitor::operator=(AllocationMonitor&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    // Best-effort cleanup of the current active scope before stealing.
    if (mActive && mAllocator) {
        mAllocator->set_context(mParent);
    }
    mName = std::move(other.mName);
    mParent = std::move(other.mParent);
    mAllocator = other.mAllocator;
    mActive = other.mActive;
    other.mAllocator = nullptr;
    other.mActive = false;
    return *this;
}

/**
 * @brief Destructor; restores the allocator's previous context if still active.
 *
 * Never throws. If improper nesting is detected (current allocator context differs from mName),
 * it prints a warning and still restores the parent.
 */
TensorAllocator::AllocationMonitor::~AllocationMonitor() noexcept {
    if (!mActive || !mAllocator) {
        return;
    }
    if (mAllocator->get_context() != mName) {
        // Never throw from a destructor. This can happen during exception unwinding and would
        // mask the original error with a terminate()/abort.
        fprintf(stderr,
                "WARNING: AllocationMonitor improper nesting: expected ctx='%s' but got ctx='%s' (restoring parent='%s')\n",
                mName.c_str(), mAllocator->get_context().c_str(), mParent.c_str());
        fflush(stderr);
    }
    mAllocator->set_context(mParent);
}

/**
 * @brief Return memory usage segmented by recorded allocation contexts.
 *
 * Produces a list of (context-name, segment) entries plus synthetic "Free", optional "Reserved", and "Other".
 * Device memory accounting uses cudaMemGetInfo and get_mem_reserved().
 *
 * @return Vector of segments suitable for visualization/reporting.
 * @throws cuda_error On CUDA API failures.
 */
std::vector<std::pair<std::string, sSegmentMemory>> TensorAllocator::get_allocation_segments() const {
    long sum = 0;
    std::vector<std::pair<std::string, sSegmentMemory>> segments;
    for (const auto& [name, amount]: m_Stats->ContextStats) {
        segments.emplace_back(name, sSegmentMemory{amount.ON_DEVICE, amount.MANAGED, amount.PINNED + amount.WRITE_CMB, amount.ON_HOST});
        sum += amount.ON_DEVICE;
    }
    std::size_t free = 0;
    std::size_t total = 0;
    long reserved = get_mem_reserved();
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    segments.emplace_back("Free", sSegmentMemory{(long)free, 0, 0, 0});
    if(reserved > 0) {
        segments.emplace_back("Reserved", sSegmentMemory{reserved, 0, 0, 0});
    }
    segments.emplace_back("Other", sSegmentMemory{(long)total - (long)free - sum - reserved, 0, 0, 0});
    return segments;
}

/**
 * @brief Set a callback invoked on each successful allocation.
 *
 * The callback receives (context, tensor-name, allocation-kind, bytes).
 *
 * @param cb Callback function; passing an empty function disables callbacks.
 */
void TensorAllocator::set_callback(std::function<void(const std::string&, const std::string&, EAllocationType, std::size_t)> cb) {
    mCallback = std::move(cb);
}

/**
 * @brief Get per-tensor allocation statistics sorted by device memory usage.
 *
 * @return Vector of (tensor_name, device_bytes) pairs sorted by size descending.
 */
std::vector<std::pair<std::string, std::size_t>> TensorAllocator::get_tensor_stats() const {
    std::vector<std::pair<std::string, std::size_t>> result;
    result.reserve(m_Stats->TensorStats.size());
    for (const auto& [name, allocs] : m_Stats->TensorStats) {
        result.emplace_back(name, static_cast<std::size_t>(allocs.ON_DEVICE));
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return result;
}
