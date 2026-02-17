// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// OffloadManager: Group-based CPU/GPU tensor offloading.
//
// Manages offloading of quantized tensors between CPU and GPU memory.
// Tensors are organized into groups (e.g., one group per expert in MoE,
// or one group per transformer layer). The manager handles:
//
// - Registering tensors to groups
// - Loading groups to GPU on demand
// - Unloading groups to CPU when GPU memory is needed
// - LRU eviction when max resident groups is exceeded
// - Async prefetching for overlapping compute and transfer

#ifndef SUROGATE_SRC_RUNTIME_QLORA_OFFLOAD_MANAGER_H
#define SUROGATE_SRC_RUNTIME_QLORA_OFFLOAD_MANAGER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime.h>

#include "utilities/tensor.h"
#include "utilities/allocator.h"
#include "runtime/qlora/quantized_tensor.h"

namespace qlora {

/// Configuration for the OffloadManager.
struct OffloadConfig {
    /// Maximum number of groups that can be resident on GPU simultaneously.
    /// When exceeded, the least recently used group is evicted.
    /// 0 = unlimited (all groups on GPU, no offloading).
    int max_resident_groups = 0;

    /// Whether to enable async prefetching.
    /// When enabled, the manager prefetches the next group while the current
    /// group is being processed.
    bool enable_prefetch = true;

    /// Number of groups to prefetch ahead.
    int prefetch_ahead = 1;

    /// Whether to use pinned (page-locked) host memory for CPU-side storage.
    /// Pinned memory enables faster CPU→GPU transfers via DMA.
    bool use_pinned_memory = true;

    /// CUDA device ID for GPU-side allocations.
    int device_id = 0;
};

/// State of a group in the offload manager.
enum class GroupState : int {
    UNLOADED = 0,  // All tensors on CPU
    LOADING,       // Transfer in progress (CPU → GPU)
    RESIDENT,      // All tensors on GPU, ready for use
    UNLOADING,     // Transfer in progress (GPU → CPU)
};

/// Statistics for an offload group.
struct GroupStats {
    int group_id = -1;
    GroupState state = GroupState::UNLOADED;
    int num_tensors = 0;
    size_t total_bytes = 0;      // Total quantized data bytes
    int64_t last_access_step = 0; // Step counter for LRU
    int load_count = 0;           // Number of times loaded to GPU
};

/// Abstract interface for group-based offloading.
///
/// Usage pattern:
///   1. Create manager with config
///   2. Register tensors to groups: register_tensor(qt, group_id)
///   3. Before using a group: load_group(group_id, stream)
///   4. Use tensors (dequantize, matmul, etc.)
///   5. Optionally prefetch next group: prefetch_group(next_id, stream)
///   6. When done with all groups in a step: new_step()
///
/// The manager automatically handles eviction when GPU memory is full.
class OffloadManager {
public:
    virtual ~OffloadManager() = default;

    /// Register a quantized tensor with a group.
    ///
    /// The tensor's data/scales/meta buffers must already be allocated
    /// (on CPU if offloading is active, on GPU if max_resident_groups == 0).
    ///
    /// @param tensor    The quantized tensor to register (stored by pointer)
    /// @param group_id  Group identifier (e.g., expert index, layer index)
    /// @param name      Human-readable name for debugging
    virtual void register_tensor(
        QuantizedTensor* tensor,
        int group_id,
        const std::string& name = "") = 0;

    /// Load a group's tensors from CPU to GPU.
    ///
    /// If the group is already resident, this is a no-op (but updates LRU).
    /// If GPU memory is full (max_resident_groups exceeded), evicts the LRU group.
    ///
    /// @param group_id  Group to load
    /// @param stream    CUDA stream for async transfers
    /// @return true if the group is now resident (or was already)
    virtual bool load_group(int group_id, cudaStream_t stream) = 0;

    /// Unload a group's tensors from GPU to CPU.
    ///
    /// Transfers quantized data back to CPU pinned memory and frees GPU buffers.
    /// No-op if the group is already unloaded.
    ///
    /// @param group_id  Group to unload
    /// @param stream    CUDA stream for async transfers
    virtual void unload_group(int group_id, cudaStream_t stream) = 0;

    /// Prefetch a group to GPU for future use.
    ///
    /// Initiates an async load on a secondary stream. The next call to
    /// load_group() for this group will synchronize on the transfer.
    ///
    /// @param group_id  Group to prefetch
    /// @param stream    CUDA stream for async prefetch (should differ from compute stream)
    virtual void prefetch_group(int group_id, cudaStream_t stream) = 0;

    /// Signal the start of a new training step.
    ///
    /// Updates internal counters for LRU tracking. Should be called once
    /// at the beginning of each forward pass.
    virtual void new_step() = 0;

    /// Check if a group is currently resident on GPU.
    [[nodiscard]] virtual bool is_resident(int group_id) const = 0;

    /// Get the current state of a group.
    [[nodiscard]] virtual GroupState get_state(int group_id) const = 0;

    /// Get statistics for a group.
    [[nodiscard]] virtual GroupStats get_stats(int group_id) const = 0;

    /// Get the total number of registered groups.
    [[nodiscard]] virtual int num_groups() const = 0;

    /// Get the number of currently resident groups on GPU.
    [[nodiscard]] virtual int num_resident() const = 0;

    /// Get the total GPU memory used by resident groups (bytes).
    [[nodiscard]] virtual size_t gpu_memory_used() const = 0;

    /// Get the total CPU memory used by offloaded groups (bytes).
    [[nodiscard]] virtual size_t cpu_memory_used() const = 0;

    /// Update max_resident_groups at runtime (e.g., after import when GPU state is known).
    virtual void set_max_resident_groups(int max_groups) = 0;

    /// Get the current max_resident_groups setting.
    [[nodiscard]] virtual int max_resident_groups() const = 0;

    /// Get max bytes among all groups (for auto-tune calculations).
    [[nodiscard]] virtual size_t max_group_bytes() const = 0;
};

/// Create an OffloadManager with the given configuration.
///
/// @param config     Offload configuration
/// @param allocator  Tensor allocator for GPU/CPU memory
/// @return           Offload manager instance (never null)
std::unique_ptr<OffloadManager> create_offload_manager(
    const OffloadConfig& config,
    const std::shared_ptr<TensorAllocator>& allocator);

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_OFFLOAD_MANAGER_H
