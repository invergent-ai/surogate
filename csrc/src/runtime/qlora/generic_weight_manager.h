// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// GenericWeightManager: Architecture-agnostic quantized weight storage and access.
//
// Stores quantized weights in a flat name-based map and provides lazy
// dequantization on access. Works with any quantization format via IQuantizer
// and supports group-based CPU/GPU offloading via OffloadManager.
//
// Usage:
//   1. Create with config and quantizer
//   2. register_weight() or import_and_quantize() to populate
//   3. get() to dequantize and access weights by name
//   4. new_step() at start of each training step to invalidate cache

#ifndef SUROGATE_SRC_RUNTIME_QLORA_GENERIC_WEIGHT_MANAGER_H
#define SUROGATE_SRC_RUNTIME_QLORA_GENERIC_WEIGHT_MANAGER_H

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "utilities/tensor.h"
#include "utilities/allocator.h"
#include "runtime/qlora/quantized_tensor.h"
#include "runtime/qlora/generic_quantizer.h"
#include "runtime/qlora/offload_manager.h"

class SafeTensorsReader;

namespace dsl { class DslWeightLoader; }

namespace qlora {

/// Configuration for GenericWeightManager.
struct GenericWeightManagerConfig {
    /// Maximum size of the dequantization buffer cache (in number of weights).
    /// When exceeded, least-recently-used dequant buffers are freed and
    /// returned to the buffer pool for reuse by other weights.
    /// 0 = unlimited (all dequant buffers kept resident, one per weight).
    int max_dequant_cache_size = 0;

    /// Whether to enable offloading.
    bool enable_offloading = false;

    /// Offload configuration (only used if enable_offloading is true).
    OffloadConfig offload_config;

    /// CUDA device ID.
    int device_id = 0;
};

/// A managed weight entry: quantized storage + lazy dequant buffer.
struct ManagedWeight {
    /// The quantized tensor (NF4/FP8/FP4 data + scales + metadata).
    QuantizedTensor quantized;

    /// BF16 dequantization buffer (same shape as original weight).
    /// In unlimited mode (max_dequant_cache_size == 0): pre-allocated at registration.
    /// In pooled mode (max_dequant_cache_size > 0): acquired from pool on access.
    Tensor dequant_buffer;

    /// Whether the dequant buffer is currently valid (populated and up-to-date).
    bool dequant_valid = false;

    /// Step at which dequant_buffer was last populated.
    int64_t dequant_step = -1;

    /// Offload group ID (-1 = no offloading, always resident).
    int offload_group = -1;

    /// Whether this weight should be quantized or stored in full precision.
    bool is_quantized_weight = true;

    /// Full-precision (BF16) tensor for weights that skip quantization
    /// (e.g., layer norms, biases, embedding).
    Tensor full_precision;

    /// Weight dimensions (stored for pool buffer acquisition in pooled mode).
    /// M is flattened rows (E*per_M for 3D), K is columns.
    int M = 0;
    int K = 0;

    /// Full tensor shape for dequant buffer (e.g., {E, per_M, K} for experts).
    /// Empty means use default {M, K}.
    std::vector<long> dequant_shape;

    /// Whether this weight currently holds a buffer from the pool.
    bool has_pool_buffer = false;

    /// Iterator into the LRU list (valid only when has_pool_buffer == true).
    /// Points to this weight's name in the LRU list for O(1) removal.
    std::list<std::string>::iterator lru_it;
};

/// Pool of reusable dequantization buffers, keyed by shape (M, K).
///
/// When max_dequant_cache_size > 0, weights don't get pre-allocated dequant
/// buffers. Instead, they acquire buffers from this pool on access and release
/// them when evicted by LRU. Same-shape buffers are reused across weights.
class DequantBufferPool {
public:
    explicit DequantBufferPool(std::shared_ptr<TensorAllocator> allocator);

    /// Acquire a buffer for the given shape. Reuses a pooled buffer if
    /// available (by element count), otherwise allocates a new one.
    Tensor acquire(int M, int K, const std::vector<long>& shape = {});

    /// Release a buffer back to the pool for reuse (keyed by element count).
    void release(int M, int K, Tensor buffer);

    /// Get the number of buffers currently in the pool (not in use).
    [[nodiscard]] int pooled_count() const;

    /// Get the total bytes of pooled (unused) buffers.
    [[nodiscard]] size_t pooled_bytes() const;

private:
    /// Shape key for pool lookup: packs (M, K) into a single uint64_t.
    static uint64_t shape_key(int M, int K) {
        return (static_cast<uint64_t>(M) << 32) | static_cast<uint32_t>(K);
    }

    std::shared_ptr<TensorAllocator> mAllocator;
    std::unordered_map<uint64_t, std::vector<Tensor>> mPool;
};

/// Architecture-agnostic quantized weight manager.
///
/// Stores weights by name and provides lazy dequantization on access.
/// Supports any quantization format via the IQuantizer interface.
///
/// When max_dequant_cache_size > 0, uses a DequantBufferPool with LRU
/// eviction to limit GPU memory usage for dequant buffers. This is critical
/// for large models where keeping all dequant buffers resident would exceed
/// GPU memory.
class GenericWeightManager {
public:
    /// @param config       Manager configuration.
    /// @param quantizer    Quantizer for format-specific quantize/dequantize.
    /// @param allocator    Tensor allocator for GPU/CPU memory.
    GenericWeightManager(
        const GenericWeightManagerConfig& config,
        std::unique_ptr<IQuantizer> quantizer,
        std::shared_ptr<TensorAllocator> allocator);

    ~GenericWeightManager();

    // Non-copyable
    GenericWeightManager(const GenericWeightManager&) = delete;
    GenericWeightManager& operator=(const GenericWeightManager&) = delete;

    // =========================================================================
    // Weight registration
    // =========================================================================

    /// Register a weight for quantized storage.
    ///
    /// Allocates quantized storage. In unlimited mode (max_dequant_cache_size
    /// == 0), also pre-allocates a dequant buffer. In pooled mode, the dequant
    /// buffer is acquired lazily from the pool on first access.
    ///
    /// @param name           Unique weight name (e.g., "blocks[0].qkv_weight")
    /// @param M              Number of rows (flattened: E*per_M for 3D experts)
    /// @param K              Number of columns
    /// @param offload_group  Offload group ID (-1 = no offloading)
    /// @param shape          Full tensor shape for dequant buffer (e.g., {E, M, K}).
    ///                       If empty, uses {M, K}. Allows 3D expert weights to
    ///                       retain their [E, per_M, K] shape after dequantization.
    void register_weight(
        const std::string& name,
        int M, int K,
        int offload_group = -1,
        const std::vector<long>& shape = {});

    /// Register a full-precision weight (not quantized).
    ///
    /// Used for weights that should not be quantized (layer norms, biases, etc.).
    /// The tensor is stored as-is and returned directly by get().
    ///
    /// @param name    Unique weight name
    /// @param tensor  The full-precision tensor to store
    void register_full_precision(
        const std::string& name,
        Tensor tensor);

    /// Quantize a BF16 tensor and store it under the given name.
    ///
    /// The weight must have been previously registered via register_weight().
    ///
    /// @param name    Weight name (must match a prior register_weight call)
    /// @param bf16    Input BF16 tensor [M, K]
    /// @param stream  CUDA stream for async quantization
    void quantize_and_store(
        const std::string& name,
        const Tensor& bf16,
        cudaStream_t stream);

    /// Quantize a single expert's BF16 data into a slice of a registered weight.
    ///
    /// Used for per-expert streaming: instead of allocating the full [E, M, K]
    /// temporary buffer, the caller loads and quantizes one expert at a time
    /// into a small [per_M, K] buffer. This method creates a QuantizedTensor
    /// sub-view pointing to the correct offset and quantizes into it.
    ///
    /// Requires that per_expert_M * K is divisible by the quantization block
    /// size so that expert boundaries align with block boundaries.
    ///
    /// @param name           Weight name (must be a registered quantized weight).
    /// @param expert_idx     Expert index (0-based).
    /// @param per_expert_M   Rows per expert (one expert's M dimension).
    /// @param bf16           Input BF16 tensor [per_expert_M, K] for this expert.
    /// @param stream         CUDA stream for async quantization.
    void quantize_expert_slice(
        const std::string& name,
        int expert_idx,
        int per_expert_M,
        const Tensor& bf16,
        cudaStream_t stream);

    // =========================================================================
    // Weight access
    // =========================================================================

    /// Get a weight by name, dequantizing lazily if needed.
    ///
    /// For quantized weights: returns a reference to the BF16 dequant buffer.
    /// For full-precision weights: returns the stored tensor directly.
    ///
    /// The returned reference is valid until the next new_step() call or until
    /// the weight's buffer is evicted (in pooled mode).
    ///
    /// @param name    Weight name
    /// @param stream  CUDA stream for dequantization (if needed)
    /// @return Reference to BF16 tensor (either dequantized or full-precision)
    Tensor& get(const std::string& name, cudaStream_t stream);

    /// Get a weight's quantized tensor (no dequantization).
    ///
    /// @param name  Weight name
    /// @return Pointer to QuantizedTensor, or nullptr if not found
    const QuantizedTensor* get_quantized(const std::string& name) const;

    /// Check if a weight exists.
    [[nodiscard]] bool has_weight(const std::string& name) const;

    /// Get all registered weight names.
    [[nodiscard]] std::vector<std::string> weight_names() const;

    /// Get the offload group ID for a weight (-1 if not offloaded or not found).
    [[nodiscard]] int get_offload_group(const std::string& name) const;

    // =========================================================================
    // Offloading
    // =========================================================================

    /// Prefetch a group's weights to GPU.
    ///
    /// If offloading is enabled, this loads the group's quantized data from
    /// CPU to GPU via the OffloadManager. No-op if offloading is disabled.
    ///
    /// @param group_id  Offload group ID
    /// @param stream    CUDA stream for async transfers
    void prefetch_group(int group_id, cudaStream_t stream);

    // =========================================================================
    // Step management
    // =========================================================================

    /// Signal the start of a new training step.
    ///
    /// Invalidates all dequantization caches, forcing re-dequantization on
    /// next access. Also advances the offload manager's step counter.
    void new_step();

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Total quantized storage bytes (all weights).
    [[nodiscard]] size_t quantized_bytes() const;

    /// Total dequant buffer bytes (all active buffers currently in use).
    [[nodiscard]] size_t dequant_buffer_bytes() const;

    /// Total full-precision storage bytes.
    [[nodiscard]] size_t full_precision_bytes() const;

    /// Number of registered weights.
    [[nodiscard]] int num_weights() const;

    /// Number of quantized weights.
    [[nodiscard]] int num_quantized() const;

    /// Number of full-precision weights.
    [[nodiscard]] int num_full_precision() const;

    /// Number of dequant buffers currently active (held by weights).
    [[nodiscard]] int num_active_dequant_buffers() const;

    /// Get the quantizer.
    [[nodiscard]] IQuantizer* quantizer() const { return mQuantizer.get(); }

    /// Get the offload manager (may be null if offloading disabled).
    [[nodiscard]] OffloadManager* offload_manager() const { return mOffloadManager.get(); }

    /// Whether pooled mode is active (max_dequant_cache_size > 0).
    [[nodiscard]] bool is_pooled() const { return mConfig.max_dequant_cache_size > 0; }

private:
    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Ensure the dequant buffer for a weight is populated.
    void ensure_dequantized(ManagedWeight& entry, const std::string& name, cudaStream_t stream);

    /// Acquire a dequant buffer for a weight in pooled mode.
    void acquire_pool_buffer(ManagedWeight& entry, const std::string& name);

    /// Evict the least recently used dequant buffer back to the pool.
    void evict_lru_buffer();

    /// Promote a weight to the front (most recently used) of the LRU list.
    void touch_lru(ManagedWeight& entry, const std::string& name);

    // =========================================================================
    // Members
    // =========================================================================

    GenericWeightManagerConfig mConfig;
    std::unique_ptr<IQuantizer> mQuantizer;
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unique_ptr<OffloadManager> mOffloadManager;

    /// All registered weights, keyed by name.
    std::unordered_map<std::string, ManagedWeight> mWeights;

    /// Current step counter (incremented by new_step()).
    int64_t mCurrentStep = 0;

    /// CUDA device properties.
    cudaDeviceProp mDeviceProps;

    /// Dequant buffer pool (only used when max_dequant_cache_size > 0).
    std::unique_ptr<DequantBufferPool> mBufferPool;

    /// LRU list: front = most recently used, back = least recently used.
    /// Contains weight names of weights that currently hold a pool buffer.
    std::list<std::string> mDequantLRU;

    /// Number of active dequant buffers (from pool, currently held by weights).
    int mActivePoolBuffers = 0;
};

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_GENERIC_WEIGHT_MANAGER_H
