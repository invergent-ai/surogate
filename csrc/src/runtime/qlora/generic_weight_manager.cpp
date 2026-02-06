// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// GenericWeightManager: Architecture-agnostic quantized weight manager.

#include "runtime/qlora/generic_weight_manager.h"

#include <stdexcept>

#include <fmt/format.h>
#include <cuda_bf16.h>

#include "utilities/utils.h"

namespace qlora {

// =============================================================================
// DequantBufferPool
// =============================================================================

DequantBufferPool::DequantBufferPool(std::shared_ptr<TensorAllocator> allocator)
    : mAllocator(std::move(allocator))
{
}

Tensor DequantBufferPool::acquire(int M, int K, const std::vector<long>& shape) {
    const uint64_t key = shape_key(M, K);
    auto it = mPool.find(key);
    if (it != mPool.end() && !it->second.empty()) {
        Tensor buf = std::move(it->second.back());
        it->second.pop_back();
        // Reshape to desired shape if needed (same total elements, different layout)
        if (!shape.empty() && buf.Rank != static_cast<int>(shape.size())) {
            buf = Tensor::from_pointer(buf.Data, buf.Device, buf.DType, shape);
        }
        return buf;
    }

    // No pooled buffer available - allocate a new one
    std::vector<long> alloc_shape;
    if (!shape.empty()) {
        alloc_shape = shape;
    } else if (K > 0) {
        alloc_shape = {static_cast<long>(M), static_cast<long>(K)};
    } else {
        alloc_shape = {static_cast<long>(M)};
    }
    return mAllocator->allocate(
        ETensorDType::BF16,
        "dequant_pool_buf",
        EAllocationType::ON_DEVICE,
        alloc_shape);
}

void DequantBufferPool::release(int M, int K, Tensor buffer) {
    const uint64_t key = shape_key(M, K);
    mPool[key].push_back(std::move(buffer));
}

int DequantBufferPool::pooled_count() const {
    int count = 0;
    for (const auto& [key, vec] : mPool) {
        count += static_cast<int>(vec.size());
    }
    return count;
}

size_t DequantBufferPool::pooled_bytes() const {
    size_t total = 0;
    for (const auto& [key, vec] : mPool) {
        for (const auto& buf : vec) {
            if (!buf.is_null()) {
                total += buf.bytes();
            }
        }
    }
    return total;
}

// =============================================================================
// Construction / Destruction
// =============================================================================

GenericWeightManager::GenericWeightManager(
    const GenericWeightManagerConfig& config,
    std::unique_ptr<IQuantizer> quantizer,
    std::shared_ptr<TensorAllocator> allocator)
    : mConfig(config)
    , mQuantizer(std::move(quantizer))
    , mAllocator(std::move(allocator))
{
    CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProps, mConfig.device_id));

    if (mConfig.enable_offloading) {
        mOffloadManager = create_offload_manager(
            mConfig.offload_config, mAllocator);
    }

    // Create buffer pool when LRU eviction is enabled
    if (mConfig.max_dequant_cache_size > 0) {
        mBufferPool = std::make_unique<DequantBufferPool>(mAllocator);
    }
}

GenericWeightManager::~GenericWeightManager() {
    // In pooled mode, release any active buffers before destroying the pool
    if (mBufferPool) {
        for (auto& [name, entry] : mWeights) {
            if (entry.has_pool_buffer && !entry.dequant_buffer.is_null()) {
                mBufferPool->release(entry.M, entry.K, std::move(entry.dequant_buffer));
                entry.has_pool_buffer = false;
            }
        }
    }
}

// =============================================================================
// Weight registration
// =============================================================================

void GenericWeightManager::register_weight(
    const std::string& name,
    int M, int K,
    int offload_group,
    const std::vector<long>& shape) {
    if (mWeights.count(name)) {
        throw std::runtime_error(
            fmt::format("GenericWeightManager: weight '{}' already registered", name));
    }

    ManagedWeight entry;
    entry.is_quantized_weight = true;
    entry.offload_group = offload_group;
    entry.M = M;
    entry.K = K;
    entry.dequant_shape = shape;

    // Determine allocation type based on offloading
    EAllocationType alloc_type = EAllocationType::ON_DEVICE;
    if (offload_group >= 0 && mConfig.enable_offloading &&
        mConfig.offload_config.max_resident_groups > 0) {
        // This weight will be offloaded: allocate quantized storage on pinned host
        alloc_type = mConfig.offload_config.use_pinned_memory
            ? EAllocationType::PINNED
            : EAllocationType::ON_HOST;
    }

    // Allocate quantized storage via IQuantizer (uses flat M*K element count)
    mQuantizer->allocate_storage(M, K, entry.quantized, *mAllocator, alloc_type, name);

    // Allocate dequant buffer: pre-allocate in unlimited mode, defer in pooled mode.
    // Use the full shape if provided (e.g., [E, M, K] for 3D expert weights).
    if (!is_pooled()) {
        std::vector<long> dequant_shape;
        if (!shape.empty()) {
            dequant_shape = shape;
        } else {
            dequant_shape = {static_cast<long>(M), static_cast<long>(K)};
        }
        entry.dequant_buffer = mAllocator->allocate(
            ETensorDType::BF16,
            fmt::format("{}.dequant", name).c_str(),
            EAllocationType::ON_DEVICE,
            dequant_shape);
    }
    // In pooled mode, dequant_buffer starts as null and is acquired on first access

    // Insert into the map FIRST, then register with the offload manager.
    // register_tensor() stores a raw QuantizedTensor* — it must point to the
    // map-resident copy, not the local `entry` which is destroyed after move.
    auto [it, inserted] = mWeights.emplace(name, std::move(entry));

    if (offload_group >= 0 && mOffloadManager) {
        mOffloadManager->register_tensor(&it->second.quantized, offload_group, name);
    }
}

void GenericWeightManager::register_full_precision(
    const std::string& name,
    Tensor tensor) {
    if (mWeights.count(name)) {
        throw std::runtime_error(
            fmt::format("GenericWeightManager: weight '{}' already registered", name));
    }

    ManagedWeight entry;
    entry.is_quantized_weight = false;
    entry.full_precision = tensor;

    mWeights.emplace(name, std::move(entry));
}

void GenericWeightManager::quantize_and_store(
    const std::string& name,
    const Tensor& bf16,
    cudaStream_t stream) {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        throw std::runtime_error(
            fmt::format("GenericWeightManager: weight '{}' not registered", name));
    }

    auto& entry = it->second;
    if (!entry.is_quantized_weight) {
        throw std::runtime_error(
            fmt::format("GenericWeightManager: weight '{}' is full-precision, "
                       "cannot quantize", name));
    }

    // Quantize BF16 input into the pre-allocated QuantizedTensor
    mQuantizer->quantize(bf16, entry.quantized, stream);

    // Invalidate dequant cache for this weight
    entry.dequant_valid = false;
}

void GenericWeightManager::quantize_expert_slice(
    const std::string& name,
    int expert_idx,
    int per_expert_M,
    const Tensor& bf16,
    cudaStream_t stream) {

    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        throw std::runtime_error(
            fmt::format("GenericWeightManager: weight '{}' not registered", name));
    }

    auto& entry = it->second;
    if (!entry.is_quantized_weight) {
        throw std::runtime_error(
            fmt::format("GenericWeightManager: weight '{}' is full-precision, "
                       "cannot quantize", name));
    }

    const auto& full = entry.quantized;
    const int K = full.K;
    const int num_experts = full.M / per_expert_M;
    const long per_expert_elems = static_cast<long>(per_expert_M) * K;

    // Verify expert boundaries align with block boundaries
    if (per_expert_elems % full.block_size != 0) {
        throw std::runtime_error(
            fmt::format("GenericWeightManager: expert elements ({}) not divisible by "
                       "block_size ({}) for '{}'", per_expert_elems, full.block_size, name));
    }

    // Build a QuantizedTensor view pointing to this expert's slice
    QuantizedTensor view;
    view.M = per_expert_M;
    view.K = K;
    view.format = full.format;
    view.block_size = full.block_size;
    view.double_quant = full.double_quant;
    view.double_quant_group_size = full.double_quant_group_size;
    view.global_scale = full.global_scale;

    // Data slice (byte offset computed from total data size / num_experts)
    const size_t data_bytes_per_expert = full.data.bytes() / num_experts;
    const long data_elems_per_expert = full.data.nelem() / num_experts;
    view.data = Tensor::from_pointer(
        static_cast<std::byte*>(full.data.Data) + expert_idx * data_bytes_per_expert,
        full.data.Device,
        full.data.DType,
        std::vector<long>{data_elems_per_expert});

    // Scales slice
    const long scales_per_expert = full.scales.nelem() / num_experts;
    const size_t scales_bytes_per_expert = full.scales.bytes() / num_experts;
    view.scales = Tensor::from_pointer(
        static_cast<std::byte*>(full.scales.Data) + expert_idx * scales_bytes_per_expert,
        full.scales.Device,
        full.scales.DType,
        std::vector<long>{scales_per_expert});

    // Meta slice (if present, for double quantization)
    if (!full.meta.is_null()) {
        const long meta_per_expert = full.meta.nelem() / num_experts;
        const size_t meta_bytes_per_expert = full.meta.bytes() / num_experts;
        view.meta = Tensor::from_pointer(
            static_cast<std::byte*>(full.meta.Data) + expert_idx * meta_bytes_per_expert,
            full.meta.Device,
            full.meta.DType,
            std::vector<long>{meta_per_expert});
    }

    // Meta2 slice (if present, for double quantization)
    if (!full.meta2.is_null()) {
        const long meta2_per_expert = full.meta2.nelem() / num_experts;
        const size_t meta2_bytes_per_expert = full.meta2.bytes() / num_experts;
        view.meta2 = Tensor::from_pointer(
            static_cast<std::byte*>(full.meta2.Data) + expert_idx * meta2_bytes_per_expert,
            full.meta2.Device,
            full.meta2.DType,
            std::vector<long>{meta2_per_expert});
    }

    // Quantize the single expert into the sub-view
    mQuantizer->quantize(bf16, view, stream);
}

// =============================================================================
// Weight access
// =============================================================================

Tensor& GenericWeightManager::get(const std::string& name, cudaStream_t stream) {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        throw std::runtime_error(
            fmt::format("GenericWeightManager: weight '{}' not found", name));
    }

    auto& entry = it->second;

    // Full-precision weights: return directly
    if (!entry.is_quantized_weight) {
        return entry.full_precision;
    }

    // Quantized weights: ensure dequantized
    ensure_dequantized(entry, name, stream);
    return entry.dequant_buffer;
}

const QuantizedTensor* GenericWeightManager::get_quantized(const std::string& name) const {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        return nullptr;
    }
    if (!it->second.is_quantized_weight) {
        return nullptr;
    }
    return &it->second.quantized;
}

bool GenericWeightManager::has_weight(const std::string& name) const {
    return mWeights.count(name) > 0;
}

std::vector<std::string> GenericWeightManager::weight_names() const {
    std::vector<std::string> names;
    names.reserve(mWeights.size());
    for (const auto& [name, _] : mWeights) {
        names.push_back(name);
    }
    return names;
}

int GenericWeightManager::get_offload_group(const std::string& name) const {
    auto it = mWeights.find(name);
    if (it == mWeights.end()) {
        return -1;
    }
    return it->second.offload_group;
}

// =============================================================================
// Offloading
// =============================================================================

void GenericWeightManager::prefetch_group(int group_id, cudaStream_t stream) {
    if (mOffloadManager) {
        mOffloadManager->prefetch_group(group_id, stream);
    }
}

// =============================================================================
// Step management
// =============================================================================

void GenericWeightManager::new_step() {
    mCurrentStep++;

    // Invalidate all dequant caches
    for (auto& [name, entry] : mWeights) {
        entry.dequant_valid = false;
    }

    // Advance offload manager step
    if (mOffloadManager) {
        mOffloadManager->new_step();
    }
}

// =============================================================================
// Statistics
// =============================================================================

size_t GenericWeightManager::quantized_bytes() const {
    size_t total = 0;
    for (const auto& [name, entry] : mWeights) {
        if (entry.is_quantized_weight) {
            total += entry.quantized.packed_bytes();
            if (!entry.quantized.scales.is_null()) {
                total += entry.quantized.scales.bytes();
            }
            if (!entry.quantized.meta.is_null()) {
                total += entry.quantized.meta.bytes();
            }
            if (!entry.quantized.meta2.is_null()) {
                total += entry.quantized.meta2.bytes();
            }
        }
    }
    return total;
}

size_t GenericWeightManager::dequant_buffer_bytes() const {
    size_t total = 0;
    for (const auto& [name, entry] : mWeights) {
        if (entry.is_quantized_weight && !entry.dequant_buffer.is_null()) {
            total += entry.dequant_buffer.bytes();
        }
    }
    return total;
}

size_t GenericWeightManager::full_precision_bytes() const {
    size_t total = 0;
    for (const auto& [name, entry] : mWeights) {
        if (!entry.is_quantized_weight && !entry.full_precision.is_null()) {
            total += entry.full_precision.bytes();
        }
    }
    return total;
}

int GenericWeightManager::num_weights() const {
    return static_cast<int>(mWeights.size());
}

int GenericWeightManager::num_quantized() const {
    int count = 0;
    for (const auto& [name, entry] : mWeights) {
        if (entry.is_quantized_weight) count++;
    }
    return count;
}

int GenericWeightManager::num_full_precision() const {
    int count = 0;
    for (const auto& [name, entry] : mWeights) {
        if (!entry.is_quantized_weight) count++;
    }
    return count;
}

int GenericWeightManager::num_active_dequant_buffers() const {
    if (!is_pooled()) {
        return num_quantized();  // All pre-allocated in unlimited mode
    }
    return mActivePoolBuffers;
}

// =============================================================================
// Internal helpers
// =============================================================================

void GenericWeightManager::ensure_dequantized(ManagedWeight& entry,
                                               const std::string& name,
                                               cudaStream_t stream) {
    // In pooled mode, update LRU even if already valid (for eviction tracking)
    if (is_pooled() && entry.has_pool_buffer) {
        touch_lru(entry, name);
    }

    // Check cache validity
    if (entry.dequant_valid && entry.dequant_step == mCurrentStep) {
        return;  // Already dequantized this step
    }

    // In pooled mode, acquire a buffer from the pool if we don't have one
    if (is_pooled() && !entry.has_pool_buffer) {
        acquire_pool_buffer(entry, name);
    }

    // If offloaded, ensure the quantized data is resident on GPU
    if (entry.offload_group >= 0 && mOffloadManager) {
        mOffloadManager->load_group(entry.offload_group, stream);
    }

    // Dequantize into the buffer
    mQuantizer->dequantize(entry.quantized, entry.dequant_buffer, stream);

    entry.dequant_valid = true;
    entry.dequant_step = mCurrentStep;
}

void GenericWeightManager::acquire_pool_buffer(ManagedWeight& entry,
                                                const std::string& name) {
    // Evict LRU buffers if we're at the limit
    while (mActivePoolBuffers >= mConfig.max_dequant_cache_size) {
        evict_lru_buffer();
    }

    // Acquire a buffer from the pool (with full shape for 3D expert weights)
    entry.dequant_buffer = mBufferPool->acquire(entry.M, entry.K, entry.dequant_shape);
    entry.has_pool_buffer = true;
    mActivePoolBuffers++;

    // Add to front of LRU list
    mDequantLRU.push_front(name);
    entry.lru_it = mDequantLRU.begin();
}

void GenericWeightManager::evict_lru_buffer() {
    if (mDequantLRU.empty()) {
        return;
    }

    // Evict the least recently used (back of list)
    const std::string& victim_name = mDequantLRU.back();
    auto it = mWeights.find(victim_name);
    if (it != mWeights.end()) {
        auto& victim = it->second;
        if (victim.has_pool_buffer && !victim.dequant_buffer.is_null()) {
            mBufferPool->release(victim.M, victim.K, std::move(victim.dequant_buffer));
            victim.dequant_buffer = Tensor{};
            victim.has_pool_buffer = false;
            victim.dequant_valid = false;
            mActivePoolBuffers--;
        }
    }

    mDequantLRU.pop_back();
}

void GenericWeightManager::touch_lru(ManagedWeight& entry, const std::string& name) {
    // Move to front of LRU list (most recently used)
    mDequantLRU.erase(entry.lru_it);
    mDequantLRU.push_front(name);
    entry.lru_it = mDequantLRU.begin();
}

}  // namespace qlora
