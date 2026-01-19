// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_OPTIMIZER_STATE_H
#define SUROGATE_SRC_MODULES_OPTIMIZER_STATE_H

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "module_concept.h"
#include "utilities/allocator.h"
#include "utilities/lazy_allocator.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"
#include "modules/mlp_utils.h"

class NCCLCommunicator;

namespace modules {

/**
 * @brief Modular optimizer state manager template
 *
 * Manages Adam optimizer state (first and second moments) for a model
 * composed of modular blocks. Supports:
 * - CPU offloading for large models
 * - Double-buffered prefetching
 * - Different dtypes for moments (FP32, BF16, FP8)
 *
 * @tparam Block The transformer block module type
 */
template<typename Block>
class ModularOptimizerStateManager {
public:
    using BlockWeights = typename Block::Weights;

    /**
     * @brief Optimizer state for a single parameter block
     *
     * Mirrors the weight structure but contains optimizer moments.
     */
    struct BlockOptState {
        // First moment (m) - same structure as BlockWeights
        BlockWeights m;

        // Second moment (v) - same structure as BlockWeights
        BlockWeights v;

        // Scales for FP8 moments (if using low-precision optimizer state)
        BlockWeights m_scales;
        BlockWeights v_scales;
    };

    /**
     * @brief Configuration for optimizer state manager
     */
    struct Config {
        int num_layers;
        typename Block::Config block_config;

        // Data types
        ETensorDType m_dtype;           ///< First moment dtype (typically FP32 or BF16)
        ETensorDType v_dtype;           ///< Second moment dtype (typically FP32)

        // Sharding
        int shard_idx;
        int num_shards;

        // Offloading
        bool offload_m = false;         ///< Offload first moments to CPU
        bool offload_v = false;         ///< Offload second moments to CPU
        bool use_zero_copy = false;
        EAllocationType offload_alloc = EAllocationType::PINNED;

        // Non-block config
        int vocab_size;
        int hidden_size;
        bool tied_embeddings;
    };

    /**
     * @brief Non-block optimizer state
     */
    struct NonBlockOptState {
        // First moments
        TensorShard m_embeddings;
        TensorShard m_lm_head;
        TensorShard m_final_norm;

        // Second moments
        TensorShard v_embeddings;
        TensorShard v_lm_head;
        TensorShard v_final_norm;

        // Optional FP8 scale tensors (local, not sharded)
        Tensor m_embeddings_scales;
        Tensor m_lm_head_scales;
        Tensor m_final_norm_scales;
        Tensor v_embeddings_scales;
        Tensor v_lm_head_scales;
        Tensor v_final_norm_scales;
    };

    /**
     * @brief Tensor containers for checkpointing (names match model weights tensor names).
     *
     * Note: checkpoints are written per-rank; the `write_safetensors()` helper will downcast
     * these TensorShards to local Tensor views, so no communicator is required.
     */
    ITensorContainer& full_m();
    ITensorContainer& full_v();
    ITensorContainer& full_m_scales();
    ITensorContainer& full_v_scales();

    ModularOptimizerStateManager(const Config& config, cudaStream_t stream,
                                  NCCLCommunicator& comm, TensorAllocator& allocator);
    ~ModularOptimizerStateManager();

    // ========================================================================
    // Optimizer pass management
    // ========================================================================

    /**
     * @brief Prepare for optimizer pass
     *
     * Allocates temporary device buffers if moments are offloaded.
     */
    void begin_optimizer(DeviceMemoryStack& memory, cudaStream_t main_stream);

    /**
     * @brief Cleanup after optimizer pass
     */
    void end_optimizer(DeviceMemoryStack& memory);

    // ========================================================================
    // Block state access (with prefetching)
    // ========================================================================

    /**
     * @brief Initiate prefetch of optimizer state for a block
     */
    void fetch_block(int layer_idx, cudaStream_t fetch_stream);

    /**
     * @brief Get first moment (m) for a block
     */
    BlockWeights& get_block_m(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get second moment (v) for a block
     */
    BlockWeights& get_block_v(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get FP8 scales for first moment (m) for a block
     */
    BlockWeights& get_block_m_scales(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get FP8 scales for second moment (v) for a block
     */
    BlockWeights& get_block_v_scales(int layer_idx, cudaStream_t stream);

    /**
     * @brief Store updated optimizer state back (if offloaded)
     */
    void store_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream);

    // ========================================================================
    // Non-block state access (not double-buffered)
    // ========================================================================

    NonBlockOptState& non_block_state() { return mNonBlockState; }

    // Direct access to full state (for checkpointing)
    std::vector<BlockOptState>& full_state() { return mBlockState; }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const Config& config() const { return mConfig; }

private:
    Config mConfig;
    TensorAllocator* mAllocator;

    // Full optimizer state (host or device depending on offload settings)
    std::vector<BlockOptState> mBlockState;
    NonBlockOptState mNonBlockState;

    // Per-block storage tensors (for bulk transfers)
    std::vector<Tensor> mBlockMStorage;
    std::vector<Tensor> mBlockVStorage;
    std::vector<Tensor> mBlockMScalesStorage;
    std::vector<Tensor> mBlockVScalesStorage;

    // Double-buffered device cache for prefetching (allocated from DeviceMemoryStack per optimizer step)
    struct BufferState {
        BlockWeights m;
        BlockWeights v;
        BlockWeights m_scales;
        BlockWeights v_scales;
        Tensor m_storage;
        Tensor v_storage;
        Tensor m_scales_storage;
        Tensor v_scales_storage;
    };
    std::array<BufferState, 2> mBuffer;

    // Buffer status
    struct BufferStatus {
        int layer_idx = -1;
        cudaEvent_t done_event = nullptr;
        bool fetch_pending = false;
        bool done = true;
    };
    std::array<BufferStatus, 2> mStatus;

    class StateContainer final : public ITensorContainer {
    public:
        enum class Kind { M, V };
        StateContainer(ModularOptimizerStateManager* owner, Kind kind) : mOwner(owner), mKind(kind) {}
        void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;
    private:
        ModularOptimizerStateManager* mOwner = nullptr;
        Kind mKind;
    };

    StateContainer mMomentumContainer{this, StateContainer::Kind::M};
    StateContainer mVarianceContainer{this, StateContainer::Kind::V};

    class ScaleContainer final : public ITensorContainer {
    public:
        enum class Kind { M, V };
        ScaleContainer(ModularOptimizerStateManager* owner, Kind kind) : mOwner(owner), mKind(kind) {}
        void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;
    private:
        ModularOptimizerStateManager* mOwner = nullptr;
        Kind mKind;
    };

    ScaleContainer mMomentumScaleContainer{this, ScaleContainer::Kind::M};
    ScaleContainer mVarianceScaleContainer{this, ScaleContainer::Kind::V};

    // Internal helpers
    void register_block_weights_lazy(LazyAllocator& alloc, BlockWeights& dst, ETensorDType dtype);
    void register_block_scales_lazy(LazyAllocator& alloc, BlockWeights& dst);
    Tensor allocate_block_state(BlockWeights& dst, ETensorDType dtype, EAllocationType kind, const char* name);
    Tensor allocate_block_scales(BlockWeights& dst, EAllocationType kind, const char* name);
    void allocate_non_block_opt_state();
    static void zero_storage(Tensor& storage, cudaStream_t stream);
    static void wait_on_fetch(BufferStatus& status, cudaStream_t stream);
    BlockWeights& get_block_from(int layer_idx, cudaStream_t stream, BlockWeights& buf);
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Block>
ModularOptimizerStateManager<Block>::ModularOptimizerStateManager(
    const Config& config, cudaStream_t stream, NCCLCommunicator& comm, TensorAllocator& allocator)
    : mConfig(config), mAllocator(&allocator) {
    (void)comm;

    // Allocate optimizer state for all blocks
    mBlockState.resize(config.num_layers);
    mBlockMStorage.resize(config.num_layers);
    mBlockVStorage.resize(config.num_layers);
    if (is_fp8_dtype(config.m_dtype)) {
        mBlockMScalesStorage.resize(config.num_layers);
    }
    if (is_fp8_dtype(config.v_dtype)) {
        mBlockVScalesStorage.resize(config.num_layers);
    }

    for (int i = 0; i < config.num_layers; ++i) {
        {
            EAllocationType kind = config.offload_m ? config.offload_alloc : EAllocationType::ON_DEVICE;
            mBlockMStorage[i] = allocate_block_state(mBlockState[i].m, config.m_dtype, kind, "opt_m_block");
            zero_storage(mBlockMStorage[i], stream);
        }
        {
            EAllocationType kind = config.offload_v ? config.offload_alloc : EAllocationType::ON_DEVICE;
            mBlockVStorage[i] = allocate_block_state(mBlockState[i].v, config.v_dtype, kind, "opt_v_block");
            zero_storage(mBlockVStorage[i], stream);
        }
        if (is_fp8_dtype(config.m_dtype)) {
            EAllocationType kind = config.offload_m ? config.offload_alloc : EAllocationType::ON_DEVICE;
            mBlockMScalesStorage[i] = allocate_block_scales(mBlockState[i].m_scales, kind, "opt_m_scales_block");
            zero_storage(mBlockMScalesStorage[i], stream);
        }
        if (is_fp8_dtype(config.v_dtype)) {
            EAllocationType kind = config.offload_v ? config.offload_alloc : EAllocationType::ON_DEVICE;
            mBlockVScalesStorage[i] = allocate_block_scales(mBlockState[i].v_scales, kind, "opt_v_scales_block");
            zero_storage(mBlockVScalesStorage[i], stream);
        }
    }

    // Allocate non-block state
    allocate_non_block_opt_state();
    if (mNonBlockState.m_embeddings.Data) zero_storage(static_cast<Tensor&>(mNonBlockState.m_embeddings), stream);
    if (mNonBlockState.v_embeddings.Data) zero_storage(static_cast<Tensor&>(mNonBlockState.v_embeddings), stream);
    if (mNonBlockState.m_final_norm.Data) zero_storage(static_cast<Tensor&>(mNonBlockState.m_final_norm), stream);
    if (mNonBlockState.v_final_norm.Data) zero_storage(static_cast<Tensor&>(mNonBlockState.v_final_norm), stream);
    if (!mConfig.tied_embeddings) {
        if (mNonBlockState.m_lm_head.Data) zero_storage(static_cast<Tensor&>(mNonBlockState.m_lm_head), stream);
        if (mNonBlockState.v_lm_head.Data) zero_storage(static_cast<Tensor&>(mNonBlockState.v_lm_head), stream);
    }

    if ((config.offload_m || config.offload_v) && !config.use_zero_copy) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaEventCreate(&mStatus[i].done_event));
        }
    }
}

template<typename Block>
ModularOptimizerStateManager<Block>::~ModularOptimizerStateManager() {
    for (int i = 0; i < 2; ++i) {
        if (mStatus[i].done_event) {
            cudaEventDestroy(mStatus[i].done_event);
        }
    }
}

template<typename Block>
void ModularOptimizerStateManager<Block>::begin_optimizer(DeviceMemoryStack& memory, cudaStream_t main_stream) {
    if ((!mConfig.offload_m && !mConfig.offload_v) || mConfig.use_zero_copy) {
        return;
    }

    CUDA_CHECK(cudaEventRecord(mStatus.at(0).done_event, main_stream));
    CUDA_CHECK(cudaEventRecord(mStatus.at(1).done_event, main_stream));

    LazyAllocator alloc;
    if (mConfig.offload_m) {
        register_block_weights_lazy(alloc, mBuffer[0].m, mConfig.m_dtype);
        mBuffer[0].m_storage = alloc.commit(memory, "opt_m_a");
        register_block_weights_lazy(alloc, mBuffer[1].m, mConfig.m_dtype);
        mBuffer[1].m_storage = alloc.commit(memory, "opt_m_b");
        if (is_fp8_dtype(mConfig.m_dtype)) {
            register_block_scales_lazy(alloc, mBuffer[0].m_scales);
            mBuffer[0].m_scales_storage = alloc.commit(memory, "opt_m_scales_a");
            register_block_scales_lazy(alloc, mBuffer[1].m_scales);
            mBuffer[1].m_scales_storage = alloc.commit(memory, "opt_m_scales_b");
        }
    }
    if (mConfig.offload_v) {
        register_block_weights_lazy(alloc, mBuffer[0].v, mConfig.v_dtype);
        mBuffer[0].v_storage = alloc.commit(memory, "opt_v_a");
        register_block_weights_lazy(alloc, mBuffer[1].v, mConfig.v_dtype);
        mBuffer[1].v_storage = alloc.commit(memory, "opt_v_b");
        if (is_fp8_dtype(mConfig.v_dtype)) {
            register_block_scales_lazy(alloc, mBuffer[0].v_scales);
            mBuffer[0].v_scales_storage = alloc.commit(memory, "opt_v_scales_a");
            register_block_scales_lazy(alloc, mBuffer[1].v_scales);
            mBuffer[1].v_scales_storage = alloc.commit(memory, "opt_v_scales_b");
        }
    }

    for (auto& stat : mStatus) {
        stat.layer_idx = -1;
        stat.fetch_pending = false;
        stat.done = true;
    }
}

template<typename Block>
void ModularOptimizerStateManager<Block>::end_optimizer(DeviceMemoryStack& memory) {
    if ((!mConfig.offload_m && !mConfig.offload_v) || mConfig.use_zero_copy) {
        return;
    }

    if (mConfig.offload_v) {
        if (is_fp8_dtype(mConfig.v_dtype)) {
            memory.free(mBuffer[1].v_scales_storage);
            memory.free(mBuffer[0].v_scales_storage);
        }
        memory.free(mBuffer[1].v_storage);
        memory.free(mBuffer[0].v_storage);
    }
    if (mConfig.offload_m) {
        if (is_fp8_dtype(mConfig.m_dtype)) {
            memory.free(mBuffer[1].m_scales_storage);
            memory.free(mBuffer[0].m_scales_storage);
        }
        memory.free(mBuffer[1].m_storage);
        memory.free(mBuffer[0].m_storage);
    }
}

template<typename Block>
void ModularOptimizerStateManager<Block>::fetch_block(int layer_idx, cudaStream_t fetch_stream) {
    if ((!mConfig.offload_m && !mConfig.offload_v) || mConfig.use_zero_copy) {
        return;
    }

    int buffer = layer_idx % 2;
    auto& status = mStatus.at(buffer);
    status.layer_idx = layer_idx;
    status.fetch_pending = true;
    status.done = false;

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.done_event, 0));

    if (mConfig.offload_m) {
        auto& dst = mBuffer.at(buffer).m_storage;
        auto& src = mBlockMStorage.at(layer_idx);
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, src.bytes(), cudaMemcpyHostToDevice, fetch_stream));
        if (is_fp8_dtype(mConfig.m_dtype)) {
            auto& dst_s = mBuffer.at(buffer).m_scales_storage;
            auto& src_s = mBlockMScalesStorage.at(layer_idx);
            CUDA_CHECK(cudaMemcpyAsync(dst_s.Data, src_s.Data, src_s.bytes(), cudaMemcpyHostToDevice, fetch_stream));
        }
    }
    if (mConfig.offload_v) {
        auto& dst = mBuffer.at(buffer).v_storage;
        auto& src = mBlockVStorage.at(layer_idx);
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, src.bytes(), cudaMemcpyHostToDevice, fetch_stream));
        if (is_fp8_dtype(mConfig.v_dtype)) {
            auto& dst_s = mBuffer.at(buffer).v_scales_storage;
            auto& src_s = mBlockVScalesStorage.at(layer_idx);
            CUDA_CHECK(cudaMemcpyAsync(dst_s.Data, src_s.Data, src_s.bytes(), cudaMemcpyHostToDevice, fetch_stream));
        }
    }

    CUDA_CHECK(cudaEventRecord(status.done_event, fetch_stream));
}

template<typename Block>
typename Block::Weights& ModularOptimizerStateManager<Block>::get_block_m(int layer_idx, cudaStream_t stream) {
    if (!mConfig.offload_m || mConfig.use_zero_copy) {
        return mBlockState[layer_idx].m;
    }
    return get_block_from(layer_idx, stream, mBuffer.at(layer_idx % 2).m);
}

template<typename Block>
typename Block::Weights& ModularOptimizerStateManager<Block>::get_block_v(int layer_idx, cudaStream_t stream) {
    if (!mConfig.offload_v || mConfig.use_zero_copy) {
        return mBlockState[layer_idx].v;
    }
    return get_block_from(layer_idx, stream, mBuffer.at(layer_idx % 2).v);
}

template<typename Block>
typename Block::Weights& ModularOptimizerStateManager<Block>::get_block_m_scales(int layer_idx, cudaStream_t stream) {
    if (!is_fp8_dtype(mConfig.m_dtype) || (!mConfig.offload_m || mConfig.use_zero_copy)) {
        return mBlockState[layer_idx].m_scales;
    }
    return get_block_from(layer_idx, stream, mBuffer.at(layer_idx % 2).m_scales);
}

template<typename Block>
typename Block::Weights& ModularOptimizerStateManager<Block>::get_block_v_scales(int layer_idx, cudaStream_t stream) {
    if (!is_fp8_dtype(mConfig.v_dtype) || (!mConfig.offload_v || mConfig.use_zero_copy)) {
        return mBlockState[layer_idx].v_scales;
    }
    return get_block_from(layer_idx, stream, mBuffer.at(layer_idx % 2).v_scales);
}

template<typename Block>
void ModularOptimizerStateManager<Block>::store_block(int layer_idx, cudaStream_t stream, cudaStream_t put_stream) {
    if ((!mConfig.offload_m && !mConfig.offload_v) || mConfig.use_zero_copy) {
        return;
    }

    int buffer = layer_idx % 2;
    auto& status = mStatus.at(buffer);
    if (status.layer_idx != layer_idx) {
        throw std::logic_error("Layer index mismatch in ModularOptimizerStateManager::store_block");
    }

    CUDA_CHECK(cudaEventRecord(status.done_event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(put_stream, status.done_event, 0));

    if (mConfig.offload_m) {
        auto& dst = mBlockMStorage.at(layer_idx);
        auto& src = mBuffer.at(buffer).m_storage;
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDeviceToHost, put_stream));
        if (is_fp8_dtype(mConfig.m_dtype)) {
            auto& dst_s = mBlockMScalesStorage.at(layer_idx);
            auto& src_s = mBuffer.at(buffer).m_scales_storage;
            CUDA_CHECK(cudaMemcpyAsync(dst_s.Data, src_s.Data, dst_s.bytes(), cudaMemcpyDeviceToHost, put_stream));
        }
    }
    if (mConfig.offload_v) {
        auto& dst = mBlockVStorage.at(layer_idx);
        auto& src = mBuffer.at(buffer).v_storage;
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDeviceToHost, put_stream));
        if (is_fp8_dtype(mConfig.v_dtype)) {
            auto& dst_s = mBlockVScalesStorage.at(layer_idx);
            auto& src_s = mBuffer.at(buffer).v_scales_storage;
            CUDA_CHECK(cudaMemcpyAsync(dst_s.Data, src_s.Data, dst_s.bytes(), cudaMemcpyDeviceToHost, put_stream));
        }
    }

    CUDA_CHECK(cudaEventRecord(status.done_event, put_stream));
    status.done = true;
}

template<typename Block>
void ModularOptimizerStateManager<Block>::zero_storage(Tensor& storage, cudaStream_t stream) {
    if (storage.Data == nullptr) return;
    if (storage.Device < 0) {
        std::memset(storage.Data, 0, storage.bytes());
    } else {
        fill_zero(storage, stream);
    }
}

template<typename Block>
void ModularOptimizerStateManager<Block>::wait_on_fetch(BufferStatus& status, cudaStream_t stream) {
    if (!status.fetch_pending) return;
    CUDA_CHECK(cudaStreamWaitEvent(stream, status.done_event, 0));
    status.fetch_pending = false;
}

template<typename Block>
void ModularOptimizerStateManager<Block>::register_block_weights_lazy(
    LazyAllocator& alloc, BlockWeights& dst, ETensorDType dtype) {
    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long M = mlp_up_rows(cfg);
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;
    long qkv_channels = HS * (HQ + 2 * HKV);
    long q_rows = HS * HQ;

    auto local_shape = [&](std::initializer_list<long> shape) -> std::vector<long> {
        std::vector<long> result(shape);
        if (!result.empty() && mConfig.num_shards > 1) {
            result[0] = div_exact(result[0], static_cast<long>(mConfig.num_shards));
        }
        return result;
    };

    if constexpr (requires { dst.ln1.weight; }) {
        alloc.allocate(&dst.ln1.weight, dtype, local_shape({C}));
    }
    if constexpr (requires { dst.ln2.weight; }) {
        alloc.allocate(&dst.ln2.weight, dtype, local_shape({C}));
    }
    if constexpr (requires { dst.attention.qkv_weight; }) {
        alloc.allocate(&dst.attention.qkv_weight, dtype, local_shape({qkv_channels, C}));
    }
    if constexpr (requires { dst.attention.qkv_bias; }) {
        if (cfg.use_qkv_bias) {
            if (!dst.attention.qkv_bias.has_value()) dst.attention.qkv_bias.emplace();
            alloc.allocate(&dst.attention.qkv_bias.value(), dtype, local_shape({qkv_channels}));
        } else {
            dst.attention.qkv_bias.reset();
        }
    }
    if constexpr (requires { dst.attention.out_weight; }) {
        alloc.allocate(&dst.attention.out_weight, dtype, local_shape({C, q_rows}));
    }
    if constexpr (requires { dst.attention.q_norm_weight; dst.attention.k_norm_weight; }) {
        if constexpr (requires { cfg.use_qk_norm; }) {
            if (cfg.use_qk_norm) {
                if (!dst.attention.q_norm_weight.has_value()) dst.attention.q_norm_weight.emplace();
                if (!dst.attention.k_norm_weight.has_value()) dst.attention.k_norm_weight.emplace();
                alloc.allocate(&dst.attention.q_norm_weight.value(), dtype, local_shape({HS}));
                alloc.allocate(&dst.attention.k_norm_weight.value(), dtype, local_shape({HS}));
            } else {
                dst.attention.q_norm_weight.reset();
                dst.attention.k_norm_weight.reset();
            }
        }
    }
    if constexpr (requires { dst.mlp_up_weight; dst.mlp_down_weight; }) {
        alloc.allocate(&dst.mlp_up_weight, dtype, local_shape({M, C}));
        alloc.allocate(&dst.mlp_down_weight, dtype, local_shape({C, D}));
    }
}

template<typename Block>
void ModularOptimizerStateManager<Block>::register_block_scales_lazy(
    LazyAllocator& alloc, BlockWeights& dst) {
    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long M = mlp_up_rows(cfg);
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;
    long qkv_channels = HS * (HQ + 2 * HKV);
    long q_rows = HS * HQ;

    auto local_shape = [&](std::initializer_list<long> shape) -> std::vector<long> {
        std::vector<long> result(shape);
        if (!result.empty() && mConfig.num_shards > 1) {
            result[0] = div_exact(result[0], static_cast<long>(mConfig.num_shards));
        }
        return result;
    };
    auto scale_shape = [&](const std::vector<long>& param_shape) -> std::vector<long> {
        std::size_t n = 1;
        for (long d : param_shape) n *= static_cast<std::size_t>(d);
        long blocks = div_ceil(static_cast<long>(n), 128L);
        return {blocks};
    };

    if constexpr (requires { dst.ln1.weight; }) {
        alloc.allocate(&dst.ln1.weight, ETensorDType::FP32, scale_shape(local_shape({C})));
    }
    if constexpr (requires { dst.ln2.weight; }) {
        alloc.allocate(&dst.ln2.weight, ETensorDType::FP32, scale_shape(local_shape({C})));
    }
    if constexpr (requires { dst.attention.qkv_weight; }) {
        alloc.allocate(&dst.attention.qkv_weight, ETensorDType::FP32, scale_shape(local_shape({qkv_channels, C})));
    }
    if constexpr (requires { dst.attention.qkv_bias; }) {
        if (cfg.use_qkv_bias) {
            if (!dst.attention.qkv_bias.has_value()) dst.attention.qkv_bias.emplace();
            alloc.allocate(&dst.attention.qkv_bias.value(), ETensorDType::FP32, scale_shape(local_shape({qkv_channels})));
        } else {
            dst.attention.qkv_bias.reset();
        }
    }
    if constexpr (requires { dst.attention.out_weight; }) {
        alloc.allocate(&dst.attention.out_weight, ETensorDType::FP32, scale_shape(local_shape({C, q_rows})));
    }
    if constexpr (requires { dst.attention.q_norm_weight; dst.attention.k_norm_weight; }) {
        if constexpr (requires { cfg.use_qk_norm; }) {
            if (cfg.use_qk_norm) {
                if (!dst.attention.q_norm_weight.has_value()) dst.attention.q_norm_weight.emplace();
                if (!dst.attention.k_norm_weight.has_value()) dst.attention.k_norm_weight.emplace();
                alloc.allocate(&dst.attention.q_norm_weight.value(), ETensorDType::FP32, scale_shape(local_shape({HS})));
                alloc.allocate(&dst.attention.k_norm_weight.value(), ETensorDType::FP32, scale_shape(local_shape({HS})));
            } else {
                dst.attention.q_norm_weight.reset();
                dst.attention.k_norm_weight.reset();
            }
        }
    }
    if constexpr (requires { dst.mlp_up_weight; dst.mlp_down_weight; }) {
        alloc.allocate(&dst.mlp_up_weight, ETensorDType::FP32, scale_shape(local_shape({M, C})));
        alloc.allocate(&dst.mlp_down_weight, ETensorDType::FP32, scale_shape(local_shape({C, D})));
    }
}

template<typename Block>
Tensor ModularOptimizerStateManager<Block>::allocate_block_state(
    BlockWeights& dst, ETensorDType dtype, EAllocationType kind, const char* name) {
    LazyAllocator alloc;
    register_block_weights_lazy(alloc, dst, dtype);
    return alloc.commit(*mAllocator, kind, name);
}

template<typename Block>
Tensor ModularOptimizerStateManager<Block>::allocate_block_scales(
    BlockWeights& dst, EAllocationType kind, const char* name) {
    LazyAllocator alloc;
    register_block_scales_lazy(alloc, dst);
    return alloc.commit(*mAllocator, kind, name);
}

template<typename Block>
typename Block::Weights& ModularOptimizerStateManager<Block>::get_block_from(int layer_idx, cudaStream_t stream, BlockWeights& buf) {
    int buffer = layer_idx % 2;
    auto& status = mStatus.at(buffer);
    if (status.layer_idx != layer_idx) {
        throw std::logic_error("Layer index mismatch in ModularOptimizerStateManager::get_block");
    }
    wait_on_fetch(status, stream);
    return buf;
}

template<typename Block>
void ModularOptimizerStateManager<Block>::allocate_non_block_opt_state() {
    const auto m_kind = mConfig.offload_m ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;
    const auto v_kind = mConfig.offload_v ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;
    long V = mConfig.vocab_size;
    long C = mConfig.hidden_size;

    // First moments
    mNonBlockState.m_embeddings = mAllocator->allocate_shard(mConfig.m_dtype, mConfig.shard_idx, mConfig.num_shards,
                                                             "opt_m_embeddings", {V, C}, m_kind);
    mNonBlockState.m_final_norm = mAllocator->allocate_shard(mConfig.m_dtype, mConfig.shard_idx, mConfig.num_shards,
                                                             "opt_m_final_norm", {C}, m_kind);

    if (!mConfig.tied_embeddings) {
        mNonBlockState.m_lm_head = mAllocator->allocate_shard(mConfig.m_dtype, mConfig.shard_idx, mConfig.num_shards,
                                                              "opt_m_lm_head", {V, C}, m_kind);
    } else {
        mNonBlockState.m_lm_head = mNonBlockState.m_embeddings;
    }

    // Second moments
    mNonBlockState.v_embeddings = mAllocator->allocate_shard(mConfig.v_dtype, mConfig.shard_idx, mConfig.num_shards,
                                                             "opt_v_embeddings", {V, C}, v_kind);
    mNonBlockState.v_final_norm = mAllocator->allocate_shard(mConfig.v_dtype, mConfig.shard_idx, mConfig.num_shards,
                                                             "opt_v_final_norm", {C}, v_kind);

    if (!mConfig.tied_embeddings) {
        mNonBlockState.v_lm_head = mAllocator->allocate_shard(mConfig.v_dtype, mConfig.shard_idx, mConfig.num_shards,
                                                              "opt_v_lm_head", {V, C}, v_kind);
    } else {
        mNonBlockState.v_lm_head = mNonBlockState.v_embeddings;
    }

    auto local_shape = [&](std::initializer_list<long> shape) -> std::vector<long> {
        std::vector<long> result(shape);
        if (!result.empty() && mConfig.num_shards > 1) {
            result[0] = div_exact(result[0], static_cast<long>(mConfig.num_shards));
        }
        return result;
    };
    auto alloc_scales = [&](Tensor& dst, const char* name, const std::vector<long>& param_shape, EAllocationType kind) {
        std::size_t n = 1;
        for (long d : param_shape) n *= static_cast<std::size_t>(d);
        long blocks = div_ceil(static_cast<long>(n), 128L);
        dst = mAllocator->allocate(ETensorDType::FP32, name, kind, {blocks});
        zero_storage(dst, /*stream=*/nullptr);
    };

    // FP8 scale tensors (local, per-rank).
    if (is_fp8_dtype(mConfig.m_dtype)) {
        alloc_scales(mNonBlockState.m_embeddings_scales, "opt_m_scales_embeddings", local_shape({V, C}), m_kind);
        alloc_scales(mNonBlockState.m_final_norm_scales, "opt_m_scales_final_norm", local_shape({C}), m_kind);
        if (!mConfig.tied_embeddings) {
            alloc_scales(mNonBlockState.m_lm_head_scales, "opt_m_scales_lm_head", local_shape({V, C}), m_kind);
        } else {
            mNonBlockState.m_lm_head_scales = mNonBlockState.m_embeddings_scales;
        }
    }
    if (is_fp8_dtype(mConfig.v_dtype)) {
        alloc_scales(mNonBlockState.v_embeddings_scales, "opt_v_scales_embeddings", local_shape({V, C}), v_kind);
        alloc_scales(mNonBlockState.v_final_norm_scales, "opt_v_scales_final_norm", local_shape({C}), v_kind);
        if (!mConfig.tied_embeddings) {
            alloc_scales(mNonBlockState.v_lm_head_scales, "opt_v_scales_lm_head", local_shape({V, C}), v_kind);
        } else {
            mNonBlockState.v_lm_head_scales = mNonBlockState.v_embeddings_scales;
        }
    }
}

template<typename Block>
ITensorContainer& ModularOptimizerStateManager<Block>::full_m() {
    return mMomentumContainer;
}

template<typename Block>
ITensorContainer& ModularOptimizerStateManager<Block>::full_v() {
    return mVarianceContainer;
}

template<typename Block>
ITensorContainer& ModularOptimizerStateManager<Block>::full_m_scales() {
    return mMomentumScaleContainer;
}

template<typename Block>
ITensorContainer& ModularOptimizerStateManager<Block>::full_v_scales() {
    return mVarianceScaleContainer;
}

template<typename Block>
void ModularOptimizerStateManager<Block>::StateContainer::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {

    auto* owner = mOwner;
    if (!owner) return;

    auto& cfg = owner->mConfig;
    auto& nb = owner->mNonBlockState;

    auto nb_embeddings = (mKind == Kind::M) ? nb.m_embeddings : nb.v_embeddings;
    auto nb_final_norm = (mKind == Kind::M) ? nb.m_final_norm : nb.v_final_norm;
    auto nb_lm_head = (mKind == Kind::M) ? nb.m_lm_head : nb.v_lm_head;

    callback("model.embed_tokens.weight", nb_embeddings);
    callback("model.norm.weight", nb_final_norm);
    if (!cfg.tied_embeddings) {
        callback("lm_head.weight", nb_lm_head);
    }

    for (int i = 0; i < cfg.num_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        auto& block = owner->mBlockState.at(i);
        auto& w = (mKind == Kind::M) ? block.m : block.v;

        callback(prefix + ".input_layernorm.weight", TensorShard(w.ln1.weight));
        callback(prefix + ".post_attention_layernorm.weight", TensorShard(w.ln2.weight));
        callback(prefix + ".self_attn.qkv.weight", TensorShard(w.attention.qkv_weight));
        if (w.attention.qkv_bias.has_value()) {
            callback(prefix + ".self_attn.qkv.bias", TensorShard(w.attention.qkv_bias.value()));
        }
        callback(prefix + ".self_attn.o_proj.weight", TensorShard(w.attention.out_weight));
        if (w.attention.q_norm_weight.has_value()) {
            callback(prefix + ".self_attn.q_norm.weight", TensorShard(w.attention.q_norm_weight.value()));
        }
        if (w.attention.k_norm_weight.has_value()) {
            callback(prefix + ".self_attn.k_norm.weight", TensorShard(w.attention.k_norm_weight.value()));
        }

        if constexpr (requires { w.mlp_up_weight; w.mlp_down_weight; }) {
            callback(prefix + ".mlp.up.weight", TensorShard(w.mlp_up_weight));
            callback(prefix + ".mlp.down_proj.weight", TensorShard(w.mlp_down_weight));
        }
    }
}

template<typename Block>
void ModularOptimizerStateManager<Block>::ScaleContainer::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {

    auto* owner = mOwner;
    if (!owner) return;

    auto& cfg = owner->mConfig;
    auto& nb = owner->mNonBlockState;

    const Tensor& nb_embeddings = (mKind == Kind::M) ? nb.m_embeddings_scales : nb.v_embeddings_scales;
    const Tensor& nb_final_norm = (mKind == Kind::M) ? nb.m_final_norm_scales : nb.v_final_norm_scales;
    const Tensor& nb_lm_head = (mKind == Kind::M) ? nb.m_lm_head_scales : nb.v_lm_head_scales;

    if (nb_embeddings.Data) callback("model.embed_tokens.weight", TensorShard(nb_embeddings));
    if (nb_final_norm.Data) callback("model.norm.weight", TensorShard(nb_final_norm));
    if (!cfg.tied_embeddings && nb_lm_head.Data) {
        callback("lm_head.weight", TensorShard(nb_lm_head));
    }

    for (int i = 0; i < cfg.num_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        auto& block = owner->mBlockState.at(i);
        auto& w = (mKind == Kind::M) ? block.m_scales : block.v_scales;

        if (w.ln1.weight.Data) callback(prefix + ".input_layernorm.weight", TensorShard(w.ln1.weight));
        if (w.ln2.weight.Data) callback(prefix + ".post_attention_layernorm.weight", TensorShard(w.ln2.weight));
        if (w.attention.qkv_weight.Data) callback(prefix + ".self_attn.qkv.weight", TensorShard(w.attention.qkv_weight));
        if (w.attention.qkv_bias.has_value() && w.attention.qkv_bias->Data) {
            callback(prefix + ".self_attn.qkv.bias", TensorShard(w.attention.qkv_bias.value()));
        }
        if (w.attention.out_weight.Data) callback(prefix + ".self_attn.o_proj.weight", TensorShard(w.attention.out_weight));
        if (w.attention.q_norm_weight.has_value() && w.attention.q_norm_weight->Data) {
            callback(prefix + ".self_attn.q_norm.weight", TensorShard(w.attention.q_norm_weight.value()));
        }
        if (w.attention.k_norm_weight.has_value() && w.attention.k_norm_weight->Data) {
            callback(prefix + ".self_attn.k_norm.weight", TensorShard(w.attention.k_norm_weight.value()));
        }

        if constexpr (requires { w.mlp_up_weight; w.mlp_down_weight; }) {
            if (w.mlp_up_weight.Data) callback(prefix + ".mlp.up.weight", TensorShard(w.mlp_up_weight));
            if (w.mlp_down_weight.Data) callback(prefix + ".mlp.down_proj.weight", TensorShard(w.mlp_down_weight));
        }
    }
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_OPTIMIZER_STATE_H
