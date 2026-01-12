// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_H
#define LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_H

#include <array>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "modules/weights/weight_manager_types.h"
#include "utilities/tensor_container.h"

class NCCLCommunicator;
class DeviceMemoryStack;

namespace modules {

/**
 * @brief Modular weight manager template
 *
 * Manages weights for a model composed of modular blocks. Provides the same
 * gather/get/release protocol as LLamaWeightsManager for prefetching overlap.
 *
 * Template parameter Block determines the weight structure per layer:
 * - Block::Weights defines the weight tensors for one transformer block
 * - Block::Config provides configuration for allocation
 */
template<typename Block>
class ModularWeightManager : public ITensorContainer {
public:
    using BlockWeights = typename Block::Weights;
    using BlockConfig = typename Block::Config;

    // Core types (kept as nested names for compatibility).
    using Config = ModularWeightManagerConfig<Block>;
    using NonBlockWeights = modules::NonBlockWeights;
    using FP8WeightCache = modules::FP8WeightCache;
    using FP4WeightCacheEntry = modules::FP4WeightCacheEntry;
    using FP4WeightCache = modules::FP4WeightCache;

    ModularWeightManager(const Config& config, TensorAllocator& allocator);
    ~ModularWeightManager();

    // ========================================================================
    // Weight access for forward/backward pass
    // ========================================================================

    void gather_block(int layer_idx, NCCLCommunicator& comm, cudaStream_t fetch_stream);
    BlockWeights& get_block(int layer_idx, cudaStream_t stream);
    void release_block(int layer_idx, cudaStream_t stream);

    // Non-block weight access
    void gather_embeddings(NCCLCommunicator& comm, cudaStream_t stream);
    Tensor& get_embeddings(cudaStream_t stream);
    void release_embeddings(cudaStream_t stream);

    void gather_final_norm(NCCLCommunicator& comm, cudaStream_t stream);
    Tensor& get_final_norm(cudaStream_t stream);
    void release_final_norm(cudaStream_t stream);

    void gather_lm_head(NCCLCommunicator& comm, cudaStream_t stream);
    Tensor& get_lm_head(cudaStream_t stream);
    void release_lm_head(cudaStream_t stream);

    // ========================================================================
    // Weight access for optimizer
    // ========================================================================

    void begin_optimizer(DeviceMemoryStack& memory, cudaStream_t stream);
    void end_optimizer(DeviceMemoryStack& memory);

    void fetch_master_block(int layer_idx, cudaStream_t fetch_stream);
    BlockWeights& get_master_block(int layer_idx, cudaStream_t stream);
    void release_master_block(int layer_idx, cudaStream_t compute_stream, cudaStream_t put_stream);

    TensorShard& get_master_embeddings();
    TensorShard& get_master_lm_head();
    TensorShard& get_master_final_norm();

    // ========================================================================
    // Initialization and I/O
    // ========================================================================

    void random_init(int seed, NCCLCommunicator& comm);
    void import_from_file(const std::string& filename, bool allow_cast, NCCLCommunicator& comm);
    void export_to_file(const std::string& filename, NCCLCommunicator& comm) const;

    void invalidate();
    void synchronize_absmax(NCCLCommunicator& comm);

    // ========================================================================
    // ITensorContainer interface for checkpointing
    // ========================================================================

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] int num_layers() const;

    // ========================================================================
    // FP8 Forward Weight Cache
    // ========================================================================

    [[nodiscard]] bool has_fp8_forward_cache() const;

    [[nodiscard]] FP8WeightCache& fp8_weight_cache();
    [[nodiscard]] const FP8WeightCache& fp8_weight_cache() const;

    using FP8CacheProvider = std::function<FP8WeightCache&()>;

    void set_fp8_cache_provider(FP8CacheProvider provider);
    void clear_fp8_cache_provider();

    // ========================================================================
    // FP4 Forward Weight Cache
    // ========================================================================

    [[nodiscard]] bool has_fp4_forward_cache() const;
    [[nodiscard]] FP4WeightCache& fp4_weight_cache();
    [[nodiscard]] const FP4WeightCache& fp4_weight_cache() const;

    [[nodiscard]] bool has_fp4_dgrad_cache() const;
    [[nodiscard]] FP4WeightCache& fp4_weight_cache_transposed();
    [[nodiscard]] const FP4WeightCache& fp4_weight_cache_transposed() const;

    void maybe_enable_fp4_persistent_cache(bool weights_static);

    void set_four_over_six(bool enable, recipes::FourOverSixErrorMetric metric = recipes::FourOverSixErrorMetric::MSE);

    [[nodiscard]] Tensor& fp4_weight_amax();
    [[nodiscard]] const Tensor& fp4_weight_amax() const;

    [[nodiscard]] Tensor& fp4_weight_amax_transposed();
    [[nodiscard]] const Tensor& fp4_weight_amax_transposed() const;

    // ========================================================================
    // Weight injection for QLoRA
    // ========================================================================

    using WeightProvider = std::function<BlockWeights&(int layer_idx, cudaStream_t stream)>;
    using NonBlockProvider = std::function<Tensor&(cudaStream_t stream)>;

    void set_weight_provider(WeightProvider provider);
    void set_embeddings_provider(NonBlockProvider provider);
    void set_final_norm_provider(NonBlockProvider provider);
    void set_lm_head_provider(NonBlockProvider provider);
    void clear_weight_provider();
    [[nodiscard]] bool has_weight_provider() const;

private:
    // External weight providers for QLoRA (when set, bypasses internal weight buffers)
    WeightProvider mExternalWeightProvider;
    NonBlockProvider mExternalEmbeddingsProvider;
    NonBlockProvider mExternalFinalNormProvider;
    NonBlockProvider mExternalLMHeadProvider;
    FP8CacheProvider mExternalFP8CacheProvider;  // For QLoRA-FP8 cache access

    Config mConfig;
    TensorAllocator* mAllocator;
    cudaDeviceProp mDeviceProp{};
    bool mStreamWeights = false;
    bool mOptimizerActive = false;
    cudaStream_t mOptimizerStream = nullptr;

    // Master weights (full precision, possibly sharded)
    std::vector<BlockWeights> mMasterBlocks;
    NonBlockWeights mMasterNonBlock;
    TensorShard mMasterEmbeddingsShardView;
    TensorShard mMasterLMHeadShardView;
    TensorShard mMasterFinalNormShardView;

    // Abs-max / scale storage for master tensors (device FP32 buffer; each tensor Stats points here).
    Tensor mAbsMaxes;

    // Work weights (possibly quantized, used for forward/backward)
    std::vector<BlockWeights> mWorkBlocks;
    NonBlockWeights mWorkNonBlock;

    // Double-buffered device cache for prefetching
    std::array<BlockWeights, 2> mPrefetchBuffer;
    std::array<GatherStatus, 2> mPrefetchStatus;

    // Per-layer status for non-streaming mode (work weights are persistent per layer).
    std::vector<GatherStatus> mLayerStatus;

    // Double-buffered master cache for optimizer
    std::array<BlockWeights, 2> mMasterBuffer;
    std::array<GatherStatus, 2> mMasterStatus;

    // Non-block gather status
    GatherStatus mEmbeddingsStatus;
    GatherStatus mFinalNormStatus;
    GatherStatus mLMHeadStatus;

    // Device staging for non-block master weights when offloaded (optimizer phase).
    NonBlockWeights mMasterNonBlockDevice;

    // Persistent quantized weights storage (when persistent_quants=true)
    std::vector<BlockWeights> mQuantBlocks;
    std::vector<int> mQuantBlockVersion;

    // Double-buffered device staging for quantized weights when offloaded
    std::array<BlockWeights, 2> mQuantBuffer;
    std::array<GatherStatus, 2> mQuantStatus;

    // FP8 forward weight cache
    FP8WeightCache mFP8WeightCache{};
    Tensor mFP8WeightStats{};
    int mFP8CacheLayerIdx = -1;

    // FP4 forward weight cache
    FP4WeightCache mFP4WeightCache{};
    FP4WeightCache mFP4WeightCacheT{};
    Tensor mFP4WeightAmax{};
    Tensor mFP4WeightAmaxT{};
    int mFP4CacheLayerIdx = -1;

    // Persistent per-layer FP4 caches
    bool mFP4PersistentCacheEnabled = false;
    bool mFP4PersistentCacheStatic = false;
    std::vector<int> mFP4PersistentCacheVersion;
    Tensor mFP4WeightAmaxAll{};
    Tensor mFP4WeightAmaxAllT{};
    std::array<Tensor, 4> mFP4WeightDataAll{};
    std::array<Tensor, 4> mFP4WeightScalesAll{};
    std::array<Tensor, 4> mFP4WeightDataAllT{};
    std::array<Tensor, 4> mFP4WeightScalesAllT{};

    // Cache versioning
    int mVersion = 0;

    // Internal helpers
    int find_free_buffer(const std::array<GatherStatus, 2>& status) const;
    void wait_for_buffer(GatherStatus& status, cudaStream_t stream) const;

    void allocate_block_weights(BlockWeights& block, ETensorDType matmul_dtype, ETensorDType other_dtype,
                                bool on_host, bool sharded);
    void allocate_non_block_weights(NonBlockWeights& weights, ETensorDType dtype, bool on_host, bool sharded);

    // FP8 weight cache helpers
    void quantize_weights_to_fp8_cache(const BlockWeights& src, cudaStream_t stream);

    // FP4 weight cache helpers
    void quantize_weights_to_fp4_cache(const BlockWeights& src, cudaStream_t stream);
    void quantize_weights_to_fp4_cache_transposed(const BlockWeights& src, cudaStream_t stream);
};

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_H
