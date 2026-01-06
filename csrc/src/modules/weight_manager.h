// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODULES_WEIGHT_MANAGER_H
#define LLMQ_SRC_MODULES_WEIGHT_MANAGER_H

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "module_concept.h"
#include "weight_schema.h"
#include "utilities/allocator.h"
#include "utilities/philox.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"

#include "kernels/kernels.h"
#include "recipes/nvfp4/nvfp4_recipe.h"

class NCCLCommunicator;
class DeviceMemoryStack;

namespace modules {

// Helper type trait to detect if a block has mlp_up_weight (dense block vs MoE)
// Must be defined early since it's used in multiple template functions.
template<typename T, typename = void>
struct has_mlp_weights : std::false_type {};

template<typename T>
struct has_mlp_weights<T, std::void_t<decltype(std::declval<T>().mlp_up_weight)>> : std::true_type {};

/**
 * @brief Status tracking for double-buffered prefetching
 */
struct GatherStatus {
    int layer_idx = -1;             ///< Which layer is stored in this buffer
    cudaEvent_t done_event = nullptr;  ///< Event to synchronize on
    bool fetch_pending = false;     ///< Whether a gather is in progress
    bool is_ready = true;           ///< Whether buffer is available for reuse
    int version = -1;               ///< Cache version for invalidation
};

/**
 * @brief Modular weight manager template
 *
 * Manages weights for a model composed of modular blocks. Provides the same
 * gather/get/release protocol as LLamaWeightsManager for prefetching overlap.
 *
 * Template parameter Block determines the weight structure per layer:
 * - Block::Weights defines the weight tensors for one transformer block
 * - Block::Config provides configuration for allocation
 *
 * Features:
 * - Double-buffered prefetching for compute/communication overlap
 * - CPU offloading support for large models
 * - Quantization support (master weights in high precision, work weights in low)
 * - NCCL sharding across multiple GPUs
 * - ITensorContainer interface for checkpoint save/load
 *
 * @tparam Block The transformer block module type (e.g., DenseTransformerBlock)
 */
template<typename Block>
class ModularWeightManager : public ITensorContainer {
public:
    using BlockWeights = typename Block::Weights;
    using BlockConfig = typename Block::Config;

    /**
     * @brief Configuration for weight manager
     */
    struct Config {
        int num_layers;
        BlockConfig block_config;

        // Data types
        ETensorDType master_dtype;      ///< Master weight dtype (typically BF16 or FP32)
        ETensorDType model_dtype;       ///< Non-matmul working dtype (typically BF16/FP32)
        ETensorDType matmul_dtype;      ///< Working dtype for linear/matmul weights (may be FP8)

        // Sharding
        int shard_idx;
        int num_shards;
        bool shard_weights = false;     ///< ZeRO-3/FSDP-like weight streaming (not fully implemented in modular path)

        // Offloading
        bool offload_master = false;    ///< Offload master weights to CPU
        bool offload_quants = false;    ///< Offload quantized weights to CPU (requires persistent_quants)
        bool use_zero_copy = false;     ///< Use zero-copy for CPU-GPU transfers
        EAllocationType offload_alloc = EAllocationType::PINNED;  ///< Host allocation kind for offloaded weights
        bool persistent_quants = false; ///< Keep quantized weights around instead of re-quantizing
        bool init_projections_to_zero = false;  ///< Init attn.out + ffn.down to zero (modded-nanogpt)

        // Non-block weights config
        int vocab_size;
        int hidden_size;
        bool tied_embeddings;

        // QLoRA: skip block weight allocation (weights provided externally via set_weight_provider)
        bool skip_block_allocation = false;

        // FP8 forward-only mode: cache FP8 weights for forward pass
        bool enable_fp8_forward = false;

        // FP4 forward-only mode: cache FP4 weights for forward pass (CUTLASS layout)
        // Optimizes datacenter GPUs where weight quantization overhead dominates
        bool enable_fp4_forward = false;

        // Four Over Six (4/6) adaptive block scaling for FP4 quantization.
        // When enabled, uses error-minimizing block scale selection (4.0 vs 6.0).
        // Only affects cached weight quantization when enable_fp4_forward is true.
        bool enable_four_over_six = false;
        recipes::FourOverSixErrorMetric four_over_six_metric = recipes::FourOverSixErrorMetric::MSE;
    };

    /**
     * @brief Non-block weights (embeddings, final norm, lm head)
     */
    struct NonBlockWeights {
        Tensor embeddings;              ///< (vocab_size, hidden_size)
        Tensor lm_head;                 ///< (vocab_size, hidden_size) - may alias embeddings
        Tensor final_norm_weight;       ///< (hidden_size,)
    };

    /**
     * @brief FP8 weight cache for FP8 forward-only mode
     *
     * When enable_fp8_forward is set, these hold the FP8 E4M3 quantized weights
     * for the forward pass. The cache is shared across layers (single buffer)
     * since weights are consumed immediately by the matmul and we only need
     * one layer's worth at a time.
     */
    struct FP8WeightCache {
        Tensor qkv_weight;      ///< FP8 E4M3, (QKV_C, C)
        Tensor o_weight;        ///< FP8 E4M3, (C, Hq*Hs)
        Tensor mlp_up_weight;   ///< FP8 E4M3, (2*D, C)
        Tensor mlp_down_weight; ///< FP8 E4M3, (C, D)
    };

    /**
     * @brief Cached FP4 weights for a single layer (CUTLASS layout).
     *
     * Pre-quantized weights eliminate the per-forward quantization overhead,
     * which is critical for datacenter GPUs (B200) where matmul is fast.
     * Each weight stores: FP4 data + block scales (FP8 E4M3) + global amax.
     *
     * The alpha correction factor is computed from (inp_amax * cached_weight_amax).
     */
    struct FP4WeightCacheEntry {
        Tensor data;        ///< FP4 packed data, shape (N, K/2)
        Tensor scales;      ///< Block scales, FP8 E4M3, CUTLASS layout
        float global_amax;  ///< Global amax used during quantization (host value)
    };

    struct FP4WeightCache {
        FP4WeightCacheEntry qkv_weight;
        FP4WeightCacheEntry o_weight;
        FP4WeightCacheEntry mlp_up_weight;
        FP4WeightCacheEntry mlp_down_weight;
    };

    ModularWeightManager(const Config& config, TensorAllocator& allocator);
    ~ModularWeightManager();

    // ========================================================================
    // Weight access for forward/backward pass
    // ========================================================================

    /**
     * @brief Initiate prefetch of block weights from CPU (if offloaded)
     *
     * Call this for layer N+1 while computing layer N for overlap.
     */
    void gather_block(int layer_idx, NCCLCommunicator& comm, cudaStream_t fetch_stream);

    /**
     * @brief Get block weights, waiting for prefetch if necessary
     *
     * Returns a reference to the weight buffer. The buffer remains valid
     * until release_block is called.
     */
    BlockWeights& get_block(int layer_idx, cudaStream_t stream);

    /**
     * @brief Release block weights, initiating offload if configured
     *
     * Signals that the weights are no longer needed for this layer.
     * If offloading is enabled, initiates D2H transfer.
     */
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

    /**
     * @brief Prepare for optimizer pass
     *
     * Allocates temporary buffers needed for optimizer if using offloading.
     */
    void begin_optimizer(DeviceMemoryStack& memory, cudaStream_t stream);

    /**
     * @brief Cleanup after optimizer pass
     */
    void end_optimizer(DeviceMemoryStack& memory);

    /**
     * @brief Fetch master weights for optimizer update
     */
    void fetch_master_block(int layer_idx, cudaStream_t fetch_stream);

    /**
     * @brief Get master weights for optimizer update
     */
    BlockWeights& get_master_block(int layer_idx, cudaStream_t stream);

    /**
     * @brief Release master weights after optimizer update
     */
    void release_master_block(int layer_idx, cudaStream_t compute_stream, cudaStream_t put_stream);

    // Non-block master weights (not double-buffered - small enough)
    TensorShard& get_master_embeddings();
    TensorShard& get_master_lm_head();
    TensorShard& get_master_final_norm();

    // ========================================================================
    // Initialization and I/O
    // ========================================================================

    /**
     * @brief Initialize weights randomly
     */
    void random_init(int seed, NCCLCommunicator& comm);

    /**
     * @brief Import weights from a safetensors file
     */
    void import_from_file(const std::string& filename, bool allow_cast, NCCLCommunicator& comm);

    /**
     * @brief Export weights to a safetensors file
     */
    void export_to_file(const std::string& filename, NCCLCommunicator& comm) const;

    /**
     * @brief Invalidate cached weights (after optimizer update)
     */
    void invalidate();

    /**
     * @brief Synchronize abs-max scales across shards
     */
    void synchronize_absmax(NCCLCommunicator& comm);

    // ========================================================================
    // ITensorContainer interface for checkpointing
    // ========================================================================

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int num_layers() const { return mConfig.num_layers; }

    // ========================================================================
    // FP8 Forward Weight Cache
    // ========================================================================

    /**
     * @brief Check if FP8 forward weight caching is enabled
     *
     * Returns true if either internal FP8 cache is enabled or an external
     * FP8 cache provider (e.g., from QLoRA FP8 weight provider) is set.
     */
    [[nodiscard]] bool has_fp8_forward_cache() const {
        return mConfig.enable_fp8_forward || (mExternalFP8CacheProvider != nullptr);
    }

    /**
     * @brief Get the FP8 weight cache for the current layer
     *
     * The cache contains FP8 E4M3 quantized weights that were prepared during
     * gather_block(). Valid only after get_block() is called for the same layer.
     * If an external FP8 cache provider is set (for QLoRA), returns from that instead.
     */
    [[nodiscard]] FP8WeightCache& fp8_weight_cache() {
        if (mExternalFP8CacheProvider) {
            return mExternalFP8CacheProvider();
        }
        return mFP8WeightCache;
    }
    [[nodiscard]] const FP8WeightCache& fp8_weight_cache() const {
        if (mExternalFP8CacheProvider) {
            return mExternalFP8CacheProvider();
        }
        return mFP8WeightCache;
    }

    /**
     * @brief FP8 cache provider callback type
     *
     * Used by QLoRA-FP8 to provide FP8 weights from the quantized weight storage.
     */
    using FP8CacheProvider = std::function<FP8WeightCache&()>;

    /**
     * @brief Set an external FP8 cache provider for QLoRA-FP8
     *
     * When set, fp8_weight_cache() will call this function to get the FP8 weight
     * cache instead of using the internal mFP8WeightCache.
     */
    void set_fp8_cache_provider(FP8CacheProvider provider) {
        mExternalFP8CacheProvider = std::move(provider);
    }

    void clear_fp8_cache_provider() {
        mExternalFP8CacheProvider = nullptr;
    }

    // ========================================================================
    // FP4 Forward Weight Cache (for datacenter GPU optimization)
    // ========================================================================

    /**
     * @brief Check if FP4 forward weight caching is enabled
     *
     * When enabled, weights are pre-quantized to FP4 with CUTLASS layout,
     * eliminating per-forward quantization overhead.
     */
    [[nodiscard]] bool has_fp4_forward_cache() const {
        return mConfig.enable_fp4_forward;
    }

    /**
     * @brief Get the FP4 weight cache for the current layer
     *
     * The cache contains pre-quantized FP4 weights (data + scales + global_amax).
     * Valid only after get_block() is called for the same layer.
     */
    [[nodiscard]] FP4WeightCache& fp4_weight_cache() {
        return mFP4WeightCache;
    }
    [[nodiscard]] const FP4WeightCache& fp4_weight_cache() const {
        return mFP4WeightCache;
    }

    /**
     * @brief Check if FP4 dgrad (transposed) weight caching is enabled.
     *
     * When enabled, we keep per-layer FP4 weights in transposed layout for the dgrad GEMM
     * (dinp = dout @ W). This avoids re-quantizing/transposing BF16 weights every backward pass
     * and is primarily beneficial on B200/B300 where FP4 GEMMs are so fast that weight prep
     * becomes the bottleneck.
     */
    [[nodiscard]] bool has_fp4_dgrad_cache() const {
        return mFP4PersistentCacheEnabled;
    }

    /**
     * @brief Get FP4 weights in transposed layout for the current layer (dgrad B matrix).
     *
     * Valid only after get_block() is called for the same layer.
     */
    [[nodiscard]] FP4WeightCache& fp4_weight_cache_transposed() {
        return mFP4WeightCacheT;
    }
    [[nodiscard]] const FP4WeightCache& fp4_weight_cache_transposed() const {
        return mFP4WeightCacheT;
    }

    /**
     * @brief Enable per-layer FP4 weight caching for Blackwell datacenter GPUs.
     *
     * Allocates persistent FP4 caches for all layers (forward + transposed/dgrad layouts).
     * Intended for LoRA/frozen-base training on B200/B300 where FP4 can be overhead-dominated.
     *
     * @param weights_static When true, cached FP4 weights are treated as immutable across steps.
     */
    void maybe_enable_fp4_persistent_cache(bool weights_static);

    /**
     * @brief Configure Four Over Six (4/6) adaptive block scaling for FP4 quantization.
     *
     * Call this to synchronize the 4/6 setting from the training recipe before
     * FP4 weight caching begins. Only affects cached weight quantization when
     * enable_fp4_forward is already enabled.
     *
     * @param enable Whether to enable 4/6 adaptive block scaling
     * @param metric Error metric for 4/6 selection (MSE, L1, or AbsMax)
     */
    void set_four_over_six(bool enable, recipes::FourOverSixErrorMetric metric = recipes::FourOverSixErrorMetric::MSE) {
        mConfig.enable_four_over_six = enable;
        mConfig.four_over_six_metric = metric;
    }

    /**
     * @brief Get the FP4 weight global amax tensor (forward cache)
     *
     * Contains 4 floats: [qkv_amax, o_amax, mlp_up_amax, mlp_down_amax]
     */
    [[nodiscard]] Tensor& fp4_weight_amax() {
        return mFP4WeightAmax;
    }
    [[nodiscard]] const Tensor& fp4_weight_amax() const {
        return mFP4WeightAmax;
    }

    /**
     * @brief Get the FP4 weight global amax tensor (transposed cache, for dgrad)
     *
     * Contains 4 floats: [qkv_amax, o_amax, mlp_up_amax, mlp_down_amax]
     * Separate from forward amax because forward may use 4/6 quantization while
     * transposed uses standard quantization, producing different amax values.
     */
    [[nodiscard]] Tensor& fp4_weight_amax_transposed() {
        return mFP4WeightAmaxT;
    }
    [[nodiscard]] const Tensor& fp4_weight_amax_transposed() const {
        return mFP4WeightAmaxT;
    }

    // ========================================================================
    // Weight injection for QLoRA
    // ========================================================================

    /**
     * @brief Set an external weight provider for QLoRA
     *
     * When set, get_block() will call this function to get weights instead
     * of using the internal weight buffers. This enables on-the-fly
     * dequantization of quantized base weights.
     *
     * @param provider Function that returns block weights for a given layer
     */
    using WeightProvider = std::function<BlockWeights&(int layer_idx, cudaStream_t stream)>;
    using NonBlockProvider = std::function<Tensor&(cudaStream_t stream)>;

    void set_weight_provider(WeightProvider provider) {
        mExternalWeightProvider = std::move(provider);
    }

    void set_embeddings_provider(NonBlockProvider provider) {
        mExternalEmbeddingsProvider = std::move(provider);
    }

    void set_final_norm_provider(NonBlockProvider provider) {
        mExternalFinalNormProvider = std::move(provider);
    }

    void set_lm_head_provider(NonBlockProvider provider) {
        mExternalLMHeadProvider = std::move(provider);
    }

    void clear_weight_provider() {
        mExternalWeightProvider = nullptr;
        mExternalEmbeddingsProvider = nullptr;
        mExternalFinalNormProvider = nullptr;
        mExternalLMHeadProvider = nullptr;
    }

    [[nodiscard]] bool has_weight_provider() const {
        return static_cast<bool>(mExternalWeightProvider);
    }

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
    bool mStreamWeights = false;           // ZeRO-3/FSDP-style weight streaming (2-buffer gather)
    bool mOptimizerActive = false;         // true between begin_optimizer/end_optimizer
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
    // These hold quantized copies of master weights to avoid re-quantization each step.
    std::vector<BlockWeights> mQuantBlocks;  ///< Per-layer quantized weights (may be on host if offload_quants)
    std::vector<int> mQuantBlockVersion;     ///< Per-layer version tracking (for invalidation)

    // Double-buffered device staging for quantized weights when offloaded (similar to master buffer)
    std::array<BlockWeights, 2> mQuantBuffer;
    std::array<GatherStatus, 2> mQuantStatus;

    // FP8 forward weight cache (when enable_fp8_forward=true)
    // Holds FP8 E4M3 quantized weights for fast forward pass matmuls.
    // Single buffer shared across layers (populated during gather_block).
    FP8WeightCache mFP8WeightCache{};
    Tensor mFP8WeightStats{};          ///< Stats buffer for FP8 weights (4 pairs = 8 floats for abs_max/scale)
    int mFP8CacheLayerIdx = -1;        ///< Which layer is currently in the FP8 cache

    // FP4 forward weight cache (when enable_fp4_forward=true)
    // Holds FP4 quantized weights with CUTLASS layout for fast forward pass matmuls.
    // Optimizes datacenter GPUs where weight quantization overhead dominates.
    FP4WeightCache mFP4WeightCache{};
    FP4WeightCache mFP4WeightCacheT{};   ///< FP4 weights in transposed layout (dgrad GEMM)
    Tensor mFP4WeightAmax{};           ///< Device buffer for global amax values (4 floats) - forward cache
    Tensor mFP4WeightAmaxT{};          ///< Device buffer for global amax values (4 floats) - transposed cache
    int mFP4CacheLayerIdx = -1;        ///< Which layer is currently in the FP4 cache

    // Persistent per-layer FP4 caches (opt-in, for Blackwell datacenter GPUs).
    bool mFP4PersistentCacheEnabled = false;
    bool mFP4PersistentCacheStatic = false;
    std::vector<int> mFP4PersistentCacheVersion;  ///< Per-layer version tracking (-1 = not cached)
    Tensor mFP4WeightAmaxAll{};                   ///< (num_layers*4,) global amax storage for all layers - forward
    Tensor mFP4WeightAmaxAllT{};                  ///< (num_layers*4,) global amax storage for all layers - transposed
    std::array<Tensor, 4> mFP4WeightDataAll{};    ///< Forward FP4 packed weights for all layers
    std::array<Tensor, 4> mFP4WeightScalesAll{};  ///< Forward FP4 block scales for all layers
    std::array<Tensor, 4> mFP4WeightDataAllT{};   ///< Transposed FP4 packed weights for dgrad
    std::array<Tensor, 4> mFP4WeightScalesAllT{}; ///< Transposed FP4 block scales for dgrad

    // Cache versioning
    int mVersion = 0;

    // Internal helpers
    int find_free_buffer(const std::array<GatherStatus, 2>& status) const;
    void wait_for_buffer(GatherStatus& status, cudaStream_t stream) const;
    void allocate_block_weights(BlockWeights& block, ETensorDType matmul_dtype, ETensorDType other_dtype, bool on_host, bool sharded);
    void allocate_non_block_weights(NonBlockWeights& weights, ETensorDType dtype, bool on_host, bool sharded);

    // FP8 weight cache helpers
    void quantize_weights_to_fp8_cache(const BlockWeights& src, cudaStream_t stream);

    // FP4 weight cache helpers
    void quantize_weights_to_fp4_cache(const BlockWeights& src, cudaStream_t stream);
    void quantize_weights_to_fp4_cache_transposed(const BlockWeights& src, cudaStream_t stream);
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Block>
ModularWeightManager<Block>::ModularWeightManager(const Config& config, TensorAllocator& allocator)
    : mConfig(config), mAllocator(&allocator) {

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProp, dev));

    // ZeRO-3/FSDP style: don't keep full per-layer weights, only a sharded master + 2 gathered buffers.
    mStreamWeights = (config.shard_weights && config.num_shards > 1);

    const bool sharded_master = (config.num_shards > 1);
    const bool separate_master_storage =
        mStreamWeights ||
        config.offload_master ||
        (config.master_dtype != config.model_dtype) ||
        (config.master_dtype != config.matmul_dtype);

    // ------------------------------------------------------------------------
    // Allocate WORK weights (full, used for forward/backward)
    // ------------------------------------------------------------------------
    // QLoRA mode: skip all weight allocation - weights are provided externally
    if (config.skip_block_allocation) {
        // All weights (including embeddings, final_norm, lm_head) will be provided via
        // set_weight_provider(), set_embeddings_provider(), etc.
        // No allocation needed here.
    } else if (mStreamWeights) {
        // Non-block work weights are persistent full tensors (used every step).
        allocate_non_block_weights(mWorkNonBlock, config.model_dtype, /*on_host=*/false, /*sharded=*/false);

        // Full block work weights are gathered into double-buffered prefetch slots.
        for (int i = 0; i < 2; ++i) {
            allocate_block_weights(mPrefetchBuffer[i], config.matmul_dtype, config.model_dtype, /*on_host=*/false, /*sharded=*/false);
            CUDA_CHECK(cudaEventCreate(&mPrefetchStatus[i].done_event));
            CUDA_CHECK(cudaEventRecord(mPrefetchStatus[i].done_event, 0));
        }
    } else {
        // Keep full per-layer work weights on device (like legacy ZeRO-1/2 path).
        mWorkBlocks.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_block_weights(mWorkBlocks[i], config.matmul_dtype, config.model_dtype, /*on_host=*/false, /*sharded=*/false);
        }
        allocate_non_block_weights(mWorkNonBlock, config.model_dtype, /*on_host=*/false, /*sharded=*/false);

        // One status/event per layer so we can prefetch (all-gather) the next layer safely.
        mLayerStatus.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            CUDA_CHECK(cudaEventCreate(&mLayerStatus[i].done_event));
            CUDA_CHECK(cudaEventRecord(mLayerStatus[i].done_event, 0));
            mLayerStatus[i].layer_idx = i;
        }
    }

    // ------------------------------------------------------------------------
    // Allocate MASTER weights (sharded, used for optimizer update/checkpointing)
    // ------------------------------------------------------------------------
    // QLoRA mode: skip master block allocation too
    if (config.skip_block_allocation) {
        // Master non-block weights only (for embeddings/final_norm checkpointing if needed)
        // Note: For QLoRA, base model weights are frozen and stored in QLoRAWeightsManager,
        // so we don't need master storage for them.
    } else if (separate_master_storage) {
        mMasterBlocks.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_block_weights(mMasterBlocks[i], config.master_dtype, config.master_dtype, config.offload_master, sharded_master);
        }
        allocate_non_block_weights(mMasterNonBlock, config.master_dtype, config.offload_master, sharded_master);
    } else {
        // Master weights are sharded views into the work weights.
        mMasterBlocks.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            auto shard = [&](const Tensor& t) -> Tensor {
                if (!sharded_master) return t;
                return static_cast<Tensor>(shard_view(t, config.shard_idx, config.num_shards));
            };

            auto& src = mWorkBlocks[i];
            auto& dst = mMasterBlocks[i];

            dst.ln1.weight = shard(src.ln1.weight);
            dst.attention.qkv_weight = shard(src.attention.qkv_weight);
            if (src.attention.qkv_bias.has_value()) {
                dst.attention.qkv_bias = shard(src.attention.qkv_bias.value());
            }
            dst.attention.out_weight = shard(src.attention.out_weight);
            dst.ln2.weight = shard(src.ln2.weight);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                dst.mlp_up_weight = shard(src.mlp_up_weight);
                dst.mlp_down_weight = shard(src.mlp_down_weight);
            }
        }

        auto shard_nb = [&](const Tensor& t) -> Tensor {
            if (!sharded_master) return t;
            return static_cast<Tensor>(shard_view(t, config.shard_idx, config.num_shards));
        };
        mMasterNonBlock.embeddings = shard_nb(mWorkNonBlock.embeddings);
        mMasterNonBlock.final_norm_weight = shard_nb(mWorkNonBlock.final_norm_weight);
        if (config.tied_embeddings) {
            mMasterNonBlock.lm_head = mMasterNonBlock.embeddings;
        } else {
            mMasterNonBlock.lm_head = shard_nb(mWorkNonBlock.lm_head);
        }
    }

    // ------------------------------------------------------------------------
    // Master shard views (TensorShard wrappers with global shapes)
    // ------------------------------------------------------------------------
    // QLoRA mode: skip shard views - non-block weights are provided externally
    if (!config.skip_block_allocation) {
        if (sharded_master) {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlock.embeddings, config.shard_idx, config.num_shards,
                                                    std::vector<long>{config.vocab_size, config.hidden_size});
            mMasterFinalNormShardView = TensorShard(mMasterNonBlock.final_norm_weight, config.shard_idx, config.num_shards,
                                                   std::vector<long>{config.hidden_size});
            if (config.tied_embeddings) {
                mMasterLMHeadShardView = mMasterEmbeddingsShardView;
            } else {
                mMasterLMHeadShardView = TensorShard(mMasterNonBlock.lm_head, config.shard_idx, config.num_shards,
                                                     std::vector<long>{config.vocab_size, config.hidden_size});
            }
        } else {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlock.embeddings);
            mMasterLMHeadShardView = TensorShard(mMasterNonBlock.lm_head);
            mMasterFinalNormShardView = TensorShard(mMasterNonBlock.final_norm_weight);
        }
    }

    // ------------------------------------------------------------------------
    // Allocate abs-max/scale storage and wire Stats pointers (master + work)
    // ------------------------------------------------------------------------
    mAbsMaxes = allocator.allocate(ETensorDType::FP32, "abs_maxes_modular", EAllocationType::ON_DEVICE,
                                   {6 + config.num_layers * 14});
    float* abs_maxes = mAbsMaxes.template get<float>();

    auto wire_nonblock = [&](NonBlockWeights& nb) {
        nb.embeddings.Stats = abs_maxes + 0;
        nb.final_norm_weight.Stats = abs_maxes + 2;
        nb.lm_head.Stats = abs_maxes + 4;
    };

    auto wire_block = [&](BlockWeights& b, int layer_idx) {
        float* a = abs_maxes + 6 + layer_idx * 14;
        b.attention.qkv_weight.Stats = a + 0;
        b.attention.out_weight.Stats = a + 2;
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            b.mlp_up_weight.Stats = a + 4;
            b.mlp_down_weight.Stats = a + 6;
        }
        if (b.attention.qkv_bias.has_value()) {
            b.attention.qkv_bias->Stats = a + 8;
        }
        b.ln1.weight.Stats = a + 10;
        b.ln2.weight.Stats = a + 12;
    };

    if (!config.skip_block_allocation) {
        wire_nonblock(mMasterNonBlock);
        for (int i = 0; i < config.num_layers; ++i) {
            wire_block(mMasterBlocks[i], i);
        }
    }

    // For single-GPU runs, set Stats directly on the full work weights too (no gather needed for propagation).
    if (config.num_shards == 1 && !config.skip_block_allocation) {
        wire_nonblock(mWorkNonBlock);
        if (!mStreamWeights) {
            for (int i = 0; i < config.num_layers; ++i) {
                wire_block(mWorkBlocks[i], i);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Status/events for non-block gather
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaEventCreate(&mEmbeddingsStatus.done_event));
    CUDA_CHECK(cudaEventCreate(&mFinalNormStatus.done_event));
    CUDA_CHECK(cudaEventCreate(&mLMHeadStatus.done_event));
    CUDA_CHECK(cudaEventRecord(mEmbeddingsStatus.done_event, 0));
    CUDA_CHECK(cudaEventRecord(mFinalNormStatus.done_event, 0));
    CUDA_CHECK(cudaEventRecord(mLMHeadStatus.done_event, 0));

    // ------------------------------------------------------------------------
    // Optimizer staging buffers for offloaded master weights (device copies)
    // ------------------------------------------------------------------------
    // QLoRA mode: skip staging buffer allocation - base weights are frozen and stored externally
    if (config.offload_master && !config.use_zero_copy && !config.skip_block_allocation) {
        for (int i = 0; i < 2; ++i) {
            allocate_block_weights(mMasterBuffer[i], config.master_dtype, config.master_dtype, /*on_host=*/false, /*sharded=*/sharded_master);
            CUDA_CHECK(cudaEventCreate(&mMasterStatus[i].done_event));
            CUDA_CHECK(cudaEventRecord(mMasterStatus[i].done_event, 0));
        }
        allocate_non_block_weights(mMasterNonBlockDevice, config.master_dtype, /*on_host=*/false, /*sharded=*/sharded_master);
    }

    // ------------------------------------------------------------------------
    // Persistent quantized weights storage (when persistent_quants=true)
    // ------------------------------------------------------------------------
    // When persistent_quants is enabled, we keep quantized copies of the master weights
    // to avoid re-quantizing each step. When offload_quants is also enabled, these
    // quantized weights are stored in pinned host memory.
    if (config.persistent_quants && !config.skip_block_allocation) {
        const bool quants_on_host = config.offload_quants;
        mQuantBlocks.resize(config.num_layers);
        mQuantBlockVersion.resize(config.num_layers, -1);  // -1 means "not yet quantized"
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_block_weights(mQuantBlocks[i], config.matmul_dtype, config.model_dtype, quants_on_host, sharded_master);
        }

        // If offloading quants and not using zero-copy, we need double-buffered device staging
        if (config.offload_quants && !config.use_zero_copy) {
            for (int i = 0; i < 2; ++i) {
                allocate_block_weights(mQuantBuffer[i], config.matmul_dtype, config.model_dtype, /*on_host=*/false, /*sharded=*/false);
                CUDA_CHECK(cudaEventCreate(&mQuantStatus[i].done_event));
                CUDA_CHECK(cudaEventRecord(mQuantStatus[i].done_event, 0));
            }
        }
    }

    // ------------------------------------------------------------------------
    // FP8 forward weight cache (when enable_fp8_forward=true)
    // ------------------------------------------------------------------------
    // Allocate a single set of FP8 weight buffers that are reused across layers.
    // Weights are quantized to FP8 during get_block and used for forward matmuls.
    // This works with both internal block weights and external providers (QLoRA).
    if (config.enable_fp8_forward) {
        const auto& bc = config.block_config;
        const long C = bc.hidden_size;
        const long Hq = bc.num_query_heads;
        const long Hkv = bc.num_kv_heads;
        const long Hs = bc.head_size;
        const long D = bc.intermediate_size;
        const long QKV_C = (Hq + 2 * Hkv) * Hs;

        // Stats buffer: 4 weights * 2 floats (abs_max, scale) = 8 floats
        mFP8WeightStats = mAllocator->allocate(ETensorDType::FP32, "fp8_weight_stats", EAllocationType::ON_DEVICE, {8});

        // Allocate FP8 weight buffers
        mFP8WeightCache.qkv_weight = mAllocator->allocate(ETensorDType::FP8_E4M3, "fp8_qkv_weight", EAllocationType::ON_DEVICE, {QKV_C, C});
        mFP8WeightCache.o_weight = mAllocator->allocate(ETensorDType::FP8_E4M3, "fp8_o_weight", EAllocationType::ON_DEVICE, {C, Hq * Hs});
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            mFP8WeightCache.mlp_up_weight = mAllocator->allocate(ETensorDType::FP8_E4M3, "fp8_mlp_up_weight", EAllocationType::ON_DEVICE, {2 * D, C});
            mFP8WeightCache.mlp_down_weight = mAllocator->allocate(ETensorDType::FP8_E4M3, "fp8_mlp_down_weight", EAllocationType::ON_DEVICE, {C, D});
        }

        // Assign Stats pointers: each weight gets 2 floats (abs_max at [0], scale at [1])
        float* stats = mFP8WeightStats.get<float>();
        mFP8WeightCache.qkv_weight.Stats = stats;      // [0], [1]
        mFP8WeightCache.o_weight.Stats = stats + 2;    // [2], [3]
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            mFP8WeightCache.mlp_up_weight.Stats = stats + 4;   // [4], [5]
            mFP8WeightCache.mlp_down_weight.Stats = stats + 6; // [6], [7]
        }
    }

    // ------------------------------------------------------------------------
    // FP4 forward weight cache (when enable_fp4_forward=true)
    // ------------------------------------------------------------------------
    // Allocate FP4 weight buffers with CUTLASS-compatible scale layout.
    // This eliminates per-forward weight quantization overhead on datacenter GPUs.
    if (config.enable_fp4_forward) {
        const auto& bc = config.block_config;
        const long C = bc.hidden_size;
        const long Hq = bc.num_query_heads;
        const long Hkv = bc.num_kv_heads;
        const long Hs = bc.head_size;
        const long D = bc.intermediate_size;
        const long QKV_C = (Hq + 2 * Hkv) * Hs;

        // Amax buffer: 4 floats for global amax values (forward cache)
        mFP4WeightAmax = mAllocator->allocate(ETensorDType::FP32, "fp4_weight_amax", EAllocationType::ON_DEVICE, {4});
        // Separate amax buffer for transposed cache (needed when forward uses 4/6 but transposed uses standard)
        mFP4WeightAmaxT = mAllocator->allocate(ETensorDType::FP32, "fp4_weight_amax_t", EAllocationType::ON_DEVICE, {4});

        // Helper to compute CUTLASS scale size
        auto cutlass_scale_size = [](long rows, long cols) -> long {
            constexpr int kBlockSize = 16;
            constexpr int kTileDim = 128;
            long num_scale_cols = (cols + kBlockSize - 1) / kBlockSize;
            long aligned_rows = ((rows + kTileDim - 1) / kTileDim) * kTileDim;
            long aligned_cols = ((num_scale_cols + 3) / 4) * 4;
            return aligned_rows * aligned_cols;
        };

        // QKV weight: (QKV_C, C)
        long qkv_scale_size = cutlass_scale_size(QKV_C, C);
        mFP4WeightCache.qkv_weight.data = mAllocator->allocate(ETensorDType::BYTE, "fp4_qkv_data", EAllocationType::ON_DEVICE, {QKV_C, C / 2});
        mFP4WeightCache.qkv_weight.scales = mAllocator->allocate(ETensorDType::BYTE, "fp4_qkv_scales", EAllocationType::ON_DEVICE, {qkv_scale_size});

        // O weight: (C, Hq*Hs)
        long o_scale_size = cutlass_scale_size(C, Hq * Hs);
        mFP4WeightCache.o_weight.data = mAllocator->allocate(ETensorDType::BYTE, "fp4_o_data", EAllocationType::ON_DEVICE, {C, (Hq * Hs) / 2});
        mFP4WeightCache.o_weight.scales = mAllocator->allocate(ETensorDType::BYTE, "fp4_o_scales", EAllocationType::ON_DEVICE, {o_scale_size});

        if constexpr (has_mlp_weights<BlockWeights>::value) {
            // MLP up weight: (2*D, C)
            long mlp_up_scale_size = cutlass_scale_size(2 * D, C);
            mFP4WeightCache.mlp_up_weight.data = mAllocator->allocate(ETensorDType::BYTE, "fp4_mlp_up_data", EAllocationType::ON_DEVICE, {2 * D, C / 2});
            mFP4WeightCache.mlp_up_weight.scales = mAllocator->allocate(ETensorDType::BYTE, "fp4_mlp_up_scales", EAllocationType::ON_DEVICE, {mlp_up_scale_size});

            // MLP down weight: (C, D)
            long mlp_down_scale_size = cutlass_scale_size(C, D);
            mFP4WeightCache.mlp_down_weight.data = mAllocator->allocate(ETensorDType::BYTE, "fp4_mlp_down_data", EAllocationType::ON_DEVICE, {C, D / 2});
            mFP4WeightCache.mlp_down_weight.scales = mAllocator->allocate(ETensorDType::BYTE, "fp4_mlp_down_scales", EAllocationType::ON_DEVICE, {mlp_down_scale_size});
        }
    }
}

template<typename Block>
ModularWeightManager<Block>::~ModularWeightManager() {
    for (int i = 0; i < 2; ++i) {
        if (mPrefetchStatus[i].done_event) {
            cudaEventDestroy(mPrefetchStatus[i].done_event);
        }
        if (mMasterStatus[i].done_event) {
            cudaEventDestroy(mMasterStatus[i].done_event);
        }
        if (mQuantStatus[i].done_event) {
            cudaEventDestroy(mQuantStatus[i].done_event);
        }
    }
    if (mEmbeddingsStatus.done_event) cudaEventDestroy(mEmbeddingsStatus.done_event);
    if (mFinalNormStatus.done_event) cudaEventDestroy(mFinalNormStatus.done_event);
    if (mLMHeadStatus.done_event) cudaEventDestroy(mLMHeadStatus.done_event);
}

template<typename Block>
void ModularWeightManager<Block>::maybe_enable_fp4_persistent_cache(bool weights_static) {
    if (!mConfig.enable_fp4_forward) return;

    // Already enabled: keep it enabled; if any caller marks weights as static, treat as static.
    if (mFP4PersistentCacheEnabled) {
        mFP4PersistentCacheStatic = mFP4PersistentCacheStatic || weights_static;
        return;
    }

    // Enable only on Blackwell GPUs by default
    const int sm_version = mDeviceProp.major * 10 + mDeviceProp.minor;
    if (sm_version < 100) {
        return;
    }

    // Don't enable for ZeRO-3/FSDP weight streaming: persistent caches would defeat the purpose.
    if (mStreamWeights) {
        return;
    }

    const auto& bc = mConfig.block_config;
    const long C = bc.hidden_size;
    const long Hq = bc.num_query_heads;
    const long Hkv = bc.num_kv_heads;
    const long Hs = bc.head_size;
    const long D = bc.intermediate_size;
    const long QKV_C = (Hq + 2 * Hkv) * Hs;

    // Helper to compute CUTLASS scale size (matches compute_nvfp4_cutlass_scale_size()).
    auto cutlass_scale_size = [](long rows, long cols) -> long {
        constexpr int kBlockSize = 16;
        constexpr int kTileDim = 128;
        long num_scale_cols = (cols + kBlockSize - 1) / kBlockSize;
        long aligned_rows = ((rows + kTileDim - 1) / kTileDim) * kTileDim;
        long aligned_cols = ((num_scale_cols + 3) / 4) * 4;
        return aligned_rows * aligned_cols;
    };

    const long L = mConfig.num_layers;

    // Amax storage: 4 floats per layer [qkv, o, mlp_up, mlp_down] - forward cache
    mFP4WeightAmaxAll = mAllocator->allocate(
        ETensorDType::FP32, "fp4_weight_amax_all", EAllocationType::ON_DEVICE, {L * 4});
    // Separate amax storage for transposed cache (needed when forward uses 4/6 but transposed uses standard)
    mFP4WeightAmaxAllT = mAllocator->allocate(
        ETensorDType::FP32, "fp4_weight_amax_all_t", EAllocationType::ON_DEVICE, {L * 4});

    // Forward (W) caches
    mFP4WeightDataAll[0] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_qkv_data_all", EAllocationType::ON_DEVICE, {L * QKV_C, C / 2});
    mFP4WeightScalesAll[0] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_qkv_scales_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(QKV_C, C)});

    mFP4WeightDataAll[1] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_o_data_all", EAllocationType::ON_DEVICE, {L * C, (Hq * Hs) / 2});
    mFP4WeightScalesAll[1] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_o_scales_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(C, Hq * Hs)});

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        mFP4WeightDataAll[2] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_up_data_all", EAllocationType::ON_DEVICE, {L * (2 * D), C / 2});
        mFP4WeightScalesAll[2] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_up_scales_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(2 * D, C)});

        mFP4WeightDataAll[3] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_down_data_all", EAllocationType::ON_DEVICE, {L * C, D / 2});
        mFP4WeightScalesAll[3] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_down_scales_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(C, D)});
    }

    // Transposed (W^T) caches for dgrad
    mFP4WeightDataAllT[0] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_qkv_data_t_all", EAllocationType::ON_DEVICE, {L * C, QKV_C / 2});
    mFP4WeightScalesAllT[0] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_qkv_scales_t_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(C, QKV_C)});

    mFP4WeightDataAllT[1] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_o_data_t_all", EAllocationType::ON_DEVICE, {L * (Hq * Hs), C / 2});
    mFP4WeightScalesAllT[1] = mAllocator->allocate(
        ETensorDType::BYTE, "fp4_o_scales_t_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(Hq * Hs, C)});

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        mFP4WeightDataAllT[2] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_up_data_t_all", EAllocationType::ON_DEVICE, {L * C, (2 * D) / 2});
        mFP4WeightScalesAllT[2] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_up_scales_t_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(C, 2 * D)});

        mFP4WeightDataAllT[3] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_down_data_t_all", EAllocationType::ON_DEVICE, {L * D, C / 2});
        mFP4WeightScalesAllT[3] = mAllocator->allocate(
            ETensorDType::BYTE, "fp4_mlp_down_scales_t_all", EAllocationType::ON_DEVICE, {L * cutlass_scale_size(D, C)});
    }

    mFP4PersistentCacheVersion.assign((std::size_t)L, -1);
    mFP4PersistentCacheEnabled = true;
    mFP4PersistentCacheStatic = weights_static;

    if (mConfig.shard_idx == 0) {
        std::fprintf(
            stderr,
            "FP4 persistent weight cache enabled (SM%d): caching FP4 weights for %ld layers (forward + dgrad, %s).\n",
            sm_version,
            L,
            weights_static ? "static" : "versioned");
    }
}

template<typename Block>
void ModularWeightManager<Block>::gather_block(int layer_idx, NCCLCommunicator& comm, cudaStream_t fetch_stream) {
    // If external weight provider is set (QLoRA), skip gather - weights are provided on-demand
    if (mExternalWeightProvider) {
        (void)layer_idx;
        (void)comm;
        (void)fetch_stream;
        return;
    }

    auto convert_into = [&](const Tensor& src, Tensor dst) {
        if (!src.Data || !dst.Data || src.nelem() == 0) return;

        // Fast path: identical storage.
        if (src.Data == dst.Data && src.DType == dst.DType) return;

        // Same dtype: memcpy (H2D/D2D/D2H as needed).
        if (src.DType == dst.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDefault, fetch_stream));
            return;
        }

        // Dtype conversion / quantization requires device-accessible input.
        if (src.Device == -1 && !mConfig.use_zero_copy) {
            throw std::runtime_error("ModularWeightManager::gather_block: dtype conversion from offloaded master weights requires --use-zero-copy");
        }

        if (dst.DType == ETensorDType::FP8_E4M3 || dst.DType == ETensorDType::FP8_E5M2 || dst.DType == ETensorDType::INT8) {
            if (!src.Stats) {
                throw std::runtime_error("ModularWeightManager::gather_block: FP8/INT8 gather requires Stats (abs_max/scale) on the source tensor");
            }
            quantize_with_abs_max(dst, /*scale_ptr=*/src.Stats + 1, src, /*abs_max=*/src.Stats, (long)src.nelem(), mDeviceProp, fetch_stream);
            return;
        }

        if (src.DType == ETensorDType::BF16 && dst.DType == ETensorDType::FP32) {
            convert_dtype(dst.get<float>(), reinterpret_cast<const nv_bfloat16*>(src.Data), src.nelem(), fetch_stream);
            return;
        }
        if (src.DType == ETensorDType::FP32 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), src.get<float>(), src.nelem(), fetch_stream);
            return;
        }
        if (src.DType == ETensorDType::FP16 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), reinterpret_cast<const half*>(src.Data), src.nelem(), fetch_stream);
            return;
        }

        throw std::runtime_error("ModularWeightManager::gather_block: unsupported dtype conversion");
    };

    if (!mStreamWeights) {
        auto& status = mLayerStatus.at(layer_idx);
        if (status.version == mVersion) return;

        CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.done_event));
        status.fetch_pending = true;
        status.is_ready = false;
        status.version = mVersion;

        auto& src = mMasterBlocks.at(layer_idx);
        auto& dst = mWorkBlocks.at(layer_idx);

        // Propagate Stats pointers (abs_max/scale) for this layer.
        dst.ln1.weight.Stats = src.ln1.weight.Stats;
        dst.attention.qkv_weight.Stats = src.attention.qkv_weight.Stats;
        dst.attention.out_weight.Stats = src.attention.out_weight.Stats;
        dst.ln2.weight.Stats = src.ln2.weight.Stats;
        if (src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            dst.attention.qkv_bias->Stats = src.attention.qkv_bias->Stats;
        }
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            dst.mlp_up_weight.Stats = src.mlp_up_weight.Stats;
            dst.mlp_down_weight.Stats = src.mlp_down_weight.Stats;
        }

        // Convert/copy local shard into the correct slice of the destination, then all-gather.
        auto shard_dst = [&](Tensor& full) -> Tensor {
            if (mConfig.num_shards == 1) return full;
            return static_cast<Tensor>(shard_view(full, mConfig.shard_idx, mConfig.num_shards));
        };

        convert_into(src.ln1.weight, shard_dst(dst.ln1.weight));
        convert_into(src.attention.qkv_weight, shard_dst(dst.attention.qkv_weight));
        if (src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            convert_into(src.attention.qkv_bias.value(), shard_dst(dst.attention.qkv_bias.value()));
        }
        convert_into(src.attention.out_weight, shard_dst(dst.attention.out_weight));
        if (src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
            convert_into(src.attention.q_norm_weight.value(), shard_dst(dst.attention.q_norm_weight.value()));
        }
        if (src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
            convert_into(src.attention.k_norm_weight.value(), shard_dst(dst.attention.k_norm_weight.value()));
        }
        convert_into(src.ln2.weight, shard_dst(dst.ln2.weight));
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(src.mlp_up_weight, shard_dst(dst.mlp_up_weight));
            convert_into(src.mlp_down_weight, shard_dst(dst.mlp_down_weight));
        }

        if (mConfig.num_shards == 1) {
            CUDA_CHECK(cudaEventRecord(status.done_event, fetch_stream));
            return;
        }

        comm.begin_transaction(fetch_stream);
        auto gather_full = [&](Tensor& full) {
            if (!full.Data || full.nelem() == 0) return;
            TensorShard local = shard_view(full, mConfig.shard_idx, mConfig.num_shards);
            local.Stats = full.Stats;
            comm.schedule_all_gather(local, full);
        };

        const auto& cfg = mConfig.block_config;
        long C = cfg.hidden_size;
        long HS = cfg.head_size;
        long HQ = cfg.num_query_heads;
        long HKV = cfg.num_kv_heads;
        long qkv_rows = HS * (HQ + 2 * HKV);
        long D = cfg.intermediate_size;

        gather_full(dst.ln1.weight);
        gather_full(dst.attention.qkv_weight);
        if (dst.attention.qkv_bias.has_value()) gather_full(dst.attention.qkv_bias.value());
        gather_full(dst.attention.out_weight);
        if (dst.attention.q_norm_weight.has_value()) gather_full(dst.attention.q_norm_weight.value());
        if (dst.attention.k_norm_weight.has_value()) gather_full(dst.attention.k_norm_weight.value());
        gather_full(dst.ln2.weight);
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            gather_full(dst.mlp_up_weight);
            gather_full(dst.mlp_down_weight);
        }

        comm.execute_transaction(status.done_event);
        return;
    }

    // Streamed ZeRO-3 mode: gather into a free prefetch buffer (double-buffered).
    int buf_idx = find_free_buffer(mPrefetchStatus);
    if (buf_idx < 0) {
        buf_idx = 0;
        wait_for_buffer(mPrefetchStatus[buf_idx], fetch_stream);
    }
    auto& status = mPrefetchStatus[buf_idx];
    if (status.layer_idx == layer_idx && status.version == mVersion) return;

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.done_event));
    status.layer_idx = layer_idx;
    status.fetch_pending = true;
    status.is_ready = false;
    status.version = mVersion;

    auto& master_src = mMasterBlocks.at(layer_idx);
    auto& dst = mPrefetchBuffer[buf_idx];

    // Prefetch buffers are reused across layers; update Stats pointers for this layer.
    dst.ln1.weight.Stats = master_src.ln1.weight.Stats;
    dst.attention.qkv_weight.Stats = master_src.attention.qkv_weight.Stats;
    dst.attention.out_weight.Stats = master_src.attention.out_weight.Stats;
    dst.ln2.weight.Stats = master_src.ln2.weight.Stats;
    if (master_src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
        dst.attention.qkv_bias->Stats = master_src.attention.qkv_bias->Stats;
    }
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        dst.mlp_up_weight.Stats = master_src.mlp_up_weight.Stats;
        dst.mlp_down_weight.Stats = master_src.mlp_down_weight.Stats;
    }

    auto shard_dst = [&](Tensor& full) -> Tensor {
        return static_cast<Tensor>(shard_view(full, mConfig.shard_idx, mConfig.num_shards));
    };

    // Determine the source for conversion: either persistent quant storage or master weights.
    // When persistent_quants is enabled, we use pre-quantized sharded weights to avoid re-quantizing.
    const bool use_persistent_quants = mConfig.persistent_quants && !mQuantBlocks.empty();

    // Helper to quantize from master to quant storage (used when quants are stale)
    auto update_quant_block = [&]() {
        auto& quant_dst = mQuantBlocks.at(layer_idx);
        // Quantize from master weights into quant storage
        convert_into(master_src.ln1.weight, quant_dst.ln1.weight);
        convert_into(master_src.attention.qkv_weight, quant_dst.attention.qkv_weight);
        if (master_src.attention.qkv_bias.has_value() && quant_dst.attention.qkv_bias.has_value()) {
            convert_into(master_src.attention.qkv_bias.value(), quant_dst.attention.qkv_bias.value());
        }
        convert_into(master_src.attention.out_weight, quant_dst.attention.out_weight);
        if (master_src.attention.q_norm_weight.has_value() && quant_dst.attention.q_norm_weight.has_value()) {
            convert_into(master_src.attention.q_norm_weight.value(), quant_dst.attention.q_norm_weight.value());
        }
        if (master_src.attention.k_norm_weight.has_value() && quant_dst.attention.k_norm_weight.has_value()) {
            convert_into(master_src.attention.k_norm_weight.value(), quant_dst.attention.k_norm_weight.value());
        }
        convert_into(master_src.ln2.weight, quant_dst.ln2.weight);
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(master_src.mlp_up_weight, quant_dst.mlp_up_weight);
            convert_into(master_src.mlp_down_weight, quant_dst.mlp_down_weight);
        }
        mQuantBlockVersion[layer_idx] = mVersion;
    };

    if (use_persistent_quants && mConfig.offload_quants && !mConfig.use_zero_copy) {
        // Offloaded quants without zero-copy: need to check if quants are stale and re-quantize.
        // For offloaded quants, we quantize to a device staging buffer, then copy D2H.
        const bool quants_stale = (mQuantBlockVersion[layer_idx] != mVersion);

        int qbuf_idx = find_free_buffer(mQuantStatus);
        if (qbuf_idx < 0) {
            qbuf_idx = layer_idx % 2;
            wait_for_buffer(mQuantStatus[qbuf_idx], fetch_stream);
        }
        CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, mQuantStatus[qbuf_idx].done_event));

        auto& quant_host = mQuantBlocks.at(layer_idx);
        auto& quant_device = mQuantBuffer[qbuf_idx];

        if (quants_stale) {
            // Quantize from master to device staging buffer
            convert_into(master_src.ln1.weight, quant_device.ln1.weight);
            convert_into(master_src.attention.qkv_weight, quant_device.attention.qkv_weight);
            if (master_src.attention.qkv_bias.has_value() && quant_device.attention.qkv_bias.has_value()) {
                convert_into(master_src.attention.qkv_bias.value(), quant_device.attention.qkv_bias.value());
            }
            convert_into(master_src.attention.out_weight, quant_device.attention.out_weight);
            convert_into(master_src.ln2.weight, quant_device.ln2.weight);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                convert_into(master_src.mlp_up_weight, quant_device.mlp_up_weight);
                convert_into(master_src.mlp_down_weight, quant_device.mlp_down_weight);
            }

            // Copy from device staging to host storage (persist for future steps)
            auto copy_d2h = [fetch_stream](const Tensor& src, Tensor& dst) {
                if (!src.Data || !dst.Data || src.nelem() == 0) return;
                CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, src.bytes(), cudaMemcpyDeviceToHost, fetch_stream));
            };
            copy_d2h(quant_device.ln1.weight, quant_host.ln1.weight);
            copy_d2h(quant_device.attention.qkv_weight, quant_host.attention.qkv_weight);
            if (quant_device.attention.qkv_bias.has_value() && quant_host.attention.qkv_bias.has_value()) {
                copy_d2h(quant_device.attention.qkv_bias.value(), quant_host.attention.qkv_bias.value());
            }
            copy_d2h(quant_device.attention.out_weight, quant_host.attention.out_weight);
            copy_d2h(quant_device.ln2.weight, quant_host.ln2.weight);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                copy_d2h(quant_device.mlp_up_weight, quant_host.mlp_up_weight);
                copy_d2h(quant_device.mlp_down_weight, quant_host.mlp_down_weight);
            }
            mQuantBlockVersion[layer_idx] = mVersion;
        } else {
            // Quants are fresh: copy from host to device staging buffer
            auto copy_h2d = [fetch_stream](const Tensor& src, Tensor& dst) {
                if (!src.Data || !dst.Data || src.nelem() == 0) return;
                CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, src.bytes(), cudaMemcpyHostToDevice, fetch_stream));
            };
            copy_h2d(quant_host.ln1.weight, quant_device.ln1.weight);
            copy_h2d(quant_host.attention.qkv_weight, quant_device.attention.qkv_weight);
            if (quant_host.attention.qkv_bias.has_value() && quant_device.attention.qkv_bias.has_value()) {
                copy_h2d(quant_host.attention.qkv_bias.value(), quant_device.attention.qkv_bias.value());
            }
            copy_h2d(quant_host.attention.out_weight, quant_device.attention.out_weight);
            copy_h2d(quant_host.ln2.weight, quant_device.ln2.weight);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                copy_h2d(quant_host.mlp_up_weight, quant_device.mlp_up_weight);
                copy_h2d(quant_host.mlp_down_weight, quant_device.mlp_down_weight);
            }
        }

        // Copy from device staging to prefetch buffer (device-to-device, same dtype - no conversion needed)
        convert_into(quant_device.ln1.weight, shard_dst(dst.ln1.weight));
        convert_into(quant_device.attention.qkv_weight, shard_dst(dst.attention.qkv_weight));
        if (quant_device.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            convert_into(quant_device.attention.qkv_bias.value(), shard_dst(dst.attention.qkv_bias.value()));
        }
        convert_into(quant_device.attention.out_weight, shard_dst(dst.attention.out_weight));
        if (quant_device.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
            convert_into(quant_device.attention.q_norm_weight.value(), shard_dst(dst.attention.q_norm_weight.value()));
        }
        if (quant_device.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
            convert_into(quant_device.attention.k_norm_weight.value(), shard_dst(dst.attention.k_norm_weight.value()));
        }
        convert_into(quant_device.ln2.weight, shard_dst(dst.ln2.weight));
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(quant_device.mlp_up_weight, shard_dst(dst.mlp_up_weight));
            convert_into(quant_device.mlp_down_weight, shard_dst(dst.mlp_down_weight));
        }

        CUDA_CHECK(cudaEventRecord(mQuantStatus[qbuf_idx].done_event, fetch_stream));
    } else if (use_persistent_quants) {
        // Persistent quants on device (or zero-copy): check if stale and re-quantize if needed.
        if (mQuantBlockVersion[layer_idx] != mVersion) {
            update_quant_block();
        }

        // Copy from quant storage to prefetch buffer
        auto& quant_src = mQuantBlocks.at(layer_idx);
        convert_into(quant_src.ln1.weight, shard_dst(dst.ln1.weight));
        convert_into(quant_src.attention.qkv_weight, shard_dst(dst.attention.qkv_weight));
        if (quant_src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            convert_into(quant_src.attention.qkv_bias.value(), shard_dst(dst.attention.qkv_bias.value()));
        }
        convert_into(quant_src.attention.out_weight, shard_dst(dst.attention.out_weight));
        if (quant_src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
            convert_into(quant_src.attention.q_norm_weight.value(), shard_dst(dst.attention.q_norm_weight.value()));
        }
        if (quant_src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
            convert_into(quant_src.attention.k_norm_weight.value(), shard_dst(dst.attention.k_norm_weight.value()));
        }
        convert_into(quant_src.ln2.weight, shard_dst(dst.ln2.weight));
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(quant_src.mlp_up_weight, shard_dst(dst.mlp_up_weight));
            convert_into(quant_src.mlp_down_weight, shard_dst(dst.mlp_down_weight));
        }
    } else {
        // No persistent quants: convert from master weights (with potential quantization).
        convert_into(master_src.ln1.weight, shard_dst(dst.ln1.weight));
        convert_into(master_src.attention.qkv_weight, shard_dst(dst.attention.qkv_weight));
        if (master_src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
            convert_into(master_src.attention.qkv_bias.value(), shard_dst(dst.attention.qkv_bias.value()));
        }
        convert_into(master_src.attention.out_weight, shard_dst(dst.attention.out_weight));
        if (master_src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
            convert_into(master_src.attention.q_norm_weight.value(), shard_dst(dst.attention.q_norm_weight.value()));
        }
        if (master_src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
            convert_into(master_src.attention.k_norm_weight.value(), shard_dst(dst.attention.k_norm_weight.value()));
        }
        convert_into(master_src.ln2.weight, shard_dst(dst.ln2.weight));
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            convert_into(master_src.mlp_up_weight, shard_dst(dst.mlp_up_weight));
            convert_into(master_src.mlp_down_weight, shard_dst(dst.mlp_down_weight));
        }
    }

    comm.begin_transaction(fetch_stream);
    auto gather_full = [&](Tensor& full) {
        if (!full.Data || full.nelem() == 0) return;
        TensorShard local = shard_view(full, mConfig.shard_idx, mConfig.num_shards);
        local.Stats = full.Stats;
        comm.schedule_all_gather(local, full);
    };
    gather_full(dst.ln1.weight);
    gather_full(dst.attention.qkv_weight);
    if (dst.attention.qkv_bias.has_value()) gather_full(dst.attention.qkv_bias.value());
    gather_full(dst.attention.out_weight);
    if (dst.attention.q_norm_weight.has_value()) gather_full(dst.attention.q_norm_weight.value());
    if (dst.attention.k_norm_weight.has_value()) gather_full(dst.attention.k_norm_weight.value());
    gather_full(dst.ln2.weight);
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        gather_full(dst.mlp_up_weight);
        gather_full(dst.mlp_down_weight);
    }
    comm.execute_transaction(status.done_event);
}

template<typename Block>
typename Block::Weights& ModularWeightManager<Block>::get_block(int layer_idx, cudaStream_t stream) {
    BlockWeights* result = nullptr;

    // If external weight provider is set (QLoRA), use it instead
    if (mExternalWeightProvider) {
        result = &mExternalWeightProvider(layer_idx, stream);
    } else if (!mStreamWeights) {
        auto& status = mLayerStatus.at(layer_idx);
        wait_for_buffer(status, stream);
        status.is_ready = false;
        result = &mWorkBlocks.at(layer_idx);
    } else {
        for (int i = 0; i < 2; ++i) {
            if (mPrefetchStatus[i].layer_idx == layer_idx) {
                wait_for_buffer(mPrefetchStatus[i], stream);
                mPrefetchStatus[i].is_ready = false;
                result = &mPrefetchBuffer[i];
                break;
            }
        }
    }

    if (!result) {
        throw std::runtime_error("Block weights not prefetched: layer " + std::to_string(layer_idx));
    }

    // Quantize weights to FP8 cache if FP8 forward mode is enabled.
    // This is done once per get_block call, and the cached FP8 weights
    // are then used by forward_qmm_fp8 for fast FP8 matmuls.
    if (mConfig.enable_fp8_forward && mFP8CacheLayerIdx != layer_idx) {
        quantize_weights_to_fp8_cache(*result, stream);
        mFP8CacheLayerIdx = layer_idx;
    }

    // Quantize weights to FP4 cache if FP4 forward mode is enabled.
    // Pre-quantized FP4 weights eliminate per-forward quantization overhead
    // on datacenter GPUs (B200) where quantization dominates runtime.
    if (mConfig.enable_fp4_forward) {
        if (mFP4PersistentCacheEnabled) {
            // Persistent per-layer FP4 caches: reuse weights across forward/backward (and across steps if static).
            const auto& cfg = mConfig.block_config;
            const long C = cfg.hidden_size;
            const long Hq = cfg.num_query_heads;
            const long Hkv = cfg.num_kv_heads;
            const long Hs = cfg.head_size;
            const long D = cfg.intermediate_size;
            const long QKV_C = (Hq + 2 * Hkv) * Hs;

            // Helper to compute CUTLASS scale size (matches compute_nvfp4_cutlass_scale_size()).
            auto cutlass_scale_size = [](long rows, long cols) -> long {
                constexpr int kBlockSize = 16;
                constexpr int kTileDim = 128;
                long num_scale_cols = (cols + kBlockSize - 1) / kBlockSize;
                long aligned_rows = ((rows + kTileDim - 1) / kTileDim) * kTileDim;
                long aligned_cols = ((num_scale_cols + 3) / 4) * 4;
                return aligned_rows * aligned_cols;
            };

            const long qkv_scale_size = cutlass_scale_size(QKV_C, C);
            const long o_scale_size = cutlass_scale_size(C, Hq * Hs);
            const long qkv_scale_size_t = cutlass_scale_size(C, QKV_C);
            const long o_scale_size_t = cutlass_scale_size(Hq * Hs, C);

            long mlp_up_scale_size = 0;
            long mlp_down_scale_size = 0;
            long mlp_up_scale_size_t = 0;
            long mlp_down_scale_size_t = 0;
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                mlp_up_scale_size = cutlass_scale_size(2 * D, C);
                mlp_down_scale_size = cutlass_scale_size(C, D);
                mlp_up_scale_size_t = cutlass_scale_size(C, 2 * D);
                mlp_down_scale_size_t = cutlass_scale_size(D, C);
            }

            auto set_views = [&](int l) {
                // Amax view: (4,) for this layer - forward cache
                mFP4WeightAmax = slice(mFP4WeightAmaxAll, 0, l * 4, l * 4 + 4);
                // Amax view: (4,) for this layer - transposed cache
                mFP4WeightAmaxT = slice(mFP4WeightAmaxAllT, 0, l * 4, l * 4 + 4);

                // Forward weights (W)
                mFP4WeightCache.qkv_weight.data = slice(mFP4WeightDataAll[0], 0, l * QKV_C, (l + 1) * QKV_C);
                mFP4WeightCache.qkv_weight.scales = slice(mFP4WeightScalesAll[0], 0, l * qkv_scale_size, (l + 1) * qkv_scale_size);

                mFP4WeightCache.o_weight.data = slice(mFP4WeightDataAll[1], 0, l * C, (l + 1) * C);
                mFP4WeightCache.o_weight.scales = slice(mFP4WeightScalesAll[1], 0, l * o_scale_size, (l + 1) * o_scale_size);

                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    mFP4WeightCache.mlp_up_weight.data = slice(mFP4WeightDataAll[2], 0, l * (2 * D), (l + 1) * (2 * D));
                    mFP4WeightCache.mlp_up_weight.scales = slice(mFP4WeightScalesAll[2], 0, l * mlp_up_scale_size, (l + 1) * mlp_up_scale_size);

                    mFP4WeightCache.mlp_down_weight.data = slice(mFP4WeightDataAll[3], 0, l * C, (l + 1) * C);
                    mFP4WeightCache.mlp_down_weight.scales = slice(mFP4WeightScalesAll[3], 0, l * mlp_down_scale_size, (l + 1) * mlp_down_scale_size);
                }

                // Transposed weights (W^T) for dgrad
                mFP4WeightCacheT.qkv_weight.data = slice(mFP4WeightDataAllT[0], 0, l * C, (l + 1) * C);
                mFP4WeightCacheT.qkv_weight.scales = slice(mFP4WeightScalesAllT[0], 0, l * qkv_scale_size_t, (l + 1) * qkv_scale_size_t);

                mFP4WeightCacheT.o_weight.data = slice(mFP4WeightDataAllT[1], 0, l * (Hq * Hs), (l + 1) * (Hq * Hs));
                mFP4WeightCacheT.o_weight.scales = slice(mFP4WeightScalesAllT[1], 0, l * o_scale_size_t, (l + 1) * o_scale_size_t);

                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    mFP4WeightCacheT.mlp_up_weight.data = slice(mFP4WeightDataAllT[2], 0, l * C, (l + 1) * C);
                    mFP4WeightCacheT.mlp_up_weight.scales = slice(mFP4WeightScalesAllT[2], 0, l * mlp_up_scale_size_t, (l + 1) * mlp_up_scale_size_t);

                    mFP4WeightCacheT.mlp_down_weight.data = slice(mFP4WeightDataAllT[3], 0, l * D, (l + 1) * D);
                    mFP4WeightCacheT.mlp_down_weight.scales = slice(mFP4WeightScalesAllT[3], 0, l * mlp_down_scale_size_t, (l + 1) * mlp_down_scale_size_t);
                }

                mFP4CacheLayerIdx = l;
            };

            // Ensure views are set for this layer (so callers can access cache tensors).
            if (mFP4CacheLayerIdx != layer_idx) {
                set_views(layer_idx);
            }

            const int wanted_version = mFP4PersistentCacheStatic ? 0 : mVersion;
            if (mFP4PersistentCacheVersion.at((std::size_t)layer_idx) != wanted_version) {
                // Quantize weights into the persistent per-layer buffers.
                quantize_weights_to_fp4_cache(*result, stream);
                quantize_weights_to_fp4_cache_transposed(*result, stream);
                mFP4PersistentCacheVersion.at((std::size_t)layer_idx) = wanted_version;
            }
        } else if (mFP4CacheLayerIdx != layer_idx) {
            // Single-buffer cache (legacy): quantize per get_block
            quantize_weights_to_fp4_cache(*result, stream);
            mFP4CacheLayerIdx = layer_idx;
        }
    }

    return *result;
}

template<typename Block>
void ModularWeightManager<Block>::release_block(int layer_idx, cudaStream_t stream) {
    // If external weight provider is set (QLoRA), skip release - provider manages its own buffers
    if (mExternalWeightProvider) {
        (void)layer_idx;
        (void)stream;
        return;
    }

    if (!mStreamWeights) {
        auto& st = mLayerStatus.at(layer_idx);
        CUDA_CHECK(cudaEventRecord(st.done_event, stream));
        st.is_ready = true;
        return;
    }

    for (int i = 0; i < 2; ++i) {
        if (mPrefetchStatus[i].layer_idx == layer_idx) {
            CUDA_CHECK(cudaEventRecord(mPrefetchStatus[i].done_event, stream));
            mPrefetchStatus[i].is_ready = true;
            return;
        }
    }
}

template<typename Block>
void ModularWeightManager<Block>::invalidate() {
    ++mVersion;
}

template<typename Block>
int ModularWeightManager<Block>::find_free_buffer(const std::array<GatherStatus, 2>& status) const {
    for (int i = 0; i < 2; ++i) {
        if (status[i].is_ready) return i;
    }
    return -1;
}

template<typename Block>
void ModularWeightManager<Block>::wait_for_buffer(GatherStatus& status, cudaStream_t stream) const {
    if (status.fetch_pending) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, status.done_event));
        status.fetch_pending = false;
    }
}

template<typename Block>
void ModularWeightManager<Block>::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {

    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;
    long qkv_rows = HS * (HQ + 2 * HKV);
    long q_rows = HS * HQ;

    auto shard = [&](const Tensor& t, const std::vector<long>& global_shape) -> TensorShard {
        if (mConfig.num_shards == 1) return TensorShard(t);
        return TensorShard(t, mConfig.shard_idx, mConfig.num_shards, global_shape);
    };

    // Non-block weights
    callback("model.embed_tokens.weight", mConfig.num_shards == 1
                                           ? TensorShard(mMasterNonBlock.embeddings)
                                           : TensorShard(mMasterNonBlock.embeddings, mConfig.shard_idx, mConfig.num_shards,
                                                        std::vector<long>{mConfig.vocab_size, mConfig.hidden_size}));
    callback("model.norm.weight", mConfig.num_shards == 1
                                     ? TensorShard(mMasterNonBlock.final_norm_weight)
                                     : TensorShard(mMasterNonBlock.final_norm_weight, mConfig.shard_idx, mConfig.num_shards,
                                                  std::vector<long>{mConfig.hidden_size}));
    if (!mConfig.tied_embeddings) {
        callback("lm_head.weight", mConfig.num_shards == 1
                                      ? TensorShard(mMasterNonBlock.lm_head)
                                      : TensorShard(mMasterNonBlock.lm_head, mConfig.shard_idx, mConfig.num_shards,
                                                   std::vector<long>{mConfig.vocab_size, mConfig.hidden_size}));
    }

    // Block weights
    for (int i = 0; i < mConfig.num_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        auto& block = mMasterBlocks[i];

        // Layer norms
        callback(prefix + ".input_layernorm.weight", shard(block.ln1.weight, {C}));
        callback(prefix + ".post_attention_layernorm.weight", shard(block.ln2.weight, {C}));

        // Attention weights (fused QKV stored as single tensor)
        callback(prefix + ".self_attn.qkv.weight", shard(block.attention.qkv_weight, {qkv_rows, C}));
        if (block.attention.qkv_bias.has_value()) {
            callback(prefix + ".self_attn.qkv.bias", shard(block.attention.qkv_bias.value(), {qkv_rows}));
        }
        callback(prefix + ".self_attn.o_proj.weight", shard(block.attention.out_weight, {C, q_rows}));
        if (block.attention.q_norm_weight.has_value()) {
            callback(prefix + ".self_attn.q_norm.weight", shard(block.attention.q_norm_weight.value(), {HS}));
        }
        if (block.attention.k_norm_weight.has_value()) {
            callback(prefix + ".self_attn.k_norm.weight", shard(block.attention.k_norm_weight.value(), {HS}));
        }

        // MLP weights - only for dense blocks
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            callback(prefix + ".mlp.up.weight", shard(block.mlp_up_weight, {2 * D, C}));
            callback(prefix + ".mlp.down_proj.weight", shard(block.mlp_down_weight, {C, D}));
        }
        // TODO: Add MoE-specific weights (router, experts) for MoE blocks
    }
}

// Block weight allocation - handles both dense and MoE blocks
template<typename Block>
void ModularWeightManager<Block>::allocate_block_weights(BlockWeights& block, ETensorDType matmul_dtype, ETensorDType other_dtype, bool on_host, bool sharded) {
    auto kind = on_host ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;
    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;
    long qkv_channels = HS * (HQ + 2 * HKV);

    auto alloc = [&](ETensorDType dtype, const char* name, const std::vector<long>& shape) -> Tensor {
        if (!sharded) {
            return mAllocator->allocate(dtype, name, kind, shape);
        }
        TensorShard shard = mAllocator->allocate_shard(dtype, mConfig.shard_idx, mConfig.num_shards, name, shape, kind);
        return static_cast<Tensor>(shard);
    };

    // Legacy parity: norm weights + biases stay in model dtype; matmul weights may be FP8.
    block.ln1.weight = alloc(other_dtype, "ln1_w", {C});

    // Attention weights
    block.attention.qkv_weight = alloc(matmul_dtype, "attn_qkv_w", {qkv_channels, C});
    if (cfg.use_qkv_bias) {
        block.attention.qkv_bias = alloc(other_dtype, "attn_qkv_b", {qkv_channels});
    }
    block.attention.out_weight = alloc(matmul_dtype, "attn_out_w", {C, HS * HQ});
    const bool use_qk_norm = [&]() -> bool {
        if constexpr (requires { cfg.use_qk_norm; }) return cfg.use_qk_norm;
        return false;
    }();
    if (use_qk_norm) {
        block.attention.q_norm_weight = alloc(other_dtype, "attn_q_norm_w", {HS});
        block.attention.k_norm_weight = alloc(other_dtype, "attn_k_norm_w", {HS});
    } else {
        block.attention.q_norm_weight.reset();
        block.attention.k_norm_weight.reset();
    }

    // LN2 weights
    block.ln2.weight = alloc(other_dtype, "ln2_w", {C});

    // MLP weights (only for dense blocks)
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        long D = cfg.intermediate_size;
        block.mlp_up_weight = alloc(matmul_dtype, "mlp_up_w", {2 * D, C});
        block.mlp_down_weight = alloc(matmul_dtype, "mlp_down_w", {C, D});
    }
    // TODO: Add MoE-specific weight allocation (router, experts) for MoE blocks
}

template<typename Block>
void ModularWeightManager<Block>::allocate_non_block_weights(NonBlockWeights& weights, ETensorDType dtype, bool on_host, bool sharded) {
    auto kind = on_host ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;
    long V = mConfig.vocab_size;
    long C = mConfig.hidden_size;

    auto alloc = [&](const char* name, const std::vector<long>& shape) -> Tensor {
        if (!sharded) {
            return mAllocator->allocate(dtype, name, kind, shape);
        }
        TensorShard shard = mAllocator->allocate_shard(dtype, mConfig.shard_idx, mConfig.num_shards, name, shape, kind);
        return static_cast<Tensor>(shard);
    };

    weights.embeddings = alloc("embeddings", {V, C});
    weights.final_norm_weight = alloc("final_norm", {C});

    if (mConfig.tied_embeddings) {
        weights.lm_head = weights.embeddings;  // Alias
    } else {
        weights.lm_head = alloc("lm_head", {V, C});
    }
}

template<typename Block>
void ModularWeightManager<Block>::gather_embeddings(NCCLCommunicator& comm, cudaStream_t stream) {
    // If external provider is set, skip gather - provider manages its own weights
    if (mExternalEmbeddingsProvider) {
        (void)comm; (void)stream;
        return;
    }
    if (mEmbeddingsStatus.version == mVersion) return;

    CUDA_CHECK(cudaStreamWaitEvent(stream, mEmbeddingsStatus.done_event));
    mEmbeddingsStatus.fetch_pending = true;
    mEmbeddingsStatus.is_ready = false;
    mEmbeddingsStatus.version = mVersion;

    // Propagate Stats pointer to the full work tensor.
    mWorkNonBlock.embeddings.Stats = mMasterNonBlock.embeddings.Stats;

    auto convert_into = [&](const Tensor& src, Tensor& dst) {
        if (!src.Data || !dst.Data || src.nelem() == 0) return;
        dst.Stats = src.Stats;
        if (src.Data == dst.Data && src.DType == dst.DType) return;
        if (src.DType == dst.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDefault, stream));
            return;
        }
        if (src.Device == -1 && !mConfig.use_zero_copy) {
            throw std::runtime_error("ModularWeightManager::gather_embeddings: dtype conversion from offloaded master weights requires --use-zero-copy");
        }
        if (dst.DType == ETensorDType::FP8_E4M3 || dst.DType == ETensorDType::FP8_E5M2 || dst.DType == ETensorDType::INT8) {
            if (!src.Stats) throw std::runtime_error("ModularWeightManager::gather_embeddings: FP8/INT8 gather requires Stats on the source tensor");
            quantize_with_abs_max(dst, /*scale_ptr=*/src.Stats + 1, src, /*abs_max=*/src.Stats, (long)src.nelem(), mDeviceProp, stream);
            return;
        }
        if (src.DType == ETensorDType::BF16 && dst.DType == ETensorDType::FP32) {
            convert_dtype(dst.get<float>(), reinterpret_cast<const nv_bfloat16*>(src.Data), src.nelem(), stream);
            return;
        }
        if (src.DType == ETensorDType::FP32 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), src.get<float>(), src.nelem(), stream);
            return;
        }
        throw std::runtime_error("ModularWeightManager::gather_embeddings: unsupported dtype conversion");
    };

    Tensor local_dst = (mConfig.num_shards == 1)
        ? mWorkNonBlock.embeddings
        : static_cast<Tensor>(shard_view(mWorkNonBlock.embeddings, mConfig.shard_idx, mConfig.num_shards));

    convert_into(mMasterNonBlock.embeddings, local_dst);

    if (mConfig.num_shards == 1) {
        CUDA_CHECK(cudaEventRecord(mEmbeddingsStatus.done_event, stream));
        return;
    }

    comm.begin_transaction(stream);
    TensorShard local = shard_view(mWorkNonBlock.embeddings, mConfig.shard_idx, mConfig.num_shards);
    local.Stats = mWorkNonBlock.embeddings.Stats;
    comm.schedule_all_gather(local, mWorkNonBlock.embeddings);
    comm.execute_transaction(mEmbeddingsStatus.done_event);
}

template<typename Block>
Tensor& ModularWeightManager<Block>::get_embeddings(cudaStream_t stream) {
    if (mExternalEmbeddingsProvider) {
        return mExternalEmbeddingsProvider(stream);
    }
    wait_for_buffer(mEmbeddingsStatus, stream);
    return mWorkNonBlock.embeddings;
}

template<typename Block>
void ModularWeightManager<Block>::release_embeddings(cudaStream_t stream) {
    if (mExternalEmbeddingsProvider) {
        (void)stream;
        return;
    }
    CUDA_CHECK(cudaEventRecord(mEmbeddingsStatus.done_event, stream));
    mEmbeddingsStatus.is_ready = true;
}

template<typename Block>
void ModularWeightManager<Block>::gather_final_norm(NCCLCommunicator& comm, cudaStream_t stream) {
    if (mExternalFinalNormProvider) {
        (void)comm; (void)stream;
        return;
    }
    if (mFinalNormStatus.version == mVersion) return;

    CUDA_CHECK(cudaStreamWaitEvent(stream, mFinalNormStatus.done_event));
    mFinalNormStatus.fetch_pending = true;
    mFinalNormStatus.is_ready = false;
    mFinalNormStatus.version = mVersion;

    mWorkNonBlock.final_norm_weight.Stats = mMasterNonBlock.final_norm_weight.Stats;

    auto convert_into = [&](const Tensor& src, Tensor& dst) {
        if (!src.Data || !dst.Data || src.nelem() == 0) return;
        dst.Stats = src.Stats;
        if (src.Data == dst.Data && src.DType == dst.DType) return;
        if (src.DType == dst.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDefault, stream));
            return;
        }
        if (src.Device == -1 && !mConfig.use_zero_copy) {
            throw std::runtime_error("ModularWeightManager::gather_final_norm: dtype conversion from offloaded master weights requires --use-zero-copy");
        }
        if (dst.DType == ETensorDType::FP8_E4M3 || dst.DType == ETensorDType::FP8_E5M2 || dst.DType == ETensorDType::INT8) {
            if (!src.Stats) throw std::runtime_error("ModularWeightManager::gather_final_norm: FP8/INT8 gather requires Stats on the source tensor");
            quantize_with_abs_max(dst, /*scale_ptr=*/src.Stats + 1, src, /*abs_max=*/src.Stats, (long)src.nelem(), mDeviceProp, stream);
            return;
        }
        if (src.DType == ETensorDType::BF16 && dst.DType == ETensorDType::FP32) {
            convert_dtype(dst.get<float>(), reinterpret_cast<const nv_bfloat16*>(src.Data), src.nelem(), stream);
            return;
        }
        if (src.DType == ETensorDType::FP32 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), src.get<float>(), src.nelem(), stream);
            return;
        }
        throw std::runtime_error("ModularWeightManager::gather_final_norm: unsupported dtype conversion");
    };

    Tensor local_dst = (mConfig.num_shards == 1)
        ? mWorkNonBlock.final_norm_weight
        : static_cast<Tensor>(shard_view(mWorkNonBlock.final_norm_weight, mConfig.shard_idx, mConfig.num_shards));
    convert_into(mMasterNonBlock.final_norm_weight, local_dst);

    if (mConfig.num_shards == 1) {
        CUDA_CHECK(cudaEventRecord(mFinalNormStatus.done_event, stream));
        return;
    }

    comm.begin_transaction(stream);
    TensorShard local = shard_view(mWorkNonBlock.final_norm_weight, mConfig.shard_idx, mConfig.num_shards);
    local.Stats = mWorkNonBlock.final_norm_weight.Stats;
    comm.schedule_all_gather(local, mWorkNonBlock.final_norm_weight);
    comm.execute_transaction(mFinalNormStatus.done_event);
}

template<typename Block>
Tensor& ModularWeightManager<Block>::get_final_norm(cudaStream_t stream) {
    if (mExternalFinalNormProvider) {
        return mExternalFinalNormProvider(stream);
    }
    wait_for_buffer(mFinalNormStatus, stream);
    return mWorkNonBlock.final_norm_weight;
}

template<typename Block>
void ModularWeightManager<Block>::release_final_norm(cudaStream_t stream) {
    if (mExternalFinalNormProvider) {
        (void)stream;
        return;
    }
    CUDA_CHECK(cudaEventRecord(mFinalNormStatus.done_event, stream));
    mFinalNormStatus.is_ready = true;
}

template<typename Block>
void ModularWeightManager<Block>::gather_lm_head(NCCLCommunicator& comm, cudaStream_t stream) {
    if (mExternalLMHeadProvider) {
        (void)comm; (void)stream;
        return;
    }
    if (mConfig.tied_embeddings) return;
    if (mLMHeadStatus.version == mVersion) return;

    CUDA_CHECK(cudaStreamWaitEvent(stream, mLMHeadStatus.done_event));
    mLMHeadStatus.fetch_pending = true;
    mLMHeadStatus.is_ready = false;
    mLMHeadStatus.version = mVersion;

    mWorkNonBlock.lm_head.Stats = mMasterNonBlock.lm_head.Stats;

    auto convert_into = [&](const Tensor& src, Tensor& dst) {
        if (!src.Data || !dst.Data || src.nelem() == 0) return;
        dst.Stats = src.Stats;
        if (src.Data == dst.Data && src.DType == dst.DType) return;
        if (src.DType == dst.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyDefault, stream));
            return;
        }
        if (src.Device == -1 && !mConfig.use_zero_copy) {
            throw std::runtime_error("ModularWeightManager::gather_lm_head: dtype conversion from offloaded master weights requires --use-zero-copy");
        }
        if (dst.DType == ETensorDType::FP8_E4M3 || dst.DType == ETensorDType::FP8_E5M2 || dst.DType == ETensorDType::INT8) {
            if (!src.Stats) throw std::runtime_error("ModularWeightManager::gather_lm_head: FP8/INT8 gather requires Stats on the source tensor");
            quantize_with_abs_max(dst, /*scale_ptr=*/src.Stats + 1, src, /*abs_max=*/src.Stats, (long)src.nelem(), mDeviceProp, stream);
            return;
        }
        if (src.DType == ETensorDType::BF16 && dst.DType == ETensorDType::FP32) {
            convert_dtype(dst.get<float>(), reinterpret_cast<const nv_bfloat16*>(src.Data), src.nelem(), stream);
            return;
        }
        if (src.DType == ETensorDType::FP32 && dst.DType == ETensorDType::BF16) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(dst.Data), src.get<float>(), src.nelem(), stream);
            return;
        }
        throw std::runtime_error("ModularWeightManager::gather_lm_head: unsupported dtype conversion");
    };

    Tensor local_dst = (mConfig.num_shards == 1)
        ? mWorkNonBlock.lm_head
        : static_cast<Tensor>(shard_view(mWorkNonBlock.lm_head, mConfig.shard_idx, mConfig.num_shards));
    convert_into(mMasterNonBlock.lm_head, local_dst);

    if (mConfig.num_shards == 1) {
        CUDA_CHECK(cudaEventRecord(mLMHeadStatus.done_event, stream));
        return;
    }

    comm.begin_transaction(stream);
    TensorShard local = shard_view(mWorkNonBlock.lm_head, mConfig.shard_idx, mConfig.num_shards);
    local.Stats = mWorkNonBlock.lm_head.Stats;
    comm.schedule_all_gather(local, mWorkNonBlock.lm_head);
    comm.execute_transaction(mLMHeadStatus.done_event);
}

template<typename Block>
Tensor& ModularWeightManager<Block>::get_lm_head(cudaStream_t stream) {
    if (mExternalLMHeadProvider) {
        return mExternalLMHeadProvider(stream);
    }
    if (mConfig.tied_embeddings) {
        if (mExternalEmbeddingsProvider) {
            return mExternalEmbeddingsProvider(stream);
        }
        wait_for_buffer(mEmbeddingsStatus, stream);
        return mWorkNonBlock.embeddings;
    }
    wait_for_buffer(mLMHeadStatus, stream);
    return mWorkNonBlock.lm_head;
}

template<typename Block>
void ModularWeightManager<Block>::release_lm_head(cudaStream_t stream) {
    if (mExternalLMHeadProvider || (mConfig.tied_embeddings && mExternalEmbeddingsProvider)) {
        (void)stream;
        return;
    }
    CUDA_CHECK(cudaEventRecord(mLMHeadStatus.done_event, stream));
    mLMHeadStatus.is_ready = true;
}

template<typename Block>
void ModularWeightManager<Block>::begin_optimizer(DeviceMemoryStack& memory, cudaStream_t stream) {
    (void)memory;
    mOptimizerActive = true;
    mOptimizerStream = stream;
    if (mAbsMaxes.Data) {
        fill_zero(mAbsMaxes, stream);
    }

    if (mConfig.offload_master && !mConfig.use_zero_copy) {
        auto copy_h2d = [&](const Tensor& src, Tensor& dst) {
            if (!src.Data || !dst.Data || src.nelem() == 0) return;
            dst.Stats = src.Stats;
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyHostToDevice, stream));
        };

        copy_h2d(mMasterNonBlock.embeddings, mMasterNonBlockDevice.embeddings);
        copy_h2d(mMasterNonBlock.final_norm_weight, mMasterNonBlockDevice.final_norm_weight);
        if (!mConfig.tied_embeddings) {
            copy_h2d(mMasterNonBlock.lm_head, mMasterNonBlockDevice.lm_head);
        } else {
            mMasterNonBlockDevice.lm_head = mMasterNonBlockDevice.embeddings;
        }

        // Point master shard views at the device staging buffers for the duration of the optimizer pass.
        if (mConfig.num_shards > 1) {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlockDevice.embeddings, mConfig.shard_idx, mConfig.num_shards,
                                                    std::vector<long>{mConfig.vocab_size, mConfig.hidden_size});
            mMasterFinalNormShardView = TensorShard(mMasterNonBlockDevice.final_norm_weight, mConfig.shard_idx, mConfig.num_shards,
                                                   std::vector<long>{mConfig.hidden_size});
            if (mConfig.tied_embeddings) {
                mMasterLMHeadShardView = mMasterEmbeddingsShardView;
            } else {
                mMasterLMHeadShardView = TensorShard(mMasterNonBlockDevice.lm_head, mConfig.shard_idx, mConfig.num_shards,
                                                     std::vector<long>{mConfig.vocab_size, mConfig.hidden_size});
            }
        } else {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlockDevice.embeddings);
            mMasterFinalNormShardView = TensorShard(mMasterNonBlockDevice.final_norm_weight);
            mMasterLMHeadShardView = TensorShard(mMasterNonBlockDevice.lm_head);
        }
    }
}

template<typename Block>
void ModularWeightManager<Block>::end_optimizer(DeviceMemoryStack& memory) {
    (void)memory;
    if (mConfig.offload_master && !mConfig.use_zero_copy) {
        if (!mOptimizerStream) {
            throw std::logic_error("ModularWeightManager::end_optimizer called without a prior begin_optimizer()");
        }
        auto copy_d2h = [&](Tensor& dst, const Tensor& src) {
            if (!dst.Data || !src.Data || src.nelem() == 0) return;
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, src.bytes(), cudaMemcpyDeviceToHost, mOptimizerStream));
        };

        copy_d2h(mMasterNonBlock.embeddings, mMasterNonBlockDevice.embeddings);
        copy_d2h(mMasterNonBlock.final_norm_weight, mMasterNonBlockDevice.final_norm_weight);
        if (!mConfig.tied_embeddings) {
            copy_d2h(mMasterNonBlock.lm_head, mMasterNonBlockDevice.lm_head);
        }

        // Restore master shard views to the (host) master tensors.
        if (mConfig.num_shards > 1) {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlock.embeddings, mConfig.shard_idx, mConfig.num_shards,
                                                    std::vector<long>{mConfig.vocab_size, mConfig.hidden_size});
            mMasterFinalNormShardView = TensorShard(mMasterNonBlock.final_norm_weight, mConfig.shard_idx, mConfig.num_shards,
                                                   std::vector<long>{mConfig.hidden_size});
            if (mConfig.tied_embeddings) {
                mMasterLMHeadShardView = mMasterEmbeddingsShardView;
            } else {
                mMasterLMHeadShardView = TensorShard(mMasterNonBlock.lm_head, mConfig.shard_idx, mConfig.num_shards,
                                                     std::vector<long>{mConfig.vocab_size, mConfig.hidden_size});
            }
        } else {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlock.embeddings);
            mMasterFinalNormShardView = TensorShard(mMasterNonBlock.final_norm_weight);
            mMasterLMHeadShardView = TensorShard(mMasterNonBlock.lm_head);
        }
    }

    mOptimizerActive = false;
    mOptimizerStream = nullptr;
}

template<typename Block>
void ModularWeightManager<Block>::fetch_master_block(int layer_idx, cudaStream_t fetch_stream) {
    if (!mConfig.offload_master || mConfig.use_zero_copy) return;

    int buf_idx = layer_idx % 2;
    auto& status = mMasterStatus[buf_idx];

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.done_event));

    if (status.layer_idx == layer_idx && status.version == mVersion) {
        return;
    }

    status.layer_idx = layer_idx;
    status.fetch_pending = true;
    status.is_ready = false;
    status.version = mVersion;

    auto& src = mMasterBlocks.at(layer_idx);
    auto& dst = mMasterBuffer.at(buf_idx);

    auto copy_h2d = [&](const Tensor& s, Tensor& d) {
        if (!s.Data || !d.Data || s.nelem() == 0) return;
        d.Stats = s.Stats;
        CUDA_CHECK(cudaMemcpyAsync(d.Data, s.Data, d.bytes(), cudaMemcpyHostToDevice, fetch_stream));
    };

    copy_h2d(src.ln1.weight, dst.ln1.weight);
    copy_h2d(src.attention.qkv_weight, dst.attention.qkv_weight);
    if (src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
        copy_h2d(src.attention.qkv_bias.value(), dst.attention.qkv_bias.value());
    }
    copy_h2d(src.attention.out_weight, dst.attention.out_weight);
    if (src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
        copy_h2d(src.attention.q_norm_weight.value(), dst.attention.q_norm_weight.value());
    }
    if (src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
        copy_h2d(src.attention.k_norm_weight.value(), dst.attention.k_norm_weight.value());
    }
    copy_h2d(src.ln2.weight, dst.ln2.weight);
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        copy_h2d(src.mlp_up_weight, dst.mlp_up_weight);
        copy_h2d(src.mlp_down_weight, dst.mlp_down_weight);
    }

    CUDA_CHECK(cudaEventRecord(status.done_event, fetch_stream));
}

template<typename Block>
typename Block::Weights& ModularWeightManager<Block>::get_master_block(int layer_idx, cudaStream_t stream) {
    if (!mConfig.offload_master || mConfig.use_zero_copy) {
        return mMasterBlocks.at(layer_idx);
    }

    int buf_idx = layer_idx % 2;
    auto& st = mMasterStatus.at(buf_idx);
    if (st.layer_idx != layer_idx) {
        throw std::runtime_error("Master weights not fetched: layer " + std::to_string(layer_idx));
    }
    wait_for_buffer(st, stream);
    return mMasterBuffer.at(buf_idx);
}

template<typename Block>
void ModularWeightManager<Block>::release_master_block(int layer_idx, cudaStream_t compute_stream, cudaStream_t put_stream) {
    if (!mConfig.offload_master || mConfig.use_zero_copy) return;

    int buf_idx = layer_idx % 2;
    auto& st = mMasterStatus.at(buf_idx);
    if (st.layer_idx != layer_idx) return;

    auto& src = mMasterBuffer.at(buf_idx);
    auto& dst = mMasterBlocks.at(layer_idx);

    // Ensure all update kernels that touched `src` are enqueued before starting D2H.
    CUDA_CHECK(cudaEventRecord(st.done_event, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(put_stream, st.done_event, 0));

    auto copy_d2h = [&](const Tensor& s, Tensor& d) {
        if (!s.Data || !d.Data || s.nelem() == 0) return;
        CUDA_CHECK(cudaMemcpyAsync(d.Data, s.Data, s.bytes(), cudaMemcpyDeviceToHost, put_stream));
    };

    copy_d2h(src.ln1.weight, dst.ln1.weight);
    copy_d2h(src.attention.qkv_weight, dst.attention.qkv_weight);
    if (src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
        copy_d2h(src.attention.qkv_bias.value(), dst.attention.qkv_bias.value());
    }
    copy_d2h(src.attention.out_weight, dst.attention.out_weight);
    if (src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
        copy_d2h(src.attention.q_norm_weight.value(), dst.attention.q_norm_weight.value());
    }
    if (src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
        copy_d2h(src.attention.k_norm_weight.value(), dst.attention.k_norm_weight.value());
    }
    copy_d2h(src.ln2.weight, dst.ln2.weight);
    if constexpr (has_mlp_weights<BlockWeights>::value) {
        copy_d2h(src.mlp_up_weight, dst.mlp_up_weight);
        copy_d2h(src.mlp_down_weight, dst.mlp_down_weight);
    }

    CUDA_CHECK(cudaEventRecord(st.done_event, put_stream));
    st.is_ready = true;
    st.fetch_pending = false;
    st.version = mVersion;

    // Make the compute stream wait so the caller's "optimizer done" event includes the D2H.
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, st.done_event, 0));
}

template<typename Block>
TensorShard& ModularWeightManager<Block>::get_master_embeddings() {
    return mMasterEmbeddingsShardView;
}

template<typename Block>
TensorShard& ModularWeightManager<Block>::get_master_lm_head() {
    return mMasterLMHeadShardView;
}

template<typename Block>
TensorShard& ModularWeightManager<Block>::get_master_final_norm() {
    return mMasterFinalNormShardView;
}

template<typename Block>
void ModularWeightManager<Block>::random_init(int seed, NCCLCommunicator& comm) {
    if (mConfig.offload_master && mMasterNonBlock.embeddings.Device == -1 && !mConfig.use_zero_copy) {
        throw std::runtime_error("ModularWeightManager::random_init: --offload-master requires --use-zero-copy for random initialization");
    }

    Philox4x32 rng(seed);

    float scale = 0.02f;
    float residual_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(mConfig.num_layers));

    for (int l = 0; l < mConfig.num_layers; ++l) {
        auto local_seeds = rng.generate(comm.rank(), l);
        auto& layer = mMasterBlocks[l];

        fill_constant(layer.ln1.weight, 1.f, layer.ln1.weight.nelem(), nullptr);
        fill_constant(layer.ln2.weight, 1.f, layer.ln2.weight.nelem(), nullptr);
        if (layer.attention.q_norm_weight.has_value()) {
            fill_constant(layer.attention.q_norm_weight.value(), 1.f, layer.attention.q_norm_weight->nelem(), nullptr);
        }
        if (layer.attention.k_norm_weight.has_value()) {
            fill_constant(layer.attention.k_norm_weight.value(), 1.f, layer.attention.k_norm_weight->nelem(), nullptr);
        }

        fill_normal(layer.attention.qkv_weight, layer.attention.qkv_weight.nelem(), 0.f, scale, seed, local_seeds[0], nullptr);
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            fill_normal(layer.mlp_up_weight, layer.mlp_up_weight.nelem(), 0.f, scale, seed, local_seeds[1], nullptr);
        }

        if (mConfig.block_config.use_qkv_bias && layer.attention.qkv_bias.has_value()) {
            fill_zero(layer.attention.qkv_bias.value(), nullptr);
        }

        if (mConfig.init_projections_to_zero) {
            fill_zero(layer.attention.out_weight, nullptr);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                fill_zero(layer.mlp_down_weight, nullptr);
            }
        } else {
            fill_normal(layer.attention.out_weight, layer.attention.out_weight.nelem(), 0.f, scale * residual_scale, seed, local_seeds[3], nullptr);
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                fill_normal(layer.mlp_down_weight, layer.mlp_down_weight.nelem(), 0.f, scale * residual_scale, seed, local_seeds[2], nullptr);
            }
        }
    }

    auto local_seeds = rng.generate(comm.rank(), mConfig.num_layers);
    fill_normal(mMasterNonBlock.embeddings, mMasterNonBlock.embeddings.nelem(), 0.f, scale, seed, local_seeds[0], nullptr);
    if (!mConfig.tied_embeddings && mMasterNonBlock.lm_head.Data != mMasterNonBlock.embeddings.Data) {
        fill_normal(mMasterNonBlock.lm_head, mMasterNonBlock.lm_head.nelem(), 0.f, scale, seed, local_seeds[1], nullptr);
    }
    fill_constant(mMasterNonBlock.final_norm_weight, 1.f, mMasterNonBlock.final_norm_weight.nelem(), nullptr);

    synchronize_absmax(comm);
    comm.barrier();
}

namespace {
    /**
     * @brief Load only the intersection of a global element range into a sharded destination tensor.
     *
     * The destination is treated as one shard of an equal-partitioned tensor (contiguous in element order).
     * This mirrors the legacy weight loader behavior and supports loading split HF tensors (q/k/v, gate/up)
     * into fused internal tensors even when shards straddle split boundaries.
     */
    inline void load_intersect(const TensorShard& dst, const SafeTensorEntry& src,
                               std::ptrdiff_t src_begin, std::ptrdiff_t src_end,
                               bool allow_cast) {
        if (src_begin >= src_end) return;

        std::ptrdiff_t dst_begin = dst.shard_offset();
        std::ptrdiff_t dst_end = dst_begin + static_cast<std::ptrdiff_t>(dst.nelem());

        if (src_begin >= dst_end) return;
        if (src_end <= dst_begin) return;

        std::ptrdiff_t slice_begin = src_begin < dst_begin ? dst_begin : src_begin;
        std::ptrdiff_t slice_end = src_end > dst_end ? dst_end : src_end;

        std::ptrdiff_t dst_offset = slice_begin - dst_begin;
        std::ptrdiff_t elements = slice_end - slice_begin;

        Tensor dst_slice = static_cast<const Tensor&>(dst);
        dst_slice.Sizes.fill(1);
        dst_slice.Rank = 1;
        dst_slice.Sizes[0] = elements;
        dst_slice.Data = dst.Data + dst_offset * get_dtype_size(dst.DType);

        src.read_raw(dst_slice, slice_begin - src_begin, elements, allow_cast);
    }
}

template<typename Block>
void ModularWeightManager<Block>::import_from_file(const std::string& filename, bool allow_cast, NCCLCommunicator& comm) {
    SafeTensorsReader reader{filename};

    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;

    const long q_rows = HS * HQ;
    const long kv_rows = HS * HKV;
    const long fused_rows = q_rows + 2 * kv_rows;

    // Load destination:
    // - ZeRO-3 / separate-master: load shards into mMaster*.
    // - ZeRO-1/2 view-master: load full weights into mWork* (master is a view into work).
    const bool load_sharded =
        mStreamWeights ||
        mConfig.offload_master ||
        (mConfig.master_dtype != mConfig.model_dtype) ||
        (mConfig.master_dtype != mConfig.matmul_dtype);
    const int load_idx = load_sharded ? mConfig.shard_idx : 0;
    const int load_num = load_sharded ? mConfig.num_shards : 1;

    auto& dst_nonblock = load_sharded ? mMasterNonBlock : mWorkNonBlock;
    auto& dst_blocks = load_sharded ? mMasterBlocks : mWorkBlocks;

    auto dst = [&](Tensor& t, const std::vector<long>& global_shape) -> TensorShard {
        if (load_num == 1) return TensorShard(t);
        return TensorShard(t, load_idx, load_num, global_shape);
    };

    // Build name -> destination mapping for direct matches (internal format + checkpoints).
	    std::unordered_map<std::string, TensorShard> named_tensors;
	    named_tensors.emplace("model.embed_tokens.weight", dst(dst_nonblock.embeddings, {mConfig.vocab_size, mConfig.hidden_size}));
	    named_tensors.emplace("model.norm.weight", dst(dst_nonblock.final_norm_weight, {mConfig.hidden_size}));
	    // HuggingFace models often include lm_head.weight even when tied embeddings are enabled.
	    // In that case, our lm_head tensor aliases embeddings, so loading is harmless.
	    named_tensors.emplace("lm_head.weight", dst(dst_nonblock.lm_head, {mConfig.vocab_size, mConfig.hidden_size}));

    for (int i = 0; i < mConfig.num_layers; ++i) {
        auto& block = dst_blocks.at(i);
        std::string prefix = "model.layers." + std::to_string(i);

        named_tensors.emplace(prefix + ".input_layernorm.weight", dst(block.ln1.weight, {C}));
        named_tensors.emplace(prefix + ".post_attention_layernorm.weight", dst(block.ln2.weight, {C}));
        named_tensors.emplace(prefix + ".self_attn.qkv.weight", dst(block.attention.qkv_weight, {fused_rows, C}));
        if (block.attention.qkv_bias.has_value()) {
            named_tensors.emplace(prefix + ".self_attn.qkv.bias", dst(block.attention.qkv_bias.value(), {fused_rows}));
        }
        named_tensors.emplace(prefix + ".self_attn.o_proj.weight", dst(block.attention.out_weight, {C, q_rows}));
        if (block.attention.q_norm_weight.has_value()) {
            named_tensors.emplace(prefix + ".self_attn.q_norm.weight", dst(block.attention.q_norm_weight.value(), {HS}));
        }
        if (block.attention.k_norm_weight.has_value()) {
            named_tensors.emplace(prefix + ".self_attn.k_norm.weight", dst(block.attention.k_norm_weight.value(), {HS}));
        }

        if constexpr (has_mlp_weights<BlockWeights>::value) {
            named_tensors.emplace(prefix + ".mlp.up.weight", dst(block.mlp_up_weight, {2 * D, C}));
            named_tensors.emplace(prefix + ".mlp.down_proj.weight", dst(block.mlp_down_weight, {C, D}));
        }
    }

    for (const auto& entry : reader.entries()) {
        const std::string& name = entry.name();

        if (auto found = named_tensors.find(name); found != named_tensors.end()) {
            load_intersect(found->second, entry, 0, (std::ptrdiff_t)found->second.global_nelem(), allow_cast);
            continue;
        }

        // Handle split Q/K/V and gate/up projections from HuggingFace format
        if (name.starts_with("model.layers.")) {
            std::size_t chars = 0;
            auto layer_idx = std::stoi(name.c_str() + 13, &chars);
            std::string suffix = name.substr(13 + chars);
            auto& block = mMasterBlocks.at(layer_idx);

            TensorShard qkv_w = TensorShard(block.attention.qkv_weight, mConfig.shard_idx, mConfig.num_shards,
                                            std::vector<long>{fused_rows, C});
            std::optional<TensorShard> qkv_b{};
            if (block.attention.qkv_bias.has_value()) {
                qkv_b = TensorShard(block.attention.qkv_bias.value(), mConfig.shard_idx, mConfig.num_shards,
                                    std::vector<long>{fused_rows});
            }
            TensorShard mlp_up{};
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                mlp_up = TensorShard(block.mlp_up_weight, mConfig.shard_idx, mConfig.num_shards,
                                     std::vector<long>{2 * D, C});
            }

            // Global split positions in fused tensors (in elements).
            const std::ptrdiff_t q_begin = 0;
            const std::ptrdiff_t k_begin = q_rows * C;
            const std::ptrdiff_t v_begin = (q_rows + kv_rows) * C;
            const std::ptrdiff_t q_end = q_rows * C;
            const std::ptrdiff_t k_end = (q_rows + kv_rows) * C;
            const std::ptrdiff_t v_end = fused_rows * C;

            if (suffix == ".self_attn.q_proj.weight") {
                load_intersect(qkv_w, entry, q_begin, q_end, allow_cast);
            } else if (suffix == ".self_attn.k_proj.weight") {
                load_intersect(qkv_w, entry, k_begin, k_end, allow_cast);
            } else if (suffix == ".self_attn.v_proj.weight") {
                load_intersect(qkv_w, entry, v_begin, v_end, allow_cast);
            } else if (suffix == ".self_attn.q_proj.bias") {
                if (qkv_b.has_value()) load_intersect(qkv_b.value(), entry, 0, q_rows, allow_cast);
            } else if (suffix == ".self_attn.k_proj.bias") {
                if (qkv_b.has_value()) load_intersect(qkv_b.value(), entry, q_rows, q_rows + kv_rows, allow_cast);
            } else if (suffix == ".self_attn.v_proj.bias") {
                if (qkv_b.has_value()) load_intersect(qkv_b.value(), entry, q_rows + kv_rows, fused_rows, allow_cast);
            } else if (suffix == ".mlp.up_proj.weight") {
                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    load_intersect(mlp_up, entry, 0, D * C, allow_cast);
                }
            } else if (suffix == ".mlp.gate_proj.weight") {
                if constexpr (has_mlp_weights<BlockWeights>::value) {
                    load_intersect(mlp_up, entry, D * C, 2 * D * C, allow_cast);
                }
            } else {
                // For other / MoE tensors, skip.
            }
        } else {
            throw std::runtime_error("Unexpected tensor name: " + name);
        }
    }

    synchronize_absmax(comm);
    comm.barrier();
}

template<typename Block>
void ModularWeightManager<Block>::export_to_file(const std::string& filename, NCCLCommunicator& comm) const {
    if (mStreamWeights && comm.world_size() > 1) {
        throw std::runtime_error("ModularWeightManager::export_to_file: export is not supported for ZeRO-3 streamed weights yet; use per-rank checkpoints or export with --gpus 1");
    }

    // Ensure full work weights are materialized before exporting (after the last optimizer step,
    // ranks may only have their local shard updated, and single-GPU runs may need a final master->work sync).
    if (!mStreamWeights &&
        (comm.world_size() > 1 ||
         mConfig.offload_master ||
         (mConfig.master_dtype != mConfig.model_dtype) ||
         (mConfig.master_dtype != mConfig.matmul_dtype))) {
        auto* self = const_cast<ModularWeightManager*>(this);
        cudaStream_t s = comm.stream();
        self->gather_embeddings(comm, s);
        self->gather_final_norm(comm, s);
        self->gather_lm_head(comm, s);
        for (int i = 0; i < mConfig.num_layers; ++i) {
            self->gather_block(i, comm, s);
            (void)self->get_block(i, s);
            self->release_block(i, s);
        }
        comm.wait_on_comms(s);
        comm.barrier();
    }

    SafeTensorWriter writer{filename};

    // Register non-split tensors (replicated => ShardIndex=0, NumShards=1; only rank0 writes).
    writer.register_tensor("model.embed_tokens.weight", TensorShard(mWorkNonBlock.embeddings));
    writer.register_tensor("model.norm.weight", TensorShard(mWorkNonBlock.final_norm_weight));
    if (!mConfig.tied_embeddings) {
        writer.register_tensor("lm_head.weight", TensorShard(mWorkNonBlock.lm_head));
    }

    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;

    // Register block tensors in HuggingFace format.
    for (int i = 0; i < mConfig.num_layers; ++i) {
        const auto& block = mWorkBlocks[i];
        std::string prefix = "model.layers." + std::to_string(i);

        const long q_rows = HS * HQ;
        const long kv_rows = HS * HKV;
        const long fused_rows = q_rows + 2 * kv_rows;

        writer.register_tensor(prefix + ".input_layernorm.weight", TensorShard(block.ln1.weight));
        writer.register_tensor(prefix + ".post_attention_layernorm.weight", TensorShard(block.ln2.weight));
        writer.register_tensor(prefix + ".self_attn.o_proj.weight", TensorShard(block.attention.out_weight));
        if (block.attention.q_norm_weight.has_value()) {
            writer.register_tensor(prefix + ".self_attn.q_norm.weight", TensorShard(block.attention.q_norm_weight.value()));
        }
        if (block.attention.k_norm_weight.has_value()) {
            writer.register_tensor(prefix + ".self_attn.k_norm.weight", TensorShard(block.attention.k_norm_weight.value()));
        }

        // Split QKV from fused tensor.
        writer.register_tensor(prefix + ".self_attn.q_proj.weight",
                               TensorShard(slice(block.attention.qkv_weight, 0, 0, q_rows)));
        writer.register_tensor(prefix + ".self_attn.k_proj.weight",
                               TensorShard(slice(block.attention.qkv_weight, 0, q_rows, q_rows + kv_rows)));
        writer.register_tensor(prefix + ".self_attn.v_proj.weight",
                               TensorShard(slice(block.attention.qkv_weight, 0, q_rows + kv_rows, fused_rows)));

        if (block.attention.qkv_bias.has_value()) {
            const auto& bias = block.attention.qkv_bias.value();
            writer.register_tensor(prefix + ".self_attn.q_proj.bias", TensorShard(slice(bias, 0, 0, q_rows)));
            writer.register_tensor(prefix + ".self_attn.k_proj.bias", TensorShard(slice(bias, 0, q_rows, q_rows + kv_rows)));
            writer.register_tensor(prefix + ".self_attn.v_proj.bias", TensorShard(slice(bias, 0, q_rows + kv_rows, fused_rows)));
        }

        if constexpr (has_mlp_weights<BlockWeights>::value) {
            writer.register_tensor(prefix + ".mlp.up_proj.weight", TensorShard(slice(block.mlp_up_weight, 0, 0, D)));
            writer.register_tensor(prefix + ".mlp.gate_proj.weight", TensorShard(slice(block.mlp_up_weight, 0, D, 2 * D)));
            writer.register_tensor(prefix + ".mlp.down_proj.weight", TensorShard(block.mlp_down_weight));
        }
    }

    writer.prepare_metadata(&comm);

    // Write non-block tensors
    writer.write_tensor("model.embed_tokens.weight", TensorShard(mWorkNonBlock.embeddings), &comm);
    writer.write_tensor("model.norm.weight", TensorShard(mWorkNonBlock.final_norm_weight), &comm);
    if (!mConfig.tied_embeddings) {
        writer.write_tensor("lm_head.weight", TensorShard(mWorkNonBlock.lm_head), &comm);
    }

    // Write block tensors
    for (int i = 0; i < mConfig.num_layers; ++i) {
        const auto& block = mWorkBlocks[i];
        std::string prefix = "model.layers." + std::to_string(i);

        const long q_rows = HS * HQ;
        const long kv_rows = HS * HKV;
        const long fused_rows = q_rows + 2 * kv_rows;

        writer.write_tensor(prefix + ".input_layernorm.weight", TensorShard(block.ln1.weight), &comm);
        writer.write_tensor(prefix + ".post_attention_layernorm.weight", TensorShard(block.ln2.weight), &comm);
        writer.write_tensor(prefix + ".self_attn.o_proj.weight", TensorShard(block.attention.out_weight), &comm);
        if (block.attention.q_norm_weight.has_value()) {
            writer.write_tensor(prefix + ".self_attn.q_norm.weight", TensorShard(block.attention.q_norm_weight.value()), &comm);
        }
        if (block.attention.k_norm_weight.has_value()) {
            writer.write_tensor(prefix + ".self_attn.k_norm.weight", TensorShard(block.attention.k_norm_weight.value()), &comm);
        }

        writer.write_tensor(prefix + ".self_attn.q_proj.weight",
                            TensorShard(slice(block.attention.qkv_weight, 0, 0, q_rows)), &comm);
        writer.write_tensor(prefix + ".self_attn.k_proj.weight",
                            TensorShard(slice(block.attention.qkv_weight, 0, q_rows, q_rows + kv_rows)), &comm);
        writer.write_tensor(prefix + ".self_attn.v_proj.weight",
                            TensorShard(slice(block.attention.qkv_weight, 0, q_rows + kv_rows, fused_rows)), &comm);

        if (block.attention.qkv_bias.has_value()) {
            const auto& bias = block.attention.qkv_bias.value();
            writer.write_tensor(prefix + ".self_attn.q_proj.bias", TensorShard(slice(bias, 0, 0, q_rows)), &comm);
            writer.write_tensor(prefix + ".self_attn.k_proj.bias", TensorShard(slice(bias, 0, q_rows, q_rows + kv_rows)), &comm);
            writer.write_tensor(prefix + ".self_attn.v_proj.bias", TensorShard(slice(bias, 0, q_rows + kv_rows, fused_rows)), &comm);
        }

        if constexpr (has_mlp_weights<BlockWeights>::value) {
            writer.write_tensor(prefix + ".mlp.up_proj.weight", TensorShard(slice(block.mlp_up_weight, 0, 0, D)), &comm);
            writer.write_tensor(prefix + ".mlp.gate_proj.weight", TensorShard(slice(block.mlp_up_weight, 0, D, 2 * D)), &comm);
            writer.write_tensor(prefix + ".mlp.down_proj.weight", TensorShard(block.mlp_down_weight), &comm);
        }
    }

    writer.finalize(&comm);
    comm.barrier();
}

template<typename Block>
void ModularWeightManager<Block>::synchronize_absmax(NCCLCommunicator& comm) {
    if (!mAbsMaxes.Data) return;
    if (mConfig.offload_master && mMasterNonBlock.embeddings.Device == -1 && !mConfig.use_zero_copy) {
        throw std::runtime_error("ModularWeightManager::synchronize_absmax: --offload-master requires --use-zero-copy (abs_max needs device-accessible weights)");
    }

    cudaStream_t stream = comm.stream();

    auto compute = [&](Tensor& t) {
        if (!t.Data || t.nelem() == 0 || !t.abs_max()) return;
        // abs_max() is only implemented for FP32/BF16. FP8 weights already carry scales
        // computed during quantization, so recomputation is unnecessary and indicates a caller bug.
        if (t.DType != ETensorDType::FP32 && t.DType != ETensorDType::BF16) return;
        
        abs_max(t.abs_max(), t, (long)t.nelem(), mDeviceProp, stream);
        if (comm.world_size() > 1) {
            comm.reduce_max(t.abs_max(), /*n=*/1, stream);
        }
    };

    compute(mMasterNonBlock.embeddings);
    compute(mMasterNonBlock.final_norm_weight);
    if (!mConfig.tied_embeddings) {
        compute(mMasterNonBlock.lm_head);
    }

    for (int i = 0; i < mConfig.num_layers; ++i) {
        auto& b = mMasterBlocks[i];
        compute(b.ln1.weight);
        compute(b.ln2.weight);
        compute(b.attention.qkv_weight);
        if (b.attention.qkv_bias.has_value()) {
            compute(b.attention.qkv_bias.value());
        }
        compute(b.attention.out_weight);
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            compute(b.mlp_up_weight);
            compute(b.mlp_down_weight);
        }
        comm.wait_on_comms(stream);
    }

    comm.barrier();
}

template<typename Block>
void ModularWeightManager<Block>::quantize_weights_to_fp8_cache(const BlockWeights& src, cudaStream_t stream) {
    // Skip if FP8 forward caching is not enabled
    if (!mConfig.enable_fp8_forward || !mFP8WeightCache.qkv_weight.Data) {
        return;
    }

    // Helper lambda to quantize a single weight tensor to FP8
    auto quantize_weight = [&](const Tensor& bf16_weight, Tensor& fp8_weight) {
        if (!bf16_weight.Data || !fp8_weight.Data) return;

        const long N = bf16_weight.nelem();
        if (N == 0) return;
        if (fp8_weight.DType != ETensorDType::FP8_E4M3) {
            // Cache is currently designed for E4M3 forward weights.
            return;
        }

        // QLoRA may provide weights already in FP8. In that case, don't recompute absmax on FP8
        // (kernels::abs_max does not support FP8 and scales are already present in Stats).
        if (bf16_weight.DType == ETensorDType::FP8_E4M3) {
            CUDA_CHECK(cudaMemcpyAsync(fp8_weight.Data, bf16_weight.Data, fp8_weight.bytes(), cudaMemcpyDefault, stream));
            if (bf16_weight.Stats && fp8_weight.Stats) {
                CUDA_CHECK(cudaMemcpyAsync(fp8_weight.Stats, bf16_weight.Stats, 2 * sizeof(float), cudaMemcpyDefault, stream));
            }
            return;
        }
        if (bf16_weight.DType != ETensorDType::BF16 && bf16_weight.DType != ETensorDType::FP32) {
            // Unsupported source dtype for on-the-fly quantization.
            return;
        }

        // Compute abs_max for this weight
        abs_max(fp8_weight.abs_max(), bf16_weight, N, mDeviceProp, stream);

        // Quantize to FP8 using the computed abs_max
        quantize_with_abs_max(fp8_weight, fp8_weight.scale(), bf16_weight, fp8_weight.abs_max(),
                              N, mDeviceProp, stream);
    };

    // Quantize all weight tensors to FP8 cache
    quantize_weight(src.attention.qkv_weight, mFP8WeightCache.qkv_weight);
    quantize_weight(src.attention.out_weight, mFP8WeightCache.o_weight);

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        quantize_weight(src.mlp_up_weight, mFP8WeightCache.mlp_up_weight);
        quantize_weight(src.mlp_down_weight, mFP8WeightCache.mlp_down_weight);
    }
}

template<typename Block>
void ModularWeightManager<Block>::quantize_weights_to_fp4_cache(const BlockWeights& src, cudaStream_t stream) {
    // Skip if FP4 forward caching is not enabled
    if (!mConfig.enable_fp4_forward || !mFP4WeightCache.qkv_weight.data.Data) {
        return;
    }

    const auto& cfg = mConfig.block_config;
    const int C = static_cast<int>(cfg.hidden_size);
    const int Hq = static_cast<int>(cfg.num_query_heads);
    const int Hkv = static_cast<int>(cfg.num_kv_heads);
    const int Hs = static_cast<int>(cfg.head_size);
    const int D = static_cast<int>(cfg.intermediate_size);
    const int QKV_C = (Hq + 2 * Hkv) * Hs;

    // Helper lambda to quantize a single weight tensor to FP4 (CUTLASS layout)
    // Weight shape is (N, K) where N is output channels, K is input channels
    auto quantize_fp4_weight = [&](const Tensor& bf16_weight, FP4WeightCacheEntry& fp4_cache,
                                   int N, int K) {
        if (!bf16_weight.Data || !fp4_cache.data.Data) return;

        // Only support BF16 source weights for now
        if (bf16_weight.DType != ETensorDType::BF16) {
            return;
        }

        // Use global amax buffer: 4 floats for qkv, o, mlp_up, mlp_down
        float* amax_ptr = mFP4WeightAmax.template get<float>();
        int amax_offset = 0;
        if (&fp4_cache == &mFP4WeightCache.qkv_weight) amax_offset = 0;
        else if (&fp4_cache == &mFP4WeightCache.o_weight) amax_offset = 1;
        else if constexpr (has_mlp_weights<BlockWeights>::value) {
            if (&fp4_cache == &mFP4WeightCache.mlp_up_weight) amax_offset = 2;
            else if (&fp4_cache == &mFP4WeightCache.mlp_down_weight) amax_offset = 3;
        }

        // Quantize to FP4 using CUTLASS-compatible layout
        // Output: packed FP4 data (N, K/2), block scales (FP8 E4M3), global amax
        if (mConfig.enable_four_over_six) {
            quantize_nvfp4_4o6_cutlass_auto_scale(
                fp4_cache.data.template get<uint8_t>(),
                fp4_cache.scales.template get<uint8_t>(),
                amax_ptr + amax_offset,
                bf16_weight.template get<nv_bfloat16>(),
                N, K,
                mConfig.four_over_six_metric,
                mDeviceProp, stream);
        } else {
            quantize_nvfp4_weight_cutlass_auto_scale(
                fp4_cache.data.template get<uint8_t>(),
                fp4_cache.scales.template get<uint8_t>(),
                amax_ptr + amax_offset,
                bf16_weight.template get<nv_bfloat16>(),
                N, K,
                mDeviceProp, stream);
        }
    };

    // Quantize all weight tensors to FP4 cache
    // QKV weight: (QKV_C, C) where QKV_C = (Hq + 2*Hkv) * Hs
    quantize_fp4_weight(src.attention.qkv_weight, mFP4WeightCache.qkv_weight, QKV_C, C);

    // O weight: (C, Hq*Hs)
    quantize_fp4_weight(src.attention.out_weight, mFP4WeightCache.o_weight, C, Hq * Hs);

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        // MLP up weight: (2*D, C) - gate+up fused
        quantize_fp4_weight(src.mlp_up_weight, mFP4WeightCache.mlp_up_weight, 2 * D, C);

        // MLP down weight: (C, D)
        quantize_fp4_weight(src.mlp_down_weight, mFP4WeightCache.mlp_down_weight, C, D);
    }
}

template<typename Block>
void ModularWeightManager<Block>::quantize_weights_to_fp4_cache_transposed(const BlockWeights& src, cudaStream_t stream) {
    // Skip if FP4 caching is not enabled or transposed cache buffers are not available
    if (!mConfig.enable_fp4_forward || !mFP4WeightCacheT.qkv_weight.data.Data) {
        return;
    }

    const auto& cfg = mConfig.block_config;
    const int C = static_cast<int>(cfg.hidden_size);
    const int Hq = static_cast<int>(cfg.num_query_heads);
    const int Hkv = static_cast<int>(cfg.num_kv_heads);
    const int Hs = static_cast<int>(cfg.head_size);
    const int D = static_cast<int>(cfg.intermediate_size);
    const int QKV_C = (Hq + 2 * Hkv) * Hs;

    // Helper lambda to quantize a single weight tensor to FP4 with transposed output layout.
    // Input weight shape is (N, K); output is W^T in packed FP4 (K, N/2).
    auto quantize_fp4_weight_t = [&](const Tensor& bf16_weight, FP4WeightCacheEntry& fp4_cache_t,
                                     int N, int K) {
        if (!bf16_weight.Data || !fp4_cache_t.data.Data) return;

        // Only support BF16 source weights for now
        if (bf16_weight.DType != ETensorDType::BF16) {
            return;
        }

        // Use separate amax buffer for transposed cache (4 floats: qkv, o, mlp_up, mlp_down)
        // This prevents overwriting the forward cache amax when using 4/6 quantization.
        float* amax_ptr = mFP4WeightAmaxT.template get<float>();
        int amax_offset = 0;
        if (&fp4_cache_t == &mFP4WeightCacheT.qkv_weight) amax_offset = 0;
        else if (&fp4_cache_t == &mFP4WeightCacheT.o_weight) amax_offset = 1;
        else if constexpr (has_mlp_weights<BlockWeights>::value) {
            if (&fp4_cache_t == &mFP4WeightCacheT.mlp_up_weight) amax_offset = 2;
            else if (&fp4_cache_t == &mFP4WeightCacheT.mlp_down_weight) amax_offset = 3;
        }

        // Note: No 4/6 variant for transpose quantization yet.
        // The transposed weights are used for dgrad, where dout (the A operand) uses 4/6.
        // Using standard quantization here is acceptable since the weights are relatively static.
        quantize_nvfp4_weight_cutlass_transpose_auto_scale(
            fp4_cache_t.data.template get<uint8_t>(),
            fp4_cache_t.scales.template get<uint8_t>(),
            amax_ptr + amax_offset,
            bf16_weight.template get<nv_bfloat16>(),
            N, K,
            mDeviceProp, stream);
    };

    // QKV weight: (QKV_C, C) -> transposed cache stores (C, QKV_C)
    quantize_fp4_weight_t(src.attention.qkv_weight, mFP4WeightCacheT.qkv_weight, QKV_C, C);

    // O weight: (C, Hq*Hs) -> transposed cache stores (Hq*Hs, C)
    quantize_fp4_weight_t(src.attention.out_weight, mFP4WeightCacheT.o_weight, C, Hq * Hs);

    if constexpr (has_mlp_weights<BlockWeights>::value) {
        // MLP up weight: (2*D, C) -> transposed cache stores (C, 2*D)
        quantize_fp4_weight_t(src.mlp_up_weight, mFP4WeightCacheT.mlp_up_weight, 2 * D, C);

        // MLP down weight: (C, D) -> transposed cache stores (D, C)
        quantize_fp4_weight_t(src.mlp_down_weight, mFP4WeightCacheT.mlp_down_weight, C, D);
    }
}

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHT_MANAGER_H
