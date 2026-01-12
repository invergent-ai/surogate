// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Heterogeneous Model Support
//
// Enables models with mixed layer types (e.g., Dense + MoE hybrid architectures
// like DeepSeek, Nemotron, or LFM2's attention + conv layers).
//
// Design:
// - Uses std::variant<BlockTypes...> to store heterogeneous blocks
// - Preserves zero-overhead CRTP pattern within each block type
// - Runtime dispatch via std::visit for forward/backward/optimizer
// - Per-layer configuration via LayerOverride in ModelConfig
//

#ifndef SUROGATE_SRC_MODULES_MODEL_HETEROGENEOUS_MODEL_H
#define SUROGATE_SRC_MODULES_MODEL_HETEROGENEOUS_MODEL_H

#include <variant>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

#include "modular_model_fwd.h"
#include "../model_config.h"
#include "../composite/transformer_block.h"
#include "../moe/moe_block.h"

namespace modules {

// ============================================================================
// Block Type Traits
// ============================================================================

/**
 * @brief Type trait to identify block types
 */
template<typename Block>
struct BlockTypeTraits {
    static constexpr bool is_dense = false;
    static constexpr bool is_moe = false;
};

template<typename Att, typename Act, typename Norm>
struct BlockTypeTraits<DenseTransformerBlock<Att, Act, Norm>> {
    static constexpr bool is_dense = true;
    static constexpr bool is_moe = false;
};

template<typename Att, typename Router, typename Norm>
struct BlockTypeTraits<MoETransformerBlock<Att, Router, Norm>> {
    static constexpr bool is_dense = false;
    static constexpr bool is_moe = true;
};

// ============================================================================
// Heterogeneous Block Storage
// ============================================================================

/**
 * @brief Variant-based block storage for heterogeneous models
 *
 * Wraps different block types in a variant, enabling per-layer type selection
 * while maintaining type safety and efficient storage.
 *
 * @tparam BlockTypes List of supported block types (e.g., DenseBlock, MoEBlock)
 */
template<typename... BlockTypes>
class HeterogeneousBlock {
public:
    using BlockVariant = std::variant<BlockTypes...>;
    using WeightsVariant = std::variant<typename BlockTypes::Weights...>;
    using ActivationsVariant = std::variant<typename BlockTypes::Activations...>;
    using GradientsVariant = std::variant<typename BlockTypes::Gradients...>;

    /**
     * @brief Construct from specific block type
     */
    template<typename Block>
    explicit HeterogeneousBlock(Block block)
        : mBlock(std::move(block)) {}

    /**
     * @brief Forward pass with variant dispatch
     *
     * @param ctx Module context
     * @param weights Weights variant (must match block type)
     * @param residual Input residual tensor
     * @param acts Activations variant (will be set to matching type)
     * @return Output tensor from forward pass
     */
    Tensor forward(ModuleContext& ctx, WeightsVariant& weights, Tensor& residual, ActivationsVariant& acts) {
        return std::visit([&](auto& block) -> Tensor {
            using BlockType = std::decay_t<decltype(block)>;
            using WeightsType = typename BlockType::Weights;
            using ActivationsType = typename BlockType::Activations;

            // Get or create properly typed weights and activations
            auto& w = std::get<WeightsType>(weights);

            // Initialize activations variant if needed
            if (!std::holds_alternative<ActivationsType>(acts)) {
                acts = ActivationsType{};
            }
            auto& a = std::get<ActivationsType>(acts);

            return block.forward(ctx, w, residual, a);
        }, mBlock);
    }

    /**
     * @brief Backward pass with variant dispatch
     */
    Tensor backward(ModuleContext& ctx, WeightsVariant& weights, ActivationsVariant& acts,
                    Tensor& grad_residual, GradientsVariant& grads, bool accumulate = false) {
        return std::visit([&](auto& block) -> Tensor {
            using BlockType = std::decay_t<decltype(block)>;
            using WeightsType = typename BlockType::Weights;
            using ActivationsType = typename BlockType::Activations;
            using GradientsType = typename BlockType::Gradients;

            auto& w = std::get<WeightsType>(weights);
            auto& a = std::get<ActivationsType>(acts);

            // Initialize gradients variant if needed
            if (!std::holds_alternative<GradientsType>(grads)) {
                grads = GradientsType{};
            }
            auto& g = std::get<GradientsType>(grads);

            return block.backward(ctx, w, a, grad_residual, g, accumulate);
        }, mBlock);
    }

    /**
     * @brief Recompute activations for gradient checkpointing
     */
    void recompute(ModuleContext& ctx, WeightsVariant& weights, Tensor& residual, ActivationsVariant& acts) {
        std::visit([&](auto& block) {
            using BlockType = std::decay_t<decltype(block)>;
            using WeightsType = typename BlockType::Weights;
            using ActivationsType = typename BlockType::Activations;

            auto& w = std::get<WeightsType>(weights);
            auto& a = std::get<ActivationsType>(acts);

            if constexpr (requires { block.recompute(ctx, w, residual, a); }) {
                block.recompute(ctx, w, residual, a);
            }
        }, mBlock);
    }

    /**
     * @brief Check if this block is a dense block
     */
    [[nodiscard]] bool is_dense() const {
        return std::visit([](const auto& block) {
            using BlockType = std::decay_t<decltype(block)>;
            return BlockTypeTraits<BlockType>::is_dense;
        }, mBlock);
    }

    /**
     * @brief Check if this block is an MoE block
     */
    [[nodiscard]] bool is_moe() const {
        return std::visit([](const auto& block) {
            using BlockType = std::decay_t<decltype(block)>;
            return BlockTypeTraits<BlockType>::is_moe;
        }, mBlock);
    }

    /**
     * @brief Get the block variant (for type-specific operations)
     */
    BlockVariant& get() { return mBlock; }
    const BlockVariant& get() const { return mBlock; }

    /**
     * @brief Get block index in the variant (useful for dispatch)
     */
    [[nodiscard]] size_t index() const { return mBlock.index(); }

private:
    BlockVariant mBlock;
};

// ============================================================================
// Default Block Type Combination
// ============================================================================

/**
 * @brief Default heterogeneous block with Dense and MoE variants
 */
using DefaultDenseBlock = DenseTransformerBlock<>;
using DefaultMoEBlock = StandardMoEBlock;
using DefaultHeterogeneousBlock = HeterogeneousBlock<DefaultDenseBlock, DefaultMoEBlock>;

// ============================================================================
// Heterogeneous Weight Manager
// ============================================================================

/**
 * @brief Weight manager for heterogeneous models
 *
 * Manages weights for models with mixed block types. Each layer stores
 * weights in a variant matching its block type.
 *
 * @tparam BlockTypes List of supported block types
 */
template<typename... BlockTypes>
class HeterogeneousWeightManager : public ITensorContainer {
public:
    using BlockType = HeterogeneousBlock<BlockTypes...>;
    using WeightsVariant = typename BlockType::WeightsVariant;

    struct Config {
        int num_layers;
        int hidden_size;
        int vocab_size;
        ETensorDType dtype;
        ETensorDType matmul_dtype;

        // Per-layer block type configuration
        // Index into BlockTypes... for each layer
        std::vector<size_t> layer_block_types;

        // Per-layer block configs (variant of Config types)
        std::vector<std::variant<typename BlockTypes::Config...>> layer_configs;

        // Sharding and offloading options
        bool shard_weights = false;
        bool offload_master = false;
        bool offload_quants = false;
        int rank = 0;
        int world = 1;
    };

    HeterogeneousWeightManager(const Config& config, TensorAllocator& allocator)
        : mConfig(config)
        , mAllocator(&allocator)
        , mMasterBlocks(config.num_layers)
        , mWorkBlocks(config.num_layers)
        , mLayerStatus(config.num_layers) {

        // Validate configuration
        if (config.layer_block_types.size() != static_cast<size_t>(config.num_layers)) {
            throw std::invalid_argument("layer_block_types size must match num_layers");
        }
        if (config.layer_configs.size() != static_cast<size_t>(config.num_layers)) {
            throw std::invalid_argument("layer_configs size must match num_layers");
        }

        // Allocate weights for each layer based on its block type
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_layer_weights(i);
        }

        // Allocate non-block weights (embeddings, final norm, lm_head)
        allocate_non_block_weights();
    }

    // ========================================================================
    // Weight access
    // ========================================================================

    /**
     * @brief Get weights for a specific layer
     */
    WeightsVariant& get_block(int layer_idx, cudaStream_t stream) {
        wait_for_gather(mLayerStatus[layer_idx], stream);
        return mWorkBlocks[layer_idx];
    }

    void gather_block(int layer_idx, NCCLCommunicator& comm, cudaStream_t stream) {
        // For now, weights are already on device - no gather needed
        // Future: implement weight streaming/prefetching
        mLayerStatus[layer_idx].is_ready = true;
    }

    void release_block(int layer_idx, cudaStream_t stream) {
        // For now, nothing to release
    }

    // Non-block weight access
    Tensor& get_embeddings(cudaStream_t stream) { return mEmbeddings; }
    Tensor& get_final_norm(cudaStream_t stream) { return mFinalNorm; }
    Tensor& get_lm_head(cudaStream_t stream) { return mLMHead; }

    // ========================================================================
    // ITensorContainer interface
    // ========================================================================

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        // Iterate non-block weights - use implicit TensorShard conversion
        callback("embed_tokens.weight", TensorShard(mEmbeddings));
        callback("norm.weight", TensorShard(mFinalNorm));
        callback("lm_head.weight", TensorShard(mLMHead));

        // Iterate per-layer weights
        for (int i = 0; i < mConfig.num_layers; ++i) {
            std::string prefix = fmt::format("layers.{}.", i);
            iterate_layer_weights(i, prefix, callback);
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int num_layers() const { return mConfig.num_layers; }

    /**
     * @brief Get block type index for a layer
     */
    [[nodiscard]] size_t get_layer_block_type(int layer_idx) const {
        return mConfig.layer_block_types[layer_idx];
    }

private:
    Config mConfig;
    TensorAllocator* mAllocator;

    // Per-layer weights (variant-based)
    std::vector<WeightsVariant> mMasterBlocks;
    std::vector<WeightsVariant> mWorkBlocks;
    std::vector<GatherStatus> mLayerStatus;

    // Non-block weights
    Tensor mEmbeddings;
    Tensor mFinalNorm;
    Tensor mLMHead;

    void wait_for_gather(GatherStatus& status, cudaStream_t stream) {
        // Wait for any pending fetch to complete
        if (status.fetch_pending && status.done_event) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, status.done_event, 0));
            status.fetch_pending = false;
        }
    }

    void allocate_layer_weights(int layer_idx) {
        const size_t block_type = mConfig.layer_block_types[layer_idx];
        const auto& layer_config = mConfig.layer_configs[layer_idx];

        // Use fold expression to dispatch to correct allocator
        allocate_layer_weights_impl<0, BlockTypes...>(layer_idx, block_type, layer_config);
    }

    template<size_t I, typename Block, typename... Rest>
    void allocate_layer_weights_impl(int layer_idx, size_t block_type,
                                     const std::variant<typename BlockTypes::Config...>& layer_config) {
        if (block_type == I) {
            using WeightsType = typename Block::Weights;
            using ConfigType = typename Block::Config;

            const auto& cfg = std::get<ConfigType>(layer_config);
            WeightsType weights;

            // Allocate weight tensors based on block type
            allocate_block_weights<Block>(weights, cfg);

            mMasterBlocks[layer_idx] = weights;
            mWorkBlocks[layer_idx] = weights;  // For now, share storage
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            allocate_layer_weights_impl<I + 1, Rest...>(layer_idx, block_type, layer_config);
        } else {
            throw std::invalid_argument("Invalid block type index");
        }
    }

    template<typename Block>
    void allocate_block_weights(typename Block::Weights& weights, const typename Block::Config& cfg) {
        // This is a placeholder - actual allocation depends on block type
        // Will be specialized for each supported block type
    }

    void allocate_non_block_weights() {
        const auto dtype = mConfig.dtype;
        const int C = mConfig.hidden_size;
        const int V = mConfig.vocab_size;

        mEmbeddings = mAllocator->allocate(dtype, "embed_tokens.weight", {V, C});
        mFinalNorm = mAllocator->allocate(dtype, "norm.weight", {C});
        mLMHead = mAllocator->allocate(dtype, "lm_head.weight", {V, C});
    }

    void iterate_layer_weights(int layer_idx, const std::string& prefix,
                               const std::function<void(std::string, const TensorShard&)>& callback) {
        // Dispatch based on block type
        std::visit([&](auto& weights) {
            using WeightsType = std::decay_t<decltype(weights)>;
            iterate_typed_weights<WeightsType>(weights, prefix, callback);
        }, mWorkBlocks[layer_idx]);
    }

    template<typename WeightsType>
    void iterate_typed_weights(WeightsType& weights, const std::string& prefix,
                               const std::function<void(std::string, const TensorShard&)>& callback) {
        // Specialized per block type - will be implemented in specializations
    }
};

// ============================================================================
// Heterogeneous Run State
// ============================================================================

/**
 * @brief Run state for heterogeneous models
 *
 * Manages activation storage for models with mixed block types.
 * Each layer stores activations in a variant matching its block type.
 *
 * @tparam BlockTypes List of supported block types
 */
template<typename... BlockTypes>
class HeterogeneousRunState : public IRunState {
public:
    using BlockType = HeterogeneousBlock<BlockTypes...>;
    using ActivationsVariant = typename BlockType::ActivationsVariant;
    using GradientsVariant = typename BlockType::GradientsVariant;

    struct Config {
        int num_layers;
        int batch_size;
        int seq_length;
        int hidden_size;
        int vocab_size;
        ETensorDType activation_dtype;
        ETensorDType grad_dtype;

        // Per-layer block type (index into BlockTypes...)
        std::vector<size_t> layer_block_types;

        // Per-layer block configs
        std::vector<std::variant<typename BlockTypes::Config...>> layer_configs;

        // Recomputation and offloading options
        bool recompute_block = false;
        bool offload_residuals = false;

        // For IRunState base class
        PretrainedConfig pretrained_config;
    };

    HeterogeneousRunState(const Config& config, TensorAllocator& allocator)
        : IRunState()  // Use default constructor - will set up base later
        , mConfig(config)
        , mAllocator(&allocator)
        , mLayerActivations(config.num_layers)
        , mLayerGradients(config.num_layers) {

        // Allocate per-layer activations/gradients based on block types
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_layer_storage(i);
        }

        // Allocate shared buffers
        allocate_shared_buffers();
    }

    // ========================================================================
    // Activation/gradient access
    // ========================================================================

    ActivationsVariant& get_activations(int layer_idx) {
        return mLayerActivations[layer_idx];
    }

    GradientsVariant& get_gradients(int layer_idx) {
        return mLayerGradients[layer_idx];
    }

    /**
     * @brief Get the residual stream tensor
     */
    Tensor& residual() { return mResidual; }

    /**
     * @brief Get gradient residual tensor
     */
    Tensor& d_residual() { return mDResidual; }

    // ========================================================================
    // Shared state access
    // ========================================================================

    Tensor& logits() { return mLogits; }
    Tensor& loss() { return mLoss; }
    Tensor& d_logits() { return mDLogits; }

    // ========================================================================
    // ModuleContext creation
    // ========================================================================

    ModuleContext create_context(int B, int T) {
        ModuleContext ctx{};
        ctx.B = B;
        ctx.T = T;
        ctx.stream = 0;  // Default stream - caller can override
        ctx.workspace = nullptr;  // TODO: Allocate workspace tensor
        ctx.cublas_handle = nullptr;  // TODO: Get from runtime
        ctx.cudnn_handle = nullptr;  // TODO: Get from runtime
        ctx.device_prop = nullptr;  // TODO: Get from runtime
        ctx.position_ids = nullptr;
        ctx.matmul_dtype = ETensorDType::BF16;
        ctx.use_quantization = false;
        return ctx;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int num_layers() const { return mConfig.num_layers; }

private:
    Config mConfig;
    TensorAllocator* mAllocator;

    // Per-layer activation/gradient storage
    std::vector<ActivationsVariant> mLayerActivations;
    std::vector<GradientsVariant> mLayerGradients;

    // Shared tensors
    Tensor mResidual;      // (B, T, C) - main residual stream
    Tensor mDResidual;     // (B, T, C) - gradient residual
    Tensor mLogits;        // (B, T, V) - output logits
    Tensor mLoss;          // (1,) - scalar loss
    Tensor mDLogits;       // (B, T, V) - logits gradient

    // Workspace for matmuls etc.
    Tensor mWorkspace;

    void allocate_layer_storage(int layer_idx) {
        const size_t block_type = mConfig.layer_block_types[layer_idx];
        const auto& layer_config = mConfig.layer_configs[layer_idx];

        // Initialize variant with correct type
        allocate_layer_storage_impl<0, BlockTypes...>(layer_idx, block_type, layer_config);
    }

    template<size_t I, typename Block, typename... Rest>
    void allocate_layer_storage_impl(int layer_idx, size_t block_type,
                                      const std::variant<typename BlockTypes::Config...>& layer_config) {
        if (block_type == I) {
            using ActivationsType = typename Block::Activations;
            using GradientsType = typename Block::Gradients;

            mLayerActivations[layer_idx] = ActivationsType{};
            mLayerGradients[layer_idx] = GradientsType{};
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            allocate_layer_storage_impl<I + 1, Rest...>(layer_idx, block_type, layer_config);
        }
    }

    void allocate_shared_buffers() {
        const int B = mConfig.batch_size;
        const int T = mConfig.seq_length;
        const int C = mConfig.hidden_size;
        const int V = mConfig.vocab_size;
        const auto act_dtype = mConfig.activation_dtype;
        const auto grad_dtype = mConfig.grad_dtype;

        mResidual = mAllocator->allocate(act_dtype, "residual", {B, T, C});
        mDResidual = mAllocator->allocate(grad_dtype, "d_residual", {B, T, C});
        mLogits = mAllocator->allocate(act_dtype, "logits", {B, T, V});
        mLoss = mAllocator->allocate(ETensorDType::FP32, "loss", {1});
        mDLogits = mAllocator->allocate(grad_dtype, "d_logits", {B, T, V});
    }
};

// ============================================================================
// Heterogeneous Transformer Model
// ============================================================================

/**
 * @brief Heterogeneous transformer model with mixed block types
 *
 * A transformer model that supports different block types per layer,
 * enabling architectures like:
 * - DeepSeek/Nemotron: some layers dense, some MoE
 * - LFM2: some layers attention, some convolution
 *
 * Uses std::variant for type-safe heterogeneous storage while preserving
 * efficient dispatch within each block type.
 *
 * @tparam BlockTypes Variadic list of supported block types
 */
template<typename... BlockTypes>
class HeterogeneousTransformerModel : public IModel {
public:
    using Block = HeterogeneousBlock<BlockTypes...>;
    using WeightManager = HeterogeneousWeightManager<BlockTypes...>;
    using RunState = HeterogeneousRunState<BlockTypes...>;

    using WeightsVariant = typename Block::WeightsVariant;
    using ActivationsVariant = typename Block::ActivationsVariant;
    using GradientsVariant = typename Block::GradientsVariant;

    /**
     * @brief Backward hook callback type
     */
    using BackwardBlockHook = std::function<void(int layer_idx, bool accumulate,
                                                  cudaStream_t stream, BackwardHookPoint point, void* context)>;

    /**
     * @brief Forward hook callback type (matches modular model signature)
     */
    using ForwardBlockHook = std::function<void(int layer_idx, cudaStream_t stream,
                                                 ForwardHookPoint point, void* context)>;

    // Forward hook points for block-level hooks (not projection-level)
    static constexpr ForwardHookPoint kPreBlockHook = ForwardHookPoint::AfterQKVProjection;
    static constexpr ForwardHookPoint kPostBlockHook = ForwardHookPoint::AfterMLPDownProjection;

    /**
     * @brief Construct heterogeneous model from configuration
     */
    HeterogeneousTransformerModel(const ModelConfig& config, const ModelOptions& options,
                                   int rank, int world,
                                   const std::shared_ptr<TensorAllocator>& alloc = nullptr)
        : mConfig(config)
        , mOptions(options)
        , mAllocator(alloc ? alloc : std::make_shared<TensorAllocator>())
        , mRank(rank)
        , mWorld(world) {

        // Validate configuration
        if (config.architecture != ArchitectureType::Hybrid) {
            throw std::invalid_argument("HeterogeneousTransformerModel requires Hybrid architecture");
        }

        // Build per-layer block configuration
        buildLayerConfigs();

        // Create blocks
        for (int i = 0; i < config.NumLayers; ++i) {
            mBlocks.push_back(createBlock(i));
        }
    }

    ~HeterogeneousTransformerModel() override = default;

    // ========================================================================
    // IModel interface implementation
    // ========================================================================

    void init_weights(NCCLCommunicator& comm) override {
        if (mWeights) {
            // Random initialization
            // TODO: Implement proper weight initialization
        }
    }

    void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) override {
        // TODO: Implement weight loading
        throw std::runtime_error("Weight import not yet implemented for heterogeneous models");
    }

    void export_weights(const std::string& file_name, NCCLCommunicator& comm) override {
        // TODO: Implement weight saving
        throw std::runtime_error("Weight export not yet implemented for heterogeneous models");
    }

    void on_restore_checkpoint(NCCLCommunicator& comm) override {
        // TODO: Implement checkpoint restoration
    }

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override {
        forward_impl(inputs, position_ids, comm, micro_step, nullptr);
    }

    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override {
        forward(inputs, position_ids, comm, micro_step);
        // TODO: Compute loss
        return 0.0f;
    }

    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override {
        backward_impl(inputs, targets, comm, grad_accum_steps, micro_step, nullptr);
    }

    void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t,
                float epsilon, float weight_decay, float grad_clip) override {
        // TODO: Implement optimizer update
        throw std::runtime_error("Optimizer update not yet implemented for heterogeneous models");
    }

    void update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) override {
        // TODO: Implement optimizer update with config
        throw std::runtime_error("Optimizer update not yet implemented for heterogeneous models");
    }

    float get_loss() const override {
        return mLoss;
    }

    ITensorContainer& weights() override {
        return *mWeights;
    }

    ITensorContainer& opt_momentum() override {
        static EmptyTensorContainer empty;
        return empty;
    }

    ITensorContainer& opt_momentum_scales() override {
        static EmptyTensorContainer empty;
        return empty;
    }

    ITensorContainer& opt_variance() override {
        static EmptyTensorContainer empty;
        return empty;
    }

    ITensorContainer& opt_variance_scales() override {
        static EmptyTensorContainer empty;
        return empty;
    }

    std::vector<std::byte> rng_state() const override {
        return {};
    }

    void set_rng_state(const std::vector<std::byte>& state) override {
        // TODO: Implement RNG state management
    }

    std::string_view model_type() const override {
        return "heterogeneous";
    }

    IRunState& get_run_state() const override {
        return *mRunState;
    }

    void allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm,
                            int B, int T, bool allocate_optimizer = true) override {
        // Build run state config
        typename RunState::Config rs_config;
        rs_config.num_layers = mConfig.NumLayers;
        rs_config.batch_size = B;
        rs_config.seq_length = T;
        rs_config.hidden_size = mConfig.HiddenSize;
        rs_config.vocab_size = mConfig.VocabSize;
        rs_config.activation_dtype = mOptions.model_dtype.value_or(ETensorDType::BF16);
        rs_config.grad_dtype = mOptions.gradient_dtype.value_or(ETensorDType::BF16);
        rs_config.layer_block_types = mLayerBlockTypes;
        rs_config.layer_configs = mLayerConfigs;
        rs_config.recompute_block = mOptions.recompute_block;
        rs_config.offload_residuals = mOptions.offload_residuals;
        rs_config.pretrained_config = static_cast<const PretrainedConfig&>(mConfig);

        mRunState = std::make_unique<RunState>(rs_config, *mAllocator);

        // Build weight manager config
        typename WeightManager::Config wm_config;
        wm_config.num_layers = mConfig.NumLayers;
        wm_config.hidden_size = mConfig.HiddenSize;
        wm_config.vocab_size = mConfig.VocabSize;
        wm_config.dtype = mOptions.model_dtype.value_or(ETensorDType::BF16);
        wm_config.matmul_dtype = mOptions.matmul_dtype.value_or(ETensorDType::BF16);
        wm_config.layer_block_types = mLayerBlockTypes;
        wm_config.layer_configs = mLayerConfigs;
        wm_config.shard_weights = mOptions.shard_weights;
        wm_config.offload_master = mOptions.offload_master;
        wm_config.offload_quants = mOptions.offload_quants;
        wm_config.rank = mRank;
        wm_config.world = mWorld;

        mWeights = std::make_unique<WeightManager>(wm_config, *mAllocator);
    }

    // ========================================================================
    // Extended interface
    // ========================================================================

    /**
     * @brief Forward pass with per-layer hook callback
     */
    void forward_with_hook(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step,
                           const ForwardBlockHook& hook) {
        forward_impl(inputs, position_ids, comm, micro_step, &hook);
    }

    /**
     * @brief Backward pass with per-layer hook callback
     */
    void backward_with_hook(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                            int grad_accum_steps, int micro_step, const BackwardBlockHook& hook) {
        backward_impl(inputs, targets, comm, grad_accum_steps, micro_step, &hook);
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const ModelConfig& config() const { return mConfig; }
    [[nodiscard]] const ModelOptions& options() const { return mOptions; }
    [[nodiscard]] int num_layers() const { return mConfig.NumLayers; }

    /**
     * @brief Check if a specific layer is MoE
     */
    [[nodiscard]] bool is_layer_moe(int layer_idx) const {
        return mBlocks[layer_idx].is_moe();
    }

    /**
     * @brief Check if a specific layer is dense
     */
    [[nodiscard]] bool is_layer_dense(int layer_idx) const {
        return mBlocks[layer_idx].is_dense();
    }

protected:
    void forward_impl(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm,
                      int micro_step, const ForwardBlockHook* hook) {
        if (!mRunState || !mWeights) {
            throw std::runtime_error("Run state not allocated - call allocate_run_state first");
        }

        const int B = inputs.Sizes[0];
        const int T = inputs.Sizes[1];
        auto ctx = mRunState->create_context(B, T);

        // Get embedding
        Tensor& embeddings = mWeights->get_embeddings(ctx.stream);
        Tensor& residual = mRunState->residual();

        // Embed input tokens
        // TODO: embedding_lookup(residual, embeddings, inputs, ctx.stream);

        // Forward through all layers
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            if (hook) {
                (*hook)(i, ctx.stream, kPreBlockHook, nullptr);
            }

            auto& weights = mWeights->get_block(i, ctx.stream);
            auto& acts = mRunState->get_activations(i);

            Tensor block_output = mBlocks[i].forward(ctx, weights, residual, acts);

            // Update residual (fused in actual implementation)
            // residual = residual + block_output

            if (hook) {
                (*hook)(i, ctx.stream, kPostBlockHook, nullptr);
            }

            mWeights->release_block(i, ctx.stream);
        }

        // Final norm and LM head
        // TODO: Implement final processing
    }

    void backward_impl(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                       int grad_accum_steps, int micro_step, const BackwardBlockHook* hook) {
        if (!mRunState || !mWeights) {
            throw std::runtime_error("Run state not allocated");
        }

        const int B = inputs.Sizes[0];
        const int T = inputs.Sizes[1];
        auto ctx = mRunState->create_context(B, T);
        const bool accumulate = (micro_step > 0);

        Tensor& d_residual = mRunState->d_residual();

        // Backward through LM head and final norm
        // TODO: Implement

        // Backward through all layers in reverse
        for (int i = mConfig.NumLayers - 1; i >= 0; --i) {
            if (hook) {
                (*hook)(i, accumulate, ctx.stream, BackwardHookPoint::BeforeLayerBackward, nullptr);
            }

            auto& weights = mWeights->get_block(i, ctx.stream);
            auto& acts = mRunState->get_activations(i);
            auto& grads = mRunState->get_gradients(i);

            // Recompute if needed
            if (mOptions.recompute_block) {
                Tensor& residual = mRunState->residual();
                mBlocks[i].recompute(ctx, weights, residual, acts);
            }

            Tensor d_input = mBlocks[i].backward(ctx, weights, acts, d_residual, grads, accumulate);

            // d_residual = d_input for next iteration
            // (handled by fused operations in actual implementation)

            if (hook) {
                (*hook)(i, accumulate, ctx.stream, BackwardHookPoint::AfterLayerBackward, nullptr);
            }

            mWeights->release_block(i, ctx.stream);
        }
    }

private:
    ModelConfig mConfig;
    ModelOptions mOptions;
    std::shared_ptr<TensorAllocator> mAllocator;
    int mRank;
    int mWorld;

    // Per-layer block type indices
    std::vector<size_t> mLayerBlockTypes;

    // Per-layer block configs (variant)
    std::vector<std::variant<typename BlockTypes::Config...>> mLayerConfigs;

    // Blocks
    std::vector<Block> mBlocks;

    // State managers
    std::unique_ptr<WeightManager> mWeights;
    std::unique_ptr<RunState> mRunState;

    // Loss value
    float mLoss = 0.0f;

    /**
     * @brief Build per-layer configuration from ModelConfig
     *
     * Uses compile-time dispatch to create the correct config type
     * for each block type in the template parameter pack.
     */
    void buildLayerConfigs() {
        mLayerBlockTypes.resize(mConfig.NumLayers);
        mLayerConfigs.resize(mConfig.NumLayers);

        // Default all layers to dense (index 0)
        std::fill(mLayerBlockTypes.begin(), mLayerBlockTypes.end(), 0);

        // Apply layer overrides
        for (const auto& override : mConfig.layer_overrides) {
            if (override.layer_idx < 0 || override.layer_idx >= mConfig.NumLayers) {
                continue;
            }

            if (override.is_moe || override.block_type == BlockType::MoE ||
                override.block_type == BlockType::SwitchMoE) {
                // MoE block (index 1)
                mLayerBlockTypes[override.layer_idx] = 1;
            }
        }

        // Build configs for each layer using proper template dispatch
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            buildLayerConfigImpl<0, BlockTypes...>(i, mLayerBlockTypes[i]);
        }
    }

    // Helper to build config for a specific block type index
    template<size_t I, typename BlockT, typename... Rest>
    void buildLayerConfigImpl(int layer_idx, size_t block_type_idx) {
        if (block_type_idx == I) {
            using ConfigType = typename BlockT::Config;
            ConfigType cfg = buildBlockConfig<BlockT>(layer_idx);
            mLayerConfigs[layer_idx] = cfg;
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            buildLayerConfigImpl<I + 1, Rest...>(layer_idx, block_type_idx);
        }
    }

    // Build a dense block config
    template<typename BlockT>
    typename std::enable_if<BlockTypeTraits<BlockT>::is_dense, typename BlockT::Config>::type
    buildBlockConfig(int layer_idx) {
        typename BlockT::Config cfg;
        cfg.hidden_size = mConfig.HiddenSize;
        cfg.num_query_heads = mConfig.NumQueryHeads;
        cfg.num_kv_heads = mConfig.NumKeyValHeads;
        cfg.head_size = mConfig.HiddenSize / mConfig.NumQueryHeads;
        cfg.rope = mConfig.Rope;
        cfg.max_seq_len = mConfig.MaxPositionEmbeddings;
        cfg.use_qkv_bias = mConfig.UseQKVBias;
        cfg.intermediate_size = mConfig.get_intermediate_size(layer_idx);
        cfg.rms_norm_eps = mConfig.RmsNormEps;
        return cfg;
    }

    // Build an MoE block config
    template<typename BlockT>
    typename std::enable_if<BlockTypeTraits<BlockT>::is_moe, typename BlockT::Config>::type
    buildBlockConfig(int layer_idx) {
        typename BlockT::Config cfg;
        cfg.hidden_size = mConfig.HiddenSize;
        cfg.num_query_heads = mConfig.NumQueryHeads;
        cfg.num_kv_heads = mConfig.NumKeyValHeads;
        cfg.head_size = mConfig.HiddenSize / mConfig.NumQueryHeads;
        cfg.rope = mConfig.Rope;
        cfg.max_seq_len = mConfig.MaxPositionEmbeddings;
        cfg.use_qkv_bias = mConfig.UseQKVBias;
        cfg.rms_norm_eps = mConfig.RmsNormEps;

        // MoE-specific config
        if (mConfig.moe_config.has_value()) {
            const auto& moe = mConfig.moe_config.value();
            cfg.num_experts = mConfig.get_num_experts(layer_idx);
            cfg.top_k = mConfig.get_top_k(layer_idx);
            cfg.aux_loss_coef = moe.router_aux_loss_coef;
            cfg.capacity_factor = moe.capacity_factor;
            cfg.use_shared_expert = moe.use_shared_expert;
            cfg.shared_expert_intermediate = moe.shared_expert_size;
        }
        cfg.intermediate_size = mConfig.get_intermediate_size(layer_idx);
        return cfg;
    }

    /**
     * @brief Create a block for a specific layer
     */
    Block createBlock(int layer_idx) {
        const size_t block_type = mLayerBlockTypes[layer_idx];
        const auto& layer_config = mLayerConfigs[layer_idx];

        return createBlockImpl<0, BlockTypes...>(block_type, layer_config);
    }

    template<size_t I, typename BlockT, typename... Rest>
    Block createBlockImpl(size_t block_type,
                          const std::variant<typename BlockTypes::Config...>& layer_config) {
        if (block_type == I) {
            using ConfigType = typename BlockT::Config;
            const auto& cfg = std::get<ConfigType>(layer_config);
            return Block(BlockT(cfg));
        }

        if constexpr (sizeof...(Rest) > 0) {
            return createBlockImpl<I + 1, Rest...>(block_type, layer_config);
        } else {
            throw std::invalid_argument("Invalid block type index");
        }
    }
};

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

/**
 * @brief Default heterogeneous model with Dense + MoE blocks
 */
using DefaultHeterogeneousModel = HeterogeneousTransformerModel<DefaultDenseBlock, DefaultMoEBlock>;

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODEL_HETEROGENEOUS_MODEL_H
