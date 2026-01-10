// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MODEL_MODULAR_MODEL_H
#define SUROGATE_SRC_MODULES_MODEL_MODULAR_MODEL_H

#include "modular_model_fwd.h"

namespace modules {

/**
 * @brief Modular transformer model template
 *
 * A flexible transformer model that can be composed from different block types.
 * Implements the IModel interface for compatibility with the training infrastructure.
 *
 * @tparam Block The transformer block type (e.g., DenseTransformerBlock)
 */
template<typename Block>
class ModularTransformerModel : public IModel {
public:
    using BlockConfig = typename Block::Config;
    using BlockWeights = typename Block::Weights;
    using BlockActivations = typename Block::Activations;
    using BlockGradients = typename Block::Gradients;


    /**
     * @brief Backward hook callback type
     */
    using BackwardBlockHook = std::function<void(int layer_idx, bool accumulate, cudaStream_t stream, BackwardHookPoint point)>;

    /**
     * @brief Forward hook callback type
     *
     * Hook implementations may modify intermediate activations in-place (e.g., add LoRA deltas).
     */
    using ForwardBlockHook = std::function<void(int layer_idx, cudaStream_t stream, ForwardHookPoint point)>;

    /**
     * @brief Construct a modular transformer model
     *
     * @param config Model configuration
     * @param options Runtime options
     * @param rank Current process rank for sharding
     * @param world World size (number of processes)
     * @param alloc Optional tensor allocator
     */
    ModularTransformerModel(const ModelConfig& config, const ModelOptions& options,
                            int rank, int world,
                            const std::shared_ptr<TensorAllocator>& alloc = nullptr);
    ~ModularTransformerModel() override;

    // ========================================================================
    // IModel interface implementation
    // ========================================================================

    void init_weights(NCCLCommunicator& comm) override;
    void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) override;
    void export_weights(const std::string& file_name, NCCLCommunicator& comm) override;
    void on_restore_checkpoint(NCCLCommunicator& comm) override;

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;
    void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t,
                float epsilon, float weight_decay, float grad_clip) override;

    /**
     * @brief Optimizer update with full configuration (supports AdamW and NorMuon)
     *
     * Dispatches to the appropriate optimizer based on config.type.
     */
    void update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) override;

    /**
     * @brief 8-bit AdamW optimizer update (internal implementation)
     */
    void update_adamw_8bit(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t,
                           float epsilon, float weight_decay, float grad_clip);

    /**
     * @brief NorMuon optimizer update (hybrid AdamW/NorMuon)
     *
     * Uses NorMuon for 2D weight matrices (attention projections, MLP weights)
     * and AdamW 8-bit for embeddings, norms, lm_head, and 0D/1D parameters.
     *
     * @param config Optimizer configuration with NorMuon hyperparameters
     * @param t Optimizer step number (for AdamW bias correction)
     */
    void update_normuon(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int t);

    /**
     * @brief Eagerly initialize optimizer state (for CUDA graph capture)
     *
     * Call this after allocate_run_state() to pre-allocate optimizer state tensors.
     * This moves allocation from the first update() call to an explicit init phase,
     * enabling CUDA graph capture of the optimizer step.
     */
    void initialize_optimizer_state(cudaStream_t stream);

    float get_loss() const override;
    ITensorContainer& weights() override;
    ITensorContainer& opt_momentum() override;
    ITensorContainer& opt_momentum_scales() override;
    ITensorContainer& opt_variance() override;
    ITensorContainer& opt_variance_scales() override;
    std::vector<std::byte> rng_state() const override;
    void set_rng_state(const std::vector<std::byte>& state) override;
    std::string_view model_type() const override;
    IRunState& get_run_state() const override;

    // ========================================================================
    // Extended interface
    // ========================================================================

    /**
     * @brief Backward pass with per-layer hook callback
     *
     * Allows external modules (like LoRA) to inject gradient computation
     * at specific points during the backward pass.
     */
    void backward_with_hook(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                            int grad_accum_steps, int micro_step,
                            const BackwardBlockHook& hook);

    /**
     * @brief Forward pass with per-layer hook callback
     */
    void forward_with_hook(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step,
                           const ForwardBlockHook& hook);

    /**
     * @brief Validate with forward hook
     */
    float validate_with_hook(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step,
                             const ForwardBlockHook& hook);

    /**
     * @brief Allocate run state for given batch/sequence sizes
     */
    void allocate_run_state(const ModelOptions& options, NCCLCommunicator& comm,
                            int B, int T, bool allocate_optimizer = true);

    /**
     * @brief Allocate run state - IModel interface override
     *
     * Converts RuntimeOptions to ModelOptions and calls the internal version.
     */
    void allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm,
                            int B, int T, bool allocate_optimizer = true) override {
        // Set the recipe from RuntimeOptions if provided
        if (options.TrainingRecipe) {
            mRecipe = options.TrainingRecipe;
        }
        allocate_run_state(ModelOptions::from_runtime_options(options), comm, B, T, allocate_optimizer);
    }

    /**
     * @brief Check if run state is allocated
     */
    [[nodiscard]] bool has_run_state() const { return mRunState != nullptr; }

    /**
     * @brief Calculate global gradient norm
     */
    void calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip);

    // ========================================================================
    // Full-step CUDA graph capture
    // ========================================================================

    /**
     * @brief Execute a complete training step, optionally using CUDA graph capture
     *
     * This method captures the entire training step (forward + backward + optimizer)
     * into a single CUDA graph for reduced launch overhead. The graph is captured
     * on the first call and replayed on subsequent calls.
     *
     * Requirements:
     * - Optimizer state must be pre-initialized via initialize_optimizer_state()
     * - Batch size and sequence length must remain constant
     * - Data must be pre-loaded into input/target buffers before calling
     *
     * @param comm NCCL communicator
     * @param loader Data loader (used to load batches into model buffers)
     * @param grad_accum_steps Number of gradient accumulation steps
     * @param learning_rate Current learning rate
     * @param beta_1 AdamW beta1
     * @param beta_2 AdamW beta2
     * @param step Current optimizer step (1-indexed)
     * @param epsilon AdamW epsilon
     * @param weight_decay Weight decay coefficient
     * @param grad_clip Gradient clipping threshold
     * @param use_graph If true, capture/replay as CUDA graph; if false, run normally
     */
    void train_step_graphed(NCCLCommunicator& comm, DataLoader& loader,
                            int grad_accum_steps,
                            float learning_rate, float beta_1, float beta_2, int step,
                            float epsilon, float weight_decay, float grad_clip,
                            bool use_graph);

    /**
     * @brief Reset the captured full-step graph
     *
     * Call this if batch size, sequence length, or model configuration changes.
     */
    void reset_full_step_graph();

    /**
     * @brief Check if full-step graph has been captured
     */
    [[nodiscard]] bool has_full_step_graph() const {
        return mFullStepGraph && mFullStepGraph->captured;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const ModelConfig& config() const { return mConfig; }
    [[nodiscard]] const ModelOptions& options() const { return mOptions; }

    ModularWeightManager<Block>& weights_manager() { return *mWeights; }
    const ModularWeightManager<Block>& weights_manager() const { return *mWeights; }
    ModularGradientManager<Block>& grads() { return *mGrads; }
    ModularRunState<Block>& run_state() { return *mRunState; }
    const ModularRunState<Block>& run_state() const { return *mRunState; }

    /**
     * @brief Set the training recipe for matmul dispatch
     *
     * The recipe controls quantization format and matmul implementation.
     * If not set, the model uses legacy conditional dispatch based on options.
     */
    void set_recipe(std::shared_ptr<recipes::Recipe> recipe) { mRecipe = std::move(recipe); }

    /**
     * @brief Get the current training recipe
     */
    [[nodiscard]] const recipes::Recipe* recipe() const { return mRecipe.get(); }
    [[nodiscard]] recipes::Recipe* recipe() { return mRecipe.get(); }

protected:
    // Forward/backward implementation helpers
    void forward_block(int layer_idx, BlockWeights& weights, BlockActivations& acts, Tensor& residual);
    void recompute_block(int layer_idx, BlockWeights& weights, BlockActivations& acts, Tensor& residual);
    void backward_block(int layer_idx, bool accumulate, BlockWeights& weights, BlockGradients& grads,
                        BlockActivations& acts, typename ModularRunState<Block>::BlockGradients& d_acts,
                        const BackwardBlockHook* hook);

    void backward_lmhead(long B, long T, int micro_step, int grad_accum_steps, NCCLCommunicator& comm);
    void reduce_loss(long B, long T, NCCLCommunicator& comm);
    void calculate_gradient_norm_impl(NCCLCommunicator& comm, float grad_clip, cudaStream_t stream);

private:
    ModelConfig mConfig;
    ModelOptions mOptions;
    std::shared_ptr<TensorAllocator> mAllocator;
    std::shared_ptr<recipes::Recipe> mRecipe = std::make_shared<recipes::BF16Recipe>();  ///< Training recipe for matmul dispatch (defaults to BF16)

    // Module instances
    EmbeddingModule mEmbedding;
    LMHeadModule mLMHead;
    RMSNormModule mFinalNorm;
    std::vector<Block> mBlocks;

    // State managers
    std::unique_ptr<ModularWeightManager<Block>> mWeights;
    std::unique_ptr<ModularGradientManager<Block>> mGrads;
    std::unique_ptr<ModularRunState<Block>> mRunState;

    // FP8 delayed scaling initialization flag
    bool mFP8ScalingInitialized = false;

    // 8-bit AdamW optimizer state
    struct AdamW8BitState {
        bool initialized = false;
        // Number of actual model parameters covered by the optimizer (sum of tensor sizes).
        size_t total_params = 0;
        // Size of the combined state buffers in elements (bytes), including padding for block alignment.
        // We pad between tensors so each tensor starts at an ADAMW8BIT_BLOCK_SIZE boundary, because the
        // single-tensor AdamW8bit kernel assumes per-tensor block indexing starting at 0.
        size_t total_state_elems = 0;
        size_t num_blocks = 0;

        // Offloading configuration (copied from options during init)
        bool offload_state = false;  // If true, state tensors are in pinned host memory
        bool use_zero_copy = false;  // If true, use zero-copy access instead of transfers

        // Quantization maps (256 entries each, on device)
        Tensor quantiles1;  // float[256] - signed quantization map for m
        Tensor quantiles2;  // float[256] - unsigned quantization map for v

        // Per-parameter 8-bit states (combined for all parameters)
        // Location: device (default), or pinned host (if offload_state && use_zero_copy)
        Tensor state1;      // uint8[total_params] - quantized first moment
        Tensor state2;      // uint8[total_params] - quantized second moment

        // Per-block absmax values
        // Location: same as state1/state2
        Tensor absmax1;     // float[num_blocks] - absmax per block for m
        Tensor absmax2;     // float[num_blocks] - absmax per block for v
    };
    std::unique_ptr<AdamW8BitState> mAdamW8BitState;

    // NorMuon optimizer state (for hybrid AdamW/NorMuon optimization)
    // NorMuon is used for 2D weight matrices (attention projections, MLP weights)
    // AdamW is used for embeddings, norms, lm_head, and 0D/1D parameters
    struct NorMuonState {
        bool initialized = false;

        // AdamW 8-bit state for embeddings, norms, lm_head
        // These are stored separately with their own state tracking
        size_t adamw_total_params = 0;
        size_t adamw_state_elems = 0;
        size_t adamw_num_blocks = 0;

        Tensor adamw_quantiles1;  // float[256] - signed quantization map for m
        Tensor adamw_quantiles2;  // float[256] - unsigned quantization map for v
        Tensor adamw_state1;      // uint8[adamw_state_elems] - quantized first moment
        Tensor adamw_state2;      // uint8[adamw_state_elems] - quantized second moment
        Tensor adamw_absmax1;     // float[adamw_num_blocks]
        Tensor adamw_absmax2;     // float[adamw_num_blocks]

        // NorMuon state for 2D weights
        size_t normuon_total_params = 0;
        size_t normuon_state_elems = 0;
        size_t normuon_num_blocks = 0;

        // 8-bit quantized momentum buffer (combined for all 2D weights)
        Tensor momentum_quantiles;  // float[256] - signed quantization map
        Tensor momentum_state;      // uint8[normuon_state_elems]
        Tensor momentum_absmax;     // float[normuon_num_blocks]

        // Variance buffers - stored per 2D weight tensor as FP32
        // Layout: vector of variance buffers, one per 2D weight
        std::vector<Tensor> variance_buffers;
        std::vector<std::pair<int, int>> variance_shapes;  // (M, N) for each buffer

        // Polar Express workspace (reused across layers)
        Tensor polar_workspace;
        size_t max_weight_M = 0;  // Max weight rows seen
        size_t max_weight_N = 0;  // Max weight cols seen

        // Temporary buffer for dequantized momentum (reused per weight)
        Tensor momentum_temp;  // BF16[max_weight_size]

        // cuBLAS handle for Polar Express matrix multiplications
        cublasHandle_t cublas_handle = nullptr;

        ~NorMuonState() {
            if (cublas_handle) {
                cublasDestroy(cublas_handle);
                cublas_handle = nullptr;
            }
        }
    };
    std::unique_ptr<NorMuonState> mNorMuonState;

    // Full-step CUDA graph capture state
    struct FullStepGraphState {
        cudaGraphExec_t graph_exec = nullptr;    // Captured graph executable
        bool captured = false;                    // Whether graph has been captured
        int captured_grad_accum_steps = 0;        // Grad accum steps when captured
        int captured_B = 0;                       // Batch size when captured
        int captured_T = 0;                       // Sequence length when captured

        // Device buffers for dynamic optimizer parameters (updated before graph launch)
        Tensor opt_params;  // float[5]: lr, beta1, beta2, eps, weight_decay
        Tensor opt_step;    // int[1]: step number

        ~FullStepGraphState() {
            if (graph_exec) {
                cudaGraphExecDestroy(graph_exec);
                graph_exec = nullptr;
            }
        }
    };
    std::unique_ptr<FullStepGraphState> mFullStepGraph;

    // Backward hook registry
    BackwardHookRegistry mHookRegistry;

    // Optimizer RNG
    std::minstd_rand mOptimizerRNG;

    // Helper: create module context from run state
    ModuleContext create_context(int B, int T);
};

// ============================================================================
// Implementation split across focused headers
// ============================================================================

#include "modular_model_lifecycle.hpp"
#include "modular_model_forward.hpp"
#include "modular_model_backward.hpp"
#include "modular_model_optimizer_adamw.hpp"
#include "modular_model_optimizer_normuon.hpp"
#include "modular_model_block_ops.hpp"
#include "modular_model_accessors.hpp"
#include "modular_model_allocation.hpp"
#include "modular_model_graph.hpp"

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODEL_MODULAR_MODEL_H
