// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MODULAR_MODEL_H
#define SUROGATE_SRC_MODULES_MODULAR_MODEL_H

#include <memory>
#include <algorithm>
#include <cstdio>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "backward_hooks.h"
#include "forward_hooks.h"
#include "gradient_manager.h"
#include "matmul_context.h"
#include "model_config.h"
#include "module_concept.h"
#include "optimizer_state.h"
#include "run_state.h"
#include "weight_manager.h"

#include "recipes/recipe.h"
#include "recipes/bf16/bf16_recipe.h"
#include "recipes/nvfp4/nvfp4_recipe.h"
#include "recipes/nvfp4/kernels/scaled_swiglu.h"

#include "primitives/attention.h"
#include "primitives/embedding.h"
#include "primitives/linear.h"
#include "primitives/rmsnorm.h"
#include "primitives/swiglu.h"
#include "composite/transformer_block.h"

#include "kernels/kernels.h"
#include "training/runtime_options.h"
#include "training/model.h"
#include "training/dataloader.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/tensor_container.h"

namespace modules {

// Forward declarations
template<typename Block> class ModularWeightManager;
template<typename Block> class ModularGradientManager;
template<typename Block> class ModularRunState;

namespace detail {

/**
 * @brief Execute a callable either directly or under a CUDA graph capture/replay.
 *
 * Mirrors the legacy training path behavior: capture the callable on @p stream, then
 * instantiate/update a cached cudaGraphExec and launch it.
 *
 * Note: If @p function depends on host-side decisions (e.g., optional hooks), the
 * caller must disable graphs in those cases to keep the graph topology stable.
 */
template<typename Function>
inline void trace_or_execute_cuda_graph(Function&& function, cudaStream_t stream,
                                        cudaGraphExec_t& instance, bool enabled) {
    if (!enabled) {
        function();
        return;
    }

    // Fast path: replay existing executable without re-capture.
    if (instance != nullptr) {
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        return;
    }

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    function();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
}

/**
 * @brief Execute a callable with stack checkpoint/restore for CUDA graph compatibility.
 *
 * When graphs are enabled and use temp_alloc() inside the captured function, we must
 * ensure the stack is in the same state before each graph replay. This overload:
 * - On first capture: saves a checkpoint after the function runs
 * - On replay: restores the stack to the checkpoint before launching the graph
 *
 * This ensures temp_alloc returns the same memory addresses that were captured in the graph.
 */
template<typename Function>
inline void trace_or_execute_cuda_graph_with_stack(Function&& function, cudaStream_t stream,
                                                    cudaGraphExec_t& instance, bool enabled,
                                                    DeviceMemoryStack& stack,
                                                    DeviceMemoryStack::Checkpoint& checkpoint) {
    if (!enabled) {
        function();
        return;
    }

    // Fast path: restore stack state and replay existing executable.
    if (instance != nullptr) {
        stack.restore(checkpoint);
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        return;
    }

    // Capture path: save checkpoint before capture so we know where to restore to.
    checkpoint = stack.checkpoint();

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    function();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
}

inline float* abs_max_ptr(Tensor& maybe_quant) {
    return maybe_quant.Data ? maybe_quant.abs_max() : nullptr;
}

// NOTE: backward_qmm_fp4 has been moved to NVFP4Recipe::backward_matmul and NVFP4SimpleRecipe::backward_matmul

/**
 * @brief Legacy backward matmul for BF16/FP8 quantized paths
 *
 * This function is only called when the active recipe returns handles_backward_matmul() = false.
 * With all current recipes (BF16, FP8 Hybrid, MXFP8, NVFP4, NVFP4Simple) returning true,
 * this is effectively dead code but kept for backwards compatibility with custom recipes.
 */
template<typename Block>
inline void backward_qmm(Tensor& dinp,
                         Tensor& dweight, std::optional<Tensor> dbias,
                         Tensor& dout, Tensor& dout_q,
                         Tensor& inp, Tensor& inp_q,
                         Tensor& weight, std::optional<Tensor> bias_buffer,
                         bool accumulate_gradient,
                         ModularRunState<Block>& rs,
                         int B, int T, int C, int OC,
                         bool reuse_inp_quant,
                         bool /*allow_fp4*/,  // unused - FP4 backward now handled by recipes
                         unsigned int /*seed*/,  // unused - FP4 backward now handled by recipes
                         cudaStream_t stream,
                         bool skip_weight_grad = false) {
    // NOTE: FP4 backward path has been moved to NVFP4Recipe::backward_matmul and
    // NVFP4SimpleRecipe::backward_matmul. This function only handles BF16/FP8 paths.

    if (weight.DType == inp.DType) {
        // Compute dinp: dinp = W^T @ dout (always needed for gradient flow)
        matmul(dinp, weight, dout, std::nullopt, nullptr, nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               C, B * T, OC, EMMTranspose::NN, /*accumulate=*/false, stream);

        // Compute dweight: dW += inp^T @ dout (skip if weights are frozen in LoRA-only mode)
        if (!skip_weight_grad) {
            matmul(dweight, inp, dout, std::nullopt, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   C, OC, B * T, EMMTranspose::NT, /*accumulate=*/accumulate_gradient, stream);

            if (dbias.has_value()) {
                if (!bias_buffer.has_value()) {
                    throw std::runtime_error("backward_qmm: dbias requested but bias_buffer not provided");
                }
                backward_bias(dbias.value(), dout, nullptr, nullptr, bias_buffer.value(),
                              B, T, OC, rs.DeviceProp, stream);
            }
        }
        return;
    }

    if (!dout_q.Data || !inp_q.Data) {
        throw std::runtime_error("backward_qmm: quant buffers not allocated");
    }
    if (!dout_q.abs_max()) {
        throw std::runtime_error("backward_qmm: dout_q missing abs_max/scale Stats");
    }

    // Quantize dout using precomputed absmax (filled by upstream kernels when grad quants are enabled).
    quantize_with_abs_max(dout_q, dout_q.scale(), dout, dout_q.abs_max(), (long)B * T * OC, rs.DeviceProp, stream);

    // Compute dinp: dinp = W^T @ dout (always needed for gradient flow)
    auto weight_tp = rs.temp_alloc(inp_q.DType, {C, OC});
    transpose(weight_tp, weight, OC, C, stream);
    matmul(dinp, weight_tp, dout_q, std::nullopt, weight.scale(), dout_q.scale(),
           rs.CublasLtHandle, rs.CuBlasWorkspace,
           C, B * T, OC, EMMTranspose::TN, /*accumulate=*/false, stream);
    rs.temp_free(weight_tp);

    // Compute dweight: dW += inp^T @ dout (skip if weights are frozen in LoRA-only mode)
    if (!skip_weight_grad) {
        auto activation_tp = rs.temp_alloc(inp_q.DType, {C, B * T});
        auto grad_tp = rs.temp_alloc(dout_q.DType, {OC, B * T});
        if (reuse_inp_quant) {
            transpose(activation_tp, inp_q, B * T, C, stream);
        } else {
            // Quantize-and-transpose using absmax from the forward pass (produced by fused ops).
            quantize_and_transpose_with_abs_max(activation_tp, activation_tp.scale(), inp, inp_q.abs_max(),
                                                B * T, C, rs.DeviceProp, stream);
        }
        transpose(grad_tp, dout_q, B * T, OC, stream);

        matmul(dweight, activation_tp, grad_tp, std::nullopt, inp_q.scale(), dout_q.scale(),
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               C, OC, B * T, EMMTranspose::TN, /*accumulate=*/accumulate_gradient, stream);

        if (dbias.has_value()) {
            if (!bias_buffer.has_value()) {
                throw std::runtime_error("backward_qmm: dbias requested but bias_buffer not provided");
            }
            backward_bias(dbias.value(), dout_q, inp_q.scale(), dout_q.scale(), bias_buffer.value(),
                          B, T, OC, rs.DeviceProp, stream);
        }

        rs.temp_free(grad_tp);
        rs.temp_free(activation_tp);
    }
}

/**
 * @brief Helper to execute recipe-driven forward matmul with full context setup
 *
 * This function handles the setup of MatmulContext for recipe->forward_matmul(),
 * including FP8 quant buffers and delayed scaling indices. Used by Phase 6
 * consolidation to reduce boilerplate in the forward loop.
 *
 * @param recipe The training recipe (must have handles_forward_matmul() == true)
 * @param out Output tensor
 * @param inp Input tensor
 * @param weight Weight tensor
 * @param bias Optional bias tensor (nullptr if no bias)
 * @param rs Run state
 * @param B Batch size
 * @param T Sequence length
 * @param C_in Input channels
 * @param C_out Output channels
 * @param layer_idx Current layer index
 * @param op Matmul operation type
 * @param inp_quant FP8 input quant buffer (for FP8 recipes)
 * @param cached_weight Optional cached FP8 weight
 * @param delayed_quantizer_idx Delayed scaling quantizer index (-1 for JIT)
 * @param stream CUDA stream
 * @param cached_fp4_data Optional cached FP4 packed weight data (CUTLASS layout)
 * @param cached_fp4_scales Optional cached FP4 block scales (FP8 E4M3, CUTLASS layout)
 * @param cached_fp4_amax Optional cached FP4 global amax pointer (device memory)
 */
template<typename Block>
inline void recipe_forward_matmul(
    const recipes::Recipe& recipe,
    Tensor& out, Tensor& inp, Tensor& weight, Tensor* bias,
    ModularRunState<Block>& rs,
    int B, int T, int C_in, int C_out,
    int layer_idx, MatmulOp op,
    Tensor* inp_quant,
    const Tensor* cached_weight,
    int delayed_quantizer_idx,
    cudaStream_t stream,
    const Tensor* cached_fp4_data = nullptr,
    const Tensor* cached_fp4_scales = nullptr,
    const float* cached_fp4_amax = nullptr)
{
    MatmulContext ctx;
    ctx.out = &out;
    ctx.inp = &inp;
    ctx.weight = &weight;
    ctx.bias = bias;
    ctx.B = B;
    ctx.T = T;
    ctx.C_in = C_in;
    ctx.C_out = C_out;
    ctx.run_state = &rs;
    ctx.stream = stream;
    ctx.layer_idx = layer_idx;
    ctx.op = op;
    ctx.inp_quant = inp_quant;
    ctx.cached_weight = cached_weight;
    ctx.delayed_quantizer_idx = delayed_quantizer_idx;
    ctx.cached_fp4_data = cached_fp4_data;
    ctx.cached_fp4_scales = cached_fp4_scales;
    ctx.cached_fp4_amax = cached_fp4_amax;

    recipe.forward_matmul(ctx);
}

/**
 * @brief Helper to execute recipe-driven backward matmul with full context setup
 *
 * @param recipe The training recipe (must have handles_backward_matmul() == true)
 * @param dinp Gradient w.r.t. input
 * @param dweight Gradient w.r.t. weight
 * @param dbias Optional gradient w.r.t. bias (nullptr if no bias)
 * @param dout Upstream gradient
 * @param inp Input activation from forward pass
 * @param weight Weight tensor
 * @param rs Run state
 * @param B Batch size
 * @param T Sequence length
 * @param C_in Input channels
 * @param C_out Output channels
 * @param layer_idx Current layer index
 * @param op Matmul operation type
 * @param accumulate Whether to accumulate into gradient buffers
 * @param skip_weight_grad Skip weight gradient (for LoRA-only)
 * @param inp_quant FP8 input quant buffer
 * @param dout_quant E5M2 gradient buffer
 * @param bias_buffer Scratch buffer for bias gradient
 * @param stream CUDA stream
 */
template<typename Block>
inline void recipe_backward_matmul(
    const recipes::Recipe& recipe,
    Tensor& dinp, Tensor& dweight, Tensor* dbias,
    Tensor& dout, Tensor& inp, Tensor& weight,
    ModularRunState<Block>& rs,
    int B, int T, int C_in, int C_out,
    int layer_idx, MatmulOp op,
    bool accumulate, bool skip_weight_grad,
    Tensor* inp_quant,
    Tensor* dout_quant,
    Tensor* bias_buffer,
    cudaStream_t stream)
{
    MatmulContext ctx;
    ctx.dinp = &dinp;
    ctx.dweight = &dweight;
    ctx.dbias = dbias;
    ctx.dout = &dout;
    ctx.inp = &inp;
    ctx.weight = &weight;
    ctx.B = B;
    ctx.T = T;
    ctx.C_in = C_in;
    ctx.C_out = C_out;
    ctx.run_state = &rs;
    ctx.stream = stream;
    ctx.layer_idx = layer_idx;
    ctx.op = op;
    ctx.accumulate = accumulate;
    ctx.skip_weight_grad = skip_weight_grad;
    ctx.inp_quant = inp_quant;
    ctx.dout_quant = dout_quant;
    ctx.bias_buffer = bias_buffer;

    recipe.backward_matmul(ctx);
}

} // namespace detail

/**
 * @brief Empty tensor container for unused ITensorContainer returns
 */
class EmptyTensorContainer : public ITensorContainer {
public:
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>&) override {}
};

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
     * @brief 8-bit AdamW optimizer update (internal implementation)
     */
    void update_adamw_8bit(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t,
                           float epsilon, float weight_decay, float grad_clip);

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
// Implementation
// ============================================================================

template<typename Block>
ModularTransformerModel<Block>::ModularTransformerModel(
    const ModelConfig& config, const ModelOptions& options,
    int rank, int world, const std::shared_ptr<TensorAllocator>& alloc)
    : mConfig(config)
    , mOptions(options)
    , mAllocator(alloc ? alloc : std::make_shared<TensorAllocator>())
    , mEmbedding({.vocab_size = config.VocabSize, .hidden_size = config.HiddenSize})
    , mLMHead({.hidden_size = config.HiddenSize, .vocab_size = config.VocabSize,
               .tied_weights = config.TiedWordEmbeddings, .num_chunks = options.lmhead_chunks})
    , mFinalNorm({.hidden_size = config.HiddenSize, .epsilon = config.RmsNormEps})
    , mOptimizerRNG(42) {

    // Create transformer blocks
    BlockConfig block_config;
    block_config.hidden_size = config.HiddenSize;
    block_config.intermediate_size = config.IntermediateSize;
    block_config.num_query_heads = config.NumQueryHeads;
    block_config.num_kv_heads = config.NumKeyValHeads;
    block_config.head_size = config.head_size();
    block_config.rms_norm_eps = config.RmsNormEps;
    block_config.rope_theta = config.RopeTheta;
    block_config.max_seq_len = config.MaxPositionEmbeddings;
    block_config.use_qkv_bias = config.UseQKVBias;
    if constexpr (requires { block_config.use_qk_norm; }) {
        block_config.use_qk_norm = config.UseQKNorm;
    }

    mBlocks.reserve(config.NumLayers);
    for (int i = 0; i < config.NumLayers; ++i) {
        mBlocks.emplace_back(block_config);
    }

    // Create weight manager
    typename ModularWeightManager<Block>::Config wm_config;
    wm_config.num_layers = config.NumLayers;
    wm_config.block_config = block_config;
    wm_config.model_dtype = options.model_dtype.value_or(config.DType);
    // For FP8 forward-only mode, keep work weights in BF16 (not FP8).
    // Weight quantization to FP8 happens on-the-fly in forward_qmm_fp8.
    // This allows backward to use BF16 weights for stability.
    wm_config.matmul_dtype = options.get_matmul_dtype();
    wm_config.master_dtype = options.master_dtype.value_or(wm_config.model_dtype);
    wm_config.shard_idx = rank;
    wm_config.num_shards = world;
    wm_config.shard_weights = options.shard_weights;
    wm_config.offload_master = options.offload_master;
    wm_config.offload_quants = options.offload_quants;
    wm_config.use_zero_copy = options.use_zero_copy;
    wm_config.offload_alloc = options.get_offload_alloc();
    wm_config.persistent_quants = options.persistent_quants;
    wm_config.init_projections_to_zero = options.init_projections_to_zero;
    wm_config.vocab_size = config.VocabSize;
    wm_config.hidden_size = config.HiddenSize;
    wm_config.tied_embeddings = config.TiedWordEmbeddings;
    wm_config.skip_block_allocation = options.skip_block_allocation;
    wm_config.enable_fp8_forward = options.enable_fp8_forward;
    wm_config.enable_fp4_forward = options.enable_fp4_forward;

    mWeights = std::make_unique<ModularWeightManager<Block>>(wm_config, *mAllocator);
}

template<typename Block>
ModularTransformerModel<Block>::~ModularTransformerModel() = default;

template<typename Block>
void ModularTransformerModel<Block>::init_weights(NCCLCommunicator& comm) {
    mWeights->random_init(42, comm);
}

template<typename Block>
void ModularTransformerModel<Block>::import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) {
    mWeights->import_from_file(file_name, allow_cast, comm);
}

template<typename Block>
void ModularTransformerModel<Block>::export_weights(const std::string& file_name, NCCLCommunicator& comm) {
    mWeights->export_to_file(file_name, comm);
}

template<typename Block>
void ModularTransformerModel<Block>::on_restore_checkpoint(NCCLCommunicator& comm) {
    mWeights->synchronize_absmax(comm);
}

template<typename Block>
ModuleContext ModularTransformerModel<Block>::create_context(int B, int T) {
    auto& rs = *mRunState;
    ModuleContext ctx;
    ctx.stream = rs.non_block_activations().encoded.stream;  // Main stream
    ctx.cublas_handle = nullptr;  // Will be set from IRunState
    ctx.cudnn_handle = nullptr;   // Will be set from IRunState
    ctx.workspace = &rs.scratch().cudnn_workspace;
    ctx.device_prop = nullptr;    // Will be set from IRunState
    ctx.B = B;
    ctx.T = T;
    ctx.position_ids = nullptr;   // Will be set per-call
    ctx.matmul_dtype = mOptions.get_matmul_dtype();
    ctx.use_quantization = mOptions.matmul_dtype.has_value() &&
                           mOptions.matmul_dtype.value() != mOptions.model_dtype.value_or(mConfig.DType);
    return ctx;
}

template<typename Block>
void ModularTransformerModel<Block>::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    forward_with_hook(inputs, position_ids, comm, micro_step, {});
}

template<typename Block>
void ModularTransformerModel<Block>::forward_with_hook(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step,
                                                       const ForwardBlockHook& hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;  // From IRunState base class
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    long C = mConfig.HiddenSize;
    long V = mConfig.VocabSize;

    // Invalidate weight cache on first micro-step
    if (micro_step == 0) {
        // Ensure no weight prefetch starts before the previous optimizer update finished.
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        mWeights->invalidate();

        // Zero recorded amaxes for FP8 delayed scaling at start of each training step
        if (rs.has_fp8_delayed_scaling()) {
            auto& fp8_state = rs.fp8_scaling_state();
            // Initialize on first use (scales to 1.0, history to 0)
            if (!mFP8ScalingInitialized) {
                fp8_state.reset(main_stream);
                mFP8ScalingInitialized = true;
            }
            fp8_state.zero_recorded_amaxes(main_stream);
        }
    }

    // Copy inputs to device
    assert(inputs.Device == -1);  // Inputs should be on host
    {
        NvtxRange r{"copy-input"};
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, main_stream));

        // Copy position IDs to GPU for RoPE
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, main_stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, main_stream));
        }
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, main_stream));
    }

    // Embedding lookup
    {
        NvtxRange emb_range("embedding");
        mWeights->gather_embeddings(comm, rs.side_stream());
        auto& emb_weights = mWeights->get_embeddings(main_stream);
        auto& encoded = rs.non_block_activations().encoded;
        encoder_forward(encoded, rs.Inputs, emb_weights, std::nullopt, B, T, C, V, main_stream);
        mWeights->release_embeddings(main_stream);
    }

    // Configuration shortcuts
    const long D = mConfig.IntermediateSize;
    const int Hq = mConfig.NumQueryHeads;
    const int Hkv = mConfig.NumKeyValHeads;
    const int Hs = mConfig.head_size();
    const long AttC = static_cast<long>(Hq) * static_cast<long>(Hs);
    const long qkv_channels = mConfig.qkv_channels();

    // RoPE frequencies and position IDs
    auto& freq_cis = rs.non_block_activations().freq_cis;
    int* pos_ids_ptr = rs.PositionIDs.template get<int>();

    // Process transformer blocks
    // Hooks can be captured safely as long as their topology is stable for the run (e.g. LoRA hooks).
    const bool use_cuda_graphs = mOptions.use_cuda_graphs;
    if (use_cuda_graphs) {
        rs.configure_forward_graphs(/*hooked=*/static_cast<bool>(hook));
    }
    mWeights->gather_block(0, comm, rs.side_stream());
    for (int l = 0; l < mConfig.NumLayers; l++) {
        NvtxRange layer_range("Layer", l);

        // Prefetch next block
        if (l != mConfig.NumLayers - 1) {
            mWeights->gather_block(l + 1, comm, rs.side_stream());
        }

        auto& weights = mWeights->get_block(l, main_stream);
        
        auto& acts = rs.simplified_acts(l);
        auto& q = rs.simplified_quant_acts(l);

        // Get the residual buffer for this layer (like legacy DeviceResiduals)
        // For layer 0: residual is the encoded input
        // For layer L>0: residual is get_residual(l-1) which gets populated by the LN1 fused op
        Tensor& residual = l == 0 ? rs.non_block_activations().encoded :
                                    rs.get_residual(l - 1, main_stream);

        // 1) First layer norm
        // The fused op writes: output_residual = inp1 + inp2, output_norm = rmsnorm(output_residual)
        // For L=0: just normalize encoded (no residual accumulation)
        // For L>0: accumulate prev.residual_att + prev.mlp_down into get_residual(l-1)
        // Determine abs_max pointer for RMSNorm: for FP8 forward, we need abs_max computed
        // so the recipe can reuse it for input quantization (FP8/FP4).
        float* ln1_abs_max_ptr = nullptr;
        if (rs.has_fp8_forward()) {
            ln1_abs_max_ptr = rs.fp8_forward_quants().ln1.abs_max();
        } else if (rs.has_fp4_forward()) {
            ln1_abs_max_ptr = rs.fp4_forward_quants().ln1_global_amax;
        } else if (rs.has_activation_quants()) {
            ln1_abs_max_ptr = q.ln1.abs_max();
        }

        if (l == 0) {
            // First layer: just normalize (no residual addition)
            rmsnorm_forward(acts.ln1, acts.ln1_rstd, residual, weights.ln1.weight,
                            ln1_abs_max_ptr,
                            mConfig.RmsNormEps, B, T, C, main_stream);
        } else {
            // Subsequent layers: compute accumulated residual and normalize
            // residual buffer = prev.residual_att + prev.mlp_down
            auto& prev = rs.simplified_acts(l - 1);
            // Write to residual buffer (get_residual(l-1)) which is separate from prev.residual_att
            fused_residual_rmsnorm_forward(residual, acts.ln1, acts.ln1_rstd,
                                           prev.residual_att, prev.mlp_down, weights.ln1.weight,
                                           ln1_abs_max_ptr,
                                           mConfig.RmsNormEps, B * T, C, main_stream);
            // If residual offload is enabled, mark the produced residual (for layer l-1) ready for D2H.
            if (mOptions.offload_residuals) {
                rs.mark_residual_ready(l - 1, main_stream);
            }
        }

        detail::trace_or_execute_cuda_graph_with_stack([&]() {
            // 2) QKV projection - recipe handles all format decisions
            {
                Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln1 : nullptr;
                const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().qkv_weight : nullptr;
                const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_LN1) : -1;

                // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                const Tensor* fp4_data = nullptr;
                const Tensor* fp4_scales = nullptr;
                const float* fp4_amax = nullptr;
                if (mWeights->has_fp4_forward_cache()) {
                    auto& fp4_cache = mWeights->fp4_weight_cache();
                    fp4_data = &fp4_cache.qkv_weight.data;
                    fp4_scales = &fp4_cache.qkv_weight.scales;
                    fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 0;  // qkv at offset 0
                }

                detail::recipe_forward_matmul(
                    *mRecipe, acts.qkv, acts.ln1, weights.attention.qkv_weight,
                    weights.attention.qkv_bias.has_value() ? &weights.attention.qkv_bias.value() : nullptr,
                    rs, (int)B, (int)T, (int)C, (int)qkv_channels,
                    l, modules::MatmulOp::QKV,
                    inp_quant, cached_weight, qidx, main_stream,
                    fp4_data, fp4_scales, fp4_amax);
            }

            if (hook) {
                hook(l, main_stream, ForwardHookPoint::AfterQKVProjection);
            }

            const bool use_qk_norm =
                weights.attention.q_norm_weight.has_value() && weights.attention.k_norm_weight.has_value();

            // 2.5) Optional Q/K head RMSNorm (Qwen3-style).
            if (use_qk_norm) {
                const int q_rows = Hq * Hs;
                qkv_head_rmsnorm_forward(
                    acts.qkv, acts.q_rstd, weights.attention.q_norm_weight.value(),
                    mConfig.RmsNormEps,
                    (int)B, (int)T, (int)qkv_channels,
                    /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
                    main_stream
                );
                qkv_head_rmsnorm_forward(
                    acts.qkv, acts.k_rstd, weights.attention.k_norm_weight.value(),
                    mConfig.RmsNormEps,
                    (int)B, (int)T, (int)qkv_channels,
                    /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
                    main_stream
                );
            }

            // 3) Apply RoPE
            Tensor& qkv_for_attn = (use_qk_norm && acts.qkv_rope.Data != nullptr) ? acts.qkv_rope : acts.qkv;
            if (use_qk_norm && acts.qkv_rope.Data != nullptr) {
                // Keep pre-RoPE QKV (after QK-norm) for backward, and store RoPE outputs separately.
                if (mOptions.use_fused_rope) {
                    rope_fused_forward(qkv_for_attn, acts.qkv, pos_ids_ptr, nullptr, mConfig.RopeTheta, B, T, Hq, Hkv, Hs, main_stream);
                } else {
                    rope_forward(qkv_for_attn, acts.qkv, freq_cis, pos_ids_ptr, nullptr, B, T, Hq, Hkv, Hs, main_stream);
                }
            } else {
                if (mOptions.use_fused_rope) {
                    rope_fused_forward(acts.qkv, acts.qkv, pos_ids_ptr, nullptr, mConfig.RopeTheta, B, T, Hq, Hkv, Hs, main_stream);
                } else {
                    rope_forward(acts.qkv, acts.qkv, freq_cis, pos_ids_ptr, nullptr, B, T, Hq, Hkv, Hs, main_stream);
                }
            }

            // 4) Attention (FlashAttention via cuDNN)
            // Match legacy behavior: use the shared CuBLAS workspace for cuDNN forward.
            attention_forward_cudnn(acts.att, acts.lse, qkv_for_attn, rs.CuBlasWorkspace,
                                    rs.CudnnHandle, B, T, Hq, Hkv, Hs, main_stream);

            // Compute abs_max for attention output (used by FP8/FP4 recipes for output projection quantization)
            if (rs.has_fp8_forward()) {
                abs_max(rs.fp8_forward_quants().att.abs_max(), acts.att, (long)acts.att.nelem(), rs.DeviceProp, main_stream);
            } else if (rs.has_fp4_forward()) {
                abs_max(rs.fp4_forward_quants().att_global_amax, acts.att, (long)acts.att.nelem(), rs.DeviceProp, main_stream);
            } else if (rs.has_activation_quants()) {
                abs_max(q.att.abs_max(), acts.att, (long)acts.att.nelem(), rs.DeviceProp, main_stream);
            }

            // 5) Output projection - recipe handles all format decisions
            {
                Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().att : nullptr;
                const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().o_weight : nullptr;
                const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_ATT) : -1;

                // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                const Tensor* fp4_data = nullptr;
                const Tensor* fp4_scales = nullptr;
                const float* fp4_amax = nullptr;
                if (mWeights->has_fp4_forward_cache()) {
                    auto& fp4_cache = mWeights->fp4_weight_cache();
                    fp4_data = &fp4_cache.o_weight.data;
                    fp4_scales = &fp4_cache.o_weight.scales;
                    fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 1;  // o at offset 1
                }

                detail::recipe_forward_matmul(
                    *mRecipe, acts.att_out, acts.att, weights.attention.out_weight,
                    nullptr,  // no bias
                    rs, (int)B, (int)T, (int)AttC, (int)C,
                    l, modules::MatmulOp::AttnOut,
                    inp_quant, cached_weight, qidx, main_stream,
                    fp4_data, fp4_scales, fp4_amax);
            }

            if (hook) {
                hook(l, main_stream, ForwardHookPoint::AfterAttnOutProjection);
            }

            // 6) Residual + LN2 (fused)
            // LN2 computes: acts.residual_att = residual + att_out, acts.ln2 = rmsnorm(acts.residual_att)
            // Determine abs_max pointer for LN2 (same logic as LN1)
            float* ln2_abs_max_ptr = nullptr;
            if (rs.has_fp8_forward()) {
                ln2_abs_max_ptr = rs.fp8_forward_quants().ln2.abs_max();
            } else if (rs.has_fp4_forward()) {
                ln2_abs_max_ptr = rs.fp4_forward_quants().ln2_global_amax;
            } else if (rs.has_activation_quants()) {
                ln2_abs_max_ptr = q.ln2.abs_max();
            }
            fused_residual_rmsnorm_forward(acts.residual_att, acts.ln2, acts.ln2_rstd,
                                           residual, acts.att_out, weights.ln2.weight,
                                           ln2_abs_max_ptr,
                                           mConfig.RmsNormEps, B * T, C, main_stream);

            // 7) MLP up projection (gate + up fused) - Dense blocks only
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                // In full-block recompute mode, don't persist the large FFN intermediates
                // (mlp_up/swiglu). Keep them stack-backed so they can overlap with other temporaries.
                bool free_ffn_temporaries = false;
                if (mOptions.recompute_block) {
                    // Use the per-layer descriptors, but back them with stack memory for this forward.
                    // This is required for LoRA forward hooks, which mutate `acts.mlp_up` and consume `acts.swiglu`.
                    if (acts.mlp_up.Data == nullptr) rs.temp_acquire(acts.mlp_up);
                    if (acts.swiglu.Data == nullptr) rs.temp_acquire(acts.swiglu);
                    free_ffn_temporaries = true;

                    // MLPUp projection (recompute path) - recipe handles all format decisions
                    {
                        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln2 : nullptr;
                        const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_up_weight : nullptr;
                        const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_LN2) : -1;

                        // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                        const Tensor* fp4_data = nullptr;
                        const Tensor* fp4_scales = nullptr;
                        const float* fp4_amax = nullptr;
                        if (mWeights->has_fp4_forward_cache()) {
                            auto& fp4_cache = mWeights->fp4_weight_cache();
                            fp4_data = &fp4_cache.mlp_up_weight.data;
                            fp4_scales = &fp4_cache.mlp_up_weight.scales;
                            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 2;  // mlp_up at offset 2
                        }

                        detail::recipe_forward_matmul(
                            *mRecipe, acts.mlp_up, acts.ln2, weights.mlp_up_weight,
                            nullptr, rs, (int)B, (int)T, (int)C, (int)(2 * D),
                            l, modules::MatmulOp::MLPUp,
                            inp_quant, cached_weight, qidx, main_stream,
                            fp4_data, fp4_scales, fp4_amax);
                    }

                    if (hook) {
                        hook(l, main_stream, ForwardHookPoint::AfterMLPUpProjection);
                    }

                    // 8) SwiGLU activation
                    // Determine abs_max pointer for swiglu output (used by FP8/FP4 recipes for MLP down quantization)
                    float* swiglu_abs_max_ptr = nullptr;
                    if (rs.has_fp8_forward()) {
                        swiglu_abs_max_ptr = rs.fp8_forward_quants().swiglu.abs_max();
                    } else if (rs.has_fp4_forward()) {
                        swiglu_abs_max_ptr = rs.fp4_forward_quants().swiglu_global_amax;
                    } else if (rs.has_activation_quants()) {
                        swiglu_abs_max_ptr = q.swiglu.abs_max();
                    }
                    if (mRecipe && mRecipe->requires_scaled_swiglu()) {
                        // Recipe-driven scaled SwiGLU forward
                        modules::SwiGLUContext ctx;
                        ctx.out = &acts.swiglu;
                        ctx.scale_out = &acts.swiglu_scale;
                        ctx.inp = &acts.mlp_up;
                        ctx.abs_max_out = swiglu_abs_max_ptr;
                        ctx.B = (int)B;
                        ctx.T = (int)T;
                        ctx.D = (int)D;
                        ctx.stream = main_stream;
                        mRecipe->swiglu_forward(ctx);
                    } else {
                        swiglu_forward(acts.swiglu, acts.mlp_up, swiglu_abs_max_ptr, B, T, D, main_stream);
                    }

                    // 9) MLP down projection (recompute path) - recipe handles all format decisions
                    {
                        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().swiglu : nullptr;
                        const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_down_weight : nullptr;
                        const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_SWIGLU) : -1;

                        // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                        const Tensor* fp4_data = nullptr;
                        const Tensor* fp4_scales = nullptr;
                        const float* fp4_amax = nullptr;
                        if (mWeights->has_fp4_forward_cache()) {
                            auto& fp4_cache = mWeights->fp4_weight_cache();
                            fp4_data = &fp4_cache.mlp_down_weight.data;
                            fp4_scales = &fp4_cache.mlp_down_weight.scales;
                            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 3;  // mlp_down at offset 3
                        }

                        detail::recipe_forward_matmul(
                            *mRecipe, acts.mlp_down, acts.swiglu, weights.mlp_down_weight,
                            nullptr, rs, (int)B, (int)T, (int)D, (int)C,
                            l, modules::MatmulOp::MLPDown,
                            inp_quant, cached_weight, qidx, main_stream,
                            fp4_data, fp4_scales, fp4_amax);
                    }
                } else {
                    // MLPUp projection (non-recompute path) - recipe handles all format decisions
                    {
                        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln2 : nullptr;
                        const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_up_weight : nullptr;
                        const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_LN2) : -1;

                        // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                        const Tensor* fp4_data = nullptr;
                        const Tensor* fp4_scales = nullptr;
                        const float* fp4_amax = nullptr;
                        if (mWeights->has_fp4_forward_cache()) {
                            auto& fp4_cache = mWeights->fp4_weight_cache();
                            fp4_data = &fp4_cache.mlp_up_weight.data;
                            fp4_scales = &fp4_cache.mlp_up_weight.scales;
                            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 2;  // mlp_up at offset 2
                        }

                        detail::recipe_forward_matmul(
                            *mRecipe, acts.mlp_up, acts.ln2, weights.mlp_up_weight,
                            nullptr, rs, (int)B, (int)T, (int)C, (int)(2 * D),
                            l, modules::MatmulOp::MLPUp,
                            inp_quant, cached_weight, qidx, main_stream,
                            fp4_data, fp4_scales, fp4_amax);
                    }

                    if (hook) {
                        hook(l, main_stream, ForwardHookPoint::AfterMLPUpProjection);
                    }

                    // 8) SwiGLU activation
                    // Determine abs_max pointer for swiglu output (used by FP8/FP4 recipes for MLP down quantization)
                    float* swiglu_abs_max_ptr2 = nullptr;
                    if (rs.has_fp8_forward()) {
                        swiglu_abs_max_ptr2 = rs.fp8_forward_quants().swiglu.abs_max();
                    } else if (rs.has_fp4_forward()) {
                        swiglu_abs_max_ptr2 = rs.fp4_forward_quants().swiglu_global_amax;
                    } else if (rs.has_activation_quants()) {
                        swiglu_abs_max_ptr2 = q.swiglu.abs_max();
                    }
                    if (mRecipe && mRecipe->requires_scaled_swiglu()) {
                        // Recipe-driven scaled SwiGLU forward
                        modules::SwiGLUContext ctx;
                        ctx.out = &acts.swiglu;
                        ctx.scale_out = &acts.swiglu_scale;
                        ctx.inp = &acts.mlp_up;
                        ctx.abs_max_out = swiglu_abs_max_ptr2;
                        ctx.B = (int)B;
                        ctx.T = (int)T;
                        ctx.D = (int)D;
                        ctx.stream = main_stream;
                        mRecipe->swiglu_forward(ctx);
                    } else {
                        swiglu_forward(acts.swiglu, acts.mlp_up, swiglu_abs_max_ptr2, B, T, D, main_stream);
                    }

                    // 9) MLP down projection (non-recompute path) - recipe handles all format decisions
                    {
                        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().swiglu : nullptr;
                        const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_down_weight : nullptr;
                        const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_SWIGLU) : -1;

                        // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                        const Tensor* fp4_data = nullptr;
                        const Tensor* fp4_scales = nullptr;
                        const float* fp4_amax = nullptr;
                        if (mWeights->has_fp4_forward_cache()) {
                            auto& fp4_cache = mWeights->fp4_weight_cache();
                            fp4_data = &fp4_cache.mlp_down_weight.data;
                            fp4_scales = &fp4_cache.mlp_down_weight.scales;
                            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 3;  // mlp_down at offset 3
                        }

                        detail::recipe_forward_matmul(
                            *mRecipe, acts.mlp_down, acts.swiglu, weights.mlp_down_weight,
                            nullptr, rs, (int)B, (int)T, (int)D, (int)C,
                            l, modules::MatmulOp::MLPDown,
                            inp_quant, cached_weight, qidx, main_stream,
                            fp4_data, fp4_scales, fp4_amax);
                    }
                }

                // Apply saved scale from scaled SwiGLU to down_proj output (nvfp4-simple)
                if (rs.has_scaled_swiglu()) {
                    recipes::nvfp4::scale_rows(
                        acts.mlp_down.template get<nv_bfloat16>(),
                        acts.swiglu_scale.template get<float>(),
                        B, T, C, main_stream);
                }

                if (hook) {
                    hook(l, main_stream, ForwardHookPoint::AfterMLPDownProjection);
                }

                // LoRA's `AfterMLPDownProjection` hook consumes `acts.swiglu`.
                // Keep recompute-block intermediates alive until after the hook runs.
                if (free_ffn_temporaries) {
                    rs.temp_free(acts.swiglu);
                    rs.temp_free(acts.mlp_up);
                }
            } else {
                // MoE block - TODO: implement expert routing
                // For now, skip MLP processing (will result in zeros)
                fill_zero(acts.mlp_down, main_stream);
            }
        }, main_stream, rs.forward_block_graph(l), use_cuda_graphs,
           rs.Stack, rs.forward_block_stack_checkpoint(l));

        mWeights->release_block(l, main_stream);
        // Offload the residual stream for the previous layer (l-1) once it's been produced.
        // Safe to overlap with subsequent reads of `residual` on the main stream because both are reads.
        if (l > 0 && mOptions.offload_residuals) {
            rs.put_residual(l - 1, rs.side_stream());
        }
    }

    // Final layer norm
    {
        NvtxRange lnf_range("LNF");
        auto& last_acts = rs.simplified_acts(mConfig.NumLayers - 1);
        mWeights->gather_final_norm(comm, rs.side_stream());
        auto& lnf_weight = mWeights->get_final_norm(main_stream);

        // Final residual: last_residual_att + last_mlp_down
        auto& final_res = rs.get_final_residual();
        fused_residual_rmsnorm_forward(final_res, rs.non_block_activations().ln_final,
                                       rs.non_block_activations().ln_final_rstd,
                                       last_acts.residual_att, last_acts.mlp_down, lnf_weight,
                                       nullptr, mConfig.RmsNormEps, B * T, C, main_stream);
        mWeights->release_final_norm(main_stream);
    }

    // Wait for input transfer to complete before returning
    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
    CUDA_CHECK(cudaEventRecord(rs.ForwardDone, main_stream));
}

template<typename Block>
float ModularTransformerModel<Block>::validate(Tensor inputs, Tensor position_ids, Tensor targets,
                                                NCCLCommunicator& comm, int micro_step) {
    return validate_with_hook(inputs, position_ids, targets, comm, micro_step, {});
}

template<typename Block>
float ModularTransformerModel<Block>::validate_with_hook(Tensor inputs, Tensor position_ids, Tensor targets,
                                                         NCCLCommunicator& comm, int micro_step,
                                                         const ForwardBlockHook& hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    const size_t V = mConfig.VocabSize;
    const size_t Vp = mConfig.VocabSize;  // Padded vocab size (same as V for now)
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    long C = mConfig.HiddenSize;
    cudaStream_t main_stream = rs.MainStream;

    // Run forward pass
    forward_with_hook(inputs, position_ids, comm, micro_step, hook);

    NvtxRange classifier_and_loss_range("classifier_and_loss");

    // Initialize losses and counts
    fill_zero(rs.Losses, main_stream);
    fill_zero(rs.ValidTokenCount, main_stream);
    fill_zero(rs.CorrectCount, main_stream);

    // Copy targets to device
    const std::size_t target_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
    if (targets.Device == -1) {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyDeviceToDevice));
    }

    // LM head: project ln_final output to vocab logits
    // Do this in chunks for memory efficiency
    long nano_batches = mOptions.lmhead_chunks;
    int nano_batch_size = static_cast<int>((B * T) / nano_batches);

    mWeights->gather_lm_head(comm, rs.side_stream());

    // Use the output tensor from non_block_activations as scratch
    rs.temp_acquire(rs.non_block_activations().output);

    for (int nano_step = 0; nano_step < nano_batches; nano_step++) {
        // Slice tensors for this nano-batch
        // Note: Update both Data pointer AND Sizes for correct matmul dimensions
        Tensor lnf_slice = rs.non_block_activations().ln_final;
        lnf_slice.Data = static_cast<std::byte*>(lnf_slice.Data) +
                         nano_step * nano_batch_size * C * get_dtype_size(lnf_slice.DType);
        lnf_slice.Sizes[0] = nano_batch_size;
        lnf_slice.Sizes[1] = C;
        lnf_slice.Rank = 2;

        Tensor tgt = rs.Targets;
        tgt.Data = static_cast<std::byte*>(tgt.Data) +
                   nano_step * nano_batch_size * get_dtype_size(tgt.DType);
        tgt.Sizes[0] = nano_batch_size;
        tgt.Rank = 1;

        Tensor losses = rs.Losses;
        losses.Data = static_cast<std::byte*>(losses.Data) +
                      nano_step * nano_batch_size * get_dtype_size(losses.DType);
        losses.Sizes[0] = nano_batch_size;
        losses.Rank = 1;

        // LM head matmul: logits = lnf @ lm_head.T
        matmul(rs.non_block_activations().output, mWeights->get_lm_head(main_stream), lnf_slice,
               std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
               V, nano_batch_size, C, EMMTranspose::TN, false, main_stream);

        // Fused classifier: softmax + cross-entropy loss + accuracy
        const float d_loss = 1.0f;
        fused_classifier(rs.non_block_activations().output, losses, d_loss, tgt,
                         &rs.ValidTokenCount, &rs.CorrectCount, nano_batch_size, V, Vp, true, main_stream);
    }

    rs.temp_free(rs.non_block_activations().output);
    mWeights->release_lm_head(main_stream);

    // Reduce loss
    reduce_loss(B, T, comm);

    // Get valid token count and correct count, then normalize loss and compute accuracy
    comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, main_stream);
    comm.all_reduce_sum_int(rs.CorrectCount.template get<int>(), /*n=*/1, main_stream);

    CUDA_CHECK(cudaMemcpyAsync(rs.NormHost, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost, main_stream));
    CUDA_CHECK(cudaMemcpyAsync(rs.AccuracyHost, rs.CorrectCount.Data, sizeof(int), cudaMemcpyDeviceToHost, main_stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    int valid_tokens = *reinterpret_cast<int*>(rs.NormHost);  // Reusing NormHost for valid token count
    int correct_tokens = *reinterpret_cast<int*>(rs.AccuracyHost);

    if (valid_tokens > 0) {
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, comm.world_size()));
        *rs.LossHost /= avg_valid;
        // Store accuracy as a percentage
        *rs.AccuracyHost = (static_cast<float>(correct_tokens) / static_cast<float>(valid_tokens)) * 100.0f;
    } else {
        *rs.LossHost = 0.0f;
        *rs.AccuracyHost = 0.0f;
    }

    return *rs.LossHost;
}

template<typename Block>
void ModularTransformerModel<Block>::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                                               int grad_accum_steps, int micro_step) {
    backward_with_hook(inputs, targets, comm, grad_accum_steps, micro_step, {});
}

template<typename Block>
void ModularTransformerModel<Block>::backward_with_hook(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                                                         int grad_accum_steps, int micro_step,
                                                         const BackwardBlockHook& hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    rs.GradAccumSteps = std::max(1, grad_accum_steps);
    rs.WorldSize = std::max(1, comm.world_size());
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    long C = mConfig.HiddenSize;
    long L = mConfig.NumLayers;

    // LoRA-only mode: skip computing base weight gradients (only compute dinp for gradient flow)
    const bool lora_only = rs.is_lora_only_mode();

    const BackwardBlockHook* hook_ptr = hook ? &hook : nullptr;

    bool last_step = micro_step == grad_accum_steps - 1;

    // Copy targets to device
    {
        NvtxRange r{"copy-targets"};
        // Make sure targets buffer is no longer needed by previous step
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.BackwardDone, 0));
        const std::size_t target_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice, rs.side_stream()));
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, rs.side_stream()));
    }

    // Initialize gradients on first micro-step
    if (micro_step == 0) {
        NvtxRange r{"zero-gradients"};
        // Zero losses and valid token count
        fill_zero(rs.Losses, main_stream);
        fill_zero(rs.ValidTokenCount, main_stream);
        mGrads->start_micro_step(rs.side_stream(), micro_step, grad_accum_steps);
        CUDA_CHECK(cudaEventRecord(rs.side_stream_event(), rs.side_stream()));
    } else {
        mGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);
    }

    // Zero the gradient of final layer norm output
    fill_zero(rs.non_block_gradients().d_ln_final, main_stream);
    // Reset the last layer residual-stream gradient (filled by final-norm backward).
    fill_zero(rs.simplified_grads((int)L - 1).d_res_ffn, main_stream);

    // Backward through LM head
    backward_lmhead(B, T, micro_step, grad_accum_steps, comm);

    if (last_step) {
        reduce_loss(B, T, comm);
        // Aggregate valid-token count across ranks (loss is reduced with ncclAvg)
        comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, main_stream);
    }

    // Backward through final norm
    bool accumulate;
    cudaStream_t fetch_stream = rs.side_stream();
    auto& d_lnf_w = mGrads->get_final_norm_full(main_stream, comm, accumulate);
    mWeights->gather_final_norm(comm, fetch_stream);
    mWeights->gather_final_norm(comm, main_stream);
    auto& lnf_weight = mWeights->get_final_norm(main_stream);
    rmsnorm_backward(rs.simplified_grads((int)L - 1).d_res_ffn,
                     d_lnf_w,
                     rs.scratch().rmsnorm_scratch,
                     rs.simplified_grads((int)L - 1).d_res_ffn,
                     rs.non_block_gradients().d_ln_final,
                     rs.get_final_residual(),
                     lnf_weight,
                     rs.non_block_activations().ln_final_rstd,
                     rs.has_grad_quants() ? rs.simplified_quant_grads().d_res_ffn.abs_max() : nullptr,
                     (int)B, (int)T, (int)C,
                     rs.DeviceProp,
                     main_stream,
                     /*skip_weight_grad=*/lora_only);
    mWeights->release_final_norm(main_stream);
    mGrads->notify_final_norm(main_stream, comm);

    // Backward through blocks
    if (mOptions.offload_residuals && L > 1) {
        rs.fetch_residual(static_cast<int>(L - 2), fetch_stream);
    }
    mWeights->gather_block(L - 1, comm, fetch_stream);
    // Hooks can be captured safely as long as their topology is stable for the run (e.g. LoRA hooks).
    // Disable per-layer graph capture when we're inside a full-step graph capture.
    const bool use_cuda_graphs = mOptions.use_cuda_graphs;
    if (use_cuda_graphs) {
        rs.configure_backward_graphs(/*hooked=*/hook_ptr != nullptr);
    }
    for (int l = L - 1; l >= 0; --l) {
        // Prefetch previous block
        if (l > 0) {
            if (l > 1) {
                rs.fetch_residual(l - 2, fetch_stream);
            }
            mWeights->gather_block(l - 1, comm, fetch_stream);
        }

        auto& block_weights = mWeights->get_block(l, main_stream);
        auto& block_grads = mGrads->get_block_full(l, main_stream, comm, accumulate);
        auto& block_acts = rs.get_block_activations(l);
        auto& block_d_acts = rs.get_block_gradients(l);

        // Recompute if needed
        Tensor& residual = l == 0 ? rs.non_block_activations().encoded :
                                    rs.get_residual(l - 1, main_stream);
        detail::trace_or_execute_cuda_graph_with_stack([&]() {
            recompute_block(l, block_weights, block_acts, residual);
            backward_block(l, accumulate, block_weights, block_grads, block_acts, block_d_acts, hook_ptr);
        }, main_stream, rs.backward_block_graph(l, accumulate), use_cuda_graphs,
           rs.Stack, rs.backward_block_stack_checkpoint(l, accumulate));

        // LN1 backward
        {
            auto& a = rs.simplified_acts(l);
            auto& da = rs.simplified_grads(l);
            Tensor* d_ln1_w = nullptr;
            if constexpr (requires { block_grads.ln1_grads.d_weight; }) {
                d_ln1_w = &block_grads.ln1_grads.d_weight;
            } else if constexpr (requires { block_grads.ln1.d_weight; }) {
                d_ln1_w = &block_grads.ln1.d_weight;
            }
            if (!d_ln1_w) {
                throw std::logic_error("ModularTransformerModel::backward_with_hook: LN1 weight gradients not available for this block type");
            }
            if (l > 0) {
                auto& prev_da = rs.simplified_grads(l - 1);
                rmsnorm_backward(prev_da.d_res_ffn,
                                 *d_ln1_w,
                                 rs.scratch().rmsnorm_scratch,
                                 da.d_res_att,
                                 da.d_ln1,
                                 residual,
                                 block_weights.ln1.weight,
                                 a.ln1_rstd,
                                 rs.has_grad_quants() ? rs.simplified_quant_grads().d_res_ffn.abs_max() : nullptr,
                                 (int)B, (int)T, (int)C,
                                 rs.DeviceProp,
                                 main_stream,
                                 /*skip_weight_grad=*/lora_only);
            } else {
                rmsnorm_backward(rs.non_block_gradients().d_embeddings,
                                 *d_ln1_w,
                                 rs.scratch().rmsnorm_scratch,
                                 da.d_res_att,
                                 da.d_ln1,
                                 residual,
                                 block_weights.ln1.weight,
                                 a.ln1_rstd,
                                 nullptr,
                                 (int)B, (int)T, (int)C,
                                 rs.DeviceProp,
                                 main_stream,
                                 /*skip_weight_grad=*/lora_only);
            }
        }

        mWeights->release_block(l, main_stream);
        mGrads->notify_block(l, main_stream, comm);

        if (l > 0) {
            rs.release_residual(l - 1, main_stream);
        }
    }

    // Embedding backward: skip in LoRA-only mode (embeddings are frozen)
    if (!lora_only) {
        auto& d_emb = mGrads->get_embeddings_full(main_stream, comm, accumulate);
        encoder_backward(d_emb,
                         rs.scratch().encoder_bwd_scratch,
                         rs.scratch().encoder_bwd_indices,
                         rs.scratch().encoder_bwd_info,
                         rs.non_block_gradients().d_embeddings,
                         rs.Inputs,
                         inputs,
                         (int)B, (int)T, (int)C,
                         static_cast<unsigned int>(mOptimizerRNG()),
                         main_stream,
                         rs.side_stream_event(),
                         rs.side_stream());
        mGrads->notify_embeddings(main_stream, comm);
    }

    // Finalize micro-step
    mGrads->end_micro_step(main_stream, comm);
    CUDA_CHECK(cudaEventRecord(rs.BackwardDone, main_stream));
    // Ensure the host-side target buffer is safe to reuse.
    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
}

template<typename Block>
void ModularTransformerModel<Block>::update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                                             int t, float epsilon, float weight_decay, float grad_clip) {
    NVTX_RANGE_FN();

    // Only 8-bit AdamW optimizer is supported
    if (!mAdamW8BitState) {
        throw std::logic_error("ModularTransformerModel::update() but no 8-bit optimizer state available");
    }
    update_adamw_8bit(comm, learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_clip);
}

template<typename Block>
void ModularTransformerModel<Block>::update_adamw_8bit(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                                                        int t, float epsilon, float weight_decay, float grad_clip) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    auto& state = *mAdamW8BitState;

    mWeights->begin_optimizer(rs.Stack, main_stream);

    // Calculate gradient norm - grad_scale is kept on device for CUDA graph compatibility
    calculate_gradient_norm(comm, grad_clip);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;

    // Lazy initialization of 8-bit state tensors
    // We need to count total parameters first
    if (!state.initialized) {
        constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
        size_t total_params = 0;
        size_t state_elems = 0;

        auto add_tensor = [&](size_t n) {
            total_params += n;
            // Ensure each tensor starts at a block boundary in the combined state buffers.
            state_elems = (state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state_elems += n;
        };
        
        // Count embedding parameters
        add_tensor(mWeights->get_master_embeddings().nelem());

        // Count final norm parameters
        add_tensor(mWeights->get_master_final_norm().nelem());

        // Count block parameters (must match order in update phase)
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            mWeights->fetch_master_block(i, main_stream);
            auto& bw = mWeights->get_master_block(i, main_stream);
            
            // Count block parameters based on block type
            if constexpr (requires { bw.ln1.weight; }) {
                add_tensor(bw.ln1.weight.nelem());
            }
            if constexpr (requires { bw.ln2.weight; }) {
                add_tensor(bw.ln2.weight.nelem());
            }
            if constexpr (requires { bw.attention.qkv_weight; }) {
                add_tensor(bw.attention.qkv_weight.nelem());
                if constexpr (requires { bw.attention.out_weight; }) {
                    add_tensor(bw.attention.out_weight.nelem());
                }
                if constexpr (requires { bw.attention.q_norm_weight; bw.attention.k_norm_weight; }) {
                    if (bw.attention.q_norm_weight.has_value()) {
                        add_tensor(bw.attention.q_norm_weight->nelem());
                    }
                    if (bw.attention.k_norm_weight.has_value()) {
                        add_tensor(bw.attention.k_norm_weight->nelem());
                    }
                }
            }
            if constexpr (has_mlp_weights<typename Block::Weights>::value) {
                add_tensor(bw.mlp_up_weight.nelem());
                add_tensor(bw.mlp_down_weight.nelem());
            }
            
            mWeights->release_master_block(i, main_stream, rs.side_stream());
        }

        // Count LM head parameters (if not tied) - must be after blocks to match update order
        if (!mConfig.TiedWordEmbeddings) {
            add_tensor(mWeights->get_master_lm_head().nelem());
        }

        state.total_params = total_params;
        state.total_state_elems = state_elems;
        state.num_blocks = (state.total_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Determine allocation location based on offload options
        // For adamw8bit, we only support zero-copy offloading (pinned memory accessed directly by GPU)
        // because the 8-bit kernel needs direct GPU access to the state tensors
        state.offload_state = mOptions.offload_optimizer;
        state.use_zero_copy = mOptions.use_zero_copy;

        EAllocationType alloc_kind = EAllocationType::ON_DEVICE;
        if (state.offload_state) {
            if (state.use_zero_copy) {
                // Zero-copy: allocate in pinned host memory, GPU accesses directly
                alloc_kind = mOptions.get_offload_alloc();
            } else {
                // Non-zero-copy offload not supported for adamw8bit (would need double-buffering)
                // Fall back to device allocation with a warning
                // Note: This could be implemented with double-buffering in the future
                alloc_kind = EAllocationType::ON_DEVICE;
            }
        }

        // Allocate 8-bit state tensors
        state.state1 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state1", alloc_kind, {(long)state.total_state_elems});
        state.state2 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state2", alloc_kind, {(long)state.total_state_elems});
        state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax1", alloc_kind, {(long)state.num_blocks});
        state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax2", alloc_kind, {(long)state.num_blocks});

        // Initialize state tensors
        init_adamw8bit_state(
            reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()),
            reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()),
            state.absmax1.template get<float>(),
            state.absmax2.template get<float>(),
            state.total_state_elems, main_stream
        );

        state.initialized = true;
    }
    
    // Track offset into combined state tensors
    size_t state_offset = 0;
    constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
    
    // Helper lambda for 8-bit update of a single tensor
    auto run_8bit_update = [&](const char* name, Tensor& val, const Tensor& grad, float wd) {
        // Align each tensor start to a block boundary. The single-tensor kernel assumes block 0
        // corresponds to the first element of the passed-in pointers.
        state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        size_t n = val.nelem();
        size_t block_offset = state_offset / BLOCK_SIZE;

        unsigned char* s1 = reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()) + state_offset;
        unsigned char* s2 = reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()) + state_offset;
        float* am1 = state.absmax1.template get<float>() + block_offset;
        float* am2 = state.absmax2.template get<float>() + block_offset;
        float* q1 = state.quantiles1.template get<float>();
        float* q2 = state.quantiles2.template get<float>();

        // Call the appropriate 8-bit update based on tensor dtype
        // grad_scale is passed as device pointer for CUDA graph compatibility
        if (val.DType == ETensorDType::FP32) {
            if (grad.DType == ETensorDType::FP32) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad.template get<float>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, main_stream
                );
            } else if (grad.DType == ETensorDType::BF16) {
                // Mixed precision: FP32 master weights with BF16 gradients.
                adamw_update_8bit(
                    val.template get<float>(),
                    grad.template get<nv_bfloat16>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, main_stream
                );
            } else {
                throw std::runtime_error(std::string("adamw8bit: unsupported grad dtype for ") + name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad.DType != ETensorDType::BF16) {
                throw std::runtime_error(std::string("adamw8bit: unsupported grad dtype for ") + name);
            }
            adamw_update_8bit(
                val.template get<nv_bfloat16>(),
                grad.template get<nv_bfloat16>(),
                s1, s2, n,
                learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                q1, q2, am1, am2, main_stream
            );
        } else {
            throw std::runtime_error(std::string("adamw8bit: unsupported dtype for ") + name);
        }

        state_offset += n;
        if (state_offset > state.total_state_elems) {
            throw std::runtime_error("adamw8bit: state buffer overflow (layout mismatch).");
        }
    };

    // Update embeddings
    run_8bit_update("embeddings", mWeights->get_master_embeddings(), 
                    mGrads->get_embeddings_shard(main_stream), weight_decay);

    // Update final norm
    run_8bit_update("final_norm", mWeights->get_master_final_norm(),
                    mGrads->get_final_norm_shard(main_stream), 0.f);

    // Update blocks
    for (int i = 0; i < mConfig.NumLayers; ++i) {
        mWeights->fetch_master_block(i, comm.stream());
        auto& bw = mWeights->get_master_block(i, main_stream);
        auto& bg = mGrads->get_block_shard(i, main_stream);

        // Norm weights (no weight decay)
        if constexpr (requires { bw.ln1.weight; }) {
            if constexpr (requires { bg.ln1_grads.d_weight; }) {
                run_8bit_update("ln1.weight", bw.ln1.weight, bg.ln1_grads.d_weight, 0.f);
            } else if constexpr (requires { bg.ln1.d_weight; }) {
                run_8bit_update("ln1.weight", bw.ln1.weight, bg.ln1.d_weight, 0.f);
            }
        }
        if constexpr (requires { bw.ln2.weight; }) {
            if constexpr (requires { bg.ln2_grads.d_weight; }) {
                run_8bit_update("ln2.weight", bw.ln2.weight, bg.ln2_grads.d_weight, 0.f);
            } else if constexpr (requires { bg.ln2.d_weight; }) {
                run_8bit_update("ln2.weight", bw.ln2.weight, bg.ln2.d_weight, 0.f);
            }
        }

        // Attention weights
        if constexpr (requires { bw.attention.qkv_weight; }) {
            if constexpr (requires { bg.attention_grads.d_qkv_weight; }) {
                run_8bit_update("attn.qkv_weight", bw.attention.qkv_weight, bg.attention_grads.d_qkv_weight, weight_decay);
            } else if constexpr (requires { bg.attention.d_qkv_weight; }) {
                run_8bit_update("attn.qkv_weight", bw.attention.qkv_weight, bg.attention.d_qkv_weight, weight_decay);
            }
            if constexpr (requires { bw.attention.out_weight; }) {
                if constexpr (requires { bg.attention_grads.d_out_weight; }) {
                    run_8bit_update("attn.out_weight", bw.attention.out_weight, bg.attention_grads.d_out_weight, weight_decay);
                } else if constexpr (requires { bg.attention.d_out_weight; }) {
                    run_8bit_update("attn.out_weight", bw.attention.out_weight, bg.attention.d_out_weight, weight_decay);
                }
            }
            if constexpr (requires { bw.attention.q_norm_weight; bw.attention.k_norm_weight; }) {
                if (bw.attention.q_norm_weight.has_value()) {
                    if constexpr (requires { bg.attention_grads.d_q_norm_weight; }) {
                        if (bg.attention_grads.d_q_norm_weight.has_value()) {
                            run_8bit_update("attn.q_norm_weight",
                                            bw.attention.q_norm_weight.value(),
                                            bg.attention_grads.d_q_norm_weight.value(),
                                            0.f);
                        }
                    }
                }
                if (bw.attention.k_norm_weight.has_value()) {
                    if constexpr (requires { bg.attention_grads.d_k_norm_weight; }) {
                        if (bg.attention_grads.d_k_norm_weight.has_value()) {
                            run_8bit_update("attn.k_norm_weight",
                                            bw.attention.k_norm_weight.value(),
                                            bg.attention_grads.d_k_norm_weight.value(),
                                            0.f);
                        }
                    }
                }
            }
        }

        // MLP weights (dense blocks only)
        if constexpr (has_mlp_weights<typename Block::Weights>::value) {
            run_8bit_update("mlp_up_weight", bw.mlp_up_weight, bg.d_mlp_up_weight, weight_decay);
            run_8bit_update("mlp_down_weight", bw.mlp_down_weight, bg.d_mlp_down_weight, weight_decay);
        }

        mWeights->release_master_block(i, main_stream, rs.side_stream());
        CUDA_CHECK(cudaEventRecord(rs.layer_update_done(i), main_stream));
    }

    // Update LM head if not tied
    if (!mConfig.TiedWordEmbeddings) {
        run_8bit_update("lm_head", mWeights->get_master_lm_head(),
                        mGrads->get_lm_head_shard(main_stream), weight_decay);
    }

    // Update FP8 delayed scaling state: roll amax history and compute new scales
    // This must happen after the forward pass (amaxes recorded) but before next forward pass uses them
    if (rs.has_fp8_delayed_scaling()) {
        delayed_scaling_update(rs.fp8_scaling_state(), main_stream);
    }

    comm.wait_on_comms(main_stream);
    mWeights->end_optimizer(rs.Stack);
    CUDA_CHECK(cudaEventRecord(rs.OptimizerDone, main_stream));
}

template<typename Block>
void ModularTransformerModel<Block>::forward_block(int layer_idx, BlockWeights& weights,
                                                    BlockActivations& acts, Tensor& residual) {
    // Delegate to block module's forward
    // This is a stub - actual implementation chains through sub-modules
}

template<typename Block>
void ModularTransformerModel<Block>::recompute_block(int layer_idx, BlockWeights& weights,
                                                      BlockActivations& acts, Tensor& residual) {
    (void)acts;
    if (!mOptions.recompute_rmsnorm &&
        !mOptions.recompute_qkv &&
        !mOptions.recompute_attention &&
        !mOptions.recompute_ffn &&
        !mOptions.recompute_swiglu &&
        !mOptions.recompute_block) {
        return;
    }

    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;

    const int B = (int)rs.B;
    const int T = (int)rs.T;
    const int C = (int)mConfig.HiddenSize;
    const int D = (int)mConfig.IntermediateSize;
    const int Hq = (int)mConfig.NumQueryHeads;
    const int Hkv = (int)mConfig.NumKeyValHeads;
    const int Hs = (int)mConfig.head_size();
    const int qkv_channels = (int)mConfig.qkv_channels();

    auto& a = rs.simplified_acts(layer_idx);
    auto& q = rs.simplified_quant_acts(layer_idx);

    // Match legacy recompute dependency rules (LLamaModel::_recompute_block).
    const bool recompute_ln1 = mOptions.recompute_rmsnorm || mOptions.recompute_attention || mOptions.recompute_block;
    const bool recompute_ln2 = mOptions.recompute_rmsnorm || mOptions.recompute_ffn || mOptions.recompute_block;
    const bool recompute_qkv = mOptions.recompute_qkv || mOptions.recompute_attention || mOptions.recompute_block;
    const bool recompute_att = mOptions.recompute_attention || mOptions.recompute_block;
    const bool recompute_mlp_up = mOptions.recompute_ffn || mOptions.recompute_block;
    const bool recompute_swiglu = mOptions.recompute_swiglu || mOptions.recompute_ffn || mOptions.recompute_block;

    // Recompute LN1
    if (recompute_ln1) {
        if (rs.has_activation_quants() && q.ln1.DType == ETensorDType::FP8_E4M3) {
            // Use fused kernel: RMSNorm + quantization in one pass
            rmsnorm_forward_quant(q.ln1, q.ln1.scale(), a.ln1_rstd,
                                  residual, weights.ln1.weight, q.ln1.abs_max(),
                                  mConfig.RmsNormEps, B, T, C, stream);
            // Note: a.ln1 not needed in recompute mode, only quantized buffer q.ln1
        } else {
            float* ln1_abs_max_ptr = nullptr;
            if (rs.has_fp4_forward()) {
                ln1_abs_max_ptr = rs.fp4_forward_quants().ln1_global_amax;
            } else if (rs.has_activation_quants()) {
                ln1_abs_max_ptr = q.ln1.abs_max();
            }
            rmsnorm_forward(a.ln1, a.ln1_rstd, residual, weights.ln1.weight,
                            ln1_abs_max_ptr,
                            mConfig.RmsNormEps, B, T, C, stream);
        }
    }

    // QKV projection + RoPE (needed for attention backward)
    if (recompute_qkv) {
        // Note: During recomputation we DON'T use delayed scaling quantizer indices
        // because we're recomputing for backward pass, not updating scale history
        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln1 : nullptr;
        const Tensor* cached_qkv = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().qkv_weight : nullptr;

        // FP4 cached weight for recomputation
        const Tensor* fp4_data = nullptr;
        const Tensor* fp4_scales = nullptr;
        const float* fp4_amax = nullptr;
        if (mWeights->has_fp4_forward_cache()) {
            auto& fp4_cache = mWeights->fp4_weight_cache();
            fp4_data = &fp4_cache.qkv_weight.data;
            fp4_scales = &fp4_cache.qkv_weight.scales;
            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 0;
        }

        detail::recipe_forward_matmul(
            *mRecipe, a.qkv, a.ln1, weights.attention.qkv_weight,
            weights.attention.qkv_bias.has_value() ? &weights.attention.qkv_bias.value() : nullptr,
            rs, B, T, C, qkv_channels,
            layer_idx, modules::MatmulOp::QKV,
            inp_quant, cached_qkv, /*delayed_quantizer_idx=*/-1, stream,
            fp4_data, fp4_scales, fp4_amax);

        if (weights.attention.q_norm_weight.has_value() && weights.attention.k_norm_weight.has_value()) {
            const int q_rows = Hq * Hs;
            qkv_head_rmsnorm_forward(
                a.qkv, a.q_rstd, weights.attention.q_norm_weight.value(),
                mConfig.RmsNormEps,
                B, T, qkv_channels,
                /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
                stream
            );
            qkv_head_rmsnorm_forward(
                a.qkv, a.k_rstd, weights.attention.k_norm_weight.value(),
                mConfig.RmsNormEps,
                B, T, qkv_channels,
                /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
                stream
            );
        }

        // RoPE: operates in-place on a.qkv (can't fuse with quantization here)
        if (mOptions.use_fused_rope) {
            rope_fused_forward(a.qkv, a.qkv, rs.PositionIDs.template get<int>(),
                               nullptr, mConfig.RopeTheta, B, T, Hq, Hkv, Hs, stream);
        } else {
            rope_forward(a.qkv, a.qkv, rs.non_block_activations().freq_cis,
                         rs.PositionIDs.template get<int>(),
                         nullptr, B, T, Hq, Hkv, Hs, stream);
        }
    }

    // Attention forward (FlashAttention): recompute att + lse for backward.
    if (recompute_att) {
        attention_forward_cudnn(a.att, a.lse, a.qkv, rs.CuBlasWorkspace,
                                rs.CudnnHandle, B, T, Hq, Hkv, Hs, stream);

        // Compute abs_max for attention output (used by FP8/FP4 recipes for output projection quantization)
        if (rs.has_fp8_forward()) {
            abs_max(rs.fp8_forward_quants().att.abs_max(), a.att, (long)a.att.nelem(), rs.DeviceProp, stream);
        } else if (rs.has_fp4_forward()) {
            abs_max(rs.fp4_forward_quants().att_global_amax, a.att, (long)a.att.nelem(), rs.DeviceProp, stream);
        } else if (rs.has_activation_quants()) {
            abs_max(q.att.abs_max(), a.att, (long)a.att.nelem(), rs.DeviceProp, stream);
        }

        // Match legacy: only recompute att_out when recomputing the whole block (needed to rebuild residual_att/LN2).
        if (mOptions.recompute_block) {
            // Note: During recomputation we DON'T use delayed scaling quantizer indices
            Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().att : nullptr;
            const Tensor* cached_o = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().o_weight : nullptr;

            // FP4 cached weight for recomputation
            const Tensor* fp4_data = nullptr;
            const Tensor* fp4_scales = nullptr;
            const float* fp4_amax = nullptr;
            if (mWeights->has_fp4_forward_cache()) {
                auto& fp4_cache = mWeights->fp4_weight_cache();
                fp4_data = &fp4_cache.o_weight.data;
                fp4_scales = &fp4_cache.o_weight.scales;
                fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 1;
            }

            detail::recipe_forward_matmul(
                *mRecipe, a.att_out, a.att, weights.attention.out_weight,
                nullptr,
                rs, B, T, Hq * Hs, C,
                layer_idx, modules::MatmulOp::AttnOut,
                inp_quant, cached_o, /*delayed_quantizer_idx=*/-1, stream,
                fp4_data, fp4_scales, fp4_amax);
        }
    }

    // Recompute LN2
    if (recompute_ln2) {
        float* ln2_abs_max_ptr = nullptr;
        if (rs.has_fp4_forward()) {
            ln2_abs_max_ptr = rs.fp4_forward_quants().ln2_global_amax;
        } else if (rs.has_activation_quants()) {
            ln2_abs_max_ptr = q.ln2.abs_max();
        }
        if (mOptions.recompute_block) {
            fused_residual_rmsnorm_forward(
                a.residual_att,
                a.ln2,
                a.ln2_rstd,
                residual,
                a.att_out,
                weights.ln2.weight,
                ln2_abs_max_ptr,
                mConfig.RmsNormEps,
                B * T,
                C,
                stream
            );
        } else {
            rmsnorm_forward(a.ln2, a.ln2_rstd, a.residual_att, weights.ln2.weight,
                            ln2_abs_max_ptr,
                            mConfig.RmsNormEps, B, T, C, stream);
        }
    }

    // Recompute MLP-up (gate+up) if needed.
    if (recompute_mlp_up) {
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            if (rs.ffn_temps_on_stack()) {
                if (a.mlp_up.Data == nullptr) rs.temp_acquire(a.mlp_up);
                if (a.swiglu.Data == nullptr) rs.temp_acquire(a.swiglu);
            }
            // Note: During recomputation we DON'T use delayed scaling quantizer indices
            Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln2 : nullptr;
            const Tensor* cached_mlp_up = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_up_weight : nullptr;

            // FP4 cached weight for recomputation
            const Tensor* fp4_data = nullptr;
            const Tensor* fp4_scales = nullptr;
            const float* fp4_amax = nullptr;
            if (mWeights->has_fp4_forward_cache()) {
                auto& fp4_cache = mWeights->fp4_weight_cache();
                fp4_data = &fp4_cache.mlp_up_weight.data;
                fp4_scales = &fp4_cache.mlp_up_weight.scales;
                fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 2;
            }

            detail::recipe_forward_matmul(
                *mRecipe, a.mlp_up, a.ln2, weights.mlp_up_weight,
                nullptr,
                rs, B, T, C, 2 * D,
                layer_idx, modules::MatmulOp::MLPUp,
                inp_quant, cached_mlp_up, /*delayed_quantizer_idx=*/-1, stream,
                fp4_data, fp4_scales, fp4_amax);
        }
    }

    // Recompute SwiGLU if needed.
    if (recompute_swiglu) {
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            if (rs.ffn_temps_on_stack()) {
                if (a.swiglu.Data == nullptr) rs.temp_acquire(a.swiglu);
            }
            // Determine abs_max pointer for swiglu output (for FP8 quantization in MLP down)
            float* swiglu_abs_max_ptr = nullptr;
            if (rs.has_fp8_forward()) {
                swiglu_abs_max_ptr = rs.fp8_forward_quants().swiglu.abs_max();
            } else if (rs.has_fp4_forward()) {
                swiglu_abs_max_ptr = rs.fp4_forward_quants().swiglu_global_amax;
            } else if (rs.has_activation_quants()) {
                swiglu_abs_max_ptr = q.swiglu.abs_max();
            }
            if (mRecipe && mRecipe->requires_scaled_swiglu()) {
                // Recipe-driven scaled SwiGLU recompute
                modules::SwiGLUContext ctx;
                ctx.out = &a.swiglu;
                ctx.scale_out = &a.swiglu_scale;
                ctx.inp = &a.mlp_up;
                ctx.abs_max_out = swiglu_abs_max_ptr;
                ctx.B = B;
                ctx.T = T;
                ctx.D = D;
                ctx.stream = stream;
                mRecipe->swiglu_forward(ctx);
            } else {
                swiglu_forward(a.swiglu, a.mlp_up, swiglu_abs_max_ptr, B, T, D, stream);
            }
        }
    }
}

template<typename Block>
void ModularTransformerModel<Block>::backward_block(int layer_idx, bool accumulate, BlockWeights& weights,
                                                     BlockGradients& grads, BlockActivations& acts,
                                                     typename ModularRunState<Block>::BlockGradients& d_acts,
                                                     const BackwardBlockHook* hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;

    // Dimensions
    const int B = (int)rs.B;
    const int T = (int)rs.T;
    const int C = (int)mConfig.HiddenSize;
    const int D = (int)mConfig.IntermediateSize;
    const int Hq = (int)mConfig.NumQueryHeads;
    const int Hkv = (int)mConfig.NumKeyValHeads;
    const int Hs = (int)mConfig.head_size();
    const int qkv_channels = (int)mConfig.qkv_channels();
    // Determine if this layer should use FP4 quantization
    const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
    const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
    const bool in_skip_range = (layer_idx < skip_first) || (layer_idx >= mConfig.NumLayers - skip_last);
    const bool allow_fp4_layer = !in_skip_range;

    // Use the simplified activation/gradient buffers (the full modular per-module
    // activation capture is still being wired up).
    auto& a = rs.simplified_acts(layer_idx);
    auto& da = rs.simplified_grads(layer_idx);
    auto& qa = rs.simplified_quant_acts(layer_idx);
    auto& qg = rs.simplified_quant_grads();

    // Keep compilation valid for non-dense block types (MoE/hybrid), even if the
    // simplified backward path isn't implemented for them yet.
    constexpr bool kDenseLike =
        requires(BlockWeights w) {
            w.ln2.weight;
            w.attention.qkv_weight;
            w.attention.out_weight;
            w.mlp_up_weight;
            w.mlp_down_weight;
        } &&
        requires(BlockGradients g) {
            g.ln2_grads.d_weight;
            g.attention_grads.d_qkv_weight;
            g.attention_grads.d_out_weight;
            g.d_mlp_up_weight;
            g.d_mlp_down_weight;
        };

    if constexpr (!kDenseLike) {
        (void)weights;
        (void)grads;
        (void)acts;
        (void)d_acts;
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeLayerBackward);
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward);
        throw std::runtime_error("ModularTransformerModel::backward_block: simplified backward is not implemented for this block type");
    } else {
        auto with_ctx = [&](const char* stage, const auto& fn) {
            try {
                fn();
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    "ModularTransformerModel::backward_block layer " + std::to_string(layer_idx) + " (" + stage + "): " + e.what());
            }
        };

        // LoRA-only mode: skip computing base weight gradients (only compute dinp for gradient flow)
        const bool lora_only = rs.is_lora_only_mode();

        // Hooks: layer start
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeLayerBackward);

        // -------------------- MLP backward --------------------
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPDownBackward);

        // In full-block recompute mode, keep the large FFN backward intermediates stack-backed.
        const bool stack_large_bwd_temps = rs.large_bwd_temps_on_stack();
        Tensor saved_d_mlp_up{};
        if (stack_large_bwd_temps) {
            if (da.d_swiglu.Data == nullptr) rs.temp_acquire(da.d_swiglu);
            // Reuse the (recomputed) mlp_up buffer in-place for d_mlp_up (matches legacy).
            saved_d_mlp_up = da.d_mlp_up;
            da.d_mlp_up = a.mlp_up;
        }

        // MLP down: swiglu -> mlp_down_weight -> d_res_ffn
        with_ctx("mlp_down:qmm", [&]() {
            if (mRecipe->handles_backward_matmul()) {
                // Recipe-driven backward (FP8 HYBRID or MXFP8)
                modules::MatmulContext ctx;
                ctx.dinp = &da.d_swiglu;
                ctx.dweight = &grads.d_mlp_down_weight;
                ctx.dbias = nullptr;
                ctx.dout = &da.d_res_ffn;
                ctx.inp = &a.swiglu;
                ctx.weight = &weights.mlp_down_weight;
                ctx.inp_quant = &qa.swiglu;
                ctx.dout_quant = &qg.d_res_ffn;
                ctx.bias_buffer = nullptr;
                ctx.B = (int)B;
                ctx.T = (int)T;
                ctx.C_in = (int)D;
                ctx.C_out = (int)C;
                ctx.run_state = &rs;
                ctx.stream = stream;
                ctx.layer_idx = layer_idx;
                ctx.op = modules::MatmulOp::MLPDown;
                ctx.accumulate = accumulate;
                ctx.skip_weight_grad = lora_only;
                ctx.seed = static_cast<unsigned int>(mOptimizerRNG());

                // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
                if (mWeights->has_fp4_dgrad_cache()) {
                    auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                    ctx.cached_fp4_data = &fp4_t.mlp_down_weight.data;
                    ctx.cached_fp4_scales = &fp4_t.mlp_down_weight.scales;
                    // Use transposed amax (separate from forward) for dgrad
                    ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 3;
                }

                mRecipe->backward_matmul(ctx);
            } else {
                // BF16/FP4 backward (used by NVFP4Recipe, NVFP4SimpleRecipe)
                detail::backward_qmm(
                    /*dinp=*/da.d_swiglu,
                    /*dweight=*/grads.d_mlp_down_weight, /*dbias=*/std::nullopt,
                    /*dout=*/da.d_res_ffn, /*dout_q=*/qg.d_res_ffn,
                    /*inp=*/a.swiglu, /*inp_q=*/qa.swiglu,
                    /*weight=*/weights.mlp_down_weight, /*bias_buffer=*/std::nullopt,
                    /*accumulate_gradient=*/accumulate,
                    rs,
                    /*B=*/B, /*T=*/T, /*C(in)=*/D, /*OC(out)=*/C,
                    /*reuse_inp_quant=*/false,
                    /*allow_fp4=*/allow_fp4_layer,
                    static_cast<unsigned int>(mOptimizerRNG()),
                    stream,
                    /*skip_weight_grad=*/lora_only
                );
            }
        });

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPDownBackward);

        // SwiGLU backward: d_mlp_up from d_swiglu and saved pre-activation (mlp_up)
        with_ctx("swiglu", [&]() {
            if (mRecipe && mRecipe->requires_scaled_swiglu()) {
                // Recipe-driven scaled SwiGLU backward
                modules::SwiGLUContext ctx;
                ctx.dinp = &da.d_mlp_up;
                ctx.dout = &da.d_swiglu;
                ctx.inp = &a.mlp_up;
                ctx.scale = &a.swiglu_scale;
                ctx.abs_max_out = rs.has_grad_quants() ? qg.d_mlp_up.abs_max() : nullptr;
                ctx.B = B;
                ctx.T = T;
                ctx.D = D;
                ctx.stream = stream;
                mRecipe->swiglu_backward(ctx);
            } else {
                swiglu_backward(da.d_mlp_up, da.d_swiglu, a.mlp_up,
                                rs.has_grad_quants() ? qg.d_mlp_up.abs_max() : nullptr,
                                B, T, D, stream);
            }
        });

        if (stack_large_bwd_temps) {
            rs.temp_free(da.d_swiglu);
            // We no longer need swiglu activations after swiglu_backward.
            if (rs.ffn_temps_on_stack()) {
                rs.temp_free(a.swiglu);
            }
        }

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPUpBackward);

        // MLP up: ln2 -> mlp_up_weight -> d_mlp_up
        with_ctx("mlp_up:qmm", [&]() {
            if (mRecipe->handles_backward_matmul()) {
                // Recipe-driven backward (FP8 HYBRID or MXFP8)
                modules::MatmulContext ctx;
                ctx.dinp = &da.d_ln2;
                ctx.dweight = &grads.d_mlp_up_weight;
                ctx.dbias = nullptr;
                ctx.dout = &da.d_mlp_up;
                ctx.inp = &a.ln2;
                ctx.weight = &weights.mlp_up_weight;
                ctx.inp_quant = &qa.ln2;
                ctx.dout_quant = &qg.d_mlp_up;
                ctx.bias_buffer = nullptr;
                ctx.B = (int)B;
                ctx.T = (int)T;
                ctx.C_in = (int)C;
                ctx.C_out = (int)(2 * D);
                ctx.run_state = &rs;
                ctx.stream = stream;
                ctx.layer_idx = layer_idx;
                ctx.op = modules::MatmulOp::MLPUp;
                ctx.accumulate = accumulate;
                ctx.skip_weight_grad = lora_only;
                ctx.seed = static_cast<unsigned int>(mOptimizerRNG());

                // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
                if (mWeights->has_fp4_dgrad_cache()) {
                    auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                    ctx.cached_fp4_data = &fp4_t.mlp_up_weight.data;
                    ctx.cached_fp4_scales = &fp4_t.mlp_up_weight.scales;
                    // Use transposed amax (separate from forward) for dgrad
                    ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 2;
                }

                mRecipe->backward_matmul(ctx);
            } else {
                // BF16/FP4 backward (used by NVFP4Recipe, NVFP4SimpleRecipe)
                detail::backward_qmm(
                    /*dinp=*/da.d_ln2,
                    /*dweight=*/grads.d_mlp_up_weight, /*dbias=*/std::nullopt,
                    /*dout=*/da.d_mlp_up, /*dout_q=*/qg.d_mlp_up,
                    /*inp=*/a.ln2, /*inp_q=*/qa.ln2,
                    /*weight=*/weights.mlp_up_weight, /*bias_buffer=*/std::nullopt,
                    /*accumulate_gradient=*/accumulate,
                    rs,
                    /*B=*/B, /*T=*/T, /*C(in)=*/C, /*OC(out)=*/2 * D,
                    /*reuse_inp_quant=*/false,
                    /*allow_fp4=*/allow_fp4_layer,
                    static_cast<unsigned int>(mOptimizerRNG()),
                    stream,
                    /*skip_weight_grad=*/lora_only
                );
            }
        });

        // LoRA's `AfterMLPUpBackward` hook consumes `da.d_mlp_up`. In recompute-block mode
        // we may be temporarily reusing `a.mlp_up` as the gradient buffer, so run the hook
        // before restoring pointers and freeing the activation buffer.
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPUpBackward);

        if (stack_large_bwd_temps) {
            da.d_mlp_up = saved_d_mlp_up;
            if (rs.ffn_temps_on_stack()) {
                rs.temp_free(a.mlp_up);
            }
        }

        // LN2 backward: accumulates residual gradient (d_res_ffn) into d_res_att
        with_ctx("ln2", [&]() {
            rmsnorm_backward(da.d_res_att, grads.ln2_grads.d_weight, rs.scratch().rmsnorm_scratch,
                             da.d_res_ffn, da.d_ln2,
                             a.residual_att, weights.ln2.weight, a.ln2_rstd,
                             rs.has_grad_quants() ? qg.d_res_att.abs_max() : nullptr,
                             B, T, C, rs.DeviceProp, stream,
                             /*skip_weight_grad=*/lora_only);
        });

        // -------------------- Attention backward --------------------
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeAttnOutBackward);

        // Output projection backward
        with_ctx("att_out:qmm", [&]() {
            if (mRecipe->handles_backward_matmul()) {
                // Recipe-driven backward (FP8 HYBRID or MXFP8)
                modules::MatmulContext ctx;
                ctx.dinp = &da.d_att;
                ctx.dweight = &grads.attention_grads.d_out_weight;
                ctx.dbias = nullptr;
                ctx.dout = &da.d_res_att;
                ctx.inp = &a.att;
                ctx.weight = &weights.attention.out_weight;
                ctx.inp_quant = &qa.att;
                ctx.dout_quant = &qg.d_res_att;
                ctx.bias_buffer = nullptr;
                ctx.B = (int)B;
                ctx.T = (int)T;
                ctx.C_in = (int)(Hq * Hs);
                ctx.C_out = (int)C;
                ctx.run_state = &rs;
                ctx.stream = stream;
                ctx.layer_idx = layer_idx;
                ctx.op = modules::MatmulOp::AttnOut;
                ctx.accumulate = accumulate;
                ctx.skip_weight_grad = lora_only;
                ctx.seed = static_cast<unsigned int>(mOptimizerRNG());

                // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
                if (mWeights->has_fp4_dgrad_cache()) {
                    auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                    ctx.cached_fp4_data = &fp4_t.o_weight.data;
                    ctx.cached_fp4_scales = &fp4_t.o_weight.scales;
                    // Use transposed amax (separate from forward) for dgrad
                    ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 1;
                }

                mRecipe->backward_matmul(ctx);
            } else {
                // BF16/FP4 backward (used by NVFP4Recipe, NVFP4SimpleRecipe)
                detail::backward_qmm(
                    /*dinp=*/da.d_att,
                    /*dweight=*/grads.attention_grads.d_out_weight, /*dbias=*/std::nullopt,
                    /*dout=*/da.d_res_att, /*dout_q=*/qg.d_res_att,
                    /*inp=*/a.att, /*inp_q=*/qa.att,
                    /*weight=*/weights.attention.out_weight, /*bias_buffer=*/std::nullopt,
                    /*accumulate_gradient=*/accumulate,
                    rs,
                    /*B=*/B, /*T=*/T, /*C(in)=*/(Hq * Hs), /*OC(out)=*/C,
                    /*reuse_inp_quant=*/false,
                    /*allow_fp4=*/allow_fp4_layer,
                    static_cast<unsigned int>(mOptimizerRNG()),
                    stream,
                    /*skip_weight_grad=*/lora_only
                );
            }
        });

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterAttnOutBackward);

	        // FlashAttention backward: produces d_qkv (post-RoPE space)
	        with_ctx("attention_backward", [&]() {
	            if (stack_large_bwd_temps && da.d_qkv.Data == nullptr) {
	                rs.temp_acquire(da.d_qkv);
	            }
	            rs.temp_acquire(rs.scratch().cudnn_workspace);
	            const Tensor& qkv_for_attn = (a.qkv_rope.Data != nullptr) ? a.qkv_rope : a.qkv;
	            const int chunks = mOptions.attention_bwd_chunks;
	            if (chunks < 1) {
	                throw std::invalid_argument("attention_bwd_chunks must be >= 1");
	            }
	            if (chunks == 1) {
	                attention_backward_cudnn(
	                    da.d_qkv,
	                    a.lse,
	                    a.att,
	                    da.d_att,
	                    qkv_for_attn,
	                    rs.scratch().cudnn_workspace,
	                    rs.CudnnHandle,
	                    B, T, Hq, Hkv, Hs,
	                    stream
	                );
	            } else {
	                const long chunk_batch_size = div_exact(static_cast<long>(B), static_cast<long>(chunks));
	                for (int i = 0; i < chunks; ++i) {
	                    Tensor d_qkv = shard_view(da.d_qkv, i, chunks);
	                    Tensor lse = shard_view(a.lse, i, chunks);
	                    Tensor att = shard_view(a.att, i, chunks);
	                    Tensor d_att = shard_view(da.d_att, i, chunks);
	                    Tensor qkv = shard_view(qkv_for_attn, i, chunks);
	                    attention_backward_cudnn(
	                        d_qkv,
	                        lse,
	                        att,
	                        d_att,
                        qkv,
                        rs.scratch().cudnn_workspace,
                        rs.CudnnHandle,
                        chunk_batch_size, T, Hq, Hkv, Hs,
                        stream
                    );
                }
            }
            rs.temp_free(rs.scratch().cudnn_workspace);
        });

	        // RoPE backward
	        with_ctx("rope_backward", [&]() {
	            if (mOptions.use_fused_rope) {
	                rope_fused_backward(
	                    da.d_qkv, da.d_qkv,
	                    rs.PositionIDs.template get<int>(),
	                    rs.has_grad_quants() ? qg.d_qkv.abs_max() : nullptr,
	                    mConfig.RopeTheta, B, T, Hq, Hkv, Hs,
	                    stream
	                );
	            } else {
	                rope_backward(
	                    da.d_qkv, da.d_qkv,
	                    rs.non_block_activations().freq_cis,
	                    rs.PositionIDs.template get<int>(),
	                    rs.has_grad_quants() ? qg.d_qkv.abs_max() : nullptr,
	                    B, T, Hq, Hkv, Hs,
	                    stream
	                );
	            }
	        });

	        // Optional Q/K head RMSNorm backward (Qwen3-style).
	        if (weights.attention.q_norm_weight.has_value() && weights.attention.k_norm_weight.has_value()) {
	            with_ctx("qk_norm_backward", [&]() {
	                // If we didn't keep pre-RoPE QKV, convert activations back to pre-RoPE space.
	                if (a.qkv_rope.Data == nullptr) {
	                    if (mOptions.use_fused_rope) {
	                        rope_fused_backward(
	                            a.qkv, a.qkv,
	                            rs.PositionIDs.template get<int>(),
	                            nullptr,
	                            mConfig.RopeTheta, B, T, Hq, Hkv, Hs,
	                            stream
	                        );
	                    } else {
	                        rope_backward(
	                            a.qkv, a.qkv,
	                            rs.non_block_activations().freq_cis,
	                            rs.PositionIDs.template get<int>(),
	                            nullptr,
	                            B, T, Hq, Hkv, Hs,
	                            stream
	                        );
	                    }
	                }

	                const int q_rows = Hq * Hs;

	                // Weight gradients must use dy (post-attention backward, pre-qk_norm_backward_dx).
	                // Skip in LoRA-only mode since QK-norm weights are frozen.
	                if constexpr (requires { grads.attention_grads.d_q_norm_weight; grads.attention_grads.d_k_norm_weight; }) {
	                    if (!lora_only) {
	                        if (grads.attention_grads.d_q_norm_weight.has_value()) {
	                            qkv_head_rmsnorm_backward_dweight(
	                                grads.attention_grads.d_q_norm_weight.value(),
	                                da.d_qkv, a.qkv, weights.attention.q_norm_weight.value(),
	                                B, T, qkv_channels,
	                                /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
	                                /*accumulate=*/accumulate,
	                                stream
	                            );
	                        }
	                        if (grads.attention_grads.d_k_norm_weight.has_value()) {
	                            qkv_head_rmsnorm_backward_dweight(
	                                grads.attention_grads.d_k_norm_weight.value(),
	                                da.d_qkv, a.qkv, weights.attention.k_norm_weight.value(),
	                                B, T, qkv_channels,
	                                /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
	                                /*accumulate=*/accumulate,
	                                stream
	                            );
	                        }
	                    }
	                }

	                // Transform dy -> dx in-place.
	                qkv_head_rmsnorm_backward_dx(
	                    da.d_qkv, a.qkv, weights.attention.q_norm_weight.value(), a.q_rstd,
	                    B, T, qkv_channels,
	                    /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
	                    stream
	                );
	                qkv_head_rmsnorm_backward_dx(
	                    da.d_qkv, a.qkv, weights.attention.k_norm_weight.value(), a.k_rstd,
	                    B, T, qkv_channels,
	                    /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
	                    stream
	                );
	            });
	        }

        // -------------------- QKV backward --------------------
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeQKVBackward);

        with_ctx("qkv:qmm", [&]() {
            std::optional<Tensor> dbias = std::nullopt;
            if (!lora_only && weights.attention.qkv_bias.has_value() && grads.attention_grads.d_qkv_bias.has_value()) {
                dbias = grads.attention_grads.d_qkv_bias.value();
            }
            if (mRecipe->handles_backward_matmul()) {
                // Recipe-driven backward (FP8 HYBRID or MXFP8)
                modules::MatmulContext ctx;
                ctx.dinp = &da.d_ln1;
                ctx.dweight = &grads.attention_grads.d_qkv_weight;
                ctx.dbias = dbias.has_value() ? &dbias.value() : nullptr;
                ctx.dout = &da.d_qkv;
                ctx.inp = &a.ln1;
                ctx.weight = &weights.attention.qkv_weight;
                ctx.inp_quant = &qa.ln1;
                ctx.dout_quant = &qg.d_qkv;
                ctx.bias_buffer = &rs.scratch().matmul_bias_scratch;
                ctx.B = (int)B;
                ctx.T = (int)T;
                ctx.C_in = (int)C;
                ctx.C_out = (int)qkv_channels;
                ctx.run_state = &rs;
                ctx.stream = stream;
                ctx.layer_idx = layer_idx;
                ctx.op = modules::MatmulOp::QKV;
                ctx.accumulate = accumulate;
                ctx.skip_weight_grad = lora_only;
                ctx.seed = static_cast<unsigned int>(mOptimizerRNG());

                // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
                if (mWeights->has_fp4_dgrad_cache()) {
                    auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                    ctx.cached_fp4_data = &fp4_t.qkv_weight.data;
                    ctx.cached_fp4_scales = &fp4_t.qkv_weight.scales;
                    // Use transposed amax (separate from forward) for dgrad
                    ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 0;
                }

                mRecipe->backward_matmul(ctx);
            } else {
                // BF16/FP4 backward (used by NVFP4Recipe, NVFP4SimpleRecipe)
                detail::backward_qmm(
                    /*dinp=*/da.d_ln1,
                    /*dweight=*/grads.attention_grads.d_qkv_weight, /*dbias=*/dbias,
                    /*dout=*/da.d_qkv, /*dout_q=*/qg.d_qkv,
                    /*inp=*/a.ln1, /*inp_q=*/qa.ln1,
                    /*weight=*/weights.attention.qkv_weight, /*bias_buffer=*/std::make_optional(rs.scratch().matmul_bias_scratch),
                    /*accumulate_gradient=*/accumulate,
                    rs,
                    /*B=*/B, /*T=*/T, /*C(in)=*/C, /*OC(out)=*/qkv_channels,
                    /*reuse_inp_quant=*/false,
                    /*allow_fp4=*/allow_fp4_layer,
                    static_cast<unsigned int>(mOptimizerRNG()),
                    stream,
                    /*skip_weight_grad=*/lora_only
                );
            }
        });

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterQKVBackward);

        if (stack_large_bwd_temps) {
            rs.temp_free(da.d_qkv);
        }

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward);
    }
}

template<typename Block>
void ModularTransformerModel<Block>::backward_lmhead(long B, long T, int micro_step, int grad_accum_steps,
                                                      NCCLCommunicator& comm) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    const size_t C = mConfig.HiddenSize;
    const size_t V = mConfig.VocabSize;
    const size_t Vp = mConfig.VocabSize;  // Padded vocab size (same for now)
    cudaStream_t main_stream = rs.MainStream;

    // LoRA-only mode: skip computing LM head weight gradients
    const bool lora_only = rs.is_lora_only_mode();

    long nano_batches = mOptions.lmhead_chunks;
    int nano_batch_size = static_cast<int>((B * T) / nano_batches);

    // Loss normalization factor
    const float d_loss = 1.0f / static_cast<float>(B * T * grad_accum_steps);

    NvtxRange classifier_range("lm-head");
    mWeights->gather_lm_head(comm, rs.side_stream());

    // Note: Losses and ValidTokenCount are zeroed in backward_with_hook on micro_step==0
    // The fused_classifier kernel accumulates: losses[idx] -= logf(prob)
    rs.temp_acquire(rs.non_block_activations().output);

    for (int nano_step = 0; nano_step < nano_batches; nano_step++) {
        if (nano_step == 0) {
            // Ensure targets have been copied to device before fused_classifier reads them.
            CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.TransferDone, 0));
            // On the first micro-step, gradients are zeroed on the side stream; ensure that's done
            // before we start writing into any grad buffers on the main stream.
            if (micro_step == 0) {
                CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.side_stream_event(), 0));
            }
        }
        // Slice tensors for this nano-batch
        // Note: Update both Data pointer AND Sizes for correct matmul dimensions
        Tensor lnf_slice = rs.non_block_activations().ln_final;
        lnf_slice.Data = static_cast<std::byte*>(lnf_slice.Data) +
                         nano_step * nano_batch_size * C * get_dtype_size(lnf_slice.DType);
        lnf_slice.Sizes[0] = nano_batch_size;
        lnf_slice.Sizes[1] = C;
        lnf_slice.Rank = 2;

        Tensor tgt = rs.Targets;
        tgt.Data = static_cast<std::byte*>(tgt.Data) +
                   nano_step * nano_batch_size * get_dtype_size(tgt.DType);
        tgt.Sizes[0] = nano_batch_size;
        tgt.Rank = 1;

        Tensor losses = rs.Losses;
        losses.Data = static_cast<std::byte*>(losses.Data) +
                      nano_step * nano_batch_size * get_dtype_size(losses.DType);
        losses.Sizes[0] = nano_batch_size;
        losses.Rank = 1;

        Tensor dlnf_slice = rs.non_block_gradients().d_ln_final;
        dlnf_slice.Data = static_cast<std::byte*>(dlnf_slice.Data) +
                          nano_step * nano_batch_size * C * get_dtype_size(dlnf_slice.DType);
        dlnf_slice.Sizes[0] = nano_batch_size;
        dlnf_slice.Sizes[1] = C;
        dlnf_slice.Rank = 2;

        // Forward: logits = lnf @ lm_head.T
        matmul(rs.non_block_activations().output, mWeights->get_lm_head(main_stream), lnf_slice,
               std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
               V, nano_batch_size, C, EMMTranspose::TN, false, main_stream);

        // Wait for targets on first nano-step
        if (nano_step == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.TransferDone, 0));
        }

        // Fused classifier: softmax + cross-entropy + sets logits to dlogits
        fused_classifier(rs.non_block_activations().output, losses, d_loss, tgt,
                         &rs.ValidTokenCount, nano_batch_size, V, Vp, true, main_stream);

        // Wait for grad zero on first micro/nano step
        if (micro_step == 0 && nano_step == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.side_stream_event(), 0));
        }

        // Backward: d_lm_head += lnf.T @ dlogits (skip in LoRA-only mode)
        if (!lora_only) {
            bool accumulate;
            auto& d_lmhead = mGrads->get_lm_head_full(main_stream, comm, accumulate);
            accumulate |= nano_step != 0;
            matmul(d_lmhead, lnf_slice, rs.non_block_activations().output,
                   std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
                   C, V, nano_batch_size, EMMTranspose::NT, accumulate, main_stream);
            if (nano_step == nano_batches - 1) {
                mGrads->notify_lm_head(main_stream, comm);
            }
        }

        // Backward: d_lnf = dlogits @ lm_head
        matmul(dlnf_slice, mWeights->get_lm_head(main_stream), rs.non_block_activations().output,
               std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
               C, nano_batch_size, V, EMMTranspose::NN, false, main_stream);
    }

    rs.temp_free(rs.non_block_activations().output);
    mWeights->release_lm_head(main_stream);
}

template<typename Block>
void ModularTransformerModel<Block>::reduce_loss(long B, long T, NCCLCommunicator& comm) {
    NVTX_RANGE_FN();
    auto& rs = *mRunState;

    // Reduce all losses within the current GPU (across all nano-batches)
    deterministic_sum(rs.Losses.template get<float>(), rs.Losses.template get<float>(), B * T, rs.MainStream);

    // Reduce loss across GPUs to a single, final float
    comm.reduce_loss(rs.Losses.template get<float>(), rs.MainStream);

    // Copy loss to host
    CUDA_CHECK(cudaMemcpyAsync(rs.LossHost, rs.Losses.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost, rs.MainStream));
}

template<typename Block>
void ModularTransformerModel<Block>::calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip) {
    NVTX_RANGE_FN();
    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    // Ensure backward has completed before reading gradients.
    CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.BackwardDone));

    detail::trace_or_execute_cuda_graph([&]() {
        calculate_gradient_norm_impl(comm, grad_clip, main_stream);
    }, main_stream, rs.global_norm_graph(), mOptions.use_cuda_graphs);

    CUDA_CHECK(cudaEventRecord(rs.NormDone, main_stream));
}

template<typename Block>
void ModularTransformerModel<Block>::calculate_gradient_norm_impl(NCCLCommunicator& comm, float grad_clip, cudaStream_t stream) {
    auto& rs = *mRunState;

    fill_zero(rs.scratch().norm_buffer, stream);

    auto norm_squared = [&](const TensorShard& grad) {
        global_norm_squared(rs.scratch().norm_buffer, grad, grad.nelem(), rs.DeviceProp, stream);
    };

    auto& emb_grad = mGrads->get_embeddings_shard(stream);
    norm_squared(emb_grad);
    auto& head_grad = mGrads->get_lm_head_shard(stream);
    if (head_grad.Data != emb_grad.Data) {
        norm_squared(head_grad);
    }
    norm_squared(mGrads->get_final_norm_shard(stream));

    for (int i = 0; i < mConfig.NumLayers; ++i) {
        auto& g = mGrads->get_block_shard(i, stream);
        if constexpr (requires { g.ln1_grads.d_weight; }) norm_squared(TensorShard(g.ln1_grads.d_weight));
        if constexpr (requires { g.ln2_grads.d_weight; }) norm_squared(TensorShard(g.ln2_grads.d_weight));

        if constexpr (requires { g.attention_grads.d_qkv_weight; }) norm_squared(TensorShard(g.attention_grads.d_qkv_weight));
        if constexpr (requires { g.attention_grads.d_qkv_bias; }) {
            if (g.attention_grads.d_qkv_bias.has_value()) {
                norm_squared(TensorShard(g.attention_grads.d_qkv_bias.value()));
            }
        }
        if constexpr (requires { g.attention_grads.d_out_weight; }) norm_squared(TensorShard(g.attention_grads.d_out_weight));
        if constexpr (requires { g.attention_grads.d_q_norm_weight; }) {
            if (g.attention_grads.d_q_norm_weight.has_value()) {
                norm_squared(TensorShard(g.attention_grads.d_q_norm_weight.value()));
            }
        }
        if constexpr (requires { g.attention_grads.d_k_norm_weight; }) {
            if (g.attention_grads.d_k_norm_weight.has_value()) {
                norm_squared(TensorShard(g.attention_grads.d_k_norm_weight.value()));
            }
        }

        if constexpr (requires { g.d_mlp_up_weight; }) norm_squared(TensorShard(g.d_mlp_up_weight));
        if constexpr (requires { g.d_mlp_down_weight; }) norm_squared(TensorShard(g.d_mlp_down_weight));
    }

    // Reduce partial sums to a single scalar on-device.
    deterministic_sum(rs.scratch().norm_buffer.template get<float>(),
                      rs.scratch().norm_buffer.template get<float>(),
                      rs.scratch().norm_buffer.nelem(),
                      stream);

    // Cross-rank reduction.
    comm.reduce_norm(rs.scratch().norm_buffer.template get<float>(), stream);

    float total_tokens = static_cast<float>(rs.B) * static_cast<float>(rs.T)
                       * static_cast<float>(std::max(1, rs.GradAccumSteps))
                       * static_cast<float>(std::max(1, comm.world_size()));
    global_norm_sqrt(rs.scratch().norm_buffer.template get<float>(), rs.NormHost, grad_clip,
                     rs.ValidTokenCount.template get<int>(), total_tokens,
                     rs.DeviceProp, stream);
}

template<typename Block>
float ModularTransformerModel<Block>::get_loss() const {
    if (!mRunState) return 0.0f;

    // Get raw loss from IRunState base class (populated by reduce_loss)
    float raw_loss = mRunState->get_loss();

    // Normalize by valid token count (similar to LLamaModel::get_loss)
    // ValidTokenCount was reduced across ranks in backward_with_hook
    int valid_tokens;
    CUDA_CHECK(cudaMemcpy(&valid_tokens, mRunState->ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost));

    if (valid_tokens > 0) {
        // ValidTokenCount is reduced across ranks (sum). Loss is reduced with ncclAvg, so
        // divide by the average valid tokens per rank for correct mean CE.
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, mRunState->WorldSize));
        return raw_loss / avg_valid;
    } else {
        return 0.0f;
    }
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::weights() {
    return *mWeights;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_momentum() {
    // 8-bit optimizer doesn't expose state as ITensorContainer
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_momentum_scales() {
    // 8-bit optimizer doesn't use FP8 scales
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_variance() {
    // 8-bit optimizer doesn't expose state as ITensorContainer
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
ITensorContainer& ModularTransformerModel<Block>::opt_variance_scales() {
    // 8-bit optimizer doesn't use FP8 scales
    static EmptyTensorContainer empty;
    return empty;
}

template<typename Block>
std::vector<std::byte> ModularTransformerModel<Block>::rng_state() const {
    std::stringstream tmp;
    static_cast<std::ostream&>(tmp) << mOptimizerRNG;
    auto view = tmp.rdbuf()->view();
    std::vector<std::byte> state;
    state.reserve(view.size());
    std::transform(view.begin(), view.end(), std::back_inserter(state),
                   [](char c) { return static_cast<std::byte>(c); });
    return state;
}

template<typename Block>
void ModularTransformerModel<Block>::set_rng_state(const std::vector<std::byte>& state) {
    std::stringstream tmp;
    tmp.write(reinterpret_cast<const char*>(state.data()), state.size());
    static_cast<std::istream&>(tmp) >> mOptimizerRNG;
}

template<typename Block>
std::string_view ModularTransformerModel<Block>::model_type() const {
    return mConfig.model_name();
}

template<typename Block>
IRunState& ModularTransformerModel<Block>::get_run_state() const {
    if (!mRunState) {
        throw std::logic_error("ModularTransformerModel::get_run_state() called before allocate_run_state()");
    }
    // ModularRunState inherits from IRunState, so this is safe
    return *mRunState;
}

template<typename Block>
void ModularTransformerModel<Block>::allocate_run_state(const ModelOptions& options, NCCLCommunicator& comm,
                                                         int B, int T, bool allocate_optimizer) {
    NVTX_RANGE_FN();

    // Synchronize Four Over Six (4/6) setting from recipe to weight manager.
    // This ensures cached FP4 weights use the same quantization method as the recipe.
    if (mRecipe) {
        if (auto* nvfp4_recipe = dynamic_cast<recipes::NVFP4Recipe*>(mRecipe.get())) {
            mWeights->set_four_over_six(
                nvfp4_recipe->uses_four_over_six(),
                nvfp4_recipe->four_over_six_metric()
            );
        }
    }

    // B200/B300 optimization: persist FP4 base weights across steps in LoRA/frozen-base mode.
    // This avoids re-quantizing/transposing BF16 base weights every iteration, which can make
    // NVFP4 slower than FP8 on very fast datacenter GPUs.
    if (options.enable_fp4_forward) {
        mWeights->maybe_enable_fp4_persistent_cache(/*weights_static=*/options.skip_base_gradients);
    }

    // Create run state config
    typename ModularRunState<Block>::Config rs_config;
    rs_config.num_layers = mConfig.NumLayers;
    rs_config.batch_size = B;
    rs_config.seq_length = T;
    rs_config.hidden_size = mConfig.HiddenSize;
    rs_config.vocab_size = mConfig.VocabSize;
    // Activations/gradients are computed and stored in a real compute dtype (BF16/FP32).
    // Some modes (e.g. FP8 QLoRA weight storage) can set the model dtype to an FP8 type;
    // we must not allocate activations as FP8, because many kernels (e.g. abs_max) only
    // support FP32/BF16 and FP8 is handled via separate quant buffers.
    ETensorDType act_dtype = options.model_dtype.value_or(mConfig.DType);
    if (is_fp8_dtype(act_dtype)) act_dtype = ETensorDType::BF16;
    rs_config.activation_dtype = act_dtype;
    // Activation-gradient tensors stay in activation dtype; FP8 training uses separate quant buffers.
    rs_config.grad_dtype = act_dtype;
    rs_config.matmul_dtype = options.get_matmul_dtype();
    rs_config.grad_quant_dtype = options.get_grad_dtype();
    rs_config.enable_fp8_forward = options.enable_fp8_forward;
    rs_config.enable_fp8_hybrid_delayed = options.enable_fp8_hybrid;  // Auto-enable delayed scaling with hybrid mode
    rs_config.fp8_scaling_config = options.fp8_scaling_config;
    rs_config.forward_matmul_dtype = options.get_forward_matmul_dtype();
    rs_config.enable_fp4_forward = options.enable_fp4_forward;
    rs_config.enable_fp4_backward = options.enable_fp4_backward;
    rs_config.enable_fp4_hadamard = options.enable_fp4_hadamard;
    rs_config.enable_scaled_swiglu = options.enable_scaled_swiglu;
    rs_config.offload_residuals = options.offload_residuals;
    rs_config.recompute_rmsnorm = options.recompute_rmsnorm;
    rs_config.recompute_qkv = options.recompute_qkv;
    rs_config.recompute_attention = options.recompute_attention;
    rs_config.recompute_ffn = options.recompute_ffn;
    rs_config.recompute_swiglu = options.recompute_swiglu;
    rs_config.recompute_block = options.recompute_block;
    rs_config.lmhead_chunks = options.lmhead_chunks;
    rs_config.attention_bwd_chunks = options.attention_bwd_chunks;
    rs_config.lora_only_mode = options.lora_only_mode;
    rs_config.use_fused_rope = options.use_fused_rope;

    // Set block config for run state
    rs_config.block_config.hidden_size = mConfig.HiddenSize;
    rs_config.block_config.intermediate_size = mConfig.IntermediateSize;
    rs_config.block_config.num_query_heads = mConfig.NumQueryHeads;
    rs_config.block_config.num_kv_heads = mConfig.NumKeyValHeads;
    rs_config.block_config.head_size = mConfig.head_size();
    rs_config.block_config.rms_norm_eps = mConfig.RmsNormEps;
    rs_config.block_config.rope_theta = mConfig.RopeTheta;
    rs_config.block_config.max_seq_len = mConfig.MaxPositionEmbeddings;
    rs_config.block_config.use_qkv_bias = mConfig.UseQKVBias;
    if constexpr (requires { rs_config.block_config.use_qk_norm; }) {
        rs_config.block_config.use_qk_norm = mConfig.UseQKNorm;
    }

    // Set PretrainedConfig for IRunState base class.
    rs_config.pretrained_config = static_cast<const PretrainedConfig&>(mConfig);

    // Two-pass stack allocation:
    // 1. First pass with a dummy stack to measure peak temporary usage
    // 2. Second pass with a properly allocated stack
    //
    // Important: the peak must include temporaries allocated during the optimizer pass
    // (e.g., device caches for offloaded Adam moments), not just construction-time
    // stack simulations inside ModularRunState.
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    DeviceMemoryStack dummy_stack(nullptr, 1024LL * 1024 * 1024 * 1024, dev);

    // First pass - construct run state (dummy_stack is moved into mRunState->Stack).
    mRunState = std::make_unique<ModularRunState<Block>>(rs_config, dummy_stack, mAllocator);

    // Allocate gradient manager
    // In LoRA mode (skip_base_gradients), we still need a gradient manager for the backward pass
    // computation flow, but we can use minimal allocation since the gradients are discarded.
    typename ModularGradientManager<Block>::Config gm_config;
    gm_config.num_layers = mConfig.NumLayers;
    gm_config.block_config = rs_config.block_config;
    // Weight gradients are stored in model dtype (legacy parity). The CLI flag --gradient-dtype
    // refers to activation-gradient quantization for FP8 matmuls, not parameter gradients.
    gm_config.grad_dtype = options.model_dtype.value_or(mConfig.DType);
    gm_config.shard_idx = comm.rank();
    gm_config.num_shards = comm.world_size();
    gm_config.shard_gradients = options.shard_gradients;
    gm_config.use_all_to_all_reduce = options.use_all_to_all_reduce;
    gm_config.offload_grads = options.offload_grads;
    gm_config.offload_alloc = options.get_offload_alloc();
    gm_config.vocab_size = mConfig.VocabSize;
    gm_config.hidden_size = mConfig.HiddenSize;
    gm_config.tied_embeddings = mConfig.TiedWordEmbeddings;
    gm_config.skip_allocation = options.skip_base_gradients;  // Skip allocating large gradient buffers

    mGrads = std::make_unique<ModularGradientManager<Block>>(42, 0, gm_config, mAllocator);

    // Allocate 8-bit AdamW optimizer state if requested
    if (allocate_optimizer) {
        mAdamW8BitState = std::make_unique<AdamW8BitState>();

        // State tensors will be initialized lazily in the first update call
        mAdamW8BitState->initialized = false;

        // Allocate quantization maps (256 entries each)
        mAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles1", {256});
        mAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles2", {256});

        // Initialize quantization maps on host then copy to device
        std::vector<float> h_quantiles1(256), h_quantiles2(256);
        create_adamw8bit_quantiles1(h_quantiles1.data());
        create_adamw8bit_quantiles2(h_quantiles2.data());
        CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles1.Data, h_quantiles1.data(),
                              256 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles2.Data, h_quantiles2.data(),
                              256 * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Get measured usage from the stack that's now inside mRunState.
    long required_size = static_cast<long>(std::max(std::size_t(1024 * 1024), mRunState->Stack.max_utilization()));  // At least 1MB
    auto high_mark = mRunState->Stack.get_high_mark();

    // Allocate real stack and replace the dummy.
    mRunState->Stack = DeviceMemoryStack{
        mAllocator->allocate(ETensorDType::BYTE, "stack", {required_size}).Data,
        static_cast<std::size_t>(required_size),
        dev
    };
    mRunState->Stack.set_high_mark(high_mark);

    comm.barrier();
}

template<typename Block>
void ModularTransformerModel<Block>::initialize_optimizer_state(cudaStream_t stream) {
    NVTX_RANGE_FN();

    if (!mAdamW8BitState) {
        throw std::runtime_error("initialize_optimizer_state: optimizer state not allocated (call allocate_run_state first)");
    }

    auto& state = *mAdamW8BitState;
    if (state.initialized) {
        return;  // Already initialized
    }

    constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
    // Count total parameters across all weight tensors and compute a padded layout.
    size_t total_params = 0;
    size_t state_elems = 0;

    auto add_tensor = [&](size_t n) {
        total_params += n;
        state_elems = (state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        state_elems += n;
    };

    // Count embedding parameters
    add_tensor(mWeights->get_master_embeddings().nelem());

    // Count final norm parameters
    add_tensor(mWeights->get_master_final_norm().nelem());

    // Count LM head parameters (if not tied)
    if (!mConfig.TiedWordEmbeddings) {
        add_tensor(mWeights->get_master_lm_head().nelem());
    }

    // Count block parameters
    for (int i = 0; i < mConfig.NumLayers; ++i) {
        mWeights->fetch_master_block(i, stream);
        auto& bw = mWeights->get_master_block(i, stream);

        // Count block parameters based on block type
        if constexpr (requires { bw.ln1.weight; }) {
            add_tensor(bw.ln1.weight.nelem());
        }
        if constexpr (requires { bw.ln2.weight; }) {
            add_tensor(bw.ln2.weight.nelem());
        }
        if constexpr (requires { bw.attention.qkv_weight; }) {
            add_tensor(bw.attention.qkv_weight.nelem());
            if constexpr (requires { bw.attention.out_weight; }) {
                add_tensor(bw.attention.out_weight.nelem());
            }
        }
        if constexpr (has_mlp_weights<typename Block::Weights>::value) {
            add_tensor(bw.mlp_up_weight.nelem());
            add_tensor(bw.mlp_down_weight.nelem());
        }

        mWeights->release_master_block(i, stream, stream);
    }

    state.total_params = total_params;
    state.total_state_elems = state_elems;
    state.num_blocks = (state.total_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Determine allocation location based on offload options
    state.offload_state = mOptions.offload_optimizer;
    state.use_zero_copy = mOptions.use_zero_copy;

    EAllocationType alloc_kind = EAllocationType::ON_DEVICE;
    if (state.offload_state) {
        if (state.use_zero_copy) {
            alloc_kind = mOptions.get_offload_alloc();
        } else {
            alloc_kind = EAllocationType::ON_DEVICE;
        }
    }

    // Allocate 8-bit state tensors
    state.state1 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state1", alloc_kind, {(long)state.total_state_elems});
    state.state2 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state2", alloc_kind, {(long)state.total_state_elems});
    state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax1", alloc_kind, {(long)state.num_blocks});
    state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax2", alloc_kind, {(long)state.num_blocks});

    // Initialize state tensors
    init_adamw8bit_state(
        reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()),
        reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()),
        state.absmax1.template get<float>(),
        state.absmax2.template get<float>(),
        state.total_state_elems, stream
    );

    state.initialized = true;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Block>
void ModularTransformerModel<Block>::reset_full_step_graph() {
    if (mFullStepGraph) {
        if (mFullStepGraph->graph_exec) {
            CUDA_CHECK(cudaGraphExecDestroy(mFullStepGraph->graph_exec));
            mFullStepGraph->graph_exec = nullptr;
        }
        mFullStepGraph->captured = false;
    }
}

template<typename Block>
void ModularTransformerModel<Block>::train_step_graphed(
    NCCLCommunicator& comm, DataLoader& loader,
    int grad_accum_steps,
    float learning_rate, float beta_1, float beta_2, int step,
    float epsilon, float weight_decay, float grad_clip,
    bool use_graph
) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    int B = rs.batch_size();
    int T = rs.seq_length();

    // Ensure optimizer state is initialized
    if (!mAdamW8BitState || !mAdamW8BitState->initialized) {
        throw std::runtime_error("train_step_graphed: optimizer state not initialized. "
                                 "Call initialize_optimizer_state() before using graphed training.");
    }

    // Initialize full-step graph state if needed
    if (!mFullStepGraph) {
        mFullStepGraph = std::make_unique<FullStepGraphState>();
        // Allocate device buffers for optimizer parameters
        mFullStepGraph->opt_params = mAllocator->allocate(ETensorDType::FP32, "graph_opt_params", {5});
        mFullStepGraph->opt_step = mAllocator->allocate(ETensorDType::INT32, "graph_opt_step", {1});
    }

    auto& graph_state = *mFullStepGraph;

    // Check if we need to re-capture the graph due to shape change
    if (graph_state.captured &&
        (graph_state.captured_grad_accum_steps != grad_accum_steps ||
         graph_state.captured_B != B ||
         graph_state.captured_T != T)) {
        reset_full_step_graph();
    }

    // Update optimizer parameters on device (always, even for graph replay)
    // This allows dynamic learning rate schedules without re-capture
    float opt_params_host[5] = {learning_rate, beta_1, beta_2, epsilon, weight_decay};
    CUDA_CHECK(cudaMemcpyAsync(graph_state.opt_params.Data, opt_params_host,
                               5 * sizeof(float), cudaMemcpyHostToDevice, main_stream));
    CUDA_CHECK(cudaMemcpyAsync(graph_state.opt_step.Data, &step,
                               sizeof(int), cudaMemcpyHostToDevice, main_stream));

    // Non-graph path: execute normally
    if (!use_graph) {
        Tensor& inputs = get_input_buffer();
        Tensor& targets = get_target_buffer();
        Tensor& position_ids = get_position_ids_buffer();

        for (int j = 0; j < grad_accum_steps; ++j) {
            loader.load_batch(inputs, targets, &position_ids);
            forward(inputs, position_ids, comm, j);
            backward(inputs, targets, comm, grad_accum_steps, j);
        }
        update(comm, learning_rate, beta_1, beta_2, step, epsilon, weight_decay, grad_clip);
        return;
    }

    // Graph replay path (if already captured)
    if (graph_state.captured) {
        // Load all batches for this step first (data loading cannot be in the graph)
        Tensor& inputs = get_input_buffer();
        Tensor& targets = get_target_buffer();
        Tensor& position_ids = get_position_ids_buffer();

        for (int j = 0; j < grad_accum_steps; ++j) {
            loader.load_batch(inputs, targets, &position_ids);
            // Note: For proper multi-accumulation graphed training, we'd need
            // to store all batches, but for now we support grad_accum_steps=1
        }

        // Launch the captured graph
        CUDA_CHECK(cudaGraphLaunch(graph_state.graph_exec, main_stream));
        return;
    }

    // Graph capture path (first execution)
    // Note: Full graph capture including optimizer is complex due to:
    // 1. Data loading must happen outside the graph
    // 2. Multiple gradient accumulation steps would need all data pre-loaded
    // 3. NCCL collectives need special handling (cudaStreamCaptureModeRelaxed)
    //
    // For now, we capture a single forward+backward+optimizer iteration.
    // For grad_accum > 1, the user should use non-graphed mode or we'd need
    // to pre-allocate buffers for all micro-batches.

    if (grad_accum_steps > 1) {
        // Fall back to non-graphed mode for grad accumulation > 1
        // (Full support would require pre-staging all batches)
        Tensor& inputs = get_input_buffer();
        Tensor& targets = get_target_buffer();
        Tensor& position_ids = get_position_ids_buffer();

        for (int j = 0; j < grad_accum_steps; ++j) {
            loader.load_batch(inputs, targets, &position_ids);
            forward(inputs, position_ids, comm, j);
            backward(inputs, targets, comm, grad_accum_steps, j);
        }
        update(comm, learning_rate, beta_1, beta_2, step, epsilon, weight_decay, grad_clip);
        return;
    }

    // Load single batch (outside graph capture)
    Tensor& inputs = get_input_buffer();
    Tensor& targets = get_target_buffer();
    Tensor& position_ids = get_position_ids_buffer();
    loader.load_batch(inputs, targets, &position_ids);
    CUDA_CHECK(cudaStreamSynchronize(main_stream));  // Ensure data is ready

    // Begin graph capture
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeThreadLocal));

    // Execute forward + backward + optimizer
    forward(inputs, position_ids, comm, 0);
    backward(inputs, targets, comm, 1, 0);
    update(comm, learning_rate, beta_1, beta_2, step, epsilon, weight_decay, grad_clip);

    // End capture
    CUDA_CHECK(cudaStreamEndCapture(main_stream, &graph));

    // Instantiate executable graph
    CUDA_CHECK(cudaGraphInstantiate(&graph_state.graph_exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));

    // Mark as captured
    graph_state.captured = true;
    graph_state.captured_grad_accum_steps = grad_accum_steps;
    graph_state.captured_B = B;
    graph_state.captured_T = T;

    // Launch the newly captured graph
    CUDA_CHECK(cudaGraphLaunch(graph_state.graph_exec, main_stream));
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODULAR_MODEL_H
