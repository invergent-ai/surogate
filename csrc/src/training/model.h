// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_TRAINING_MODEL_H
#define SUROGATE_SRC_TRAINING_MODEL_H

#include <cstddef>
#include <memory>
#include <string_view>
#include <tuple>
#include <vector>

#include "utilities/stack.h"
#include "utilities/tensor.h"
#include "config/pretrained_config.h"
#include "modules/optimizers/optimizer_config.h"

class ITensorContainer;
class NCCLCommunicator;
class TensorAllocator;
class DataLoader;
struct RuntimeOptions;

namespace modules { class FP8ScalingState; }

typedef struct cudnnContext* cudnnHandle_t;
typedef struct cublasLtContext* cublasLtHandle_t;

class IRunState;

//! \brief Abstract model base class.
//! \details Provides access to the different underlying tensor containers.
class IModel {
public:
    virtual ~IModel() = default;
    //! \brief Runs the forward pass until just before the logit calculation
    //! \details This function is asynchronous. You need to wait on `run_state.ForwardDone`
    //! before accessing any of the results (or run subsequent work on `run_state.MainStream`).
    //! However, it is guaranteed that `inputs` have been copied to the GPU-side buffer
    //! before this function returns.
    //! Note: We do not calculate the logits here, so that we have more freedom to optimize
    //! this large matmul, e.g., calculating it in chunks, by including it in the backward pass.
    virtual void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) = 0;

    //! \brief Runs the forward pass and calculates the loss w.r.t. `targets`.
    virtual float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) = 0;

    //! \brief Runs the backward pass
    //! \details This function is asynchronous. You need to wait on `run_state.BackwardDone`
    //! before accessing any of the results (or run subsequent work on `run_state.MainStream`).
    //! However, it is guaranteed that `inputs` and `targets` have been copied to the GPU-side buffer
    //! before this function returns.
    virtual void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) = 0;

    //! \brief Runs the AdamW update step.
    //! \details Runs asynchronously, signalling completion through the OptimizerDone event.
    virtual void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon, float weight_decay, float grad_clip) = 0;

    //! \brief Runs the optimizer update with full configuration.
    //! \details Supports AdamW 8-bit and NorMuon (hybrid AdamW/NorMuon) optimizers.
    //! NorMuon uses orthogonalized momentum for 2D weights and AdamW for other parameters.
    //! Default implementation dispatches to AdamW update for backwards compatibility.
    virtual void update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step);

    //! Gets the loss of the preceding validate or backward call (forward does _not_ calculate the loss)
    virtual float get_loss() const;

    //! Gets the gradient norm of the preceding update call.
    float get_norm() const;

    //! Gets the accuracy of the preceding validate call (percentage of correct predictions)
    virtual float get_accuracy() const;

    //! Gets the tensor into which model inputs are to be placed.
    Tensor& get_input_buffer();

    //! Gets the tensor into which model targets are to be placed.
    Tensor& get_target_buffer();

    //! Gets the tensor into which model position ids are to be placed.
    Tensor& get_position_ids_buffer();

    //! Model (master) weights. Sharded.
    virtual ITensorContainer& weights() = 0;

    //! (First order) momentum. Sharded.
    virtual ITensorContainer& opt_momentum() = 0;

    //! (First order) momentum. Sharded.
    virtual ITensorContainer& opt_momentum_scales() = 0;

    //! Second order moments. Sharded.
    virtual ITensorContainer& opt_variance() = 0;

    //! Second order moments FP8 scales (if applicable). Sharded.
    virtual ITensorContainer& opt_variance_scales() = 0;

    //! Get the current RNG state
    virtual std::vector<std::byte> rng_state() const = 0;

    //! Set the RNG state from checkpoint data
    virtual void set_rng_state(const std::vector<std::byte>& state) = 0;

    //! Randomly initialize the model weights.
    virtual void init_weights(NCCLCommunicator& comm) = 0;

    //! Import the model weights from a file. This may be different than just reading into `weights()`,
    //! because it may involve dtype conversion (`allow_cast=true`), and even rearrange some data
    //! (e.g., fused vs unfused QKV)
    virtual void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) = 0;

    //! This function needs to be called after the model has been restored from a checkpoint.
    virtual void on_restore_checkpoint(NCCLCommunicator& comm) = 0;

    //! Export the model weights to a safetensors file.
    virtual void export_weights(const std::string& file_name, NCCLCommunicator& comm) = 0;

    //! Allocate run state buffers. Must be called before forward/backward.
    //! @param options Model options
    //! @param comm NCCL communicator for distributed setup
    //! @param B Batch size
    //! @param T Sequence length
    //! @param allocate_optimizer Whether to allocate optimizer state
    virtual void allocate_run_state(const struct RuntimeOptions& options, NCCLCommunicator& comm,
                                     int B, int T, bool allocate_optimizer) {
        // Default no-op for models that allocate run state in constructor
    }

    //! Save LoRA adapter to a checkpoint directory (no-op for non-LoRA models)
    virtual void save_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) {}

    //! Load LoRA adapter from a checkpoint directory (no-op for non-LoRA models)
    virtual void load_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) {}

    //! Get the model type identifier
    virtual std::string_view model_type() const = 0;

    //! Get a const reference to the model's RunState.
    virtual IRunState& get_run_state() const = 0;
};


/*!
 *  \brief Architecture-agnostic base class for model run states
 *  \details Contains model run data that is independent of the actual
 *  model architecture, e.g., cublas handles, generic cuda events, etc.
 */
class IRunState {
    friend class IModel;
    friend std::string save_checkpoint(std::string checkpoint_directory, int step, IModel& model,
        const DataLoader* loader, NCCLCommunicator& comm);
    friend void load_checkpoint(std::string checkpoint_directory, int step, IModel& model,
        DataLoader* loader, NCCLCommunicator& comm);
public:
    IRunState() = default;
    IRunState(PretrainedConfig config, long batch_size, long seq_len, std::shared_ptr<TensorAllocator> alloc);
    ~IRunState();

    IRunState(const IRunState&) = delete;
    IRunState& operator=(const IRunState&) = delete;
    IRunState(IRunState&& other) noexcept;
    IRunState& operator=(IRunState&& other) noexcept;

    //! gets the accumulated loss after the last backward operation.
    //! only should be called after the last backward step (i.e., where `micro_step==grad_accum_steps`)
    //! will block the caller until `backward` is done, i.e., the function is safe to call without
    //! additional synchronization.
    float get_loss() const;

    //! gets the global gradient norm.
    //! will block the caller until `update` has finished the norm calculation,
    //! i.e., the function is safe to call without additional synchronization.
    float get_norm() const;

    //! gets the accuracy after the last validate operation.
    //! will block the caller until `validate` is done, i.e., the function is safe to call without
    //! additional synchronization.
    float get_accuracy() const;

    // temporary buffers
    Tensor temp_alloc(ETensorDType dtype, const std::vector<long>& shape);
    void temp_acquire(Tensor& target);
    void temp_free(Tensor& tensor);

    PretrainedConfig Config;
    long B;     //!< Batch size
    long T;     //!< Sequence length
    int GradAccumSteps = 1;  //!< Most recent grad-accumulation steps (for loss/grad normalization)
    int WorldSize = 1;       //!< Data-parallel world size (used for masked-token normalization)

    std::shared_ptr<TensorAllocator> Allocator;
    DeviceMemoryStack Stack;

    Tensor Inputs;                      // (B, T) Int32
    Tensor PositionIDs;                 // (B, T) Int32 - explicit position IDs
    Tensor Targets;                     // (B, T) Int32
    Tensor Losses;                      // (B, T) FP32
    Tensor ValidTokenCount;             // (1,) Int32 - count of non-masked tokens
    Tensor CorrectCount;                // (1,) Int32 - count of correct predictions

    float* NormHost = nullptr;          // single value
    float* LossHost = nullptr;          // single value
    float* AccuracyHost = nullptr;      // single value for accuracy

    std::pair<float, float> record_step(float loss, float norm);

    cudaDeviceProp DeviceProp;
    int DeviceId = -1;

    cudaStream_t MainStream = nullptr;

    cudaEvent_t ForwardDone   = nullptr;       //!< recorded at the end of the forward pass
    cudaEvent_t BackwardDone  = nullptr;       //!< recorded at the end of the backward pass
    cudaEvent_t TransferDone  = nullptr;       //!< recorded once CPU-side buffers have been copied to GPU
    cudaEvent_t NormDone      = nullptr;       //!< recorded after norm calculation completes
    cudaEvent_t OptimizerDone = nullptr;       //!< recorded after the optimizer completes

    cudnnHandle_t CudnnHandle = nullptr;
    cublasLtHandle_t CublasLtHandle = nullptr;
    Tensor CuBlasWorkspace;
    Tensor CutlassWorkspace;  ///< Workspace for CUTLASS SM120 MX FP8 operations

    // events for debugging timings
    void setup_timing_events(int micro_steps);

    cudaEvent_t TimingOptimizerStart = nullptr;
    cudaEvent_t TimingOptimizerEnd   = nullptr;
    std::vector<cudaEvent_t> TimingForwardStart;
    std::vector<cudaEvent_t> TimingForwardEnd;
    std::vector<cudaEvent_t> TimingHeadStart;
    std::vector<cudaEvent_t> TimingHeadEnd;
    std::vector<cudaEvent_t> TimingBackwardStart;
    std::vector<cudaEvent_t> TimingBackwardEnd;

    Tensor Inputs_CPU;      // (B, T) Int32
    Tensor PositionIDs_CPU; // (B, T) Int32
    Tensor Targets_CPU;     // (B, T) Int32

    // =========================================================================
    // Virtual accessors for recipe-driven matmul dispatch
    // =========================================================================
    // These methods provide type-erased access to ModularRunState features.
    // Default implementations return nullptr/false. ModularRunState overrides.

    /// @brief Check if FP8 forward buffers are available
    [[nodiscard]] virtual bool has_fp8_forward() const { return false; }

    /// @brief Check if FP8 HYBRID backward mode is enabled
    [[nodiscard]] virtual bool has_fp8_hybrid_backward() const { return false; }

    /// @brief Check if FP8 delayed scaling is enabled
    [[nodiscard]] virtual bool has_fp8_delayed_scaling() const { return false; }

    /// @brief Check if FP4 forward is enabled
    [[nodiscard]] virtual bool has_fp4_forward() const { return false; }

    /// @brief Check if FP4 backward is enabled
    [[nodiscard]] virtual bool has_fp4_backward() const { return false; }

    /// @brief Check if activation quant buffers are allocated
    [[nodiscard]] virtual bool has_activation_quants() const { return false; }

    /// @brief Check if gradient quant buffers are allocated
    [[nodiscard]] virtual bool has_grad_quants() const { return false; }

    /// @brief Get FP8 forward quant buffer for a matmul operation (nullptr if not available)
    [[nodiscard]] virtual Tensor* get_fp8_forward_buffer(int /*op*/) { return nullptr; }

    /// @brief Get FP4 forward buffers for a matmul operation (data, scales, amax)
    [[nodiscard]] virtual std::tuple<Tensor*, Tensor*, float*> get_fp4_forward_buffers(int /*op*/) {
        return {nullptr, nullptr, nullptr};
    }

    /// @brief Get Hadamard workspace for FP4 operations (nullptr if not available)
    [[nodiscard]] virtual Tensor* get_hadamard_workspace() { return nullptr; }

    /// @brief Get activation quant buffer for a layer (nullptr if not available)
    [[nodiscard]] virtual Tensor* get_activation_quant_buffer(int /*layer_idx*/, int /*op*/) { return nullptr; }

    /// @brief Get gradient quant buffer for a matmul operation (nullptr if not available)
    [[nodiscard]] virtual Tensor* get_gradient_quant_buffer(int /*op*/) { return nullptr; }

    /// @brief Get FP8 scaling state for delayed scaling (nullptr if not available)
    [[nodiscard]] virtual modules::FP8ScalingState* get_fp8_scaling_state() { return nullptr; }

private:
    // ring-buffers that keep a history of past losses
    struct OutlierDetector {
        OutlierDetector(int window_size=100);

        void record(float value);
        float eval(float value) const;

        void re_evaluate();
        void reset(int window_size, int index, std::vector<float> values);

        int mWindowSize;
        int mIndex = 0;
        std::vector<float> mValues;

        double mSum = 0.0;
        double mSumSq = 0.0;
    };

    OutlierDetector LossOutliers;
    OutlierDetector NormOutliers;
};

#endif //SUROGATE_SRC_TRAINING_MODEL_H
