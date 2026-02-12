// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_BINDING_PY_TRAIN_H
#define SUROGATE_SRC_BINDING_PY_TRAIN_H

#include <memory>
#include <string>
#include <utility>
#include <thread>
#include <functional>
#include <vector>
#include <cuda_runtime.h>

#include "config/pretrained_config.h"
#include "runtime/training/runtime_options.h"
#include "config/lora_adapter_config.h"
#include "runtime/qlora/qlora_config.h"
#include "runtime/optimizers/optimizer_config.h"

class DataLoader;
class IModel;
class IGPUUtilTracker;
struct GPUUtilInfo;
struct sSegmentMemory;
class CommunicatorThreadsPack;
class NCCLCommunicator;

//! \brief A multi-GPU trainer wrapper to be used for python bindings
//! \details When wrapping the C++ Surogate core for Python, the  main source of difficulty is handling
//! multi-GPU support. The C++ version supports both multi-process and multi-thread, with
//! multi-thread being the more interesting (due to cudaMemcpy) option.
//! However, mapping multi-threading to python is problematic due to GIL (maybe that will be better once
//! free-threaded python is widely used); hence, this wrapper is used to hide all worker threads
//! from the python interface.
//!
//! Internally, we start up one thread per GPU, and keep track of its training state (`sThreadContext`).
//! Each interface function wraps the desired model call into a std::function that gets sent to the thread
//! context. Each thread runs an infinite loop, and picks up the work it has been sent. Interface functions
//! only return once the work is done. If the work function does not synchronize with the GPU, "done" in this
//! case means that the CPU execution has finished, but the GPU might still be busy. This allows overlap of
//! python execution with GPU execution.
//!
//! As a consequence of this implementation strategy, data loading in python will be slightly different than in the
//! C++ implementation. For C++, each thread has its own DataLoader, providing `B*T` tokens each step. For python,
//! we have only one interface-visible thread, which gets `nGPU*B*T` tokens per step, and splits them into `B*T`-sized
//! chunks for each GPU.
class MultiGPUPyTrainer
{
public:
    //! Single-node constructor (original)
    MultiGPUPyTrainer(int ngpus, const PretrainedConfig& config, RuntimeOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv, std::optional<LoRAAdapterConfig> lora_config = std::nullopt, std::optional<modules::QLoRAConfig> qlora_config = std::nullopt);

    //! Multi-node constructor (for Ray distributed training)
    MultiGPUPyTrainer(int ngpus, int node_rank, int num_nodes,
                      const void* nccl_id, const void* node_master_nccl_id,
                      const PretrainedConfig& config, RuntimeOptions options,
                      int batch_size, int seq_len, int grad_accum,
                      bool memcpy_all_gather, bool memcpy_send_recv,
                      std::optional<LoRAAdapterConfig> lora_config = std::nullopt,
                      std::optional<modules::QLoRAConfig> qlora_config = std::nullopt);

    ~MultiGPUPyTrainer();

    void import_weights(std::string path);
    void export_model(std::string path);
    void export_adapter(std::string path, std::string base_model_path = "");
    void init_weights();
    void load_checkpoint(std::string directory, int step);
    void save_checkpoint(std::string directory, int step);
    void step(const std::int32_t* inputs, const std::int32_t* targets, const std::int32_t* position_ids = nullptr);
    float validate(const std::int32_t* inputs, const std::int32_t* targets, const std::int32_t* position_ids = nullptr);
    std::pair<float, float> update_with_config(const optimizers::OptimizerConfig& config, int step);
    std::pair<float, float> train_step_graphed(const std::int32_t* inputs, const std::int32_t* targets,
                                               const std::int32_t* position_ids,
                                               const optimizers::OptimizerConfig& config, int step);
    void stop();

    std::vector<GPUUtilInfo> get_gpu_info();

    // MoE stats (returns {aux_loss, z_loss, expert_utilization, load_imbalance, valid})
    // Returns zeros with valid=false for non-MoE models
    std::tuple<float, float, float, float, bool> get_moe_stats();

    int world_size() const;
    int local_world_size() const { return static_cast<int>(mContexts.size()); }
    int batch_size() const { return B; }
    int seq_length() const { return T; }
    int grad_accumulation() const { return mGradAccumulation; }
    const PretrainedConfig& config() const { return *mConfig; }
    const RuntimeOptions& options() const { return mOptions; }
    bool is_qlora() const { return mLoRAConfig.has_value() && mQLoRAConfig.has_value() && mQLoRAConfig->is_quantized(); }

    std::vector<std::pair<std::string, sSegmentMemory>> get_allocations(int gpu_id);
    std::vector<std::pair<std::string, long>> get_stack_info(int gpu_id);
    std::vector<std::pair<std::string, Tensor>> get_gradients(int gpu_id);
    std::vector<std::pair<std::string, Tensor>> get_lora_gradients(int gpu_id);
    int get_valid_token_count(int gpu_id);
    void set_visual_inputs(const std::int32_t* visual_pos_masks,
                           const float* visual_embeds,
                           const std::vector<const float*>& deepstack_visual_embeds);

private:
    std::unique_ptr<PretrainedConfig> mConfig;  // unique_ptr to preserve polymorphism
    RuntimeOptions mOptions;
    std::optional<LoRAAdapterConfig> mLoRAConfig;
    std::optional<modules::QLoRAConfig> mQLoRAConfig;
    int B;
    int T;

    int mTrainMicroStep = 0;
    int mEvalStep = 0;
    int mGradAccumulation = 1;

    std::unique_ptr<CommunicatorThreadsPack> mThreads;
    struct sFullStepGraphState {
        cudaGraphExec_t graph_exec = nullptr;
        bool captured = false;
        int captured_B = 0;
        int captured_T = 0;
        int captured_grad_accum = 0;
        Tensor opt_params;
        Tensor opt_step;
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        std::vector<Tensor> position_ids;
    };
    struct sThreadContext {
        NCCLCommunicator* Communicator;
        std::unique_ptr<IModel> Model;
        std::unique_ptr<IGPUUtilTracker> GPUUtil;
        std::unique_ptr<sFullStepGraphState> FullStepGraph;
        std::function<void(sThreadContext& ctx)> Work;
    };
    std::vector<sThreadContext> mContexts;
    std::mutex mGlobalMutex;
    std::atomic<bool> mIsRunning = false;
    std::atomic<bool> mHasCrashed = false;
    std::atomic<int> mIsReady = 0;
    std::atomic<int> mWorkDone = 0;

    std::function<void(sThreadContext& ctx)> fetch_work(sThreadContext& ctx);
    void run_work(std::function<void(sThreadContext& ctx)> work, int idx=-1);
    void main_loop(NCCLCommunicator& comm);
};


#endif //SUROGATE_SRC_BINDING_PY_TRAIN_H
