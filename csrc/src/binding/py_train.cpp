// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "py_train.h"

#include <filesystem>

#include <fmt/format.h>

#include "utilities/gpu_info.h"
#include "training/checkpoint.h"
#include "training/dataloader.h"
#include "training/logging.h"
#include "utilities/comm.h"
#include "kernels/kernels.h"
#include "training/model.h"
#include "modules/model_config.h"
#include "modules/model_factory.h"
#include "modules/composite/transformer_block.h"
#include "modules/lora/lora_model.h"

/**
 * @brief Construct a multi-GPU trainer and launch one worker thread per GPU.
 *
 * Creates NCCL communicators and per-rank worker threads, then waits until all workers
 * have initialized their model/run-state and the trainer is ready to accept work.
 *
 * @param ngpus Number of GPUs to use. If 0, use all visible CUDA devices.
 * @param config Model architecture configuration (layer counts, hidden sizes, etc.).
 * @param options Runtime/training options (precision, sharding, etc.).
 * @param batch_size Per-GPU micro-batch size (B).
 * @param seq_len Sequence length (T).
 * @param grad_accum Number of micro-steps to accumulate before calling update().
 * @param memcpy_all_gather If true, use memcpy-based path for all_gather (implementation-defined).
 * @param memcpy_send_recv If true, use memcpy-based path for send/recv (implementation-defined).
 *
 * @throws std::runtime_error If requested GPUs exceed available device count.
 */
MultiGPUPyTrainer::MultiGPUPyTrainer(int ngpus, PretrainedConfig config, RuntimeOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv, std::optional<LoRAAdapterConfig> lora_config, std::optional<modules::QLoRAConfig> qlora_config) :
    mConfig(config), mOptions(options), mLoRAConfig(lora_config), mQLoRAConfig(qlora_config), B(batch_size), T(seq_len), mGradAccumulation(grad_accum)
{
    int gpus_available = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpus_available));
    if (ngpus == 0) {
        ngpus = gpus_available;
    }

    if (ngpus > gpus_available) {
        throw std::runtime_error(fmt::format("Requested {} GPUs, only {} available", ngpus, gpus_available));
    }
    mContexts.resize(ngpus);
    mThreads = NCCLCommunicator::launch_threads_communicators(
       ngpus, memcpy_all_gather, memcpy_send_recv,
       [&](NCCLCommunicator& comm) {
           try {
               this->main_loop(comm);
           } catch (...) {
               mHasCrashed = true;
               throw;
           }
       });

    while(!mIsRunning && !mHasCrashed) {
        std::this_thread::yield();
    }
}

/**
 * @brief Stop all workers and join their threads.
 *
 * Signals termination, synchronizes CUDA devices for each initialized context, then joins
 * the NCCL worker threads. Intended to ensure all outstanding GPU work completes before exit.
 */
MultiGPUPyTrainer::~MultiGPUPyTrainer() {
    mIsRunning = false;

    // make sure all work has finished
    for(auto& ctx : mContexts) {
        if(ctx.Communicator) {
            CUDA_CHECK(cudaSetDevice(ctx.Communicator->rank()));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mThreads->join();
}

/**
 * @brief Import model weights from disk on all ranks.
 *
 * Runs a synchronized "work item" across all worker threads/ranks.
 *
 * @param path Path to the weights source (format handled by IModel::import_weights()).
 */
void MultiGPUPyTrainer::import_weights(std::string path) {
    run_work([this, path](sThreadContext& ctx) {
        ctx.Model->import_weights(path, true, *ctx.Communicator);

        // Print memory breakdown if enabled (rank 0 only)
        if (mOptions.DebugMemoryBreakdown && ctx.Communicator->rank() == 0) {
            auto& rs = ctx.Model->get_run_state();
            if (rs.Allocator) {
                auto stats = rs.Allocator->get_allocation_segments();
                auto stack_stats = rs.Stack.get_allocation_stats();

                // Build memory breakdown context
                MemoryBreakdownContext breakdown_ctx;
                breakdown_ctx.enabled = true;
                breakdown_ctx.allocator = rs.Allocator.get();
                breakdown_ctx.hidden_size = mConfig.HiddenSize;
                breakdown_ctx.intermediate_size = mConfig.IntermediateSize;
                breakdown_ctx.num_layers = mConfig.NumLayers;
                breakdown_ctx.batch_size = B;
                breakdown_ctx.seq_length = T;

                // Get QLoRA stats if applicable
                using DenseBlock = modules::DenseTransformerBlock<>;
                if (auto* lora_model = dynamic_cast<modules::ModularLoRAModel<DenseBlock>*>(ctx.Model.get())) {
                    if (lora_model->qlora_enabled()) {
                        breakdown_ctx.qlora_quantized_bytes = lora_model->qlora_quantized_weights_bytes();
                        breakdown_ctx.qlora_savings_ratio = lora_model->qlora_memory_savings_ratio();
                    }
                }

                // Use a temporary logger to print the breakdown
                TrainingRunLogger logger("", 0, TrainingRunLogger::VERBOSE);
                logger.log_allocator(stats, stack_stats, breakdown_ctx);
            }
        }
    });
}


/**
 * @brief Export the model configuration and weights to a directory.
 *
 * Creates the directory (and parents) if needed, writes `config.json`, and writes
 * `model.safetensors`. Executed as a synchronized work item across ranks (with internal
 * coordination handled by the model/communicator).
 *
 * @param path Output directory path.
 */
void MultiGPUPyTrainer::export_model(std::string path) {
    run_work([path](sThreadContext& ctx) {
        std::filesystem::path p(path);
        std::filesystem::create_directories(p);

        if (ctx.Communicator->rank() == 0) {
            save_pretrained_config(ctx.Model->get_run_state().Config, (p / "config.json").c_str());
        }
        ctx.Model->export_weights((p / "model.safetensors").c_str(), *ctx.Communicator);
    });
}

/**
 * @brief Export LoRA adapter weights to a directory.
 *
 * Creates the directory (and parents) if needed, writes `adapter_model.safetensors` and
 * `adapter_config.json` in PEFT-compatible format. Only works if the model is a LoRAModel.
 * Executed as a synchronized work item across ranks.
 *
 * @param path Output directory path.
 * @param base_model_path Optional path/name of base model for adapter_config.json.
 * @throws std::runtime_error If the model is not a LoRAModel.
 */
void MultiGPUPyTrainer::export_adapter(std::string path, std::string base_model_path) {
    run_work([path, base_model_path](sThreadContext& ctx) {
        using DenseBlock = modules::DenseTransformerBlock<>;
        auto* lora_model = dynamic_cast<modules::ModularLoRAModel<DenseBlock>*>(ctx.Model.get());
        if (!lora_model) throw std::runtime_error("export_adapter: Model is not a modular LoRA model.");
        lora_model->export_adapter(path, *ctx.Communicator, base_model_path);
    });
}

/**
 * @brief Initialize model weights on all ranks.
 *
 * Executes model initialization as a synchronized work item across worker threads/ranks.
 */
void MultiGPUPyTrainer::init_weights() {
    run_work([](sThreadContext& ctx) {
        ctx.Model->init_weights(*ctx.Communicator);
    });
}

/**
 * @brief Load a training checkpoint.
 *
 * Loads model state for the given training step from a checkpoint directory.
 * Executed as a synchronized work item across ranks.
 *
 * @param directory Checkpoint root directory.
 * @param step Checkpoint step index to load.
 */
void MultiGPUPyTrainer::load_checkpoint(std::string directory, int step) {
    run_work([directory, step](sThreadContext& ctx) {
        ::load_checkpoint(directory, step, *ctx.Model, nullptr, *ctx.Communicator);
    });
}

/**
 * @brief Save a training checkpoint.
 *
 * Saves model state for the given training step into a checkpoint directory.
 * Executed as a synchronized work item across ranks.
 *
 * @param directory Checkpoint root directory.
 * @param step Checkpoint step index to save.
 */
void MultiGPUPyTrainer::save_checkpoint(std::string directory, int step) {
    run_work([directory, step](sThreadContext& ctx) {
        ::save_checkpoint(directory, step, *ctx.Model, nullptr, *ctx.Communicator);
    });
}

/**
 * @brief Run one training micro-step (forward + backward) on all ranks.
 *
 * Copies host input/target token IDs into each rank's device-visible input/target buffers,
 * then runs forward/backward for the current micro-step index.
 *
 * Buffer layout expectation:
 * - `inputs` contains `world_size * B * T` int32 tokens laid out contiguously.
 * - Rank `i` reads from `inputs + i * B * T`.
 * Same for `targets`.
 *
 * @param inputs Pointer to host int32 token IDs for all ranks (see layout above).
 * @param targets Pointer to host int32 target token IDs for all ranks (see layout above).
 *
 * @throws std::runtime_error If called more than `grad_accum` times without an update().
 */
void MultiGPUPyTrainer::step(const std::int32_t* inputs, const std::int32_t* targets) {
    for(int i = 0; i < mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
    }

    if(mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(fmt::format("step: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
    }

    run_work([micro_idx = mTrainMicroStep, micro_batches = mGradAccumulation](sThreadContext& ctx) {
        Tensor inputs = ctx.Model->get_input_buffer();
        Tensor position_ids = ctx.Model->get_position_ids_buffer();
        Tensor targets = ctx.Model->get_target_buffer();
        ctx.Model->forward(inputs, position_ids, *ctx.Communicator, micro_idx);
        ctx.Model->backward(inputs, targets, *ctx.Communicator, micro_batches, micro_idx);
    });
    ++mTrainMicroStep;
}

/**
 * @brief Run one validation step and return the loss (rank 0).
 *
 * Copies host input/target token IDs into each rank's buffers (same layout as step()),
 * then runs validation. The returned loss is taken from rank 0.
 *
 * @param inputs Pointer to host int32 token IDs for all ranks.
 * @param targets Pointer to host int32 target token IDs for all ranks.
 * @return Loss value computed on rank 0 for this validation micro-step.
 */
float MultiGPUPyTrainer::validate(const std::int32_t* inputs, const std::int32_t* targets) {
    for(int i = 0; i < mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
    }

    float loss;

    run_work([micro_idx = mEvalStep, &loss](sThreadContext& ctx) {
        Tensor inputs = ctx.Model->get_input_buffer();
        Tensor position_ids = ctx.Model->get_position_ids_buffer();
        Tensor targets = ctx.Model->get_target_buffer();
        float calc_loss = ctx.Model->validate(inputs, position_ids, targets, *ctx.Communicator, micro_idx);
        if (ctx.Communicator->rank() == 0) {
            loss = calc_loss;
        }
    });

    ++mEvalStep;

    return loss;
}

/**
 * @brief Apply optimizer update after gradient accumulation.
 *
 * Executes the optimizer step on all ranks, synchronizes the device, then reads back
 * loss and norm from rank 0 context. Resets internal micro-step counters so the next
 * forward will re-gather as needed.
 *
 * Note: The implementation passes `step + 1` to the model update (1-based step).
 *
 * @param lr Learning rate.
 * @param beta1 Adam beta1 coefficient.
 * @param beta2 Adam beta2 coefficient.
 * @param step Zero-based global optimization step index (converted to 1-based internally).
 * @param weight_decay Weight decay coefficient.
 * @param grad_clip Gradient clipping threshold (implementation-defined norm type).
 * @return Pair of (loss, grad_norm). Loss is already normalized by valid token count from get_loss().
 */
std::pair<float, float> MultiGPUPyTrainer::update(float lr, float beta1, float beta2, int step, float epsilon, float weight_decay, float grad_clip) {
    run_work([=](sThreadContext& ctx) {
        ctx.Model->update(*ctx.Communicator, lr, beta1, beta2, step + 1, epsilon, weight_decay, grad_clip);
        CUDA_CHECK(cudaDeviceSynchronize());

    });
    float step_loss, step_norm;

    auto& ctx = mContexts.at(0);
    step_loss = ctx.Model->get_loss();
    step_norm = ctx.Model->get_norm();

    // ensure we're re-gathering on next forward for eval and train
    mTrainMicroStep = 0;
    mEvalStep = 0;

    // Note: get_loss() already returns loss normalized by valid token count,
    // so we don't normalize further here (unlike the old code that divided by B*T*grad_accum)
    return {step_loss, step_norm};
}


/**
 * @brief Query per-GPU utilization information for all ranks.
 *
 * Executes a synchronized work item across ranks and returns the most recent utilization
 * snapshot for each rank/device.
 *
 * @return Vector indexed by rank containing utilization/telemetry info.
 */
std::vector<GPUUtilInfo> MultiGPUPyTrainer::get_gpu_info() {
    std::vector<GPUUtilInfo> infos(mContexts.size());
    run_work([&](sThreadContext& ctx) {
         infos[ctx.Communicator->rank()] = ctx.GPUUtil->update();
    });
    return infos;
}

/**
 * @brief Request all worker threads to stop processing new work.
 *
 * This sets the running flag to false. The destructor additionally synchronizes and joins.
 */
void MultiGPUPyTrainer::stop() {
    mIsRunning = false;
}

/**
 * @brief Fetch the next queued work item for a specific worker context.
 *
 * Called from the worker thread. If no work is available, returns an empty function.
 * Access is protected by a global mutex since work assignment is shared across ranks.
 *
 * @param ctx The calling worker's thread context.
 * @return A callable work item to execute, or an empty function if none is pending.
 */
auto MultiGPUPyTrainer::fetch_work(sThreadContext& ctx) -> std::function<void(sThreadContext & ctx)> {
    std::lock_guard<std::mutex> lock(mGlobalMutex);
    if (!ctx.Work) {
        std::this_thread::yield();
        return {};
    } else {
        auto work = std::move(ctx.Work);
        return work;
    }
}

/**
 * @brief Schedule a work item to run on one rank or all ranks, and wait for completion.
 *
 * If @p idx >= 0, schedules the work only on that rank and treats all other ranks as already done.
 * If @p idx < 0, schedules the work on all ranks.
 *
 * This call blocks until the scheduled work completes (as indicated by @c mWorkDone),
 * or propagates worker thread exceptions.
 *
 * @param work Callable executed within each worker thread, receiving that rank's context.
 * @param idx Target rank index, or -1 to run on all ranks.
 *
 * @throws Rethrows any exception encountered in worker threads.
 */
void MultiGPUPyTrainer::run_work(std::function<void(sThreadContext & ctx)> work, int idx) {
    {
        std::lock_guard<std::mutex> lock(mGlobalMutex);

        if (idx >= 0) {
            mWorkDone = mContexts.size() - 1;
            mContexts.at(idx).Work = work;
        } else {
            mWorkDone = 0;
            for (auto& ctx: mContexts) {
                ctx.Work = work;
            }
        }
    }

    while(mWorkDone.load() < mContexts.size()) {
        if(mThreads->has_exception()) {
            stop();
            mThreads->join(); // will throw, ending the loop
        }
        std::this_thread::yield();
    }
}

/**
 * @brief Worker thread entry point for a given NCCL communicator rank.
 *
 * Initializes per-rank resources (communicator pointer, GPU util tracker, model, run-state),
 * signals readiness, then repeatedly polls for work via fetch_work() and executes it.
 *
 * @param comm Rank-local NCCL communicator (provides rank/world_size and collectives).
 *
 * @throws std::runtime_error If another worker reports a crash during startup waiting phase.
 */
void MultiGPUPyTrainer::main_loop(NCCLCommunicator& comm) {
    sThreadContext& ctx = mContexts.at(comm.rank());

    ctx.Communicator = &comm;
    ctx.GPUUtil = IGPUUtilTracker::create();

    auto allocator = std::make_shared<TensorAllocator>();

    modules::ModelConfig mod_config = modules::ModelConfig::from_llama_config(mConfig);
            modules::ModelOptions mod_options = modules::ModelOptions::from_runtime_options(mOptions);

    std::unique_ptr<IModel> model_storage = modules::ModelFactory::create(
        mod_config, mod_options, comm.rank(), comm.world_size(), allocator);

    if (mLoRAConfig.has_value()) {
        if (mod_config.architecture != modules::ArchitectureType::Dense) {
            throw std::runtime_error("MultiGPUPyTrainer: LoRA requires dense architecture");
        }

        // Convert LoRAAdapterConfig -> modular LoRA config.
        modules::ModularLoRAConfig mod_lora;
        mod_lora.rank = mLoRAConfig->Rank;
        mod_lora.alpha = mLoRAConfig->Alpha;
        mod_lora.dropout = mLoRAConfig->Dropout;
        mod_lora.dtype = mLoRAConfig->DType;
        mod_lora.init_a_kaiming = mLoRAConfig->InitAKaimingUniform;
        mod_lora.use_rs_lora = mLoRAConfig->UseRSLoRA;
        mod_lora.targets.clear();
        if (mLoRAConfig->TargetModules.count("all") > 0) {
            mod_lora.with_all();
        } else {
            for (const auto& name : mLoRAConfig->TargetModules) {
                if (name == "q_proj") mod_lora.targets.insert(modules::LoRATarget::Q_PROJ);
                else if (name == "k_proj") mod_lora.targets.insert(modules::LoRATarget::K_PROJ);
                else if (name == "v_proj") mod_lora.targets.insert(modules::LoRATarget::V_PROJ);
                else if (name == "o_proj") mod_lora.targets.insert(modules::LoRATarget::O_PROJ);
                else if (name == "gate_proj") mod_lora.targets.insert(modules::LoRATarget::GATE_PROJ);
                else if (name == "up_proj") mod_lora.targets.insert(modules::LoRATarget::UP_PROJ);
                else if (name == "down_proj") mod_lora.targets.insert(modules::LoRATarget::DOWN_PROJ);
            }
        }

        using DenseBlock = modules::DenseTransformerBlock<>;
        using DenseModel = modules::ModularTransformerModel<DenseBlock>;
        auto* dense_ptr = dynamic_cast<DenseModel*>(model_storage.get());
        if (!dense_ptr) {
            throw std::runtime_error("MultiGPUPyTrainer: modular factory returned non-dense model under dense config");
        }

        // Build QLoRA config if provided
        modules::QLoRAConfig qlora_config;
        if (mQLoRAConfig.has_value()) {
            qlora_config = mQLoRAConfig.value();
        }

        std::unique_ptr<DenseModel> dense_base(static_cast<DenseModel*>(model_storage.release()));
        ctx.Model = std::make_unique<modules::ModularLoRAModel<DenseBlock>>(
            std::move(dense_base), mod_lora, mOptions, comm, allocator, qlora_config);
    } else {
        ctx.Model = std::move(model_storage);
    }

    ctx.Model->allocate_run_state(mOptions, comm, B, T, /*allocate_optimizer=*/true);

    // Default position IDs: [0..T-1] for each sequence in the batch.
    // This keeps Python-side training/tests deterministic even when callers do not provide
    // explicit position ids (unlike the C++ training binary which can load them from .bin files).
    {
        auto* pos = ctx.Model->get_position_ids_buffer().get<std::int32_t>();
        if (!pos) {
            throw std::runtime_error("PositionIDs buffer is not INT32 (unexpected)");
        }
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                pos[b * T + t] = t;
            }
        }
    }

    if (mIsReady.fetch_add(1) == comm.world_size() - 1) {
        mIsRunning = true;
    };

    while (!mIsRunning.load()) {
        std::this_thread::yield();
        if(mHasCrashed.load()) throw std::runtime_error("Another worker has crashed, exiting.");
    }

    while (mIsRunning.load()) {
        if (auto work = fetch_work(ctx); work) {
            work(ctx);
            mWorkDone.fetch_add(1);
        } else {
            std::this_thread::yield();
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();

    // free resources
    ctx.Model.reset();
    ctx.GPUUtil.reset();
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Return the NCCL world size (number of ranks / GPUs).
 *
 * @return World size as reported by rank 0 communicator.
 */
int MultiGPUPyTrainer::world_size() const {
    return mContexts.at(0).Communicator->world_size();
}

/**
 * @brief Get allocator segment allocation info for a specific GPU/rank.
 *
 * Executes a work item only on the specified rank and returns that rank's allocator segments.
 *
 * @param gpu_id Rank/GPU index to query.
 * @return Vector of (segment_name, segment_memory_stats) pairs for that rank.
 */
std::vector<std::pair<std::string, sSegmentMemory>> MultiGPUPyTrainer::get_allocations(int gpu_id) {
    std::vector<std::pair<std::string, sSegmentMemory>> result;
    run_work([&result](sThreadContext& ctx) {
        auto& rs = ctx.Model->get_run_state();
        if (rs.Allocator) {
            result = rs.Allocator->get_allocation_segments();
        }
    }, gpu_id);
    return result;
}

/**
 * @brief Get run-state stack allocation statistics for a specific GPU/rank.
 *
 * Executes a work item only on the specified rank and returns stack allocation stats.
 *
 * @param gpu_id Rank/GPU index to query.
 * @return Vector of (stat_name, stat_value) pairs for that rank.
 */
std::vector<std::pair<std::string, long>> MultiGPUPyTrainer::get_stack_info(int gpu_id) {
    std::vector<std::pair<std::string, long>> result;
    run_work([&result](sThreadContext& ctx) {
        result = ctx.Model->get_run_state().Stack.get_allocation_stats();
    }, gpu_id);
    return result;
}

/**
 * @brief Collect gradient tensors for a specific GPU/rank.
 *
 * Executes a work item only on the specified rank, synchronizes the device to ensure
 * gradient buffers are ready, then returns a list of named gradient tensor shards.
 * Returned tensor names follow a HuggingFace-like naming convention.
 *
 * @param gpu_id Rank/GPU index to query.
 * @return Vector of (parameter_name, gradient_tensor_shard) pairs for that rank.
 */
std::vector<std::pair<std::string, Tensor>> MultiGPUPyTrainer::get_gradients(int gpu_id) {
    std::vector<std::pair<std::string, Tensor>> result;
    run_work([&result](sThreadContext& ctx) {
        using DenseBlock = modules::DenseTransformerBlock<>;
        using DenseModel = modules::ModularTransformerModel<DenseBlock>;
        DenseModel* base_model = nullptr;
        if (auto* m = dynamic_cast<DenseModel*>(ctx.Model.get())) {
            base_model = m;
        } else if (auto* lora = dynamic_cast<modules::ModularLoRAModel<DenseBlock>*>(ctx.Model.get())) {
            base_model = &lora->base_model();
        }
        if (!base_model) {
            throw std::runtime_error("get_gradients: unsupported model type for gradient inspection");
        }

        const auto& config = base_model->config();
        auto& rs = base_model->get_run_state();
        cudaStream_t stream = rs.MainStream;
        auto& grads = base_model->grads();

        CUDA_CHECK(cudaDeviceSynchronize());
        result.emplace_back("model.embed_tokens.weight", static_cast<Tensor>(grads.get_embeddings_shard(stream)));
        if (!config.TiedWordEmbeddings) {
            result.emplace_back("lm_head.weight", static_cast<Tensor>(grads.get_lm_head_shard(stream)));
        }
        result.emplace_back("model.norm.weight", static_cast<Tensor>(grads.get_final_norm_shard(stream)));

        for (int l = 0; l < config.NumLayers; l++) {
            std::string prefix = "model.layers." + std::to_string(l);
            auto& block = grads.get_block_shard(l, stream);

            result.emplace_back(prefix + ".input_layernorm.weight", block.ln1_grads.d_weight);
            result.emplace_back(prefix + ".post_attention_layernorm.weight", block.ln2_grads.d_weight);

            result.emplace_back(prefix + ".self_attn.qkv.weight", block.attention_grads.d_qkv_weight);
            if (block.attention_grads.d_qkv_bias.has_value()) {
                result.emplace_back(prefix + ".self_attn.qkv.bias", block.attention_grads.d_qkv_bias.value());
            }
            result.emplace_back(prefix + ".self_attn.o_proj.weight", block.attention_grads.d_out_weight);

            result.emplace_back(prefix + ".mlp.up.weight", block.d_mlp_up_weight);
            result.emplace_back(prefix + ".mlp.down_proj.weight", block.d_mlp_down_weight);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }, gpu_id);
    return result;
}

std::vector<std::pair<std::string, Tensor>> MultiGPUPyTrainer::get_lora_gradients(int gpu_id) {
    std::vector<std::pair<std::string, Tensor>> result;
    run_work([&result](sThreadContext& ctx) {
        using DenseBlock = modules::DenseTransformerBlock<>;
        auto* lora_model = dynamic_cast<modules::ModularLoRAModel<DenseBlock>*>(ctx.Model.get());
        if (!lora_model) throw std::runtime_error("get_lora_gradients: model is not a modular LoRA model");

        const auto& config = lora_model->base_model().config();
        CUDA_CHECK(cudaDeviceSynchronize());

        auto add_layer = [&](const std::string& module_prefix,
                             const std::optional<modules::LoRALayerWeights<Tensor>>& layer) {
            if (!layer.has_value()) return;
            if (layer->A.Data) result.emplace_back(module_prefix + ".lora_A.weight", layer->A);
            if (layer->B.Data) result.emplace_back(module_prefix + ".lora_B.weight", layer->B);
        };

        for (int l = 0; l < config.NumLayers; ++l) {
            bool unused_accumulate = false;
            auto& block = lora_model->lora_grads().get_block_full(l, /*stream=*/nullptr, *ctx.Communicator, unused_accumulate);
            std::string prefix = fmt::format("base_model.model.model.layers.{}", l);

            add_layer(prefix + ".self_attn.q_proj", block.attention.q);
            add_layer(prefix + ".self_attn.k_proj", block.attention.k);
            add_layer(prefix + ".self_attn.v_proj", block.attention.v);
            add_layer(prefix + ".self_attn.o_proj", block.attention.o);

            add_layer(prefix + ".mlp.gate_proj", block.mlp.gate);
            add_layer(prefix + ".mlp.up_proj", block.mlp.up);
            add_layer(prefix + ".mlp.down_proj", block.mlp.down);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }, gpu_id);
    return result;
}
