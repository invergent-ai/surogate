// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "py_train.h"

#include <array>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <fmt/format.h>

#include "utilities/gpu_info.h"
#include "utilities/dtype.h"
#include "runtime/training/checkpoint.h"
#include "runtime/training/dataloader.h"
#include "runtime/training/logging.h"
#include "utilities/comm.h"
#include "kernels/kernels.h"
#include "runtime/training/model.h"
#include "runtime/core/model_factory.h"
#include "runtime/lora/lora_config.h"
#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_grad_store.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/optimizers/normuon.h"

namespace {
bool env_enabled(const char* name) {
    if (!name || !*name) {
        return false;
    }
    const char* value = std::getenv(name);
    if (!value) {
        return false;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0) {
        return false;
    }
    return true;
}

}  // namespace

namespace {
void copy_from_float(void* dst, ETensorDType dtype, const float* src, std::size_t n) {
    if (!dst || n == 0) {
        return;
    }
    if (!src) {
        std::memset(dst, 0, n * get_dtype_size(dtype));
        return;
    }
    switch (dtype) {
        case ETensorDType::FP32: {
            std::memcpy(dst, src, n * sizeof(float));
            break;
        }
        case ETensorDType::BF16: {
            auto* out = reinterpret_cast<nv_bfloat16*>(dst);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = __float2bfloat16(src[i]);
            }
            break;
        }
        case ETensorDType::FP16: {
            auto* out = reinterpret_cast<half*>(dst);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = __float2half(src[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("set_visual_inputs: unsupported dtype for visual embeds");
    }
}

void zero_tensor(Tensor& t) {
    if (t.Data && t.bytes() > 0) {
        std::memset(t.Data, 0, static_cast<std::size_t>(t.bytes()));
    }
}
}  // namespace

static void fill_sequential_position_ids(std::int32_t* dst, int planes, int B, int T) {
    for (int p = 0; p < planes; ++p) {
        for (int b = 0; b < B; ++b) {
            std::int32_t* row = dst + (p * B + b) * T;
            for (int t = 0; t < T; ++t) {
                row[t] = t;
            }
        }
    }
}

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
MultiGPUPyTrainer::MultiGPUPyTrainer(int ngpus, const PretrainedConfig& config, RuntimeOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv, std::optional<LoRAAdapterConfig> lora_config, std::optional<modules::QLoRAConfig> qlora_config) :
    mConfig(config.clone()), mOptions(options), mLoRAConfig(lora_config), mQLoRAConfig(qlora_config), B(batch_size), T(seq_len), mGradAccumulation(grad_accum)
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
    mThreads = NCCLCommunicator::launch_communicators(
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
 * @brief Construct a multi-GPU trainer for multi-node training (Ray).
 *
 * Creates NCCL communicators using externally-provided NCCL IDs for cross-node coordination.
 * Used when training is orchestrated by Ray, where NCCL IDs are shared via Ray's object store.
 *
 * @param ngpus Number of local GPUs on this node.
 * @param node_rank This node's rank (0 to num_nodes-1).
 * @param num_nodes Total number of nodes in the cluster.
 * @param nccl_id 128-byte NCCL unique ID for global communicator (shared across all nodes).
 *                Node master communicator is derived internally via ncclCommSplit.
 * @param config Model architecture configuration.
 * @param options Runtime/training options.
 * @param batch_size Per-GPU micro-batch size.
 * @param seq_len Sequence length.
 * @param grad_accum Number of micro-steps before optimizer update.
 * @param memcpy_all_gather Enable memcpy-based all-gather.
 * @param memcpy_send_recv Enable memcpy-based send/recv.
 * @param lora_config Optional LoRA configuration.
 * @param qlora_config Optional QLoRA configuration.
 */
MultiGPUPyTrainer::MultiGPUPyTrainer(int ngpus, int node_rank, int num_nodes,
                                     const void* nccl_id,
                                     const PretrainedConfig& config, RuntimeOptions options,
                                     int batch_size, int seq_len, int grad_accum,
                                     bool memcpy_all_gather, bool memcpy_send_recv,
                                     std::optional<LoRAAdapterConfig> lora_config,
                                     std::optional<modules::QLoRAConfig> qlora_config) :
    mConfig(config.clone()), mOptions(options), mLoRAConfig(lora_config), mQLoRAConfig(qlora_config), B(batch_size), T(seq_len), mGradAccumulation(grad_accum)
{
    int gpus_available = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpus_available));
    if (ngpus == 0) {
        ngpus = gpus_available;
    }

    if (ngpus > gpus_available) {
        throw std::runtime_error(fmt::format("Requested {} GPUs, only {} available", ngpus, gpus_available));
    }

    // Copy NCCL ID to owned storage. The caller's buffer (nanobind nb::bytes) may be
    // destroyed before the worker threads read the ID in ncclCommInitRank.
    std::array<std::byte, 128> nccl_id_owned{};
    std::memcpy(nccl_id_owned.data(), nccl_id, 128);

    mContexts.resize(ngpus);
    mThreads = NCCLCommunicator::launch_communicators_multinode(
       ngpus, node_rank, num_nodes, nccl_id_owned.data(),
       memcpy_all_gather, memcpy_send_recv,
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
    // Use local_rank() for cudaSetDevice, and don't throw from destructor
    for(auto& ctx : mContexts) {
        if(ctx.Communicator) {
            cudaError_t err = cudaSetDevice(ctx.Communicator->local_rank());
            if (err == cudaSuccess) {
                cudaDeviceSynchronize();
            }
            // Ignore errors - we're in destructor, possibly after a crash
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    mThreads->join();
}

/**
 * @brief Set the path to a PEFT adapter to merge into base weights during import.
 *
 * Must be called before import_weights(). The adapter's LoRA deltas will be
 * applied to the BF16 base weights before quantization (QLoRA) or storage (BF16).
 */
void MultiGPUPyTrainer::set_adapter_path(std::string path) {
    run_work([path](sThreadContext& ctx) {
        if (auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get())) {
            dsl_model->set_adapter_path(path);
        }
    });
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

        // Schedule deferred QLoRA offloading auto-tune.  The actual tuning
        // runs after step 0 completes (inside invalidate_cache) when all lazy
        // runtime allocations are settled and gpu_free reflects steady-state.
        if (auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get())) {
            dsl_model->auto_tune_offloading();
        }

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
                breakdown_ctx.hidden_size = mConfig->HiddenSize;
                breakdown_ctx.intermediate_size = mConfig->IntermediateSize;
                breakdown_ctx.num_layers = mConfig->NumLayers;
                breakdown_ctx.batch_size = B;
                breakdown_ctx.seq_length = T;

                // Get QLoRA stats if applicable
                auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
                if (!dsl_model) {
                    throw std::runtime_error("memory breakdown requires DSL model");
                }
                if (dsl_model->qlora_enabled()) {
                    breakdown_ctx.qlora_quantized_bytes = dsl_model->qlora_quantized_weights_bytes();
                    breakdown_ctx.qlora_savings_ratio = dsl_model->qlora_memory_savings_ratio();
                }

                // Use a temporary logger to print the breakdown
                TrainingRunLogger logger("", 0, TrainingRunLogger::VERBOSE);
                logger.log_allocator(stats, stack_stats, breakdown_ctx);
            }
        }
    });
}


/**
 * @brief Import weights from external GPU pointers (zero-copy from vLLM).
 *
 * Quantized base weights are borrowed from external GPU memory (no disk I/O).
 * Non-quantized weights (norms, biases, embeddings) are loaded from SafeTensors on disk.
 *
 * @param safetensors_path Path to HuggingFace SafeTensors (for non-quantized weights).
 * @param per_gpu_weights  Per-GPU external weight descriptors (one vector per local GPU).
 */
void MultiGPUPyTrainer::import_weights_from_external(
        std::string safetensors_path,
        std::vector<std::vector<qlora::ExternalWeight>> per_gpu_weights) {
    if (per_gpu_weights.size() != mContexts.size()) {
        throw std::runtime_error(fmt::format(
            "import_weights_from_external: expected {} GPU weight sets, got {}",
            mContexts.size(), per_gpu_weights.size()));
    }

    // Distribute per-GPU weights to each context
    for (int i = 0; i < (int)mContexts.size(); ++i) {
        mContexts[i].Work = nullptr;  // clear any stale work
    }

    // Store per-GPU weights in a shared vector (captured by reference)
    run_work([this, &safetensors_path, &per_gpu_weights](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("import_weights_from_external: DSL model required");
        }

        const int local_rank = ctx.Communicator->local_rank();
        dsl_model->import_weights_from_external(
            safetensors_path,
            per_gpu_weights[local_rank],
            *ctx.Communicator);

        // Schedule deferred QLoRA offloading auto-tune
        dsl_model->auto_tune_offloading();
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
            save_pretrained_config(*ctx.Model->get_run_state().Config, (p / "config.json").c_str());
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
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("export_adapter: DSL model required");
        }
        if (!dsl_model->lora_enabled()) {
            throw std::runtime_error("export_adapter: DSL model is not configured for LoRA");
        }
        dsl_model->export_adapter(path, *ctx.Communicator, base_model_path);
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
void MultiGPUPyTrainer::step(const std::int32_t* inputs, const std::int32_t* targets, const std::int32_t* position_ids) {
    for(int i = 0; i < mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        if (!ctx.Model) {
            throw std::runtime_error(fmt::format("step: ctx[{}].Model is null", i));
        }
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t pos_stride = static_cast<std::size_t>(pos_planes) * static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            std::memcpy(pb, position_ids + i * pos_stride, pos_stride * sizeof(std::int32_t));
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if(mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(fmt::format("step: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
    }

    const bool do_timing = mOptions.TriggerTimingEvents;
    run_work([micro_idx = mTrainMicroStep, micro_batches = mGradAccumulation, do_timing](sThreadContext& ctx) {
        auto& rs = ctx.Model->get_run_state();
        if (do_timing && rs.TimingForwardStart.empty()) {
            rs.setup_timing_events(micro_batches);
        }
        Tensor inputs = ctx.Model->get_input_buffer();
        Tensor position_ids = ctx.Model->get_position_ids_buffer();
        Tensor targets = ctx.Model->get_target_buffer();
        if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingForwardStart[micro_idx], rs.MainStream));
        ctx.Model->forward(inputs, position_ids, *ctx.Communicator, micro_idx);
        if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingForwardEnd[micro_idx], rs.MainStream));
        if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingBackwardStart[micro_idx], rs.MainStream));
        try {
            ctx.Model->backward(inputs, targets, *ctx.Communicator, micro_batches, micro_idx);
        } catch (const std::exception& e) {
            std::cerr << "backward threw: " << e.what() << std::endl;
            throw;
        }
        if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingBackwardEnd[micro_idx], rs.MainStream));
    });
    ++mTrainMicroStep;
}

void MultiGPUPyTrainer::set_visual_inputs(const std::int32_t* visual_pos_masks,
                                          const float* visual_embeds,
                                          const std::vector<const float*>& deepstack_visual_embeds) {
    if (!mConfig || !mConfig->UseVisualInputs) {
        if (visual_pos_masks || visual_embeds || !deepstack_visual_embeds.empty()) {
            throw std::runtime_error("set_visual_inputs: visual inputs requested but model config has UseVisualInputs=false");
        }
        return;
    }

    const int world = local_world_size();
    const std::size_t mask_stride = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    const std::size_t embed_stride = mask_stride * static_cast<std::size_t>(mConfig->HiddenSize);
    for (int i = 0; i < world; ++i) {
        const std::int32_t* mask_ptr = visual_pos_masks ? (visual_pos_masks + i * mask_stride) : nullptr;
        const float* embed_ptr = visual_embeds ? (visual_embeds + i * embed_stride) : nullptr;
        std::vector<const float*> deepstack_ptrs;
        deepstack_ptrs.reserve(deepstack_visual_embeds.size());
        for (const float* base_ptr : deepstack_visual_embeds) {
            deepstack_ptrs.push_back(base_ptr ? (base_ptr + i * embed_stride) : nullptr);
        }
        run_work([mask_ptr, embed_ptr, deepstack_ptrs](sThreadContext& ctx) {
            auto& rs = ctx.Model->get_run_state();
            if (!rs.VisualPosMasks_CPU.Data || !rs.VisualEmbeds_CPU.Data) {
                if (mask_ptr || embed_ptr || !deepstack_ptrs.empty()) {
                    throw std::runtime_error("set_visual_inputs: visual buffers not allocated in run state");
                }
                return;
            }

            if (mask_ptr) {
                std::memcpy(rs.VisualPosMasks_CPU.Data, mask_ptr, rs.VisualPosMasks_CPU.bytes());
            } else {
                zero_tensor(rs.VisualPosMasks_CPU);
            }

            copy_from_float(rs.VisualEmbeds_CPU.Data, rs.VisualEmbeds_CPU.DType, embed_ptr,
                            rs.VisualEmbeds_CPU.nelem());

            if (!rs.DeepstackVisualEmbeds_CPU.empty()) {
                for (std::size_t j = 0; j < rs.DeepstackVisualEmbeds_CPU.size(); ++j) {
                    Tensor& dst = rs.DeepstackVisualEmbeds_CPU[j];
                    const float* src = (j < deepstack_ptrs.size()) ? deepstack_ptrs[j] : nullptr;
                    copy_from_float(dst.Data, dst.DType, src, dst.nelem());
                }
            }
        }, i);
    }
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
float MultiGPUPyTrainer::validate(const std::int32_t* inputs, const std::int32_t* targets, const std::int32_t* position_ids) {
    for(int i = 0; i < mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t pos_stride = static_cast<std::size_t>(pos_planes) * static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            std::memcpy(pb, position_ids + i * pos_stride, pos_stride * sizeof(std::int32_t));
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
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
 * @brief Apply optimizer update with full configuration support.
 *
 * Supports AdamW 8-bit and NorMuon optimizers based on config.type.
 * NorMuon uses orthogonalized momentum for 2D weights (attention, MLP)
 * and AdamW 8-bit for embeddings, norms, and lm_head.
 *
 * @param config Optimizer configuration with all hyperparameters.
 * @param step Zero-based global optimization step index (converted to 1-based internally).
 * @return Pair of (loss, grad_norm).
 */
std::pair<float, float> MultiGPUPyTrainer::update_with_config(const optimizers::OptimizerConfig& config, int step) {
    const bool do_timing = mOptions.TriggerTimingEvents;
    run_work([&, do_timing](sThreadContext& ctx) {
        auto& rs = ctx.Model->get_run_state();
        if (do_timing && rs.TimingOptimizerStart) {
            CUDA_CHECK(cudaEventRecord(rs.TimingOptimizerStart, rs.MainStream));
        }
        ctx.Model->update_with_config(*ctx.Communicator, config, step + 1);
        if (do_timing && rs.TimingOptimizerEnd) {
            CUDA_CHECK(cudaEventRecord(rs.TimingOptimizerEnd, rs.MainStream));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    if (do_timing) {
        print_timing_breakdown(step, mGradAccumulation);
    }

    float step_loss, step_norm;
    auto& ctx = mContexts.at(0);
    step_loss = ctx.Model->get_loss();
    step_norm = ctx.Model->get_norm();

    // ensure we're re-gathering on next forward for eval and train
    mTrainMicroStep = 0;
    mEvalStep = 0;

    return {step_loss, step_norm};
}

std::pair<float, float> MultiGPUPyTrainer::train_step_graphed(const std::int32_t* inputs,
                                                              const std::int32_t* targets,
                                                              const std::int32_t* position_ids,
                                                              const optimizers::OptimizerConfig& config,
                                                              int step) {
    const int local_gpus = static_cast<int>(mContexts.size());
    const int micro_steps = mGradAccumulation;
    const std::size_t stride = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    const int pos_planes = (mContexts.empty() || !mContexts.front().Model)
        ? 1
        : ((mContexts.front().Model->get_position_ids_buffer().Rank == 3)
              ? static_cast<int>(mContexts.front().Model->get_position_ids_buffer().Sizes[0])
              : 1);
    const std::size_t pos_stride = stride * static_cast<std::size_t>(pos_planes);

    run_work([&](sThreadContext& ctx) {
        auto& rs = ctx.Model->get_run_state();
        if (!rs.Allocator) {
            throw std::runtime_error("train_step_graphed: missing allocator");
        }

        if (!ctx.FullStepGraph) {
            ctx.FullStepGraph = std::make_unique<sFullStepGraphState>();
        }
        auto& gs = *ctx.FullStepGraph;

        // Reset graph if shape or accumulation changed.
        if (gs.captured && (gs.captured_B != B || gs.captured_T != T || gs.captured_grad_accum != micro_steps)) {
            if (gs.graph_exec) {
                CUDA_CHECK(cudaGraphExecDestroy(gs.graph_exec));
                gs.graph_exec = nullptr;
            }
            gs.captured = false;
        }

        // Allocate per-micro-step pinned buffers if needed.
        if (gs.inputs.size() != static_cast<size_t>(micro_steps) ||
            gs.targets.size() != static_cast<size_t>(micro_steps) ||
            gs.position_ids.size() != static_cast<size_t>(micro_steps) ||
            gs.captured_B != B || gs.captured_T != T) {
            gs.inputs.clear();
            gs.targets.clear();
            gs.position_ids.clear();
            gs.inputs.reserve(micro_steps);
            gs.targets.reserve(micro_steps);
            gs.position_ids.reserve(micro_steps);

            const int rank = ctx.Communicator->local_rank();
            for (int j = 0; j < micro_steps; ++j) {
                auto in_name = fmt::format("graph_inputs_cpu_ms{}_rank{}", j, rank);
                auto tgt_name = fmt::format("graph_targets_cpu_ms{}_rank{}", j, rank);
                auto pos_name = fmt::format("graph_pos_ids_cpu_ms{}_rank{}", j, rank);
                gs.inputs.push_back(rs.Allocator->allocate(ETensorDType::INT32, in_name.c_str(), EAllocationType::PINNED, {B, T}));
                gs.targets.push_back(rs.Allocator->allocate(ETensorDType::INT32, tgt_name.c_str(), EAllocationType::PINNED, {B, T}));
                if (pos_planes > 1) {
                    gs.position_ids.push_back(rs.Allocator->allocate(ETensorDType::INT32, pos_name.c_str(), EAllocationType::PINNED, {pos_planes, B, T}));
                } else {
                    gs.position_ids.push_back(rs.Allocator->allocate(ETensorDType::INT32, pos_name.c_str(), EAllocationType::PINNED, {B, T}));
                }
            }

            gs.captured_B = B;
            gs.captured_T = T;
            gs.captured_grad_accum = micro_steps;
        }

        // Allocate device-side optimizer parameter buffers if needed.
        // Use the maximum size to support both AdamW and NorMuon
        constexpr int max_opt_params = std::max(optimizers::ADAMW_GRAPH_PARAM_COUNT,
                                                 optimizers::NORMUON_GRAPH_PARAM_COUNT);
        if (!gs.opt_params.Data) {
            auto name = fmt::format("graph_opt_params_rank{}", ctx.Communicator->local_rank());
            gs.opt_params = rs.Allocator->allocate(ETensorDType::FP32, name.c_str(), EAllocationType::ON_DEVICE,
                                                  {max_opt_params});
        }
        if (!gs.opt_step.Data) {
            auto name = fmt::format("graph_opt_step_rank{}", ctx.Communicator->local_rank());
            gs.opt_step = rs.Allocator->allocate(ETensorDType::INT32, name.c_str(), EAllocationType::ON_DEVICE, {1});
        }

        // Stage inputs/targets/position_ids for all micro-steps.
        const int rank = ctx.Communicator->local_rank();
        for (int j = 0; j < micro_steps; ++j) {
            const std::size_t offset = (static_cast<std::size_t>(j) * static_cast<std::size_t>(local_gpus) + static_cast<std::size_t>(rank)) * stride;
            const std::size_t pos_offset = (static_cast<std::size_t>(j) * static_cast<std::size_t>(local_gpus) + static_cast<std::size_t>(rank)) * pos_stride;
            std::memcpy(gs.inputs[j].Data, inputs + offset, stride * sizeof(std::int32_t));
            std::memcpy(gs.targets[j].Data, targets + offset, stride * sizeof(std::int32_t));
            if (position_ids) {
                std::memcpy(gs.position_ids[j].Data, position_ids + pos_offset, pos_stride * sizeof(std::int32_t));
            } else {
                fill_sequential_position_ids(reinterpret_cast<std::int32_t*>(gs.position_ids[j].Data), pos_planes, B, T);
            }
        }

        // Update optimizer parameters on device (dynamic LR/step support).
        const int opt_step_host = step + 1;
        if (config.type == optimizers::OptimizerType::NORMUON) {
            // NorMuon graph params layout:
            // [0] = normuon_lr, [1] = normuon_momentum, [2] = normuon_beta2, [3] = weight_decay
            // [4] = adamw_lr, [5] = adamw_beta1, [6] = adamw_beta2, [7] = adamw_eps
            float opt_params_host[optimizers::NORMUON_GRAPH_PARAM_COUNT] = {
                config.normuon_lr > 0 ? config.normuon_lr : config.learning_rate,
                config.normuon_momentum,
                config.normuon_beta2,
                config.weight_decay,
                config.learning_rate,
                config.adamw_beta1,
                config.adamw_beta2,
                config.adamw_epsilon
            };
            CUDA_CHECK(cudaMemcpyAsync(gs.opt_params.Data, opt_params_host,
                                       sizeof(opt_params_host), cudaMemcpyHostToDevice, rs.MainStream));
        } else {
            float opt_params_host[optimizers::ADAMW_GRAPH_PARAM_COUNT] = {
                config.learning_rate, config.adamw_beta1, config.adamw_beta2, config.adamw_epsilon, config.weight_decay
            };
            CUDA_CHECK(cudaMemcpyAsync(gs.opt_params.Data, opt_params_host,
                                       sizeof(opt_params_host), cudaMemcpyHostToDevice, rs.MainStream));
        }
        CUDA_CHECK(cudaMemcpyAsync(gs.opt_step.Data, &opt_step_host,
                                   sizeof(opt_step_host), cudaMemcpyHostToDevice, rs.MainStream));

        // If graphs are disabled or unsupported, fall back to eager execution.
        if (!mOptions.UseCudaGraphs) {
            const bool do_timing = mOptions.TriggerTimingEvents;
            if (do_timing && rs.TimingForwardStart.empty()) {
                rs.setup_timing_events(micro_steps);
            }
            for (int j = 0; j < micro_steps; ++j) {
                rs.Targets_CPU = gs.targets[j];
                if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingForwardStart[j], rs.MainStream));
                ctx.Model->forward(gs.inputs[j], gs.position_ids[j], *ctx.Communicator, j);
                if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingForwardEnd[j], rs.MainStream));
                if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingBackwardStart[j], rs.MainStream));
                ctx.Model->backward(gs.inputs[j], gs.targets[j], *ctx.Communicator, micro_steps, j);
                if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingBackwardEnd[j], rs.MainStream));
            }
            if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingOptimizerStart, rs.MainStream));
            ctx.Model->update_with_config(*ctx.Communicator, config, opt_step_host);
            if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingOptimizerEnd, rs.MainStream));
            CUDA_CHECK(cudaDeviceSynchronize());
            return;
        }

        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("train_step_graphed: only supported for DSL models");
        }
        if (config.type != optimizers::OptimizerType::ADAMW_8BIT &&
            config.type != optimizers::OptimizerType::NORMUON) {
            throw std::runtime_error("train_step_graphed: only supports AdamW 8-bit or NorMuon optimizer");
        }

        // CUDA graph capture path (both AdamW and NorMuon support graph capture)
        const bool warmup_full_graph = !gs.captured
                                       && !env_enabled("SUROGATE_DSL_GRAPH_SKIP_WARMUP");
        const bool warmup_skip_bwd = env_enabled("SUROGATE_DSL_GRAPH_WARMUP_SKIP_BWD");
        const bool prev_internal_graphs = dsl_model->internal_graphs_enabled();
        if (prev_internal_graphs) {
            dsl_model->set_internal_graphs_enabled(false);
        }
        if (warmup_full_graph) {
            auto rng_state = dsl_model->rng_state();
            for (int j = 0; j < micro_steps; ++j) {
                rs.Targets_CPU = gs.targets[j];
                dsl_model->forward(gs.inputs[j], gs.position_ids[j], *ctx.Communicator, j);
                if (!warmup_skip_bwd) {
                    dsl_model->backward(gs.inputs[j], gs.targets[j], *ctx.Communicator, micro_steps, j);
                }
            }
            dsl_model->zero_grads(rs.MainStream);
            dsl_model->set_rng_state(rng_state);
            CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
        }

        if (!gs.captured) {
            dsl_model->prepare_optimizer_state_for_graph(*ctx.Communicator, config);
            // Ensure the main stream is idle before beginning capture (no external dependencies).
            CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
            cudaGraph_t graph = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(rs.MainStream, cudaStreamCaptureModeThreadLocal));
            for (int j = 0; j < micro_steps; ++j) {
                rs.Targets_CPU = gs.targets[j];
                dsl_model->forward(gs.inputs[j], gs.position_ids[j], *ctx.Communicator, j);
                dsl_model->backward(gs.inputs[j], gs.targets[j], *ctx.Communicator, micro_steps, j);
            }
            dsl_model->update_with_graph_params(*ctx.Communicator, config,
                                               gs.opt_params.template get<float>(),
                                               gs.opt_step.template get<int>());
            CUDA_CHECK(cudaStreamEndCapture(rs.MainStream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&gs.graph_exec, graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            gs.captured = true;
        }

        if (prev_internal_graphs) {
            dsl_model->set_internal_graphs_enabled(true);
        }

        CUDA_CHECK(cudaGraphLaunch(gs.graph_exec, rs.MainStream));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Refresh loss/norm on host after full-step graph launch.
        CUDA_CHECK(cudaMemcpy(rs.LossHost, rs.Losses.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost));
        auto& dsl_rs = dynamic_cast<dsl::DslRunState&>(dsl_model->get_run_state());
        if (dsl_model->lora_enabled()) {
            auto& lora_rs = dsl_model->lora_run_state();
            CUDA_CHECK(cudaMemcpy(dsl_rs.NormHost, lora_rs.norm_buffer.template get<float>(),
                                  sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpy(dsl_rs.NormHost, dsl_rs.scratch().norm_buffer.template get<float>(),
                                  sizeof(float), cudaMemcpyDeviceToHost));
        }
    });

    // Post-step memory breakdown (step 1 only, after optimizer states are allocated)
    if (step == 1 && mOptions.DebugMemoryBreakdown) {
        auto& ctx0 = mContexts.at(0);
        auto& rs0 = ctx0.Model->get_run_state();
        if (rs0.Allocator) {
            auto stats = rs0.Allocator->get_allocation_segments();
            auto stack_stats = rs0.Stack.get_allocation_stats();
            MemoryBreakdownContext breakdown_ctx;
            breakdown_ctx.enabled = true;
            breakdown_ctx.allocator = rs0.Allocator.get();
            breakdown_ctx.hidden_size = mConfig->HiddenSize;
            breakdown_ctx.intermediate_size = mConfig->IntermediateSize;
            breakdown_ctx.num_layers = mConfig->NumLayers;
            breakdown_ctx.batch_size = B;
            breakdown_ctx.seq_length = T;

            auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx0.Model.get());
            if (dsl_model && dsl_model->qlora_enabled()) {
                breakdown_ctx.qlora_quantized_bytes = dsl_model->qlora_quantized_weights_bytes();
                breakdown_ctx.qlora_savings_ratio = dsl_model->qlora_memory_savings_ratio();
            }

            fprintf(stderr, "\n[Post-Step-0 Memory Breakdown]\n");
            TrainingRunLogger logger("", 0, TrainingRunLogger::VERBOSE);
            logger.log_allocator(stats, stack_stats, breakdown_ctx);
        }
    }

    if (mOptions.TriggerTimingEvents) {
        print_timing_breakdown(step, micro_steps);
    }

    auto& ctx = mContexts.at(0);
    float step_loss = ctx.Model->get_loss();
    float step_norm = ctx.Model->get_norm();

    mTrainMicroStep = 0;
    mEvalStep = 0;

    return {step_loss, step_norm};
}


/**
 * @brief Print a per-phase timing breakdown for the last training step.
 *
 * Uses CUDA timing events recorded around forward, backward, and optimizer phases
 * to compute and print elapsed time for each phase. Only reads events from rank 0.
 *
 * @param step Global step index (for display).
 * @param micro_steps Number of gradient accumulation micro-steps.
 */
void MultiGPUPyTrainer::print_timing_breakdown(int step, int micro_steps) {
    auto& rs = mContexts.at(0).Model->get_run_state();
    const int n = std::min(micro_steps, static_cast<int>(rs.TimingForwardStart.size()));
    if (n == 0) return;

    float total_fwd_ms = 0, total_bwd_ms = 0, opt_ms = 0;
    std::vector<float> fwd_ms(n), bwd_ms(n);

    for (int j = 0; j < n; ++j) {
        if (!rs.TimingForwardStart[j] || !rs.TimingForwardEnd[j]) break;
        CUDA_CHECK(cudaEventElapsedTime(&fwd_ms[j], rs.TimingForwardStart[j], rs.TimingForwardEnd[j]));
        total_fwd_ms += fwd_ms[j];
        if (rs.TimingBackwardStart[j] && rs.TimingBackwardEnd[j]) {
            CUDA_CHECK(cudaEventElapsedTime(&bwd_ms[j], rs.TimingBackwardStart[j], rs.TimingBackwardEnd[j]));
            total_bwd_ms += bwd_ms[j];
        }
    }
    if (rs.TimingOptimizerStart && rs.TimingOptimizerEnd) {
        CUDA_CHECK(cudaEventElapsedTime(&opt_ms, rs.TimingOptimizerStart, rs.TimingOptimizerEnd));
    }

    float total_ms = total_fwd_ms + total_bwd_ms + opt_ms;
    fprintf(stderr, "[Time Breakdown] step=%d  fwd: %.1fms  bwd: %.1fms  opt: %.1fms  total: %.1fms",
            step, total_fwd_ms, total_bwd_ms, opt_ms, total_ms);

    if (n > 1) {
        fprintf(stderr, "\n");
        for (int j = 0; j < n; ++j) {
            fprintf(stderr, "  micro[%d] fwd: %.1fms  bwd: %.1fms\n", j, fwd_ms[j], bwd_ms[j]);
        }
    } else {
        fprintf(stderr, "\n");
    }
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
 * @brief Get MoE training statistics from the last forward pass.
 *
 * Returns accumulated MoE metrics from rank 0's run state. For non-MoE models,
 * returns zeros with valid=false.
 *
 * @return Tuple of (aux_loss, z_loss, expert_utilization, load_imbalance, valid)
 */
std::tuple<float, float, float, float, bool> MultiGPUPyTrainer::get_moe_stats() {
    auto& ctx = mContexts.at(0);
    auto& rs = ctx.Model->get_run_state();
    auto stats = rs.get_moe_stats();
    return {stats.aux_loss, stats.z_loss, stats.expert_utilization, stats.load_imbalance, stats.valid};
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
    static int work_id = 0;
    int current_work_id = work_id++;
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
    sThreadContext& ctx = mContexts.at(comm.local_rank());

    ctx.Communicator = &comm;
    ctx.GPUUtil = IGPUUtilTracker::create();

    // Initialize EP sub-communicators if EP is enabled
    if (mOptions.EPSize > 1) {
        comm.init_ep_groups(mOptions.EPSize);
    }

    auto allocator = std::make_shared<TensorAllocator>();

    if (mLoRAConfig.has_value()) {
        // Convert LoRAAdapterConfig -> modular LoRA config
        modules::ModularLoRAConfig mod_lora;
        mod_lora.rank = mLoRAConfig->Rank;
        mod_lora.alpha = mLoRAConfig->Alpha;
        mod_lora.dropout = mLoRAConfig->Dropout;
        mod_lora.dtype = mLoRAConfig->DType;
        mod_lora.init_a_kaiming = mLoRAConfig->InitAKaimingUniform;
        mod_lora.use_rs_lora = mLoRAConfig->UseRSLoRA;
        mod_lora.train_router = mLoRAConfig->TrainRouter;
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
                else if (name == "gate_up_proj") mod_lora.targets.insert(modules::LoRATarget::GATE_UP_PROJ);
                else if (name == "up_proj") mod_lora.targets.insert(modules::LoRATarget::UP_PROJ);
                else if (name == "down_proj") mod_lora.targets.insert(modules::LoRATarget::DOWN_PROJ);
            }
        }

        // Build QLoRA config if provided
        modules::QLoRAConfig qlora_config;
        if (mQLoRAConfig.has_value()) {
            qlora_config = mQLoRAConfig.value();
        }

        // Use factory to create LoRA model with proper architecture dispatch
        ctx.Model = modules::ModelFactory::create_lora_from_pretrained_config(
            *mConfig, mod_lora, mOptions, comm, allocator, qlora_config);
    } else {
        // Use factory to create base model with proper architecture dispatch
        ctx.Model = modules::ModelFactory::create_from_pretrained_config(
            *mConfig, mOptions, comm.rank(), comm.world_size(), allocator);
    }

    // DEBUG: GPU memory after model creation (before run state)
    if (mOptions.DebugMemoryBreakdown && comm.rank() == 0) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "[DEBUG-MEM] After model creation: GPU used="
                  << (total_mem - free_mem) / (1024*1024) << " MiB, free="
                  << free_mem / (1024*1024) << " MiB, total="
                  << total_mem / (1024*1024) << " MiB" << std::endl;
    }

    ctx.Model->allocate_run_state(mOptions, comm, B, T, /*allocate_optimizer=*/true);

    // DEBUG: GPU memory after run state allocation
    if (mOptions.DebugMemoryBreakdown && comm.rank() == 0) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "[DEBUG-MEM] After run state alloc: GPU used="
                  << (total_mem - free_mem) / (1024*1024) << " MiB, free="
                  << free_mem / (1024*1024) << " MiB" << std::endl;
    }

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

    // Use local GPU count, not global world_size. Each node has its own trainer instance
    // with its own mIsReady counter, so we sync when all LOCAL threads are ready.
    if (mIsReady.fetch_add(1) == static_cast<int>(mContexts.size()) - 1) {
        mIsRunning = true;
    };

    while (!mIsRunning.load()) {
        std::this_thread::yield();
        if(mHasCrashed.load()) throw std::runtime_error("Another worker has crashed, exiting.");
    }

    int loop_iteration = 0;
    while (mIsRunning.load()) {
        if (auto work = fetch_work(ctx); work) {
            try {
                work(ctx);
            } catch (const std::exception& e) {
                std::cerr << "work threw exception: " << e.what() << std::endl;
                throw;
            }
            mWorkDone.fetch_add(1);
        } else {
            std::this_thread::yield();
        }
        loop_iteration++;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();

    // free resources
    if (ctx.FullStepGraph) {
        if (ctx.FullStepGraph->graph_exec) {
            CUDA_CHECK(cudaGraphExecDestroy(ctx.FullStepGraph->graph_exec));
            ctx.FullStepGraph->graph_exec = nullptr;
        }
        ctx.FullStepGraph.reset();
    }
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
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("get_gradients: DSL model required");
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto& grads = dsl_model->grads();
        const auto& grad_map = grads.grads();
        result.reserve(grads.param_names().size());
        for (const auto& name : grads.param_names()) {
            auto it = grad_map.find(name);
            if (it == grad_map.end()) {
                continue;
            }
            result.emplace_back(name, it->second);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }, gpu_id);
    return result;
}

std::vector<std::pair<std::string, Tensor>> MultiGPUPyTrainer::get_lora_gradients(int gpu_id) {
    std::vector<std::pair<std::string, Tensor>> result;
    run_work([&result](sThreadContext& ctx) {
        // Helper to add LoRA layer gradients
        auto add_layer = [&](const std::string& module_prefix,
                             const std::optional<modules::LoRALayerWeights<Tensor>>& layer) {
            if (!layer.has_value()) return;
            if (layer->A.Data) result.emplace_back(module_prefix + ".lora_A.weight", layer->A);
            if (layer->B.Data) result.emplace_back(module_prefix + ".lora_B.weight", layer->B);
        };
        auto add_grouped_layer = [&](const std::string& module_prefix,
                                     const std::optional<modules::LoRAGroupedLayerWeights<Tensor>>& layer) {
            if (!layer.has_value()) return;
            if (layer->A.Data) result.emplace_back(module_prefix + ".lora_A.weight", layer->A);
            if (layer->B.Data) result.emplace_back(module_prefix + ".lora_B.weight", layer->B);
        };
        auto contains_ci = [](std::string_view haystack, std::string_view needle) {
            auto to_lower = [](std::string_view in) {
                std::string out(in);
                for (auto& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                return out;
            };
            const std::string h = to_lower(haystack);
            const std::string n = to_lower(needle);
            return h.find(n) != std::string::npos;
        };
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("get_lora_gradients: DSL model required");
        }
        if (!dsl_model->lora_enabled()) {
            throw std::runtime_error("get_lora_gradients: DSL model is not configured for LoRA");
        }
        const auto& config = *ctx.Model->get_run_state().Config;
        const bool is_nemotron = (config.Architecture == PretrainedConfig::NEMOTRON_H) ||
                                 contains_ci(config.ArchitectureName, "nemotron") ||
                                 contains_ci(config.ModelTypeName, "nemotron");
        const bool is_moe = dsl_model->is_moe_model();
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int l = 0; l < config.NumLayers; ++l) {
            bool unused_accumulate = false;
            auto& block = dsl_model->lora_grads().get_block_full(l, /*stream=*/nullptr, *ctx.Communicator, unused_accumulate);
            std::string prefix;
            if (is_nemotron) {
                prefix = fmt::format("base_model.model.backbone.layers.{}", l);
            } else {
                prefix = fmt::format("base_model.model.model.layers.{}", l);
            }

            // Attention LoRA (same for dense and MoE)
            if (is_nemotron) {
                const std::string mixer_prefix = prefix + ".mixer";
                add_layer(mixer_prefix + ".q_proj", block.attention.q);
                add_layer(mixer_prefix + ".k_proj", block.attention.k);
                add_layer(mixer_prefix + ".v_proj", block.attention.v);
                add_layer(mixer_prefix + ".o_proj", block.attention.o);
            } else {
                add_layer(prefix + ".self_attn.q_proj", block.attention.q);
                add_layer(prefix + ".self_attn.k_proj", block.attention.k);
                add_layer(prefix + ".self_attn.v_proj", block.attention.v);
                add_layer(prefix + ".self_attn.o_proj", block.attention.o);
            }

            if (is_moe) {
                if (block.moe.use_grouped) {
                    std::string expert_prefix;
                    if (is_nemotron) {
                        expert_prefix = fmt::format("{}.mixer.experts", prefix);
                    } else {
                        expert_prefix = fmt::format("{}.mlp.experts", prefix);
                    }
                    add_grouped_layer(expert_prefix + ".gate_proj", block.moe.grouped.gate);
                    add_grouped_layer(expert_prefix + ".gate_up_proj", block.moe.grouped.gate_up);
                    add_grouped_layer(expert_prefix + ".up_proj", block.moe.grouped.up);
                    add_grouped_layer(expert_prefix + ".down_proj", block.moe.grouped.down);
                } else {
                for (int e = 0; e < (int)block.moe.experts.size(); ++e) {
                    auto& expert = block.moe.experts[e];
                    std::string expert_prefix;
                    if (is_nemotron) {
                        expert_prefix = fmt::format("{}.mixer.experts.{}", prefix, e);
                    } else {
                        expert_prefix = fmt::format("{}.mlp.experts.{}", prefix, e);
                    }
                    add_layer(expert_prefix + ".gate_proj", expert.gate);
                    add_layer(expert_prefix + ".gate_up_proj", expert.gate_up);
                    add_layer(expert_prefix + ".up_proj", expert.up);
                    add_layer(expert_prefix + ".down_proj", expert.down);
                }
                }
            } else {
                if (is_nemotron) {
                    const std::string mixer_prefix = prefix + ".mixer";
                    add_layer(mixer_prefix + ".gate_proj", block.mlp.gate);
                    add_layer(mixer_prefix + ".up_proj", block.mlp.up);
                    add_layer(mixer_prefix + ".down_proj", block.mlp.down);
                } else {
                    add_layer(prefix + ".mlp.gate_proj", block.mlp.gate);
                    add_layer(prefix + ".mlp.up_proj", block.mlp.up);
                    add_layer(prefix + ".mlp.down_proj", block.mlp.down);
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }, gpu_id);
    return result;
}

std::vector<std::pair<std::string, Tensor>> MultiGPUPyTrainer::get_lora_weights(int gpu_id) {
    std::vector<std::pair<std::string, Tensor>> result;
    run_work([&result](sThreadContext& ctx) {
        auto add_layer = [&](const std::string& module_prefix,
                             const std::optional<modules::LoRALayerWeights<Tensor>>& layer) {
            if (!layer.has_value()) return;
            if (layer->A.Data) result.emplace_back(module_prefix + ".lora_A.weight", layer->A);
            if (layer->B.Data) result.emplace_back(module_prefix + ".lora_B.weight", layer->B);
        };
        auto add_grouped_layer = [&](const std::string& module_prefix,
                                     const std::optional<modules::LoRAGroupedLayerWeights<Tensor>>& layer) {
            if (!layer.has_value()) return;
            if (layer->A.Data) result.emplace_back(module_prefix + ".lora_A.weight", layer->A);
            if (layer->B.Data) result.emplace_back(module_prefix + ".lora_B.weight", layer->B);
        };
        auto contains_ci = [](std::string_view haystack, std::string_view needle) {
            auto to_lower = [](std::string_view in) {
                std::string out(in);
                for (auto& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                return out;
            };
            const std::string h = to_lower(haystack);
            const std::string n = to_lower(needle);
            return h.find(n) != std::string::npos;
        };
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("get_lora_weights: DSL model required");
        }
        if (!dsl_model->lora_enabled()) {
            throw std::runtime_error("get_lora_weights: DSL model is not configured for LoRA");
        }
        const auto& config = *ctx.Model->get_run_state().Config;
        const bool is_nemotron = (config.Architecture == PretrainedConfig::NEMOTRON_H) ||
                                 contains_ci(config.ArchitectureName, "nemotron") ||
                                 contains_ci(config.ModelTypeName, "nemotron");
        const bool is_moe = dsl_model->is_moe_model();
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int l = 0; l < config.NumLayers; ++l) {
            auto& block = dsl_model->lora_weights().get_block(l, /*stream=*/nullptr);
            std::string prefix;
            if (is_nemotron) {
                prefix = fmt::format("base_model.model.backbone.layers.{}", l);
            } else {
                prefix = fmt::format("base_model.model.model.layers.{}", l);
            }

            // Attention LoRA
            if (is_nemotron) {
                const std::string mixer_prefix = prefix + ".mixer";
                add_layer(mixer_prefix + ".q_proj", block.attention.q);
                add_layer(mixer_prefix + ".k_proj", block.attention.k);
                add_layer(mixer_prefix + ".v_proj", block.attention.v);
                add_layer(mixer_prefix + ".o_proj", block.attention.o);
            } else {
                add_layer(prefix + ".self_attn.q_proj", block.attention.q);
                add_layer(prefix + ".self_attn.k_proj", block.attention.k);
                add_layer(prefix + ".self_attn.v_proj", block.attention.v);
                add_layer(prefix + ".self_attn.o_proj", block.attention.o);
            }

            if (is_moe) {
                if (block.moe.use_grouped) {
                    std::string expert_prefix;
                    if (is_nemotron) {
                        expert_prefix = fmt::format("{}.mixer.experts", prefix);
                    } else {
                        expert_prefix = fmt::format("{}.mlp.experts", prefix);
                    }
                    add_grouped_layer(expert_prefix + ".gate_proj", block.moe.grouped.gate);
                    add_grouped_layer(expert_prefix + ".gate_up_proj", block.moe.grouped.gate_up);
                    add_grouped_layer(expert_prefix + ".up_proj", block.moe.grouped.up);
                    add_grouped_layer(expert_prefix + ".down_proj", block.moe.grouped.down);
                } else {
                    for (int e = 0; e < (int)block.moe.experts.size(); ++e) {
                        auto& expert = block.moe.experts[e];
                        std::string expert_prefix;
                        if (is_nemotron) {
                            expert_prefix = fmt::format("{}.mixer.experts.{}", prefix, e);
                        } else {
                            expert_prefix = fmt::format("{}.mlp.experts.{}", prefix, e);
                        }
                        add_layer(expert_prefix + ".gate_proj", expert.gate);
                        add_layer(expert_prefix + ".gate_up_proj", expert.gate_up);
                        add_layer(expert_prefix + ".up_proj", expert.up);
                        add_layer(expert_prefix + ".down_proj", expert.down);
                    }
                }
            } else {
                if (is_nemotron) {
                    const std::string mixer_prefix = prefix + ".mixer";
                    add_layer(mixer_prefix + ".gate_proj", block.mlp.gate);
                    add_layer(mixer_prefix + ".up_proj", block.mlp.up);
                    add_layer(mixer_prefix + ".down_proj", block.mlp.down);
                } else {
                    add_layer(prefix + ".mlp.gate_proj", block.mlp.gate);
                    add_layer(prefix + ".mlp.up_proj", block.mlp.up);
                    add_layer(prefix + ".mlp.down_proj", block.mlp.down);
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }, gpu_id);
    return result;
}

std::vector<float> MultiGPUPyTrainer::compute_logprobs(const std::int32_t* input_ids,
                                                        const std::int32_t* targets,
                                                        int B, int T, bool use_lora,
                                                        const std::int32_t* position_ids,
                                                        const float* temperatures) {
    std::vector<float> result;
    run_work([&result, input_ids, targets, B, T, use_lora, position_ids, temperatures](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("compute_logprobs: model is not a DslModel");
        }
        auto logprobs = dsl_model->compute_logprobs(input_ids, targets, B, T, use_lora,
                                                    *ctx.Communicator, position_ids, temperatures);
        if (ctx.Communicator->local_rank() == 0) {
            result = std::move(logprobs);
        }
    });
    return result;
}

void MultiGPUPyTrainer::step_with_custom_loss(
        const std::int32_t* inputs,
        const std::int32_t* targets,
        const float* per_token_grads,
        const std::int32_t* position_ids,
        const float* temperatures) {
    // Distribute inputs, targets, and position_ids to each GPU's CPU-side buffers.
    for (int i = 0; i < (int)mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        if (!ctx.Model) {
            throw std::runtime_error(fmt::format("step_with_custom_loss: ctx[{}].Model is null", i));
        }
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t pos_stride = static_cast<std::size_t>(pos_planes) *
                                       static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            std::memcpy(pb, position_ids + i * pos_stride, pos_stride * sizeof(std::int32_t));
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(fmt::format("step_with_custom_loss: micro_step {} >= grad_accumulation {}",
                                             mTrainMicroStep, mGradAccumulation));
    }

    run_work([micro_idx = mTrainMicroStep, micro_batches = mGradAccumulation,
              per_token_grads, temperatures, B = this->B, T = this->T](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("step_with_custom_loss: model is not a DslModel");
        }

        Tensor inputs_tensor = ctx.Model->get_input_buffer();
        Tensor position_ids_tensor = ctx.Model->get_position_ids_buffer();
        Tensor targets_tensor = ctx.Model->get_target_buffer();

        // Each GPU receives its own slice of the per_token_grads buffer.
        const int gpu_rank = ctx.Communicator->local_rank();
        const float* grads_for_this_gpu = per_token_grads +
                                          static_cast<std::ptrdiff_t>(gpu_rank) * B * T;
        const float* temps_for_this_gpu = nullptr;
        if (temperatures) {
            temps_for_this_gpu = temperatures + static_cast<std::ptrdiff_t>(gpu_rank) * B * T;
        }

        dsl_model->step_with_custom_loss(inputs_tensor, position_ids_tensor, targets_tensor,
                                          grads_for_this_gpu, micro_batches, micro_idx,
                                          *ctx.Communicator, temps_for_this_gpu);
    });

    ++mTrainMicroStep;
}

int MultiGPUPyTrainer::get_valid_token_count(int gpu_id) {
    int result = 0;
    run_work([&result](sThreadContext& ctx) {
        auto& rs = ctx.Model->get_run_state();
        if (!rs.ValidTokenCount.Data) {
            result = 0;
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(&result, rs.ValidTokenCount.Data, sizeof(int),
                                   cudaMemcpyDeviceToHost, rs.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
    }, gpu_id);
    return result;
}
