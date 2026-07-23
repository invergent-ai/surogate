// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "py_train.h"

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <thread>
#include <tuple>
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
#include "runtime/dsl/shared_master_store.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/dsl_grad_store.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/executor/graph_executor.h"
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

inline int host_batch_row_for_local_rank(int local_rank, int ep_size) {
    // Default EP behavior: ranks in the same EP group consume the same host row.
    // This keeps non-expert forward/backward numerically aligned across EP ranks.
    if (ep_size > 1) {
        return local_rank / ep_size;
    }
    return local_rank;
}

inline int round_up_pow2_int(int value) {
    if (value <= 1) return 1;
    int rounded = 1;
    while (rounded < value && rounded < (1 << 30)) {
        rounded <<= 1;
    }
    return rounded;
}

long env_long_mb(const char* name, long default_value) {
    const char* value = std::getenv(name);
    if (!value) {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return (end != value && parsed >= 0) ? parsed : default_value;
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
        default: throw std::runtime_error("set_visual_inputs: unsupported dtype for visual embeds");
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
MultiGPUPyTrainer::MultiGPUPyTrainer(int ngpus,
                                     const PretrainedConfig& config,
                                     RuntimeOptions options,
                                     int batch_size,
                                     int seq_len,
                                     int grad_accum,
                                     bool memcpy_all_gather,
                                     bool memcpy_send_recv,
                                     std::optional<LoRAAdapterConfig> lora_config,
                                     std::optional<modules::QLoRAConfig> qlora_config)
    : mConfig(config.clone()),
      mOptions(options),
      mLoRAConfig(lora_config),
      mQLoRAConfig(qlora_config),
      B(batch_size),
      T(seq_len),
      mGradAccumulation(grad_accum) {
    int gpus_available = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpus_available));
    if (ngpus == 0) {
        ngpus = gpus_available;
    }

    if (ngpus > gpus_available) {
        throw std::runtime_error(fmt::format("Requested {} GPUs, only {} available", ngpus, gpus_available));
    }
    mContexts.resize(ngpus);
    init_async_slots(ngpus);
    mThreads =
        NCCLCommunicator::launch_communicators(ngpus, memcpy_all_gather, memcpy_send_recv, [&](NCCLCommunicator& comm) {
            try {
                this->main_loop(comm);
            } catch (...) {
                mHasCrashed = true;
                throw;
            }
        });

    while (!mIsRunning && !mHasCrashed) {
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
MultiGPUPyTrainer::MultiGPUPyTrainer(int ngpus,
                                     int node_rank,
                                     int num_nodes,
                                     const void* nccl_id,
                                     const PretrainedConfig& config,
                                     RuntimeOptions options,
                                     int batch_size,
                                     int seq_len,
                                     int grad_accum,
                                     bool memcpy_all_gather,
                                     bool memcpy_send_recv,
                                     std::optional<LoRAAdapterConfig> lora_config,
                                     std::optional<modules::QLoRAConfig> qlora_config)
    : mConfig(config.clone()),
      mOptions(options),
      mLoRAConfig(lora_config),
      mQLoRAConfig(qlora_config),
      B(batch_size),
      T(seq_len),
      mGradAccumulation(grad_accum) {
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
    init_async_slots(ngpus);
    mThreads = NCCLCommunicator::launch_communicators_multinode(ngpus,
                                                                node_rank,
                                                                num_nodes,
                                                                nccl_id_owned.data(),
                                                                memcpy_all_gather,
                                                                memcpy_send_recv,
                                                                [&](NCCLCommunicator& comm) {
                                                                    try {
                                                                        this->main_loop(comm);
                                                                    } catch (...) {
                                                                        mHasCrashed = true;
                                                                        throw;
                                                                    }
                                                                });

    while (!mIsRunning && !mHasCrashed) {
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

    const bool crashed = mHasCrashed.load() || (mThreads && mThreads->has_exception());
    if (crashed) {
        // Workers may be stuck in NCCL collectives waiting for a crashed rank,
        // and their pending kernels would also make cudaDeviceSynchronize hang.
        // Abort the comms so everyone can exit, and skip the device syncs. If
        // they stay wedged (driver-lock deadlocks inside NCCL), force-exit: a
        // dead process releases its GPUs, a deadlocked one pins them forever.
        mThreads->abort_async();
        if (!mThreads->wait_exit_for(15000)) {
            fprintf(stderr,
                    "FATAL: MultiGPUPyTrainer workers wedged in NCCL teardown after a crash; "
                    "forcing process exit to avoid a deadlocked trainer\n");
            fflush(nullptr);
            std::_Exit(134);
        }
    } else {
        // make sure all work has finished
        // Use local_rank() for cudaSetDevice, and don't throw from destructor
        for (auto& ctx : mContexts) {
            if (ctx.Communicator) {
                cudaError_t err = cudaSetDevice(ctx.Communicator->local_rank());
                if (err == cudaSuccess) {
                    cudaDeviceSynchronize();
                }
                // Ignore errors - we're in destructor, possibly after a crash
            }
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    try {
        mThreads->join();
    } catch (const std::exception& e) {
        // The exception was already surfaced to the caller via run_work/wait_gpu;
        // rethrowing from a destructor would terminate the process.
        fprintf(stderr, "MultiGPUPyTrainer teardown after error: %s\n", e.what());
        fflush(stderr);
    } catch (...) {
        fprintf(stderr, "MultiGPUPyTrainer teardown after unknown error\n");
        fflush(stderr);
    }

    // Free the cross-GPU shared base masters (after all streaming work has finished).
    dsl::shared_master_store().clear();
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
void MultiGPUPyTrainer::import_weights_from_external(std::string safetensors_path,
                                                     std::vector<std::vector<qlora::ExternalWeight>> per_gpu_weights) {
    if (per_gpu_weights.size() != mContexts.size()) {
        throw std::runtime_error(fmt::format("import_weights_from_external: expected {} GPU weight sets, got {}",
                                             mContexts.size(),
                                             per_gpu_weights.size()));
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
        dsl_model->import_weights_from_external(safetensors_path, per_gpu_weights[local_rank], *ctx.Communicator);

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
    run_work([](sThreadContext& ctx) { ctx.Model->init_weights(*ctx.Communicator); });
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
 * - `inputs` contains `local_gpus * B * T` int32 tokens laid out contiguously.
 * - EP disabled: rank `i` reads row `i`.
 * - EP enabled: ranks in the same EP group read the same row (`row = local_rank / ep_size`).
 * Same rule applies to `targets`.
 *
 * @param inputs Pointer to host int32 token IDs for all ranks (see layout above).
 * @param targets Pointer to host int32 target token IDs for all ranks (see layout above).
 *
 * @throws std::runtime_error If called more than `grad_accum` times without an update().
 */
void MultiGPUPyTrainer::step(const std::int32_t* inputs,
                             const std::int32_t* targets,
                             const std::int32_t* position_ids) {
    if (mOptions.SequenceChunks > 1) {
        step_chunked(inputs, targets, position_ids, mOptions.SequenceChunks);
        return;
    }
    const int ep_size = std::max(1, mOptions.EPSize);
    for (int i = 0; i < mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        if (!ctx.Model) {
            throw std::runtime_error(fmt::format("step: ctx[{}].Model is null", i));
        }
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
        const int src_row = host_batch_row_for_local_rank(i, ep_size);

        std::memcpy(ib, inputs + src_row * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + src_row * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            // Python binding provides 2D [B, T] position IDs (one plane per GPU).
            // For mRoPE models the buffer is [3, B, T] — replicate the single plane.
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(src_row) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(
            fmt::format("step: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
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

namespace {

/// Slice one chunk out of [rows, B, N*T] host arrays into the per-model
/// input/target/position buffers. Position ids get the chunk's global offset
/// so RoPE sees absolute positions.
void copy_chunk_inputs_for_ctx(IModel& model,
                               const std::int32_t* inputs,
                               const std::int32_t* targets,
                               const std::int32_t* position_ids,
                               int src_row,
                               int B,
                               int T,
                               long T_full,
                               long base_pos) {
    auto* ib = model.get_input_buffer().get<std::int32_t>();
    auto* tb = model.get_target_buffer().get<std::int32_t>();
    Tensor pos_buf = model.get_position_ids_buffer();
    auto* pb = pos_buf.get<std::int32_t>();
    const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;

    for (int b = 0; b < B; ++b) {
        const long src_off = (static_cast<long>(src_row) * B + b) * T_full + base_pos;
        std::memcpy(ib + static_cast<long>(b) * T, inputs + src_off, T * sizeof(std::int32_t));
        std::memcpy(tb + static_cast<long>(b) * T, targets + src_off, T * sizeof(std::int32_t));
    }
    for (int p = 0; p < pos_planes; ++p) {
        for (int b = 0; b < B; ++b) {
            std::int32_t* dst = pb + (static_cast<long>(p) * B + b) * T;
            if (position_ids) {
                const long src_off = (static_cast<long>(src_row) * B + b) * T_full + base_pos;
                std::memcpy(dst, position_ids + src_off, T * sizeof(std::int32_t));
            } else {
                for (int t = 0; t < T; ++t) {
                    dst[t] = static_cast<std::int32_t>(base_pos + t);
                }
            }
        }
    }
}

}  // namespace

void MultiGPUPyTrainer::step_chunked(const std::int32_t* inputs,
                                     const std::int32_t* targets,
                                     const std::int32_t* position_ids,
                                     int seq_chunks) {
    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(
            fmt::format("step_chunked: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
    }
    const int ep_size = std::max(1, mOptions.EPSize);
    const long T_full = static_cast<long>(T) * seq_chunks;
    const int accum_eff = mGradAccumulation * seq_chunks;

    auto copy_chunk = [&](int c) {
        const long base_pos = static_cast<long>(c) * T;
        for (int i = 0; i < static_cast<int>(mContexts.size()); ++i) {
            auto& ctx = mContexts.at(i);
            if (!ctx.Model) {
                throw std::runtime_error(fmt::format("step_chunked: ctx[{}].Model is null", i));
            }
            const int src_row = host_batch_row_for_local_rank(i, ep_size);
            copy_chunk_inputs_for_ctx(*ctx.Model, inputs, targets, position_ids, src_row, B, T, T_full, base_pos);
        }
    };

    // Phase A — KV sweep: forward chunks left-to-right filling the per-layer
    // attention KV caches. Saved-tensor persistence is skipped; the loss op
    // lives in backward, so this pass pays layers only.
    for (int c = 0; c < seq_chunks; ++c) {
        copy_chunk(c);
        const int micro_eff = mTrainMicroStep * seq_chunks + c;
        run_work([c, seq_chunks, micro_eff](sThreadContext& ctx) {
            ctx.Model->set_sequence_chunk(c, seq_chunks);
            Tensor in = ctx.Model->get_input_buffer();
            Tensor pos = ctx.Model->get_position_ids_buffer();
            ctx.Model->forward_no_save(in, pos, *ctx.Communicator, micro_eff);
        });
    }

    // Phase B — reverse re-forward + backward. Each chunk re-forwards against
    // the frozen KV prefix (regenerating its saved tensors), then runs its
    // backward; dK/dV accumulate across chunks in the executor's FP32
    // accumulators, completed exactly because chunks run last-to-first.
    // Grad zero/accumulate, LoRA micro bookkeeping and loss reduction all key
    // off the effective micro index, which counts processed chunks.
    run_work([](sThreadContext& ctx) { ctx.Model->zero_sequence_chunk_dkv(); });
    for (int c = seq_chunks - 1; c >= 0; --c) {
        copy_chunk(c);
        const int micro_eff = mTrainMicroStep * seq_chunks + (seq_chunks - 1 - c);
        run_work([c, seq_chunks, micro_eff, accum_eff](sThreadContext& ctx) {
            ctx.Model->set_sequence_chunk(c, seq_chunks);
            Tensor in = ctx.Model->get_input_buffer();
            Tensor pos = ctx.Model->get_position_ids_buffer();
            Tensor tgt = ctx.Model->get_target_buffer();
            ctx.Model->forward(in, pos, *ctx.Communicator, micro_eff);
            ctx.Model->backward(in, tgt, *ctx.Communicator, accum_eff, micro_eff);
        });
    }
    run_work([](sThreadContext& ctx) { ctx.Model->set_sequence_chunk(-1, 0); });
    ++mTrainMicroStep;
}

void MultiGPUPyTrainer::set_visual_inputs(const std::int32_t* visual_pos_masks,
                                          const float* visual_embeds,
                                          const std::vector<const float*>& deepstack_visual_embeds) {
    if (!mConfig || !mConfig->UseVisualInputs) {
        if (visual_pos_masks || visual_embeds || !deepstack_visual_embeds.empty()) {
            throw std::runtime_error(
                "set_visual_inputs: visual inputs requested but model config has UseVisualInputs=false");
        }
        return;
    }

    const int world = local_world_size();
    const int ep_size = std::max(1, mOptions.EPSize);
    const std::size_t mask_stride = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
    const std::size_t embed_stride = mask_stride * static_cast<std::size_t>(mConfig->HiddenSize);
    for (int i = 0; i < world; ++i) {
        const int src_row = host_batch_row_for_local_rank(i, ep_size);
        const std::int32_t* mask_ptr = visual_pos_masks ? (visual_pos_masks + src_row * mask_stride) : nullptr;
        const float* embed_ptr = visual_embeds ? (visual_embeds + src_row * embed_stride) : nullptr;
        std::vector<const float*> deepstack_ptrs;
        deepstack_ptrs.reserve(deepstack_visual_embeds.size());
        for (const float* base_ptr : deepstack_visual_embeds) {
            deepstack_ptrs.push_back(base_ptr ? (base_ptr + src_row * embed_stride) : nullptr);
        }
        run_work(
            [mask_ptr, embed_ptr, deepstack_ptrs](sThreadContext& ctx) {
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

                copy_from_float(rs.VisualEmbeds_CPU.Data,
                                rs.VisualEmbeds_CPU.DType,
                                embed_ptr,
                                rs.VisualEmbeds_CPU.nelem());

                if (!rs.DeepstackVisualEmbeds_CPU.empty()) {
                    for (std::size_t j = 0; j < rs.DeepstackVisualEmbeds_CPU.size(); ++j) {
                        Tensor& dst = rs.DeepstackVisualEmbeds_CPU[j];
                        const float* src = (j < deepstack_ptrs.size()) ? deepstack_ptrs[j] : nullptr;
                        copy_from_float(dst.Data, dst.DType, src, dst.nelem());
                    }
                }
            },
            i);
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
float MultiGPUPyTrainer::validate(const std::int32_t* inputs,
                                  const std::int32_t* targets,
                                  const std::int32_t* position_ids) {
    const int ep_size = std::max(1, mOptions.EPSize);
    for (int i = 0; i < mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
        const int src_row = host_batch_row_for_local_rank(i, ep_size);

        std::memcpy(ib, inputs + src_row * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + src_row * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            // Python binding provides 2D [B, T] position IDs (one plane per GPU).
            // For mRoPE models the buffer is [3, B, T] — replicate the single plane.
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(src_row) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
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
    // Chunked-sequence training: step arrays and pinned staging carry the
    // full sequence (T_step = T * seq_chunks); the graph runs chunk views.
    const int seq_chunks = std::max(1, mOptions.SequenceChunks);
    const int T_step = T * seq_chunks;
    const std::size_t stride = static_cast<std::size_t>(B) * static_cast<std::size_t>(T_step);
    const int pos_planes = (mContexts.empty() || !mContexts.front().Model)
                               ? 1
                               : ((mContexts.front().Model->get_position_ids_buffer().Rank == 3)
                                      ? static_cast<int>(mContexts.front().Model->get_position_ids_buffer().Sizes[0])
                                      : 1);

    // Chunked-sequence: trailing all-padding chunks are skipped exactly —
    // no valid target lives there and causal attention never looks ahead, so
    // neither loss nor any gradient depends on them. Computed on the FULL
    // host batch (all local ranks' rows) so every rank derives the same
    // count and the collective schedules stay aligned. (Single-node only:
    // multi-node would need a cross-node reduction of the count.)
    std::vector<int> chunk_count_per_micro(static_cast<std::size_t>(micro_steps), seq_chunks);
    if (seq_chunks > 1 && targets) {
        const int rows_per_micro = local_gpus * B;
        for (int j = 0; j < micro_steps; ++j) {
            long last_valid = -1;
            for (int r = 0; r < rows_per_micro; ++r) {
                const std::int32_t* row =
                    targets + (static_cast<std::size_t>(j) * rows_per_micro + r) * static_cast<std::size_t>(T_step);
                for (long t = static_cast<long>(T_step) - 1; t > last_valid; --t) {
                    if (row[t] != -100) {
                        last_valid = t;
                        break;
                    }
                }
                if (last_valid >= static_cast<long>(T_step) - 1) break;
            }
            const int need = static_cast<int>((last_valid + static_cast<long>(T)) / T);
            chunk_count_per_micro[static_cast<std::size_t>(j)] = std::min(seq_chunks, std::max(1, need));
        }
    }

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
        if (gs.captured && (gs.captured_B != B || gs.captured_T != T_step || gs.captured_grad_accum != micro_steps)) {
            gs.reset_capture();
        }

        // Allocate per-micro-step pinned buffers if needed.
        if (gs.inputs.size() != static_cast<size_t>(micro_steps) ||
            gs.targets.size() != static_cast<size_t>(micro_steps) ||
            gs.position_ids.size() != static_cast<size_t>(micro_steps) || gs.captured_B != B ||
            gs.captured_T != T_step) {
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
                gs.inputs.push_back(rs.Allocator->allocate(
                    ETensorDType::INT32, in_name.c_str(), EAllocationType::PINNED, {B, T_step}));
                gs.targets.push_back(rs.Allocator->allocate(
                    ETensorDType::INT32, tgt_name.c_str(), EAllocationType::PINNED, {B, T_step}));
                if (pos_planes > 1) {
                    gs.position_ids.push_back(rs.Allocator->allocate(ETensorDType::INT32,
                                                                     pos_name.c_str(),
                                                                     EAllocationType::PINNED,
                                                                     {pos_planes, B, T_step}));
                } else {
                    gs.position_ids.push_back(rs.Allocator->allocate(
                        ETensorDType::INT32, pos_name.c_str(), EAllocationType::PINNED, {B, T_step}));
                }
            }

            gs.captured_B = B;
            gs.captured_T = T_step;
            gs.captured_grad_accum = micro_steps;
        }

        // Allocate device-side optimizer parameter buffers if needed.
        // Use the maximum size to support both AdamW and NorMuon
        constexpr int max_opt_params =
            std::max(optimizers::ADAMW_GRAPH_PARAM_COUNT, optimizers::NORMUON_GRAPH_PARAM_COUNT);
        if (!gs.opt_params.Data) {
            auto name = fmt::format("graph_opt_params_rank{}", ctx.Communicator->local_rank());
            gs.opt_params =
                rs.Allocator->allocate(ETensorDType::FP32, name.c_str(), EAllocationType::ON_DEVICE, {max_opt_params});
        }
        if (!gs.opt_step.Data) {
            auto name = fmt::format("graph_opt_step_rank{}", ctx.Communicator->local_rank());
            gs.opt_step = rs.Allocator->allocate(ETensorDType::INT32, name.c_str(), EAllocationType::ON_DEVICE, {1});
        }

        // Stage inputs/targets/position_ids for all micro-steps.
        const int rank = ctx.Communicator->local_rank();
        const int ep_size = std::max(1, mOptions.EPSize);
        const int src_row = host_batch_row_for_local_rank(rank, ep_size);
        for (int j = 0; j < micro_steps; ++j) {
            const std::size_t offset = (static_cast<std::size_t>(j) * static_cast<std::size_t>(local_gpus) +
                                        static_cast<std::size_t>(src_row)) *
                                       stride;
            const std::size_t pos_row_offset = (static_cast<std::size_t>(j) * static_cast<std::size_t>(local_gpus) +
                                                static_cast<std::size_t>(src_row)) *
                                               stride;
            std::memcpy(gs.inputs[j].Data, inputs + offset, stride * sizeof(std::int32_t));
            std::memcpy(gs.targets[j].Data, targets + offset, stride * sizeof(std::int32_t));
            if (position_ids) {
                auto* dst = reinterpret_cast<std::int32_t*>(gs.position_ids[j].Data);
                const auto* src = position_ids + static_cast<std::ptrdiff_t>(pos_row_offset);
                if (pos_planes > 1) {
                    // Python passes 2D [rows, T] position IDs.
                    // Expand to [planes, B, T] by replicating the same packed IDs
                    // across planes (required to preserve doc-boundary resets).
                    for (int p = 0; p < pos_planes; ++p) {
                        std::memcpy(dst + static_cast<std::size_t>(p) * stride, src, stride * sizeof(std::int32_t));
                    }
                } else {
                    std::memcpy(dst, src, stride * sizeof(std::int32_t));
                }
            } else {
                fill_sequential_position_ids(reinterpret_cast<std::int32_t*>(gs.position_ids[j].Data),
                                             pos_planes,
                                             B,
                                             T_step);
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
                config.adamw_epsilon};
            CUDA_CHECK(cudaMemcpyAsync(gs.opt_params.Data,
                                       opt_params_host,
                                       sizeof(opt_params_host),
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        } else {
            float opt_params_host[optimizers::ADAMW_GRAPH_PARAM_COUNT] = {config.learning_rate,
                                                                          config.adamw_beta1,
                                                                          config.adamw_beta2,
                                                                          config.adamw_epsilon,
                                                                          config.weight_decay};
            CUDA_CHECK(cudaMemcpyAsync(gs.opt_params.Data,
                                       opt_params_host,
                                       sizeof(opt_params_host),
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        CUDA_CHECK(cudaMemcpyAsync(gs.opt_step.Data,
                                   &opt_step_host,
                                   sizeof(opt_step_host),
                                   cudaMemcpyHostToDevice,
                                   rs.MainStream));

        // Detect packed sequences with document boundaries in any micro-step.
        // When present, the captured H2D memcpys for cu_seqlens read from
        // per-ms slices of mCuSeqlensCpu and the captured kernel grid
        // bakes in MAX_NUM_DOCS so any topology (within cap) replays
        // correctly. Cap-and-pad mode is enabled before
        // cudaStreamBeginCapture below; per-launch cu_seqlens recomputes
        // happen right before cudaGraphLaunch.
        // Mirror dsl::compute_doc_masking: only WITHIN-ROW transitions
        // count as doc boundaries. Multi-row batches (B>1) without packed
        // sequences would otherwise look like they have "boundaries" at
        // every row transition (pos[T] -> 0), which is wrong.
        bool has_doc_boundaries = false;
        int current_max_num_docs = 0;
        std::vector<std::vector<std::int32_t>> per_ms_cu_seqlens(micro_steps);
        std::vector<int> per_ms_total_q(micro_steps, 0);
        if (mOptions.DocMasking) {
            for (int j = 0; j < micro_steps; ++j) {
                const auto* pos = reinterpret_cast<const std::int32_t*>(gs.position_ids[j].Data);
                auto& cu = per_ms_cu_seqlens[j];
                cu.clear();
                cu.push_back(0);
                for (int b = 0; b < B; ++b) {
                    const int row_base = b * T_step;
                    int doc_start = row_base;
                    for (int t = 1; t < T_step; ++t) {
                        const int idx = row_base + t;
                        if (pos[idx] - pos[idx - 1] != 1) {
                            const int doc_len = idx - doc_start;
                            if (doc_len > 0) cu.push_back(cu.back() + doc_len);
                            doc_start = idx;
                            has_doc_boundaries = true;
                        }
                    }
                    const int last_len = (b + 1) * T_step - doc_start;
                    if (last_len > 0) cu.push_back(cu.back() + last_len);
                }
                per_ms_total_q[j] = static_cast<int>(B * T_step);
                current_max_num_docs = std::max(current_max_num_docs, std::max(0, static_cast<int>(cu.size()) - 1));
            }
        }

        // Outer full-step capture handles sample_packing via the cap-and-pad
        // cu_seqlens machinery (commits 05423b6, e4da4f8 + the saved-tensor
        // policy fix): per-ms cu_seqlens slices in mCuSeqlensCpu, captured
        // H2D memcpys with stable host source pointers, and a captured
        // kernel grid baked at worst-case dims (params.b = MAX_NUM_DOCS,
        // seqlen = total_q). SUROGATE_DEBUG_FORCE_OUTER_CAPTURE is now
        // redundant for the doc-masking gate but kept as an opt-in escape
        // hatch for diagnostic runs. The eager fallback below remains the
        // path used when CUDA graphs are entirely disabled.
        const bool force_outer_capture = std::getenv("SUROGATE_DEBUG_FORCE_OUTER_CAPTURE") != nullptr;
        (void)force_outer_capture;

        if (!mOptions.UseCudaGraphs) {
            const bool do_timing = mOptions.TriggerTimingEvents;
            if (do_timing && rs.TimingForwardStart.empty()) {
                rs.setup_timing_events(micro_steps);
            }
            // Under force-full-capture + doc_masking, GraphExecutor captures
            // mForwardGraph at ms=0 and replays it for ms>0 within the same
            // step. The captured H2D memcpys bake in the source pinned-buffer
            // address (gs.inputs[0], gs.targets[0], gs.position_ids[0]).
            // If we pass gs.inputs[j] for j>0, the replay ignores j's buffer
            // and re-reads [0]'s contents — so ms>0 forwards see ms=0's
            // tokens. Route every micro-step through the [0] buffer and
            // stage fresh data there per micro-step so captured replays
            // read the correct current tokens.
            const bool force_full_capture = std::getenv("SUROGATE_FORCE_FULL_GRAPH_CAPTURE") != nullptr;
            const bool stage_through_zero = force_full_capture && has_doc_boundaries && micro_steps > 1;
            const std::size_t pos_bytes_per_plane = stride * sizeof(std::int32_t);
            const std::size_t pos_total_bytes =
                (pos_planes > 1 ? static_cast<std::size_t>(pos_planes) : 1) * pos_bytes_per_plane;
            if (seq_chunks > 1) {
                // ----------------------------------------------------------
                // Chunked-sequence schedule (KV-checkpointed chunks).
                // Phase A walks chunks left-to-right filling the per-layer
                // attention KV caches; phase B walks right-to-left,
                // re-forwarding each chunk against the frozen KV prefix and
                // running its backward with exact dK/dV accumulation.
                // Effective micro indices count processed chunks, so grad
                // zero/accumulate, LoRA bookkeeping and loss reduction key
                // off them exactly as plain grad accumulation would.
                // ----------------------------------------------------------
                if (B != 1) {
                    throw std::runtime_error("chunked-sequence training requires per-device batch size 1");
                }
                int accum_eff = 0;
                for (int j = 0; j < micro_steps; ++j) accum_eff += chunk_count_per_micro[static_cast<std::size_t>(j)];
                int micro_base = 0;
                const std::size_t chunk_bytes_off = static_cast<std::size_t>(T) * sizeof(std::int32_t);
                auto chunk_view = [&](Tensor& full, int c) {
                    Tensor v = full;
                    v.Sizes[v.Rank - 1] = T;
                    v.Data = static_cast<std::byte*>(full.Data) + static_cast<std::size_t>(c) * chunk_bytes_off;
                    return v;
                };
                // Position ids may carry mRoPE planes ([planes, B, T_step]);
                // a chunk slice is then non-contiguous — stage it through a
                // small pinned scratch (planes x B x T).
                auto pos_chunk_view = [&](Tensor& full, int c) -> Tensor {
                    if (pos_planes <= 1) return chunk_view(full, c);
                    if (!gs.chunk_pos_scratch.Data) {
                        auto name = fmt::format("chunk_pos_scratch_rank{}", ctx.Communicator->local_rank());
                        gs.chunk_pos_scratch = rs.Allocator->allocate(
                            ETensorDType::INT32, name.c_str(), EAllocationType::PINNED, {pos_planes, B, T});
                    }
                    auto* dst = gs.chunk_pos_scratch.get<std::int32_t>();
                    const auto* src = reinterpret_cast<const std::int32_t*>(full.Data);
                    for (int pl = 0; pl < pos_planes; ++pl) {
                        std::memcpy(dst + static_cast<std::size_t>(pl) * T,
                                    src + static_cast<std::size_t>(pl) * T_step + static_cast<std::size_t>(c) * T,
                                    static_cast<std::size_t>(T) * sizeof(std::int32_t));
                    }
                    return gs.chunk_pos_scratch;
                };
                const bool chunk_trace = std::getenv("SUROGATE_CHUNK_TRACE") != nullptr;
                // Per-chunk document geometry from position ids: packed rows
                // restart positions at each document, so a chunk's segments
                // and its contiguous KV window fall out of one linear scan.
                // Unpacked rows (absolute positions) reduce to one segment
                // with win_start 0 — the same code path covers both.
                auto build_chunk_pack = [&](const std::int32_t* pos_row, int c) {
                    IModel::ChunkPackMeta m;
                    const int t0 = c * T;
                    const int t1 = t0 + T;
                    // Document starts are ROW-RELATIVE: position resets mark
                    // boundaries, and the row start is always one — packers
                    // split documents across rows, so a row can begin
                    // mid-document with continued position values (the
                    // partial head attends within the row only, matching the
                    // dense doc-masking semantics).
                    int win_start = 0;  // last doc start <= t0
                    for (int t = 1; t <= t0; ++t) {
                        if (pos_row[t] != pos_row[t - 1] + 1) win_start = t;
                    }
                    m.win_start = win_start;
                    m.cu_q = {0};
                    m.cu_k = {0};
                    int seg_start = t0;
                    int doc_start = win_start;
                    auto close_seg = [&](int seg_end) {
                        const int q_len = seg_end - seg_start;
                        const int k_len = seg_end - doc_start;
                        m.cu_q.push_back(m.cu_q.back() + q_len);
                        m.cu_k.push_back(m.cu_k.back() + k_len);
                        m.max_q = std::max(m.max_q, q_len);
                        m.max_k = std::max(m.max_k, k_len);
                    };
                    for (int t = t0 + 1; t < t1; ++t) {
                        if (pos_row[t] != pos_row[t - 1] + 1) {
                            close_seg(t);
                            seg_start = t;
                            doc_start = t;
                        }
                    }
                    close_seg(t1);
                    m.num_segs = static_cast<int>(m.cu_q.size()) - 1;
                    m.kv_len = t1 - m.win_start;
                    if (std::getenv("SUROGATE_CHUNK_TRACE") && ctx.Communicator->rank() == 0) {
                        fprintf(stderr, "[pack] c=%d win=%d kv=%d segs=%d maxq=%d maxk=%d\n", c, m.win_start,
                                m.kv_len, m.num_segs, m.max_q, m.max_k);
                        fflush(stderr);
                    }
                    return m;
                };
                for (int j = 0; j < micro_steps; ++j) {
                    const int eff_chunks = chunk_count_per_micro[static_cast<std::size_t>(j)];
                    for (int c = 0; c < eff_chunks; ++c) {
                        Tensor in_v = chunk_view(gs.inputs[j], c);
                        Tensor pos_v = pos_chunk_view(gs.position_ids[j], c);
                        Tensor tgt_v = chunk_view(gs.targets[j], c);
                        const auto pack =
                            build_chunk_pack(reinterpret_cast<const std::int32_t*>(gs.position_ids[j].Data), c);
                        ctx.Model->set_sequence_chunk(c, seq_chunks, &pack);
                        // The loss op lives in the forward graph — stage the
                        // chunk's real targets so phase A's loss terms are
                        // sane (they are wiped by the micro-0 zeroing before
                        // phase B accumulates the reported values).
                        rs.Targets_CPU = tgt_v;
                        if (chunk_trace && ctx.Communicator->rank() == 0) {
                            fprintf(stderr, "[chunk] phaseA j=%d c=%d\n", j, c);
                            fflush(stderr);
                        }
                        ctx.Model->forward(in_v, pos_v, *ctx.Communicator, micro_base + c);
                        ::surogate::tick_watchdog_heartbeat();
                    }
                    ctx.Model->zero_sequence_chunk_dkv();
                    for (int c = eff_chunks - 1; c >= 0; --c) {
                        Tensor in_v = chunk_view(gs.inputs[j], c);
                        Tensor pos_v = pos_chunk_view(gs.position_ids[j], c);
                        Tensor tgt_v = chunk_view(gs.targets[j], c);
                        const int micro_eff = micro_base + (eff_chunks - 1 - c);
                        auto pack =
                            build_chunk_pack(reinterpret_cast<const std::int32_t*>(gs.position_ids[j].Data), c);
                        pack.reuse_ep = true;  // phase A cached this chunk's plan+splits
                        ctx.Model->set_sequence_chunk(c, seq_chunks, &pack);
                        rs.Targets_CPU = tgt_v;
                        if (chunk_trace && ctx.Communicator->rank() == 0) {
                            fprintf(stderr, "[chunk] phaseB j=%d c=%d micro_eff=%d\n", j, c, micro_eff);
                            fflush(stderr);
                        }
                        ctx.Model->forward(in_v, pos_v, *ctx.Communicator, micro_eff);
                        ::surogate::tick_watchdog_heartbeat();
                        ctx.Model->backward(in_v, tgt_v, *ctx.Communicator, accum_eff, micro_eff);
                        ::surogate::tick_watchdog_heartbeat();
                    }
                    ctx.Model->set_sequence_chunk(-1, 0);
                    micro_base += eff_chunks;
                }
            } else
            for (int j = 0; j < micro_steps; ++j) {
                if (stage_through_zero && j > 0) {
                    std::memcpy(gs.inputs[0].Data, gs.inputs[j].Data, stride * sizeof(std::int32_t));
                    std::memcpy(gs.targets[0].Data, gs.targets[j].Data, stride * sizeof(std::int32_t));
                    std::memcpy(gs.position_ids[0].Data, gs.position_ids[j].Data, pos_total_bytes);
                }
                Tensor& in_t = stage_through_zero ? gs.inputs[0] : gs.inputs[j];
                Tensor& tgt_t = stage_through_zero ? gs.targets[0] : gs.targets[j];
                Tensor& pos_t = stage_through_zero ? gs.position_ids[0] : gs.position_ids[j];
                rs.Targets_CPU = tgt_t;
                if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingForwardStart[j], rs.MainStream));
                ctx.Model->forward(in_t, pos_t, *ctx.Communicator, j);
                if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingForwardEnd[j], rs.MainStream));
                if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingBackwardStart[j], rs.MainStream));
                ctx.Model->backward(in_t, tgt_t, *ctx.Communicator, micro_steps, j);
                if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingBackwardEnd[j], rs.MainStream));
            }
            if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingOptimizerStart, rs.MainStream));
            ctx.Model->update_with_config(*ctx.Communicator, config, opt_step_host);
            if (do_timing) CUDA_CHECK(cudaEventRecord(rs.TimingOptimizerEnd, rs.MainStream));
            CUDA_CHECK(cudaDeviceSynchronize());
            return;
        }

        if (seq_chunks > 1) {
            throw std::runtime_error(
                "chunked-sequence training requires use_cuda_graphs: false (eager full-step path)");
        }
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("train_step_graphed: only supported for DSL models");
        }
        if (config.type != optimizers::OptimizerType::ADAMW && config.type != optimizers::OptimizerType::ADAMW_8BIT &&
            config.type != optimizers::OptimizerType::NORMUON) {
            throw std::runtime_error("train_step_graphed: only supports AdamW, AdamW 8-bit or NorMuon optimizer");
        }

        // CUDA graph capture path (both AdamW and NorMuon support graph capture)
        enum class FullStepGraphMode {
            Full,
            ForwardBackward,
            ForwardOnly
        };
        const char* graph_mode_env = std::getenv("SUROGATE_FULLSTEP_GRAPH_MODE");
        FullStepGraphMode graph_mode = FullStepGraphMode::Full;
        if (graph_mode_env) {
            if (std::strcmp(graph_mode_env, "fwd_bwd") == 0) {
                graph_mode = FullStepGraphMode::ForwardBackward;
            } else if (std::strcmp(graph_mode_env, "fwd") == 0) {
                graph_mode = FullStepGraphMode::ForwardOnly;
            }
        }
        const bool do_graph_backward = (graph_mode != FullStepGraphMode::ForwardOnly);
        const bool do_graph_update = (graph_mode == FullStepGraphMode::Full);
        if (graph_mode != FullStepGraphMode::Full) {
            static bool graph_mode_warned = false;
            if (!graph_mode_warned && ctx.Communicator && ctx.Communicator->rank() == 0) {
                fprintf(stderr,
                        "[CUDA graphs] SUROGATE_FULLSTEP_GRAPH_MODE=%s (debug mode)\n",
                        graph_mode == FullStepGraphMode::ForwardBackward ? "fwd_bwd" : "fwd");
                graph_mode_warned = true;
            }
        }

        const bool warmup_full_graph = !gs.captured && !env_enabled("SUROGATE_DSL_GRAPH_SKIP_WARMUP");
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
                if (do_graph_backward && !warmup_skip_bwd) {
                    dsl_model->backward(gs.inputs[j], gs.targets[j], *ctx.Communicator, micro_steps, j);
                }
            }
            dsl_model->zero_grads(rs.MainStream);
            dsl_model->set_rng_state(rng_state);
            CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));

            // Full-step graph capture bakes stack addresses into the graph, so
            // Python cannot safely shrink the DSL stack after step 0. Do the
            // one-shot shrink here: warmup has measured the real high-water
            // mark, and capture has not started yet.
            if (!env_enabled("SUROGATE_DISABLE_STACK_SHRINK")) {
                const long safety_mb = env_long_mb("SUROGATE_FULLSTEP_STACK_SHRINK_SAFETY_MB", 512);
                const long min_savings_mb = env_long_mb("SUROGATE_FULLSTEP_STACK_SHRINK_MIN_SAVINGS_MB", 128);
                auto* dsl_rs = dynamic_cast<dsl::DslRunState*>(&rs);
                if (dsl_rs) {
                    const long old_size = static_cast<long>(dsl_rs->Stack.capacity());
                    const long new_size =
                        dsl_rs->shrink_stack_to_high_water_mark(safety_mb * 1024 * 1024, min_savings_mb * 1024 * 1024);
                    if (new_size > 0) {
                        dsl_model->invalidate_cuda_graphs();
                        if (ctx.Communicator && ctx.Communicator->rank() == 0) {
                            fprintf(stderr,
                                    "[CUDA graphs] DSL stack shrunk before full-step capture: %ld MiB -> %ld MiB\n",
                                    old_size / (1024 * 1024),
                                    new_size / (1024 * 1024));
                        }
                    }
                }
            }
        }

        // After warmup compiles the graph, check for capture-unsafe ops
        // (e.g. JIT Triton GDR kernels). Full-step graph capture cannot wrap
        // these ops, so fall back to the per-step eager path where
        // GraphExecutor uses split-attention per-segment CUDA graphs internally.
        if (dsl_model->has_capture_unsafe_ops()) {
            if (prev_internal_graphs) {
                dsl_model->set_internal_graphs_enabled(true);
            }
            for (int j = 0; j < micro_steps; ++j) {
                rs.Targets_CPU = gs.targets[j];
                dsl_model->forward(gs.inputs[j], gs.position_ids[j], *ctx.Communicator, j);
                dsl_model->backward(gs.inputs[j], gs.targets[j], *ctx.Communicator, micro_steps, j);
            }
            dsl_model->update_with_config(*ctx.Communicator, config, opt_step_host);
            CUDA_CHECK(cudaDeviceSynchronize());
            return;
        }

        int max_num_docs = 64;
        bool max_num_docs_from_env = false;
        if (const char* env = std::getenv("SUROGATE_OUTER_CAPTURE_MAX_DOCS")) {
            char* end = nullptr;
            const long val = std::strtol(env, &end, 10);
            if (end != env && val > 0) {
                max_num_docs = static_cast<int>(val);
                max_num_docs_from_env = true;
            }
        }
        if (has_doc_boundaries && current_max_num_docs > max_num_docs) {
            if (max_num_docs_from_env) {
                throw std::runtime_error(
                    fmt::format("outer capture doc cap exceeded: current batch has {} docs, cap is {}. "
                                "Increase SUROGATE_OUTER_CAPTURE_MAX_DOCS.",
                                current_max_num_docs,
                                max_num_docs));
            }
            const int old_max_num_docs = max_num_docs;
            max_num_docs = round_up_pow2_int(current_max_num_docs);
            static bool warned_auto_doc_cap = false;
            if (!warned_auto_doc_cap && ctx.Communicator && ctx.Communicator->rank() == 0) {
                std::fprintf(stderr,
                             "[CUDA graphs] outer capture doc cap auto-raised: %d -> %d docs\n",
                             old_max_num_docs,
                             max_num_docs);
                warned_auto_doc_cap = true;
            }
        }
        if (gs.captured) {
            if (has_doc_boundaries && (gs.captured_doc_cap <= 0 || current_max_num_docs > gs.captured_doc_cap)) {
                gs.reset_capture();
            } else if (!has_doc_boundaries && gs.captured_doc_cap > 0) {
                has_doc_boundaries = true;
                current_max_num_docs = 1;
                const int total_q = static_cast<int>(B * T);
                for (auto& cu : per_ms_cu_seqlens) {
                    cu.assign({0, total_q});
                }
            }
        }

        if (!gs.captured) {
            dsl_model->prepare_optimizer_state_for_graph(*ctx.Communicator, config);
            // The bwd_cross_layer arena is sized lazily during eager
            // backward (warmup just ran one); grow it now to fit the
            // observed high-water mark before capture begins so the
            // captured backward never falls back to cudaMalloc.
            dsl_model->prepare_bwd_cross_layer_for_capture();
            // Warmup forward syncs LoRA master weights into compute/work
            // weights and marks each block current. If we capture immediately
            // after that, LoRA get_block() skips the master->work sync and the
            // graph never records those copy/convert nodes. Replays would then
            // keep using the warmup adapter weights while the optimizer updates
            // only the master weights.
            if (dsl_model->lora_enabled()) {
                dsl_model->lora_weights().advance_sync_generation();
            }
            // Cap-and-pad: when packed sequences are present, lock the
            // captured kernel grid + scalar args (params.b, seqlen) to
            // worst case so a single captured graph replays correctly
            // for any per-step cu_seqlens topology within
            // SUROGATE_OUTER_CAPTURE_MAX_DOCS, or the observed topology when
            // the env cap is unset. cu_seqlens contents are updated per-launch
            // from a stable pinned host buffer below, before cudaGraphLaunch.
            if (has_doc_boundaries) {
                const int capture_doc_cap = std::min(max_num_docs, round_up_pow2_int(current_max_num_docs));
                dsl_model->enable_doc_masking_pad_to_max(capture_doc_cap, micro_steps);
                gs.captured_doc_cap = capture_doc_cap;
                // Stage warmup-step cu_seqlens for each ms so the captured
                // forward sees consistent shapes during capture.
                const int total_q = static_cast<int>(B * T);
                for (int j = 0; j < micro_steps; ++j) {
                    auto& cu = per_ms_cu_seqlens[j];
                    if (cu.size() <= 1) {
                        // No boundaries this ms — synthesize a single
                        // doc covering the full B*T span.
                        cu.assign({0, total_q});
                    }
                    dsl_model->stage_cu_seqlens_for_micro_step(j, cu.data(), static_cast<int>(cu.size()), total_q);
                }
            }
            // Ensure the main stream is idle before beginning capture (no external dependencies).
            CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
            auto stack_cp = rs.Stack.checkpoint();
            gs.stack_top = stack_cp.top;
            gs.stack_alloc_count = stack_cp.alloc_count;
            gs.has_stack_checkpoint = true;
            cudaGraph_t graph = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(rs.MainStream, cudaStreamCaptureModeThreadLocal));
            for (int j = 0; j < micro_steps; ++j) {
                rs.Targets_CPU = gs.targets[j];
                dsl_model->forward(gs.inputs[j], gs.position_ids[j], *ctx.Communicator, j);
                if (do_graph_backward) {
                    dsl_model->backward(gs.inputs[j], gs.targets[j], *ctx.Communicator, micro_steps, j);
                }
            }
            if (do_graph_update) {
                dsl_model->update_with_graph_params(*ctx.Communicator,
                                                    config,
                                                    gs.opt_params.template get<float>(),
                                                    gs.opt_step.template get<int>());
            }
            CUDA_CHECK(cudaStreamEndCapture(rs.MainStream, &graph));
            // Diagnostic: walk captured graph and report memcpy nodes whose
            // host source/dest pointers look suspicious (stack-resident, or
            // outside the pinned-buffer / device-arena ranges we know are
            // stable). Helps localize V#3 — a captured op that references
            // a transient host allocation valid only at capture time.
            if (std::getenv("SUROGATE_DEBUG_DUMP_CAPTURED_GRAPH") != nullptr) {
                size_t num_nodes = 0;
                CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
                std::vector<cudaGraphNode_t> nodes(num_nodes);
                CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &num_nodes));
                fprintf(stderr, "[graph-dump] %zu nodes\n", num_nodes);
                std::map<int, int> kind_counts;
                std::vector<std::tuple<int, void*, void*, size_t, int>> memcpys;  // idx, src, dst, bytes, kind
                for (size_t i = 0; i < num_nodes; ++i) {
                    cudaGraphNodeType t = cudaGraphNodeTypeEmpty;
                    CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &t));
                    kind_counts[static_cast<int>(t)]++;
                    if (t == cudaGraphNodeTypeMemcpy) {
                        cudaMemcpy3DParms params{};
                        cudaError_t err = cudaGraphMemcpyNodeGetParams(nodes[i], &params);
                        if (err == cudaSuccess) {
                            void* src = params.srcPtr.ptr;
                            void* dst = params.dstPtr.ptr;
                            size_t bytes = params.extent.width * std::max<size_t>(params.extent.height, 1) *
                                           std::max<size_t>(params.extent.depth, 1);
                            memcpys.emplace_back(static_cast<int>(i), src, dst, bytes, static_cast<int>(params.kind));
                        }
                    }
                }
                fprintf(stderr, "[graph-dump] node kinds:");
                for (const auto& [k, c] : kind_counts) {
                    fprintf(stderr, " %d=%d", k, c);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "[graph-dump] %zu memcpy nodes:\n", memcpys.size());
                for (const auto& [idx, src, dst, bytes, kind] : memcpys) {
                    const char* kname = "?";
                    if (kind == cudaMemcpyHostToDevice)
                        kname = "H2D";
                    else if (kind == cudaMemcpyDeviceToHost)
                        kname = "D2H";
                    else if (kind == cudaMemcpyDeviceToDevice)
                        kname = "D2D";
                    else if (kind == cudaMemcpyHostToHost)
                        kname = "H2H";
                    fprintf(stderr, "  node[%d] %s src=%p dst=%p bytes=%zu\n", idx, kname, src, dst, bytes);
                }
            }
            CUDA_CHECK(cudaGraphInstantiate(&gs.graph_exec, graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            gs.captured = true;
        }

        if (prev_internal_graphs) {
            dsl_model->set_internal_graphs_enabled(true);
        }

        if (gs.has_stack_checkpoint) {
            rs.Stack.restore(DeviceMemoryStack::Checkpoint{gs.stack_top, gs.stack_alloc_count});
        }
        // Per-launch cu_seqlens update: write each ms's cu_seqlens into
        // its slice of the pinned host buffer. The captured H2D memcpys
        // pick up the new contents at the next launch (host pointer is
        // stable, contents change). Without this, the captured graph
        // would replay step 0's cu_seqlens for every subsequent step.
        if (has_doc_boundaries) {
            const int total_q = static_cast<int>(B * T);
            for (int j = 0; j < micro_steps; ++j) {
                auto& cu = per_ms_cu_seqlens[j];
                if (cu.size() <= 1) {
                    cu.assign({0, total_q});
                }
                dsl_model->stage_cu_seqlens_for_micro_step(j, cu.data(), static_cast<int>(cu.size()), total_q);
            }
        }
        CUDA_CHECK(cudaGraphLaunch(gs.graph_exec, rs.MainStream));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Refresh loss/norm on host after full-step graph launch.
        CUDA_CHECK(cudaMemcpy(rs.LossHost, rs.Losses.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost));
        auto& dsl_rs = dynamic_cast<dsl::DslRunState&>(dsl_model->get_run_state());
        if (dsl_model->lora_enabled()) {
            auto& lora_rs = dsl_model->lora_run_state();
            CUDA_CHECK(cudaMemcpy(dsl_rs.NormHost,
                                  lora_rs.norm_buffer.template get<float>(),
                                  sizeof(float),
                                  cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpy(dsl_rs.NormHost,
                                  dsl_rs.scratch().norm_buffer.template get<float>(),
                                  sizeof(float),
                                  cudaMemcpyDeviceToHost));
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
    fprintf(stderr,
            "[Time Breakdown] step=%d  fwd: %.1fms  bwd: %.1fms  opt: %.1fms  total: %.1fms",
            step,
            total_fwd_ms,
            total_bwd_ms,
            opt_ms,
            total_ms);

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
    run_work([&](sThreadContext& ctx) { infos[ctx.Communicator->rank()] = ctx.GPUUtil->update(); });
    return infos;
}

/**
 * @brief Get MoE training statistics from the last forward pass.
 *
 * Returns accumulated MoE metrics from rank 0's run state. For non-MoE models,
 * returns zeros with valid=false.
 *
 * @return Tuple of MoE load/router diagnostics plus valid flag.
 */
std::tuple<float, float, float, float, float, float, float, float, float, float, bool>
MultiGPUPyTrainer::get_moe_stats() {
    auto& ctx = mContexts.at(0);
    auto& rs = ctx.Model->get_run_state();
    auto stats = rs.get_moe_stats();
    return {stats.aux_loss,
            stats.z_loss,
            stats.expert_utilization,
            stats.load_imbalance,
            stats.active_experts,
            stats.max_expert_fraction,
            stats.min_active_expert_fraction,
            stats.load_cv,
            stats.router_entropy,
            stats.router_confidence,
            stats.valid};
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
auto MultiGPUPyTrainer::fetch_work(sThreadContext& ctx) -> std::function<void(sThreadContext& ctx)> {
    std::lock_guard<std::mutex> lock(mGlobalMutex);
    if (!ctx.Work) {
        return {};
    } else {
        auto work = std::move(ctx.Work);
        return work;
    }
}

void MultiGPUPyTrainer::init_async_slots(std::size_t n) {
    mCtxPending = std::make_unique<std::atomic<int>[]>(n);
    mCtxDone = std::make_unique<std::atomic<int>[]>(n);
    for (std::size_t i = 0; i < n; ++i) {
        mCtxPending[i].store(0, std::memory_order_relaxed);
        mCtxDone[i].store(0, std::memory_order_relaxed);
    }
}

// Launch `work` on GPU `gpu` without blocking the caller; the worker thread for that
// GPU picks it up via fetch_work and bumps mCtxDone when finished. Blocks only if that
// GPU still has an outstanding async item (the natural is_idle backpressure).
void MultiGPUPyTrainer::dispatch_async(std::function<void(sThreadContext& ctx)> work, int gpu) {
    if (mHasCrashed.load()) {
        throw std::runtime_error(
            "MultiGPUPyTrainer: a worker crashed earlier; the trainer is defunct (restart the process)");
    }
    wait_gpu(gpu);  // ensure the previous async item on this GPU has completed
    {
        std::lock_guard<std::mutex> lock(mGlobalMutex);
        mContexts.at(static_cast<std::size_t>(gpu)).Work = std::move(work);
    }
    mCtxPending[static_cast<std::size_t>(gpu)].fetch_add(1, std::memory_order_release);
}

// Block until GPU `gpu`'s outstanding async work has finished (Done caught up to
// Pending). Propagates worker exceptions like run_work.
void MultiGPUPyTrainer::wait_gpu(int gpu) {
    const auto g = static_cast<std::size_t>(gpu);
    while (mCtxDone[g].load(std::memory_order_acquire) < mCtxPending[g].load(std::memory_order_acquire)) {
        if (mThreads->has_exception()) {
            stop();
            // Surviving ranks may be blocked in NCCL collectives waiting for the
            // crashed rank; abort the comms (detached — ncclCommAbort can wedge on
            // driver locks) so they can exit, then surface the root-cause error.
            mThreads->abort_async();
            if (mThreads->wait_exit_for(15000)) {
                mThreads->join();  // will throw, ending the loop
            } else {
                fprintf(stderr,
                        "MultiGPUPyTrainer: workers still wedged after NCCL abort; "
                        "surfacing stored error without join\n");
                fflush(stderr);
                mThreads->rethrow_exception();  // will throw, ending the loop
            }
        }
        std::this_thread::yield();
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
void MultiGPUPyTrainer::run_work(std::function<void(sThreadContext& ctx)> work, int idx) {
    if (mHasCrashed.load()) {
        // A worker died (its exception was already rethrown to the caller once);
        // the surviving workers may be wedged in NCCL and will never process new
        // work — fail fast instead of spinning on mWorkDone forever.
        throw std::runtime_error(
            "MultiGPUPyTrainer: a worker crashed earlier; the trainer is defunct (restart the process)");
    }
    static int work_id = 0;
    int current_work_id = work_id++;
    {
        std::lock_guard<std::mutex> lock(mGlobalMutex);

        if (idx >= 0) {
            mWorkDone = mContexts.size() - 1;
            mContexts.at(idx).Work = work;
            // Keep the async per-GPU counters consistent: the worker bumps Done for
            // every work item (sync or async), so the sync path must bump Pending too.
            if (mCtxPending) mCtxPending[static_cast<std::size_t>(idx)].fetch_add(1, std::memory_order_release);
        } else {
            mWorkDone = 0;
            for (std::size_t i = 0; i < mContexts.size(); ++i) {
                mContexts[i].Work = work;
                if (mCtxPending) mCtxPending[i].fetch_add(1, std::memory_order_release);
            }
        }
    }

    auto last_progress = std::chrono::steady_clock::now();
    std::size_t last_done = mWorkDone.load() + ::surogate::watchdog_heartbeat();
    while (mWorkDone.load() < mContexts.size()) {
        if (mThreads->has_exception()) {
            stop();
            // Surviving ranks may be blocked in NCCL collectives waiting for the
            // crashed rank; abort the comms (detached — ncclCommAbort can wedge on
            // driver locks) so they can exit, then surface the root-cause error.
            mThreads->abort_async();
            if (mThreads->wait_exit_for(15000)) {
                mThreads->join();  // will throw, ending the loop
            } else {
                fprintf(stderr,
                        "MultiGPUPyTrainer: workers still wedged after NCCL abort; "
                        "surfacing stored error without join\n");
                fflush(stderr);
                mThreads->rethrow_exception();  // will throw, ending the loop
            }
        }
        // Lost-wakeup watchdog: a worker occasionally misses a driver/futex wakeup
        // under heavy multi-threaded CUDA load and waits forever on an already-met
        // condition. A no-op signal forces EINTR + recheck. 45s is far above any
        // legitimate op; poking a healthy worker is harmless.
        const std::size_t done_now = mWorkDone.load() + ::surogate::watchdog_heartbeat();
        if (done_now != last_done) {
            last_done = done_now;
            last_progress = std::chrono::steady_clock::now();
        } else if (std::chrono::steady_clock::now() - last_progress > std::chrono::seconds(45)) {
            fprintf(stderr, "MultiGPUPyTrainer: no worker progress for 45s; poking workers (lost-wakeup recovery)\n");
            fflush(stderr);
            mThreads->poke_workers();
            last_progress = std::chrono::steady_clock::now();
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
                if (name == "q_proj")
                    mod_lora.targets.insert(modules::LoRATarget::Q_PROJ);
                else if (name == "k_proj")
                    mod_lora.targets.insert(modules::LoRATarget::K_PROJ);
                else if (name == "v_proj")
                    mod_lora.targets.insert(modules::LoRATarget::V_PROJ);
                else if (name == "o_proj")
                    mod_lora.targets.insert(modules::LoRATarget::O_PROJ);
                else if (name == "gate_proj")
                    mod_lora.targets.insert(modules::LoRATarget::GATE_PROJ);
                else if (name == "gate_up_proj")
                    mod_lora.targets.insert(modules::LoRATarget::GATE_UP_PROJ);
                else if (name == "up_proj")
                    mod_lora.targets.insert(modules::LoRATarget::UP_PROJ);
                else if (name == "down_proj")
                    mod_lora.targets.insert(modules::LoRATarget::DOWN_PROJ);
            }
        }

        // Build QLoRA config if provided
        modules::QLoRAConfig qlora_config;
        if (mQLoRAConfig.has_value()) {
            qlora_config = mQLoRAConfig.value();
        }

        // Use factory to create LoRA model with proper architecture dispatch
        ctx.Model = modules::ModelFactory::create_lora_from_pretrained_config(*mConfig,
                                                                              mod_lora,
                                                                              mOptions,
                                                                              comm,
                                                                              allocator,
                                                                              qlora_config);
    } else {
        // Use factory to create base model with proper architecture dispatch
        ctx.Model = modules::ModelFactory::create_from_pretrained_config(*mConfig,
                                                                         mOptions,
                                                                         comm.rank(),
                                                                         comm.world_size(),
                                                                         allocator);
    }

    // DEBUG: GPU memory after model creation (before run state)
    if (mOptions.DebugMemoryBreakdown && comm.rank() == 0) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "[DEBUG-MEM] After model creation: GPU used=" << (total_mem - free_mem) / (1024 * 1024)
                  << " MiB, free=" << free_mem / (1024 * 1024) << " MiB, total=" << total_mem / (1024 * 1024) << " MiB"
                  << std::endl;
    }

    ctx.Model->allocate_run_state(mOptions, comm, B, T, /*allocate_optimizer=*/true);

    // DEBUG: GPU memory after run state allocation
    if (mOptions.DebugMemoryBreakdown && comm.rank() == 0) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "[DEBUG-MEM] After run state alloc: GPU used=" << (total_mem - free_mem) / (1024 * 1024)
                  << " MiB, free=" << free_mem / (1024 * 1024) << " MiB" << std::endl;
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
        if (mHasCrashed.load()) throw std::runtime_error("Another worker has crashed, exiting.");
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
            // Per-GPU async completion: mark this GPU's outstanding item done.
            if (mCtxDone) mCtxDone[static_cast<std::size_t>(comm.local_rank())].fetch_add(1, std::memory_order_release);
        } else {
            std::this_thread::yield();
        }
        loop_iteration++;
    }
    if (mHasCrashed.load()) {
        // A peer crashed: it will never reach the barrier, and this device may
        // hold aborted collective kernels — skip the syncs and free best-effort.
        try {
            if (ctx.FullStepGraph) {
                ctx.FullStepGraph->reset_capture();
                ctx.FullStepGraph.reset();
            }
            ctx.Model.reset();
            ctx.GPUUtil.reset();
        } catch (...) {
            // Teardown after a crash; the root cause is already recorded.
        }
        return;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();

    // free resources
    if (ctx.FullStepGraph) {
        ctx.FullStepGraph->reset_capture();
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
    run_work(
        [&result](sThreadContext& ctx) {
            auto& rs = ctx.Model->get_run_state();
            if (rs.Allocator) {
                result = rs.Allocator->get_allocation_segments();
            }
        },
        gpu_id);
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
    run_work([&result](sThreadContext& ctx) { result = ctx.Model->get_run_state().Stack.get_allocation_stats(); },
             gpu_id);
    return result;
}

// =============================================================================
// Phase-tree / region / layout introspection (design/buffer-runtime-v4.md).
// =============================================================================
//
// Each getter captures the trainer's (B, T) and force-compiles the forward +
// backward DSL graphs before collecting. The debug collectors read
// `CompiledGraph::tensor_meta` + `phase_tree`, which are only populated by
// `compile_graphs()`, and the trainer constructor itself does NOT trigger
// that — it happens lazily on first forward/backward. Noop for non-DSL
// models (the dynamic_cast to DslModel fails harmlessly).

std::vector<dsl::DebugTensorEntry> MultiGPUPyTrainer::get_debug_tensor_layout() {
    std::vector<dsl::DebugTensorEntry> result;
    const long b = B;
    const long t = T;
    run_work(
        [&result, b, t](sThreadContext& ctx) {
            auto* m = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!m) {
                return;
            }
            if (auto* exec = m->graph_executor()) {
                exec->ensure_graphs_compiled(b, t);
            }
            result = dsl::collect_tensor_layout(*m);
        },
        /*gpu_id=*/0);
    return result;
}

dsl::DebugArenaSummary MultiGPUPyTrainer::get_debug_arena_summary() {
    dsl::DebugArenaSummary result{};
    const long b = B;
    const long t = T;
    run_work(
        [&result, b, t](sThreadContext& ctx) {
            auto* m = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!m) {
                return;
            }
            if (auto* exec = m->graph_executor()) {
                exec->ensure_graphs_compiled(b, t);
            }
            result = dsl::collect_arena_summary(*m);
        },
        /*gpu_id=*/0);
    return result;
}

dsl::DebugDescriptorSummary MultiGPUPyTrainer::get_debug_descriptor_summary() {
    dsl::DebugDescriptorSummary result{};
    const long b = B;
    const long t = T;
    run_work(
        [&result, b, t](sThreadContext& ctx) {
            auto* m = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!m) {
                return;
            }
            if (auto* exec = m->graph_executor()) {
                exec->ensure_graphs_compiled(b, t);
            }
            result = dsl::collect_descriptor_summary(*m);
        },
        /*gpu_id=*/0);
    return result;
}

dsl::DebugFusionPreview MultiGPUPyTrainer::get_debug_fusion_preview() {
    dsl::DebugFusionPreview result{};
    const long b = B;
    const long t = T;
    run_work(
        [&result, b, t](sThreadContext& ctx) {
            auto* m = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!m) {
                return;
            }
            if (auto* exec = m->graph_executor()) {
                exec->ensure_graphs_compiled(b, t);
            }
            result = dsl::collect_fusion_preview(*m);
        },
        /*gpu_id=*/0);
    return result;
}

dsl::DebugBufferPlanSummary MultiGPUPyTrainer::get_debug_buffer_plan_summary() {
    dsl::DebugBufferPlanSummary result{};
    run_work(
        [&result](sThreadContext& ctx) {
            auto* m = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!m) {
                return;
            }
            result = dsl::collect_buffer_plan_summary(*m);
        },
        /*gpu_id=*/0);
    return result;
}

dsl::DebugPhaseTree MultiGPUPyTrainer::get_debug_phase_tree(bool is_backward) {
    dsl::DebugPhaseTree result;
    const long b = B;
    const long t = T;
    run_work(
        [&result, is_backward, b, t](sThreadContext& ctx) {
            auto* m = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!m) {
                return;
            }
            if (auto* exec = m->graph_executor()) {
                exec->ensure_graphs_compiled(b, t);
            }
            result = dsl::collect_phase_tree(*m, is_backward);
        },
        /*gpu_id=*/0);
    return result;
}

std::vector<dsl::DebugAliasingPair> MultiGPUPyTrainer::get_debug_static_aliasing() {
    std::vector<dsl::DebugAliasingPair> result;
    const long b = B;
    const long t = T;
    run_work(
        [&result, b, t](sThreadContext& ctx) {
            auto* m = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!m) {
                return;
            }
            if (auto* exec = m->graph_executor()) {
                exec->ensure_graphs_compiled(b, t);
            }
            result = dsl::collect_static_aliasing(*m);
        },
        /*gpu_id=*/0);
    return result;
}

dsl::DebugTensorResolution
MultiGPUPyTrainer::get_debug_tensor_resolution(const std::string& name, int tid, bool is_backward) {
    dsl::DebugTensorResolution result;
    const long b = B;
    const long t = T;
    run_work(
        [&result, &name, tid, is_backward, b, t](sThreadContext& ctx) {
            auto* m = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!m) {
                return;
            }
            if (auto* exec = m->graph_executor()) {
                exec->ensure_graphs_compiled(b, t);
            }
            result = dsl::resolve_tensor(*m, name, tid, is_backward);
        },
        /*gpu_id=*/0);
    return result;
}

std::vector<std::pair<long, long>> MultiGPUPyTrainer::shrink_stack_after_warmup(long safety_bytes,
                                                                                long min_savings_bytes) {
    std::vector<std::pair<long, long>> results(mContexts.size(), {0L, 0L});
    run_work([&results, safety_bytes, min_savings_bytes](sThreadContext& ctx) {
        auto* dsl_rs = dynamic_cast<dsl::DslRunState*>(&ctx.Model->get_run_state());
        if (!dsl_rs) {
            return;  // Only the DSL run-state has a shrinkable stack.
        }
        const long old_size = static_cast<long>(dsl_rs->Stack.capacity());
        const long new_size = dsl_rs->shrink_stack_to_high_water_mark(safety_bytes, min_savings_bytes);
        if (new_size > 0) {
            // The new stack buffer lives at a different device address, so
            // every CUDA-graph capture that hardcoded the old buffer's
            // pointers is invalid. Tear down:
            //   1. The per-trainer "full-step" graph capture (inputs +
            //      forward + backward + optimizer rolled into one graph).
            //   2. Every graph owned by the GraphExecutor — whole-graph,
            //      per-layer forward/backward, and split-attention
            //      per-segment captures — plus their stack checkpoints.
            // They re-capture on step 1 against the new buffer; one extra
            // capture pass is cheap vs. carrying the inflated upfront
            // allocation for the whole training run.
            if (ctx.FullStepGraph) {
                ctx.FullStepGraph->reset_capture();
            }
            if (auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get())) {
                dsl_model->invalidate_cuda_graphs();
            }
        }
        results[static_cast<std::size_t>(ctx.Communicator->rank())] = {new_size, old_size};
    });
    return results;
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
    run_work(
        [&result](sThreadContext& ctx) {
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
        },
        gpu_id);
    return result;
}

// ---- Debug-only dispatch-PP sub-range parity ------------------------------
// Stage host token ids into GPU 0's device input buffer and fill sequential
// position ids, mirroring step()'s single-GPU staging.
#define DISPATCH_PP_DBG_STAGE(ctx, inputs_ptr, targets_ptr)                                          \
    do {                                                                                             \
        auto* _ib = (ctx).Model->get_input_buffer().get<std::int32_t>();                             \
        std::memcpy(_ib, (inputs_ptr), static_cast<std::size_t>(B) * T * sizeof(std::int32_t));      \
        if ((targets_ptr) != nullptr) {                                                              \
            auto* _tb = (ctx).Model->get_target_buffer().get<std::int32_t>();                        \
            std::memcpy(_tb, (targets_ptr), static_cast<std::size_t>(B) * T * sizeof(std::int32_t)); \
        }                                                                                            \
        Tensor _pos = (ctx).Model->get_position_ids_buffer();                                        \
        auto* _pb = _pos.get<std::int32_t>();                                                        \
        const int _planes = (_pos.Rank == 3) ? static_cast<int>(_pos.Sizes[0]) : 1;                  \
        fill_sequential_position_ids(_pb, _planes, B, T);                                            \
    } while (0)

std::vector<float> MultiGPUPyTrainer::dispatch_pp_forward_hidden(const std::int32_t* inputs) {
    std::vector<float> result;
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!model) throw std::runtime_error("dispatch_pp_forward_hidden: DSL model required");
            DISPATCH_PP_DBG_STAGE(ctx, inputs, nullptr);
            auto out = model->dispatch_pp_forward_hidden(ctx.Model->get_input_buffer(),
                                                         ctx.Model->get_position_ids_buffer(),
                                                         *ctx.Communicator);
            if (ctx.Communicator->local_rank() == 0) result = std::move(out);
        },
        0);
    return result;
}

std::vector<float> MultiGPUPyTrainer::dispatch_pp_forward_subranges(const std::int32_t* inputs, int split_after_block) {
    std::vector<float> result;
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!model) throw std::runtime_error("dispatch_pp_forward_subranges: DSL model required");
            DISPATCH_PP_DBG_STAGE(ctx, inputs, nullptr);
            auto out = model->dispatch_pp_forward_subranges(ctx.Model->get_input_buffer(),
                                                            ctx.Model->get_position_ids_buffer(),
                                                            *ctx.Communicator,
                                                            split_after_block);
            if (ctx.Communicator->local_rank() == 0) result = std::move(out);
        },
        0);
    return result;
}

std::vector<float> MultiGPUPyTrainer::dispatch_pp_grad_norms_whole(const std::int32_t* inputs,
                                                                   const std::int32_t* targets) {
    std::vector<float> result;
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!model) throw std::runtime_error("dispatch_pp_grad_norms_whole: DSL model required");
            DISPATCH_PP_DBG_STAGE(ctx, inputs, targets);
            auto out = model->dispatch_pp_grad_norms_whole(ctx.Model->get_input_buffer(),
                                                           ctx.Model->get_target_buffer(),
                                                           ctx.Model->get_position_ids_buffer(),
                                                           *ctx.Communicator);
            if (ctx.Communicator->local_rank() == 0) result = std::move(out);
        },
        0);
    return result;
}

std::vector<float> MultiGPUPyTrainer::dispatch_pp_grad_norms_subranges(const std::int32_t* inputs,
                                                                       const std::int32_t* targets,
                                                                       int split_after_block) {
    std::vector<float> result;
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!model) throw std::runtime_error("dispatch_pp_grad_norms_subranges: DSL model required");
            DISPATCH_PP_DBG_STAGE(ctx, inputs, targets);
            auto out = model->dispatch_pp_grad_norms_subranges(ctx.Model->get_input_buffer(),
                                                               ctx.Model->get_target_buffer(),
                                                               ctx.Model->get_position_ids_buffer(),
                                                               *ctx.Communicator,
                                                               split_after_block);
            if (ctx.Communicator->local_rank() == 0) result = std::move(out);
        },
        0);
    return result;
}

std::vector<float> MultiGPUPyTrainer::dispatch_pp_forward_hidden_multigpu(const std::int32_t* inputs,
                                                                          const std::vector<int>& los,
                                                                          const std::vector<int>& his) {
    if (los.size() != his.size() || los.empty()) {
        throw std::runtime_error("dispatch_pp_forward_hidden_multigpu: bad stage ranges");
    }
    const int ngpu = static_cast<int>(mContexts.size());
    const int num_stages = static_cast<int>(los.size());
    std::vector<float> result;
    // The fused-residual block returns (mlp_out, residual_after_attn): block hi+1's
    // first op reads blocks[hi].res_att (the residual accumulator after attention)
    // and blocks[hi].mlp_down (x = the previous block's MLP output), folding x in.
    // Hand both over by name through host memory between stage GPUs.
    std::vector<std::pair<std::string, std::vector<std::byte>>> boundary;

    for (int si = 0; si < num_stages; ++si) {
        const int gpu = si % ngpu;
        const int lo = los[static_cast<std::size_t>(si)];
        const int hi = his[static_cast<std::size_t>(si)];
        const bool is_last = (si == num_stages - 1);
        std::vector<std::pair<std::string, std::vector<std::byte>>> inject_named = boundary;
        std::vector<std::pair<std::string, std::vector<std::byte>>> next_boundary;
        std::vector<float> final_hidden;

        run_work(
            [&](sThreadContext& ctx) {
                auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
                if (!model) throw std::runtime_error("dispatch_pp_forward_hidden_multigpu: DSL model required");
                DISPATCH_PP_DBG_STAGE(ctx, inputs, nullptr);
                model->dispatch_pp_forward_stage(ctx.Model->get_input_buffer(),
                                                 ctx.Model->get_position_ids_buffer(),
                                                 *ctx.Communicator,
                                                 lo,
                                                 hi,
                                                 std::move(inject_named),
                                                 /*preserve_output=*/!is_last);
                auto* ge = model->graph_executor();
                if (is_last) {
                    final_hidden = ge->last_block_hidden_f32();
                } else {
                    const std::string res_name = "blocks[" + std::to_string(hi) + "].res_att";
                    const std::string x_name = "blocks[" + std::to_string(hi) + "].mlp_down";
                    next_boundary.emplace_back(res_name, ge->read_named_bytes(res_name));
                    next_boundary.emplace_back(x_name, ge->read_named_bytes(x_name));
                    // Drop the preserved stage's stack allocations now that the
                    // boundary is read, so this GPU is clean if reused (round-robin).
                    ge->restore_stage_base();
                }
            },
            gpu);

        if (is_last) {
            result = std::move(final_hidden);
        } else {
            boundary = std::move(next_boundary);
        }
    }
    return result;
}

std::vector<float> MultiGPUPyTrainer::dispatch_pp_grad_norms_multigpu(const std::int32_t* inputs,
                                                                      const std::int32_t* targets,
                                                                      const std::vector<int>& los,
                                                                      const std::vector<int>& his) {
    if (los.size() != his.size() || los.empty()) {
        throw std::runtime_error("dispatch_pp_grad_norms_multigpu: bad stage ranges");
    }
    const int ngpu = static_cast<int>(mContexts.size());
    const int num_stages = static_cast<int>(los.size());
    const int num_layers = his.back() + 1;
    std::vector<float> result(static_cast<std::size_t>(num_layers), 0.0f);
    // Backward boundary gradients (d_blocks[lo-1].res_att / .mlp_down) handed from
    // a higher stage's GPU to the next lower stage's GPU through host memory.
    std::vector<std::pair<std::string, std::vector<std::byte>>> boundary;

    // Backward visits stages in reverse forward order: the stage owning the last
    // block (and the loss) runs first.
    for (int si = num_stages - 1; si >= 0; --si) {
        const int gpu = si % ngpu;
        const int lo = los[static_cast<std::size_t>(si)];
        const int hi = his[static_cast<std::size_t>(si)];
        const bool is_loss_stage = (si == num_stages - 1);
        std::vector<std::pair<std::string, std::vector<std::byte>>> inject_named = boundary;
        std::vector<std::pair<std::string, std::vector<std::byte>>> next_boundary;
        std::vector<float> stage_norms;

        run_work(
            [&](sThreadContext& ctx) {
                auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
                if (!model) throw std::runtime_error("dispatch_pp_grad_norms_multigpu: DSL model required");
                DISPATCH_PP_DBG_STAGE(ctx, inputs, targets);
                model->dispatch_pp_backward_stage(ctx.Model->get_input_buffer(),
                                                  ctx.Model->get_target_buffer(),
                                                  ctx.Model->get_position_ids_buffer(),
                                                  *ctx.Communicator,
                                                  lo,
                                                  hi,
                                                  is_loss_stage,
                                                  /*fwd_inject=*/{},  // harness: whole-from-start forward fallback
                                                  std::move(inject_named));
                auto* ge = model->graph_executor();
                stage_norms = ge->block_grad_norms();
                if (lo > 0) {
                    const std::string rn = "d_blocks[" + std::to_string(lo - 1) + "].res_att";
                    const std::string xn = "d_blocks[" + std::to_string(lo - 1) + "].mlp_down";
                    next_boundary.emplace_back(rn, ge->read_named_bytes(rn));
                    next_boundary.emplace_back(xn, ge->read_named_bytes(xn));
                }
            },
            gpu);

        for (int L = lo; L <= hi; ++L) {
            result[static_cast<std::size_t>(L)] = stage_norms[static_cast<std::size_t>(L)];
        }
        boundary = std::move(next_boundary);
    }
    return result;
}

float MultiGPUPyTrainer::dispatch_pp_train_step(const std::int32_t* inputs,
                                                const std::int32_t* targets,
                                                const optimizers::OptimizerConfig& opt_config,
                                                int step_idx) {
    float loss = 0.0f;
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!model) throw std::runtime_error("dispatch_pp_train_step: DSL model required");
            DISPATCH_PP_DBG_STAGE(ctx, inputs, targets);
            const float l = model->dispatch_pp_train_step(ctx.Model->get_input_buffer(),
                                                          ctx.Model->get_target_buffer(),
                                                          ctx.Model->get_position_ids_buffer(),
                                                          *ctx.Communicator,
                                                          opt_config,
                                                          step_idx);
            if (ctx.Communicator->local_rank() == 0) loss = l;
        },
        0);
    return loss;
}

float MultiGPUPyTrainer::dispatch_pp_train_step_multigpu(const std::int32_t* inputs,
                                                         const std::int32_t* targets,
                                                         const std::vector<int>& los,
                                                         const std::vector<int>& his,
                                                         const optimizers::OptimizerConfig& opt_config,
                                                         int step_idx,
                                                         bool stale,
                                                         int num_microbatches) {
    if (los.size() != his.size() || los.empty()) {
        throw std::runtime_error("dispatch_pp_train_step_multigpu: bad stage ranges");
    }
    const int ngpu = static_cast<int>(mContexts.size());
    const int num_stages = static_cast<int>(los.size());
    const int M = std::max(1, num_microbatches);
    const long mb_stride = static_cast<long>(B) * static_cast<long>(T);  // tokens per microbatch
    float loss = 0.0f;

    using Boundary = std::vector<std::pair<std::string, std::vector<std::byte>>>;

    // 0. Reset each GPU's compute stack to its clean per-step base. The dispatch
    //    sub-range forwards/backwards run with skip_finalize (so boundary tensors and
    //    saves survive the cross-GPU reads), which leaves residue on the bump-allocated
    //    stack; without this reset it accumulates step over step and overflows.
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (model && model->graph_executor()) model->graph_executor()->dispatch_reset_stack();
        },
        -1);

    // 1. Forward-dispatch stages in order with cross-GPU boundary handoff. Capture
    //    each stage's INPUT boundary (block lo-1's residual) so the backward pass can
    //    re-forward just that one stage instead of the whole model -- this bounds
    //    resident activations to a single stage. Only stages [0..N-2] need to run:
    //    each produces the input boundary for the next stage. The last stage's output
    //    is consumed by nothing here (its loss is recomputed in the backward pass), so
    //    running it would just leak resident state. After reading a stage's boundary,
    //    restore the stage base so the preserved activations are reclaimed and the
    //    (reused) GPU starts the next step clean -- otherwise saves accumulate on the
    //    compute stack step over step and eventually overflow it.
    // Forward STAGE-LEVEL PIPELINE: each forward stage [0..N-2] is dispatched to GPU s%N
    // and runs ALL M microbatches there with the stage's weights held RESIDENT (gathered
    // once via the enlarged prefetch, reused for every microbatch) -> the stage streams
    // once per step. Stages run concurrently across GPUs, pipelined by per-(stage,
    // microbatch) ready flags: stage s microbatch m waits for stage s-1 microbatch m's
    // boundary. dispatch_async's is_idle backpressure bounds the in-flight depth to N
    // stages. stage_inputs[s][m] (input to stage s, microbatch m) is captured for the
    // backward; the last stage's forward is recomputed in the backward pass.
    std::vector<std::vector<Boundary>> stage_inputs(static_cast<std::size_t>(num_stages),
                                                    std::vector<Boundary>(static_cast<std::size_t>(M)));
    {
        const int last_fwd = num_stages - 2;
        std::vector<std::vector<Boundary>> fwd_out(static_cast<std::size_t>(num_stages),
                                                   std::vector<Boundary>(static_cast<std::size_t>(M)));
        const int nflags = std::max(1, num_stages * M);
        std::unique_ptr<std::atomic<int>[]> ready(new std::atomic<int>[nflags]);
        for (int i = 0; i < nflags; ++i)
            ready[i].store(0, std::memory_order_relaxed);
        std::atomic<int>* readyp = ready.get();
        for (int s = 0; s <= last_fwd; ++s) {
            const int lo = los[static_cast<std::size_t>(s)];
            const int hi = his[static_cast<std::size_t>(s)];
            std::vector<Boundary>* my_out = &fwd_out[static_cast<std::size_t>(s)];
            std::vector<Boundary>* up_out = (s > 0) ? &fwd_out[static_cast<std::size_t>(s - 1)] : nullptr;
            std::vector<Boundary>* sin = &stage_inputs[static_cast<std::size_t>(s)];
            dispatch_async(
                [this, inputs, mb_stride, M, s, lo, hi, my_out, up_out, sin, readyp](sThreadContext& ctx) {
                    auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
                    if (!model) throw std::runtime_error("dispatch_pp_train_step_multigpu: DSL model required");
                    auto* ge = model->graph_executor();
                    const std::string rn = "blocks[" + std::to_string(hi) + "].res_att";
                    const std::string xn = "blocks[" + std::to_string(hi) + "].mlp_down";
                    for (int m = 0; m < M; ++m) {
                        Boundary inj;
                        if (up_out) {
                            while (readyp[(s - 1) * M + m].load(std::memory_order_acquire) == 0)
                                std::this_thread::yield();
                            inj = (*up_out)[static_cast<std::size_t>(m)];
                        }
                        (*sin)[static_cast<std::size_t>(m)] = inj;
                        DISPATCH_PP_DBG_STAGE(ctx, inputs + static_cast<std::size_t>(m) * mb_stride, nullptr);
                        model->dispatch_pp_forward_stage(ctx.Model->get_input_buffer(),
                                                         ctx.Model->get_position_ids_buffer(),
                                                         *ctx.Communicator,
                                                         lo,
                                                         hi,
                                                         std::move(inj),
                                                         /*preserve_output=*/true);
                        Boundary o;
                        o.emplace_back(rn, ge->read_named_bytes(rn));
                        o.emplace_back(xn, ge->read_named_bytes(xn));
                        ge->restore_stage_base();  // reclaim this microbatch's preserved activations
                        (*my_out)[static_cast<std::size_t>(m)] = std::move(o);
                        readyp[s * M + m].store(1, std::memory_order_release);
                    }
                },
                s % ngpu);
        }
        for (int g = 0; g < ngpu; ++g)
            wait_gpu(g);
        for (int m = 0; m < M; ++m)
            stage_inputs[static_cast<std::size_t>(num_stages - 1)][static_cast<std::size_t>(m)] =
                (last_fwd >= 0) ? fwd_out[static_cast<std::size_t>(last_fwd)][static_cast<std::size_t>(m)] : Boundary{};
    }

    // 2. Backward WAVEFRONT (mirror of the forward). Grads are pre-zeroed on every GPU,
    //    then each backward_stage accumulates (micro_step=1, no per-task zero) -- the
    //    diagonal hits a stage's microbatches out of m-order, so order-independent
    //    accumulation is required. In diagonal wave w, microbatch m runs backward step
    //    t=w-m -> stage s=num_stages-1-t on GPU s%N; the grad boundary d_blocks[lo-1].*
    //    is handed down per microbatch. After the wavefront, each stage's accumulated
    //    grads are collected once from its GPU.
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (model) model->dispatch_pp_zero_grads();
        },
        -1);

    std::vector<double> loss_per_mb(static_cast<std::size_t>(M), 0.0);
    // Valid (non-pad) tokens per microbatch, counted by the loss stage on its GPU. Summed
    // after the wavefront and published to the optimizer GPU for valid-token grad-norm scaling.
    std::vector<int> valid_per_mb(static_cast<std::size_t>(M), 0);
    {
        // grad_out[s][m] = stage s's grad-input boundary (d_blocks[lo-1].*) for microbatch
        // m, consumed by stage s-1. Each backward stage runs on GPU s%N, all M microbatches
        // resident (grads accumulated, micro_step=1 since pre-zeroed). Pipelined: stage s
        // microbatch m waits for stage s+1 microbatch m's grad boundary.
        std::vector<std::vector<Boundary>> grad_out(static_cast<std::size_t>(num_stages),
                                                    std::vector<Boundary>(static_cast<std::size_t>(M)));
        const int nflags = std::max(1, num_stages * M);
        std::unique_ptr<std::atomic<int>[]> ready(new std::atomic<int>[nflags]);
        for (int i = 0; i < nflags; ++i)
            ready[i].store(0, std::memory_order_relaxed);
        std::atomic<int>* readyp = ready.get();
        std::vector<double>* lpm = &loss_per_mb;
        std::vector<int>* vtpm = &valid_per_mb;
        for (int s = num_stages - 1; s >= 0; --s) {
            const int lo = los[static_cast<std::size_t>(s)];
            const int hi = his[static_cast<std::size_t>(s)];
            const bool is_loss = (s == num_stages - 1);
            std::vector<Boundary>* my_g = &grad_out[static_cast<std::size_t>(s)];
            std::vector<Boundary>* up_g = (s < num_stages - 1) ? &grad_out[static_cast<std::size_t>(s + 1)] : nullptr;
            std::vector<Boundary>* sin = &stage_inputs[static_cast<std::size_t>(s)];
            dispatch_async(
                [this, inputs, targets, mb_stride, M, s, lo, hi, is_loss, my_g, up_g, sin, readyp, lpm, vtpm](
                    sThreadContext& ctx) {
                    auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
                    if (!model) throw std::runtime_error("dispatch_pp_train_step_multigpu: DSL model required");
                    auto* ge = model->graph_executor();
                    const std::string rn = "d_blocks[" + std::to_string(lo - 1) + "].res_att";
                    const std::string xn = "d_blocks[" + std::to_string(lo - 1) + "].mlp_down";
                    for (int m = 0; m < M; ++m) {
                        Boundary ginj;
                        if (up_g) {
                            while (readyp[(s + 1) * M + m].load(std::memory_order_acquire) == 0)
                                std::this_thread::yield();
                            ginj = (*up_g)[static_cast<std::size_t>(m)];
                        }
                        Boundary finj = (*sin)[static_cast<std::size_t>(m)];
                        DISPATCH_PP_DBG_STAGE(ctx,
                                              inputs + static_cast<std::size_t>(m) * mb_stride,
                                              targets + static_cast<std::size_t>(m) * mb_stride);
                        model->dispatch_pp_backward_stage(ctx.Model->get_input_buffer(),
                                                          ctx.Model->get_target_buffer(),
                                                          ctx.Model->get_position_ids_buffer(),
                                                          *ctx.Communicator,
                                                          lo,
                                                          hi,
                                                          is_loss,
                                                          std::move(finj),
                                                          std::move(ginj),
                                                          /*micro_step=*/1,  // pre-zeroed -> always accumulate
                                                          /*total_micro=*/M);
                        if (is_loss) {
                            (*lpm)[static_cast<std::size_t>(m)] = static_cast<double>(model->dispatch_pp_raw_loss());
                            (*vtpm)[static_cast<std::size_t>(m)] = model->dispatch_pp_loss_valid_tokens();
                        }
                        if (lo > 0) {
                            Boundary g;
                            g.emplace_back(rn, ge->read_named_bytes(rn));
                            g.emplace_back(xn, ge->read_named_bytes(xn));
                            (*my_g)[static_cast<std::size_t>(m)] = std::move(g);
                        }
                        // Reclaim this microbatch's backward activations before the next.
                        // Grads persist in the grad store and the grad boundary is already
                        // read out above; one GPU runs all M microbatches of this stage, so
                        // without this the compute stack accumulates M x stage activations.
                        // restore_stage_base is a no-op here (the stage base is only
                        // checkpointed in the forward), so reset to the per-step base.
                        ge->dispatch_reset_stack();
                        readyp[s * M + m].store(1, std::memory_order_release);
                    }
                },
                s % ngpu);
        }
        for (int g = 0; g < ngpu; ++g)
            wait_gpu(g);
    }
    {
        double ls = 0.0;
        for (double x : loss_per_mb)
            ls += x;
        loss = static_cast<float>(ls / static_cast<double>(M));
    }
    int step_valid_tokens = 0;
    for (int v : valid_per_mb)
        step_valid_tokens += v;

    // Collect each stage's accumulated grads from its GPU (grads stay resident on the
    // stage's GPU until read here, after the whole backward wavefront).
    std::vector<std::pair<std::string, std::vector<std::byte>>> collected;
    for (int si = 0; si < num_stages; ++si) {
        const int gpu = si % ngpu;
        const int lo = los[static_cast<std::size_t>(si)];
        const int hi = his[static_cast<std::size_t>(si)];
        const bool is_loss = (si == num_stages - 1);
        std::vector<std::pair<std::string, std::vector<std::byte>>> stage_grads;
        run_work(
            [&, lo, hi, is_loss](sThreadContext& ctx) {
                auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
                if (model)
                    stage_grads = model->dispatch_pp_read_block_grads(lo,
                                                                      hi,
                                                                      /*include_head=*/is_loss,
                                                                      /*include_embed=*/lo == 0);
            },
            gpu);
        for (auto& g : stage_grads)
            collected.push_back(std::move(g));
    }

    // 2. Optimizer + broadcast. In synchronous mode, apply this step's grads now.
    //    In one-step-stale mode, defer: apply the *previous* step's grads (so the
    //    weights this step just trained on are one update behind — the RoundPipe v1
    //    staleness), then stash this step's grads for the next call.
    if (stale) {
        if (!mDispatchPpPendingGrads.empty()) {
            // The deferred grads belong to the previous step — scale by its valid-token count.
            dispatch_pp_apply_grads_(mDispatchPpPendingGrads,
                                     opt_config,
                                     ++mDispatchPpAppliedStep,
                                     mDispatchPpPendingValidTokens);
        }
        mDispatchPpPendingGrads = std::move(collected);
        mDispatchPpPendingValidTokens = step_valid_tokens;
    } else {
        dispatch_pp_apply_grads_(collected, opt_config, step_idx + 1, step_valid_tokens);
    }
    return loss;
}

void MultiGPUPyTrainer::dispatch_pp_flush_pending(const optimizers::OptimizerConfig& opt_config) {
    if (mDispatchPpPendingGrads.empty()) return;
    dispatch_pp_apply_grads_(mDispatchPpPendingGrads,
                             opt_config,
                             ++mDispatchPpAppliedStep,
                             mDispatchPpPendingValidTokens);
    mDispatchPpPendingGrads.clear();
}

void MultiGPUPyTrainer::dispatch_pp_apply_grads_(
    const std::vector<std::pair<std::string, std::vector<std::byte>>>& collected,
    const optimizers::OptimizerConfig& opt_config,
    int opt_step_1based,
    int valid_tokens) {
    // GPU 0 holds the master replica: write the collected grads into its store and
    // run the optimizer there.
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!model) throw std::runtime_error("dispatch_pp_apply_grads_: DSL model required");
            // The loss GPU counted valid tokens; publish the step total here so the optimizer
            // (on this master GPU) scales the grad norm per valid token, not per padded token.
            model->set_dispatch_pp_valid_tokens(valid_tokens);
            model->dispatch_pp_write_grads(collected);
            mDispatchPpLastGradNorm =
                model->dispatch_pp_apply_optimizer(*ctx.Communicator, opt_config, opt_step_1based);
        },
        0);
    // Broadcast GPU 0's updated weights to every replica so the pool is consistent
    // for the next step's stages (which run on any GPU).
    std::vector<std::pair<std::string, std::vector<std::byte>>> master;
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (model) master = model->dispatch_pp_read_weights();
        },
        0);
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (model) model->dispatch_pp_write_weights(master);
        },
        -1);
}
#undef DISPATCH_PP_DBG_STAGE

std::unordered_map<std::string, std::size_t> MultiGPUPyTrainer::dispatch_pp_weight_residency() {
    std::unordered_map<std::string, std::size_t> out;
    run_work(
        [&](sThreadContext& ctx) {
            auto* model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!model) throw std::runtime_error("dispatch_pp_weight_residency: DSL model required");
            auto* wm = model->weight_manager();
            if (!wm) {
                // No weight manager => weights are fully resident (not streamed);
                // there are no streaming slots to report.
                out["total_persistent_bytes"] = 0;
                out["gpu_prefetch_buffer_bytes"] = 0;
                out["prefetch_slot_count"] = 0;
                return;
            }
            out["total_persistent_bytes"] = wm->total_persistent_bytes();
            out["gpu_prefetch_buffer_bytes"] = wm->gpu_prefetch_buffer_bytes();
            out["prefetch_slot_count"] = static_cast<std::size_t>(wm->prefetch_slot_count());
        },
        0);
    return out;
}

std::vector<std::pair<std::string, Tensor>> MultiGPUPyTrainer::get_lora_gradients(int gpu_id) {
    std::vector<std::pair<std::string, Tensor>> result;
    run_work(
        [&result](sThreadContext& ctx) {
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
                    for (auto& c : out)
                        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
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
            CUDA_CHECK(cudaDeviceSynchronize());

            for (int l = 0; l < config.NumLayers; ++l) {
                bool unused_accumulate = false;
                auto& block =
                    dsl_model->lora_grads().get_block_full(l, /*stream=*/nullptr, *ctx.Communicator, unused_accumulate);
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

                // Dense MLP LoRA (present in dense and hybrid non-MoE blocks).
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

                // MoE LoRA (if this layer is an MoE block).
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
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        },
        gpu_id);
    return result;
}

std::vector<std::pair<std::string, Tensor>> MultiGPUPyTrainer::get_lora_weights(int gpu_id) {
    std::vector<std::pair<std::string, Tensor>> result;
    run_work(
        [&result](sThreadContext& ctx) {
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
                    for (auto& c : out)
                        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
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

                // Dense MLP LoRA (present in dense and hybrid non-MoE blocks).
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

                // MoE LoRA (if this layer is an MoE block).
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
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        },
        gpu_id);
    return result;
}

std::vector<float> MultiGPUPyTrainer::compute_logprobs(const std::int32_t* input_ids,
                                                       const std::int32_t* targets,
                                                       int B,
                                                       int T,
                                                       bool use_lora,
                                                       const std::int32_t* position_ids,
                                                       const float* temperatures) {
    std::vector<float> result;
    run_work([&result, input_ids, targets, B, T, use_lora, position_ids, temperatures](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("compute_logprobs: model is not a DslModel");
        }
        auto logprobs =
            dsl_model
                ->compute_logprobs(input_ids, targets, B, T, use_lora, *ctx.Communicator, position_ids, temperatures);
        if (ctx.Communicator->local_rank() == 0) {
            result = std::move(logprobs);
        }
    });
    return result;
}

void MultiGPUPyTrainer::step_with_custom_loss(const std::int32_t* inputs,
                                              const std::int32_t* targets,
                                              const float* per_token_grads,
                                              const std::int32_t* position_ids,
                                              const float* temperatures) {
    // Distribute inputs, targets, and position_ids to each GPU's CPU-side buffers.
    const int ep_size = std::max(1, mOptions.EPSize);
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
        const std::size_t pos_stride =
            static_cast<std::size_t>(pos_planes) * static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
        const int src_row = host_batch_row_for_local_rank(i, ep_size);

        std::memcpy(ib, inputs + src_row * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + src_row * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            // Python binding provides 2D [B, T] position IDs (one plane per GPU).
            // For mRoPE models the buffer is [3, B, T] — replicate the single plane.
            const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(src_row) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(fmt::format("step_with_custom_loss: micro_step {} >= grad_accumulation {}",
                                             mTrainMicroStep,
                                             mGradAccumulation));
    }

    run_work([micro_idx = mTrainMicroStep,
              micro_batches = mGradAccumulation,
              per_token_grads,
              temperatures,
              B = this->B,
              T = this->T](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("step_with_custom_loss: model is not a DslModel");
        }

        Tensor inputs_tensor = ctx.Model->get_input_buffer();
        Tensor position_ids_tensor = ctx.Model->get_position_ids_buffer();
        Tensor targets_tensor = ctx.Model->get_target_buffer();

        // Each GPU receives its own slice of the per_token_grads buffer.
        const int gpu_rank = ctx.Communicator->local_rank();
        const int gpu_ep_size = ctx.Communicator->ep_size();
        const int src_row = host_batch_row_for_local_rank(gpu_rank, gpu_ep_size);
        const float* grads_for_this_gpu = per_token_grads + static_cast<std::ptrdiff_t>(src_row) * B * T;
        const float* temps_for_this_gpu = nullptr;
        if (temperatures) {
            temps_for_this_gpu = temperatures + static_cast<std::ptrdiff_t>(src_row) * B * T;
        }

        dsl_model->step_with_custom_loss(inputs_tensor,
                                         position_ids_tensor,
                                         targets_tensor,
                                         grads_for_this_gpu,
                                         micro_batches,
                                         micro_idx,
                                         *ctx.Communicator,
                                         temps_for_this_gpu);
    });

    ++mTrainMicroStep;
}

void MultiGPUPyTrainer::step_grpo_native(const std::int32_t* inputs,
                                         const std::int32_t* targets,
                                         const float* inference_logprobs,
                                         const float* advantages,
                                         const std::uint8_t* loss_mask,
                                         const std::int32_t* sample_starts,
                                         const std::int32_t* sample_ends,
                                         int sample_count,
                                         const std::int32_t* position_ids,
                                         const float* temperatures,
                                         const float* teacher_logprobs,
                                         float loss_scale,
                                         float ipo_mask_low,
                                         float ipo_mask_high,
                                         float adv_tau,
                                         float teacher_tau,
                                         float kl_tau) {
    const int ep_size = std::max(1, mOptions.EPSize);
    for (int i = 0; i < (int)mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        if (!ctx.Model) {
            throw std::runtime_error(fmt::format("step_grpo_native: ctx[{}].Model is null", i));
        }
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
        const int src_row = host_batch_row_for_local_rank(i, ep_size);

        std::memcpy(ib, inputs + src_row * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + src_row * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(src_row) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(
            fmt::format("step_grpo_native: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
    }

    const dsl::GrpoNativeLossConfig loss_config{
        .loss_scale = loss_scale,
        .ipo_mask_low = ipo_mask_low,
        .ipo_mask_high = ipo_mask_high,
        .adv_tau = adv_tau,
        .teacher_tau = teacher_tau,
        .kl_tau = kl_tau,
    };

    run_work([micro_idx = mTrainMicroStep,
              micro_batches = mGradAccumulation,
              inference_logprobs,
              advantages,
              loss_mask,
              sample_starts,
              sample_ends,
              sample_count,
              temperatures,
              teacher_logprobs,
              loss_config,
              B = this->B,
              T = this->T](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("step_grpo_native: model is not a DslModel");
        }

        Tensor inputs_tensor = ctx.Model->get_input_buffer();
        Tensor position_ids_tensor = ctx.Model->get_position_ids_buffer();
        Tensor targets_tensor = ctx.Model->get_target_buffer();

        const int gpu_rank = ctx.Communicator->local_rank();
        const int gpu_ep_size = ctx.Communicator->ep_size();
        const int src_row = host_batch_row_for_local_rank(gpu_rank, gpu_ep_size);
        const float* temps_for_this_gpu = nullptr;
        if (temperatures) {
            temps_for_this_gpu = temperatures + static_cast<std::ptrdiff_t>(src_row) * B * T;
        }

        dsl_model->step_grpo_native(inputs_tensor,
                                    position_ids_tensor,
                                    targets_tensor,
                                    inference_logprobs,
                                    advantages,
                                    loss_mask,
                                    sample_starts,
                                    sample_ends,
                                    sample_count,
                                    micro_batches,
                                    micro_idx,
                                    *ctx.Communicator,
                                    loss_config,
                                    temps_for_this_gpu,
                                    teacher_logprobs);
    });

    ++mTrainMicroStep;
}

void MultiGPUPyTrainer::step_with_kd(const std::int32_t* inputs,
                                     const std::int32_t* targets,
                                     const std::int32_t* kd_ids,
                                     const float* kd_logprobs,
                                     const std::int32_t* position_ids,
                                     int top_k,
                                     float temperature,
                                     float kd_weight,
                                     float ce_weight) {
    const int ep_size = std::max(1, mOptions.EPSize);
    for (int i = 0; i < (int)mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        if (!ctx.Model) {
            throw std::runtime_error(fmt::format("step_with_kd: ctx[{}].Model is null", i));
        }
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
        const int src_row = host_batch_row_for_local_rank(i, ep_size);

        std::memcpy(ib, inputs + src_row * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + src_row * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(src_row) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(
            fmt::format("step_with_kd: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
    }

    const dsl::KdLossConfig kd_config{
        .top_k = top_k,
        .temperature = temperature,
        .kd_weight = kd_weight,
        .ce_weight = ce_weight,
    };

    run_work([micro_idx = mTrainMicroStep,
              micro_batches = mGradAccumulation,
              kd_ids,
              kd_logprobs,
              kd_config,
              B = this->B,
              T = this->T](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("step_with_kd: model is not a DslModel");
        }

        Tensor inputs_tensor = ctx.Model->get_input_buffer();
        Tensor position_ids_tensor = ctx.Model->get_position_ids_buffer();
        Tensor targets_tensor = ctx.Model->get_target_buffer();

        const int gpu_rank = ctx.Communicator->local_rank();
        const int gpu_ep_size = ctx.Communicator->ep_size();
        const int src_row = host_batch_row_for_local_rank(gpu_rank, gpu_ep_size);
        const std::ptrdiff_t kd_stride = static_cast<std::ptrdiff_t>(B) * static_cast<std::ptrdiff_t>(T) *
                                         static_cast<std::ptrdiff_t>(kd_config.top_k);

        dsl_model->step_with_kd(inputs_tensor,
                                position_ids_tensor,
                                targets_tensor,
                                kd_ids + src_row * kd_stride,
                                kd_logprobs + src_row * kd_stride,
                                micro_batches,
                                micro_idx,
                                *ctx.Communicator,
                                kd_config);
    });

    ++mTrainMicroStep;
}

float MultiGPUPyTrainer::get_kd_loss() {
    float result = 0.0f;
    run_work([&result](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("get_kd_loss: model is not a DslModel");
        }
        // Consume on every rank so accumulators reset; report rank 0 only.
        const float kd_sum = dsl_model->consume_kd_loss_sum();
        if (ctx.Communicator->local_rank() != 0) {
            return;
        }
        auto& rs = dsl_model->get_run_state();
        int valid_tokens = 0;
        if (rs.ValidTokenCount.Data) {
            CUDA_CHECK(cudaMemcpy(&valid_tokens, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost));
        }
        const int world_size = std::max(1, ctx.Communicator->world_size());
        const float avg_valid = valid_tokens > 0 ? static_cast<float>(valid_tokens) / world_size : 0.0f;
        result = avg_valid > 0.0f ? kd_sum / avg_valid : 0.0f;
    });
    return result;
}

std::unordered_map<std::string, float> MultiGPUPyTrainer::get_grpo_native_metrics() {
    std::unordered_map<std::string, float> result;
    run_work([&result](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("get_grpo_native_metrics: model is not a DslModel");
        }
        if (ctx.Communicator->local_rank() != 0) {
            return;
        }
        const auto metrics = dsl_model->consume_grpo_native_metrics();
        result = {
            {"policy_loss", metrics.policy_loss},
            {"mismatch_kl", metrics.mismatch_kl},
            {"masked_mismatch_kl", metrics.masked_mismatch_kl},
            {"unmasked_mismatch_kl", metrics.unmasked_mismatch_kl},
            {"is_masked", metrics.is_masked},
            {"is_masked_low", metrics.is_masked_low},
            {"is_masked_high", metrics.is_masked_high},
            {"teacher_kl", metrics.teacher_kl},
            {"keep_tokens", metrics.keep_tokens},
            {"total_tokens", metrics.total_tokens},
        };
    });
    return result;
}

void MultiGPUPyTrainer::step_dpo_native(const std::int32_t* inputs,
                                        const std::int32_t* targets,
                                        const float* ref_logprobs,
                                        const std::uint8_t* loss_mask,
                                        const std::int32_t* sample_starts,
                                        const std::int32_t* sample_ends,
                                        int sample_count,
                                        const std::int32_t* pair_chosen,
                                        const std::int32_t* pair_rejected,
                                        int pair_count,
                                        const std::int32_t* position_ids,
                                        float loss_scale,
                                        float beta,
                                        int length_norm,
                                        DpoHostLayout layout) {
    const int ep_size = std::max(1, mOptions.EPSize);
    const int host_rows = std::max(1, (int)mContexts.size() / ep_size);
    if (layout.token_rows != 1 && layout.token_rows != host_rows) {
        throw std::invalid_argument(fmt::format(
            "step_dpo_native: per-token arrays have {} rows; expected 1 (shared) or {} (one per host batch row)",
            layout.token_rows,
            host_rows));
    }
    if (layout.sample_rows != 1 && layout.sample_rows != host_rows) {
        throw std::invalid_argument(fmt::format(
            "step_dpo_native: sample/pair arrays have {} rows; expected 1 (shared) or {} (one per host batch row)",
            layout.sample_rows,
            host_rows));
    }
    if (layout.token_len != 0 && layout.token_len != static_cast<long>(B) * static_cast<long>(T)) {
        throw std::invalid_argument(
            fmt::format("step_dpo_native: per-token arrays have {} elements per row; engine expects B*T = {}x{} = {}",
                        layout.token_len,
                        B,
                        T,
                        static_cast<long>(B) * static_cast<long>(T)));
    }
    if (layout.input_cols != 0 && layout.input_cols != static_cast<long>(T)) {
        throw std::invalid_argument(
            fmt::format("step_dpo_native: input_ids/targets have {} columns; engine expects T = {}",
                        layout.input_cols,
                        T));
    }
    if (layout.input_rows != 0 && layout.input_rows < static_cast<long>(host_rows) * static_cast<long>(B)) {
        throw std::invalid_argument(
            fmt::format("step_dpo_native: input_ids/targets have {} rows; engine consumes {} host rows x B = {}",
                        layout.input_rows,
                        host_rows,
                        static_cast<long>(host_rows) * static_cast<long>(B)));
    }
    mDpoShardedRows = (layout.token_rows > 1);
    for (int i = 0; i < (int)mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        if (!ctx.Model) {
            throw std::runtime_error(fmt::format("step_dpo_native: ctx[{}].Model is null", i));
        }
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
        const int src_row = host_batch_row_for_local_rank(i, ep_size);

        std::memcpy(ib, inputs + src_row * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + src_row * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(src_row) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(
            fmt::format("step_dpo_native: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
    }

    const dsl::DpoNativeLossConfig loss_config{
        .loss_scale = loss_scale,
        .beta = beta,
        .length_norm = (length_norm != 0),
    };

    run_work([micro_idx = mTrainMicroStep,
              micro_batches = mGradAccumulation,
              ref_logprobs,
              loss_mask,
              sample_starts,
              sample_ends,
              sample_count,
              pair_chosen,
              pair_rejected,
              pair_count,
              loss_config,
              layout,
              B = this->B,
              T = this->T](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("step_dpo_native: model is not a DslModel");
        }

        Tensor inputs_tensor = ctx.Model->get_input_buffer();
        Tensor position_ids_tensor = ctx.Model->get_position_ids_buffer();
        Tensor targets_tensor = ctx.Model->get_target_buffer();

        const int gpu_rank = ctx.Communicator->local_rank();
        const int gpu_ep_size = ctx.Communicator->ep_size();
        const int src_row = host_batch_row_for_local_rank(gpu_rank, gpu_ep_size);
        const std::ptrdiff_t token_offset = static_cast<std::ptrdiff_t>(src_row) * B * T;

        const float* ref_row = ref_logprobs;
        const std::uint8_t* loss_mask_row = loss_mask;
        if (layout.token_rows > 1) {
            ref_row += token_offset;
            loss_mask_row += token_offset;
        }
        const std::int32_t* starts_row = sample_starts;
        const std::int32_t* ends_row = sample_ends;
        const std::int32_t* pair_chosen_row = pair_chosen;
        const std::int32_t* pair_rejected_row = pair_rejected;
        int row_sample_count = sample_count;
        int row_pair_count = pair_count;
        if (layout.sample_rows > 1) {
            starts_row += static_cast<std::ptrdiff_t>(src_row) * sample_count;
            ends_row += static_cast<std::ptrdiff_t>(src_row) * sample_count;
            int n = 0;
            while (n < sample_count && starts_row[n] >= 0 && ends_row[n] >= 0) {
                ++n;
            }
            row_sample_count = n;
            // Pair rows share the sample-row stride; entries are padded with -1.
            pair_chosen_row += static_cast<std::ptrdiff_t>(src_row) * pair_count;
            pair_rejected_row += static_cast<std::ptrdiff_t>(src_row) * pair_count;
            int p = 0;
            while (p < pair_count && pair_chosen_row[p] >= 0 && pair_rejected_row[p] >= 0) {
                ++p;
            }
            row_pair_count = p;
        }

        dsl_model->step_dpo_native(inputs_tensor,
                                   position_ids_tensor,
                                   targets_tensor,
                                   ref_row,
                                   loss_mask_row,
                                   starts_row,
                                   ends_row,
                                   row_sample_count,
                                   pair_chosen_row,
                                   pair_rejected_row,
                                   row_pair_count,
                                   micro_batches,
                                   micro_idx,
                                   *ctx.Communicator,
                                   loss_config);
    });

    ++mTrainMicroStep;
}

std::unordered_map<std::string, float> MultiGPUPyTrainer::get_dpo_native_metrics() {
    std::vector<dsl::DpoNativeMetrics> per_rank(mContexts.size());
    run_work([&per_rank](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("get_dpo_native_metrics: model is not a DslModel");
        }
        per_rank.at(static_cast<std::size_t>(ctx.Communicator->local_rank())) = dsl_model->consume_dpo_native_metrics();
    });

    // Sharded layout: each data-parallel rank saw a distinct slice of pairs, so
    // combine means weighted by each rank's pair count. Replicated layout: every
    // rank is identical, take rank 0.
    const int ep_size = std::max(1, mOptions.EPSize);
    dsl::DpoNativeMetrics agg;
    if (mDpoShardedRows) {
        float total_pairs = 0.0f;
        for (int r = 0; r < (int)per_rank.size(); r += ep_size) {
            const auto& m = per_rank[static_cast<std::size_t>(r)];
            const float w = std::max(m.pair_count, 0.0f);
            agg.loss += m.loss * w;
            agg.accuracy += m.accuracy * w;
            agg.margin += m.margin * w;
            total_pairs += m.pair_count;
        }
        const float denom = std::max(total_pairs, 1.0f);
        agg.loss /= denom;
        agg.accuracy /= denom;
        agg.margin /= denom;
        agg.pair_count = total_pairs;
    } else {
        agg = per_rank.at(0);
    }

    return {
        {"dpo_loss", agg.loss},
        {"dpo_accuracy", agg.accuracy},
        {"dpo_margin", agg.margin},
        {"dpo_pairs", agg.pair_count},
    };
}
std::vector<float> MultiGPUPyTrainer::compute_ref_logprobs_dpo(const std::int32_t* inputs,
                                                               const std::int32_t* targets,
                                                               const std::int32_t* position_ids,
                                                               int input_rows) {
    const int ep_size = std::max(1, mOptions.EPSize);
    const int engine_host_rows = std::max(1, (int)mContexts.size() / ep_size);
    if (input_rows % B != 0) {
        throw std::invalid_argument(
            fmt::format("compute_ref_logprobs_dpo: input_rows {} not a multiple of B {}", input_rows, B));
    }
    const int host_rows = input_rows / B;
    if (host_rows != engine_host_rows) {
        throw std::invalid_argument(fmt::format("compute_ref_logprobs_dpo: inputs have {} host rows; engine consumes "
                                                "{} (one B-block per data-parallel rank)",
                                                host_rows,
                                                engine_host_rows));
    }
    const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

    for (int i = 0; i < (int)mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        if (!ctx.Model) {
            throw std::runtime_error(fmt::format("compute_ref_logprobs_dpo: ctx[{}].Model is null", i));
        }
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const int src_row = host_batch_row_for_local_rank(i, ep_size);

        std::memcpy(ib, inputs + static_cast<std::ptrdiff_t>(src_row) * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + static_cast<std::ptrdiff_t>(src_row) * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(src_row) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    std::vector<float> result(static_cast<std::size_t>(host_rows) * bt, 0.0f);
    run_work([&result, ep_size, bt, B = this->B, T = this->T](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("compute_ref_logprobs_dpo: model is not a DslModel");
        }
        const int gpu_rank = ctx.Communicator->local_rank();
        const int gpu_ep_size = ctx.Communicator->ep_size();
        // Within an EP group every rank holds the same host row; only the group's
        // first rank writes it back so distinct rows don't race or duplicate.
        if (gpu_ep_size > 1 && (gpu_rank % gpu_ep_size) != 0) {
            (void)dsl_model->compute_ref_logprobs(ctx.Model->get_input_buffer(),
                                                  ctx.Model->get_position_ids_buffer(),
                                                  ctx.Model->get_target_buffer(),
                                                  *ctx.Communicator);
            return;
        }
        const int src_row = host_batch_row_for_local_rank(gpu_rank, gpu_ep_size);
        auto logprobs = dsl_model->compute_ref_logprobs(ctx.Model->get_input_buffer(),
                                                        ctx.Model->get_position_ids_buffer(),
                                                        ctx.Model->get_target_buffer(),
                                                        *ctx.Communicator);
        std::copy(logprobs.begin(), logprobs.end(), result.begin() + static_cast<std::ptrdiff_t>(src_row) * bt);
    });
    return result;
}

std::vector<float> MultiGPUPyTrainer::forward_for_grpo(const std::int32_t* inputs,
                                                       const std::int32_t* targets,
                                                       const std::int32_t* position_ids,
                                                       const float* temperatures) {
    // Distribute inputs, targets, and position_ids to each GPU's CPU-side buffers.
    const int ep_size = std::max(1, mOptions.EPSize);
    for (int i = 0; i < (int)mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        if (!ctx.Model) {
            throw std::runtime_error(fmt::format("forward_for_grpo: ctx[{}].Model is null", i));
        }
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
        const int src_row = host_batch_row_for_local_rank(i, ep_size);

        std::memcpy(ib, inputs + src_row * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + src_row * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(src_row) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(
            fmt::format("forward_for_grpo: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
    }

    std::vector<float> result;
    run_work([&result,
              micro_idx = mTrainMicroStep,
              micro_batches = mGradAccumulation,
              temperatures,
              B = this->B,
              T = this->T](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("forward_for_grpo: model is not a DslModel");
        }

        Tensor inputs_tensor = ctx.Model->get_input_buffer();
        Tensor position_ids_tensor = ctx.Model->get_position_ids_buffer();
        Tensor targets_tensor = ctx.Model->get_target_buffer();

        const int gpu_rank = ctx.Communicator->local_rank();
        const int gpu_ep_size = ctx.Communicator->ep_size();
        const int src_row = host_batch_row_for_local_rank(gpu_rank, gpu_ep_size);
        const float* temps_for_this_gpu = nullptr;
        if (temperatures) {
            temps_for_this_gpu = temperatures + static_cast<std::ptrdiff_t>(src_row) * B * T;
        }

        auto logprobs = dsl_model->forward_for_grpo(inputs_tensor,
                                                    position_ids_tensor,
                                                    targets_tensor,
                                                    micro_batches,
                                                    micro_idx,
                                                    *ctx.Communicator,
                                                    temps_for_this_gpu);
        if (ctx.Communicator->local_rank() == 0) {
            result = std::move(logprobs);
        }
    });
    // Don't increment mTrainMicroStep — that happens in backward_grpo.
    return result;
}

void MultiGPUPyTrainer::backward_grpo(const float* per_token_grads) {
    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(
            fmt::format("backward_grpo: micro_step {} >= grad_accumulation {}", mTrainMicroStep, mGradAccumulation));
    }

    run_work(
        [micro_idx = mTrainMicroStep, micro_batches = mGradAccumulation, per_token_grads, B = this->B, T = this->T](
            sThreadContext& ctx) {
            auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!dsl_model) {
                throw std::runtime_error("backward_grpo: model is not a DslModel");
            }

            Tensor inputs_tensor = ctx.Model->get_input_buffer();
            Tensor targets_tensor = ctx.Model->get_target_buffer();

            const int gpu_rank = ctx.Communicator->local_rank();
            const int gpu_ep_size = ctx.Communicator->ep_size();
            const int src_row = host_batch_row_for_local_rank(gpu_rank, gpu_ep_size);
            const float* grads_for_this_gpu = per_token_grads + static_cast<std::ptrdiff_t>(src_row) * B * T;

            dsl_model->backward_grpo(inputs_tensor,
                                     targets_tensor,
                                     grads_for_this_gpu,
                                     micro_batches,
                                     micro_idx,
                                     *ctx.Communicator);
        });

    ++mTrainMicroStep;
}

int MultiGPUPyTrainer::get_valid_token_count(int gpu_id) {
    int result = 0;
    run_work(
        [&result](sThreadContext& ctx) {
            auto& rs = ctx.Model->get_run_state();
            if (!rs.ValidTokenCount.Data) {
                result = 0;
                return;
            }
            CUDA_CHECK(
                cudaMemcpyAsync(&result, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
        },
        gpu_id);
    return result;
}
