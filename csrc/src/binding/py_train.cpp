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
#include <chrono>
#include <random>

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
#include "runtime/dsl/dsl_run_state.h"
#include "runtime/dsl/dsl_param_store.h"
#include "runtime/dsl/graph_executor.h"
#include "runtime/infer/generation_engine.h"
#include "runtime/infer/continuous_engine.h"
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

uint64_t default_generation_seed() {
    static std::atomic<uint64_t> counter{0};
    const uint64_t c = counter.fetch_add(1, std::memory_order_relaxed);
    const uint64_t t = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::random_device rd;
    const uint64_t r = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    // SplitMix64-style mixing.
    uint64_t x = t ^ r ^ (c + 0x9e3779b97f4a7c15ULL);
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

uint64_t generation_seed_from_env_or_default() {
    const char* raw = std::getenv("SUROGATE_GENERATE_SEED");
    if (!raw || !*raw) {
        return default_generation_seed();
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || (end && *end != '\0')) {
        return default_generation_seed();
    }
    return static_cast<uint64_t>(parsed);
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

struct MultiGPUPyTrainer::GenerationSessionBundle {
    int prompt_count = 0;
    bool use_lora = false;
    std::vector<std::vector<int>> rank_prompt_indices;
    std::vector<int> global_to_rank;
    std::vector<int> global_to_local;
    std::vector<std::unique_ptr<infer::GenerationSession>> per_rank_sessions;
};

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
    std::vector<std::uint64_t> open_session_ids;
    {
        std::lock_guard<std::mutex> lock(mGenerationSessionsMutex);
        open_session_ids.reserve(mGenerationSessions.size());
        for (const auto& it : mGenerationSessions) {
            open_session_ids.push_back(it.first);
        }
    }
    for (std::uint64_t session_id : open_session_ids) {
        try {
            close_generation_session(session_id);
        } catch (...) {
            // Best-effort cleanup in destructor.
        }
    }

    mIsRunning = false;

    // make sure all work has finished
    // Use communicator device index for cudaSetDevice, and don't throw from destructor
    for(auto& ctx : mContexts) {
        if(ctx.Communicator) {
            cudaError_t err = cudaSetDevice(ctx.Communicator->device_index());
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
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            // Python binding provides 2D [B, T] position IDs (one plane per GPU).
            // For mRoPE models the buffer is [3, B, T] — replicate the single plane.
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
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
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            // Python binding provides 2D [B, T] position IDs (one plane per GPU).
            // For mRoPE models the buffer is [3, B, T] — replicate the single plane.
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(bt);
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
            gs.has_stack_checkpoint = false;
            gs.stack_top = nullptr;
            gs.stack_alloc_count = 0;
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

        // Detect packed sequences with document boundaries in any micro-step.
        // When present, fall back to eager execution because CUDA graph replay
        // cannot handle per-step cu_seqlens updates for Flash Attention varlen.
        bool has_doc_boundaries = false;
        if (mOptions.DocMasking) {
            for (int j = 0; j < micro_steps && !has_doc_boundaries; ++j) {
                const auto* pos = reinterpret_cast<const std::int32_t*>(gs.position_ids[j].Data);
                for (int t = 1; t < B * T; ++t) {
                    if (pos[t] - pos[t - 1] != 1) {
                        has_doc_boundaries = true;
                        break;
                    }
                }
            }
        }

        // If graphs are disabled or packed sequences need doc masking, use eager execution.
        // When doc boundaries are present but CUDA graphs are enabled, the GraphExecutor
        // uses split-attention mode: non-attention ops are captured as per-segment CUDA
        // graphs, while FlashAttention runs eagerly with per-step cu_seqlens.
        if (!mOptions.UseCudaGraphs || has_doc_boundaries) {
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
        if (config.type != optimizers::OptimizerType::ADAMW &&
            config.type != optimizers::OptimizerType::ADAMW_8BIT &&
            config.type != optimizers::OptimizerType::NORMUON) {
            throw std::runtime_error("train_step_graphed: only supports AdamW, AdamW 8-bit or NorMuon optimizer");
        }

        // CUDA graph capture path (both AdamW and NorMuon support graph capture)
        enum class FullStepGraphMode { Full, ForwardBackward, ForwardOnly };
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
                if (do_graph_backward && !warmup_skip_bwd) {
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
                dsl_model->update_with_graph_params(*ctx.Communicator, config,
                                                   gs.opt_params.template get<float>(),
                                                   gs.opt_step.template get<int>());
            }
            CUDA_CHECK(cudaStreamEndCapture(rs.MainStream, &graph));
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
    modules::QLoRAConfig qlora_config;
    if (mQLoRAConfig.has_value()) {
        qlora_config = mQLoRAConfig.value();
    }

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

        // Use factory to create LoRA model with proper architecture dispatch
        ctx.Model = modules::ModelFactory::create_lora_from_pretrained_config(
            *mConfig, mod_lora, mOptions, comm, allocator, qlora_config);
    } else {
        // Use factory to create base model with proper architecture dispatch
        ctx.Model = modules::ModelFactory::create_from_pretrained_config(
            *mConfig, mOptions, comm.rank(), comm.world_size(), allocator, qlora_config);
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
            // Python binding provides 2D [B, T] position IDs (one plane per GPU).
            // For mRoPE models the buffer is [3, B, T] — replicate the single plane.
            const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
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

std::vector<float> MultiGPUPyTrainer::forward_for_grpo(
        const std::int32_t* inputs,
        const std::int32_t* targets,
        const std::int32_t* position_ids,
        const float* temperatures) {
    // Distribute inputs, targets, and position_ids to each GPU's CPU-side buffers.
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

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(fmt::format("forward_for_grpo: micro_step {} >= grad_accumulation {}",
                                             mTrainMicroStep, mGradAccumulation));
    }

    std::vector<float> result;
    run_work([&result, micro_idx = mTrainMicroStep, micro_batches = mGradAccumulation,
              temperatures, B = this->B, T = this->T](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("forward_for_grpo: model is not a DslModel");
        }

        Tensor inputs_tensor = ctx.Model->get_input_buffer();
        Tensor position_ids_tensor = ctx.Model->get_position_ids_buffer();
        Tensor targets_tensor = ctx.Model->get_target_buffer();

        const int gpu_rank = ctx.Communicator->local_rank();
        const float* temps_for_this_gpu = nullptr;
        if (temperatures) {
            temps_for_this_gpu = temperatures + static_cast<std::ptrdiff_t>(gpu_rank) * B * T;
        }

        auto logprobs = dsl_model->forward_for_grpo(inputs_tensor, position_ids_tensor, targets_tensor,
                                                      micro_batches, micro_idx,
                                                      *ctx.Communicator, temps_for_this_gpu);
        if (ctx.Communicator->local_rank() == 0) {
            result = std::move(logprobs);
        }
    });
    // Don't increment mTrainMicroStep — that happens in backward_grpo.
    return result;
}

void MultiGPUPyTrainer::backward_grpo(const float* per_token_grads) {
    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(fmt::format("backward_grpo: micro_step {} >= grad_accumulation {}",
                                             mTrainMicroStep, mGradAccumulation));
    }

    run_work([micro_idx = mTrainMicroStep, micro_batches = mGradAccumulation,
              per_token_grads, B = this->B, T = this->T](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("backward_grpo: model is not a DslModel");
        }

        Tensor inputs_tensor = ctx.Model->get_input_buffer();
        Tensor targets_tensor = ctx.Model->get_target_buffer();

        const int gpu_rank = ctx.Communicator->local_rank();
        const float* grads_for_this_gpu = per_token_grads +
                                          static_cast<std::ptrdiff_t>(gpu_rank) * B * T;

        dsl_model->backward_grpo(inputs_tensor, targets_tensor, grads_for_this_gpu,
                                  micro_batches, micro_idx, *ctx.Communicator);
    });

    ++mTrainMicroStep;
}

MultiGPUPyTrainer::NativeGrpoStepMetrics MultiGPUPyTrainer::step_with_native_grpo(
        const std::int32_t* inputs,
        const std::int32_t* targets,
        const float* inference_logprobs,
        const float* advantages,
        const std::uint8_t* loss_mask,
        float kl_tau,
        float adv_tau,
        float ipo_mask_low,
        float ipo_mask_high,
        float loss_scale,
        int rollout_batch_rows,
        const std::int32_t* position_ids,
        const float* temperatures) {
    const int world = static_cast<int>(mContexts.size());
    if (rollout_batch_rows < 0) {
        rollout_batch_rows = B;
    }
    const bool rollout_broadcast = (rollout_batch_rows == B);
    const bool rollout_sharded = (rollout_batch_rows == B * world);
    if (!rollout_broadcast && !rollout_sharded) {
        throw std::runtime_error(
            fmt::format("step_with_native_grpo: invalid rollout rows {} (expected {} broadcast or {} sharded)",
                        rollout_batch_rows, B, B * world));
    }

    for (int i = 0; i < (int)mContexts.size(); ++i) {
        auto& ctx = mContexts.at(i);
        if (!ctx.Model) {
            throw std::runtime_error(fmt::format("step_with_native_grpo: ctx[{}].Model is null", i));
        }
        auto* ib = ctx.Model->get_input_buffer().get<std::int32_t>();
        auto* tb = ctx.Model->get_target_buffer().get<std::int32_t>();
        Tensor pos_buf = ctx.Model->get_position_ids_buffer();
        auto* pb = pos_buf.get<std::int32_t>();
        const int pos_planes = (pos_buf.Rank == 3) ? static_cast<int>(pos_buf.Sizes[0]) : 1;
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

        std::memcpy(ib, inputs + i * B * T, B * T * sizeof(std::int32_t));
        std::memcpy(tb, targets + i * B * T, B * T * sizeof(std::int32_t));
        if (position_ids) {
            const auto* src = position_ids + static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(bt);
            for (int p = 0; p < pos_planes; ++p) {
                std::memcpy(pb + p * bt, src, bt * sizeof(std::int32_t));
            }
        } else {
            fill_sequential_position_ids(pb, pos_planes, B, T);
        }
    }

    if (mTrainMicroStep >= mGradAccumulation) {
        throw std::runtime_error(fmt::format("step_with_native_grpo: micro_step {} >= grad_accumulation {}",
                                             mTrainMicroStep, mGradAccumulation));
    }

    std::vector<NativeGrpoStepMetrics> per_rank_metrics(static_cast<std::size_t>(world));
    run_work([micro_idx = mTrainMicroStep, micro_batches = mGradAccumulation,
              inference_logprobs, advantages, loss_mask,
              rollout_broadcast,
              kl_tau, adv_tau, ipo_mask_low, ipo_mask_high, loss_scale,
              temperatures, B = this->B, T = this->T,
              &per_rank_metrics](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("step_with_native_grpo: model is not a DslModel");
        }

        Tensor inputs_tensor = ctx.Model->get_input_buffer();
        Tensor position_ids_tensor = ctx.Model->get_position_ids_buffer();
        Tensor targets_tensor = ctx.Model->get_target_buffer();

        const int gpu_rank = ctx.Communicator->local_rank();
        const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
        const std::ptrdiff_t rollout_off = rollout_broadcast
            ? 0
            : static_cast<std::ptrdiff_t>(gpu_rank) * static_cast<std::ptrdiff_t>(bt);
        const float* inf_lp_for_this_gpu = inference_logprobs + rollout_off;
        const float* adv_for_this_gpu = advantages + rollout_off;
        const std::uint8_t* mask_for_this_gpu = loss_mask + rollout_off;

        const float* temps_for_this_gpu = nullptr;
        if (temperatures) {
            temps_for_this_gpu = temperatures + static_cast<std::ptrdiff_t>(gpu_rank) * B * T;
        }

        auto metrics = dsl_model->step_with_native_grpo(
            inputs_tensor,
            position_ids_tensor,
            targets_tensor,
            inf_lp_for_this_gpu,
            adv_for_this_gpu,
            mask_for_this_gpu,
            kl_tau,
            adv_tau,
            ipo_mask_low,
            ipo_mask_high,
            loss_scale,
            micro_batches,
            micro_idx,
            *ctx.Communicator,
            temps_for_this_gpu);
        auto& out = per_rank_metrics[static_cast<std::size_t>(gpu_rank)];
        out.policy_loss = metrics.policy_loss;
        out.mismatch_kl = metrics.mismatch_kl;
        out.is_masked = metrics.is_masked;
        out.keep_tokens = metrics.keep_tokens;
        out.total_tokens = metrics.total_tokens;
    });

    NativeGrpoStepMetrics result{};
    double policy_num = 0.0;
    double kl_num = 0.0;
    double masked_num = 0.0;
    int total_tokens = 0;
    int keep_tokens = 0;
    for (const auto& m : per_rank_metrics) {
        total_tokens += m.total_tokens;
        keep_tokens += m.keep_tokens;
        policy_num += static_cast<double>(m.policy_loss) * static_cast<double>(m.total_tokens);
        kl_num += static_cast<double>(m.mismatch_kl) * static_cast<double>(m.total_tokens);
        masked_num += static_cast<double>(m.is_masked) * static_cast<double>(m.total_tokens);
    }
    result.total_tokens = total_tokens;
    result.keep_tokens = keep_tokens;
    if (total_tokens > 0) {
        result.policy_loss = static_cast<float>(policy_num / static_cast<double>(total_tokens));
        result.mismatch_kl = static_cast<float>(kl_num / static_cast<double>(total_tokens));
        result.is_masked = static_cast<float>(masked_num / static_cast<double>(total_tokens));
    }

    ++mTrainMicroStep;
    return result;
}

MultiGPUPyTrainer::GenerationResult MultiGPUPyTrainer::generate(
        const std::vector<std::vector<int32_t>>& prompts,
        int num_completions,
        int max_gen_len,
        float temperature,
        int eos_token_id,
        bool use_lora,
        bool use_cuda_graphs,
        int top_k,
        float top_p,
        float min_p,
        int prefill_chunk_size,
        float repetition_penalty) {
    if (num_completions <= 0) {
        throw std::invalid_argument("generate: num_completions must be > 0");
    }

    GenerationResult result;
    if (prompts.empty()) {
        return result;
    }

    struct RankShardOutput {
        std::vector<int> prompt_indices;
        std::vector<infer::Trajectory> trajectories;
    };

    const int local_world = static_cast<int>(mContexts.size());
    const int prompt_count = static_cast<int>(prompts.size());
    const uint64_t base_seed = generation_seed_from_env_or_default();
    std::vector<RankShardOutput> rank_outputs(static_cast<std::size_t>(local_world));

    // Shard prompts across local GPUs (round-robin by prompt index).
    // Each rank generates only its shard, then we stitch outputs back into
    // canonical global order [prompt0 completion0..N-1, prompt1 ...].
    run_work([&](sThreadContext& ctx) {
        const int local_rank = ctx.Communicator->local_rank();
        std::vector<std::vector<int32_t>> local_prompts;
        std::vector<int> local_prompt_indices;
        for (int prompt_idx = local_rank; prompt_idx < prompt_count; prompt_idx += local_world) {
            local_prompts.push_back(prompts[static_cast<std::size_t>(prompt_idx)]);
            local_prompt_indices.push_back(prompt_idx);
        }
        if (local_prompts.empty()) {
            return;
        }

        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("generate: model is not a DslModel");
        }

        auto* graph_exec = dsl_model->graph_executor();
        if (!graph_exec) {
            throw std::runtime_error("generate: model has no GraphExecutor");
        }

        auto& run_state = dsl_model->dsl_run_state();
        auto& params = dsl_model->param_store();
        const auto& model_config = dsl_model->model_config();

        // Build generation config
        infer::GenerationEngineConfig gen_config{};
        gen_config.max_gen_len = max_gen_len;
        gen_config.temperature = temperature;
        gen_config.top_k = top_k;
        gen_config.top_p = top_p;
        gen_config.min_p = min_p;
        gen_config.prefill_chunk_size = std::max(0, prefill_chunk_size);
        gen_config.repetition_penalty = repetition_penalty;
        gen_config.eos_token_id = eos_token_id;
        gen_config.num_completions = num_completions;
        gen_config.use_cuda_graphs = use_cuda_graphs;
        gen_config.seed = base_seed + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(local_rank);

        // Create generation engine using the model's own stack as the arena.
        // The stack is shared with training — generation allocates from it,
        // and restores the checkpoint at the end (memory overlay).
        infer::GenerationEngine engine(run_state, params, model_config, mOptions);

        // Build LoRA forward hook if requested.
        modules::ForwardHook hook_storage;
        const modules::ForwardHook* hook_ptr = nullptr;
        if (use_lora) {
            int max_prompt_len = 0;
            for (const auto& prompt : local_prompts) {
                max_prompt_len = std::max(max_prompt_len, static_cast<int>(prompt.size()));
            }
            const int total_batch = static_cast<int>(local_prompts.size()) * std::max(1, num_completions);
            const int lora_bt_capacity = std::max(total_batch, max_prompt_len);
            hook_ptr = dsl_model->make_forward_hook(
                /*use_lora=*/true,
                /*ensure_B=*/std::max(1, lora_bt_capacity),
                /*ensure_T=*/1,
                *ctx.Communicator,
                hook_storage);
        }

        // Generation may recompile the graph for decode dimensions via
        // execute_decode_step -> compile_graphs(). After generation finishes,
        // force_recompile() ensures the next training operation gets fresh
        // graphs for training dimensions (B_train, T_train).

        // Generate using the model's stack as arena.
        // N=1 uses the non-GRPO API surface, but it now also runs paged KV.
        std::vector<infer::Trajectory> trajectories;
        if (num_completions == 1) {
            trajectories = engine.generate(
                local_prompts, gen_config, run_state.Stack,
                *graph_exec, *ctx.Communicator, hook_ptr);
        } else {
            trajectories = engine.generate_grpo(
                local_prompts, gen_config, run_state.Stack,
                *graph_exec, *ctx.Communicator, hook_ptr);
        }

        rank_outputs.at(static_cast<std::size_t>(local_rank)).prompt_indices = std::move(local_prompt_indices);
        rank_outputs.at(static_cast<std::size_t>(local_rank)).trajectories = std::move(trajectories);
    });

    const std::size_t total_outputs = static_cast<std::size_t>(prompt_count) * static_cast<std::size_t>(num_completions);
    result.tokens.resize(total_outputs);
    result.logprobs.resize(total_outputs);
    result.prompt_lens.resize(total_outputs);
    result.completion_lens.resize(total_outputs);

    for (int rank = 0; rank < local_world; ++rank) {
        auto& shard = rank_outputs[static_cast<std::size_t>(rank)];
        const std::size_t expected = shard.prompt_indices.size() * static_cast<std::size_t>(num_completions);
        if (shard.trajectories.size() != expected) {
            throw std::runtime_error(
                fmt::format("generate: shard output size mismatch on rank {} (expected {}, got {})",
                            rank, expected, shard.trajectories.size()));
        }
        for (std::size_t local_prompt_pos = 0; local_prompt_pos < shard.prompt_indices.size(); ++local_prompt_pos) {
            const int global_prompt_idx = shard.prompt_indices[local_prompt_pos];
            for (int completion_idx = 0; completion_idx < num_completions; ++completion_idx) {
                const std::size_t local_seq_idx =
                    local_prompt_pos * static_cast<std::size_t>(num_completions)
                    + static_cast<std::size_t>(completion_idx);
                const std::size_t global_seq_idx =
                    static_cast<std::size_t>(global_prompt_idx) * static_cast<std::size_t>(num_completions)
                    + static_cast<std::size_t>(completion_idx);
                auto& traj = shard.trajectories[local_seq_idx];
                result.tokens[global_seq_idx] = std::move(traj.tokens);
                result.logprobs[global_seq_idx] = std::move(traj.logprobs);
                result.prompt_lens[global_seq_idx] = traj.prompt_len;
                result.completion_lens[global_seq_idx] = traj.completion_len;
            }
        }
    }

    return result;
}

std::uint64_t MultiGPUPyTrainer::open_generation_session(
        const std::vector<std::vector<int32_t>>& prompts,
        int max_gen_len,
        float temperature,
        int eos_token_id,
        bool use_lora,
        bool use_cuda_graphs,
        int top_k,
        float top_p,
        float min_p,
        int prefill_chunk_size,
        float repetition_penalty) {
    if (prompts.empty()) {
        throw std::invalid_argument("open_generation_session: prompts must be non-empty");
    }
    if (max_gen_len <= 0) {
        throw std::invalid_argument("open_generation_session: max_gen_len must be > 0");
    }

    const int local_world = static_cast<int>(mContexts.size());
    const int prompt_count = static_cast<int>(prompts.size());
    const uint64_t base_seed = generation_seed_from_env_or_default();

    auto bundle = std::make_unique<GenerationSessionBundle>();
    bundle->prompt_count = prompt_count;
    bundle->use_lora = use_lora;
    bundle->rank_prompt_indices.resize(static_cast<std::size_t>(local_world));
    bundle->global_to_rank.assign(static_cast<std::size_t>(prompt_count), -1);
    bundle->global_to_local.assign(static_cast<std::size_t>(prompt_count), -1);
    bundle->per_rank_sessions.resize(static_cast<std::size_t>(local_world));

    for (int rank = 0; rank < local_world; ++rank) {
        auto& shard = bundle->rank_prompt_indices[static_cast<std::size_t>(rank)];
        for (int prompt_idx = rank; prompt_idx < prompt_count; prompt_idx += local_world) {
            const int local_pos = static_cast<int>(shard.size());
            shard.push_back(prompt_idx);
            bundle->global_to_rank[static_cast<std::size_t>(prompt_idx)] = rank;
            bundle->global_to_local[static_cast<std::size_t>(prompt_idx)] = local_pos;
        }
    }

    std::uint64_t session_id = 0;
    {
        std::lock_guard<std::mutex> lock(mGenerationSessionsMutex);
        session_id = mNextGenerationSessionId++;
        if (session_id == 0) {
            session_id = mNextGenerationSessionId++;
        }
    }

    try {
        run_work([&, session_id](sThreadContext& ctx) {
            const int local_rank = ctx.Communicator->local_rank();
            const auto& shard_indices =
                bundle->rank_prompt_indices[static_cast<std::size_t>(local_rank)];
            if (shard_indices.empty()) {
                return;
            }

            std::vector<std::vector<int32_t>> local_prompts;
            local_prompts.reserve(shard_indices.size());
            int max_prompt_len = 0;
            for (int prompt_idx : shard_indices) {
                const auto& prompt = prompts[static_cast<std::size_t>(prompt_idx)];
                max_prompt_len = std::max(max_prompt_len, static_cast<int>(prompt.size()));
                local_prompts.push_back(prompt);
            }

            auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
            if (!dsl_model) {
                throw std::runtime_error("open_generation_session: model is not a DslModel");
            }
            auto* graph_exec = dsl_model->graph_executor();
            if (!graph_exec) {
                throw std::runtime_error("open_generation_session: model has no GraphExecutor");
            }
            auto& run_state = dsl_model->dsl_run_state();
            auto& params = dsl_model->param_store();
            const auto& model_config = dsl_model->model_config();

            infer::GenerationEngineConfig gen_config{};
            gen_config.max_gen_len = max_gen_len;
            gen_config.temperature = temperature;
            gen_config.top_k = top_k;
            gen_config.top_p = top_p;
            gen_config.min_p = min_p;
            gen_config.prefill_chunk_size = std::max(0, prefill_chunk_size);
            gen_config.repetition_penalty = repetition_penalty;
            gen_config.eos_token_id = eos_token_id;
            gen_config.num_completions = 1;
            gen_config.use_cuda_graphs = use_cuda_graphs;
            gen_config.seed = base_seed + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(local_rank);

            modules::ForwardHook hook_storage;
            const modules::ForwardHook* hook_ptr = nullptr;
            if (use_lora) {
                const int lora_bt_capacity = std::max(
                    static_cast<int>(local_prompts.size()), max_prompt_len);
                hook_ptr = dsl_model->make_forward_hook(
                    /*use_lora=*/true,
                    /*ensure_B=*/std::max(1, lora_bt_capacity),
                    /*ensure_T=*/1,
                    *ctx.Communicator,
                    hook_storage);
            }

            auto session = std::make_unique<infer::GenerationSession>(
                run_state, params, model_config, mOptions);
            session->init(
                local_prompts, gen_config, run_state.Stack, *graph_exec, *ctx.Communicator, hook_ptr);
            bundle->per_rank_sessions[static_cast<std::size_t>(local_rank)] = std::move(session);
        });
    } catch (...) {
        try {
            run_work([&](sThreadContext& ctx) {
                const int local_rank = ctx.Communicator->local_rank();
                auto& session = bundle->per_rank_sessions[static_cast<std::size_t>(local_rank)];
                if (!session) {
                    return;
                }
                auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
                if (!dsl_model) {
                    session.reset();
                    return;
                }
                auto* graph_exec = dsl_model->graph_executor();
                if (!graph_exec) {
                    session.reset();
                    return;
                }
                session->close(*graph_exec);
                session.reset();
            });
        } catch (...) {
            // Ignore cleanup errors while propagating original exception.
        }
        throw;
    }

    {
        std::lock_guard<std::mutex> lock(mGenerationSessionsMutex);
        mGenerationSessions.emplace(session_id, std::move(bundle));
    }
    return session_id;
}

MultiGPUPyTrainer::GenerationSessionStepResult MultiGPUPyTrainer::step_generation_session(
        std::uint64_t session_id,
        int step_tokens,
        const std::vector<int>& force_finished_rows) {
    if (step_tokens <= 0) {
        throw std::invalid_argument("step_generation_session: step_tokens must be > 0");
    }

    GenerationSessionBundle* bundle = nullptr;
    {
        std::lock_guard<std::mutex> lock(mGenerationSessionsMutex);
        auto it = mGenerationSessions.find(session_id);
        if (it == mGenerationSessions.end()) {
            throw std::runtime_error(
                fmt::format("step_generation_session: unknown session_id {}", session_id));
        }
        bundle = it->second.get();
    }

    const int local_world = static_cast<int>(mContexts.size());
    std::vector<std::vector<int>> force_finished_local(static_cast<std::size_t>(local_world));
    for (int row : force_finished_rows) {
        if (row < 0 || row >= bundle->prompt_count) {
            continue;
        }
        const int rank = bundle->global_to_rank[static_cast<std::size_t>(row)];
        const int local_row = bundle->global_to_local[static_cast<std::size_t>(row)];
        if (rank >= 0 && rank < local_world && local_row >= 0) {
            force_finished_local[static_cast<std::size_t>(rank)].push_back(local_row);
        }
    }

    struct RankStepOutput {
        std::vector<std::vector<int32_t>> tokens;
        std::vector<int32_t> completion_lens;
        std::vector<int32_t> finished;
        bool all_finished = true;
    };
    std::vector<RankStepOutput> rank_outputs(static_cast<std::size_t>(local_world));

    run_work([&](sThreadContext& ctx) {
        const int local_rank = ctx.Communicator->local_rank();
        auto& session = bundle->per_rank_sessions[static_cast<std::size_t>(local_rank)];
        if (!session) {
            rank_outputs[static_cast<std::size_t>(local_rank)].all_finished = true;
            return;
        }

        const auto& rows = force_finished_local[static_cast<std::size_t>(local_rank)];
        if (!rows.empty()) {
            session->mark_finished(rows);
        }

        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("step_generation_session: model is not a DslModel");
        }
        auto* graph_exec = dsl_model->graph_executor();
        if (!graph_exec) {
            throw std::runtime_error("step_generation_session: model has no GraphExecutor");
        }

        modules::ForwardHook hook_storage;
        const modules::ForwardHook* hook_ptr = nullptr;
        if (bundle->use_lora) {
            hook_ptr = dsl_model->make_forward_hook(
                /*use_lora=*/true,
                /*ensure_B=*/std::max(1, session->batch_size()),
                /*ensure_T=*/1,
                *ctx.Communicator,
                hook_storage);
        }

        auto step_out = session->step(step_tokens, *graph_exec, *ctx.Communicator, hook_ptr);
        auto& out = rank_outputs[static_cast<std::size_t>(local_rank)];
        out.tokens = std::move(step_out.tokens);
        out.completion_lens = std::move(step_out.completion_lens);
        out.finished = std::move(step_out.finished);
        out.all_finished = step_out.all_finished;
    });

    GenerationSessionStepResult result;
    result.tokens.resize(static_cast<std::size_t>(bundle->prompt_count));
    result.completion_lens.assign(static_cast<std::size_t>(bundle->prompt_count), 0);
    result.finished.assign(static_cast<std::size_t>(bundle->prompt_count), 1);
    result.all_finished = true;

    for (int rank = 0; rank < local_world; ++rank) {
        const auto& shard_indices = bundle->rank_prompt_indices[static_cast<std::size_t>(rank)];
        if (shard_indices.empty()) {
            continue;
        }
        auto& out = rank_outputs[static_cast<std::size_t>(rank)];
        if (out.tokens.size() != shard_indices.size()
            || out.completion_lens.size() != shard_indices.size()
            || out.finished.size() != shard_indices.size()) {
            throw std::runtime_error(
                fmt::format("step_generation_session: rank {} output size mismatch", rank));
        }
        for (std::size_t local_pos = 0; local_pos < shard_indices.size(); ++local_pos) {
            const int global_prompt_idx = shard_indices[local_pos];
            result.tokens[static_cast<std::size_t>(global_prompt_idx)] =
                std::move(out.tokens[local_pos]);
            result.completion_lens[static_cast<std::size_t>(global_prompt_idx)] =
                static_cast<int>(out.completion_lens[local_pos]);
            result.finished[static_cast<std::size_t>(global_prompt_idx)] =
                out.finished[local_pos] != 0 ? 1 : 0;
            if (result.finished[static_cast<std::size_t>(global_prompt_idx)] == 0) {
                result.all_finished = false;
            }
        }
    }

    return result;
}

void MultiGPUPyTrainer::close_generation_session(std::uint64_t session_id) {
    std::unique_ptr<GenerationSessionBundle> bundle;
    {
        std::lock_guard<std::mutex> lock(mGenerationSessionsMutex);
        auto it = mGenerationSessions.find(session_id);
        if (it == mGenerationSessions.end()) {
            return;
        }
        bundle = std::move(it->second);
        mGenerationSessions.erase(it);
    }
    if (!bundle) {
        return;
    }

    GenerationSessionBundle* bundle_ptr = bundle.get();
    run_work([&](sThreadContext& ctx) {
        const int local_rank = ctx.Communicator->local_rank();
        auto& session = bundle_ptr->per_rank_sessions[static_cast<std::size_t>(local_rank)];
        if (!session) {
            return;
        }
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            session.reset();
            return;
        }
        auto* graph_exec = dsl_model->graph_executor();
        if (!graph_exec) {
            session.reset();
            return;
        }
        session->close(*graph_exec);
        session.reset();
    });
}

// ============================================================================
// Continuous generation engine (iteration-level continuous batching)
// ============================================================================

std::uint64_t MultiGPUPyTrainer::create_continuous_engine(
        int max_num_seqs, int max_seq_len,
        float gpu_memory_utilization, bool use_cuda_graphs,
        int min_activation_mb) {
    if (max_num_seqs <= 0) {
        throw std::invalid_argument("create_continuous_engine: max_num_seqs must be > 0");
    }

    auto engine = std::make_unique<infer::ContinuousGenerationEngine>();
    std::uint64_t engine_id = 0;

    // For now, use rank 0 only (single-GPU continuous engine).
    run_work([&](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("create_continuous_engine: model is not a DslModel");
        }
        auto& run_state = dsl_model->dsl_run_state();
        auto& params = dsl_model->param_store();
        const auto& model_config = dsl_model->model_config();

        // Compute page budget based on GPU memory utilization.
        const std::size_t free_arena = run_state.Stack.unused_capacity();
        const std::size_t usable = static_cast<std::size_t>(
            static_cast<double>(free_arena) * gpu_memory_utilization);

        const int Hkv = model_config.NumKeyValHeads;
        const int Hs = model_config.head_size();
        const int num_layers = model_config.NumLayers;
        constexpr int kPageBlockSize = 256;
        const int sm = run_state.DeviceProp.major * 10 + run_state.DeviceProp.minor;
        const bool fp8_kv_cache = env_enabled("SUROGATE_ENABLE_FP8_KV_CACHE") && sm >= 89;
        const int elem_bytes = fp8_kv_cache ? 1 : 2;
        const std::size_t page_bytes_per_pool =
            static_cast<std::size_t>(num_layers) * kPageBlockSize * Hkv * Hs * elem_bytes;
        const std::size_t fp8_scale_bytes =
            fp8_kv_cache
                ? static_cast<std::size_t>(2) * num_layers * kPageBlockSize * Hkv * sizeof(float)
                : 0;  // k_scales + v_scales
        const std::size_t page_bytes_total = 2 * page_bytes_per_pool + fp8_scale_bytes;

        // Reserve space for decode buffers (~max_num_seqs * V * 4 for logits, etc.)
        // AND activation scratch (FFN intermediates, delta-rule scratch, etc.).
        const std::size_t decode_buffer_estimate =
            static_cast<std::size_t>(max_num_seqs)
            * static_cast<std::size_t>(model_config.VocabSize) * sizeof(float)
            + static_cast<std::size_t>(max_num_seqs) * 256 * 16;  // misc buffers
        const std::size_t activation_reserve =
            static_cast<std::size_t>(std::max(min_activation_mb, 256)) * 1024 * 1024;
        const std::size_t total_reserve = decode_buffer_estimate + activation_reserve;

        const std::size_t kv_budget =
            (usable > total_reserve) ? (usable - total_reserve) : 0;
        int total_pages = static_cast<int>(kv_budget / page_bytes_total);
        total_pages = std::max(total_pages, max_num_seqs);  // at least 1 page per seq

        engine->init(
            max_num_seqs, max_seq_len, total_pages,
            use_cuda_graphs,
            run_state, params, model_config, mOptions, run_state.Stack);

        // Pre-warm prefill graphs for common prompt lengths to eliminate
        // compilation latency on the first real request.
        auto* ge = dsl_model->graph_executor();
        if (ge) {
            engine->warm_prefill_graphs(*ge, *ctx.Communicator);
        }
    }, /*idx=*/0);

    {
        std::lock_guard<std::mutex> lock(mContinuousEnginesMutex);
        engine_id = mNextContinuousEngineId++;
        mContinuousEngines[engine_id] = std::move(engine);
    }
    return engine_id;
}

int MultiGPUPyTrainer::engine_add_sequence(
        std::uint64_t engine_id,
        const std::vector<int32_t>& prompt_ids,
        int max_gen_len, float temperature, int32_t eos_token_id,
        int top_k, float top_p, float min_p,
        int prefill_chunk_size) {
    infer::ContinuousGenerationEngine* eng = nullptr;
    {
        std::lock_guard<std::mutex> lock(mContinuousEnginesMutex);
        auto it = mContinuousEngines.find(engine_id);
        if (it == mContinuousEngines.end()) {
            throw std::runtime_error("engine_add_sequence: invalid engine_id");
        }
        eng = it->second.get();
    }

    int slot_id = -1;
    run_work([&](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("engine_add_sequence: model is not a DslModel");
        }
        auto* graph_exec = dsl_model->graph_executor();
        if (!graph_exec) {
            throw std::runtime_error("engine_add_sequence: model has no GraphExecutor");
        }
        slot_id = eng->add_sequence(
            prompt_ids, max_gen_len, temperature, eos_token_id,
            top_k, top_p, min_p, prefill_chunk_size,
            *graph_exec, *ctx.Communicator);
    }, /*idx=*/0);

    return slot_id;
}

std::vector<int> MultiGPUPyTrainer::engine_add_sequences_batch(
        std::uint64_t engine_id,
        const std::vector<std::vector<int32_t>>& prompts,
        int max_gen_len, float temperature, int32_t eos_token_id,
        int top_k, float top_p, float min_p,
        int prefill_chunk_size) {
    infer::ContinuousGenerationEngine* eng = nullptr;
    {
        std::lock_guard<std::mutex> lock(mContinuousEnginesMutex);
        auto it = mContinuousEngines.find(engine_id);
        if (it == mContinuousEngines.end()) {
            throw std::runtime_error("engine_add_sequences_batch: invalid engine_id");
        }
        eng = it->second.get();
    }

    std::vector<int> slot_ids;
    run_work([&](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("engine_add_sequences_batch: model is not a DslModel");
        }
        auto* graph_exec = dsl_model->graph_executor();
        if (!graph_exec) {
            throw std::runtime_error("engine_add_sequences_batch: model has no GraphExecutor");
        }
        infer::ContinuousGenerationEngine::BatchAddConfig cfg;
        cfg.max_gen_len = max_gen_len;
        cfg.temperature = temperature;
        cfg.eos_token_id = eos_token_id;
        cfg.top_k = top_k;
        cfg.top_p = top_p;
        cfg.min_p = min_p;
        cfg.prefill_chunk_size = prefill_chunk_size;
        slot_ids = eng->add_sequences_batch(prompts, cfg, *graph_exec, *ctx.Communicator);
    }, /*idx=*/0);

    return slot_ids;
}

MultiGPUPyTrainer::ContinuousStepResult MultiGPUPyTrainer::engine_step(
        std::uint64_t engine_id, int step_tokens) {
    infer::ContinuousGenerationEngine* eng = nullptr;
    {
        std::lock_guard<std::mutex> lock(mContinuousEnginesMutex);
        auto it = mContinuousEngines.find(engine_id);
        if (it == mContinuousEngines.end()) {
            throw std::runtime_error("engine_step: invalid engine_id");
        }
        eng = it->second.get();
    }

    ContinuousStepResult result;
    run_work([&](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) {
            throw std::runtime_error("engine_step: model is not a DslModel");
        }
        auto* graph_exec = dsl_model->graph_executor();
        if (!graph_exec) {
            throw std::runtime_error("engine_step: model has no GraphExecutor");
        }
        auto step = eng->step(step_tokens, *graph_exec, *ctx.Communicator);
        result.slot_ids = std::move(step.slot_ids);
        result.tokens = std::move(step.tokens);
        result.finished = std::move(step.finished);
        result.completion_lens = std::move(step.completion_lens);
    }, /*idx=*/0);

    return result;
}

MultiGPUPyTrainer::FlatStepResult MultiGPUPyTrainer::engine_flat_step(
        std::uint64_t engine_id,
        const std::vector<std::vector<int32_t>>& new_prompts,
        int max_gen_len, float temperature, int32_t eos_token_id,
        int top_k, float top_p, float min_p, int prefill_chunk_size) {
    infer::ContinuousGenerationEngine* eng = nullptr;
    {
        std::lock_guard<std::mutex> lock(mContinuousEnginesMutex);
        auto it = mContinuousEngines.find(engine_id);
        if (it == mContinuousEngines.end()) {
            throw std::runtime_error("engine_flat_step: invalid engine_id");
        }
        eng = it->second.get();
    }

    FlatStepResult result;
    run_work([&](sThreadContext& ctx) {
        auto* dsl_model = dynamic_cast<dsl::DslModel*>(ctx.Model.get());
        if (!dsl_model) throw std::runtime_error("engine_flat_step: not a DslModel");
        auto* ge = dsl_model->graph_executor();
        if (!ge) throw std::runtime_error("engine_flat_step: no GraphExecutor");

        infer::ContinuousGenerationEngine::FlatStepConfig cfg;
        cfg.max_gen_len = max_gen_len;
        cfg.temperature = temperature;
        cfg.eos_token_id = eos_token_id;
        cfg.top_k = top_k;
        cfg.top_p = top_p;
        cfg.min_p = min_p;
        cfg.prefill_chunk_size = prefill_chunk_size;

        auto r = eng->flat_step(new_prompts, cfg, *ge, *ctx.Communicator);
        result.new_slot_ids = std::move(r.new_slot_ids);
        result.active_slot_ids = std::move(r.active_slot_ids);
        result.sampled_tokens = std::move(r.sampled_tokens);
        result.finished = std::move(r.finished);
        result.completion_lens = std::move(r.completion_lens);
    }, 0);

    return result;
}

void MultiGPUPyTrainer::engine_release_slot(
        std::uint64_t engine_id, int slot_id) {
    infer::ContinuousGenerationEngine* eng = nullptr;
    {
        std::lock_guard<std::mutex> lock(mContinuousEnginesMutex);
        auto it = mContinuousEngines.find(engine_id);
        if (it == mContinuousEngines.end()) return;
        eng = it->second.get();
    }
    eng->release_slot(slot_id);
}

void MultiGPUPyTrainer::engine_destroy(std::uint64_t engine_id) {
    std::unique_ptr<infer::ContinuousGenerationEngine> eng;
    {
        std::lock_guard<std::mutex> lock(mContinuousEnginesMutex);
        auto it = mContinuousEngines.find(engine_id);
        if (it == mContinuousEngines.end()) return;
        eng = std::move(it->second);
        mContinuousEngines.erase(it);
    }
    if (eng) {
        eng->destroy();
    }
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
