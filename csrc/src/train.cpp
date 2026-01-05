// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#include "utilities/safetensors.h"
#include "utilities/gpu_info.h"
#include "utilities/sol.h"
#include "utilities/comm.h"

#include "kernels/kernels.h"

#include "training/logging.h"
#include "training/dataloader.h"
#include "training/schedule.h"
#include "training/checkpoint.h"

#include "training/runtime_options.h"
#include "config/pretrained_config.h"
#include "config/lora_adapter_config.h"
#include "recipes/recipe_factory.h"
#include "modules/modules.h"
#include "modules/lora/lora_model.h"
#include "modules/qlora/qlora_config.h"
#include "modules/qlora/fp8_weights.h"
#include <chrono>
#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <CLI/CLI.hpp>
#include <fmt/chrono.h>

/**
 * @brief Run validation on the provided eval dataloader and log results.
 *
 * @param test_loader Evaluation dataloader; its state is reset to the beginning for the run.
 * @param model Model instance used for forward/validate.
 * @param logger Training run logger used to emit eval events/metrics.
 * @param epoch Fractional epoch value reported to the logger.
 * @param step Global optimizer step used for logging/NVTX range naming.
 * @param comm Communicator used for distributed validation behavior.
 * @param max_steps Maximum number of eval batches to process (0 means no batches; caller controls semantics).
 * @param inputs Pre-allocated input tensor buffer (device) filled by the dataloader.
 * @param targets Pre-allocated target tensor buffer (device) filled by the dataloader.
 * @return Mean validation loss over processed batches (0 if no batches were available).
 */
float run_evaluation(DataLoader& test_loader, IModel& model, TrainingRunLogger& logger, float epoch, int step,
                     NCCLCommunicator& comm, int max_steps, Tensor& inputs, Tensor& targets, Tensor& position_ids);

/**
 * @brief CLI11 lexical cast hook for parsing ETensorDType options.
 *
 * @param input String value provided on the command line.
 * @param output Parsed tensor dtype.
 * @return Always true (parsing errors are handled by dtype_from_str).
 */
bool lexical_cast(const std::string& input, ETensorDType& output) {
    output = dtype_from_str(input);
    return true;
}

/**
 * @brief Replace all occurrences of a substring within a string.
 *
 * @param haystack Input string to search/modify (copied by value).
 * @param needle Substring to replace.
 * @param replacement Replacement substring.
 * @return Modified string with all replacements applied.
 */
std::string replace(std::string haystack, const std::string& needle, const std::string& replacement) {
    size_t pos = haystack.find(needle);
    while (pos != std::string::npos) {
        haystack.replace(pos, needle.size(), replacement);
        pos = haystack.find(needle, pos + needle.size());
    }
    return std::move(haystack);
}

namespace CLI::detail {
    template<>
    constexpr const char* type_name<ETensorDType>() {
        return "DTYPE";
    }
}

namespace {
/**
 * @brief Read a single scalar from a device tensor and convert it to float (debug helper).
 *
 * @param t Device tensor to read from (must be FP32 or BF16 for a non-NaN result).
 * @param index Element index (0-based).
 * @return Scalar value as float, or NaN if out-of-range/unsupported dtype/null data.
 */
float read_device_scalar(const Tensor& t, long index) {
    if (!t.Data || index < 0 || index >= t.nelem()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (t.DType == ETensorDType::FP32) {
        float v = 0.0f;
        CUDA_CHECK(cudaMemcpy(&v, reinterpret_cast<const float*>(t.Data) + index, sizeof(v), cudaMemcpyDeviceToHost));
        return v;
    }
    if (t.DType == ETensorDType::BF16) {
        nv_bfloat16 v;
        CUDA_CHECK(cudaMemcpy(&v, reinterpret_cast<const nv_bfloat16*>(t.Data) + index, sizeof(v), cudaMemcpyDeviceToHost));
        return __bfloat162float(v);
    }
    return std::numeric_limits<float>::quiet_NaN();
}

/**
 * @brief Detect whether the current CUDA device supports FP8 tensor-core operations.
 * @return True if the active device has SM89 (Ada) or SM90+ (Hopper+).
 */
bool device_supports_fp8() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
    // FP8 tensor-core support starts with Ada (SM89) and Hopper (SM90+).
    return (prop.major > 8) || (prop.major == 8 && prop.minor >= 9);
}

void validate_zero_copy_support_or_throw() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    int can_map_host = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&can_map_host, cudaDevAttrCanMapHostMemory, device));
    if (!can_map_host) {
        throw std::runtime_error("--use-zero-copy requested but this CUDA device cannot map host memory (cudaDevAttrCanMapHostMemory=0)");
    }

    void* host_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&host_ptr, 4096, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("--use-zero-copy requested but cudaHostAlloc(cudaHostAllocMapped) failed: ") + cudaGetErrorString(err));
    }

    void* dev_ptr = nullptr;
    err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    cudaFreeHost(host_ptr);
    if (err != cudaSuccess || dev_ptr == nullptr) {
        throw std::runtime_error(std::string("--use-zero-copy requested but cudaHostGetDevicePointer failed: ") + cudaGetErrorString(err));
    }
}
} // namespace

/**
 * @brief End-to-end training runner: CLI config parsing, distributed setup, training loop, eval/checkpoint/export.
 *
 * “Parameters” are stored as public fields so CLI11 can bind options directly.
 */
struct TrainingRunner {
    /// Micro-batch size per optimizer micro-step (per rank).
    int B = 4;
    /// Sequence length (tokens) per sample.
    int T = 1024;

    /// Max optimizer steps (-1 derives from epochs and dataset size).
    int MaxSteps = -1;
    /// Number of epochs (ignored if --steps is set).
    int NumEpochs = 1;

    /// Base learning rate.
    float LearningRate = 1e-5f;
    /// Number of warmup steps at start of training.
    int WarmupSteps = 0;
    /// Number of cooldown steps at the end (anneals LR to zero with 1-sqrt schedule).
    int CoolDownSteps = 0;
    /// Final learning-rate fraction applied by the schedule (e.g., 0.1 => end LR is 10% of base).
    float FinalLrFraction = 0.0f;
    /// LR schedule type: "cosine", "linear", or project-specific variants (e.g. "wsd").
    std::string LrScheduleType = "cosine";

    /// Adam beta1.
    float Beta1 = 0.9f;
    /// Adam beta2.
    float Beta2 = 0.999f;
    /// Global gradient clipping threshold (<=0 disables clipping).
    float GradClip = 1.0f;
    /// AdamW weight decay for matrix parameters.
    float WeightDecay = 0.0f;
    /// Adam epsilon.
    float Epsilon = 1e-8f;
    /// Gradient accumulation steps (micro-batches per optimizer step).
    int GradAccSteps = 4;

    /// If true, initialize model randomly instead of loading weights from disk/HF cache.
    bool FromScratch = false;

    /// Model parameter dtype (e.g., BF16/FP32/FP8 depending on support and options).
    ETensorDType ModelDType = ETensorDType::BF16;
    /// HF model directory (local path) or HF repo id (resolved via cache helper).
    std::string ModelRootPath = ".";
    /// Original user-supplied `--model` value (HF repo id or local path). `ModelRootPath` may be resolved to a cache path.
    std::string ModelReference = ".";
    /// Human-readable run name used in paths/logs (supports %n expansion in templates).
    std::string RunName = "surogate";

    /// Training token file path.
    std::string TrainFile = "";
    /// RNG seed for training dataloader sharding/order.
    std::uint64_t TrainLoaderSeed = 0x83b45442ull;
    /// Evaluation token file path.
    std::string EvalFile = "";
    /// RNG seed for evaluation dataloader sharding/order.
    std::uint64_t EvalLoaderSeed = 0x384b4524ull;
    /// Output directory template for final exported model (%n expanded).
    std::string OutDir = "output/%n";
    /// Checkpoint directory template (%n expanded).
    std::string CkptDir = "output/checkpoint-%n";
    /// Log file template (%n expanded).
    std::string LogFile = fmt::format("logs/%n-{:%FT%H_%M}.json", std::chrono::system_clock::now());
    /// Evaluate every N optimizer steps (<=0 disables periodic eval).
    int EvalEvery = 100;
    /// Max eval batches per periodic evaluation.
    int EvalNumSteps = 100;
    /// Final evaluation batches: -1=full eval set, 0=skip, >0=limit to that many batches.
    int FinalEvalNumSteps = -1;
    /// Checkpoint every N optimizer steps (<=0 disables periodic checkpoints).
    int CkptEvery = 100;
    /// Number of checkpoints to keep (-1 keeps all, >0 keeps last N; majors are preserved).
    int CkptToKeep = -1;
    /// Every Nth checkpoint is marked “major” and not cleaned up (-1 disables).
    int MajorCkptEvery = -1;
    /// Log GPU utilization every N steps (0 disables).
    int LogGPUEvery = 0;
    /// If true, export final model weights/config after training.
    bool SaveFinalModel = true;

    /// Optional checkpoint step to continue from; absent means no resume; negative uses latest.
    std::optional<int> ContinueFromCheckpoint;

    /// Number of GPUs to use (0 means “all available” for threads backend; MPI sets this).
    int NGPUs = 0;
    /// Use memcpy for all-gather in threads backend.
    bool MemcpyAllGather = false;
    /// Use memcpy for send/recv in threads backend (all-to-all).
    bool MemcpySendRecv = false;

    // LoRA configuration
    /// Enable LoRA adapter training (base model frozen).
    bool UseLora = false;
    /// LoRA rank (adapter bottleneck dimension).
    int LoraRank = 8;
    /// LoRA alpha scaling factor.
    float LoraAlpha = 16.0f;
    /// LoRA dropout probability.
    float LoraDropout = 0.0f;
    /// LoRA master weights dtype (optimizer/export). Compute uses model-dtype.
    ETensorDType LoraDType = ETensorDType::FP32;
    /// Comma-separated module list to apply LoRA to (e.g. "q_proj,k_proj,...").
    std::string LoraTargetModules = "q_proj,k_proj,v_proj,o_proj";
    /// Use RSLoRA scaling (sqrt(rank) denominator).
    bool LoraUseRSLoRA = false;

    // QLoRA configuration (quantized base weights + LoRA adapters)
    /// Enable FP8 QLoRA mode (base weights stored as FP8 with per-block scales).
    bool UseQLoRAFP8 = false;
    /// Enable NVFP4 QLoRA mode (base weights stored as FP4 E2M1 with two-level block scales).
    /// Requires Blackwell GPU (SM100+).
    bool UseQLoRAFP4 = false;
    /// Block size for per-block quantization in FP8 QLoRA (64, 128, or 256).
    int QLoRABlockSize = 128;
    /// Enable Four Over Six (4/6) adaptive block scaling for NVFP4 QLoRA quantization.
    bool QLoRAFourOverSix = true;

    // Modular architecture system
    /// Architecture type for modular system: "dense", "moe", or "hybrid".
    std::string ArchType = "dense";
    /// Number of experts for MoE architectures.
    int NumExperts = 8;
    /// Top-k experts per token for MoE routing.
    int MoETopK = 2;
    /// Enable shared expert (Nemotron/DeepSeek style) for MoE.
    bool UseSharedExpert = false;
    /// Auxiliary loss coefficient for MoE load balancing.
    float MoEAuxLossCoef = 0.01f;
    /// Capacity factor for MoE expert buffers.
    float MoECapacityFactor = 1.25f;

    /// Low-level runtime options (recompute, sharding, cuda graphs, optimizer state dtypes, etc.).
    RuntimeOptions Options;

    /// ZeRO redundancy optimization level (1=optimizer, 2=+gradients, 3=+weights).
    int ZeroLevel = 1;

    /// If >=0, log allocations >= this many bytes (set from MiB CLI value).
    int LogAllocations = -1;
    int DebugLogAbsMaxes = -1;

    /// True if the user specified --use-zero-copy or --no-use-zero-copy.
    bool UseZeroCopySpecified = false;
    /// Startup timestamp used to report setup time.
    std::chrono::steady_clock::time_point BeginStartup;

    /**
     * @brief Parse command-line options into this runner's fields (CLI11).
     * @param argc Argument count.
     * @param argv Argument vector.
     */
    void load_training_config(int argc, const char** argv);

    /**
     * @brief Launch training either via MPI communicator (multi-process) or threads backend (single process).
     * @param argc Argument count (forwarded for logging).
     * @param argv Argument vector (forwarded for logging).
     */
    void launch_training(int argc, const char** argv);

    /**
     * @brief Execute the full training loop on the current rank (data, model init/load, train/eval/ckpt/export).
     * @param argc Argument count (used for logger command capture).
     * @param argv Argument vector (used for logger command capture).
     * @param comm Initialized communicator providing rank/world_size collectives.
     */
    void run_training(int argc, const char** argv, NCCLCommunicator& comm);


    /**
     * @brief Build a modules::MoEConfig from parsed CLI options.
     * @return MoEConfig populated with MoE parameters.
     */
    modules::MoEConfig create_moe_config() const {
        modules::MoEConfig moe;
        moe.num_experts = NumExperts;
        moe.top_k = MoETopK;
        moe.use_shared_expert = UseSharedExpert;
        moe.router_aux_loss_coef = MoEAuxLossCoef;
        moe.capacity_factor = MoECapacityFactor;
        return moe;
    }

    /**
     * @brief Build a modules::ModelConfig from PretrainedConfig and parsed CLI options.
     * @param base_config The loaded PretrainedConfig.
     * @return ModelConfig for the modular system.
     */
    modules::ModelConfig create_modular_config(const PretrainedConfig& base_config) const {
        using namespace modules;

        // Start from PretrainedConfig conversion
        ModelConfig config = ModelConfig::from_pretrained_config(base_config);

        // Set architecture type
        if (iequals(ArchType, "moe")) {
            config.architecture = ArchitectureType::MoE;
            config.moe_config = create_moe_config();
        } else if (iequals(ArchType, "hybrid")) {
            config.architecture = ArchitectureType::Hybrid;
            config.moe_config = create_moe_config();
        } else {
            config.architecture = ArchitectureType::Dense;
        }

        return config;
    }

    /**
     * @brief Build a modules::ModelOptions from RuntimeOptions.
     * @return ModelOptions for the modular system.
     */
    modules::ModelOptions create_modular_options() const {
        return modules::ModelOptions::from_runtime_options(Options);
    }


    /**
     * @brief Build a LoRAAdapterConfig from parsed CLI options.
     * @return LoRAAdapterConfig with Rank=0 if LoRA is disabled; otherwise fully populated config.
     */
    LoRAAdapterConfig create_lora_config() const {
        if (!UseLora) {
            return LoRAAdapterConfig{.Rank = 0};  // Disabled LoRA
        }
        
        // Parse target modules from comma-separated string
        std::set<std::string> target_modules;
        std::string module;
        std::istringstream stream(LoraTargetModules);
        while (std::getline(stream, module, ',')) {
            // Trim whitespace
            size_t start = module.find_first_not_of(" \t");
            size_t end = module.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
                target_modules.insert(module.substr(start, end - start + 1));
            }
        }
        
        return LoRAAdapterConfig{
            .Rank = LoraRank,
            .Alpha = LoraAlpha,
            .Dropout = LoraDropout,
            .TargetModules = target_modules,
            .DType = LoraDType,
            .UseRSLoRA = LoraUseRSLoRA
        };
    }
};

void TrainingRunner::load_training_config(int argc, const char** argv) {
    BeginStartup = std::chrono::steady_clock::now();
    CLI::App app;

    std::string matmul_dtype = "";
    std::string gradient_dtype = "";
    std::string master_dtype = "";

    Options.KeepAllActivations = false;
    Options.RecomputeSwiGLu = false;
    Options.RecomputeRMSNorm = false;
    Options.RecomputeFFN = false;
    Options.UseCudaGraphs = true;

    app.add_option("--model", ModelRootPath, "Path to the huggingface model directory or name of a HF model that is cache locally.");
    auto from_scratch = app.add_flag("--from-scratch", FromScratch, "Train the model from a random initialization");
    app.add_flag("--init-proj-to-zero", Options.InitProjectionsToZero, "Init (ffn.down and att.out) projections to zero, as in modded-nanogpt")->needs(from_scratch);
    app.add_option("--matmul-dtype", matmul_dtype,
                   "Which dtype to use for matmuls. Defaults to model-dtype. Note: recipes override forward/backward matmul dtypes.");
    app.add_option("--gradient-dtype", gradient_dtype,
                   "Which dtype to use for (activation) gradients / backward matmul policy. Defaults to matmul-dtype. Note: recipes may override backward dtype.");
    app.add_option("--master-dtype", master_dtype,
                   "Master weight dtype used for optimizer updates (e.g. FP32 for more stable full fine-tuning). Defaults to model-dtype.");
    app.add_option("--model-dtype", ModelDType, "Which dtype to use for model");
    app.add_option("--batch,--batch-size", B, "micro-batch size")->check(CLI::PositiveNumber);
    app.add_option("--seq-len,--seq-length", T, "sequence length")->check(CLI::PositiveNumber);
    app.add_option("--lmhead-chunks", Options.LMHeadChunks, "Run the LM-Head in chunks to avoid materializing the large logit tensor.")->check(CLI::PositiveNumber);
    app.add_option("--attn-bwd-chunks", Options.AttBwdChunks, "Run the attention backward pass in chunks, to avoid having to materialize a large workspace tensor.")->check(CLI::PositiveNumber);

    // debug
    app.add_option("--name", RunName, "Associate a name with this run. This will not influence any computations. You can use %n as part of specifying log, output, and checkpoint file names.");
    app.add_option("--debug-log-allocations", LogAllocations, "Log all memory allocations larger than the given number (in MiB)");
    app.add_flag("--debug-time-breakdown", Options.TriggerTimingEvents, "Log additional timing information");
    app.add_flag("--debug-log-abs-max", DebugLogAbsMaxes, "Log abs-maxes every n steps");

    // optimizer
    app.add_option("--lr,--learning-rate", LearningRate, "Base learning rate")->check(CLI::NonNegativeNumber);
    app.add_option("--warmup", WarmupSteps, "Number of warmup steps.")->check(CLI::NonNegativeNumber);;
    app.add_option("--cooldown", CoolDownSteps, "Number of cool-down steps, using 1-sqrt() to anneal learning rate to zero");
    app.add_option("--final-lr-fraction", FinalLrFraction, "Fraction of base lr to use for the final steps.")->check(CLI::NonNegativeNumber);
    app.add_option("--lr-schedule", LrScheduleType, "Learning rate schedule function: Cosine or Linear");
    app.add_option("--beta-1", Beta1, "Beta 1 for Adam")->check(CLI::NonNegativeNumber);
    app.add_option("--beta-2", Beta2, "Beta 2 for Adam")->check(CLI::NonNegativeNumber);
    app.add_option("--grad-accumulation", GradAccSteps, "number of micro-batches per optimizer step")->check(CLI::PositiveNumber);
    app.add_option("--grad-clip", GradClip, "Gradient clipping");
    app.add_option("--weight-decay", WeightDecay, "Weight decay for matrix parameters")->check(CLI::NonNegativeNumber);
    app.add_option("--adam-epsilon", Epsilon, "Epsilon to use for AdamW")->check(CLI::NonNegativeNumber);

    auto steps_opt = app.add_option("--steps", MaxSteps, "Number of training steps");
    app.add_option("--epochs", NumEpochs, "Number of training steps")->excludes(steps_opt)->check(CLI::PositiveNumber);
    app.add_option("--log-gpu-util", LogGPUEvery, "Log the gpu utilization every n steps. Set to 0 to disable.");
    app.add_option("--eval-every-n-steps", EvalEvery, "How many optimizer steps between evaluations");
    app.add_option("--eval-num-steps", EvalNumSteps, "How many batches of eval data to use");
    app.add_option("--final-eval-num-steps", FinalEvalNumSteps, "Final evaluation batches after training: -1=full eval set (default), 0=skip, >0=limit.");

    app.add_option("--train-file", TrainFile, "Tokens for training");
    app.add_option("--train-seed", TrainLoaderSeed, "Seed for the training dataloader");
    app.add_option("--eval-file", EvalFile, "Tokens for validation");
    app.add_option("--eval-seed", EvalLoaderSeed, "Seed for the eval loader");
    app.add_option("--out-dir", OutDir, "Where to save the trained model");
    app.add_flag("--save,!--no-save", SaveFinalModel, "Enable/disable exporting the final model weights/config.");
    app.add_option("--checkpoint-dir", CkptDir, "Directory in which to save checkpoints.");
    app.add_option("--ckpt-interval", CkptEvery, "How many optimizer steps between checkpoints");
    app.add_option("--ckpt-keep-n", CkptToKeep, "Clean up old checkpoints, only preserving the latest n.");
    app.add_option("--ckpt-major", MajorCkptEvery, "Make every nth checkpoint a major checkpoint, which does not get cleaned up.");
    auto continue_from_checkpoint = app.add_option("--continue", ContinueFromCheckpoint,
        "Continue from checkpoint. If no number is given, uses the latest checkpoint")->expected(0, 1)->default_str("-1");

    app.add_option("--log-file", LogFile, "Where to save the training log");
    app.add_option("--gpus", NGPUs, "How many GPUs to use for training.");

    //  options
    app.add_flag("--recompute-swiglu", Options.RecomputeSwiGLu, "Recompute swiglu during the backward pass to save activation memory");
    app.add_flag("--recompute-norm", Options.RecomputeRMSNorm, "Recompute rms-norms during the backward pass to save activation memory");
    app.add_flag("--recompute-ffn", Options.RecomputeFFN, "Recompute the feed-forward block during the backward pass to save activation memory");
    app.add_flag("--recompute-qkv", Options.RecomputeQKV, "Recompute the qkv projections during the backward pass");
    app.add_flag("--recompute-att", Options.RecomputeAtt, "Recompute the attention block during the backward pass");
    app.add_flag("--recompute-block", Options.RecomputeBlock, "Recompute the entire transformer block");
    app.add_flag("--offload-residual", Options.OffloadResidual, "Offload the residual of the feed-forward block to host memory");
    app.add_flag("--use-cuda-graphs,!--no-use-cuda-graphs", Options.UseCudaGraphs, "Enable/disable use of cuda graphs");
    app.add_option("--zero-level", ZeroLevel, "Zero redundancy level: 1 - sharded optimizer; 2 - sharded gradients; 3 - sharded weights");
    app.add_flag("--offload-master", Options.OffloadMaster, "Store master weights in pinned host memory.");
    auto persist = app.add_flag("--persistent-quants", Options.PersistentQuants, "Keep quantized weights around, instead of re-quantizing");
    app.add_flag("--offload-quants", Options.OffloadQuants, "Store quantized weights in pinned host memory.")->needs(persist);
    app.add_flag("--offload-optimizer", Options.OffloadOptimizer, "Store optimizer state in pinned host memory.");
    app.add_flag("--offload-gradients", Options.OffloadGrads, "Offload gradients to pinned host memory.");
    auto use_zero_copy = app.add_flag("--use-zero-copy,!--no-use-zero-copy", Options.UseZeroCopy, "Use ZeroCopy memory access, instead of double-buffered cudaMemcpy, for offloaded optimizer states. On consumer cards, DMA appears to be much slower, whereas on professional cards it is faster.");
    app.add_flag("--memcpy-all-gather", MemcpyAllGather, "Use memcpy to perform all-gathers. Currently only supported by the threads backend.");
    auto memcpy_send_recv = app.add_flag("--memcpy-send-recv", MemcpySendRecv, "Use memcpy to perform send/receive (all-to-all). Currently only supported by the threads backend.");
    auto all_to_all_reduce = app.add_flag("--all-to-all-reduce", Options.UseAllToAllReduce, "Uses an all-to-all-based reduce algorithm. Implies --memcpy-send-recv.");
    app.add_flag("--write-combined", Options.UseWriteCombined, "Uses write-combined memory for offloaded tensors.");

    // Training recipe (quantization strategy)
    std::string recipe_name = "bf16";  // Default: no quantization
    app.add_option("--recipe", recipe_name,
        "Training recipe: bf16 (default), fp8-hybrid, nvfp4")
        ->check(CLI::IsMember({"bf16", "fp8-hybrid", "nvfp4"}));

    // Recipe-specific options (only used when corresponding recipe is selected)
    std::string fp4_backend_str = "cudnn";
    app.add_option("--fp8-amax-history", Options.RecipeOptions.fp8_amax_history_len,
        "FP8 delayed scaling amax history length (default: 1024, for fp8-hybrid recipe)")
        ->check(CLI::PositiveNumber);
    app.add_option("--fp4-backend", fp4_backend_str,
        "FP4 matmul backend: cudnn (default) or cutlass (for nvfp4 recipe)")
        ->check(CLI::IsMember({"cudnn", "cutlass"}, CLI::ignore_case));
    app.add_flag("--no-fp4-hadamard", Options.RecipeOptions.fp4_disable_rht,
        "Disable Random Hadamard Transform for NVFP4 recipe");
    app.add_flag("--no-fp4-stochastic-rounding", Options.RecipeOptions.fp4_disable_stochastic_rounding,
        "Disable stochastic rounding for NVFP4 gradient quantization");
    app.add_option("--skip-quant-first-layers", Options.RecipeOptions.skip_quant_first_layers,
        "Skip quantization for the first N transformer layers (embedding layers kept in BF16)")
        ->check(CLI::NonNegativeNumber);
    app.add_option("--skip-quant-last-layers", Options.RecipeOptions.skip_quant_last_layers,
        "Skip quantization for the last N transformer layers (lm_head layers kept in BF16)")
        ->check(CLI::NonNegativeNumber);

    app.add_flag("--fused-rope", Options.UseFusedRope,
        "Use fused RoPE kernel with on-the-fly cos/sin computation (saves memory, reduces bandwidth)");

    // LoRA options
    app.add_flag("--lora,--use-lora", UseLora, "Enable LoRA adapter training (freezes base model)");
    app.add_option("--lora-rank", LoraRank, "LoRA rank (low-rank decomposition dimension)")->check(CLI::PositiveNumber);
    app.add_option("--lora-alpha", LoraAlpha, "LoRA alpha scaling factor")->check(CLI::NonNegativeNumber);
    app.add_option("--lora-dropout", LoraDropout, "LoRA dropout probability")->check(CLI::NonNegativeNumber);
    app.add_option("--lora-dtype", LoraDType, "LoRA master weights dtype for optimizer/export (compute uses model-dtype)")
        ->default_val("fp32");
    app.add_option("--lora-target-modules", LoraTargetModules, "Comma-separated list of modules to apply LoRA to (e.g., 'q_proj,k_proj,v_proj,o_proj')");
    app.add_flag("--lora-use-rslora", LoraUseRSLoRA, "Use RSLoRA scaling (sqrt(rank) instead of rank in denominator)");

    // QLoRA options (quantized base weights)
    app.add_flag("--qlora-fp8", UseQLoRAFP8, "Enable FP8 QLoRA mode (base weights quantized to FP8 with per-block scales)")
        ->needs("--lora");
    app.add_flag("--qlora-fp4", UseQLoRAFP4, "Enable NVFP4 QLoRA mode (base weights quantized to FP4 E2M1). Requires Blackwell GPU (SM100+)")
        ->needs("--lora")
        ->excludes("--qlora-fp8");
    app.add_option("--qlora-block-size", QLoRABlockSize, "Block size for per-block quantization in FP8 QLoRA (64, 128, or 256)")
        ->check(CLI::IsMember({64, 128, 256}));
    app.add_flag("--qlora-four-over-six,!--no-qlora-four-over-six", QLoRAFourOverSix,
        "Enable Four Over Six (4/6) adaptive block scaling for NVFP4 QLoRA (default: enabled)")
        ->needs("--qlora-fp4");

    // Modular architecture system
    app.add_option("--arch", ArchType, "Architecture type: dense, moe, or hybrid")
        ->check(CLI::IsMember({"dense", "moe", "hybrid"}, CLI::ignore_case));
        
    app.add_option("--num-experts", NumExperts, "Number of experts for MoE architectures")->check(CLI::PositiveNumber);
    app.add_option("--moe-top-k", MoETopK, "Top-k experts per token for MoE routing")->check(CLI::PositiveNumber);
    app.add_flag("--moe-shared-expert", UseSharedExpert, "Enable shared expert (Nemotron/DeepSeek style)");
    app.add_option("--moe-aux-loss", MoEAuxLossCoef, "Auxiliary loss coefficient for MoE load balancing")->check(CLI::NonNegativeNumber);
    app.add_option("--moe-capacity", MoECapacityFactor, "Capacity factor for MoE expert buffers")->check(CLI::PositiveNumber);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::exit(app.exit(e));
    }

    // Parse FP4 backend option
    if (iequals(fp4_backend_str, "cutlass")) {
        Options.RecipeOptions.fp4_backend = EMatmulBackend::CUTLASS;
    } else {
        Options.RecipeOptions.fp4_backend = EMatmulBackend::CUBLASLT;  // cuDNN uses cuBLASLt path
    }

    // Create recipe from CLI options
    Options.TrainingRecipe = recipes::RecipeFactory::create(recipe_name, Options.RecipeOptions);

    // Set matmul backend from recipe (each recipe knows its required/preferred backend)
    Options.MatmulBackend = Options.TrainingRecipe->matmul_backend();

    // Validate FP4 hardware support (requires Blackwell SM100+)
    if (Options.fp4_enabled()) {
        if (!device_supports_fp4()) {
            throw std::runtime_error(
                "FP4 training requires Blackwell GPU (SM100+). "
                "Your current GPU does not support native FP4 operations. "
                "Use --recipe=bf16 or --recipe=fp8-hybrid.");
        }

        auto is_primary_process = []() -> bool {
            if (const char* r = std::getenv("OMPI_COMM_WORLD_RANK")) {
                return std::string(r) == "0";
            }
            return true;
        };

        if (is_primary_process()) {
            const auto& recipe = *Options.TrainingRecipe;

            // Use unified recipe interface for configuration
            bool is_cutlass = recipe.matmul_backend() == EMatmulBackend::CUTLASS;
            bool rht_enabled = recipe.requires_hadamard_workspace();
            bool sr_enabled = recipe.quant_bwd_grad().stochastic_rounding;
            bool scaled_swiglu = recipe.requires_scaled_swiglu();

            std::cerr
                << "FP4 training enabled: using FP4 E2M1 with two-level block scaling.\n"
                << "  - Recipe: " << Options.recipe_name() << "\n"
                << "  - Backend: " << (is_cutlass ? "CUTLASS" : "cuDNN") << "\n"
                << "  - Forward: FP4 quantization\n"
                << "  - Backward: FP4 with " << (sr_enabled ? "stochastic rounding" : "no stochastic rounding") << "\n"
                << "  - RHT: " << (rht_enabled ? "enabled" : "disabled") << "\n";
            if (scaled_swiglu) {
                std::cerr << "  - Scaled SwiGLU: enabled\n";
            }
        }
    }

    if (all_to_all_reduce->count() > 0) {
        // --all-to-all-reduce requires point-to-point send/recv; treat it as an alias for enabling memcpy send/recv.
        MemcpySendRecv = true;
    }

    UseZeroCopySpecified = (use_zero_copy->count() > 0);

    // Preserve original `--model` value before we potentially resolve HF repo ids to local cache paths.
    ModelReference = ModelRootPath;

    if (!std::filesystem::exists(ModelRootPath)) {
        if (ModelRootPath.find('/') != std::string::npos) {
            std::string hf_path = get_hf_model_files(ModelRootPath);
            if (hf_path.empty()) {
                throw std::runtime_error("Could not find model files for " + ModelRootPath);
            }
            ModelRootPath = hf_path;
        }
    }

    Options.MatmulType = matmul_dtype.empty() ? std::optional<ETensorDType>{} : dtype_from_str(matmul_dtype);
    Options.GradientType = gradient_dtype.empty() ? std::optional<ETensorDType>{} : dtype_from_str(gradient_dtype);
    Options.MasterDType = master_dtype.empty() ? std::optional<ETensorDType>{} : dtype_from_str(master_dtype);

    if (Options.MasterDType.has_value()) {
        const auto dt = Options.MasterDType.value();
        if (dt != ETensorDType::FP32 && dt != ETensorDType::BF16) {
            throw std::runtime_error("--master-dtype must be FP32 or BF16");
        }
    }

    switch (ZeroLevel) {
    case 0:
        std::cerr << "Warning: ZeRO-level 0 not supported, defaulting to 1" << std::endl;
        break;
    case 1:
        break;
    case 3:
        Options.ShardWeights = true;
        [[fallthrough]];
    case 2:
        Options.ShardGradients = true;
        break;
    default:
        std::cerr << "Warning: Invalid ZeRO-level " << ZeroLevel << std::endl;
    }

    // Validate dependencies after ZeroLevel has been applied
    if (Options.PersistentQuants && !Options.ShardWeights) {
        std::cerr << "Error: --persistent-quants requires --zero-level 3 (which shards weights)" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (Options.RecomputeBlock) {
        Options.RecomputeAtt = true;
        Options.RecomputeFFN = true;
        Options.RecomputeRMSNorm = true;
    }

    if (Options.RecomputeAtt) {
        Options.RecomputeQKV = true;
    }
    if (Options.RecomputeFFN) {
        Options.RecomputeSwiGLu = true;
    }

    LogAllocations *= 1024 * 1024;

    LogFile = replace(LogFile, "%n", RunName);
    OutDir = replace(OutDir, "%n", RunName);
    CkptDir = replace(CkptDir, "%n", RunName);
}

void TrainingRunner::launch_training(int argc, const char** argv) {
    if (const char* mpi_world_size = getenv("OMPI_COMM_WORLD_SIZE"); mpi_world_size) {
        // MPI code path -- this region is entered by all processes, so no additional
        // launching necessary
        NGPUs = std::stoi(mpi_world_size);
        if (NGPUs != 0 && NGPUs != std::stoi(getenv("OMPI_COMM_WORLD_SIZE"))) {
            throw std::runtime_error("Number of GPUs does not match OMPI_COMM_WORLD_SIZE");
        }
        auto comm = NCCLCommunicator::make_mpi_communicator();
        run_training(argc, argv, *comm);
    } else {
        // Threads code path -- launch one thread per GPU
        int gpus_available = 0;
        CUDA_CHECK(cudaGetDeviceCount(&gpus_available));
        if (NGPUs == 0) {
            NGPUs = gpus_available;
        }

        if (NGPUs > gpus_available) {
            std::cerr << "Error: requested " << NGPUs << " GPUs, but only " << gpus_available << " found" << std::endl;
            std::exit(1);
        }
        // Use the blocking runner so exceptions get rethrown from `join()` (not a noexcept destructor).
        NCCLCommunicator::run_threads_communicators(
            NGPUs, MemcpyAllGather, MemcpySendRecv,
            [&](NCCLCommunicator& comm) { run_training(argc, argv, comm); });
    }
}

/**
 * @brief Load the model configuration for a given root directory/name.
 *
 * @param root Path to a model directory (must contain config.json) or a model name (used for from-scratch presets).
 * @param from_scratch If true, allow creating a config from a known model name when config.json is absent.
 * @param dtype Desired dtype for the created/loaded config.
 * @return Parsed or synthesized PretrainedConfig; exits the process on unrecoverable errors.
 */
PretrainedConfig create_config(const std::string& root, bool from_scratch, ETensorDType dtype) {
    std::string config_path = root + "/config.json";
    if(std::filesystem::exists(config_path)) {
        return load_pretrained_config(config_path.c_str(), dtype);
    } else if(from_scratch) {
        auto cfg = create_pretrained_config_from_name(root, dtype);
        return cfg;
    } else {
        std::cerr << "Could not find model config at " << config_path << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void TrainingRunner::run_training(int argc, const char** argv, NCCLCommunicator& comm) {
    // local printf that prints only on rank 0
    auto printf = [enabled = comm.rank() == 0](const char* fmt, const auto& ... args) {
        if (enabled)
            ::printf(fmt, args...);
    };

    std::unique_ptr<IGPUUtilTracker> gpu_util = IGPUUtilTracker::create();
    int total_batch_size = B * T * comm.world_size() * GradAccSteps;

    // Fail fast if zero-copy was requested but is not supported at runtime.
    // This avoids hard-to-debug crashes when kernels access unmapped host pointers.
    if (Options.UseZeroCopy && (Options.OffloadMaster || Options.OffloadOptimizer || Options.OffloadGrads || Options.OffloadQuants)) {
        try {
            validate_zero_copy_support_or_throw();
        } catch (const std::exception& e) {
            const std::string hint = UseZeroCopySpecified
                ? " (try --no-use-zero-copy)"
                : " (auto-enabled; you can force staging with --no-use-zero-copy)";
            throw std::runtime_error(std::string(e.what()) + hint);
        }
    }

    std::string model_path = ModelRootPath + "/model.safetensors";
    if (!std::filesystem::exists(model_path)) {
        model_path = ModelRootPath + "/model.safetensors.index.json";
    }
    PretrainedConfig config = create_config(ModelRootPath, FromScratch, ModelDType);
    Options.ModelType = config.DType;

    TrainingRunLogger logger(LogFile, comm.rank(), TrainingRunLogger::VERBOSE);
    logger.log_cmd(argc, argv);

    DataLoader train_loader({TrainFile}, T, comm.rank(), comm.world_size(), TrainLoaderSeed);
    DataLoader test_loader({EvalFile}, T, comm.rank(), comm.world_size(), EvalLoaderSeed);

    int steps_per_epoch = train_loader.num_tokens() / total_batch_size;
    if (MaxSteps == -1) {
        MaxSteps = steps_per_epoch * NumEpochs;
    }

    logger.log_options({
        {"name",               RunName},
        {"recompute-swiglu",   Options.RecomputeSwiGLu},
        {"recompute-norm",     Options.RecomputeRMSNorm},
        {"recompute-ffn",      Options.RecomputeFFN},
        {"recompute-qkv",      Options.RecomputeQKV},
        {"recompute-att",      Options.RecomputeAtt},
        {"recompute-block",    Options.RecomputeBlock},
        {"lm-head-chunks",     Options.LMHeadChunks},
        {"attn-bwd-chunks",    Options.AttBwdChunks},
        {"offload-master",     Options.OffloadMaster},
        {"offload-quants",     Options.OffloadQuants},
        {"offload-optimizer",  Options.OffloadOptimizer},
        {"use-zero-copy",      Options.UseZeroCopy},
        {"zero-level",         ZeroLevel},
        {"shard-weights",      Options.ShardWeights},
        {"shard-gradients",    Options.ShardGradients},
        {"persistent-quants",  Options.PersistentQuants},
        {"cuda-graphs",        Options.UseCudaGraphs},
        {"all-to-all-reduce",  Options.UseAllToAllReduce},
        {"memcpy-all-gather",  MemcpyAllGather},
        {"memcpy-send-recv",   MemcpySendRecv},
        {"use-write-combined", Options.UseWriteCombined},
        {"matmul-dtype",       dtype_to_str(Options.matmul_dtype())},
        {"gradient-dtype",     dtype_to_str(Options.grad_dtype())},
        {"model-dtype",        dtype_to_str(ModelDType)},
        {"recipe",             std::string(Options.recipe_name())},
        {"optimizer",          std::string("adamw8bit")},

        {"learning-rate",      LearningRate},
        {"warmup",             WarmupSteps},
        {"cooldown",           CoolDownSteps},
        {"final-lr",           FinalLrFraction},
        {"lr-schedule",        LrScheduleType},
        {"beta-1",             Beta1},
        {"beta-2",             Beta2},
        {"adam-epsilon",       Epsilon},
        {"grad-clip",          GradClip},
        {"weight-decay",       WeightDecay},
        {"grad-accumulation",  GradAccSteps},

        {"micro-batch",        B},
        {"seq-len",            T},
        {"total-batch",        total_batch_size},
        {"steps-per-epoch",    steps_per_epoch},
        {"steps",              MaxSteps},
        {"eval-every-n-steps", EvalEvery},
        {"eval-num-steps",     EvalNumSteps},
        {"ckpt-every-n-steps", CkptEvery},
        {"ckpt-keep",          CkptToKeep},
        {"ckpt-major",         MajorCkptEvery},

        {"train-loader-seed",  (std::int64_t)TrainLoaderSeed},
        {"eval-loader-seed",   (std::int64_t)EvalLoaderSeed},

        {"arch",               std::string{config.model_name()}},
        {"vocab-size",         config.VocabSize},
        {"hidden-size",        config.HiddenSize},
        {"ffn-size",           config.IntermediateSize},
        {"num-layers",         config.NumLayers},
        {"tied-word-emb",      config.TiedWordEmbeddings},
        {"num-heads",          config.NumQueryHeads},
        {"num-kv-heads",       config.NumKeyValHeads},

        {"train-file",         TrainFile},
        {"eval-file",          EvalFile},
        {"checkpoint-dir",     CkptDir},
        {"out-dir",            OutDir},
        {"log-file",           LogFile},

        // LoRA configuration
        {"lora-enabled",       UseLora},
        {"lora-rank",          LoraRank},
        {"lora-alpha",         LoraAlpha},
        {"lora-dropout",       LoraDropout},
        {"lora-dtype",         dtype_to_str(LoraDType)},
        {"lora-target-modules", LoraTargetModules},
        {"lora-use-rslora",    LoraUseRSLoRA},

        // QLoRA configuration
        {"qlora-fp8",          UseQLoRAFP8},
        {"qlora-fp4",          UseQLoRAFP4},
        {"qlora-four-over-six", QLoRAFourOverSix},

        {"arch-type",          ArchType},
        {"num-experts",        NumExperts},
        {"moe-top-k",          MoETopK},
        {"moe-shared-expert",  UseSharedExpert},
        {"moe-aux-loss-coef",  MoEAuxLossCoef},
        {"moe-capacity-factor", MoECapacityFactor}
    });

    logger.log_gpu_model(comm);
  
   
    // Preflight: make unsupported dtype choices fail fast with a helpful message.
    // Without this, some failures can occur deep in cuBLASLt (and may be masked by terminate()).
    if (is_fp8_dtype(Options.matmul_dtype()) || is_fp8_dtype(Options.grad_dtype())) {
        if (!device_supports_fp8()) {
            throw std::runtime_error(fmt::format(
                "FP8 requested (matmul-dtype={}, gradient-dtype={}) but this GPU does not support FP8 tensor cores. "
                "Use --matmul-dtype=bf16 (and optionally --gradient-dtype=bf16).",
                dtype_to_str(Options.matmul_dtype()), dtype_to_str(Options.grad_dtype())));
        }
    }

    auto allocator = std::make_shared<TensorAllocator>();
    if (LogAllocations >= 0 && comm.rank() == 0) {
        allocator->set_callback([this](const std::string& ctx, const std::string& name, EAllocationType kind, std::size_t amount){
            if(amount >= LogAllocations) {
                const char* kind_str;
                switch(kind) {
                    case EAllocationType::ON_DEVICE: kind_str = "GPU "; break;
                    case EAllocationType::MANAGED: kind_str = "USM "; break;
                    case EAllocationType::WRITE_CMB:
                    case EAllocationType::PINNED:
                    case EAllocationType::ON_HOST: kind_str = "HOST"; break;
                }
                ::printf("%s  %15s - %15s: %4zu MiB\n", kind_str, ctx.c_str(), name.c_str(), amount / 1024 / 1024);
            }
        });
    }

    logger.log_sol_estimate(get_transformer_ops(
                                config.NumLayers * ((long)config.HiddenSize * (config.IntermediateSize * 3 + config.HiddenSize * 1 + config.qkv_channels())),
                                Options.forward_matmul_dtype(), (long)config.VocabSize * config.HiddenSize, config.DType,
                                config.NumQueryHeads * config.head_size(), config.NumLayers, T),
                            comm.world_size());

    // Note: cannot check for exact equality, because vocab_size differs in tokenizer vs model (implicit padding)
    if (train_loader.vocab_size() > 0 && train_loader.vocab_size() > config.VocabSize) {
        std::cerr << "\033[1;31mError: model vocab size " << config.VocabSize
                  << " does not match training data vocab size " << train_loader.vocab_size() << "\033[0m" << std::endl;
        std::exit(1);
    }
    if (test_loader.vocab_size() > 0 && test_loader.vocab_size() > config.VocabSize) {
        std::cerr << "\033[1;31mError: model vocab size " << config.VocabSize
                  << " does not match validation data vocab size " << test_loader.vocab_size() << "\033[0m"
                  << std::endl;
        std::exit(1);
    }

    int latest_step = -1;
    if (ContinueFromCheckpoint.has_value()) {
        if (ContinueFromCheckpoint.value() < 0) {
            latest_step = find_latest_checkpoint(CkptDir);
        } else {
            latest_step = ContinueFromCheckpoint.value();
        }
        if (latest_step < 0) {
            std::cerr << "No checkpoint found in " << CkptDir << std::endl;
            std::cerr << " starting from scratch" << std::endl;
        }
    }

	    // ========================================================================
	    // Model Creation: Modular system (only)
	    // ========================================================================
	    std::unique_ptr<IModel> model_storage;
	    IModel* model = nullptr;

	    // Create modular configuration
	    modules::ModelConfig mod_config = create_modular_config(config);
	    modules::ModelOptions mod_options = create_modular_options();

	    // QLoRA: skip block weight allocation
	    if (UseLora && (UseQLoRAFP8 || UseQLoRAFP4)) {
	        mod_options.skip_block_allocation = true;
	    }

	    // Log architecture info
	    const char* arch_name = "dense";
	    if (mod_config.architecture == modules::ArchitectureType::MoE) {
	        arch_name = "moe";
	        if (mod_config.moe_config.has_value()) {
	            const auto& moe = mod_config.moe_config.value();
	            printf("MoE config: %d experts, top-%d routing, aux_loss=%.4f, capacity=%.2f%s\n",
	                   moe.num_experts, moe.top_k, moe.router_aux_loss_coef, moe.capacity_factor,
	                   moe.use_shared_expert ? ", shared expert" : "");
	        }
	    } else if (mod_config.architecture == modules::ArchitectureType::Hybrid) {
	        arch_name = "hybrid";
	    }
	    printf("Architecture: %s\n", arch_name);

	    // Create model via factory
	    model_storage = modules::ModelFactory::create(
	        mod_config, mod_options, comm.rank(), comm.world_size(), allocator);
	    if (UseLora) {
	        if (mod_config.architecture != modules::ArchitectureType::Dense) {
	            throw std::runtime_error("--lora currently supports only --arch=dense");
	        }

	        LoRAAdapterConfig lora_config = create_lora_config();
	        printf("LoRA enabled: rank=%d, alpha=%.1f, targets=%s%s\n",
	               lora_config.Rank, lora_config.Alpha, LoraTargetModules.c_str(),
	               lora_config.UseRSLoRA ? " (RSLoRA)" : "");

            // Build QLoRA config if enabled
            modules::QLoRAConfig qlora_config;
            if (UseQLoRAFP8) {
                qlora_config = modules::QLoRAConfigBuilder()
                    .fp8()
                    .block_size(QLoRABlockSize)
                    .build();
                printf("QLoRA-FP8 enabled: block_size=%d (base weights quantized to FP8 with per-block scales)\n",
                       QLoRABlockSize);
                printf("Expected memory savings: ~50%% for base model weights\n");
            } else if (UseQLoRAFP4) {
                qlora_config = modules::QLoRAConfigBuilder()
                    .nvfp4()
                    .four_over_six(QLoRAFourOverSix)
                    .build();
                printf("QLoRA-FP4 enabled: NVFP4 (E2M1) with two-level block scaling%s\n",
                       QLoRAFourOverSix ? " + 4/6 adaptive scaling" : "");
                printf("Expected memory savings: ~75%% for base model weights\n");
            }

	        // Convert LoRAAdapterConfig -> modular LoRA config (modules/lora).
	        modules::ModularLoRAConfig mod_lora;
	        mod_lora.rank = lora_config.Rank;
	        mod_lora.alpha = lora_config.Alpha;
	        mod_lora.dropout = lora_config.Dropout;
	        // LoRA master weights are always FP32 for optimizer compatibility (8-bit AdamW).
	        // The compute dtype (for matmuls) is derived from model-dtype automatically.
	        mod_lora.dtype = lora_config.DType;  // FP32 by default
	        mod_lora.init_a_kaiming = lora_config.InitAKaimingUniform;
	        mod_lora.use_rs_lora = lora_config.UseRSLoRA;
	        mod_lora.targets.clear();
	        if (lora_config.TargetModules.count("all") > 0) {
	            mod_lora.with_all();
	        } else {
	            for (const auto& name : lora_config.TargetModules) {
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
	            throw std::runtime_error("Internal error: modular dense model factory returned unexpected type");
	        }
	        std::unique_ptr<DenseModel> dense_base(static_cast<DenseModel*>(model_storage.release()));
	        model_storage = std::make_unique<modules::ModularLoRAModel<DenseBlock>>(
	            std::move(dense_base), mod_lora, Options, comm, allocator, qlora_config);
	        model = model_storage.get();

	        if (comm.rank() == 0) {
	            auto* lora = dynamic_cast<modules::ModularLoRAModel<DenseBlock>*>(model);
	            if (lora) {
	                const std::size_t lora_params = lora->lora_num_parameters();
	                const std::size_t lora_bytes_total = modules::lora_bytes(mod_config, mod_lora);
	                printf("LoRA trainable parameters: %zu (%.2f MiB @ %s)\n",
	                       lora_params, (double)lora_bytes_total / 1024.0 / 1024.0, dtype_to_str(mod_lora.dtype));
	                if (lora->qlora_enabled()) {
	                    printf("QLoRA mode active: base weights will be quantized on load\n");
	                }
	            }
	        }
	    } else {
	        model = model_storage.get();
	    }

	    // Allocate run state for the selected model (base or LoRA wrapper)
	    model->allocate_run_state(Options, comm, B, T, /*allocate_optimizer=*/true);

	    if (latest_step >= 0) {
	        auto log = logger.log_section_start(0, fmt::format("Loading checkpoint {} from `{}`", latest_step, CkptDir.c_str()));
	        load_checkpoint(CkptDir, latest_step, *model, &train_loader, comm);
	    } else if (FromScratch) {
	        auto log = logger.log_section_start(0, "Initializing model from scratch");
	        model->init_weights(comm);
	        latest_step = 0;
	    } else {
	        auto log = logger.log_section_start(0, fmt::format("Loading model from `{}`", model_path.c_str()));
	        model->import_weights(model_path, true, comm);
	        latest_step = 0;
	    }

    logger.log_dataset(train_loader, test_loader);

	    // Log allocator stats (modular and legacy)
	    {
	        auto& rs = model->get_run_state();
	        if (rs.Allocator) {
	            logger.log_allocator(rs.Allocator->get_allocation_segments(), rs.Stack.get_allocation_stats());
	        }
	    }

    Tensor inputs = model->get_input_buffer();
    Tensor targets = model->get_target_buffer();
    Tensor position_ids = model->get_position_ids_buffer();

    std::unique_ptr<ISchedule> lr_schedule;
    int end_steps = MaxSteps - CoolDownSteps;
    if (iequals(LrScheduleType, "cosine")) {
        lr_schedule = std::make_unique<CosineSchedule>(LearningRate, end_steps, WarmupSteps, LearningRate * FinalLrFraction);
    } else if (iequals(LrScheduleType, "linear")) {
        lr_schedule = std::make_unique<LinearSchedule>(LearningRate, LearningRate * FinalLrFraction, end_steps, WarmupSteps);
    } else if (iequals(LrScheduleType, "wsd")) {
        lr_schedule = std::make_unique<LinearSchedule>(LearningRate, LearningRate, end_steps, WarmupSteps);
    } else {
        throw std::invalid_argument("Unknown learning rate schedule: " + LrScheduleType);
    }

    logger.log_message(0, fmt::format("Starting training for {} steps ({:.2f} epochs in total)",
        MaxSteps, float(MaxSteps) / steps_per_epoch));
    logger.log_message(0, fmt::format("Setup took {} seconds",
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - BeginStartup).count()));


    for (int step = latest_step; step < MaxSteps; ++step) {
        bool run_eval = false;
        if (!train_loader.has_next(GradAccSteps * B)) {
            train_loader.advance_epoch();
            run_eval = true;
        }
        if (EvalEvery > 0 && step % EvalEvery == 0 && step > 0) {
            run_eval = true;
        }

        if (CkptEvery > 0 && step % CkptEvery == 0 && step > latest_step) {
            auto log = logger.log_section_start(step, fmt::format("saving checkpoint to `{}`", CkptDir.c_str()));
            std::string save_path = save_checkpoint(CkptDir, step, *model, &train_loader, comm);
            if(CkptToKeep > 0) {
                auto cleaned = clean_old_checkpoints(CkptDir, CkptToKeep, MajorCkptEvery);
                logger.log_message(0, fmt::format("Cleaned {} checkpoints", cleaned.size()));
            }
        }

        if (run_eval) {
            run_evaluation(test_loader, *model, logger, train_loader.epoch() + 0.01f * train_loader.progress(), step, comm, EvalNumSteps,
                           inputs, targets, position_ids);
        }

        NvtxRange range("step", step);
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        float lr = lr_schedule->eval(step);
        // learning-rate cooldown
        if (step > end_steps) {
            int cds = step - end_steps;
            float f = static_cast<float>(cds) / static_cast<float>(CoolDownSteps);
            // 1 - sqrt recommended by https://arxiv.org/pdf/2405.18392
            lr = lr_schedule->eval(end_steps) * (1.f - sqrtf(f));
        }

        for (int j = 0; j < GradAccSteps; ++j) {
            train_loader.load_batch(inputs, targets, &position_ids);
            model->forward(inputs, position_ids, comm, j);
            model->backward(inputs, targets, comm, GradAccSteps, j);
        }

        if (LogGPUEvery > 0 && step % LogGPUEvery == 0) {
            logger.log_gpu_state(step, 0, gpu_util->update());
        }

        model->update(comm, lr, Beta1, Beta2, step + 1, Epsilon, WeightDecay, GradClip);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float step_loss = model->get_loss();
        float step_norm = model->get_norm();
        auto [loss_z, grad_z] = model->get_run_state().record_step(step_loss, step_norm);
        if (loss_z > 10.f) {
            logger.log_message(step, fmt::format("Loss outlier with z-score: {}", loss_z));
        }
        if (grad_z > 10.f) {
            logger.log_message(step, fmt::format("Gradient norm outlier with z-score: {}", grad_z));
        }
        logger.log_step(step, train_loader.epoch() + 0.01f*train_loader.progress(), B*T*GradAccSteps*comm.world_size(), narrow<int>(ms), step_norm, step_loss, lr);

        if (Options.TriggerTimingEvents) {
            // timing breakdown (works via IRunState)
            auto& rs = model->get_run_state();
            printf("%s", "\nTiming breakdown:\n");
            for (int i = 0; i < GradAccSteps; ++i) {
                float fwd = 0.0f, bwd = 0.0f, head = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&fwd, rs.TimingForwardStart[i], rs.TimingForwardEnd[i]));
                CUDA_CHECK(cudaEventElapsedTime(&head, rs.TimingHeadStart[i], rs.TimingHeadEnd[i]));
                CUDA_CHECK(cudaEventElapsedTime(&bwd, rs.TimingBackwardStart[i], rs.TimingBackwardEnd[i]));
                // Note: head events are nested in bwd, so subtract times.
                printf("  fwd %7.2fms, head %7.2fms, bwd %7.2fms\n", fwd, head, bwd - head);
            }
            float opt = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&opt, rs.TimingOptimizerStart, rs.TimingOptimizerEnd));
            printf("  opt %7.2fms\n", opt);
            printf("%s", "\n");
        }
    }

    if (FinalEvalNumSteps != 0) {
        int final_batches = FinalEvalNumSteps < 0 ? test_loader.num_chunks() : FinalEvalNumSteps;
        logger.log_message(0, fmt::format("Running final evaluation ({} batches)...", final_batches));
        float loss = run_evaluation(test_loader, *model, logger,
                                    train_loader.epoch() + 0.01f * train_loader.progress(),
                                    MaxSteps, comm, final_batches, inputs, targets, position_ids);
        float accuracy = model->get_accuracy();
        logger.log_message(0, fmt::format("Done. validation loss {:10f}, accuracy {:.2f}%", loss, accuracy));
    } else {
        logger.log_message(0, "Skipping final evaluation.");
    }

    if (SaveFinalModel) {
        auto log = logger.log_section_start(MaxSteps, fmt::format("Saving model to `{}`", OutDir.c_str()));
        std::filesystem::path p(OutDir);
        // Directory creation is safe to run on every rank.
        std::filesystem::create_directories(p);

        if (UseLora) {
            // For LoRA, export adapter weights and config (adapter_model.safetensors + adapter_config.json)
            // Prefer the original `--model` reference for portability (e.g. HF repo id),
            // rather than the resolved local cache path used for loading.
            const std::string base_ref = ModelReference.empty() ? ModelRootPath : ModelReference;

            if (auto* lora = dynamic_cast<modules::ModularLoRAModel<modules::DenseTransformerBlock<>>*>(model)) {
                lora->export_adapter(OutDir, comm, base_ref);
            }
            else {
                // Fallback: save whatever adapter the model exposes (may omit adapter_config.json).
                model->save_lora_checkpoint(OutDir, comm);
            }
        } else {
            // For full model training, export the complete model
            if (comm.rank() == 0) {
                save_pretrained_config(config, (p / "config.json").c_str());
            }
            model->export_weights((p / "model.safetensors").c_str(), comm);
        }

        // copy config files from source model, if we have them and they don't exist already
        // Only rank 0 should touch extra filesystem artifacts to avoid multi-rank races.
        if (comm.rank() == 0 && std::filesystem::exists(ModelRootPath)) {
            auto maybe_copy = [root = std::filesystem::path(ModelRootPath), dest = std::filesystem::path(p)](const char* file_name) {
                const auto src = root / file_name;
                const auto dst = dest / file_name;
                if (!std::filesystem::exists(src)) {
                    return;
                }
                std::error_code ec;
                // Avoid errors when re-running into an existing output directory.
                std::filesystem::copy_file(src, dst, std::filesystem::copy_options::skip_existing, ec);
                if (ec) {
                    // Do not fail training when optional metadata copies fail.
                    std::cerr << "Warning: failed to copy `" << src.string() << "` -> `" << dst.string()
                              << "`: " << ec.message() << std::endl;
                }
            };

            maybe_copy("tokenizer_config.json");
            maybe_copy("generation_config.json");
            maybe_copy("merges.txt");
            maybe_copy("vocab.json");
            maybe_copy("tokenizer.json");
        }
    } else {
        logger.log_message(0, "Skipping final model export (--no-save).");
    }
}

/**
 * @brief Run validation on the provided eval dataloader and log results.
 *
 * @param test_loader Evaluation dataloader; its state is reset to the beginning for the run.
 * @param model Model instance used for forward/validate.
 * @param logger Training run logger used to emit eval events/metrics.
 * @param epoch Fractional epoch value reported to the logger.
 * @param step Global optimizer step used for logging/NVTX range naming.
 * @param comm Communicator used for distributed validation behavior.
 * @param max_steps Maximum number of eval batches to process (0 means no batches; caller controls semantics).
 * @param inputs Pre-allocated input tensor buffer (device) filled by the dataloader.
 * @param targets Pre-allocated target tensor buffer (device) filled by the dataloader.
 * @return Mean validation loss over processed batches (0 if no batches were available).
 */
float run_evaluation(DataLoader& test_loader, IModel& model, TrainingRunLogger& logger, float epoch, int step,
                     NCCLCommunicator& comm, int max_steps, Tensor& inputs, Tensor& targets, Tensor& position_ids) {
    NvtxRange range("validate", step);
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    test_loader.set_state(test_loader.seed(), 0, 0, 0);
    float loss = 0.f;
    std::int64_t total_valid_tokens = 0;
    std::int64_t total_correct_tokens = 0;
    int batches = 0;
    while (test_loader.has_next(div_exact((int)inputs.nelem(), test_loader.seq_len())) && batches < max_steps) {
        test_loader.load_batch(inputs, targets, &position_ids);
        loss += model.validate(inputs, position_ids, targets, comm, batches);

        // Token-weighted accuracy aggregation.
        // `validate()` computes per-token top-1 accuracy and stores counts in the run state;
        // averaging batch percentages can be misleading when the last batch is partial or masks differ.
        auto& rs = model.get_run_state();
        int valid_tokens = 0;
        int correct_tokens = 0;
        CUDA_CHECK(cudaMemcpy(&valid_tokens, rs.ValidTokenCount.Data, sizeof(valid_tokens), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&correct_tokens, rs.CorrectCount.Data, sizeof(correct_tokens), cudaMemcpyDeviceToHost));
        total_valid_tokens += static_cast<std::int64_t>(valid_tokens);
        total_correct_tokens += static_cast<std::int64_t>(correct_tokens);
        batches++;
    }
    static bool warning = true;
    if (warning && batches == 0) {
        std::cerr << "WARNING: insufficient validation data: " << test_loader.num_tokens() << " need at least "
                  << comm.world_size() * test_loader.seq_len() << std::endl;
        warning = false;
        return 0.f;
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    float avg_loss = loss / batches;
    float avg_accuracy = 0.0f;
    if (total_valid_tokens > 0) {
        avg_accuracy = (static_cast<float>(total_correct_tokens) / static_cast<float>(total_valid_tokens)) * 100.0f;
    }

    // Use valid token count for eval throughput reporting (masked tokens excluded).
    logger.log_eval(step, epoch, static_cast<int>(std::min<std::int64_t>(total_valid_tokens, std::numeric_limits<int>::max())), narrow<int>(ms), avg_loss);

    // Store accuracy in model for later retrieval
    // We'll just use the last accuracy value which is the average
    auto& rs = model.get_run_state();
    *rs.AccuracyHost = avg_accuracy;

    return avg_loss;
}

/**
 * @brief Program entry point. Creates a TrainingRunner, parses CLI args, and launches training.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success; nonzero on failure (errors are printed to stderr).
 */
int main(int argc, const char** argv) {
    try {
        TrainingRunner runner;
        runner.load_training_config(argc, argv);
        runner.launch_training(argc, argv);
        return 0;
    } catch (const std::exception& e) {
        ::fprintf(stderr, "ERROR: %s\n", e.what());
        fflush(stderr);
        return EXIT_FAILURE;
    }
}
