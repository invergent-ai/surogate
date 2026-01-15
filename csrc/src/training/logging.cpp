// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "logging.h"

#include <cmath>
#include <filesystem>
#include <map>
#include <algorithm>

#include <fmt/core.h>
#include <fmt/chrono.h>
#include <cuda_runtime.h>

#include "dataloader.h"
#include "utilities/comm.h"
#include "utilities/gpu_info.h"
#include "utilities/utils.h"
#include "utilities/allocator.h"
#include "utilities/stack.h"
#include "utilities/sol.h"
#include <iostream>

/**
 * @brief Create a logger that writes a JSON array to @p file_name (rank 0 only).
 *
 * On rank 0, ensures the parent directory exists, opens the file for output,
 * and initializes it as a JSON array (writes "[ ... ]").
 *
 * @param file_name Output path for the JSON log.
 * @param rank MPI/NCCL rank; only rank 0 writes the JSON file and prints most output.
 * @param verbosity Verbosity level controlling stdout printing.
 */
TrainingRunLogger::TrainingRunLogger(const std::string& file_name, int rank, EVerbosity verbosity) :
    mFileName(std::move(file_name)), mRank(rank), mVerbosity(verbosity)
{
    if(mRank == 0) {
        auto log_path = std::filesystem::path(mFileName).parent_path();
        if (!log_path.empty()) {
            std::filesystem::create_directories(log_path);
        }
        mLogFile.open(mFileName, std::fstream::out);
        mLogFile << "[\n";
        mLogFile << "\n]\n";
    }
}

/**
 * @brief Destructor; closes the log file if open.
 */
TrainingRunLogger::~TrainingRunLogger()
{
    if(mLogFile.is_open()) mLogFile.close();
}

/**
 * @brief Format a token count as a human-readable short string (k/M/B/T).
 *
 * Uses integer or one-decimal formatting depending on magnitude.
 *
 * @param num_tokens Number of tokens.
 * @return Formatted string (e.g., " 123k", " 1.2M", "  12B").
 */
std::string fmt_token_count(long num_tokens) {
    if(num_tokens < 1'000'000 ) {
        return fmt::format("{:4d}k", num_tokens / 1'000);
    } else if(num_tokens < 20'000'000 ) {
        return fmt::format("{:4.1f}M", float(num_tokens / 1'000) / 1000.f);
    } else if(num_tokens < 1'000'000'000 ) {
        return fmt::format("{:4d}M", num_tokens / 1'000'000);
    } else if(num_tokens < 20'000'000'000 ) {
        return fmt::format("{:4.1f}B", float(num_tokens / 1'000'000) / 1000.f);
    } else if(num_tokens < 1'000'000'000'000 ) {
        return fmt::format("{:4d}B", num_tokens / 1'000'000'000);
    } else if(num_tokens < 20'000'000'000'000 ) {
        return fmt::format("{:4.1f}T", float(num_tokens / 1'000'000'000) / 1000.f);
    } else {
        return fmt::format("{:4d}T", num_tokens / 1'000'000'000'000);
    }
}

/**
 * @brief Create a single JSON log line describing a DataLoader state.
 *
 * Fields include split name, current time, file/token counts, indices, and seed.
 *
 * @param loader DataLoader to summarize.
 * @param split Split name (e.g., "train", "eval").
 * @return JSON object line (as string) ready for insertion into the log array.
 */
std::string format_data_loader(const DataLoader& loader, const char* split) {
    return fmt::format(R"(  {{"log": "dataset", "split": "{}", "time": "{}", "step": 0, "files": {}, "tokens": {}, "file_index": {}, "chunk_index": {}, "seed": {}}})",
        split, std::chrono::system_clock::now(), loader.num_files(), loader.num_tokens(), loader.file_index(), loader.chunk_index(), loader.seed());
}

/**
 * @brief Log dataset metadata for train and eval loaders (rank 0 only).
 *
 * Emits JSON lines and optionally prints a readable summary to stdout.
 *
 * @param train_loader Training DataLoader.
 * @param eval_loader Evaluation DataLoader.
 */
void TrainingRunLogger::log_dataset(const DataLoader& train_loader, const DataLoader& eval_loader) {
    if(mRank != 0) return;
    log_line(format_data_loader(train_loader, "train"));
    log_line(format_data_loader(eval_loader, "eval"));

    if (mVerbosity >= 0) {
        printf("[Dataset]\n");

        printf(" train: %s tokens\n", fmt_token_count(train_loader.num_tokens()).c_str());
        for (int i = 0; i < train_loader.num_files(); ++i) {
            if (i < 10 || mVerbosity >= 1) {
                printf("   %s : %10d\n", train_loader.file_name(i).c_str(), train_loader.file_tokens(i));
            }
        }
        printf(" eval: %s tokens\n", fmt_token_count(eval_loader.num_tokens()).c_str());
        for (int i = 0; i < eval_loader.num_files(); ++i) {
            if (i < 10 || mVerbosity >= 1) {
                printf("   %s : %10d\n", eval_loader.file_name(i).c_str(), eval_loader.file_tokens(i));
            }
        }
        printf("\n");
    }
}

/**
 * @brief Log configuration options (rank 0 only).
 *
 * Each option is written as a JSON log line and optionally printed.
 *
 * @param options Vector of (name, value) pairs; value may be bool, int64, float, or std::string.
 */
void TrainingRunLogger::log_options(const std::vector<std::pair<std::string_view, std::variant<bool, std::int64_t, float, std::string>>>& options) {
    if(mRank != 0) return;

    int option_length = 0;
    for(auto& [name, value]: options) {
        auto log = [&](auto&& v){
            if(std::is_same_v<std::remove_cvref_t<decltype(v)>, std::string>) {
                log_line(fmt::format(R"(  {{"log": "option", "time": "{}", "step": 0, "name": "{}", "value": "{}"}})",
                                     std::chrono::system_clock::now(), name, v));
            } else {
                log_line(fmt::format(R"(  {{"log": "option", "time": "{}", "step": 0, "name": "{}", "value": {}}})",
                                     std::chrono::system_clock::now(), name, v));
            }
        };
        option_length = std::max(option_length, static_cast<int>(name.size()));
        std::visit(log, value);
    }
}

/**
 * @brief Format tokens-per-second as a short string.
 *
 * @param eval_tokens Number of tokens processed.
 * @param duration_ms Duration in milliseconds for processing @p eval_tokens.
 * @return "-" if @p duration_ms is 0, else a right-aligned TPS string (plain or "k").
 */
std::string format_tps(long eval_tokens, long duration_ms) {
    if (duration_ms == 0) {
        return "-";
    }

    long tps = 1000ll * eval_tokens / duration_ms;
    if(tps < 100'000) {
        return fmt::format("{:5}", tps);
    } else {
        return fmt::format("{:4}k", tps / 1000);
    }
}


/**
 * @brief Format a FLOP count for display as M or G.
 *
 * @param flop Total floating-point operations.
 * @return Formatted string with units (" M" or " G").
 */
std::string format_flop(long flop) {
    if(flop > 2'000'000'000) {
        return fmt::format("{:8.1f} G", float(flop) / 1'000'000'000.f);
    } else {
        return fmt::format("{:8.1f} M", float(flop) / 1'000'000.f);
    }
}

/**
 * @brief Format a duration given in milliseconds.
 *
 * @param duration_ms Duration in milliseconds.
 * @return Formatted string in "ms" or "s" depending on magnitude.
 */
std::string format_time(int duration_ms) {
    if (duration_ms >= 100'000) {
        return fmt::format("{:5d}  s", duration_ms / 1000);
    } else {
        return fmt::format("{:5d} ms", duration_ms);
    }
}

/**
 * @brief Log a training step (rank 0 only).
 *
 * Updates running totals used to compute average training loss between evals.
 * Writes a JSON line and optionally prints a compact progress line including TPS
 * and an optional speed-of-light estimate when available.
 *
 * @param step Global step index.
 * @param epoch Fractional epoch progress (used to compute percent-within-epoch).
 * @param step_tokens Tokens processed this step.
 * @param duration_ms Step duration in milliseconds.
 * @param norm Gradient/parameter norm (as reported by training loop).
 * @param loss Training loss for this step.
 * @param lr Learning rate for this step.
 */
void TrainingRunLogger::log_step(int step, float epoch, int step_tokens, int duration_ms, float norm, float loss, float lr)
{
    if(mRank != 0) return;
    mTotalTrainingLoss += loss;
    ++mTotalTrainingSteps;

    if(mVerbosity >= 0) {
        float iptr;
        float progress = 100.f * std::modf(epoch,  &iptr);

        // Tokens per second
        long tps = (duration_ms > 0) ? (1000ll * step_tokens / duration_ms) : 0;

        // Loss trend indicator
        char trend = ' ';
        if (mPreviousLoss > 0) {
            if (loss < mPreviousLoss) {
                trend = '\\';
            } else if (loss > mPreviousLoss) {
                trend = '/';
            }
        }
        mPreviousLoss = loss;

        // Update norm moving average
        if (mNormSampleCount < 100) {
            mNormMovingAverage = (mNormMovingAverage * mNormSampleCount + norm) / (mNormSampleCount + 1);
            ++mNormSampleCount;
        } else {
            // Exponential moving average with alpha=0.1
            mNormMovingAverage = 0.9f * mNormMovingAverage + 0.1f * norm;
        }

        // Check if norm exceeds threshold
        char norm_flag = ' ';
        if (mNormSampleCount >= 10 && mNormMovingAverage > 0 && norm > 5.0f * mNormMovingAverage) {
            norm_flag = '!';
        }

        // Speed of light percentage
        std::string sol_str = "";
        if (mExpectedTimePerToken > 0) {
            long peak = mExpectedTimePerToken * step_tokens / 1'000'000;
            double ratio = static_cast<double>(peak) / static_cast<double>(duration_ms);
            sol_str = fmt::format(" | sol {:4.1f}", ratio * 100.0);
        }

        // ETA calculation
        std::string eta_str = "";
        if (mRemainingTokens > 0 && mTotalTokens > 0) {
            mRemainingTokens -= step_tokens;
            auto elapsed = std::chrono::steady_clock::now() - mTrainingStartTime;
            long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            long tokens_processed = mTotalTokens - mRemainingTokens;

            if (tokens_processed > 0 && elapsed_ms > 0) {
                double tokens_per_ms = static_cast<double>(tokens_processed) / elapsed_ms;
                long eta_ms = static_cast<long>(mRemainingTokens / tokens_per_ms);

                long eta_hours = eta_ms / (1000 * 60 * 60);
                long eta_minutes = (eta_ms % (1000 * 60 * 60)) / (1000 * 60);
                eta_str = fmt::format(" | eta {:02d}h{:02d}m", eta_hours, eta_minutes);
            }
        }

        printf(":: step %7d [%5.1f%%] %c loss %6.4f | norm %c%6.4f | %5.1fk tps | %5d ms%s%s\n",
               step, progress, trend, loss, norm_flag, norm, tps / 1000.0f, duration_ms, sol_str.c_str(), eta_str.c_str());
        fflush(stdout);
    }
    log_line(fmt::format(R"(  {{"log": "step", "time": "{}", "step": {}, "epoch": {}, "step_tokens": {}, "duration_ms": {}, "norm": {}, "loss": {}, "lr": {}}})",
        std::chrono::system_clock::now(), step, epoch, step_tokens, duration_ms, norm, loss, lr ));
}

/**
 * @brief Log a training step with MoE metrics (rank 0 only).
 *
 * Extended version for MoE models that includes router metrics in console output.
 * Updates running totals used to compute average training loss between evals.
 * Writes JSON lines for both step and moe logs.
 *
 * @param step Global step index.
 * @param epoch Fractional epoch progress (used to compute percent-within-epoch).
 * @param step_tokens Tokens processed this step.
 * @param duration_ms Step duration in milliseconds.
 * @param norm Gradient/parameter norm (as reported by training loop).
 * @param loss Training loss for this step.
 * @param lr Learning rate for this step.
 * @param moe_aux_loss MoE auxiliary load balancing loss.
 * @param moe_z_loss MoE router z-loss.
 * @param moe_load_imbalance MoE load imbalance ratio.
 */
void TrainingRunLogger::log_step(int step, float epoch, int step_tokens, int duration_ms, float norm, float loss, float lr,
                                  float moe_aux_loss, float moe_z_loss, float moe_load_imbalance)
{
    if(mRank != 0) return;
    mTotalTrainingLoss += loss;
    ++mTotalTrainingSteps;

    if(mVerbosity >= 0) {
        float iptr;
        float progress = 100.f * std::modf(epoch,  &iptr);

        // Tokens per second
        long tps = (duration_ms > 0) ? (1000ll * step_tokens / duration_ms) : 0;

        // Loss trend indicator
        char trend = ' ';
        if (mPreviousLoss > 0) {
            if (loss < mPreviousLoss) {
                trend = '\\';
            } else if (loss > mPreviousLoss) {
                trend = '/';
            }
        }
        mPreviousLoss = loss;

        // Update norm moving average
        if (mNormSampleCount < 100) {
            mNormMovingAverage = (mNormMovingAverage * mNormSampleCount + norm) / (mNormSampleCount + 1);
            ++mNormSampleCount;
        } else {
            // Exponential moving average with alpha=0.1
            mNormMovingAverage = 0.9f * mNormMovingAverage + 0.1f * norm;
        }

        // Check if norm exceeds threshold
        char norm_flag = ' ';
        if (mNormSampleCount >= 10 && mNormMovingAverage > 0 && norm > 5.0f * mNormMovingAverage) {
            norm_flag = '!';
        }

        // Speed of light percentage
        std::string sol_str = "";
        if (mExpectedTimePerToken > 0) {
            long peak = mExpectedTimePerToken * step_tokens / 1'000'000;
            double ratio = static_cast<double>(peak) / static_cast<double>(duration_ms);
            sol_str = fmt::format(" | sol {:4.1f}", ratio * 100.0);
        }

        // ETA calculation
        std::string eta_str = "";
        if (mRemainingTokens > 0 && mTotalTokens > 0) {
            mRemainingTokens -= step_tokens;
            auto elapsed = std::chrono::steady_clock::now() - mTrainingStartTime;
            long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            long tokens_processed = mTotalTokens - mRemainingTokens;

            if (tokens_processed > 0 && elapsed_ms > 0) {
                double tokens_per_ms = static_cast<double>(tokens_processed) / elapsed_ms;
                long eta_ms = static_cast<long>(mRemainingTokens / tokens_per_ms);

                long eta_hours = eta_ms / (1000 * 60 * 60);
                long eta_minutes = (eta_ms % (1000 * 60 * 60)) / (1000 * 60);
                eta_str = fmt::format(" | eta {:02d}h{:02d}m", eta_hours, eta_minutes);
            }
        }

        // MoE metrics string
        std::string moe_str = fmt::format(" | aux {:.4f} | imbal {:.2f}", moe_aux_loss, moe_load_imbalance);

        printf(":: step %7d [%5.1f%%] %c loss %6.4f | norm %c%6.4f | %5.1fk tps | %5d ms%s%s%s\n",
               step, progress, trend, loss, norm_flag, norm, tps / 1000.0f, duration_ms, moe_str.c_str(), sol_str.c_str(), eta_str.c_str());
        fflush(stdout);
    }
    log_line(fmt::format(R"(  {{"log": "step", "time": "{}", "step": {}, "epoch": {}, "step_tokens": {}, "duration_ms": {}, "norm": {}, "loss": {}, "lr": {}, "moe_aux_loss": {:.6f}, "moe_z_loss": {:.6f}, "moe_load_imbalance": {:.4f}}})",
        std::chrono::system_clock::now(), step, epoch, step_tokens, duration_ms, norm, loss, lr, moe_aux_loss, moe_z_loss, moe_load_imbalance));
}

/**
 * @brief Log an evaluation result (rank 0 only).
 *
 * Prints an eval line (verbosity-dependent), resets accumulated training-loss totals,
 * and writes a JSON line with eval loss and timing.
 *
 * @param step Global step index at which eval was run.
 * @param epoch Fractional epoch progress.
 * @param eval_tokens Tokens processed during evaluation.
 * @param duration_ms Evaluation duration in milliseconds.
 * @param loss Evaluation loss.
 */
void TrainingRunLogger::log_eval(int step, float epoch, int eval_tokens, int duration_ms, float loss)
{
    if(mRank != 0) return;
    if(mVerbosity >= -1) {
        float train_avg = static_cast<float>(mTotalTrainingLoss / std::max(mTotalTrainingSteps, 1));
        float gap = loss - train_avg;

        // Tokens per second
        long tps = (duration_ms > 0) ? (1000ll * eval_tokens / duration_ms) : 0;

        printf("\x1b[1m>> eval                      loss %6.4f | gap %+7.4f | %5.1fk tps | %5d ms\x1b[22m\n",
               loss, gap, tps / 1000.0f, duration_ms);
        fflush(stdout);
    }
    mTotalTrainingLoss = 0;
    mTotalTrainingSteps = 0;
    log_line(fmt::format(R"(  {{"log": "eval", "time": "{}", "step": {}, "epoch": {}, "eval_tokens": {}, "duration_ms": {}, "loss": {}}})",
        std::chrono::system_clock::now(), step, epoch, eval_tokens, duration_ms, loss ));
}

/**
 * @brief Log MoE training statistics (rank 0 only).
 *
 * Writes metrics specific to Mixture-of-Experts models to help monitor router health
 * and expert utilization during training.
 *
 * @param step Global step index.
 * @param aux_loss Load balancing auxiliary loss (summed across layers).
 * @param z_loss Router z-loss for regularization (summed across layers).
 * @param expert_utilization Fraction of experts receiving tokens (0-1 average across layers).
 * @param load_imbalance Ratio of max to mean token counts per expert (1.0 = perfect balance).
 */
void TrainingRunLogger::log_moe_stats(int step, float aux_loss, float z_loss, float expert_utilization, float load_imbalance)
{
    if (mRank != 0) return;
    log_line(fmt::format(R"(  {{"log": "moe", "time": "{}", "step": {}, "aux_loss": {:.6f}, "z_loss": {:.6f}, "expert_utilization": {:.4f}, "load_imbalance": {:.4f}}})",
        std::chrono::system_clock::now(), step, aux_loss, z_loss, expert_utilization, load_imbalance));
}

/**
 * @brief Log instantaneous GPU utilization/telemetry for a given step.
 *
 * Writes a JSON line and optionally prints a short human-readable summary.
 *
 * @param step Global step index.
 * @param gpu_id Device index being reported.
 * @param gpu_util Sampled utilization/telemetry metrics.
 */
void TrainingRunLogger::log_gpu_state(int step, int gpu_id, const GPUUtilInfo& gpu_util)
{
    log_line(fmt::format(R"(  {{"log": "gpu", "time": "{}", "step": {}, "id": {}, "clock": {}, "max_clock": {}, "fan": {}, "power": {}, "power_limit": {}, "temperature": {}, "temp_slowdown": {}, "gpu_util": {}, "mem_util": {}, "throttle": "{}", "dram_free": {}, "pcie_rx": {}, "pcie_tx": {}}})",
       std::chrono::system_clock::now(), step, gpu_id, gpu_util.clock, gpu_util.max_clock, gpu_util.fan,
       gpu_util.power, gpu_util.power_limit, gpu_util.temperature, gpu_util.temp_slowdown, gpu_util.gpu_utilization,
       gpu_util.mem_utilization, gpu_util.throttle_reason, gpu_util.mem_free, gpu_util.pcie_rx, gpu_util.pcie_tx ));
    if(mVerbosity >= 1) {
        printf("[G] step %5d [gpu %2d] | power %4d W   | temp %3d /%3d°C | clock %4d MHz\n",
               step, gpu_id, gpu_util.power / 1000, gpu_util.temperature, gpu_util.temp_slowdown, gpu_util.clock);
        printf("Mem Usage %ld MiB | PCI↓%5d MiB/s| PCI↑%5d MiB/s\n", 
            static_cast<long>(gpu_util.mem_used / 1024 / 1024),
            static_cast<int>(gpu_util.pcie_rx / 1024 / 1024),  static_cast<int>(gpu_util.pcie_tx / 1024 / 1024));
    }
}

/**
 * @brief Collect and log GPU model/state information across all ranks.
 *
 * Each rank gathers its CUDA device properties and memory information, then rank 0
 * logs one JSON line per rank/device and optionally prints a summary.
 *
 * @param comm NCCL communicator used to gather per-rank GPU info to host.
 */
void TrainingRunLogger::log_gpu_model(NCCLCommunicator& comm)
{
    struct sGPUInfoMessage {
        std::chrono::system_clock::time_point time;
        int rank;
        int device_id;
        cudaDeviceProp prop;
        int driver_version;
        int runtime_version;
        std::size_t mem_free;
        std::size_t mem_total;
        std::size_t mem_reserved;
    } msg;

    msg.time = std::chrono::system_clock::now();
    msg.rank = comm.rank();
    CUDA_CHECK(cudaGetDevice(&msg.device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&msg.prop, msg.device_id));
    CUDA_CHECK(cudaDriverGetVersion( &msg.driver_version ));
    CUDA_CHECK(cudaRuntimeGetVersion( &msg.runtime_version ));
    CUDA_CHECK(cudaMemGetInfo(&msg.mem_free, &msg.mem_total));
    msg.mem_reserved = get_mem_reserved();

    auto all_gpus = comm.host_gather(msg);
    if(mRank == 0) {
        for (auto& d: all_gpus) {
            std::string uuid;
            for (char& byte: d.prop.uuid.bytes) {
                uuid += fmt::format("{:02x}", byte);
            }
            std::string line =
                fmt::format(
                    R"(  {{"log": "gpu-model", "time": "{}", "rank": {}, "step": 0, "id": {}, "name": "{}", "l2_size": {}, "sm_count": {}, "major": {}, "minor": {}, "memory": {}, "free": {}, "reserved": {}, "uuid": "{}", "ecc": {}, "shared_mem": {}, "cuda_driver": {}, "cuda_runtime": {}}})",
                    d.time, d.rank, d.device_id, d.prop.name, d.prop.l2CacheSize, d.prop.multiProcessorCount, d.prop.major,
                    d.prop.minor, d.prop.totalGlobalMem, d.mem_free, d.mem_reserved, uuid,
                    d.prop.ECCEnabled, d.prop.sharedMemPerMultiprocessor, d.driver_version, d.runtime_version
                );
            log_line(line);

            if(mVerbosity >= 1 || (mVerbosity >= 0 && d.rank == 0)) {
                printf("[System %d]\n", d.rank);
                printf("  Device %d: %s\n", d.device_id, d.prop.name);
                printf("  CUDA version: driver %d, runtime %d\n", d.driver_version, d.runtime_version);
                printf("  Memory: %zu MiB / %zu MiB\n", (d.mem_total-d.mem_free) / 1024 / 1024, d.mem_total / 1024 / 1024);
                printf("\n");
            }
        }
    }
}

/**
 * @brief Log the command line used to start the run (rank 0 only).
 *
 * Writes a JSON line containing argv as an array of strings.
 *
 * @param argc Argument count.
 * @param argv Argument vector; expected to be @p argc entries.
 */
void TrainingRunLogger::log_cmd(int argc, const char** argv)
{
    if(mRank != 0) return;
    std::string cmd = fmt::format(R"(  {{"log": "cmd", "time": "{}", "step": 0, "cmd": [)", std::chrono::system_clock::now());
    for (int i = 0; i < argc; i++)
    {
        if (i != 0) cmd += ", ";
        cmd += fmt::format("\"{}\"", argv[i]);
    }
    cmd += "]}";
    log_line(cmd);
}

/**
 * @brief Log a speed-of-light (SOL) throughput estimate and peak rates.
 *
 * Computes an expected time-per-token (ns/token) based on GPU model and operation mix,
 * converts to tokens/s, prints a summary (verbosity-dependent), writes a JSON log line,
 * and stores the expected time-per-token for later step-time SOL reporting.
 *
 * @param ops Vector of (dtype, flop) pairs describing operation mix; expected layout:
 *            [0]=blocks, [1]=lm_head, [2]=attention.
 * @param world_size Number of ranks; used to scale per-token estimate across distributed setup.
 */
void TrainingRunLogger::log_sol_estimate(std::vector<std::pair<ETensorDType, long>> ops, int world_size) {
    if (mRank != 0) return;
    if (ops.size() < 3) return;

    auto gpu_name = get_gpu_name();

    // Check if FP8 is being used (blocks dtype is the main indicator)
    bool uses_fp8 = (ops[0].first == ETensorDType::FP8_E4M3 || ops[0].first == ETensorDType::FP8_E5M2);

    // Get spec sheet peaks first (before benchmark updates them)
    const float spec_bf16_peak = get_peak_rate(gpu_name.c_str(), ETensorDType::BF16);
    const float spec_fp8_peak = get_peak_rate(gpu_name.c_str(), ETensorDType::FP8_E4M3);

    // Run appropriate benchmark to measure real peak - this updates the cached measured value
    // used by get_peak_rate() and estimate_speed_of_light()
    double true_bf16_rate = 0.0;
    double true_fp8_rate = 0.0;
    // if (spec_bf16_peak > 0.0f) {
    //     true_bf16_rate = measure_real_peak();
    // }
    // if (uses_fp8 && spec_fp8_peak > 0.0f) {
    //     true_fp8_rate = measure_real_peak_fp8();
    // }

    // Now compute SOL estimate using measured values via get_peak_rate()
    const long estimate_ns = estimate_speed_of_light(gpu_name.c_str(), ops);
    long ns_per_token = -1;
    if (estimate_ns > 0 && world_size > 0) {
        ns_per_token = estimate_ns / world_size;
        if (ns_per_token <= 0) {
            ns_per_token = -1;
        }
    }

    long tps = -1;
    if (ns_per_token > 0) {
        tps = 1'000'000'000l / ns_per_token;
    }

    const float tf32_peak = get_peak_rate(gpu_name.c_str(), ETensorDType::FP32);
    const float bf16_peak = get_peak_rate(gpu_name.c_str(), ETensorDType::BF16);
    const float fp16_peak = get_peak_rate(gpu_name.c_str(), ETensorDType::FP16);
    const float fp8_peak  = get_peak_rate(gpu_name.c_str(), ETensorDType::FP8_E4M3);

    if (mVerbosity >= 0) {
        printf("%s", "[Speed of Light]\n");

        auto log_speed_if_needed = [&](ETensorDType dtype, const char* name){
            if(ops[0].first == dtype || ops[1].first == dtype || ops[2].first == dtype) {
                auto rate = get_peak_rate(gpu_name.c_str(), dtype);
                if(rate > 0) {
                    printf("  Peak %s: %8.1f TFLOP/s\n", name, rate);
                }
            }
        };

        log_speed_if_needed(ETensorDType::FP32, "TF32");
        log_speed_if_needed(ETensorDType::BF16, "BF16");
        log_speed_if_needed(ETensorDType::FP16, "FP16");
        log_speed_if_needed(ETensorDType::FP8_E4M3, " FP8");
        log_speed_if_needed(ETensorDType::FP4_E2M1, " FP4");

        // Show benchmark results for the dtype being used
        if (uses_fp8 && spec_fp8_peak > 0.0f && true_fp8_rate > 0.0) {
            float rate = static_cast<float>((true_fp8_rate / 1e12) / spec_fp8_peak);
            if(rate < 0.85) {
                printf("  \033[31;1mBenchmark:  %6.1f%%         of spec sheet (FP8)\033[0m\n", rate * 100);
            } else {
                printf("  Benchmark:  %6.1f%%         of spec sheet (FP8)\n", rate * 100);
            }
        } else if (spec_bf16_peak > 0.0f && true_bf16_rate > 0.0) {
            float rate = static_cast<float>((true_bf16_rate / 1e12) / spec_bf16_peak);
            if(rate < 0.85) {
                printf("  \033[31;1mBenchmark:  %6.1f%%         of spec sheet\033[0m\n", rate  * 100);
            } else {
                printf("  Benchmark:  %6.1f%%         of spec sheet\n", rate  * 100);
            }
        }

        printf("  Blocks:    %sFLOP   in %s\n", format_flop(ops[0].second).c_str(), dtype_to_str(ops[0].first));
        printf("  LM-Head:   %sFLOP   in %s\n", format_flop(ops[1].second).c_str(), dtype_to_str(ops[1].first));
        printf("  Attention: %sFLOP   in %s\n", format_flop(ops[2].second).c_str(), dtype_to_str(ops[2].first));
        if (tps > 0) {
            printf("  SOL:       %8ld tok/s\n", tps);
        } else {
            printf("  SOL:       unavailable\n");
        }
        printf("%s", "\n");
    }

    log_line(fmt::format(R"(  {{"log": "sol", "time": "{}", "rank": {}, "step": {}, "blocks": {}, "lm_head": {}, "attention": {}, "tps": {}, "tf32_peak": {}, "bf16_peak": {}, "fp16_peak": {}, "fp8_peak": {}}})",
                         std::chrono::system_clock::now(), mRank, 0, ops[0].second, ops[1].second, ops[2].second, tps,
                         tf32_peak, bf16_peak, fp16_peak, fp8_peak));

    mExpectedTimePerToken = ns_per_token;
}

/**
 * @brief Append one JSON object line to the log file (and emit callback if set).
 *
 * The file is maintained as a valid JSON array by seeking near the end and
 * overwriting the array closing tokens. The caller should pass a complete JSON
 * object (no trailing comma).
 *
 * @param line JSON object line to append.
 */
void TrainingRunLogger::log_line(std::string_view line) {
    if(mCallback)
        mCallback(line);

    mLogFile.seekp(-3, std::ios::end);  // overwrite the array closing part
    if (!mFirst)
    {
        mLogFile << ",\n";
    }
    mLogFile << line << "\n]" << std::endl;
    mFirst = false;
}

namespace {
/**
 * @brief Print detailed memory breakdown for raining optimization.
 *
 * Analyzes memory usage by category and compares against Unsloth's baseline.
 */
void print_memory_breakdown(
    const MemoryBreakdownContext& ctx,
    const std::vector<std::pair<std::string, sSegmentMemory>>& stats,
    const std::vector<std::pair<std::string, long>>& stack_stats)
{
    printf("\n");
    printf("================================================================================\n");
    printf("                              MEMORY BREAKDOWN \n");
    printf("================================================================================\n\n");

    // Query CUDA memory stats
    std::size_t cuda_free = 0, cuda_total = 0;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    const std::size_t cuda_used = cuda_total - cuda_free;

    // Print allocator segment summary
    printf("[Allocator Segments]\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("%-35s %12s %12s %12s\n", "Segment", "Device", "Managed", "Pinned");
    printf("--------------------------------------------------------------------------------\n");
    for (const auto& [name, amount]: stats) {
        printf("%-35s %12.1f %12.1f %12.1f\n", name.c_str(),
               amount.OnDevice / 1024.0 / 1024.0,
               amount.Managed / 1024.0 / 1024.0,
               amount.PinnedHost / 1024.0 / 1024.0);
    }
    printf("--------------------------------------------------------------------------------\n");
    printf("\n");

    // Get per-tensor stats for detailed breakdown
    auto tensor_stats = ctx.allocator->get_tensor_stats();

    // Categorize tensors
    struct CategoryInfo {
        std::size_t bytes = 0;
        std::vector<std::pair<std::string, std::size_t>> items;
    };
    std::map<std::string, CategoryInfo> categories;
    std::size_t total_device = 0;

    auto categorize = [](const std::string& name) -> std::string {
        // QLoRA-specific buffers
        if (name.find("dequant") != std::string::npos ||
            name.find("fp8_quant") != std::string::npos ||
            name.find("fp4_quant") != std::string::npos) {
            return "QLoRA Dequant Buffers";
        }
        if (name.find("qlora") != std::string::npos ||
            name.find("fp8_weight") != std::string::npos ||
            name.find("fp4_weight") != std::string::npos ||
            name.find("_nf4") != std::string::npos ||
            name.find("_absmax") != std::string::npos ||
            name.find("_scales") != std::string::npos) {
            return "QLoRA Quantized Weights";
        }
        // LoRA adapters
        if (name.find("lora_") != std::string::npos) {
            return "LoRA Adapters";
        }
        // Activations (per-layer tensors)
        if (name == "ln1" || name == "ln2" || name == "qkv" || name == "att" ||
            name == "att_out" || name == "residual_att" || name == "mlp_up" ||
            name == "swiglu" || name == "mlp_down" || name == "lse" ||
            name.find("_rstd") != std::string::npos ||
            name.find("_shared") != std::string::npos) {
            return "Activations (per-layer)";
        }
        // Gradients
        if (name.find("d_") == 0 || name.find("grad") != std::string::npos) {
            return "Gradients";
        }
        // Optimizer state
        if (name.find("opt_") == 0 || name.find("momentum") != std::string::npos ||
            name.find("variance") != std::string::npos) {
            return "Optimizer State";
        }
        // Model weights (embeddings, projections)
        if (name.find("embed") != std::string::npos || name.find("wte") != std::string::npos ||
            name.find("lm_head") != std::string::npos || name.find("_proj") != std::string::npos ||
            name.find("_weight") != std::string::npos || name.find("_bias") != std::string::npos) {
            return "Model Weights";
        }
        // Workspace/temporary buffers
        if (name.find("workspace") != std::string::npos || name.find("_ws") != std::string::npos ||
            name.find("tmp") != std::string::npos || name.find("buffer") != std::string::npos) {
            return "Workspace/Temp";
        }
        return "Other";
    };

    for (const auto& [name, bytes] : tensor_stats) {
        if (bytes == 0) continue;
        total_device += bytes;
        std::string category = categorize(name);
        categories[category].bytes += bytes;
        if (bytes >= 1024 * 1024) {  // Track items >= 1MB
            categories[category].items.emplace_back(name, bytes);
        }
    }

    // Print category summary
    printf("Memory by Category:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("%-35s %12s   %s\n", "Category", "Size (MiB)", "Top Tensors");
    printf("--------------------------------------------------------------------------------\n");

    std::vector<std::pair<std::string, CategoryInfo*>> sorted_cats;
    for (auto& [cat, info] : categories) {
        sorted_cats.emplace_back(cat, &info);
    }
    std::sort(sorted_cats.begin(), sorted_cats.end(),
              [](const auto& a, const auto& b) { return a.second->bytes > b.second->bytes; });

    for (const auto& [cat, info] : sorted_cats) {
        printf("%-35s %12.1f", cat.c_str(), info->bytes / 1024.0 / 1024.0);

        // Show top 3 tensors in this category
        if (!info->items.empty()) {
            printf("   ");
            int count = 0;
            for (const auto& [name, bytes] : info->items) {
                if (count++ >= 3) break;
                if (count > 1) printf(", ");
                printf("%s(%.0fM)", name.c_str(), bytes / 1024.0 / 1024.0);
            }
        }
        printf("\n");
    }
    printf("--------------------------------------------------------------------------------\n");
    printf("%-35s %12.1f\n", "TOTAL DEVICE MEMORY", total_device / 1024.0 / 1024.0);
    printf("\n");

    // QLoRA-specific breakdown
    if (ctx.qlora_quantized_bytes > 0) {
        printf("QLoRA-Specific Memory:\n");
        printf("  Quantized base weights:     %8.1f MiB\n",
               ctx.qlora_quantized_bytes / 1024.0 / 1024.0);
        printf("  Memory savings ratio:       %8.1fx vs FP16\n",
               ctx.qlora_savings_ratio);
        printf("\n");
    }

    // Stack breakdown (temporary allocations during forward/backward)
    if (!stack_stats.empty()) {
        printf("Stack Allocations (high-water mark):\n");
        printf("--------------------------------------------------------------------------------\n");
        std::size_t stack_total = 0;
        for (const auto& [name, bytes] : stack_stats) {
            stack_total += bytes;
            if (bytes >= 1024 * 1024) {  // Show items >= 1MB
                printf("  %-40s %10.1f MiB\n", name.c_str(), bytes / 1024.0 / 1024.0);
            }
        }
        printf("--------------------------------------------------------------------------------\n");
        printf("  %-40s %10.1f MiB\n", "STACK TOTAL", stack_total / 1024.0 / 1024.0);
        printf("\n");
    }

    // Detailed tensor list (top 20 largest)
    printf("Top 20 Largest Tensors:\n");
    printf("--------------------------------------------------------------------------------\n");
    int count = 0;
    for (const auto& [name, bytes] : tensor_stats) {
        if (bytes < 1024 * 1024 || count++ >= 20) break;
        printf("  %-40s %10.1f MiB\n", name.c_str(), bytes / 1024.0 / 1024.0);
    }
    printf("\n");

    // Model info
    const std::size_t C = ctx.hidden_size;
    const std::size_t L = ctx.num_layers;
    const std::size_t D = ctx.intermediate_size;
    const int B = ctx.batch_size;
    const int T = ctx.seq_length;

    printf("Model Configuration:\n");
    printf("  Hidden size (C):            %8zu\n", C);
    printf("  Intermediate size (D):      %8zu\n", D);
    printf("  Num layers (L):             %8zu\n", L);
    printf("  Batch size (B):             %8d\n", B);
    printf("  Sequence length (T):        %8d\n", T);
    printf("\n");

    // Theoretical memory estimates
    const std::size_t activation_per_layer_bf16 = B * T * (
        C +       // ln1
        C +       // ln2
        C * 3 +   // qkv (approx, depends on GQA)
        C +       // att
        C +       // att_out
        C +       // residual_att
        2 * D +   // mlp_up (gate + up)
        D +       // swiglu
        C         // mlp_down
    ) * 2;  // BF16 = 2 bytes

    printf("Theoretical Activation Memory (BF16, no sharing):\n");
    printf("  Per layer:                  %8.1f MiB\n", activation_per_layer_bf16 / 1024.0 / 1024.0);
    printf("  All %zu layers:              %8.1f MiB\n", L, (activation_per_layer_bf16 * L) / 1024.0 / 1024.0);
    printf("\n");

    // CUDA Memory Breakdown (tracked vs untracked)
    printf("CUDA Memory Analysis:\n");
    printf("--------------------------------------------------------------------------------\n");
    const double cuda_used_mib = cuda_used / 1024.0 / 1024.0;
    const double tracked_mib = total_device / 1024.0 / 1024.0;
    const double untracked_mib = cuda_used_mib - tracked_mib;
    printf("  CUDA total used (nvidia-smi): %8.1f MiB\n", cuda_used_mib);
    printf("  Tracked by TensorAllocator:   %8.1f MiB\n", tracked_mib);
    printf("  Untracked CUDA overhead:      %8.1f MiB (%.1f%%)\n",
           untracked_mib, (untracked_mib / cuda_used_mib) * 100.0);
    printf("\n");
    printf("  Untracked memory breakdown (estimated):\n");
    printf("    - CUDA context:             ~200-400 MiB\n");
    printf("    - cuDNN handles/workspace:  ~200-500 MiB\n");
    printf("    - cuBLAS handles:           ~50-100 MiB\n");
    printf("    - Memory fragmentation:     variable\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("\n");

    // Optimization suggestions based on actual memory usage
    printf("Optimization Suggestions:\n");
    bool has_suggestions = false;
    if (categories["Activations (per-layer)"].bytes > 2ULL * 1024 * 1024 * 1024) {
        printf("  - High activation memory (%.0f MiB): try --recompute-block or --recompute-ffn\n",
               categories["Activations (per-layer)"].bytes / 1024.0 / 1024.0);
        has_suggestions = true;
    }
    if (categories["Gradients"].bytes > 1ULL * 1024 * 1024 * 1024) {
        printf("  - High gradient memory (%.0f MiB): try --offload-gradients or --shard-gradients\n",
               categories["Gradients"].bytes / 1024.0 / 1024.0);
        has_suggestions = true;
    }
    if (categories["Optimizer State"].bytes > 1ULL * 1024 * 1024 * 1024) {
        printf("  - High optimizer memory (%.0f MiB): try --offload-optimizer\n",
               categories["Optimizer State"].bytes / 1024.0 / 1024.0);
        has_suggestions = true;
    }
    if (categories["QLoRA Dequant Buffers"].bytes > 500ULL * 1024 * 1024) {
        printf("  - Large dequant buffers (%.0f MiB): consider fusing dequant into matmul kernels\n",
               categories["QLoRA Dequant Buffers"].bytes / 1024.0 / 1024.0);
        has_suggestions = true;
    }
    if (categories["Model Weights"].bytes > 1ULL * 1024 * 1024 * 1024 && ctx.qlora_quantized_bytes > 0) {
        printf("  - Non-quantized weights detected (%.0f MiB): verify QLoRA is enabled correctly\n",
               categories["Model Weights"].bytes / 1024.0 / 1024.0);
        has_suggestions = true;
    }
    if (!has_suggestions) {
        printf("  (No major optimization opportunities detected)\n");
    }

    printf("\n================================================================================\n\n");
}
} // anonymous namespace

/**
 * @brief Log allocator statistics and (optionally) print a readable summary (rank 0 only).
 *
 * @param stats Per-allocator segment memory statistics keyed by name.
 * @param stack_info Additional stack-related memory info keyed by name.
 */
void TrainingRunLogger::log_allocator(
        const std::vector<std::pair<std::string, sSegmentMemory>>& stats,
        const std::vector<std::pair<std::string, long>>& stack_info)
{
    // Call the extended version with disabled breakdown
    MemoryBreakdownContext ctx{};
    ctx.enabled = false;
    log_allocator(stats, stack_info, ctx);
}

void TrainingRunLogger::log_allocator(
        const std::vector<std::pair<std::string, sSegmentMemory>>& stats,
        const std::vector<std::pair<std::string, long>>& stack_info,
        const MemoryBreakdownContext& breakdown_ctx)
{
    if (mRank != 0) return;
    std::string stat_str = "[";
    bool first = true;
    for (auto& [name, amount]: stats) {
        if (!first) stat_str += ", ";
        first = false;
        stat_str += fmt::format("{{\"name\": \"{}\", \"device\": {}, \"managed\": {}, \"pinned\": {}, \"pageable\": {}}}",
                                name, amount.OnDevice, amount.Managed, amount.PinnedHost, amount.PageableHost);
    }
    stat_str += "]";
    std::string line = fmt::format(R"(  {{"log": "allocator", "time": "{}", "step": 0, "stats": {}}})", std::chrono::system_clock::now(), stat_str);
    log_line(line);

    // Print detailed memory breakdown if enabled (includes [Allocator State])
    if (breakdown_ctx.enabled && breakdown_ctx.allocator) {
        print_memory_breakdown(breakdown_ctx, stats, stack_info);
    }
}


void TrainingRunLogger::log_abs_maxes(int step, const std::vector<std::pair<std::string, float>>& abs_maxes) {
    if (mRank != 0) return;
    std::string abs_maxes_str = "[\n          ";
    int count = 0;
    for (auto& [name, max]: abs_maxes) {
        if (count != 0) abs_maxes_str += ", ";
        ++count;
        if (count % 10 == 0) {
            abs_maxes_str += "\n          ";
        }
        abs_maxes_str += fmt::format(R"({{"name": "{}", "value": {}}})", name, max);
    }
    abs_maxes_str += "]";

    if (mVerbosity >= 1) {
        printf("[Abs Maxes]\n");
        for (auto& [name, max]: abs_maxes) {
            printf("  %16s: %10.4f \n", name.c_str(), max);
        }
        printf("\n");
    }

    std::string line = fmt::format(R"(  {{"log": "abs-maxes", "time": "{}", "step": {}, "abs_maxes": {}}})",
        std::chrono::system_clock::now(), step, abs_maxes_str);
    log_line(line);
}

/**
 * @brief Set a callback invoked for each JSON log line before file append.
 *
 * @param cb Callback taking the JSON line as a string_view; may be empty/null.
 */
void TrainingRunLogger::set_callback(std::function<void(std::string_view)> cb) {
    mCallback = std::move(cb);
}

/**
 * @brief Set the total training tokens for ETA calculation.
 *
 * @param total_tokens Total number of tokens to process during training.
 */
void TrainingRunLogger::set_training_tokens(long total_tokens) {
    mTotalTokens = total_tokens;
    mRemainingTokens = total_tokens;
    mTrainingStartTime = std::chrono::steady_clock::now();
}

/**
 * @brief Log an informational message (rank 0 only).
 *
 * Prints to stdout (verbosity-dependent) and writes a JSON "info" record.
 *
 * @param step Step associated with this message.
 * @param msg Message text.
 */
void TrainingRunLogger::log_message(int step, const std::string& msg) {
    if(mRank != 0) return;
    if(mVerbosity >= 0) {
        fprintf(stdout, "%s\n", msg.c_str());
    }
    log_line(fmt::format(R"(  {{"log": "info", "time": "{}", "step": {}, "message": "{}"}})",
                         std::chrono::system_clock::now(), step, msg ));
}

/**
 * @brief Begin a timed logging section (rank 0 only).
 *
 * Stores section metadata in the logger and returns an RAII handle that will
 * call log_section_end() on destruction (when constructed with a valid logger).
 *
 * @param step Step associated with this section.
 * @param info Human-readable description printed to stdout and stored in JSON.
 * @return RAII_Section handle; on non-zero ranks, contains nullptr and is a no-op.
 */
TrainingRunLogger::RAII_Section TrainingRunLogger::log_section_start(int step, const std::string& info) {
    if(mRank != 0) return RAII_Section{nullptr};
    mSectionInfo = info;
    mSectionStep = step;
    mSectionStart = std::chrono::steady_clock::now();
    if(mVerbosity >= 0) {
        printf("%s ...\n", info.data());
    }
    return RAII_Section{this};
}

/**
 * @brief End the current timed section and emit its duration (rank 0 only).
 *
 * Computes elapsed time since log_section_start(), writes a JSON "info" record
 * including duration_ms, and prints a completion line (verbosity-dependent).
 */
void TrainingRunLogger::log_section_end() {
    auto duration = std::chrono::steady_clock::now() - mSectionStart;
    long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    if(mRank != 0) return;
    log_line(fmt::format(R"(  {{"log": "info", "time": "{}", "step": {}, "message": "{}", "duration_ms": {}}})",
                         std::chrono::system_clock::now(), mSectionStep, mSectionInfo, milliseconds ));

    if(mVerbosity >= 0) {
        if(milliseconds < 2000) {
            printf("  done in %ld ms\n\n", milliseconds);
        } else {
            printf("  done in %ld s\n\n", milliseconds / 1000);
        }
    }
}
