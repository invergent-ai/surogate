// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_TRAINING_LOGGING_H
#define SUROGATE_TRAINING_LOGGING_H

#include <fstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <functional>
#include <chrono>

struct GPUUtilInfo;
struct sSegmentMemory;
class NCCLCommunicator;
class DataLoader;
class TensorAllocator;
enum class ETensorDType : int;

/**
 * @brief Context for detailed memory breakdown analysis (QLoRA optimization).
 */
struct MemoryBreakdownContext {
    const TensorAllocator* allocator = nullptr;  ///< Allocator for tensor stats
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_layers = 0;
    int batch_size = 0;
    int seq_length = 0;
    std::size_t qlora_quantized_bytes = 0;  ///< QLoRA quantized weight bytes (0 if not QLoRA)
    float qlora_savings_ratio = 0.0f;       ///< QLoRA memory savings ratio (0 if not QLoRA)
    bool enabled = false;                    ///< Whether to print detailed breakdown
};

class TrainingRunLogger
{
public:
    enum EVerbosity {
        SILENT = -2,
        QUIET = -1,
        DEFAULT = 0,
        VERBOSE = 1
    };

    TrainingRunLogger(const std::string& file_name, int rank, EVerbosity verbosity);
    ~TrainingRunLogger();

    void log_sol_estimate(std::vector<std::pair<ETensorDType, long>> ops, int world_size);
    void set_callback(std::function<void(std::string_view)> cb);

    void log_cmd(int argc, const char** argv);
    void log_options(const std::vector<std::pair<std::string_view, std::variant<bool, std::int64_t, float, std::string>>>& options);
    void log_gpu_model(NCCLCommunicator& comm);
    void log_dataset(const DataLoader& train_loader, const DataLoader& eval_loader);
    void log_step(int step, float epoch, int step_tokens, int duration_ms, float norm, float loss, float lr);
    void log_eval(int step, float epoch, int eval_tokens, int duration_ms, float loss);
    void log_gpu_state(int step, int gpu_id, const GPUUtilInfo& gpu_util);
    void log_allocator(
        const std::vector<std::pair<std::string, sSegmentMemory>>& stats,
        const std::vector<std::pair<std::string, long>>& stack_info
        );
    void log_allocator(
        const std::vector<std::pair<std::string, sSegmentMemory>>& stats,
        const std::vector<std::pair<std::string, long>>& stack_info,
        const MemoryBreakdownContext& breakdown_ctx
        );
    void log_abs_maxes(int step, const std::vector<std::pair<std::string, float>>& abs_maxes);

    // call at the beginning and end of a section of processing.
    // will record the time between the two calls
    class RAII_Section {
    public:
        ~RAII_Section() noexcept {
            if(mLogger)
                mLogger->log_section_end();
        };
    private:
        RAII_Section(TrainingRunLogger* l) : mLogger(l) {}
        RAII_Section(RAII_Section&&) = default;
        TrainingRunLogger* mLogger;

        friend class TrainingRunLogger;
    };

    void log_message(int step, const std::string& msg);
    RAII_Section log_section_start(int step, const std::string& info);
    void log_section_end();
private:
    void log_line(std::string_view line);
    std::string mFileName;
    std::fstream mLogFile;
    bool mFirst = true;

    int mRank;
    EVerbosity mVerbosity;

    // running mean for training loss
    double mTotalTrainingLoss = 0.0;
    int mTotalTrainingSteps = 0;

    // to estimate ETA
    int mRemainingTokens = -1;

    // to estimate MFU
    long mExpectedTimePerToken = -1;

    // arbitrary callback for log lines
    std::function<void(std::string_view)> mCallback;

    // log section is a two-step process, here we safe intermediaries
    std::string mSectionInfo;
    int mSectionStep;
    std::chrono::steady_clock::time_point mSectionStart;
};

#endif //SUROGATE_TRAINING_LOGGING_H
