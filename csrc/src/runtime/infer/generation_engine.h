// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// GenerationEngine: synchronous native generation for training/inference paths.
//
// Orchestrates prefill + decode loop using the same model weights and GPU
// context as the training engine. No separate process, no weight broadcast.

#ifndef SUROGATE_SRC_RUNTIME_INFER_GENERATION_ENGINE_H
#define SUROGATE_SRC_RUNTIME_INFER_GENERATION_ENGINE_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/infer/kv_cache.h"
#include "runtime/infer/decode_context.h"
#include "runtime/infer/decode_state.h"
#include "runtime/core/forward_hooks.h"

namespace modules {
struct ModelConfig;
}

class NCCLCommunicator;
struct RuntimeOptions;

namespace dsl {
class DslRunState;
class DslParamStore;
class DslGradStore;
class GraphExecutor;
}

namespace infer {

/// Configuration for the generation engine.
struct GenerationEngineConfig {
    int max_gen_len = 512;          // Maximum tokens to generate
    float temperature = 1.0f;       // Sampling temperature (0 = greedy)
    int top_k = 0;                  // Top-K sampling (0 = disabled)
    float top_p = 1.0f;            // Top-P / nucleus sampling (1.0 = disabled)
    float min_p = 0.0f;            // Min-P sampling (0.0 = disabled)
    float repetition_penalty = 1.0f; // Repetition penalty (1.0 = disabled)
    int eos_token_id = 2;           // End-of-sequence token
    uint64_t seed = 42;             // RNG seed for reproducible sampling
    int num_completions = 1;        // N completions per prompt (Phase 3: GRPO multi-completion)
    bool use_cuda_graphs = false;   // Capture decode step as CUDA graph (Phase 4)
    int prefill_chunk_size = 256;   // Chunk size for prompt prefill (0 = disabled)

    // Persistent system prefix: if non-empty, this token sequence is prefilled
    // once and its KV-cache survives across generate() calls via arena prefix
    // boundary. Subsequent calls skip re-prefilling the system prefix.
    std::vector<int32_t> system_prefix_tokens;
};

/// GenerationEngine: prefill + decode loop for batch generation.
///
/// Usage:
///   GenerationEngine engine(run_state, weights, config, options);
///   engine.init(arena);
///   auto trajectories = engine.generate(prompts, graph_executor, comm);
class GenerationEngine {
public:
    GenerationEngine(dsl::DslRunState& run_state,
                     dsl::DslParamStore& weights,
                     const modules::ModelConfig& config,
                     const RuntimeOptions& options);

    /// Generate completions for a batch of prompts (1 completion each).
    /// Uses the paged-KV decode backend with N=1 for consistency.
    ///
    /// @param prompts        [num_prompts][prompt_len] token IDs (host, ragged)
    /// @param gen_config     Generation parameters (num_completions forced to 1)
    /// @param arena          Arena to allocate KV-cache from
    /// @param graph_executor Model graph executor
    /// @param comm           NCCL communicator
    /// @param hook           Optional LoRA forward hook (nullptr = base model)
    /// @return Trajectories [num_prompts] with generated tokens and logprobs
    std::vector<Trajectory> generate(
        const std::vector<std::vector<int32_t>>& prompts,
        const GenerationEngineConfig& gen_config,
        DeviceMemoryStack& arena,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook = nullptr);

    /// Generate M×N completions for GRPO (M prompts × N completions each).
    ///
    /// For each prompt, the KV-cache is prefilled once and broadcast to N
    /// sequence slots. All M×N sequences then decode in parallel. This saves
    /// (N-1) × prefill_cost per prompt compared to naive N× prefill.
    ///
    /// @param prompts        [M][prompt_len] token IDs (host, ragged). M prompts.
    /// @param gen_config     Generation parameters. num_completions = N.
    /// @param arena          Arena to allocate KV-cache from
    /// @param graph_executor Model graph executor
    /// @param comm           NCCL communicator
    /// @param hook           Optional LoRA forward hook (nullptr = base model)
    /// @return Trajectories [M*N] — first N are completions for prompt 0, etc.
    std::vector<Trajectory> generate_grpo(
        const std::vector<std::vector<int32_t>>& prompts,
        const GenerationEngineConfig& gen_config,
        DeviceMemoryStack& arena,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook = nullptr);

private:
    dsl::DslRunState& mRunState;
    dsl::DslParamStore& mWeights;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;
};

/// Incremental generation output for one decode chunk.
struct GenerationSessionStepResult {
    std::vector<std::vector<int32_t>> tokens;  // [batch_size] newly generated IDs for this step chunk
    std::vector<int32_t> completion_lens;      // [batch_size] total generated tokens so far
    std::vector<int32_t> finished;             // [batch_size] 1 if finished, else 0
    bool all_finished = false;
};

/// Persistent decode session for serving continuous scheduling (N=1 completions).
///
/// Lifecycle:
///   init(prompts, cfg, ...)
///   step(k) repeatedly
///   close()
class GenerationSession {
public:
    GenerationSession(dsl::DslRunState& run_state,
                      dsl::DslParamStore& weights,
                      const modules::ModelConfig& config,
                      const RuntimeOptions& options);
    ~GenerationSession();

    void init(const std::vector<std::vector<int32_t>>& prompts,
              const GenerationEngineConfig& gen_config,
              DeviceMemoryStack& arena,
              dsl::GraphExecutor& graph_executor,
              NCCLCommunicator& comm,
              const modules::ForwardHook* hook = nullptr);

    GenerationSessionStepResult step(
        int max_steps,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook = nullptr);

    /// Force-finish specific sequence rows (e.g., stop-string hit on host).
    void mark_finished(const std::vector<int>& rows);

    void close(dsl::GraphExecutor& graph_executor);

    bool initialized() const { return mInitialized; }
    bool all_finished() const { return mAllFinished; }
    int batch_size() const { return mBatchSize; }
    int generated_steps() const { return mGeneratedSteps; }
    int max_gen_len() const { return mGenConfig.max_gen_len; }

private:
    dsl::DslRunState& mRunState;
    dsl::DslParamStore& mWeights;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;

    bool mInitialized = false;
    bool mAllFinished = false;
    DeviceMemoryStack* mArena = nullptr;
    dsl::GraphExecutor* mGraphExecutor = nullptr;
    DeviceMemoryStack::Checkpoint mArenaCheckpoint{};
    GenerationEngineConfig mGenConfig{};
    uint64_t mRngOffset = 0;
    int mGeneratedSteps = 0;
    bool mFullStepCudaGraphEnabled = false;
    bool mFullStepCudaGraphPrimed = false;
    bool mDecodeCudaGraphPrimed = false;
    cudaGraphExec_t mFullStepCudaGraphExec = nullptr;
    DeviceMemoryStack::Checkpoint mFullStepCudaGraphCheckpoint{};

    int mBatchSize = 0;
    int mMaxTotalLen = 0;
    int mVocabSize = 0;
    int32_t mFallbackTokenId = 0;

    PagedKVCache mPagedKV;

    int* mSeqLensGpu = nullptr;
    int32_t* mCuSeqlensQGpu = nullptr;
    int32_t* mSeqlensKGpu = nullptr;
    int32_t* mPositionIdsGpu = nullptr;
    int32_t* mLastTokensGpu = nullptr;
    float* mLogitsGpu = nullptr;
    float* mProbsGpu = nullptr;
    void* mSoftmaxWs = nullptr;
    std::size_t mSoftmaxWsBytes = 0;
    int32_t* mSampledTokensGpu = nullptr;
    float* mLogprobsGpu = nullptr;
    int32_t* mCompletionLensGpu = nullptr;
    int* mActiveCountGpu = nullptr;
    int* mFinishedGpu = nullptr;
    uint64_t* mSamplingOffsetsGpu = nullptr;
    float* mTemperatureGpu = nullptr;
    int* mPrefillBlockTableGpu = nullptr;

    std::vector<int32_t> mPromptLensHost;
    std::vector<int> mSeqLensHost;
    std::vector<int> mFinishedHost;
    std::vector<int32_t> mCompletionLensHost;
    std::vector<int32_t> mSampledTokensHost;
    std::vector<int32_t> mStepSampledTokensHost;
    std::vector<int> mStepFinishedBeforeHost;
    std::vector<uint64_t> mSamplingOffsetsHost;

    infer::DecodeState mDecodeState{};

    std::unordered_map<int, void*> mRecurrentStates;
    std::unordered_map<int, std::size_t> mRecurrentStateBytes;
    std::unordered_map<int, void*> mConvStates;
    std::unordered_map<int, std::size_t> mConvStateBytes;
};

}  // namespace infer

#endif  // SUROGATE_SRC_RUNTIME_INFER_GENERATION_ENGINE_H
