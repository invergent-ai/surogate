// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DecodeExecutor: executes a single decode step through the model graph with KV-cache.
//
// Design: reuses the existing compiled forward graph but intercepts
// attention and RoPE ops to route through the KV-cache path.
// For each decode step (T=1), the executor:
//   1. Embeds the last token
//   2. For each layer:
//      a. RMSNorm
//      b. QKV projection (matmul)
//      c. RoPE on Q and K
//      d. Append K/V to cache
//      e. FlashAttention (Q attends to full KV-cache)
//      f. Output projection
//      g. RMSNorm + MLP
//   3. Final RMSNorm + LM head → logits

#ifndef SUROGATE_SRC_RUNTIME_INFER_DECODE_EXECUTOR_H
#define SUROGATE_SRC_RUNTIME_INFER_DECODE_EXECUTOR_H

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/infer/kv_cache.h"
#include "runtime/infer/decode_context.h"
#include "runtime/core/forward_hooks.h"
#include "utilities/tensor.h"

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
class CompiledExecutor;
struct CompiledGraph;
}

namespace infer {

/// Result of a single decode step.
struct DecodeStepResult {
    Tensor logits;              // [batch_size, vocab_size] logits on GPU
    bool all_finished = false;  // True when all sequences have hit EOS
};

/// DecodeExecutor executes single-token decode steps through the model with KV-cache.
///
/// Usage:
///   DecodeExecutor executor(graph_executor, run_state, ...);
///   executor.init(decode_context);
///   // Prefill:
///   executor.prefill(prompt_tokens, prompt_lens);
///   // Decode loop:
///   while (!ctx.state.all_finished()) {
///       auto result = executor.decode_step(last_tokens);
///       // sample from result.logits, update last_tokens, check EOS
///   }
class DecodeExecutor {
public:
    DecodeExecutor(dsl::DslRunState& run_state,
                   dsl::DslParamStore& weights,
                   const modules::ModelConfig& config,
                   const RuntimeOptions& options);
    ~DecodeExecutor();

    /// Initialize for a new generation batch.
    void init(DecodeContext& ctx);

    /// Prefill: process prompt tokens to populate KV-cache.
    /// prompt_tokens: [batch_size, max_prompt_len] (padded, host)
    /// prompt_lens: [batch_size] actual prompt lengths (host)
    /// Uses the existing full-sequence forward to process prompts.
    void prefill(const int32_t* prompt_tokens,
                 const int* prompt_lens, int batch_size, int max_prompt_len,
                 dsl::GraphExecutor& graph_executor,
                 NCCLCommunicator& comm,
                 const modules::ForwardHook* hook = nullptr);

    /// Execute one decode step.
    /// last_tokens: [batch_size] token IDs to process (GPU)
    /// Returns logits [batch_size, vocab_size] for sampling.
    DecodeStepResult decode_step(const int32_t* last_tokens_gpu,
                                 dsl::GraphExecutor& graph_executor,
                                 NCCLCommunicator& comm,
                                 const modules::ForwardHook* hook = nullptr);

    /// Sync streaming logprobs to host trajectories (call after each decode step).
    void sync_trajectories();

private:
    dsl::DslRunState& mRunState;
    dsl::DslParamStore& mWeights;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;

    DecodeContext* mCtx = nullptr;

    // Decode step counter (for position IDs)
    int mDecodeStep = 0;
};

}  // namespace infer

#endif  // SUROGATE_SRC_RUNTIME_INFER_DECODE_EXECUTOR_H
