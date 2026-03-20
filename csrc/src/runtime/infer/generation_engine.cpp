// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/infer/generation_engine.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/infer/decode_state.h"
#include "runtime/dsl/dsl_run_state.h"
#include "runtime/dsl/dsl_param_store.h"
#include "runtime/dsl/graph_executor.h"
#include "runtime/core/model_config.h"
#include "runtime/training/runtime_options.h"
#include "kernels/attention_decode.h"
#include "kernels/sampling.h"
#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace infer {

namespace {

inline bool is_valid_token_id(const int32_t token_id, const int vocab_size) {
    return token_id >= 0 && token_id < vocab_size;
}

inline bool sanitize_logits_enabled() {
    static int enabled = -1;
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char* env = std::getenv("SUROGATE_SANITIZE_LOGITS");
    enabled = (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    return enabled != 0;
}

inline int round_up_prompt_len_bucket(
        int observed_max_prompt_len,
        int max_gen_len,
        const dsl::DslRunState& run_state) {
    if (observed_max_prompt_len <= 0) {
        return observed_max_prompt_len;
    }

    // Limit by trainer sequence capacity when known.
    int prompt_cap = observed_max_prompt_len;
    if (run_state.Inputs.Rank >= 2) {
        const int seq_cap = static_cast<int>(run_state.Inputs.Sizes[1]);
        if (seq_cap > max_gen_len) {
            prompt_cap = std::max(observed_max_prompt_len, seq_cap - max_gen_len);
        }
    }

    static constexpr int kBuckets[] = {
        32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1536, 2048
    };
    for (int b : kBuckets) {
        if (b >= observed_max_prompt_len && b <= prompt_cap) {
            return b;
        }
    }
    return std::min(prompt_cap, observed_max_prompt_len);
}

void validate_prompt_tokens_or_throw(const std::vector<std::vector<int32_t>>& prompts,
                                     const int vocab_size,
                                     const char* caller) {
    for (std::size_t prompt_idx = 0; prompt_idx < prompts.size(); ++prompt_idx) {
        const auto& prompt = prompts[prompt_idx];
        for (std::size_t tok_idx = 0; tok_idx < prompt.size(); ++tok_idx) {
            const int32_t tok = prompt[tok_idx];
            if (!is_valid_token_id(tok, vocab_size)) {
                throw std::invalid_argument(
                    std::string(caller)
                    + ": prompt token out of range at prompt "
                    + std::to_string(prompt_idx)
                    + ", token "
                    + std::to_string(tok_idx)
                    + ": id="
                    + std::to_string(tok)
                    + ", vocab_size="
                    + std::to_string(vocab_size));
            }
        }
    }
}

}  // namespace

GenerationEngine::GenerationEngine(dsl::DslRunState& run_state,
                                   dsl::DslParamStore& weights,
                                   const modules::ModelConfig& config,
                                   const RuntimeOptions& options)
    : mRunState(run_state)
    , mWeights(weights)
    , mConfig(config)
    , mOptions(options)
{}

std::vector<Trajectory> GenerationEngine::generate(
        const std::vector<std::vector<int32_t>>& prompts,
        const GenerationEngineConfig& gen_config,
        DeviceMemoryStack& arena,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook) {

    const int batch_size = static_cast<int>(prompts.size());
    if (batch_size == 0) return {};

    // Find max prompt length and round up to a stable bucket to improve
    // decode graph reuse across calls with slightly different prompt lengths.
    int max_prompt_len_observed = 0;
    for (const auto& p : prompts) {
        max_prompt_len_observed = std::max(max_prompt_len_observed, static_cast<int>(p.size()));
    }
    const int max_prompt_len =
        round_up_prompt_len_bucket(max_prompt_len_observed, gen_config.max_gen_len, mRunState);

    const int max_total_len = max_prompt_len + gen_config.max_gen_len;
    const int Hkv = mConfig.NumKeyValHeads;
    const int Hs = mConfig.head_size();
    const int num_layers = mConfig.NumLayers;
    const int V = mConfig.VocabSize;

    validate_prompt_tokens_or_throw(prompts, V, "GenerationEngine::generate");

    // Save arena checkpoint for cleanup
    auto arena_cp = arena.checkpoint();

    // ========================================================================
    // Allocate KV-cache
    // ========================================================================
    KVCache kv_cache(num_layers, batch_size, max_total_len, Hkv, Hs, KVDType::BF16);
    kv_cache.allocate(arena);

    // ========================================================================
    // Allocate GPU buffers
    // ========================================================================
    auto* seq_lens_gpu = reinterpret_cast<int*>(
        arena.allocate(static_cast<std::size_t>(batch_size) * sizeof(int), "gen_seq_lens"));
    auto* cu_seqlens_q_gpu = reinterpret_cast<int32_t*>(
        arena.allocate(static_cast<std::size_t>(batch_size + 1) * sizeof(int32_t), "gen_cu_q"));
    auto* seqused_k_gpu = reinterpret_cast<int32_t*>(
        arena.allocate(static_cast<std::size_t>(batch_size) * sizeof(int32_t), "gen_seqused_k"));
    auto* position_ids_gpu = reinterpret_cast<int32_t*>(
        arena.allocate(static_cast<std::size_t>(batch_size) * sizeof(int32_t), "gen_pos_ids"));
    auto* last_tokens_gpu = reinterpret_cast<int32_t*>(
        arena.allocate(static_cast<std::size_t>(batch_size) * sizeof(int32_t), "gen_last_tokens"));

    // Logits buffer: [batch_size, vocab_size] — used by fused LM head in decode mode
    auto* logits_gpu = reinterpret_cast<float*>(
        arena.allocate(static_cast<std::size_t>(batch_size) * V * sizeof(float), "gen_logits"));
    // Probs buffer (reuse logits memory for in-place softmax when temperature > 0)
    auto* probs_gpu = logits_gpu;  // in-place softmax

    // EOS finished mask on GPU (0 = active, 1 = finished)
    auto* finished_gpu = reinterpret_cast<int*>(
        arena.allocate(static_cast<std::size_t>(batch_size) * sizeof(int), "gen_finished"));
    CUDA_CHECK(cudaMemsetAsync(finished_gpu, 0, static_cast<std::size_t>(batch_size) * sizeof(int),
                               mRunState.MainStream));

    // Sampled token IDs and logprobs on GPU
    auto* sampled_tokens_gpu = reinterpret_cast<int32_t*>(
        arena.allocate(static_cast<std::size_t>(batch_size) * sizeof(int32_t), "gen_sampled"));
    auto* logprobs_gpu = reinterpret_cast<float*>(
        arena.allocate(static_cast<std::size_t>(batch_size) * sizeof(float), "gen_logprobs"));

    // Softmax workspace (FlashInfer needs this for large vocab multi-block reduction)
    const std::size_t softmax_workspace_bytes = static_cast<std::size_t>(batch_size) * 256 * 8;  // generous
    void* softmax_workspace = arena.allocate(softmax_workspace_bytes, "gen_softmax_ws");

    // Temperature buffer (per-sequence, for softmax)
    float* temperature_gpu = nullptr;
    if (gen_config.temperature > 0.0f && gen_config.temperature != 1.0f) {
        temperature_gpu = reinterpret_cast<float*>(
            arena.allocate(static_cast<std::size_t>(batch_size) * sizeof(float), "gen_temperature"));
        std::vector<float> temp_host(static_cast<std::size_t>(batch_size), gen_config.temperature);
        CUDA_CHECK(cudaMemcpyAsync(temperature_gpu, temp_host.data(),
                                   static_cast<std::size_t>(batch_size) * sizeof(float),
                                   cudaMemcpyHostToDevice, mRunState.MainStream));
    }

    // Create a copy stream for async D2H of (token, logprob) per decode step
    cudaStream_t copy_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&copy_stream));

    // Pinned host buffers for streaming D2H
    int32_t* pinned_tokens = nullptr;
    float* pinned_logprobs = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned_tokens, static_cast<std::size_t>(batch_size) * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&pinned_logprobs, static_cast<std::size_t>(batch_size) * sizeof(float)));

    // ========================================================================
    // Initialize state
    // ========================================================================
    std::vector<Trajectory> trajectories(static_cast<std::size_t>(batch_size));
    std::vector<int> seq_lens_host(static_cast<std::size_t>(batch_size), 0);
    std::vector<bool> finished(static_cast<std::size_t>(batch_size), false);
    int num_active = batch_size;

    std::vector<int> prompt_lens_host(static_cast<std::size_t>(batch_size), 0);
    std::vector<int> prefill_lens_host(static_cast<std::size_t>(batch_size), 0);
    int max_prefill_len = 0;
    for (int b = 0; b < batch_size; ++b) {
        const int prompt_len = static_cast<int>(prompts[static_cast<std::size_t>(b)].size());
        if (prompt_len <= 0) {
            throw std::invalid_argument("GenerationEngine::generate: prompts must be non-empty");
        }
        const int prefill_len = prompt_len - 1;
        prompt_lens_host[static_cast<std::size_t>(b)] = prompt_len;
        prefill_lens_host[static_cast<std::size_t>(b)] = prefill_len;
        max_prefill_len = std::max(max_prefill_len, prefill_len);
        trajectories[static_cast<std::size_t>(b)].prompt_len = prompt_len;
        trajectories[static_cast<std::size_t>(b)].tokens = prompts[static_cast<std::size_t>(b)];
    }

    // Recurrent state map for delta-rule / SSM layers (Qwen3.5 hybrid)
    std::unordered_map<int, void*> recurrent_states;
    std::unordered_map<int, void*> conv_states;

    // Set up decode state
    infer::DecodeState decode_state{};
    decode_state.recurrent_states = &recurrent_states;
    decode_state.conv_states = &conv_states;
    decode_state.finished_gpu = finished_gpu;
    decode_state.k_data = kv_cache.k_data;
    decode_state.v_data = kv_cache.v_data;
    decode_state.per_buffer_bytes = kv_cache.per_buffer_bytes;
    decode_state.seq_lens_gpu = seq_lens_gpu;
    decode_state.cu_seqlens_q_gpu = cu_seqlens_q_gpu;
    decode_state.cu_seqlens_k_gpu = seqused_k_gpu;  // reused as seqused_k
    decode_state.max_seq_len = max_total_len;
    decode_state.num_kv_heads = Hkv;
    decode_state.head_dim = Hs;
    decode_state.fp8 = false;
    decode_state.logits_out_gpu = logits_gpu;
    decode_state.vocab_size = V;

    // ========================================================================
    // Prefill: process prompt prefix (all tokens except the last one) in a
    // single full-sequence forward. The first decode step then consumes the
    // last prompt token to produce the first generated token.
    // dispatch_rope bulk-stores K/V to the KV-cache; attention runs normally.
    // ========================================================================
    if (max_prefill_len > 0) {
        // Pad prompt prefixes into [batch_size, max_prefill_len] row-major
        const std::size_t padded_size = static_cast<std::size_t>(batch_size) * max_prefill_len;
        std::vector<int32_t> tokens_host(padded_size, 0);
        std::vector<int32_t> pos_ids_host(padded_size);
        for (int b = 0; b < batch_size; ++b) {
            const auto& p = prompts[static_cast<std::size_t>(b)];
            const int prefill_len = prefill_lens_host[static_cast<std::size_t>(b)];
            for (int t = 0; t < max_prefill_len; ++t) {
                const std::size_t idx = static_cast<std::size_t>(b) * max_prefill_len + t;
                tokens_host[idx] = t < prefill_len ? p[static_cast<std::size_t>(t)] : 0;
                pos_ids_host[idx] = t;
            }
        }

        graph_executor.execute_prefill(
            static_cast<long>(batch_size), static_cast<long>(max_prefill_len),
            tokens_host.data(), pos_ids_host.data(),
            decode_state, comm, hook);
    }

    // Update seq_lens to reflect prefilled prefix lengths.
    for (int b = 0; b < batch_size; ++b) {
        seq_lens_host[static_cast<std::size_t>(b)] = prefill_lens_host[static_cast<std::size_t>(b)];
    }

    // ========================================================================
    // Decode loop: generate tokens autoregressively
    // ========================================================================
    // After prefill, last_tokens_gpu holds the last prompt token for each seq.
    // We need to set it to the last token of each prompt for the first decode step.
    {
        std::vector<int32_t> first_decode_tokens(static_cast<std::size_t>(batch_size));
        for (int b = 0; b < batch_size; ++b) {
            const auto& p = prompts[static_cast<std::size_t>(b)];
            first_decode_tokens[static_cast<std::size_t>(b)] = p.back();
        }
        CUDA_CHECK(cudaMemcpyAsync(last_tokens_gpu, first_decode_tokens.data(),
                                   static_cast<std::size_t>(batch_size) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, mRunState.MainStream));
    }

    // RNG state for sampling
    uint64_t rng_offset = 0;

    // Event for copy stream synchronization
    cudaEvent_t compute_done;
    CUDA_CHECK(cudaEventCreate(&compute_done));

    // Pre-allocate host buffers used every step (avoid heap allocs in hot loop)
    std::vector<int32_t> pos_ids_host(static_cast<std::size_t>(batch_size));
    std::vector<int> finished_host_buf(static_cast<std::size_t>(batch_size), 0);

    for (int step = 0; step < gen_config.max_gen_len && num_active > 0; ++step) {
        // Upload seq_lens and build cu_seqlens/seqused_k
        CUDA_CHECK(cudaMemcpyAsync(seq_lens_gpu, seq_lens_host.data(),
                                   static_cast<std::size_t>(batch_size) * sizeof(int),
                                   cudaMemcpyHostToDevice, mRunState.MainStream));

        if (gen_config.use_cuda_graphs) {
            decode_state.max_seqlen_k = max_prompt_len + gen_config.max_gen_len;
        } else {
            decode_state.max_seqlen_k = *std::max_element(seq_lens_host.begin(), seq_lens_host.end()) + 1;
        }

        fill_decode_cu_seqlens(cu_seqlens_q_gpu, seqused_k_gpu,
                               seq_lens_gpu, batch_size, mRunState.MainStream);

        // Position IDs
        for (int b = 0; b < batch_size; ++b) {
            pos_ids_host[static_cast<std::size_t>(b)] = seq_lens_host[static_cast<std::size_t>(b)];
        }
        CUDA_CHECK(cudaMemcpyAsync(position_ids_gpu, pos_ids_host.data(),
                                   static_cast<std::size_t>(batch_size) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, mRunState.MainStream));

        // Mask finished sequences: replace their token with pad (0)
        mask_finished_tokens(last_tokens_gpu, finished_gpu, batch_size, mRunState.MainStream);

        // Execute decode step → logits in logits_gpu [batch_size, V]
        graph_executor.execute_decode_step(
            static_cast<long>(batch_size), last_tokens_gpu, position_ids_gpu,
            decode_state, comm, hook, gen_config.use_cuda_graphs);
        if (sanitize_logits_enabled()) {
            // Match HF remove_invalid_values behavior when explicitly enabled.
            sampling_sanitize_logits(logits_gpu, batch_size, V, mRunState.MainStream);
        }

        // ----------------------------------------------------------------
        // Sample from logits
        // ----------------------------------------------------------------
        if (gen_config.temperature <= 0.0f) {
            // Greedy (argmax)
            sampling_argmax(logits_gpu, sampled_tokens_gpu, batch_size, V, mRunState.MainStream);
        } else {
            // Temperature-scaled softmax → probs
            sampling_softmax(logits_gpu, probs_gpu, temperature_gpu,
                             batch_size, V, softmax_workspace, softmax_workspace_bytes,
                             mRunState.MainStream);

            // Apply filtered sampling strategy
            if (gen_config.top_k > 0 && gen_config.top_p < 1.0f) {
                sampling_top_k_top_p(probs_gpu, sampled_tokens_gpu,
                                     gen_config.top_k, gen_config.top_p,
                                     batch_size, V, /*deterministic=*/false,
                                     gen_config.seed, rng_offset, mRunState.MainStream);
            } else if (gen_config.top_k > 0) {
                sampling_top_k(probs_gpu, sampled_tokens_gpu, gen_config.top_k,
                               batch_size, V, /*deterministic=*/false,
                               gen_config.seed, rng_offset, mRunState.MainStream);
            } else if (gen_config.top_p < 1.0f) {
                sampling_top_p(probs_gpu, sampled_tokens_gpu, gen_config.top_p,
                               batch_size, V, /*deterministic=*/false,
                               gen_config.seed, rng_offset, mRunState.MainStream);
            } else if (gen_config.min_p > 0.0f) {
                sampling_min_p(probs_gpu, sampled_tokens_gpu, gen_config.min_p,
                               batch_size, V, /*deterministic=*/false,
                               gen_config.seed, rng_offset, mRunState.MainStream);
            } else {
                // Plain categorical sampling
                sampling_from_probs(probs_gpu, sampled_tokens_gpu, batch_size, V,
                                    /*deterministic=*/false, gen_config.seed, rng_offset,
                                    mRunState.MainStream);
            }
        }
        // Ensure finished rows remain masked for subsequent decode steps.
        mask_finished_tokens(sampled_tokens_gpu, finished_gpu, batch_size, mRunState.MainStream);
        rng_offset++;

        // Extract logprobs: logprob[i] = log p(sampled_token_i | logits_i)
        if (gen_config.temperature <= 0.0f) {
            sampling_softmax(logits_gpu, probs_gpu, /*temperature=*/nullptr,
                             batch_size, V, softmax_workspace, softmax_workspace_bytes,
                             mRunState.MainStream);
        }
        sampling_extract_logprob(probs_gpu, sampled_tokens_gpu, logprobs_gpu,
                                 batch_size, V, mRunState.MainStream);

        // ----------------------------------------------------------------
        // Streaming D2H: copy (token, logprob) to pinned host on copy stream
        // ----------------------------------------------------------------
        CUDA_CHECK(cudaEventRecord(compute_done, mRunState.MainStream));
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, compute_done, 0));

        CUDA_CHECK(cudaMemcpyAsync(pinned_tokens, sampled_tokens_gpu,
                                   static_cast<std::size_t>(batch_size) * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(pinned_logprobs, logprobs_gpu,
                                   static_cast<std::size_t>(batch_size) * sizeof(float),
                                   cudaMemcpyDeviceToHost, copy_stream));

        // Wait for D2H to complete before reading host buffers
        CUDA_CHECK(cudaStreamSynchronize(copy_stream));

        // Feed sampled tokens back for next step.
        CUDA_CHECK(cudaMemcpyAsync(last_tokens_gpu, sampled_tokens_gpu,
                                   static_cast<std::size_t>(batch_size) * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));

        // ----------------------------------------------------------------
        // EOS detection + trajectory update
        // ----------------------------------------------------------------
        bool finished_changed = false;
        for (int b = 0; b < batch_size; ++b) {
            if (finished[static_cast<std::size_t>(b)]) continue;

            const int32_t token = pinned_tokens[b];
            const float logprob = pinned_logprobs[b];

            trajectories[static_cast<std::size_t>(b)].tokens.push_back(token);
            trajectories[static_cast<std::size_t>(b)].logprobs.push_back(logprob);
            seq_lens_host[static_cast<std::size_t>(b)]++;

            if (token == gen_config.eos_token_id) {
                finished[static_cast<std::size_t>(b)] = true;
                finished_changed = true;
                trajectories[static_cast<std::size_t>(b)].completion_len = step + 1;
                num_active--;
            }
        }

        // Upload finished mask to GPU for next step's EOS masking
        if (finished_changed) {
            for (int b = 0; b < batch_size; ++b) {
                finished_host_buf[static_cast<std::size_t>(b)] = finished[static_cast<std::size_t>(b)] ? 1 : 0;
            }
            CUDA_CHECK(cudaMemcpyAsync(finished_gpu, finished_host_buf.data(),
                                       static_cast<std::size_t>(batch_size) * sizeof(int),
                                       cudaMemcpyHostToDevice, mRunState.MainStream));
        }
    }

    // Set completion_len for sequences that didn't hit EOS
    for (int b = 0; b < batch_size; ++b) {
        if (!finished[static_cast<std::size_t>(b)]) {
            trajectories[static_cast<std::size_t>(b)].completion_len =
                static_cast<int>(trajectories[static_cast<std::size_t>(b)].tokens.size())
                - trajectories[static_cast<std::size_t>(b)].prompt_len;
        }
    }
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(compute_done));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaFreeHost(pinned_tokens));
    CUDA_CHECK(cudaFreeHost(pinned_logprobs));

    // Free recurrent state buffers (allocated via cudaMalloc, not arena)
    for (auto& [_, ptr] : recurrent_states) {
        if (ptr) cudaFree(ptr);
    }
    for (auto& [_, ptr] : conv_states) {
        if (ptr) cudaFree(ptr);
    }

    // Invalidate decode graph before arena restore: captured graph arguments
    // include generation arena pointers that become stale after restore.
    graph_executor.invalidate_decode_graph();
    arena.restore(arena_cp);

    return trajectories;
}

std::vector<Trajectory> GenerationEngine::generate_grpo(
        const std::vector<std::vector<int32_t>>& prompts,
        const GenerationEngineConfig& gen_config,
        DeviceMemoryStack& arena,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook) {

    const int M = static_cast<int>(prompts.size());  // number of prompts
    const int N = gen_config.num_completions;         // completions per prompt
    if (M == 0 || N <= 0) return {};

    const int total_batch = M * N;  // total sequences

    // Find max prompt length and round up to a stable bucket to improve
    // decode graph reuse across calls with slightly different prompt lengths.
    int max_prompt_len_observed = 0;
    for (const auto& p : prompts) {
        max_prompt_len_observed = std::max(max_prompt_len_observed, static_cast<int>(p.size()));
    }
    const int max_prompt_len =
        round_up_prompt_len_bucket(max_prompt_len_observed, gen_config.max_gen_len, mRunState);

    const int max_total_len = max_prompt_len + gen_config.max_gen_len;
    const int Hkv = mConfig.NumKeyValHeads;
    const int Hs = mConfig.head_size();
    const int num_layers = mConfig.NumLayers;
    const int V = mConfig.VocabSize;

    validate_prompt_tokens_or_throw(prompts, V, "GenerationEngine::generate_grpo");

    auto arena_cp = arena.checkpoint();

    // ========================================================================
    // Allocate paged KV-cache for M×N sequences with shared prefix pages
    // ========================================================================
    constexpr int kPageBlockSize = 256;
    PagedKVCache paged_kv;
    paged_kv.configure(M, N, num_layers, max_prompt_len, gen_config.max_gen_len,
                       Hkv, Hs, kPageBlockSize, KVDType::BF16);
    paged_kv.allocate(arena);

    // ========================================================================
    // Allocate GPU buffers (sized for total_batch)
    // ========================================================================
    auto alloc = [&](std::size_t bytes, const char* name) {
        return arena.allocate(bytes, name);
    };
    auto* seq_lens_gpu = reinterpret_cast<int*>(
        alloc(static_cast<std::size_t>(total_batch) * sizeof(int), "grpo_seq_lens"));
    auto* cu_seqlens_q_gpu = reinterpret_cast<int32_t*>(
        alloc(static_cast<std::size_t>(total_batch + 1) * sizeof(int32_t), "grpo_cu_q"));
    auto* seqused_k_gpu = reinterpret_cast<int32_t*>(
        alloc(static_cast<std::size_t>(total_batch) * sizeof(int32_t), "grpo_seqused_k"));
    auto* position_ids_gpu = reinterpret_cast<int32_t*>(
        alloc(static_cast<std::size_t>(total_batch) * sizeof(int32_t), "grpo_pos_ids"));
    auto* last_tokens_gpu = reinterpret_cast<int32_t*>(
        alloc(static_cast<std::size_t>(total_batch) * sizeof(int32_t), "grpo_last_tokens"));
    auto* logits_gpu = reinterpret_cast<float*>(
        alloc(static_cast<std::size_t>(total_batch) * V * sizeof(float), "grpo_logits"));
    float* probs_gpu = logits_gpu;
    const std::size_t grpo_softmax_ws_bytes = static_cast<std::size_t>(total_batch) * 256 * 8;
    void* grpo_softmax_ws = alloc(grpo_softmax_ws_bytes, "grpo_softmax_ws");
    auto* sampled_tokens_gpu = reinterpret_cast<int32_t*>(
        alloc(static_cast<std::size_t>(total_batch) * sizeof(int32_t), "grpo_sampled"));
    auto* logprobs_gpu = reinterpret_cast<float*>(
        alloc(static_cast<std::size_t>(total_batch) * sizeof(float), "grpo_logprobs"));
    auto* generated_tokens_gpu = reinterpret_cast<int32_t*>(
        alloc(static_cast<std::size_t>(gen_config.max_gen_len) * static_cast<std::size_t>(total_batch) * sizeof(int32_t),
              "grpo_generated_tokens"));
    auto* generated_logprobs_gpu = reinterpret_cast<float*>(
        alloc(static_cast<std::size_t>(gen_config.max_gen_len) * static_cast<std::size_t>(total_batch) * sizeof(float),
              "grpo_generated_logprobs"));
    auto* completion_lens_gpu = reinterpret_cast<int32_t*>(
        alloc(static_cast<std::size_t>(total_batch) * sizeof(int32_t), "grpo_completion_lens"));
    CUDA_CHECK(cudaMemsetAsync(completion_lens_gpu, 0,
                               static_cast<std::size_t>(total_batch) * sizeof(int32_t), mRunState.MainStream));
    auto* active_count_gpu = reinterpret_cast<int*>(
        alloc(sizeof(int), "grpo_active_count"));
    auto* grpo_finished_gpu = reinterpret_cast<int*>(
        alloc(static_cast<std::size_t>(total_batch) * sizeof(int), "grpo_finished"));
    CUDA_CHECK(cudaMemsetAsync(grpo_finished_gpu, 0,
                               static_cast<std::size_t>(total_batch) * sizeof(int), mRunState.MainStream));

    float* temperature_gpu = nullptr;
    if (gen_config.temperature > 0.0f && gen_config.temperature != 1.0f) {
        temperature_gpu = reinterpret_cast<float*>(
            alloc(static_cast<std::size_t>(total_batch) * sizeof(float), "grpo_temp"));
        std::vector<float> temp_host(static_cast<std::size_t>(total_batch), gen_config.temperature);
        CUDA_CHECK(cudaMemcpyAsync(temperature_gpu, temp_host.data(),
                                   static_cast<std::size_t>(total_batch) * sizeof(float),
                                   cudaMemcpyHostToDevice, mRunState.MainStream));
    }

    // ========================================================================
    // Initialize state for all M×N sequences
    // ========================================================================
    std::vector<Trajectory> trajectories(static_cast<std::size_t>(total_batch));
    std::vector<int> seq_lens_host(static_cast<std::size_t>(total_batch), 0);

    std::vector<int> prompt_lens_host(static_cast<std::size_t>(total_batch));
    std::vector<int> prefill_lens_per_prompt(static_cast<std::size_t>(M), 0);
    for (int m = 0; m < M; ++m) {
        const int plen = static_cast<int>(prompts[static_cast<std::size_t>(m)].size());
        if (plen <= 0) {
            throw std::invalid_argument("GenerationEngine::generate_grpo: prompts must be non-empty");
        }
        prefill_lens_per_prompt[static_cast<std::size_t>(m)] = plen - 1;
        for (int n = 0; n < N; ++n) {
            const int idx = m * N + n;
            prompt_lens_host[static_cast<std::size_t>(idx)] = plen;
            trajectories[static_cast<std::size_t>(idx)].prompt_len = plen;
            trajectories[static_cast<std::size_t>(idx)].tokens = prompts[static_cast<std::size_t>(m)];
        }
    }

    // Block table GPU buffer for B=1 prefill
    auto* prefill_block_table_gpu = reinterpret_cast<int*>(
        alloc(static_cast<std::size_t>(paged_kv.max_pages_per_seq) * sizeof(int), "grpo_prefill_bt"));

    // ========================================================================
    // Prefill: for each prompt, run prefix-only forward (all tokens except the
    // last one) to populate shared prefix pages.
    // ========================================================================
    for (int m = 0; m < M; ++m) {
        const auto& prompt = prompts[static_cast<std::size_t>(m)];
        const int plen = static_cast<int>(prompt.size());
        const int prefill_len = prefill_lens_per_prompt[static_cast<std::size_t>(m)];

        // Allocate prefix pages for source slot.
        const int prefix_pages = paged_kv.alloc_prefix_pages(m, prefill_len);
        const int full_prefix_pages = prefill_len / kPageBlockSize;
        const int partial_prefix_tokens = prefill_len % kPageBlockSize;
        const int src_slot = m * N;

        // Share only full prefix pages.
        if (full_prefix_pages > 0) {
            paged_kv.share_prefix_pages(m, full_prefix_pages);
        }

        // Materialize private copies for the last partially-filled prefix page.
        // Sharing that page would alias continuation writes across completions.
        int partial_virtual_page = -1;
        if (partial_prefix_tokens > 0) {
            partial_virtual_page = full_prefix_pages;
            for (int n = 1; n < N; ++n) {
                const int dst_slot = src_slot + n;
                const int dst_phys = paged_kv.alloc_page();
                paged_kv.set_block_table(dst_slot, partial_virtual_page, dst_phys);
            }
        }

        // Upload block table for source slot (B=1)
        CUDA_CHECK(cudaMemcpyAsync(
            prefill_block_table_gpu,
            paged_kv.block_table_host.data() + static_cast<std::size_t>(src_slot) * paged_kv.max_pages_per_seq,
            static_cast<std::size_t>(paged_kv.max_pages_per_seq) * sizeof(int),
            cudaMemcpyHostToDevice, mRunState.MainStream));

        // Prefill decode state (B=1, paged, prefill_mode=true)
        infer::DecodeState prompt_ds{};
        prompt_ds.paged = true;
        prompt_ds.k_pages = paged_kv.k_pages;
        prompt_ds.v_pages = paged_kv.v_pages;
        prompt_ds.per_pool_bytes = paged_kv.per_pool_bytes();
        prompt_ds.block_table_gpu = prefill_block_table_gpu;
        prompt_ds.block_table_stride = paged_kv.max_pages_per_seq;
        prompt_ds.page_block_size = kPageBlockSize;
        prompt_ds.total_pages = paged_kv.total_pages;
        prompt_ds.num_kv_heads = Hkv;
        prompt_ds.head_dim = Hs;
        prompt_ds.logits_out_gpu = logits_gpu;
        prompt_ds.vocab_size = V;
        // prefill_mode is set by execute_prefill()

        if (prefill_len > 0) {
            // Build token and position arrays for this prompt prefix.
            std::vector<int32_t> tokens_host(static_cast<std::size_t>(prefill_len));
            std::vector<int32_t> pos_ids_host(static_cast<std::size_t>(prefill_len));
            for (int t = 0; t < prefill_len; ++t) {
                tokens_host[static_cast<std::size_t>(t)] = prompt[static_cast<std::size_t>(t)];
                pos_ids_host[static_cast<std::size_t>(t)] = t;
            }

            graph_executor.execute_prefill(
                1L, static_cast<long>(prefill_len),
                tokens_host.data(), pos_ids_host.data(),
                prompt_ds, comm, hook);
        }

        // Copy the initialized prefix tail from source partial page into each
        // completion's private partial page (all layers, K and V).
        if (partial_prefix_tokens > 0 && N > 1) {
            const int src_phys = paged_kv.get_block_table(src_slot, partial_virtual_page);
            for (int n = 1; n < N; ++n) {
                const int dst_slot = src_slot + n;
                const int dst_phys = paged_kv.get_block_table(dst_slot, partial_virtual_page);
                paged_kv.copy_prefix_tokens_between_pages(
                    src_phys, dst_phys, partial_prefix_tokens, mRunState.MainStream);
            }
        }

        for (int n = 0; n < N; ++n) {
            seq_lens_host[static_cast<std::size_t>(m * N + n)] = prefill_len;
        }
    }

    // Pre-allocate all missing pages for decode once. This removes per-step
    // host-side page-boundary checks and block-table uploads from the hot loop.
    {
        for (int i = 0; i < total_batch; ++i) {
            for (int vp = 0; vp < paged_kv.max_pages_per_seq; ++vp) {
                if (paged_kv.get_block_table(i, vp) < 0) {
                    paged_kv.alloc_suffix_page(i, vp);
                }
            }
        }
        paged_kv.upload_block_table(mRunState.MainStream);
    }

    // ========================================================================
    // Decode loop: all M×N sequences decode in parallel
    // ========================================================================
    std::unordered_map<int, void*> grpo_recurrent_states;
    std::unordered_map<int, void*> grpo_conv_states;

    infer::DecodeState decode_state{};
    decode_state.recurrent_states = &grpo_recurrent_states;
    decode_state.conv_states = &grpo_conv_states;
    decode_state.paged = true;
    decode_state.k_pages = paged_kv.k_pages;
    decode_state.v_pages = paged_kv.v_pages;
    decode_state.per_pool_bytes = paged_kv.per_pool_bytes();
    decode_state.block_table_gpu = paged_kv.block_table_gpu;
    decode_state.block_table_stride = paged_kv.max_pages_per_seq;
    decode_state.page_block_size = kPageBlockSize;
    decode_state.total_pages = paged_kv.total_pages;
    decode_state.seq_lens_gpu = seq_lens_gpu;
    decode_state.cu_seqlens_q_gpu = cu_seqlens_q_gpu;
    decode_state.cu_seqlens_k_gpu = seqused_k_gpu;
    decode_state.num_kv_heads = Hkv;
    decode_state.head_dim = Hs;
    decode_state.finished_gpu = grpo_finished_gpu;
    decode_state.logits_out_gpu = logits_gpu;
    decode_state.vocab_size = V;

    // Set initial last_tokens to the last token of each prompt
    {
        std::vector<int32_t> first_decode_tokens(static_cast<std::size_t>(total_batch));
        for (int m = 0; m < M; ++m) {
            const auto& p = prompts[static_cast<std::size_t>(m)];
            for (int n = 0; n < N; ++n) {
                first_decode_tokens[static_cast<std::size_t>(m * N + n)] = p.back();
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(last_tokens_gpu, first_decode_tokens.data(),
                                   static_cast<std::size_t>(total_batch) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, mRunState.MainStream));
    }
    CUDA_CHECK(cudaMemcpyAsync(seq_lens_gpu, seq_lens_host.data(),
                               static_cast<std::size_t>(total_batch) * sizeof(int),
                               cudaMemcpyHostToDevice, mRunState.MainStream));

    uint64_t rng_offset = 0;
    decode_state.max_seqlen_k = max_total_len;
    int generated_steps = gen_config.max_gen_len;
    int active_count_host = total_batch;
    const int stop_check_start = std::max(16, gen_config.max_gen_len / 2);
    const int stop_check_interval = std::max(16, gen_config.max_gen_len / 8);

    for (int step = 0; step < gen_config.max_gen_len; ++step) {

        // ----------------------------------------------------------------
        // Upload metadata for this decode step
        // ----------------------------------------------------------------
        fill_decode_cu_seqlens(cu_seqlens_q_gpu, seqused_k_gpu,
                               seq_lens_gpu, total_batch, mRunState.MainStream);
        CUDA_CHECK(cudaMemcpyAsync(position_ids_gpu, seq_lens_gpu,
                                   static_cast<std::size_t>(total_batch) * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));

        // Mask finished sequences
        mask_finished_tokens(last_tokens_gpu, grpo_finished_gpu, total_batch, mRunState.MainStream);

        // ----------------------------------------------------------------
        // GPU compute: model forward + sampling
        // ----------------------------------------------------------------
        graph_executor.execute_decode_step(
            static_cast<long>(total_batch), last_tokens_gpu, position_ids_gpu,
            decode_state, comm, hook, gen_config.use_cuda_graphs);
        if (sanitize_logits_enabled()) {
            // Match HF remove_invalid_values behavior when explicitly enabled.
            sampling_sanitize_logits(logits_gpu, total_batch, V, mRunState.MainStream);
        }

        // Check for async errors after decode step (catches OOB writes in
        // LoRA hooks, attention, etc. that would otherwise surface later
        // during training and be hard to diagnose).
        CUDA_CHECK(cudaGetLastError());

        if (gen_config.temperature <= 0.0f) {
            sampling_argmax(logits_gpu, sampled_tokens_gpu, total_batch, V, mRunState.MainStream);
        } else {
            sampling_softmax(logits_gpu, probs_gpu, temperature_gpu,
                             total_batch, V, grpo_softmax_ws, grpo_softmax_ws_bytes,
                             mRunState.MainStream);
            if (gen_config.top_k > 0 && gen_config.top_p < 1.0f) {
                sampling_top_k_top_p(probs_gpu, sampled_tokens_gpu,
                                     gen_config.top_k, gen_config.top_p,
                                     total_batch, V, /*deterministic=*/false,
                                     gen_config.seed, rng_offset, mRunState.MainStream);
            } else if (gen_config.top_k > 0) {
                sampling_top_k(probs_gpu, sampled_tokens_gpu, gen_config.top_k,
                               total_batch, V, /*deterministic=*/false,
                               gen_config.seed, rng_offset, mRunState.MainStream);
            } else if (gen_config.top_p < 1.0f) {
                sampling_top_p(probs_gpu, sampled_tokens_gpu, gen_config.top_p,
                               total_batch, V, /*deterministic=*/false,
                               gen_config.seed, rng_offset, mRunState.MainStream);
            } else if (gen_config.min_p > 0.0f) {
                sampling_min_p(probs_gpu, sampled_tokens_gpu, gen_config.min_p,
                               total_batch, V, /*deterministic=*/false,
                               gen_config.seed, rng_offset, mRunState.MainStream);
            } else {
                sampling_from_probs(probs_gpu, sampled_tokens_gpu, total_batch, V,
                                    /*deterministic=*/false, gen_config.seed, rng_offset,
                                    mRunState.MainStream);
            }
        }
        mask_finished_tokens(sampled_tokens_gpu, grpo_finished_gpu, total_batch, mRunState.MainStream);
        rng_offset++;

        if (gen_config.temperature <= 0.0f) {
            sampling_softmax(logits_gpu, probs_gpu, nullptr,
                             total_batch, V, grpo_softmax_ws, grpo_softmax_ws_bytes,
                             mRunState.MainStream);
        }
        sampling_extract_logprob(probs_gpu, sampled_tokens_gpu, logprobs_gpu,
                                 total_batch, V, mRunState.MainStream);

        // Store step outputs on GPU; host copies are done once after decode.
        const std::size_t step_off =
            static_cast<std::size_t>(step) * static_cast<std::size_t>(total_batch);
        CUDA_CHECK(cudaMemcpyAsync(
            generated_tokens_gpu + step_off, sampled_tokens_gpu,
            static_cast<std::size_t>(total_batch) * sizeof(int32_t),
            cudaMemcpyDeviceToDevice, mRunState.MainStream));
        CUDA_CHECK(cudaMemcpyAsync(
            generated_logprobs_gpu + step_off, logprobs_gpu,
            static_cast<std::size_t>(total_batch) * sizeof(float),
            cudaMemcpyDeviceToDevice, mRunState.MainStream));

        // Feed sampled tokens back for next step.
        CUDA_CHECK(cudaMemcpyAsync(last_tokens_gpu, sampled_tokens_gpu,
                                   static_cast<std::size_t>(total_batch) * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
        update_generation_state(
            sampled_tokens_gpu,
            grpo_finished_gpu,
            seq_lens_gpu,
            completion_lens_gpu,
            static_cast<int32_t>(gen_config.eos_token_id),
            total_batch,
            mRunState.MainStream);

        if ((step + 1) >= stop_check_start &&
            (((step + 1 - stop_check_start) % stop_check_interval) == 0 ||
             (step + 1) == gen_config.max_gen_len)) {
            count_active_sequences(
                grpo_finished_gpu, active_count_gpu, total_batch, mRunState.MainStream);
            CUDA_CHECK(cudaMemcpyAsync(&active_count_host, active_count_gpu, sizeof(int),
                                       cudaMemcpyDeviceToHost, mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            if (active_count_host == 0) {
                generated_steps = step + 1;
                break;
            }
        }
    }

    std::vector<int32_t> completion_lens_host(static_cast<std::size_t>(total_batch), 0);
    std::vector<int32_t> generated_tokens_host(
        static_cast<std::size_t>(generated_steps) * static_cast<std::size_t>(total_batch), 0);
    std::vector<float> generated_logprobs_host(
        static_cast<std::size_t>(generated_steps) * static_cast<std::size_t>(total_batch), 0.0f);
    CUDA_CHECK(cudaMemcpyAsync(completion_lens_host.data(), completion_lens_gpu,
                               static_cast<std::size_t>(total_batch) * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, mRunState.MainStream));
    CUDA_CHECK(cudaMemcpyAsync(generated_tokens_host.data(), generated_tokens_gpu,
                               generated_tokens_host.size() * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, mRunState.MainStream));
    CUDA_CHECK(cudaMemcpyAsync(generated_logprobs_host.data(), generated_logprobs_gpu,
                               generated_logprobs_host.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, mRunState.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));

    for (int i = 0; i < total_batch; ++i) {
        int cl = completion_lens_host[static_cast<std::size_t>(i)];
        cl = std::max(0, std::min(cl, generated_steps));
        trajectories[static_cast<std::size_t>(i)].completion_len = cl;
        for (int step = 0; step < cl; ++step) {
            const std::size_t off =
                static_cast<std::size_t>(step) * static_cast<std::size_t>(total_batch)
                + static_cast<std::size_t>(i);
            trajectories[static_cast<std::size_t>(i)].tokens.push_back(generated_tokens_host[off]);
            trajectories[static_cast<std::size_t>(i)].logprobs.push_back(generated_logprobs_host[off]);
        }
    }
    for (auto& [_, ptr] : grpo_recurrent_states) {
        if (ptr) cudaFree(ptr);
    }
    for (auto& [_, ptr] : grpo_conv_states) {
        if (ptr) cudaFree(ptr);
    }

    // Invalidate decode graph before arena restore: captured graph arguments
    // include generation arena pointers that become stale after restore.
    graph_executor.invalidate_decode_graph();

    // Clear any prefix boundary that generation may have left, so
    // arena.restore() can fully rewind the stack to pre-generation state.
    arena.clear_prefix_boundary();
    arena.restore(arena_cp);

    return trajectories;
}

}  // namespace infer
