// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/infer/generation_engine.h"

#include <algorithm>
#include <cstring>
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

    // Find max prompt length
    int max_prompt_len = 0;
    for (const auto& p : prompts) {
        max_prompt_len = std::max(max_prompt_len, static_cast<int>(p.size()));
    }

    const int max_total_len = max_prompt_len + gen_config.max_gen_len;
    const int Hkv = mConfig.NumKeyValHeads;
    const int Hs = mConfig.head_size();
    const int num_layers = mConfig.NumLayers;
    const int V = mConfig.VocabSize;

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
                                     batch_size, V, /*deterministic=*/true,
                                     gen_config.seed, rng_offset, mRunState.MainStream);
            } else if (gen_config.top_k > 0) {
                sampling_top_k(probs_gpu, sampled_tokens_gpu, gen_config.top_k,
                               batch_size, V, /*deterministic=*/true,
                               gen_config.seed, rng_offset, mRunState.MainStream);
            } else if (gen_config.top_p < 1.0f) {
                sampling_top_p(probs_gpu, sampled_tokens_gpu, gen_config.top_p,
                               batch_size, V, /*deterministic=*/true,
                               gen_config.seed, rng_offset, mRunState.MainStream);
            } else if (gen_config.min_p > 0.0f) {
                sampling_min_p(probs_gpu, sampled_tokens_gpu, gen_config.min_p,
                               batch_size, V, /*deterministic=*/true,
                               gen_config.seed, rng_offset, mRunState.MainStream);
            } else {
                // Plain categorical sampling
                sampling_from_probs(probs_gpu, sampled_tokens_gpu, batch_size, V,
                                    /*deterministic=*/true, gen_config.seed, rng_offset,
                                    mRunState.MainStream);
            }
        }
        rng_offset++;

        // Extract logprobs: logprob[i] = log(softmax(logits)[i, sampled_token[i]])
        // If we did greedy (no softmax), compute softmax now for logprobs
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

        // ----------------------------------------------------------------
        // Update sampled tokens as input for next step
        // ----------------------------------------------------------------
        // Copy sampled tokens back to last_tokens_gpu for next decode step
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

    // Find max prompt length
    int max_prompt_len = 0;
    for (const auto& p : prompts) {
        max_prompt_len = std::max(max_prompt_len, static_cast<int>(p.size()));
    }

    const int max_total_len = max_prompt_len + gen_config.max_gen_len;
    const int Hkv = mConfig.NumKeyValHeads;
    const int Hs = mConfig.head_size();
    const int num_layers = mConfig.NumLayers;
    const int V = mConfig.VocabSize;

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

    cudaStream_t copy_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&copy_stream));
    int32_t* pinned_tokens = nullptr;
    float* pinned_logprobs = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned_tokens, static_cast<std::size_t>(total_batch) * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&pinned_logprobs, static_cast<std::size_t>(total_batch) * sizeof(float)));

    // ========================================================================
    // Initialize state for all M×N sequences
    // ========================================================================
    std::vector<Trajectory> trajectories(static_cast<std::size_t>(total_batch));
    std::vector<int> seq_lens_host(static_cast<std::size_t>(total_batch), 0);
    std::vector<bool> finished(static_cast<std::size_t>(total_batch), false);
    int num_active = total_batch;

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

        // Allocate prefix pages and share to all N sequences
        const int prefix_pages = paged_kv.alloc_prefix_pages(m, prefill_len);
        paged_kv.share_prefix_pages(m, prefix_pages);

        // Upload block table for source slot (B=1)
        CUDA_CHECK(cudaMemcpyAsync(
            prefill_block_table_gpu,
            paged_kv.block_table_host.data() + static_cast<std::size_t>(m * N) * paged_kv.max_pages_per_seq,
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

        // Allocate first suffix page for each of the N sequences
        const int suffix_start_page = prefix_pages;
        for (int n = 0; n < N; ++n) {
            paged_kv.alloc_suffix_page(m * N + n, suffix_start_page);
        }

        for (int n = 0; n < N; ++n) {
            seq_lens_host[static_cast<std::size_t>(m * N + n)] = prefill_len;
        }
    }

    // Upload the full block table to GPU for the decode phase
    paged_kv.upload_block_table(mRunState.MainStream);

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

    uint64_t rng_offset = 0;
    cudaEvent_t compute_done;
    CUDA_CHECK(cudaEventCreate(&compute_done));

    // Pre-allocate host buffers used every step (avoid heap allocs in hot loop)
    std::vector<int32_t> pos_ids_host(static_cast<std::size_t>(total_batch));
    std::vector<int> finished_host(static_cast<std::size_t>(total_batch), 0);

    // Double-buffered pinned memory for overlapped D2H copies.
    // While the GPU computes step N+1, the CPU processes step N's results
    // from the other buffer — no cudaStreamSynchronize in the hot path.
    int32_t* pinned_tokens_b = nullptr;
    float* pinned_logprobs_b = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned_tokens_b, static_cast<std::size_t>(total_batch) * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&pinned_logprobs_b, static_cast<std::size_t>(total_batch) * sizeof(float)));
    int32_t* db_tokens[2] = { pinned_tokens, pinned_tokens_b };
    float*   db_logprobs[2] = { pinned_logprobs, pinned_logprobs_b };
    cudaEvent_t copy_done[2];
    CUDA_CHECK(cudaEventCreate(&copy_done[0]));
    CUDA_CHECK(cudaEventCreate(&copy_done[1]));
    int pending_buf = -1;  // which buffer has pending D2H (-1 = none)
    int pending_step = -1;

    for (int step = 0; step < gen_config.max_gen_len && num_active > 0; ++step) {
        const int cur_buf = step & 1;

        // ----------------------------------------------------------------
        // Process PREVIOUS step's D2H results (overlapped with this step's
        // H2D uploads below). On the first step there's nothing to process.
        // ----------------------------------------------------------------
        if (pending_buf >= 0) {
            CUDA_CHECK(cudaEventSynchronize(copy_done[pending_buf]));
            bool finished_changed = false;
            for (int i = 0; i < total_batch; ++i) {
                if (finished[static_cast<std::size_t>(i)]) continue;
                const int32_t token = db_tokens[pending_buf][i];
                const float logprob = db_logprobs[pending_buf][i];
                trajectories[static_cast<std::size_t>(i)].tokens.push_back(token);
                trajectories[static_cast<std::size_t>(i)].logprobs.push_back(logprob);
                seq_lens_host[static_cast<std::size_t>(i)]++;
                if (token == gen_config.eos_token_id) {
                    finished[static_cast<std::size_t>(i)] = true;
                    finished_changed = true;
                    trajectories[static_cast<std::size_t>(i)].completion_len = pending_step + 1;
                    num_active--;
                }
            }
            if (finished_changed) {
                for (int i = 0; i < total_batch; ++i) {
                    finished_host[static_cast<std::size_t>(i)] = finished[static_cast<std::size_t>(i)] ? 1 : 0;
                }
                CUDA_CHECK(cudaMemcpyAsync(grpo_finished_gpu, finished_host.data(),
                                           static_cast<std::size_t>(total_batch) * sizeof(int),
                                           cudaMemcpyHostToDevice, mRunState.MainStream));
            }
            if (num_active <= 0) break;
        }

        // ----------------------------------------------------------------
        // Paged KV: allocate suffix pages for sequences crossing a page boundary
        // ----------------------------------------------------------------
        {
            bool block_table_dirty = false;
            for (int i = 0; i < total_batch; ++i) {
                if (finished[static_cast<std::size_t>(i)]) continue;
                const int new_pos = seq_lens_host[static_cast<std::size_t>(i)];
                const int virtual_page = new_pos / kPageBlockSize;
                if (paged_kv.get_block_table(i, virtual_page) < 0) {
                    paged_kv.alloc_suffix_page(i, virtual_page);
                    block_table_dirty = true;
                }
            }
            if (block_table_dirty) {
                paged_kv.upload_block_table(mRunState.MainStream);
            }
        }

        // ----------------------------------------------------------------
        // Upload metadata for this decode step
        // ----------------------------------------------------------------
        CUDA_CHECK(cudaMemcpyAsync(seq_lens_gpu, seq_lens_host.data(),
                                   static_cast<std::size_t>(total_batch) * sizeof(int),
                                   cudaMemcpyHostToDevice, mRunState.MainStream));
        if (gen_config.use_cuda_graphs) {
            decode_state.max_seqlen_k = max_prompt_len + gen_config.max_gen_len;
        } else {
            decode_state.max_seqlen_k = *std::max_element(seq_lens_host.begin(), seq_lens_host.end()) + 1;
        }

        fill_decode_cu_seqlens(cu_seqlens_q_gpu, seqused_k_gpu,
                               seq_lens_gpu, total_batch, mRunState.MainStream);

        for (int i = 0; i < total_batch; ++i) {
            pos_ids_host[static_cast<std::size_t>(i)] = seq_lens_host[static_cast<std::size_t>(i)];
        }
        CUDA_CHECK(cudaMemcpyAsync(position_ids_gpu, pos_ids_host.data(),
                                   static_cast<std::size_t>(total_batch) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, mRunState.MainStream));

        // Mask finished sequences
        mask_finished_tokens(last_tokens_gpu, grpo_finished_gpu, total_batch, mRunState.MainStream);

        // ----------------------------------------------------------------
        // GPU compute: model forward + sampling
        // ----------------------------------------------------------------
        graph_executor.execute_decode_step(
            static_cast<long>(total_batch), last_tokens_gpu, position_ids_gpu,
            decode_state, comm, hook, gen_config.use_cuda_graphs);

        if (gen_config.temperature <= 0.0f) {
            sampling_argmax(logits_gpu, sampled_tokens_gpu, total_batch, V, mRunState.MainStream);
        } else {
            sampling_softmax(logits_gpu, probs_gpu, temperature_gpu,
                             total_batch, V, grpo_softmax_ws, grpo_softmax_ws_bytes,
                             mRunState.MainStream);
            sampling_from_probs(probs_gpu, sampled_tokens_gpu, total_batch, V,
                                /*deterministic=*/true, gen_config.seed, rng_offset,
                                mRunState.MainStream);
        }
        rng_offset++;

        if (gen_config.temperature <= 0.0f) {
            sampling_softmax(logits_gpu, probs_gpu, nullptr,
                             total_batch, V, grpo_softmax_ws, grpo_softmax_ws_bytes,
                             mRunState.MainStream);
        }
        sampling_extract_logprob(probs_gpu, sampled_tokens_gpu, logprobs_gpu,
                                 total_batch, V, mRunState.MainStream);

        // Feed sampled tokens back as input for next step (D2D, no sync needed)
        CUDA_CHECK(cudaMemcpyAsync(last_tokens_gpu, sampled_tokens_gpu,
                                   static_cast<std::size_t>(total_batch) * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));

        // ----------------------------------------------------------------
        // Async D2H into double buffer (no sync — will be read next iteration)
        // ----------------------------------------------------------------
        CUDA_CHECK(cudaEventRecord(compute_done, mRunState.MainStream));
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream, compute_done, 0));
        CUDA_CHECK(cudaMemcpyAsync(db_tokens[cur_buf], sampled_tokens_gpu,
                                   static_cast<std::size_t>(total_batch) * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(db_logprobs[cur_buf], logprobs_gpu,
                                   static_cast<std::size_t>(total_batch) * sizeof(float),
                                   cudaMemcpyDeviceToHost, copy_stream));
        CUDA_CHECK(cudaEventRecord(copy_done[cur_buf], copy_stream));

        pending_buf = cur_buf;
        pending_step = step;
    }

    // ----------------------------------------------------------------
    // Drain: process the last pending D2H buffer
    // ----------------------------------------------------------------
    if (pending_buf >= 0) {
        CUDA_CHECK(cudaEventSynchronize(copy_done[pending_buf]));
        for (int i = 0; i < total_batch; ++i) {
            if (finished[static_cast<std::size_t>(i)]) continue;
            const int32_t token = db_tokens[pending_buf][i];
            const float logprob = db_logprobs[pending_buf][i];
            trajectories[static_cast<std::size_t>(i)].tokens.push_back(token);
            trajectories[static_cast<std::size_t>(i)].logprobs.push_back(logprob);
            seq_lens_host[static_cast<std::size_t>(i)]++;
            if (token == gen_config.eos_token_id) {
                finished[static_cast<std::size_t>(i)] = true;
                trajectories[static_cast<std::size_t>(i)].completion_len = pending_step + 1;
            }
        }
    }

    for (int i = 0; i < total_batch; ++i) {
        if (!finished[static_cast<std::size_t>(i)]) {
            trajectories[static_cast<std::size_t>(i)].completion_len =
                static_cast<int>(trajectories[static_cast<std::size_t>(i)].tokens.size())
                - trajectories[static_cast<std::size_t>(i)].prompt_len;
        }
    }

    CUDA_CHECK(cudaEventDestroy(compute_done));
    CUDA_CHECK(cudaEventDestroy(copy_done[0]));
    CUDA_CHECK(cudaEventDestroy(copy_done[1]));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaFreeHost(pinned_tokens));
    CUDA_CHECK(cudaFreeHost(pinned_logprobs));
    CUDA_CHECK(cudaFreeHost(pinned_tokens_b));
    CUDA_CHECK(cudaFreeHost(pinned_logprobs_b));
    for (auto& [_, ptr] : grpo_recurrent_states) {
        if (ptr) cudaFree(ptr);
    }
    for (auto& [_, ptr] : grpo_conv_states) {
        if (ptr) cudaFree(ptr);
    }

    // Invalidate the cached decode CUDA graph before restoring the arena.
    // The graph captures pointers into the arena (KV cache, logits, etc.)
    // that become stale after arena.restore().
    graph_executor.invalidate_decode_graph();

    arena.restore(arena_cp);

    return trajectories;
}

}  // namespace infer
