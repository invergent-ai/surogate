// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/infer/generation_engine.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
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

inline bool env_flag_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "False") != 0 &&
           std::strcmp(v, "FALSE") != 0;
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

inline int resolve_prefill_chunk_size(
        const GenerationEngineConfig& gen_config,
        const int max_prefill_len) {
    if (max_prefill_len <= 0) {
        return 0;
    }
    if (gen_config.prefill_chunk_size <= 0) {
        return max_prefill_len;
    }
    return std::max(1, std::min(gen_config.prefill_chunk_size, max_prefill_len));
}

inline bool supports_fp8_kv_cache(const cudaDeviceProp& prop) {
    // FP8 KV-cache is opt-in.
    if (!env_flag_enabled("SUROGATE_ENABLE_FP8_KV_CACHE")) {
        return false;
    }
    // FP8 E4M3 KV kernels require FP8-capable Tensor Cores (SM89+).
    const int sm = prop.major * 10 + prop.minor;
    return sm >= 89;
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
    // Non-GRPO generation now runs through the same paged-KV decode path as
    // GRPO with N=1, so we keep one optimized implementation.
    GenerationEngineConfig paged_cfg = gen_config;
    paged_cfg.num_completions = 1;
    return generate_grpo(prompts, paged_cfg, arena, graph_executor, comm, hook);
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
    int max_prefill_len_observed = 0;
    for (const auto& p : prompts) {
        max_prompt_len_observed = std::max(max_prompt_len_observed, static_cast<int>(p.size()));
        max_prefill_len_observed = std::max(
            max_prefill_len_observed,
            std::max(0, static_cast<int>(p.size()) - 1));
    }
    const int max_prompt_len =
        round_up_prompt_len_bucket(max_prompt_len_observed, gen_config.max_gen_len, mRunState);
    int prefill_chunk_size = resolve_prefill_chunk_size(gen_config, max_prefill_len_observed);
    if (prefill_chunk_size > 0 && mRunState.Inputs.Rank >= 2) {
        const int static_t_cap = static_cast<int>(mRunState.Inputs.Sizes[1]);
        if (static_t_cap > 0) {
            prefill_chunk_size = std::max(1, std::min(prefill_chunk_size, static_t_cap));
        }
    }

    const int max_total_len = max_prompt_len + gen_config.max_gen_len;
    const int Hkv = mConfig.NumKeyValHeads;
    const int Hs = mConfig.head_size();
    const int num_layers = mConfig.NumLayers;
    const int V = mConfig.VocabSize;
    const int32_t fallback_token_id = is_valid_token_id(
        static_cast<int32_t>(gen_config.eos_token_id), V
    ) ? static_cast<int32_t>(gen_config.eos_token_id) : int32_t{0};

    validate_prompt_tokens_or_throw(prompts, V, "GenerationEngine::generate_grpo");
    // Generation can receive prompts longer than the trainer's static T.
    // Ensure RoPE cache covers all positions touched by prefill+decode.
    mRunState.ensure_rope_freq_capacity(mConfig, max_total_len);

    auto arena_cp = arena.checkpoint();

    // ========================================================================
    // Allocate paged KV-cache for M×N sequences with shared prefix pages
    // ========================================================================
    constexpr int kPageBlockSize = 256;
    const KVDType kv_dtype =
        supports_fp8_kv_cache(mRunState.DeviceProp) ? KVDType::FP8_E4M3 : KVDType::BF16;
    const bool fp8_kv_cache = kv_dtype == KVDType::FP8_E4M3;
    PagedKVCache paged_kv;
    paged_kv.configure(M, N, num_layers, max_prompt_len, gen_config.max_gen_len,
                       Hkv, Hs, kPageBlockSize, kv_dtype);
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

    // Cross-prompt full-page prefix sharing:
    // If two prompts share the same leading page-aligned prefix, later prompts
    // can alias those KV pages and skip prefill on the shared leading pages.
    // Keep this disabled for recurrent/Mamba models where hidden state must
    // also be carried across the skipped region.
    const bool has_recurrent_prefix_state = [&]() -> bool {
        if (mConfig.MambaSsmStateSize > 0 || mConfig.MambaConvKernel > 1 || mConfig.MambaNumHeads > 0) {
            return true;
        }
        for (int layer = 0; layer < mConfig.NumLayers; ++layer) {
            if (mConfig.get_block_type(layer) == modules::BlockType::Mamba) {
                return true;
            }
        }
        return false;
    }();
    const bool enable_cross_prompt_prefix_sharing = !has_recurrent_prefix_state;

    std::vector<int> share_from_prompt(static_cast<std::size_t>(M), -1);
    std::vector<int> shared_full_pages_per_prompt(static_cast<std::size_t>(M), 0);
    if (enable_cross_prompt_prefix_sharing) {
        for (int m = 0; m < M; ++m) {
            const int full_pages_m = prefill_lens_per_prompt[static_cast<std::size_t>(m)] / kPageBlockSize;
            if (full_pages_m <= 0) {
                continue;
            }

            int best_src = -1;
            int best_shared_pages = 0;
            for (int src = 0; src < m; ++src) {
                const int full_pages_src = prefill_lens_per_prompt[static_cast<std::size_t>(src)] / kPageBlockSize;
                const int max_pages = std::min(full_pages_m, full_pages_src);
                int shared_pages = 0;
                while (shared_pages < max_pages) {
                    const int off = shared_pages * kPageBlockSize;
                    if (!std::equal(
                            prompts[static_cast<std::size_t>(m)].begin() + off,
                            prompts[static_cast<std::size_t>(m)].begin() + off + kPageBlockSize,
                            prompts[static_cast<std::size_t>(src)].begin() + off)) {
                        break;
                    }
                    ++shared_pages;
                }
                if (shared_pages > best_shared_pages) {
                    best_shared_pages = shared_pages;
                    best_src = src;
                }
            }

            if (best_shared_pages > 0) {
                share_from_prompt[static_cast<std::size_t>(m)] = best_src;
                shared_full_pages_per_prompt[static_cast<std::size_t>(m)] = best_shared_pages;
            }
        }
    }

    // Block table GPU buffer for B=1 prefill
    auto* prefill_block_table_gpu = reinterpret_cast<int*>(
        alloc(static_cast<std::size_t>(paged_kv.max_pages_per_seq) * sizeof(int), "grpo_prefill_bt"));
    std::vector<int32_t> prefill_tokens_chunk(
        static_cast<std::size_t>(std::max(1, prefill_chunk_size)), 0);
    std::vector<int32_t> prefill_pos_chunk(
        static_cast<std::size_t>(std::max(1, prefill_chunk_size)), 0);

    // Shared recurrent/conv state maps for decode.
    // Recurrent state is prefilled per-prompt (B=1) then scattered into these
    // full-batch [M*N, H, K, V] buffers before decode starts.
    std::unordered_map<int, void*> grpo_recurrent_states;
    std::unordered_map<int, std::size_t> grpo_recurrent_state_bytes;
    std::unordered_map<int, void*> grpo_conv_states;
    std::unordered_map<int, std::size_t> grpo_conv_state_bytes;

    // ========================================================================
    // Prefill: for each prompt, run prefix-only forward (all tokens except the
    // last one) to populate shared prefix pages.
    // ========================================================================
    for (int m = 0; m < M; ++m) {
        const auto& prompt = prompts[static_cast<std::size_t>(m)];
        const int plen = static_cast<int>(prompt.size());
        const int prefill_len = prefill_lens_per_prompt[static_cast<std::size_t>(m)];

        const int full_prefix_pages = prefill_len / kPageBlockSize;
        const int partial_prefix_tokens = prefill_len % kPageBlockSize;
        const int src_slot = m * N;
        const int shared_full_pages = std::min(
            full_prefix_pages,
            shared_full_pages_per_prompt[static_cast<std::size_t>(m)]);
        const int share_src_prompt = share_from_prompt[static_cast<std::size_t>(m)];

        // Share leading full prefix pages from an earlier prompt with the same
        // page-aligned prefix, then allocate the remaining full pages.
        if (shared_full_pages > 0 && share_src_prompt >= 0) {
            const int share_src_slot = share_src_prompt * N;
            for (int p = 0; p < shared_full_pages; ++p) {
                paged_kv.set_block_table(src_slot, p, paged_kv.get_block_table(share_src_slot, p));
            }
        }
        for (int p = shared_full_pages; p < full_prefix_pages; ++p) {
            paged_kv.set_block_table(src_slot, p, paged_kv.alloc_page());
        }
        int partial_virtual_page = -1;
        if (partial_prefix_tokens > 0) {
            partial_virtual_page = full_prefix_pages;
            paged_kv.set_block_table(src_slot, partial_virtual_page, paged_kv.alloc_page());
        }

        // Share only full prefix pages.
        if (full_prefix_pages > 0) {
            paged_kv.share_prefix_pages(m, full_prefix_pages);
        }

        // Materialize private copies for the last partially-filled prefix page.
        // Sharing that page would alias continuation writes across completions.
        if (partial_prefix_tokens > 0) {
            for (int n = 1; n < N; ++n) {
                const int dst_slot = src_slot + n;
                paged_kv.set_block_table(dst_slot, partial_virtual_page, paged_kv.alloc_page());
            }
        }

        // Upload block table for source slot (B=1)
        CUDA_CHECK(cudaMemcpyAsync(
            prefill_block_table_gpu,
            paged_kv.block_table_host.data() + static_cast<std::size_t>(src_slot) * paged_kv.max_pages_per_seq,
            static_cast<std::size_t>(paged_kv.max_pages_per_seq) * sizeof(int),
            cudaMemcpyHostToDevice, mRunState.MainStream));

        // Per-prompt recurrent-state capture during prefill (B=1 source slot).
        std::unordered_map<int, void*> prompt_recurrent_states;
        std::unordered_map<int, std::size_t> prompt_recurrent_state_bytes;
        std::unordered_map<int, void*> prompt_conv_states;
        std::unordered_map<int, std::size_t> prompt_conv_state_bytes;

        // Prefill decode state (B=1, paged, prefill_mode=true)
        infer::DecodeState prompt_ds{};
        prompt_ds.paged = true;
        prompt_ds.fp8 = fp8_kv_cache;
        prompt_ds.k_pages = paged_kv.k_pages;
        prompt_ds.v_pages = paged_kv.v_pages;
        prompt_ds.k_scales = paged_kv.k_scales;
        prompt_ds.v_scales = paged_kv.v_scales;
        prompt_ds.k_scales_paged_fp8 = paged_kv.k_scales;
        prompt_ds.v_scales_paged_fp8 = paged_kv.v_scales;
        prompt_ds.per_pool_bytes = paged_kv.per_pool_bytes();
        prompt_ds.block_table_gpu = prefill_block_table_gpu;
        prompt_ds.block_table_stride = paged_kv.max_pages_per_seq;
        prompt_ds.page_block_size = kPageBlockSize;
        prompt_ds.total_pages = paged_kv.total_pages;
        prompt_ds.num_kv_heads = Hkv;
        prompt_ds.head_dim = Hs;
        prompt_ds.logits_out_gpu = logits_gpu;
        prompt_ds.vocab_size = V;
        prompt_ds.recurrent_states = &prompt_recurrent_states;
        prompt_ds.recurrent_state_bytes = &prompt_recurrent_state_bytes;
        prompt_ds.conv_states = &prompt_conv_states;
        prompt_ds.conv_state_bytes = &prompt_conv_state_bytes;
        prompt_ds.prefill_pos_offset = 0;
        // prefill_mode is set by execute_prefill()

        const int prefill_start = shared_full_pages * kPageBlockSize;
        if (prefill_len > prefill_start) {
            const int chunk_size = std::max(1, prefill_chunk_size);
            for (int chunk_start = prefill_start; chunk_start < prefill_len; chunk_start += chunk_size) {
                const int chunk_len = std::min(chunk_size, prefill_len - chunk_start);
                for (int t = 0; t < chunk_len; ++t) {
                    const int absolute_t = chunk_start + t;
                    prefill_tokens_chunk[static_cast<std::size_t>(t)] =
                        prompt[static_cast<std::size_t>(absolute_t)];
                    prefill_pos_chunk[static_cast<std::size_t>(t)] = absolute_t;
                }
                prompt_ds.prefill_pos_offset = chunk_start;

                graph_executor.execute_prefill(
                    1L, static_cast<long>(chunk_len),
                    prefill_tokens_chunk.data(), prefill_pos_chunk.data(),
                    prompt_ds, comm, hook);
            }
        }

        // Scatter per-prompt recurrent state [1,H,K,V] to each completion slot
        // in the decode batch buffer [M*N,H,K,V].
        for (const auto& [layer_idx, src_ptr] : prompt_recurrent_states) {
            if (!src_ptr) {
                continue;
            }
            auto it_bytes = prompt_recurrent_state_bytes.find(layer_idx);
            if (it_bytes == prompt_recurrent_state_bytes.end() || it_bytes->second == 0) {
                throw std::runtime_error(
                    "GenerationEngine::generate_grpo: missing recurrent state bytes for layer "
                    + std::to_string(layer_idx));
            }
            const std::size_t per_seq_bytes = it_bytes->second;

            void*& dst_base = grpo_recurrent_states[layer_idx];
            auto it_global_bytes = grpo_recurrent_state_bytes.find(layer_idx);
            if (!dst_base) {
                const std::size_t total_bytes =
                    per_seq_bytes * static_cast<std::size_t>(total_batch);
                CUDA_CHECK(cudaMalloc(&dst_base, total_bytes));
                CUDA_CHECK(cudaMemsetAsync(dst_base, 0, total_bytes, mRunState.MainStream));
                grpo_recurrent_state_bytes[layer_idx] = per_seq_bytes;
            } else if (it_global_bytes == grpo_recurrent_state_bytes.end() ||
                       it_global_bytes->second != per_seq_bytes) {
                throw std::runtime_error(
                    "GenerationEngine::generate_grpo: recurrent state byte-size mismatch for layer "
                    + std::to_string(layer_idx));
            }

            for (int n = 0; n < N; ++n) {
                const int dst_slot = src_slot + n;
                auto* dst_ptr = static_cast<char*>(dst_base)
                              + static_cast<std::size_t>(dst_slot) * per_seq_bytes;
                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, per_seq_bytes,
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            }
        }

        for (auto& [_, ptr] : prompt_recurrent_states) {
            if (ptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
        }

        // Scatter per-prompt causal-conv state [1, conv_dim, kernel-1] into
        // decode batch buffer [M*N, conv_dim, kernel-1].
        for (const auto& [layer_idx, src_ptr] : prompt_conv_states) {
            if (!src_ptr) {
                continue;
            }
            auto it_bytes = prompt_conv_state_bytes.find(layer_idx);
            if (it_bytes == prompt_conv_state_bytes.end() || it_bytes->second == 0) {
                throw std::runtime_error(
                    "GenerationEngine::generate_grpo: missing conv state bytes for layer "
                    + std::to_string(layer_idx));
            }
            const std::size_t per_seq_bytes = it_bytes->second;

            void*& dst_base = grpo_conv_states[layer_idx];
            auto it_global_bytes = grpo_conv_state_bytes.find(layer_idx);
            if (!dst_base) {
                const std::size_t total_bytes =
                    per_seq_bytes * static_cast<std::size_t>(total_batch);
                CUDA_CHECK(cudaMalloc(&dst_base, total_bytes));
                CUDA_CHECK(cudaMemsetAsync(dst_base, 0, total_bytes, mRunState.MainStream));
                grpo_conv_state_bytes[layer_idx] = per_seq_bytes;
            } else if (it_global_bytes == grpo_conv_state_bytes.end() ||
                       it_global_bytes->second != per_seq_bytes) {
                throw std::runtime_error(
                    "GenerationEngine::generate_grpo: conv state byte-size mismatch for layer "
                    + std::to_string(layer_idx));
            }

            for (int n = 0; n < N; ++n) {
                const int dst_slot = src_slot + n;
                auto* dst_ptr = static_cast<char*>(dst_base)
                              + static_cast<std::size_t>(dst_slot) * per_seq_bytes;
                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, per_seq_bytes,
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            }
        }

        for (auto& [_, ptr] : prompt_conv_states) {
            if (ptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
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
    infer::DecodeState decode_state{};
    decode_state.recurrent_states = &grpo_recurrent_states;
    decode_state.recurrent_state_bytes = &grpo_recurrent_state_bytes;
    decode_state.conv_states = &grpo_conv_states;
    decode_state.conv_state_bytes = &grpo_conv_state_bytes;
    decode_state.paged = true;
    decode_state.fp8 = fp8_kv_cache;
    decode_state.k_pages = paged_kv.k_pages;
    decode_state.v_pages = paged_kv.v_pages;
    decode_state.k_scales = paged_kv.k_scales;
    decode_state.v_scales = paged_kv.v_scales;
    decode_state.k_scales_paged_fp8 = paged_kv.k_scales;
    decode_state.v_scales_paged_fp8 = paged_kv.v_scales;
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
    decode_state.prefill_pos_offset = 0;

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
    bool decode_cuda_graphs_primed = !gen_config.use_cuda_graphs;

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
            decode_state, comm, hook,
            gen_config.use_cuda_graphs && decode_cuda_graphs_primed);
        if (!decode_cuda_graphs_primed && gen_config.use_cuda_graphs) {
            // Warm-up eager once so lazy allocations happen before first capture.
            decode_cuda_graphs_primed = true;
        }
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
            const bool unconstrained_sampling =
                (gen_config.top_k <= 0) && (gen_config.top_p >= 1.0f) && (gen_config.min_p <= 0.0f);
            if (unconstrained_sampling && gen_config.temperature == 1.0f) {
                // Fuse softmax + categorical sampling for the default generation path.
                sampling_from_logits(logits_gpu, sampled_tokens_gpu, total_batch, V,
                                     /*deterministic=*/false,
                                     gen_config.seed, rng_offset, mRunState.MainStream);
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
        }
        mask_finished_tokens(sampled_tokens_gpu, grpo_finished_gpu, total_batch, mRunState.MainStream);
        // Guarantee sampled IDs stay in-vocab before feedback / host export.
        sampling_sanitize_token_ids(
            sampled_tokens_gpu,
            total_batch,
            V,
            fallback_token_id,
            /*invalid_count=*/nullptr,
            mRunState.MainStream);
        rng_offset++;

        if (gen_config.temperature <= 0.0f) {
            // Greedy: compute log-softmax directly from logits (skip materializing probs).
            sampling_extract_logprob_from_logits(logits_gpu, sampled_tokens_gpu, logprobs_gpu,
                                                  nullptr, total_batch, V, mRunState.MainStream);
        } else if (gen_config.top_k <= 0 && gen_config.top_p >= 1.0f &&
                   gen_config.min_p <= 0.0f && gen_config.temperature == 1.0f) {
            // sampling_from_logits path above does not materialize probs.
            sampling_extract_logprob_from_logits(logits_gpu, sampled_tokens_gpu, logprobs_gpu,
                                                 nullptr, total_batch, V, mRunState.MainStream);
        } else {
            // Non-greedy: probs already computed by sampling_softmax above.
            sampling_extract_logprob(probs_gpu, sampled_tokens_gpu, logprobs_gpu,
                                     total_batch, V, mRunState.MainStream);
        }

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

    // Clear any prefix boundary that generation may have left, so
    // arena.restore() can fully rewind the stack to pre-generation state.
    arena.clear_prefix_boundary();
    arena.restore(arena_cp);

    return trajectories;
}

}  // namespace infer
