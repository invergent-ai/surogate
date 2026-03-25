// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "binding/py_train.h"
#include "serve/scheduler.h"
#include "tokenizer/tokenizer.h"
#include "third_party/cpp-httplib/httplib.h"

namespace serve {

struct NativeHttpServerConfig {
    std::string host = "0.0.0.0";
    int port = 8000;
    std::string model_id = "native";
    std::string api_key;
    std::string model_dir;

    int eos_token_id = 2;
    int max_context_len = 4096;
    int runtime_seq_len = 4096;

    // Scheduler (vLLM v1 style)
    int max_batch_sequences = 64;
    int max_num_batched_tokens = 2048;         // unified token budget per step
    int long_prefill_token_threshold = 0;      // 0 = auto (4% of max_seq_len)
    int max_num_partial_prefills = 1;          // max new requests per step

    int stream_batch_tokens = 16;

    // Engine
    float gpu_memory_utilization = 0.9f;
    bool use_cuda_graphs = true;
    int continuous_min_activation_mb = 256;
    int continuous_engine_max_seq_len = 4096;

    // Generation defaults
    int max_gen_len = 512;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;

    int multi_decode_steps = 1;
    int max_http_threads = 0;
    bool enable_loop_trace = false;
};

class NativeHttpServer {
public:
    NativeHttpServer(MultiGPUPyTrainer* trainer, NativeHttpServerConfig cfg);
    ~NativeHttpServer();

    void serve();
    void stop();

private:
    struct GenerationParams {
        int max_tokens = 1;
        float temperature = 1.0f;
        int top_k = 0;
        float top_p = 1.0f;
        float min_p = 0.0f;
        float repetition_penalty = 1.0f;
        bool ignore_eos = false;
    };

    struct GeneratedChoice {
        int index = 0;
        std::string text;
        std::vector<int32_t> token_ids;
        std::string finish_reason = "stop";
        int prompt_tokens = 0;
        int completion_tokens = 0;
    };

    struct TokenQueue {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<int32_t> q;
        bool closed = false;

        void push(int32_t tok);
        void close();
        bool try_pop(int32_t& tok);
        bool pop_blocking(int32_t& tok);
    };

    struct PendingGeneration {
        std::vector<int32_t> prompt_ids;
        std::vector<std::string> stop_strings;
        GenerationParams params;
        std::shared_ptr<TokenQueue> stream_queue;

        std::mutex mu;
        std::condition_variable cv;
        bool done = false;
        std::vector<GeneratedChoice> result;
        std::exception_ptr error;
    };

    struct ActiveGeneration {
        std::shared_ptr<PendingGeneration> pending;
        std::vector<int32_t> generated_ids;
        bool finished = false;
        std::string finish_reason = "stop";
        // Unified token tracking (mirrors vLLM's Request).
        int num_prompt_tokens = 0;     // total prompt length
        int num_computed_tokens = 0;   // advanced immediately after scheduling
        bool is_prefill_chunk = true;  // true while num_computed_tokens < num_prompt_tokens
    };

    struct ContextLengthError : public std::runtime_error {
        std::string param;
        ContextLengthError(std::string message, std::string param_name)
            : std::runtime_error(std::move(message)), param(std::move(param_name)) {}
    };

    // HTTP setup
    void setup_routes();
    bool authorize_or_401(const std::unordered_multimap<std::string, std::string>& headers,
                          std::string& out_body) const;

    // Request prep
    std::pair<std::vector<int32_t>, GenerationParams> prepare_chat_generation(
        const nlohmann::json& req,
        std::vector<std::string>& out_stop_strings) const;
    std::pair<std::vector<int32_t>, GenerationParams> prepare_completion_generation(
        const nlohmann::json& req,
        std::vector<std::string>& out_stop_strings) const;

    // Generation lifecycle
    std::shared_ptr<PendingGeneration> enqueue_generation(
        std::vector<int32_t> prompt_ids,
        const GenerationParams& params,
        std::vector<std::string> stop_strings,
        bool with_stream_queue);
    std::vector<GeneratedChoice> wait_generation(const std::shared_ptr<PendingGeneration>& pending) const;

    // Scheduler
    void scheduler_loop();
    void finalize_request(ActiveGeneration& item);
    void fail_pending(const std::shared_ptr<PendingGeneration>& pending, const std::exception_ptr& eptr) const;

    // Tokenization helpers
    std::string messages_to_prompt_text(const nlohmann::json& messages) const;
    std::vector<int32_t> sanitize_token_ids(const std::vector<int32_t>& ids, bool& changed) const;
    int32_t sanitize_token(int32_t tok, bool& replaced) const;
    std::string safe_decode(const std::vector<int32_t>& ids) const;
    std::string strip_thinking_preamble(const std::string& text) const;
    std::pair<std::string, bool> apply_stop(const std::string& text, const std::vector<std::string>& stop_strings) const;
    std::pair<std::vector<int32_t>, int> fit_prompt_and_max_tokens(
        const std::vector<int32_t>& prompt_ids,
        int max_tokens,
        const std::string& param_name) const;

    static std::vector<std::string> normalize_stop(const nlohmann::json& req);

    // JSON helpers
    static nlohmann::json usage_from_choices(const std::vector<GeneratedChoice>& choices);
    static std::string now_unix_seconds_string();
    static int now_unix_seconds();
    std::string next_request_id(const char* prefix);

    MultiGPUPyTrainer* trainer_;
    NativeHttpServerConfig cfg_;

    tokenizer::Tokenizer tokenizer_;
    int32_t eos_id_ = 2;
    int32_t token_id_limit_ = 0;
    int32_t fallback_token_id_ = 0;

    mutable std::atomic<uint64_t> next_request_id_{1};

    mutable std::mutex pending_mu_;
    std::condition_variable pending_cv_;
    std::deque<std::shared_ptr<PendingGeneration>> pending_;

    std::atomic<bool> shutdown_{false};
    std::atomic<bool> scheduler_exited_{false};
    std::atomic<bool> scheduler_failed_{false};
    mutable std::mutex scheduler_error_mu_;
    std::string scheduler_error_msg_;
    std::thread scheduler_thread_;
    std::atomic<std::uint64_t> engine_id_{0};

    // Direct inference context — bypasses MultiGPUPyTrainer::run_work for the hot path.
    // Extracted once after engine creation, used by scheduler + collect threads.
    MultiGPUPyTrainer::InferenceContext infer_ctx_{};

    // Unified scheduler (vLLM v1 style).
    std::unique_ptr<Scheduler> scheduler_;

    std::unique_ptr<httplib::Server> server_;
};

}  // namespace serve
