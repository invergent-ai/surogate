// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "serve/native_http_server.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <csignal>
#include <unistd.h>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "third_party/cpp-httplib/httplib.h"
#include "runtime/infer/continuous_engine.h"
#include "utilities/utils.h"  // CUDA_CHECK

namespace serve {

namespace {

// Global pointer for signal handler — only one NativeHttpServer can be active.
std::atomic<NativeHttpServer*> g_active_server{nullptr};

void signal_handler(int sig) {
    const char* name = (sig == SIGINT) ? "SIGINT" : "SIGTERM";
    // write() is async-signal-safe, fprintf/cerr are not.
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "\n[surogate][cpp-http] received %s, stopping...\n", name);
    if (len > 0) {
        auto ignored = write(STDERR_FILENO, buf, static_cast<size_t>(len));
        (void)ignored;
    }
    auto* srv = g_active_server.load(std::memory_order_relaxed);
    if (srv) {
        srv->stop();
    }
}

using json = nlohmann::json;

inline bool env_flag_enabled(const char* name, bool default_value = false) {
    const char* v = std::getenv(name);
    if (!v || !*v) {
        return default_value;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "0" || s == "false" || s == "no" || s == "off") {
        return false;
    }
    return true;
}

inline int env_int(const char* name, int default_value, int min_value = std::numeric_limits<int>::min()) {
    const char* v = std::getenv(name);
    if (!v || !*v) {
        return default_value;
    }
    try {
        const int parsed = std::stoi(v);
        return std::max(min_value, parsed);
    } catch (...) {
        return default_value;
    }
}

inline std::string json_dump(const json& v) {
    return v.dump(-1, ' ', false, json::error_handler_t::replace);
}

inline std::string sse_data(const json& payload) {
    return std::string("data: ") + json_dump(payload) + "\n\n";
}

inline std::string sse_done() {
    return "data: [DONE]\n\n";
}

inline std::string header_value_ci(const std::unordered_multimap<std::string, std::string>& headers,
                                   const std::string& key) {
    for (const auto& kv : headers) {
        if (kv.first.size() != key.size()) {
            continue;
        }
        bool equal = true;
        for (size_t i = 0; i < key.size(); ++i) {
            const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(kv.first[i])));
            const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(key[i])));
            if (a != b) {
                equal = false;
                break;
            }
        }
        if (equal) {
            return kv.second;
        }
    }
    return "";
}

}  // namespace

void NativeHttpServer::TokenQueue::push(int32_t tok) {
    std::lock_guard<std::mutex> lock(mu);
    if (closed) {
        return;
    }
    q.push_back(tok);
    cv.notify_one();
}

void NativeHttpServer::TokenQueue::close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cv.notify_all();
}

bool NativeHttpServer::TokenQueue::try_pop(int32_t& tok) {
    std::lock_guard<std::mutex> lock(mu);
    if (q.empty()) {
        return false;
    }
    tok = q.front();
    q.pop_front();
    return true;
}

bool NativeHttpServer::TokenQueue::pop_blocking(int32_t& tok) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&]() { return closed || !q.empty(); });
    if (q.empty()) {
        return false;
    }
    tok = q.front();
    q.pop_front();
    return true;
}

NativeHttpServer::NativeHttpServer(MultiGPUPyTrainer* trainer, NativeHttpServerConfig cfg)
    : trainer_(trainer), cfg_(std::move(cfg)), tokenizer_(tokenizer::Tokenizer::from_pretrained(cfg_.model_dir)) {
    if (!trainer_) {
        throw std::invalid_argument("NativeHttpServer: trainer is null");
    }
    if (cfg_.model_dir.empty()) {
        throw std::invalid_argument("NativeHttpServer: model_dir is required");
    }

    eos_id_ = cfg_.eos_token_id > 0 ? cfg_.eos_token_id : tokenizer_.eos_token_id();
    if (eos_id_ < 0) {
        eos_id_ = 2;
    }
    token_id_limit_ = std::max(1, tokenizer_.vocab_size());

    int32_t fallback = tokenizer_.encode_single_token("<unk>");
    if (fallback < 0 || fallback >= token_id_limit_) {
        fallback = eos_id_;
    }
    if (fallback < 0 || fallback >= token_id_limit_) {
        fallback = 0;
    }
    fallback_token_id_ = fallback;

    if (cfg_.continuous_engine_max_seq_len <= 0) {
        cfg_.continuous_engine_max_seq_len = std::max(2, cfg_.runtime_seq_len);
    }
}

NativeHttpServer::~NativeHttpServer() {
    stop();
}

void NativeHttpServer::stop() {
    shutdown_.store(true);
    pending_cv_.notify_all();
    if (server_) {
        server_->stop();
    }
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }
}

void NativeHttpServer::serve() {
    shutdown_.store(false);
    scheduler_exited_.store(false);
    scheduler_failed_.store(false);
    {
        std::lock_guard<std::mutex> lock(scheduler_error_mu_);
        scheduler_error_msg_.clear();
    }

    // Install signal handlers so Ctrl+C cleanly stops the server.
    g_active_server.store(this, std::memory_order_relaxed);
    const auto prev_sigint = std::signal(SIGINT, signal_handler);
    const auto prev_sigterm = std::signal(SIGTERM, signal_handler);

    server_ = std::make_unique<httplib::Server>();
    server_->set_default_headers({{"Server", "surogate-cpp"}});
    if (cfg_.max_http_threads > 0) {
        const int n = cfg_.max_http_threads;
        server_->new_task_queue = [n] { return new httplib::ThreadPool(n); };
    }
    setup_routes();

    scheduler_thread_ = std::thread([this]() { this->scheduler_loop(); });

    std::cerr << fmt::format(
        "[surogate][cpp-http] listening on {}:{} model={} max_seq={} max_batch_seqs={}\n",
        cfg_.host,
        cfg_.port,
        cfg_.model_id,
        cfg_.runtime_seq_len,
        cfg_.max_batch_sequences);

    const bool ok = server_->listen(cfg_.host.c_str(), cfg_.port);
    if (!ok && !shutdown_.load()) {
        stop();
        // Restore previous signal handlers before throwing.
        std::signal(SIGINT, prev_sigint);
        std::signal(SIGTERM, prev_sigterm);
        g_active_server.store(nullptr, std::memory_order_relaxed);
        throw std::runtime_error(fmt::format(
            "NativeHttpServer: failed to listen on {}:{}",
            cfg_.host,
            cfg_.port));
    }

    stop();

    // Restore previous signal handlers.
    std::signal(SIGINT, prev_sigint);
    std::signal(SIGTERM, prev_sigterm);
    g_active_server.store(nullptr, std::memory_order_relaxed);
}

std::string NativeHttpServer::next_request_id(const char* prefix) {
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    return fmt::format("{}-{:x}{:x}", prefix, static_cast<unsigned long long>(now), static_cast<unsigned long long>(id));
}

int NativeHttpServer::now_unix_seconds() {
    return static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string NativeHttpServer::now_unix_seconds_string() {
    return std::to_string(now_unix_seconds());
}

bool NativeHttpServer::authorize_or_401(
    const std::unordered_multimap<std::string, std::string>& headers,
    std::string& out_body) const {
    if (cfg_.api_key.empty()) {
        return true;
    }
    std::string auth = header_value_ci(headers, "authorization");
    if (auth == ("Bearer " + cfg_.api_key)) {
        return true;
    }
    out_body = json_dump(json{
        {"error", {
            {"message", "Invalid API key"},
            {"type", "invalid_request_error"},
            {"code", "unauthorized"}
        }}
    });
    return false;
}

std::vector<std::string> NativeHttpServer::normalize_stop(const json& req) {
    std::vector<std::string> out;
    const auto it = req.find("stop");
    if (it == req.end() || it->is_null()) {
        return out;
    }
    if (it->is_string()) {
        const std::string s = it->get<std::string>();
        if (!s.empty()) {
            out.push_back(s);
        }
        return out;
    }
    if (it->is_array()) {
        for (const auto& v : *it) {
            if (!v.is_string()) {
                continue;
            }
            const std::string s = v.get<std::string>();
            if (!s.empty()) {
                out.push_back(s);
            }
        }
    }
    return out;
}

std::string NativeHttpServer::messages_to_prompt_text(const json& messages) const {
    std::vector<tokenizer::ChatMessage> normalized;
    normalized.reserve(messages.size());
    for (const auto& m : messages) {
        tokenizer::ChatMessage msg;
        msg.role = m.value("role", "user");

        const auto content_it = m.find("content");
        if (content_it == m.end() || content_it->is_null()) {
            msg.content.clear();
        } else if (content_it->is_string()) {
            msg.content = content_it->get<std::string>();
        } else if (content_it->is_array()) {
            std::string content;
            bool first = true;
            for (const auto& part : *content_it) {
                std::string piece;
                if (part.is_string()) {
                    piece = part.get<std::string>();
                } else if (part.is_object()) {
                    if (part.contains("text") && part["text"].is_string()) {
                        piece = part["text"].get<std::string>();
                    } else if (part.value("type", "") == "text" && part.contains("text") && part["text"].is_string()) {
                        piece = part["text"].get<std::string>();
                    }
                }
                if (piece.empty()) {
                    continue;
                }
                if (!first) {
                    content.push_back('\n');
                }
                first = false;
                content += piece;
            }
            msg.content = std::move(content);
        } else {
            msg.content = content_it->dump();
        }

        normalized.push_back(std::move(msg));
    }

    try {
        return tokenizer_.apply_chat_template(normalized, true);
    } catch (...) {
        std::ostringstream oss;
        for (const auto& m : normalized) {
            oss << m.role << ": " << m.content << "\n";
        }
        oss << "assistant:";
        return oss.str();
    }
}

std::pair<std::vector<int32_t>, int> NativeHttpServer::fit_prompt_and_max_tokens(
    const std::vector<int32_t>& prompt_ids,
    int max_tokens,
    const std::string& param_name) const {
    const int runtime_seq_len = std::max(2, cfg_.runtime_seq_len);
    const int requested = std::max(1, max_tokens);
    const int prompt_len = static_cast<int>(prompt_ids.size());
    const int total = prompt_len + requested;
    if (total > runtime_seq_len) {
        throw ContextLengthError(
            fmt::format(
                "This model's maximum context length is {} tokens, but you requested {} tokens "
                "({} in the prompt; {} for the completion). Please reduce the prompt or max_tokens.",
                runtime_seq_len,
                total,
                prompt_len,
                requested),
            param_name);
    }
    return {prompt_ids, requested};
}

std::pair<std::vector<int32_t>, NativeHttpServer::GenerationParams>
NativeHttpServer::prepare_chat_generation(
    const json& req,
    std::vector<std::string>& out_stop_strings) const {
    const std::string requested_model = req.value("model", "");
    if (!requested_model.empty() && requested_model != cfg_.model_id) {
        throw std::invalid_argument(fmt::format("Unknown model '{}'", requested_model));
    }

    const int n = req.value("n", 1);
    if (n <= 0) {
        throw std::invalid_argument("`n` must be > 0");
    }
    if (n != 1) {
        throw std::invalid_argument("`n>1` is not supported in native C++ server yet");
    }

    const auto messages_it = req.find("messages");
    if (messages_it == req.end() || !messages_it->is_array() || messages_it->empty()) {
        throw std::invalid_argument("`messages` must be a non-empty array");
    }

    const std::string prompt_text = messages_to_prompt_text(*messages_it);
    auto prompt_ids = tokenizer_.encode(prompt_text, false);

    int max_tokens = cfg_.max_gen_len;
    if (req.contains("max_completion_tokens") && req["max_completion_tokens"].is_number_integer()) {
        max_tokens = req["max_completion_tokens"].get<int>();
    } else if (req.contains("max_tokens") && req["max_tokens"].is_number_integer()) {
        max_tokens = req["max_tokens"].get<int>();
    }

    const float temperature = req.value("temperature", cfg_.temperature);
    const float top_p = req.value("top_p", cfg_.top_p);
    const int top_k = req.value("top_k", cfg_.top_k);
    const float min_p = req.value("min_p", cfg_.min_p);
    const float repetition_penalty = req.value("repetition_penalty", cfg_.repetition_penalty);
    const bool ignore_eos = req.value("ignore_eos", false);

    auto fit = fit_prompt_and_max_tokens(prompt_ids, max_tokens, "messages");

    GenerationParams params;
    params.max_tokens = std::max(1, fit.second);
    params.temperature = temperature;
    params.top_k = std::max(0, top_k);
    params.top_p = top_p > 0.0f ? top_p : 1.0f;
    params.min_p = std::max(0.0f, min_p);
    params.repetition_penalty = repetition_penalty > 0.0f ? repetition_penalty : 1.0f;
    params.ignore_eos = ignore_eos;

    out_stop_strings = normalize_stop(req);
    return {std::move(fit.first), params};
}

std::pair<std::vector<int32_t>, NativeHttpServer::GenerationParams>
NativeHttpServer::prepare_completion_generation(
    const json& req,
    std::vector<std::string>& out_stop_strings) const {
    const std::string requested_model = req.value("model", "");
    if (!requested_model.empty() && requested_model != cfg_.model_id) {
        throw std::invalid_argument(fmt::format("Unknown model '{}'", requested_model));
    }

    const int n = req.value("n", 1);
    if (n <= 0) {
        throw std::invalid_argument("`n` must be > 0");
    }
    if (n != 1) {
        throw std::invalid_argument("`n>1` is not supported in native C++ server yet");
    }

    std::string prompt_text;
    const auto pit = req.find("prompt");
    if (pit == req.end()) {
        throw std::invalid_argument("`prompt` is required");
    }
    if (pit->is_string()) {
        prompt_text = pit->get<std::string>();
    } else if (pit->is_array()) {
        if (pit->empty()) {
            throw std::invalid_argument("`prompt` cannot be empty");
        }
        if (pit->size() > 1) {
            throw std::invalid_argument("This server supports one prompt per request");
        }
        if (!(*pit)[0].is_string()) {
            throw std::invalid_argument("`prompt[0]` must be a string");
        }
        prompt_text = (*pit)[0].get<std::string>();
    } else {
        throw std::invalid_argument("`prompt` must be string or [string]");
    }

    auto prompt_ids = tokenizer_.encode(prompt_text, true);
    const int max_tokens = req.value("max_tokens", cfg_.max_gen_len);

    const float temperature = req.value("temperature", cfg_.temperature);
    const float top_p = req.value("top_p", cfg_.top_p);
    const int top_k = req.value("top_k", cfg_.top_k);
    const float min_p = req.value("min_p", cfg_.min_p);
    const float repetition_penalty = req.value("repetition_penalty", cfg_.repetition_penalty);
    const bool ignore_eos = req.value("ignore_eos", false);

    auto fit = fit_prompt_and_max_tokens(prompt_ids, max_tokens, "prompt");

    GenerationParams params;
    params.max_tokens = std::max(1, fit.second);
    params.temperature = temperature;
    params.top_k = std::max(0, top_k);
    params.top_p = top_p > 0.0f ? top_p : 1.0f;
    params.min_p = std::max(0.0f, min_p);
    params.repetition_penalty = repetition_penalty > 0.0f ? repetition_penalty : 1.0f;
    params.ignore_eos = ignore_eos;

    out_stop_strings = normalize_stop(req);
    return {std::move(fit.first), params};
}

std::shared_ptr<NativeHttpServer::PendingGeneration> NativeHttpServer::enqueue_generation(
    std::vector<int32_t> prompt_ids,
    const GenerationParams& params,
    std::vector<std::string> stop_strings,
    bool with_stream_queue) {
    auto pending = std::make_shared<PendingGeneration>();
    pending->prompt_ids = std::move(prompt_ids);
    pending->params = params;
    pending->stop_strings = std::move(stop_strings);
    if (with_stream_queue) {
        pending->stream_queue = std::make_shared<TokenQueue>();
    }

    if (shutdown_.load(std::memory_order_relaxed)
        || scheduler_failed_.load(std::memory_order_relaxed)
        || scheduler_exited_.load(std::memory_order_relaxed)) {
        std::string msg = "native scheduler is not running";
        {
            std::lock_guard<std::mutex> lock(scheduler_error_mu_);
            if (!scheduler_error_msg_.empty()) {
                msg = fmt::format("native scheduler is not running: {}", scheduler_error_msg_);
            }
        }
        fail_pending(pending, std::make_exception_ptr(std::runtime_error(msg)));
        return pending;
    }

    {
        std::lock_guard<std::mutex> lock(pending_mu_);
        pending_.push_back(pending);
    }
    pending_cv_.notify_one();
    return pending;
}

std::vector<NativeHttpServer::GeneratedChoice> NativeHttpServer::wait_generation(
    const std::shared_ptr<PendingGeneration>& pending) const {
    std::unique_lock<std::mutex> lock(pending->mu);
    pending->cv.wait(lock, [&]() { return pending->done; });
    if (pending->error) {
        std::rethrow_exception(pending->error);
    }
    return pending->result;
}

void NativeHttpServer::fail_pending(
    const std::shared_ptr<PendingGeneration>& pending,
    const std::exception_ptr& eptr) const {
    {
        std::lock_guard<std::mutex> lock(pending->mu);
        pending->error = eptr;
        pending->done = true;
    }
    pending->cv.notify_all();
    if (pending->stream_queue) {
        pending->stream_queue->close();
    }
}

std::vector<NativeHttpServer::ActiveGeneration> NativeHttpServer::drain_pending_for_prefill(
    int max_new,
    int max_prompt_tokens) {
    std::vector<ActiveGeneration> out;
    out.reserve(static_cast<size_t>(std::max(0, max_new)));

    int used_prompt_tokens = 0;
    const int token_budget = std::max(1, max_prompt_tokens);

    std::lock_guard<std::mutex> lock(pending_mu_);
    while (!pending_.empty() && static_cast<int>(out.size()) < max_new) {
        auto pending = pending_.front();
        const int prefill_tokens = std::max(1, static_cast<int>(pending->prompt_ids.size()) - 1);
        if (!out.empty() && (used_prompt_tokens + prefill_tokens > token_budget)) {
            break;
        }
        pending_.pop_front();
        out.push_back(ActiveGeneration{pending});
        used_prompt_tokens += prefill_tokens;
        if (used_prompt_tokens >= token_budget) {
            break;
        }
    }

    return out;
}

int32_t NativeHttpServer::sanitize_token(int32_t tok, bool& replaced) const {
    if (tok >= 0 && tok < token_id_limit_) {
        replaced = false;
        return tok;
    }
    replaced = true;
    return fallback_token_id_;
}

std::vector<int32_t> NativeHttpServer::sanitize_token_ids(const std::vector<int32_t>& ids, bool& changed) const {
    std::vector<int32_t> out;
    out.reserve(ids.size());
    changed = false;
    for (int32_t tok : ids) {
        bool replaced = false;
        out.push_back(sanitize_token(tok, replaced));
        changed = changed || replaced;
    }
    return out;
}

std::string NativeHttpServer::safe_decode(const std::vector<int32_t>& ids) const {
    if (ids.empty()) {
        return {};
    }
    bool changed = false;
    auto cleaned = sanitize_token_ids(ids, changed);
    std::vector<int32_t> visible;
    visible.reserve(cleaned.size());
    for (int32_t tok : cleaned) {
        if (tok == eos_id_) {
            continue;
        }
        if (tokenizer_.is_special_token(tok)) {
            continue;
        }
        visible.push_back(tok);
    }
    try {
        return tokenizer_.decode(visible);
    } catch (...) {
        return {};
    }
}

std::pair<std::string, bool> NativeHttpServer::apply_stop(
    const std::string& text,
    const std::vector<std::string>& stop_strings) const {
    if (stop_strings.empty()) {
        return {text, false};
    }
    size_t best_pos = std::string::npos;
    for (const auto& s : stop_strings) {
        if (s.empty()) {
            continue;
        }
        const auto pos = text.find(s);
        if (pos == std::string::npos) {
            continue;
        }
        if (best_pos == std::string::npos || pos < best_pos) {
            best_pos = pos;
        }
    }
    if (best_pos == std::string::npos) {
        return {text, false};
    }
    return {text.substr(0, best_pos), true};
}

void NativeHttpServer::finalize_request(ActiveGeneration& item) {
    bool had_invalid = false;
    auto clean_ids = sanitize_token_ids(item.generated_ids, had_invalid);
    const std::string raw_text = safe_decode(clean_ids);
    auto [trimmed_text, stop_hit] = apply_stop(raw_text, item.pending->stop_strings);

    std::vector<int32_t> out_ids;
    if (stop_hit) {
        out_ids = tokenizer_.encode(trimmed_text, false);
    } else {
        out_ids = std::move(clean_ids);
    }

    std::string finish_reason = item.finish_reason;
    if (stop_hit) {
        finish_reason = "stop";
    } else if (had_invalid) {
        finish_reason = "length";
    }

    GeneratedChoice c;
    c.index = 0;
    c.text = std::move(trimmed_text);
    c.token_ids = std::move(out_ids);
    c.finish_reason = std::move(finish_reason);
    c.prompt_tokens = static_cast<int>(item.pending->prompt_ids.size());
    c.completion_tokens = static_cast<int>(c.token_ids.size());

    {
        std::lock_guard<std::mutex> lock(item.pending->mu);
        item.pending->result = {std::move(c)};
        item.pending->done = true;
    }
    item.pending->cv.notify_all();
    if (item.pending->stream_queue) {
        item.pending->stream_queue->close();
    }
}

void NativeHttpServer::scheduler_loop() {
    std::uint64_t engine_id = 0;
    std::unordered_map<int, ActiveGeneration> slot_map;
    scheduler_exited_.store(false, std::memory_order_relaxed);
    scheduler_failed_.store(false, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(scheduler_error_mu_);
        scheduler_error_msg_.clear();
    }

    try {
        engine_id = trainer_->create_continuous_engine(
            cfg_.max_batch_sequences,
            cfg_.continuous_engine_max_seq_len,
            cfg_.gpu_memory_utilization,
            cfg_.use_cuda_graphs,
            cfg_.continuous_min_activation_mb,
            cfg_.max_num_batched_tokens);
        engine_id_.store(engine_id, std::memory_order_relaxed);

        // Extract raw pointers for direct engine access (bypasses run_work).
        infer_ctx_ = trainer_->extract_inference_context(engine_id);
        CUDA_CHECK(cudaSetDevice(infer_ctx_.device_id));

        auto* engine = infer_ctx_.engine;
        auto* graph_exec = infer_ctx_.graph_executor;
        auto* nccl_comm = infer_ctx_.communicator;

        // Collector thread calls cudaEventSynchronize + Phase 8 directly on its
        // own thread — no run_work serialization. This is the key difference from
        // the old design: the scheduler can launch the next step while the
        // collector is still blocking on the previous step's GPU completion.
        std::mutex collect_mu;
        std::condition_variable_any collect_cv;
        bool collect_requested = false;
        bool collect_inflight = false;
        bool collect_ready = false;
        bool collect_stop = false;
        std::exception_ptr collect_error;
        std::optional<infer::ContinuousGenerationEngine::FlatStepResult> collected_step;

        std::jthread collect_thread([&](std::stop_token st) {
            CUDA_CHECK(cudaSetDevice(infer_ctx_.device_id));
            while (true) {
                std::unique_lock<std::mutex> lock(collect_mu);
                const bool ready = collect_cv.wait(lock, st, [&]() { return collect_stop || collect_requested; });
                if (!ready || collect_stop) {
                    break;
                }
                collect_requested = false;
                collect_inflight = true;
                lock.unlock();

                try {
                    // Direct call — cudaEventSynchronize happens on THIS thread,
                    // not the GPU worker thread. No run_work round-trip.
                    auto step = engine->flat_step_collect();
                    lock.lock();
                    collected_step = std::move(step);
                    collect_error = nullptr;
                    collect_ready = true;
                    collect_inflight = false;
                    lock.unlock();
                } catch (...) {
                    lock.lock();
                    collect_error = std::current_exception();
                    collect_ready = true;
                    collect_inflight = false;
                    lock.unlock();
                }
                collect_cv.notify_all();
            }
        });

        int step_count = 0;
        const bool sched_debug = env_flag_enabled("SUROGATE_SERVE_SCHED_DEBUG", false);

        while (!shutdown_.load(std::memory_order_relaxed)) {
            const auto t_loop_start = std::chrono::high_resolution_clock::now();

            // Pull collected step from collector thread (if ready).
            std::optional<infer::ContinuousGenerationEngine::FlatStepResult> collected;
            {
                std::lock_guard<std::mutex> lock(collect_mu);
                if (collect_ready) {
                    if (collect_error) {
                        std::rethrow_exception(collect_error);
                    }
                    collected = std::move(collected_step);
                    collected_step.reset();
                    collect_ready = false;
                }
            }

            if (collected.has_value()) {
                if (sched_debug) {
                    std::cerr << fmt::format(
                        "[sched-dbg] step={} collected: active_slots={} sampled={} finished={} completion_lens={}\n",
                        step_count,
                        collected->active_slot_ids.size(),
                        collected->sampled_tokens.size(),
                        collected->finished.size(),
                        collected->completion_lens.size());
                    for (size_t i = 0; i < collected->active_slot_ids.size(); ++i) {
                        std::cerr << fmt::format(
                            "[sched-dbg]   slot[{}] sid={} tok={} fin={} clen={} in_slot_map={}\n",
                            i,
                            collected->active_slot_ids[i],
                            i < collected->sampled_tokens.size() ? collected->sampled_tokens[i] : -1,
                            i < collected->finished.size() ? collected->finished[i] : -1,
                            i < collected->completion_lens.size() ? collected->completion_lens[i] : -1,
                            slot_map.count(collected->active_slot_ids[i]));
                    }
                }

                std::vector<int> release_slots;
                release_slots.reserve(collected->active_slot_ids.size());
                for (size_t i = 0; i < collected->active_slot_ids.size(); ++i) {
                    const int sid = collected->active_slot_ids[i];
                    auto it = slot_map.find(sid);
                    if (it == slot_map.end()) {
                        if (sched_debug) {
                            std::cerr << fmt::format("[sched-dbg]   sid={} NOT in slot_map, skipping\n", sid);
                        }
                        continue;
                    }
                    auto& item = it->second;
                    if (item.finished) {
                        if (sched_debug) {
                            std::cerr << fmt::format("[sched-dbg]   sid={} already finished, skipping\n", sid);
                        }
                        continue;
                    }

                    const int32_t raw_tok = collected->sampled_tokens.size() > i ? collected->sampled_tokens[i] : fallback_token_id_;
                    const int fin = collected->finished.size() > i ? collected->finished[i] : 0;
                    const int clen = collected->completion_lens.size() > i ? collected->completion_lens[i] : 0;

                    if (clen == 0 && !fin) {
                        if (sched_debug) {
                            std::cerr << fmt::format("[sched-dbg]   sid={} clen=0 fin=0 (prefilling), skipping\n", sid);
                        }
                        continue;
                    }

                    bool tok_replaced = false;
                    const int32_t emitted = sanitize_token(raw_tok, tok_replaced);
                    item.generated_ids.push_back(emitted);
                    if (item.pending->stream_queue) {
                        item.pending->stream_queue->push(emitted);
                    }

                    bool should_finish = false;
                    if (!item.pending->params.ignore_eos && emitted == eos_id_) {
                        should_finish = true;
                    }
                    if (tok_replaced) {
                        should_finish = true;
                    }
                    if (!should_finish && !item.pending->stop_strings.empty()) {
                        auto partial = safe_decode(item.generated_ids);
                        auto stop_res = apply_stop(partial, item.pending->stop_strings);
                        if (stop_res.second) {
                            should_finish = true;
                        }
                    }
                    if (!should_finish && static_cast<int>(item.generated_ids.size()) >= item.pending->params.max_tokens) {
                        should_finish = true;
                    }
                    if (!should_finish && fin && !item.pending->params.ignore_eos) {
                        should_finish = true;
                    }

                    if (should_finish) {
                        if (sched_debug) {
                            std::cerr << fmt::format(
                                "[sched-dbg]   sid={} FINISH tok={} gen_count={} has_stream={}\n",
                                sid, emitted, item.generated_ids.size(),
                                item.pending->stream_queue != nullptr);
                        }
                        item.finished = true;
                        if (!item.generated_ids.empty() && item.generated_ids.back() == eos_id_) {
                            item.finish_reason = "stop";
                        } else if (!item.pending->stop_strings.empty()) {
                            auto partial = safe_decode(item.generated_ids);
                            auto stop_res = apply_stop(partial, item.pending->stop_strings);
                            item.finish_reason = stop_res.second ? "stop" : "length";
                        } else {
                            item.finish_reason = "length";
                        }
                        finalize_request(item);
                        release_slots.push_back(sid);
                    }
                }

                if (sched_debug) {
                    std::cerr << fmt::format(
                        "[sched-dbg] step={} releasing {} slots, slot_map_before={}\n",
                        step_count, release_slots.size(), slot_map.size());
                }
                for (int sid : release_slots) {
                    engine->release_slot(sid);
                    slot_map.erase(sid);
                }
                if (sched_debug) {
                    std::cerr << fmt::format(
                        "[sched-dbg] step={} slot_map_after={}\n",
                        step_count, slot_map.size());
                }
            }

            // Check if a step is still inflight in the collector thread.
            bool step_inflight = false;
            {
                std::lock_guard<std::mutex> lock(collect_mu);
                step_inflight = collect_inflight || collect_requested || collect_ready;
            }

            // If a step is inflight, wait for it to complete before draining
            // new requests. Draining while inflight would pop requests from
            // pending_ that we can't launch yet, permanently losing them.
            if (step_inflight) {
                std::unique_lock<std::mutex> lock(collect_mu);
                collect_cv.wait(lock, [&]() {
                    return shutdown_.load(std::memory_order_relaxed) || collect_ready || collect_error;
                });
                continue;  // loop back to process collected results at the top
            }

            // Drain new requests only when we are ready to launch immediately.
            const int free_slots = std::max(0, cfg_.max_batch_sequences - static_cast<int>(slot_map.size()));
            std::size_t pending_size = 0;
            {
                std::lock_guard<std::mutex> lock(pending_mu_);
                pending_size = pending_.size();
            }
            auto prefill_items = drain_pending_for_prefill(
                std::max(0, std::min(free_slots, cfg_.prefill_max_new_sequences)),
                cfg_.prefill_budget_tokens);

            if (sched_debug && (!prefill_items.empty() || step_count % 100 == 0)) {
                std::cerr << fmt::format(
                    "[sched-dbg] step={} drain: free_slots={} pending_before={} drained={} slot_map={}\n",
                    step_count, free_slots, pending_size, prefill_items.size(), slot_map.size());
            }

            std::vector<std::vector<int32_t>> new_prompts;
            std::vector<int> new_max_tokens;
            new_prompts.reserve(prefill_items.size());
            new_max_tokens.reserve(prefill_items.size());
            for (auto& item : prefill_items) {
                int max_tokens = item.pending->params.max_tokens;
                const int prompt_len = static_cast<int>(item.pending->prompt_ids.size());
                if (prompt_len + max_tokens > cfg_.runtime_seq_len) {
                    max_tokens = std::max(1, cfg_.runtime_seq_len - prompt_len);
                }
                new_prompts.push_back(item.pending->prompt_ids);
                new_max_tokens.push_back(max_tokens);
            }

            if (slot_map.empty() && new_prompts.empty()) {
                if (sched_debug && pending_size > 0) {
                    std::cerr << fmt::format(
                        "[sched-dbg] step={} IDLE but pending_size={} free_slots={} — requests may be stuck!\n",
                        step_count, pending_size, free_slots);
                }
                std::unique_lock<std::mutex> lock(pending_mu_);
                pending_cv_.wait_for(lock, std::chrono::milliseconds(1), [&]() {
                    return shutdown_.load(std::memory_order_relaxed) || !pending_.empty();
                });
                continue;
            }

            GenerationParams first_params;
            if (!prefill_items.empty()) {
                first_params = prefill_items.front().pending->params;
            } else if (!slot_map.empty()) {
                first_params = slot_map.begin()->second.pending->params;
            }

            const bool any_ignore_eos = [&]() {
                for (const auto& kv : slot_map) {
                    if (kv.second.pending->params.ignore_eos) {
                        return true;
                    }
                }
                for (const auto& item : prefill_items) {
                    if (item.pending->params.ignore_eos) {
                        return true;
                    }
                }
                return false;
            }();
            const int32_t eos_for_step = any_ignore_eos ? -1 : eos_id_;
            const int max_gen = !new_max_tokens.empty()
                ? *std::max_element(new_max_tokens.begin(), new_max_tokens.end())
                : std::min(cfg_.max_gen_len, 128);

            // Launch flat_step — we are guaranteed !step_inflight here.
            if (sched_debug) {
                std::cerr << fmt::format(
                    "[sched-dbg] step={} LAUNCH: new_prompts={} slot_map={} max_gen={} eos={}\n",
                    step_count, new_prompts.size(), slot_map.size(), max_gen, eos_for_step);
            }
            // Direct engine call — no run_work serialization.
            infer::ContinuousGenerationEngine::FlatStepConfig step_cfg;
            step_cfg.max_gen_len = max_gen;
            step_cfg.temperature = first_params.temperature;
            step_cfg.eos_token_id = eos_for_step;
            step_cfg.top_k = first_params.top_k;
            step_cfg.top_p = first_params.top_p;
            step_cfg.min_p = first_params.min_p;
            step_cfg.prefill_chunk_size = cfg_.prefill_chunk_size;
            auto new_slot_ids = engine->flat_step_launch(
                new_prompts, step_cfg, *graph_exec, *nccl_comm);

            std::vector<ActiveGeneration> failed_prefills;
            for (size_t i = 0; i < prefill_items.size(); ++i) {
                const int sid = (i < new_slot_ids.size()) ? new_slot_ids[i] : -1;
                if (sched_debug) {
                    std::cerr << fmt::format(
                        "[sched-dbg]   prefill[{}] sid={} prompt_len={}\n",
                        i, sid, prefill_items[i].pending->prompt_ids.size());
                }
                if (sid < 0) {
                    failed_prefills.push_back(std::move(prefill_items[i]));
                } else {
                    slot_map[sid] = std::move(prefill_items[i]);
                }
            }
            if (!failed_prefills.empty()) {
                if (sched_debug) {
                    std::cerr << fmt::format(
                        "[sched-dbg] step={} re-queuing {} failed prefills\n",
                        step_count, failed_prefills.size());
                }
                std::lock_guard<std::mutex> lock(pending_mu_);
                for (auto it = failed_prefills.rbegin(); it != failed_prefills.rend(); ++it) {
                    pending_.push_front(it->pending);
                }
            }

            {
                std::lock_guard<std::mutex> lock(collect_mu);
                collect_requested = true;
            }
            collect_cv.notify_one();

            ++step_count;
            if (cfg_.enable_loop_trace && (step_count <= 30 || step_count % 50 == 0)) {
                const auto t_end = std::chrono::high_resolution_clock::now();
                const double total_ms = std::chrono::duration<double, std::milli>(t_end - t_loop_start).count();
                std::size_t pending_size = 0;
                {
                    std::lock_guard<std::mutex> lock(pending_mu_);
                    pending_size = pending_.size();
                }
                std::cerr << fmt::format(
                    "[surogate][cpp-http][loop] step={} total={:.2f}ms new={} active={} pending={}\n",
                    step_count,
                    total_ms,
                    new_prompts.size(),
                    slot_map.size(),
                    pending_size);
            }
        }

        {
            std::lock_guard<std::mutex> lock(collect_mu);
            collect_stop = true;
        }
        collect_cv.notify_all();
        if (collect_thread.joinable()) {
            collect_thread.join();
        }
    } catch (...) {
        const auto eptr = std::current_exception();
        std::string err = "unknown scheduler error";
        try {
            if (eptr) {
                std::rethrow_exception(eptr);
            }
        } catch (const std::exception& e) {
            err = e.what();
        } catch (...) {
        }
        {
            std::lock_guard<std::mutex> lock(scheduler_error_mu_);
            scheduler_error_msg_ = err;
        }
        scheduler_failed_.store(true, std::memory_order_relaxed);
        std::cerr << fmt::format("[surogate][cpp-http] scheduler loop failed: {}\n", err);

        for (auto& kv : slot_map) {
            fail_pending(kv.second.pending, eptr);
        }
        std::deque<std::shared_ptr<PendingGeneration>> pending_left;
        {
            std::lock_guard<std::mutex> lock(pending_mu_);
            pending_left.swap(pending_);
        }
        for (auto& p : pending_left) {
            fail_pending(p, eptr);
        }
    }

    if (engine_id != 0) {
        if (infer_ctx_.engine) {
            for (const auto& kv : slot_map) {
                infer_ctx_.engine->release_slot(kv.first);
            }
        }
        // Destroy goes through trainer (one-time cleanup).
        trainer_->engine_destroy(engine_id);
        infer_ctx_ = {};
    }
    scheduler_exited_.store(true, std::memory_order_relaxed);
    pending_cv_.notify_all();
}

nlohmann::json NativeHttpServer::usage_from_choices(const std::vector<GeneratedChoice>& choices) {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    if (!choices.empty()) {
        prompt_tokens = choices.front().prompt_tokens;
        for (const auto& c : choices) {
            completion_tokens += c.completion_tokens;
        }
    }
    return json{
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", prompt_tokens + completion_tokens},
    };
}

void NativeHttpServer::setup_routes() {
    if (!server_) {
        return;
    }

    using json = nlohmann::json;

    auto to_headers = [](const httplib::Headers& in) {
        std::unordered_multimap<std::string, std::string> out;
        for (const auto& kv : in) {
            out.emplace(kv.first, kv.second);
        }
        return out;
    };

    auto set_json = [](httplib::Response& res, int status, const json& payload) {
        res.status = status;
        res.set_content(json_dump(payload), "application/json; charset=utf-8");
    };

    auto set_context_length_error = [&](httplib::Response& res, const std::string& message, const std::string& param) {
        set_json(
            res,
            400,
            json{
                {"error",
                 {
                     {"message", message},
                     {"type", "invalid_request_error"},
                     {"param", param},
                     {"code", "context_length_exceeded"},
                 }},
            });
    };

    auto set_invalid_request = [&](httplib::Response& res, int status, const std::string& message) {
        set_json(
            res,
            status,
            json{
                {"error",
                 {
                     {"message", message},
                     {"type", "invalid_request_error"},
                 }},
            });
    };

    server_->set_exception_handler([&](const httplib::Request&, httplib::Response& res, const std::exception_ptr& eptr) {
        std::string message = "internal server error";
        try {
            if (eptr) {
                std::rethrow_exception(eptr);
            }
        } catch (const std::exception& e) {
            message = e.what();
        } catch (...) {
        }
        set_json(
            res,
            500,
            json{
                {"error",
                 {
                     {"message", message},
                     {"type", "server_error"},
                 }},
            });
    });

    server_->Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        set_json(res, 200, json{{"status", "ok"}});
    });
    server_->Get("/v1/health", [&](const httplib::Request&, httplib::Response& res) {
        set_json(res, 200, json{{"status", "ok"}});
    });

    auto v1_root = [&](const httplib::Request&, httplib::Response& res) {
        set_json(res, 200, json{{"status", "ok"}});
    };
    server_->Get("/v1", v1_root);
    server_->Post("/v1", v1_root);
    server_->Options("/v1", v1_root);

    server_->Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
        std::string auth_err;
        if (!authorize_or_401(to_headers(req.headers), auth_err)) {
            res.status = 401;
            res.set_content(auth_err, "application/json; charset=utf-8");
            return;
        }
        const int created = now_unix_seconds();
        set_json(
            res,
            200,
            json{
                {"object", "list"},
                {"data",
                 json::array(
                     {json{
                         {"id", cfg_.model_id},
                         {"object", "model"},
                         {"created", created},
                         {"owned_by", "surogate"},
                         {"root", cfg_.model_id},
                     }})},
            });
    });

    server_->Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        std::string auth_err;
        if (!authorize_or_401(to_headers(req.headers), auth_err)) {
            res.status = 401;
            res.set_content(auth_err, "application/json; charset=utf-8");
            return;
        }

        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            set_invalid_request(res, 400, fmt::format("Invalid JSON body: {}", e.what()));
            return;
        }

        const std::string request_id = next_request_id("chatcmpl");
        const int created = now_unix_seconds();
        const bool stream = body.value("stream", false);

        std::vector<std::string> stop_strings;
        std::vector<int32_t> prompt_ids;
        GenerationParams params;
        try {
            auto prepared = prepare_chat_generation(body, stop_strings);
            prompt_ids = std::move(prepared.first);
            params = prepared.second;
        } catch (const ContextLengthError& e) {
            set_context_length_error(res, e.what(), e.param);
            return;
        } catch (const std::invalid_argument& e) {
            set_invalid_request(res, 400, e.what());
            return;
        } catch (const std::exception& e) {
            set_invalid_request(res, 500, e.what());
            return;
        }

        if (!stream) {
            try {
                auto pending = enqueue_generation(
                    std::move(prompt_ids),
                    params,
                    std::move(stop_strings),
                    false);
                auto choices = wait_generation(pending);
                auto usage = usage_from_choices(choices);

                json out_choices = json::array();
                for (const auto& c : choices) {
                    out_choices.push_back(
                        json{
                            {"index", c.index},
                            {"message", json{{"role", "assistant"}, {"content", c.text}}},
                            {"finish_reason", c.finish_reason},
                        });
                }

                set_json(
                    res,
                    200,
                    json{
                        {"id", request_id},
                        {"object", "chat.completion"},
                        {"created", created},
                        {"model", cfg_.model_id},
                        {"choices", std::move(out_choices)},
                        {"usage", std::move(usage)},
                    });
            } catch (const std::exception& e) {
                set_invalid_request(res, 500, e.what());
            }
            return;
        }

        auto pending = enqueue_generation(
            std::move(prompt_ids),
            params,
            std::move(stop_strings),
            true);

        bool include_usage = false;
        if (body.contains("stream_options") && body["stream_options"].is_object()) {
            include_usage = body["stream_options"].value("include_usage", false);
        }

        struct ChatStreamState {
            enum Phase {
                ROLE,
                TOKENS,
                FINISH,
                USAGE,
                DONE
            };
            Phase phase = ROLE;
            std::shared_ptr<PendingGeneration> pending;
            std::string request_id;
            int created = 0;
            bool include_usage = false;
            std::vector<GeneratedChoice> choices;
            bool has_choices = false;
            bool had_error = false;
        };

        auto state = std::make_shared<ChatStreamState>();
        state->pending = pending;
        state->request_id = request_id;
        state->created = created;
        state->include_usage = include_usage;

        res.status = 200;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, state](size_t, httplib::DataSink& sink) -> bool {
                std::string chunk;
                while (chunk.empty()) {
                    if (state->phase == ChatStreamState::ROLE) {
                        chunk = sse_data(
                            json{
                                {"id", state->request_id},
                                {"object", "chat.completion.chunk"},
                                {"created", state->created},
                                {"model", cfg_.model_id},
                                {"choices",
                                 json::array({json{
                                     {"index", 0},
                                     {"delta", json{{"role", "assistant"}}},
                                     {"finish_reason", nullptr},
                                 }})},
                            });
                        state->phase = ChatStreamState::TOKENS;
                        break;
                    }
                    if (state->phase == ChatStreamState::TOKENS) {
                        int32_t tok = 0;
                        if (!state->pending->stream_queue || !state->pending->stream_queue->pop_blocking(tok)) {
                            state->phase = ChatStreamState::FINISH;
                            continue;
                        }

                        std::vector<int32_t> batch{tok};
                        int32_t next_tok = 0;
                        while (static_cast<int>(batch.size()) < std::max(1, cfg_.stream_batch_tokens)
                               && state->pending->stream_queue->try_pop(next_tok)) {
                            batch.push_back(next_tok);
                        }

                        const std::string piece = safe_decode(batch);
                        if (piece.empty()) {
                            continue;
                        }

                        chunk = sse_data(
                            json{
                                {"id", state->request_id},
                                {"object", "chat.completion.chunk"},
                                {"created", state->created},
                                {"model", cfg_.model_id},
                                {"choices",
                                 json::array({json{
                                     {"index", 0},
                                     {"delta", json{{"content", piece}}},
                                     {"finish_reason", nullptr},
                                 }})},
                            });
                        break;
                    }
                    if (state->phase == ChatStreamState::FINISH) {
                        if (!state->has_choices && !state->had_error) {
                            try {
                                state->choices = wait_generation(state->pending);
                                state->has_choices = true;
                            } catch (...) {
                                state->had_error = true;
                            }
                        }
                        if (state->had_error || !state->has_choices) {
                            chunk = sse_done();
                            sink.write(chunk.data(), chunk.size());
                            sink.done();
                            return false;
                        }

                        const std::string finish_reason = state->choices.empty()
                                                              ? "stop"
                                                              : state->choices.front().finish_reason;
                        chunk = sse_data(
                            json{
                                {"id", state->request_id},
                                {"object", "chat.completion.chunk"},
                                {"created", state->created},
                                {"model", cfg_.model_id},
                                {"choices",
                                 json::array({json{
                                     {"index", 0},
                                     {"delta", json::object()},
                                     {"finish_reason", finish_reason},
                                 }})},
                            });
                        state->phase = state->include_usage ? ChatStreamState::USAGE : ChatStreamState::DONE;
                        break;
                    }
                    if (state->phase == ChatStreamState::USAGE) {
                        chunk = sse_data(
                            json{
                                {"id", state->request_id},
                                {"object", "chat.completion.chunk"},
                                {"created", state->created},
                                {"model", cfg_.model_id},
                                {"choices", json::array()},
                                {"usage", usage_from_choices(state->choices)},
                            });
                        state->phase = ChatStreamState::DONE;
                        break;
                    }
                    if (state->phase == ChatStreamState::DONE) {
                        chunk = sse_done();
                        sink.write(chunk.data(), chunk.size());
                        sink.done();
                        return false;
                    }
                }

                if (!chunk.empty()) {
                    sink.write(chunk.data(), chunk.size());
                }
                return true;
            },
            [state](bool) mutable {
                state.reset();
            });
    });

    server_->Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        std::string auth_err;
        if (!authorize_or_401(to_headers(req.headers), auth_err)) {
            res.status = 401;
            res.set_content(auth_err, "application/json; charset=utf-8");
            return;
        }

        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            set_invalid_request(res, 400, fmt::format("Invalid JSON body: {}", e.what()));
            return;
        }

        const std::string request_id = next_request_id("cmpl");
        const int created = now_unix_seconds();
        const bool stream = body.value("stream", false);

        std::vector<std::string> stop_strings;
        std::vector<int32_t> prompt_ids;
        GenerationParams params;
        try {
            auto prepared = prepare_completion_generation(body, stop_strings);
            prompt_ids = std::move(prepared.first);
            params = prepared.second;
        } catch (const ContextLengthError& e) {
            set_context_length_error(res, e.what(), e.param);
            return;
        } catch (const std::invalid_argument& e) {
            set_invalid_request(res, 400, e.what());
            return;
        } catch (const std::exception& e) {
            set_invalid_request(res, 500, e.what());
            return;
        }

        if (!stream) {
            try {
                auto pending = enqueue_generation(
                    std::move(prompt_ids),
                    params,
                    std::move(stop_strings),
                    false);
                auto choices = wait_generation(pending);
                auto usage = usage_from_choices(choices);

                json out_choices = json::array();
                for (const auto& c : choices) {
                    out_choices.push_back(
                        json{
                            {"index", c.index},
                            {"text", c.text},
                            {"finish_reason", c.finish_reason},
                        });
                }

                set_json(
                    res,
                    200,
                    json{
                        {"id", request_id},
                        {"object", "text_completion"},
                        {"created", created},
                        {"model", cfg_.model_id},
                        {"choices", std::move(out_choices)},
                        {"usage", std::move(usage)},
                    });
            } catch (const std::exception& e) {
                set_invalid_request(res, 500, e.what());
            }
            return;
        }

        auto pending = enqueue_generation(
            std::move(prompt_ids),
            params,
            std::move(stop_strings),
            true);

        bool include_usage = false;
        if (body.contains("stream_options") && body["stream_options"].is_object()) {
            include_usage = body["stream_options"].value("include_usage", false);
        }

        struct CompletionStreamState {
            enum Phase {
                TOKENS,
                FINISH,
                USAGE,
                DONE
            };
            Phase phase = TOKENS;
            std::shared_ptr<PendingGeneration> pending;
            std::string request_id;
            int created = 0;
            bool include_usage = false;
            std::vector<GeneratedChoice> choices;
            bool has_choices = false;
            bool had_error = false;
        };

        auto state = std::make_shared<CompletionStreamState>();
        state->pending = pending;
        state->request_id = request_id;
        state->created = created;
        state->include_usage = include_usage;

        res.status = 200;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, state](size_t, httplib::DataSink& sink) -> bool {
                std::string chunk;
                while (chunk.empty()) {
                    if (state->phase == CompletionStreamState::TOKENS) {
                        int32_t tok = 0;
                        if (!state->pending->stream_queue || !state->pending->stream_queue->pop_blocking(tok)) {
                            state->phase = CompletionStreamState::FINISH;
                            continue;
                        }

                        std::vector<int32_t> batch{tok};
                        int32_t next_tok = 0;
                        while (static_cast<int>(batch.size()) < std::max(1, cfg_.stream_batch_tokens)
                               && state->pending->stream_queue->try_pop(next_tok)) {
                            batch.push_back(next_tok);
                        }

                        const std::string piece = safe_decode(batch);
                        if (piece.empty()) {
                            continue;
                        }

                        chunk = sse_data(
                            json{
                                {"id", state->request_id},
                                {"object", "text_completion"},
                                {"created", state->created},
                                {"model", cfg_.model_id},
                                {"choices",
                                 json::array({json{
                                     {"index", 0},
                                     {"text", piece},
                                     {"finish_reason", nullptr},
                                 }})},
                            });
                        break;
                    }
                    if (state->phase == CompletionStreamState::FINISH) {
                        if (!state->has_choices && !state->had_error) {
                            try {
                                state->choices = wait_generation(state->pending);
                                state->has_choices = true;
                            } catch (...) {
                                state->had_error = true;
                            }
                        }
                        if (state->had_error || !state->has_choices) {
                            chunk = sse_done();
                            sink.write(chunk.data(), chunk.size());
                            sink.done();
                            return false;
                        }

                        const std::string finish_reason = state->choices.empty()
                                                              ? "stop"
                                                              : state->choices.front().finish_reason;
                        chunk = sse_data(
                            json{
                                {"id", state->request_id},
                                {"object", "text_completion"},
                                {"created", state->created},
                                {"model", cfg_.model_id},
                                {"choices",
                                 json::array({json{
                                     {"index", 0},
                                     {"text", ""},
                                     {"finish_reason", finish_reason},
                                 }})},
                            });
                        state->phase = state->include_usage ? CompletionStreamState::USAGE : CompletionStreamState::DONE;
                        break;
                    }
                    if (state->phase == CompletionStreamState::USAGE) {
                        chunk = sse_data(
                            json{
                                {"id", state->request_id},
                                {"object", "text_completion"},
                                {"created", state->created},
                                {"model", cfg_.model_id},
                                {"choices", json::array()},
                                {"usage", usage_from_choices(state->choices)},
                            });
                        state->phase = CompletionStreamState::DONE;
                        break;
                    }
                    if (state->phase == CompletionStreamState::DONE) {
                        chunk = sse_done();
                        sink.write(chunk.data(), chunk.size());
                        sink.done();
                        return false;
                    }
                }

                if (!chunk.empty()) {
                    sink.write(chunk.data(), chunk.size());
                }
                return true;
            },
            [state](bool) mutable {
                state.reset();
            });
    });
}

}  // namespace serve
