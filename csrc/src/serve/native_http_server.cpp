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
#include <string_view>
#include <thread>

#include <csignal>
#include <unistd.h>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "third_party/cpp-httplib/httplib.h"
#include "runtime/infer/continuous_engine.h"
#include "utilities/crash_handler.h"
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

inline std::string ltrim_ascii_whitespace(std::string text) {
    size_t pos = 0;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    if (pos == 0) {
        return text;
    }
    return text.substr(pos);
}

inline bool contains_ci(std::string_view haystack, std::string_view needle) {
    if (needle.empty()) {
        return true;
    }
    auto it = std::search(
        haystack.begin(),
        haystack.end(),
        needle.begin(),
        needle.end(),
        [](char a, char b) {
            return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
        });
    return it != haystack.end();
}

inline bool is_stateful_chunking_model(const NativeHttpServerConfig& cfg, const PretrainedConfig& model_cfg) {
    switch (model_cfg.Architecture) {
        case PretrainedConfig::QWEN3_5:
        case PretrainedConfig::QWEN3_5_MOE:
        case PretrainedConfig::NEMOTRON_H:
            return true;
        default:
            break;
    }

    if (contains_ci(cfg.model_id, "qwen3.5") || contains_ci(cfg.model_id, "qwen3_5") ||
        contains_ci(cfg.model_id, "qwen3-5") || contains_ci(cfg.model_id, "nemotron-h") ||
        contains_ci(cfg.model_id, "nemotron_h")) {
        return true;
    }

    if (contains_ci(cfg.model_dir, "qwen3.5") || contains_ci(cfg.model_dir, "qwen3_5") ||
        contains_ci(cfg.model_dir, "qwen3-5") || contains_ci(cfg.model_dir, "nemotron-h") ||
        contains_ci(cfg.model_dir, "nemotron_h")) {
        return true;
    }

    if (contains_ci(model_cfg.ModelTypeName, "qwen3_5") || contains_ci(model_cfg.ModelTypeName, "qwen3.5") ||
        contains_ci(model_cfg.ArchitectureName, "qwen3_5") ||
        contains_ci(model_cfg.ArchitectureName, "qwen3.5") ||
        contains_ci(model_cfg.ModelTypeName, "nemotron_h") ||
        contains_ci(model_cfg.ArchitectureName, "nemotron_h")) {
        return true;
    }

    return false;
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
    const int tokenizer_vocab = tokenizer_.vocab_size();
    const int model_vocab = trainer_->config().VocabSize;
    token_id_limit_ = std::max({1, tokenizer_vocab, model_vocab});

    int32_t fallback = -1;
    try {
        const std::string unk = tokenizer_.special_token("unk_token");
        if (!unk.empty()) {
            fallback = tokenizer_.encode_single_token(unk);
        }
    } catch (...) {
    }
    if (fallback < 0 || fallback >= token_id_limit_) {
        try {
            fallback = tokenizer_.encode_single_token("<unk>");
        } catch (...) {
            fallback = -1;
        }
    }
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
        std::cerr << fmt::format(
            "[sched-reject] Rejecting request: shutdown={} failed={} exited={} msg={}\n",
            shutdown_.load(), scheduler_failed_.load(), scheduler_exited_.load(), msg);
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

std::string NativeHttpServer::strip_thinking_preamble(const std::string& text) const {
    constexpr const char* kThinkOpen = "<think>";
    constexpr const char* kThinkClose = "</think>";
    constexpr size_t kThinkCloseLen = 8;

    const auto close_pos = text.find(kThinkClose);
    if (close_pos == std::string::npos) {
        return text;
    }

    bool should_strip = false;
    const auto open_pos = text.find(kThinkOpen);
    if (open_pos != std::string::npos && open_pos < close_pos) {
        should_strip = true;
    } else {
        // Some tokenizers emit the closing tag text but hide the opening tag as
        // a special token; treat an early `</think>` as a hidden-reasoning preamble.
        constexpr size_t kMaxReasoningPrefixChars = 16384;
        if (close_pos <= kMaxReasoningPrefixChars
            && (close_pos == 0
                || std::isspace(static_cast<unsigned char>(text[close_pos - 1])) != 0)) {
            should_strip = true;
        }
    }
    if (!should_strip) {
        return text;
    }

    const size_t answer_start = close_pos + kThinkCloseLen;
    return ltrim_ascii_whitespace(text.substr(answer_start));
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
    const std::string visible_text = strip_thinking_preamble(raw_text);
    auto [trimmed_text, stop_hit] = apply_stop(visible_text, item.pending->stop_strings);

    std::vector<int32_t> out_ids;
    if (stop_hit || trimmed_text != raw_text) {
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

        // Create unified scheduler (vLLM v1 style).
        SchedulerConfig sched_cfg;
        sched_cfg.max_num_batched_tokens = cfg_.max_num_batched_tokens;
        sched_cfg.max_num_seqs = cfg_.max_batch_sequences;
        sched_cfg.long_prefill_token_threshold = cfg_.long_prefill_token_threshold;
        sched_cfg.max_num_partial_prefills = cfg_.max_num_partial_prefills;
        sched_cfg.max_seq_len = cfg_.continuous_engine_max_seq_len;
        const bool disable_stateful_chunking =
            env_flag_enabled("SUROGATE_SERVE_DISABLE_CHUNKED_PREFILL_FOR_STATEFUL", true) &&
            is_stateful_chunking_model(cfg_, trainer_->config());
        if (disable_stateful_chunking) {
            sched_cfg.enable_chunked_prefill = false;
            sched_cfg.long_prefill_token_threshold = std::max(
                sched_cfg.long_prefill_token_threshold,
                std::max(sched_cfg.max_seq_len, cfg_.runtime_seq_len));
            std::cerr << fmt::format(
                "[surogate][cpp-http] stateful model detected (arch={}): disabling chunked prefill, threshold={}\n",
                trainer_->config().model_name(),
                sched_cfg.long_prefill_token_threshold);
        }
        scheduler_ = std::make_unique<Scheduler>(sched_cfg);

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
                const int Nsteps = std::max(1, collected->decode_steps);
                const int Bactive = static_cast<int>(collected->active_slot_ids.size());
                for (int i = 0; i < Bactive; ++i) {
                    const int sid = collected->active_slot_ids[i];
                    auto it = slot_map.find(sid);
                    if (it == slot_map.end()) continue;
                    auto& item = it->second;
                    if (item.finished) continue;

                    const int fin = (i < static_cast<int>(collected->finished.size())) ? collected->finished[i] : 0;
                    const int clen = (i < static_cast<int>(collected->completion_lens.size())) ? collected->completion_lens[i] : 0;
                    if (clen == 0 && !fin) continue;  // still prefilling

                    // Process all N tokens for this slot (N=1 for prefill/single-step).
                    // Use completion_lens to determine how many tokens the engine actually
                    // generated — this is authoritative vs trying to detect padding values.
                    const int prev_gen = static_cast<int>(item.generated_ids.size());
                    const int engine_total_gen = clen;  // engine's total generated_count after all steps
                    const int tokens_this_step = std::max(0, engine_total_gen - prev_gen);
                    const int tokens_to_process = std::min(tokens_this_step, Nsteps);

                    bool should_finish = false;
                    for (int s = 0; s < tokens_to_process && !should_finish; ++s) {
                        const size_t tok_idx = static_cast<size_t>(s) * Bactive + i;
                        if (tok_idx >= collected->sampled_tokens.size()) break;
                        const int32_t raw_tok = collected->sampled_tokens[tok_idx];

                        bool tok_replaced = false;
                        const int32_t emitted = sanitize_token(raw_tok, tok_replaced);
                        item.generated_ids.push_back(emitted);
                        if (item.pending->stream_queue) {
                            item.pending->stream_queue->push(emitted);
                        }

                        if (!item.pending->params.ignore_eos && emitted == eos_id_) should_finish = true;
                        if (tok_replaced) should_finish = true;
                        if (!should_finish && static_cast<int>(item.generated_ids.size()) >= item.pending->params.max_tokens) {
                            should_finish = true;
                        }
                    }

                    // Check engine-reported finish flag
                    if (!should_finish && fin && !item.pending->params.ignore_eos) {
                        should_finish = true;
                    }
                    // Check stop strings once (after all tokens emitted)
                    if (!should_finish && !item.pending->stop_strings.empty()) {
                        auto partial = safe_decode(item.generated_ids);
                        if (apply_stop(partial, item.pending->stop_strings).second) {
                            should_finish = true;
                        }
                    }

                    if (should_finish) {
                        item.finished = true;
                        if (!item.generated_ids.empty() && item.generated_ids.back() == eos_id_) {
                            item.finish_reason = "stop";
                        } else if (!item.pending->stop_strings.empty()) {
                            auto partial = safe_decode(item.generated_ids);
                            item.finish_reason = apply_stop(partial, item.pending->stop_strings).second ? "stop" : "length";
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

            // === UNIFIED SCHEDULING (vLLM v1 style) ===
            // Build slot views for the scheduler.
            std::vector<SlotView> slot_views;
            slot_views.reserve(slot_map.size());
            for (const auto& [sid, item] : slot_map) {
                SlotView sv;
                sv.finished = item.finished;
                sv.is_prefill_chunk = item.is_prefill_chunk;
                sv.num_prompt_tokens = item.num_prompt_tokens;
                sv.num_computed_tokens = item.num_computed_tokens;
                slot_views.push_back(sv);
            }

            // Peek at pending prompt lengths (without popping).
            std::vector<int> pending_lens;
            {
                std::lock_guard<std::mutex> lock(pending_mu_);
                pending_lens.reserve(pending_.size());
                for (const auto& p : pending_) {
                    pending_lens.push_back(static_cast<int>(p->prompt_ids.size()));
                }
            }

            const int free_slots = std::max(0,
                cfg_.max_batch_sequences - static_cast<int>(slot_map.size()));
            auto sched_output = scheduler_->schedule(
                slot_views.data(),
                static_cast<int>(slot_views.size()),
                static_cast<int>(pending_lens.size()),
                free_slots,
                pending_lens.data(),
                static_cast<int>(pending_lens.size()));

            if (!sched_output.has_work) {
                std::unique_lock<std::mutex> lock(pending_mu_);
                pending_cv_.wait_for(lock, std::chrono::milliseconds(1), [&]() {
                    return shutdown_.load(std::memory_order_relaxed) || !pending_.empty();
                });
                continue;
            }

            // Pop newly admitted requests from pending_.
            std::vector<std::shared_ptr<PendingGeneration>> new_items;
            std::vector<std::vector<int32_t>> new_prompts;
            if (sched_output.num_new_admitted > 0) {
                std::lock_guard<std::mutex> lock(pending_mu_);
                for (int i = 0; i < sched_output.num_new_admitted && !pending_.empty(); ++i) {
                    new_items.push_back(pending_.front());
                    new_prompts.push_back(pending_.front()->prompt_ids);
                    pending_.pop_front();
                }
            }

            // Build generation params from the first available request.
            GenerationParams first_params;
            if (!new_items.empty()) {
                first_params = new_items.front()->params;
            } else if (!slot_map.empty()) {
                first_params = slot_map.begin()->second.pending->params;
            }

            const bool any_ignore_eos = [&]() {
                for (const auto& kv : slot_map) {
                    if (kv.second.pending->params.ignore_eos) return true;
                }
                for (const auto& item : new_items) {
                    if (item->params.ignore_eos) return true;
                }
                return false;
            }();
            const int32_t eos_for_step = any_ignore_eos ? -1 : eos_id_;

            int max_gen = std::min(cfg_.max_gen_len, 128);
            for (const auto& item : new_items) {
                int mt = item->params.max_tokens;
                const int pl = static_cast<int>(item->prompt_ids.size());
                if (pl + mt > cfg_.runtime_seq_len) {
                    mt = std::max(1, cfg_.runtime_seq_len - pl);
                }
                max_gen = std::max(max_gen, mt);
            }

            // Launch flat_step — direct engine call.
            infer::ContinuousGenerationEngine::FlatStepConfig step_cfg;
            step_cfg.max_gen_len = max_gen;
            step_cfg.temperature = first_params.temperature;
            step_cfg.eos_token_id = eos_for_step;
            step_cfg.top_k = first_params.top_k;
            step_cfg.top_p = first_params.top_p;
            step_cfg.min_p = first_params.min_p;
            step_cfg.prefill_chunk_size = sched_output.prefill_chunk_size;
            step_cfg.multi_decode_steps = cfg_.multi_decode_steps;
            auto new_slot_ids = engine->flat_step_launch(
                new_prompts, step_cfg, *graph_exec, *nccl_comm);

            // Register new slots in slot_map.
            for (size_t i = 0; i < new_items.size(); ++i) {
                const int sid = (i < new_slot_ids.size()) ? new_slot_ids[i] : -1;
                if (sid < 0) {
                    std::lock_guard<std::mutex> lock(pending_mu_);
                    pending_.push_front(new_items[i]);
                } else {
                    ActiveGeneration ag;
                    ag.pending = new_items[i];
                    ag.num_prompt_tokens = static_cast<int>(ag.pending->prompt_ids.size());
                    ag.num_computed_tokens = 0;
                    ag.is_prefill_chunk = true;
                    slot_map[sid] = std::move(ag);
                }
            }

            // Advance num_computed_tokens IMMEDIATELY (before GPU starts).
            // Mirrors vLLM's _update_after_schedule() (scheduler.py line 964).
            for (auto& [sid, item] : slot_map) {
                if (item.finished) continue;
                if (!item.is_prefill_chunk) {
                    item.num_computed_tokens += 1;
                } else {
                    const int remaining = item.num_prompt_tokens - item.num_computed_tokens;
                    const int chunk = std::min(remaining, sched_output.prefill_chunk_size);
                    item.num_computed_tokens += std::max(1, chunk);
                    item.is_prefill_chunk = (item.num_computed_tokens < item.num_prompt_tokens);
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
        if (err.find("Stack trace:") == std::string::npos) {
            try {
                err += "\n\nStack trace:\n";
                err += surogate::capture_stacktrace(1, 64);
            } catch (...) {
            }
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
            std::vector<int32_t> emitted_ids;
            size_t emitted_chars = 0;
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

                        state->emitted_ids.insert(
                            state->emitted_ids.end(),
                            batch.begin(),
                            batch.end());
                        const std::string visible =
                            strip_thinking_preamble(safe_decode(state->emitted_ids));
                        if (visible.size() <= state->emitted_chars) {
                            continue;
                        }
                        const std::string piece = visible.substr(state->emitted_chars);
                        state->emitted_chars = visible.size();

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
            std::vector<int32_t> emitted_ids;
            size_t emitted_chars = 0;
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

                        state->emitted_ids.insert(
                            state->emitted_ids.end(),
                            batch.begin(),
                            batch.end());
                        const std::string visible =
                            strip_thinking_preamble(safe_decode(state->emitted_ids));
                        if (visible.size() <= state->emitted_chars) {
                            continue;
                        }
                        const std::string piece = visible.substr(state->emitted_chars);
                        state->emitted_chars = visible.size();

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
