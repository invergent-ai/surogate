#pragma once

#include "serve/generation_service.h"
#include "serve/response_store.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <map>
#include <string>
#include <thread>

namespace sinfer::serve {

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

class HttpServer {
public:
    explicit HttpServer(ServeOptions options);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    /// An additional model served from this process; routed by its served id.
    void attach_extra(GenerationService& service);
    /// Overcommit mode: the scheduler wakes and evicts models per request.
    void attach_scheduler(class ModelScheduler& scheduler);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    /// GET /kv_stats: per-model KV pool physical occupancy, for elastic-KV sizing.
    void handle_kv_stats(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;

    // The process-wide console logger serializes lines from request and reporter threads.
    void log_line(const std::string& line);
    void log_request_start(const RequestLogContext& context);
    void log_request_rejected(const RequestRejectionLogContext& context);
    void log_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void log_request_error(const RequestLogContext& context, const std::string& message);
    void log_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();

    GenerationService* service_ = nullptr;
    /// Adapters this server may serve; empty unless --enable-lora named some.
    ServeOptions options_;
    std::string public_model_id_;
    /// Extra models by served id. Built at attach time, read-only afterwards.
    std::map<std::string, GenerationService*> extra_services_;
    /// The service a request routed to, bound per HTTP worker thread for the
    /// handler's duration; falls back to the primary.
    static thread_local GenerationService* t_routed_service;
    class ModelScheduler* scheduler_ = nullptr;
    /// Per-service startup geometry (weights, resolved KV capacity) for /kv_stats, so serving it
    /// never has to reach into a running engine. Written at attach, read-only afterwards.
    std::map<const GenerationService*, sinfer::MemorySummary> attached_memory_;
    int device_ = 0;
    [[nodiscard]] GenerationService& svc() const {
        return t_routed_service != nullptr ? *t_routed_service : *service_;
    }
    /// Routes `model`: an extra's id, the primary id, or a primary adapter
    /// (writes `lora_adapter`). Throws ApiException 404 otherwise.
    GenerationService& route_model(const std::string& model, std::string* lora_adapter);
    /// The service a management endpoint (?model=) addresses; primary default.
    GenerationService& routed_management_service(const httplib::Request& req);
    ResponseStore response_store_;
    JsonlRequestLog request_jsonl_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
};

} // namespace sinfer::serve
