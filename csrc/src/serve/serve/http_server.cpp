#include "serve/http_server.h"

#include "serve/lora_registry.h"
#include "serve/model_scheduler.h"

#include "serve/anthropic_schema.h"
#include "serve/console_log.h"
#include "serve/decisions_schema.h"
#include "serve/http_socket.h"
#include "serve/openai_schema.h"
#include "serve/openrouter_catalog.h"
#include "serve/request_log.h"
#include "serve/translate.h"

#include "core/sleep.h"
#include "core/unified_memory.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <cuda_runtime.h>
#include <cstdlib>
#include <sstream>

namespace sinfer::serve {
namespace {

struct StreamingRequest {
    explicit StreamingRequest(PreparedRequest request) : prepared(std::move(request)) {}

    PreparedRequest prepared;
    std::atomic<bool> cancelled{false};
    bool started = false;
};

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

void write_stream_item(httplib::DataSink& sink, StreamingRequest& request,
                       const std::string& item) {
    if (request.cancelled.load(std::memory_order_acquire) ||
        (sink.is_writable && !sink.is_writable()) || !sink.write(item.data(), item.size())) {
        request.cancelled.store(true, std::memory_order_release);
        throw ClientDisconnected();
    }
}

/// write_stream_item for the output callbacks, which run inside generation. A client that has
/// gone away marks the stream cancelled instead of throwing through the engine: generation then
/// stops at its next step and returns what it produced, so the request_done record carries the
/// tokens actually generated -- what a caller settles a dropped stream from.
void offer_stream_item(httplib::DataSink& sink, StreamingRequest& request, const std::string& item) {
    try {
        write_stream_item(sink, request, item);
    } catch (const ClientDisconnected&) {
        // Recorded in request.cancelled; is_cancelled reports it to the engine.
    }
}

/// The caller's X-Request-Id, if it is safe to log and echo (see client_request_id()). A request
/// carrying several is given none: which one a proxy chain meant is not knowable here.
std::string request_id_of(const httplib::Request& request) {
    const std::string header(kClientRequestIdHeader);
    if (request.get_header_value_count(header) != 1) { return {}; }
    return client_request_id(request.get_header_value(header));
}

/// A caller that sends ids this server cannot use would otherwise see its records carry
/// `client_request_id: null` with no hint why. Said once per process, without the value: it is
/// exactly what the sanitizer keeps out of the log.
void warn_rejected_request_id() {
    static std::atomic<bool> warned{false};
    if (warned.exchange(true, std::memory_order_relaxed)) { return; }
    write_console_log(ConsoleLogLevel::Warning,
                      "ignoring an X-Request-Id that is not one header of 1 to 128 visible ASCII "
                      "characters; such ids are neither echoed nor logged (reported once)");
}

void set_owned_content(httplib::Response& response, std::string body,
                       std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.hold_resource(std::move(lifetime));
}

void write_error(httplib::Response& res, const ApiError& error) {
    res.status = error.status;
    res.set_content(make_error_body(error), "application/json");
}

// Anthropic-shaped error body ({"type":"error","error":{...}}), used by the
// /v1/messages endpoints so Claude clients see the error format they expect.
void write_messages_error(httplib::Response& res, const ApiError& error) {
    res.status = error.status;
    res.set_content(make_messages_error_body(error), "application/json");
}

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_error(res, error);
}

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

ThroughputReport make_throughput_report(const sinfer::RuntimeStats& previous,
                                        const sinfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .scheduler         = current,
    };
}

/// Runs `action` on every engine in `targets` -- the primary's data-parallel replicas, or one
/// extra -- at once (each moves its own memory over its own link), and rethrows the first
/// failure once every engine has finished, so one failing replica does not leave the rest
/// untried.
template <typename Action>
void on_every_engine(const std::vector<GenerationService*>& targets, const Action& action) {
    if (targets.size() == 1) {
        action(*targets.front());
        return;
    }
    std::vector<std::future<void>> runs;
    runs.reserve(targets.size());
    for (GenerationService* target : targets) {
        runs.push_back(std::async(std::launch::async, [&action, target] { action(*target); }));
    }
    std::exception_ptr failure;
    for (auto& run : runs) {
        try {
            run.get();
        } catch (...) {
            if (!failure) { failure = std::current_exception(); }
        }
    }
    if (failure) { std::rethrow_exception(failure); }
}

/// Loads (or replaces) adapter `name` on every engine in `targets` -- the primary's data-parallel
/// replicas, or one extra -- at once: an update waits for each engine's admitted requests to let
/// go of the adapter, and those waits overlap instead of adding up. If any engine fails, an
/// adapter that was new is unloaded again from the engines that took it, so the replicas never
/// disagree about whether it exists, and the first failure is rethrown as it was raised.
void load_lora_adapter_everywhere(const std::vector<GenerationService*>& targets,
                                  const std::string& name, const std::string& path) {
    if (targets.size() == 1) {
        targets.front()->load_lora_adapter(name, path);
        return;
    }
    std::vector<bool> existed(targets.size());
    for (std::size_t i = 0; i < targets.size(); ++i) { existed[i] = targets[i]->lora_slot(name) >= 0; }
    std::vector<std::atomic<bool>> loaded(targets.size());
    std::exception_ptr failure;
    try {
        on_every_engine(targets, [&](GenerationService& target) {
            target.load_lora_adapter(name, path);
            const auto at = std::find(targets.begin(), targets.end(), &target) - targets.begin();
            loaded[static_cast<std::size_t>(at)].store(true);
        });
        return;
    } catch (...) { failure = std::current_exception(); }
    for (std::size_t i = 0; i < targets.size(); ++i) {
        if (loaded[i] && !existed[i]) {
            try {
                targets[i]->unload_lora_adapter(name);
            } catch (const std::exception& e) {
                write_console_log(ConsoleLogLevel::Warning,
                                  "lora: adapter '" + name + "' could not be rolled back on replica " +
                                      std::to_string(i) + ": " + e.what());
            }
        }
    }
    std::rethrow_exception(failure);
}

/// The primary's counters across its data-parallel replicas: counters and request counts add up,
/// and the page geometry is the same on each.
sinfer::RuntimeStats sum_runtime_stats(const std::vector<GenerationService*>& services) {
    if (services.empty()) { return {}; }
    sinfer::RuntimeStats total = services.front()->runtime_stats();
    for (std::size_t i = 1; i < services.size(); ++i) {
        const sinfer::RuntimeStats s = services[i]->runtime_stats();
        total.computed_prefill_tokens += s.computed_prefill_tokens;
        total.committed_decode_tokens += s.committed_decode_tokens;
        total.decode_rounds += s.decode_rounds;
        total.decode_row_rounds += s.decode_row_rounds;
        total.packed_prefill_rounds += s.packed_prefill_rounds;
        total.packed_prefill_prompts += s.packed_prefill_prompts;
        total.running_requests += s.running_requests;
        total.prefilling_requests += s.prefilling_requests;
        total.decode_ready_requests += s.decode_ready_requests;
        total.waiting_requests += s.waiting_requests;
        total.reserving_requests += s.reserving_requests;
        total.kv_pages_mapped += s.kv_pages_mapped;
        total.kv_pages += s.kv_pages;
        total.kv_pages_entitled += s.kv_pages_entitled;
        total.kv_pages_in_use += s.kv_pages_in_use;
        total.kv_pages_resident_at_granule += s.kv_pages_resident_at_granule;
        total.admission_unblocked_heads += s.admission_unblocked_heads;
        total.device_oom_rounds += s.device_oom_rounds;
    }
    return total;
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.scheduler.running_requests != 0 ||
           report.scheduler.waiting_requests != 0;
}

} // namespace

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    if (response.status != 413 || !response.body.empty()) {
        return httplib::Server::HandlerResponse::Unhandled;
    }

    ApiError error;
    error.status  = 413;
    error.type    = "invalid_request_error";
    error.code    = "request_too_large";
    error.message = "request body exceeds the configured payload limit of " +
                    std::to_string(options.max_request_bytes) + " bytes";
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_messages_error(response, error);
    } else {
        write_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

HttpPoolSizes http_pool_sizes(const ServeOptions& options) {
    // Mirrors `extra_model_options()` in serve_options.cpp for the `0 = inherit the primary's`
    // rule; that function is the source of truth, but it builds a whole ServeOptions per extra
    // model, which is a lot of copying to read one number.
    //
    // Keep this sum equal to the executor's `max_outstanding_`
    // (runtime/engine/concurrent_executor.h): the reason an unbounded queue is safe is that a
    // request which gets a worker is then admitted or refused with a 429 by that bound.
    std::size_t serving =
        static_cast<std::size_t>(options.max_concurrency) + options.max_pending_requests;
    // Each data-parallel replica is an engine of its own with the primary's bounds
    // (`replica_options()`).
    if (options.data_parallel && options.devices.size() > 1) {
        serving *= options.devices.size();
    }
    for (const auto& extra : options.extra_models) {
        serving += (extra.max_num_seqs != 0 ? extra.max_num_seqs : options.max_concurrency)
                   + static_cast<std::size_t>(options.max_pending_requests);
    }
    return HttpPoolSizes{.workers = serving + 1, .queued = 0};
}

HttpServer::HttpServer(ServeOptions options)
    : options_(std::move(options)),
      openrouter_catalog_(options_.openrouter_models_file.empty()
                              ? nlohmann::json() : load_openrouter_catalog(options_.openrouter_models_file)),
      admission_(options_.rate_limit_rps, options_.rate_limit_burst, options_.max_inflight_requests),
      thinking_admission_(0, 1, options_.max_thinking_requests),
      image_admission_(0, 1, options_.max_image_requests),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path) {
    const HttpPoolSizes sizes = http_pool_sizes(options_);
    server_.new_task_queue = [sizes] {
        return new httplib::ThreadPool(sizes.workers, sizes.queued);
    };
    // One request per queue entry, because cpp-httplib enqueues a task per
    // *connection* and that task then serves the socket for up to
    // CPPHTTPLIB_KEEPALIVE_MAX_COUNT requests (process_server_socket_core). A
    // caller that keeps sending therefore holds its worker for the whole burst,
    // not for one request, and anything behind it waits for the burst rather
    // than for a request. That is how a weight update ends up queued behind a
    // step's worth of rollouts: the rollout clients are pooled and never idle,
    // while the admin client opens a fresh connection and joins the back.
    //
    // Measured at 128 in flight against a bound of 24: the admin POST waited
    // 132s without this line and 0.0s with it.
    //
    // It is only safe BESIDE an admission bound sized to the caller, which is
    // what `http_pool_sizes` above and `grpo/utils/capacity.py` arrange for a
    // training run. On its own it made things worse, not better: shedding starts
    // happening for real, so the same measurement lost 259 rollouts against
    // main's 195, because every caller past the bound now gets an immediate 429
    // instead of waiting for a worker. A plain `surogate serve` caller, which has
    // nobody to size its bound to the run, still sees 429s past the default
    // `max_num_seqs + max_pending_requests` for exactly this reason.
    //
    // The cost is a connect per request. This engine is reached over loopback by
    // GRPO rollouts and in-cluster by `surogate serve`, where that is noise next
    // to a generation; it would not be free for a WAN client.
    server_.set_keep_alive_max_count(1);
    server_.set_payload_max_length(options_.max_request_bytes);
    // Before bind(): the listening socket carries the option to every connection.
    disable_nagle(server_);
    register_routes();
}

void HttpServer::log_line(const std::string& line) {
    write_console_log(ConsoleLogLevel::Info, line);
}

void HttpServer::log_request_start(const RequestLogContext& context) {
    log_line(format_request_start(context));
    request_jsonl_.write_request_start(context);
}

void HttpServer::log_request_rejected(const RequestRejectionLogContext& context) {
    log_line(format_request_rejected(context));
    request_jsonl_.write_request_rejected(context);
}

void HttpServer::log_request_done(const RequestLogContext& context,
                                  const GenerationOutcome& outcome) {
    log_line(format_request_done(context, outcome));
    request_jsonl_.write_request_done(context, outcome);
}

void HttpServer::log_request_error(const RequestLogContext& context, const std::string& message) {
    log_line(format_request_error(context, message));
    request_jsonl_.write_request_error(context, message);
}

void HttpServer::log_stream_failure(const RequestLogContext& context, const std::string& message,
                                    bool record_written) {
    if (record_written) {
        log_line(format_request_error(context, message));
    } else {
        log_request_error(context, message);
    }
}

void HttpServer::log_cancelled_stream(const RequestLogContext& context,
                                      const PreparedRequest& prepared) {
    GenerationOutcome outcome;
    outcome.prompt_tokens = prepared.prompt_tokens;
    outcome.finish_reason = sinfer::FinishReason::Cancelled;
    log_request_done(context, outcome);
}

void HttpServer::settle_abandoned_stream(GenerationService& service, PreparedRequest& prepared,
                                         const RequestLogContext& context) noexcept {
    try {
        // A sleeping model's worker is parked until something wakes it, and waking it only to
        // cancel this request would be a waste: the request is dropped with the PreparedRequest,
        // which abandons it without waiting.
        if (service.is_sleeping()) {
            log_cancelled_stream(context, prepared);
            return;
        }
        try {
            const GenerationOutcome outcome = service.run(prepared, nullptr, [] { return true; });
            log_request_done(context, outcome);
        } catch (const ApiException& e) {
            if (e.error().code == "client_disconnected") {
                log_cancelled_stream(context, prepared);
            } else {
                log_request_error(context, e.error().message);
            }
        } catch (const std::exception& e) { log_request_error(context, e.what()); }
    } catch (...) {
        // The releaser runs in the response's destructor; logging here is best effort.
    }
}

void HttpServer::log_throughput(const ThroughputReport& report) {
    log_line(format_throughput(report));
    if (std::getenv("SUROGATE_SERVE_MEM_TRACE") != nullptr) {
        int device = 0;
        cudaGetDevice(&device);
        std::size_t free_bytes = 0, total_bytes = 0;
        device_mem_get_info(&free_bytes, &total_bytes);
        std::ostringstream trace;
        trace << "mem-trace device=" << device << " sleepable="
              << (service_->resident_bytes(device) >> 20) << " MiB free="
              << (free_bytes >> 20) << " MiB";
        log_line(trace.str());
    }
    request_jsonl_.write_throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    // One line for the primary model, summed over its data-parallel replicas.
    sinfer::RuntimeStats previous   = sum_runtime_stats(replicas_);
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_for(lock, interval, [this] { return stats_stopping_; })) { break; }
        }

        const sinfer::RuntimeStats current = sum_runtime_stats(replicas_);
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { log_throughput(report); }
        previous      = current;
        previous_time = now;
    }

    const sinfer::RuntimeStats current = sum_runtime_stats(replicas_);
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    if (tail.computed_prefill_tokens != 0 || tail.committed_decode_tokens != 0 ||
        tail.decode_rounds != 0) {
        log_throughput(tail);
    }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Allow-Headers", "Authorization, Content-Type, X-Request-Id"},
             {"Access-Control-Expose-Headers", "X-Request-Id, Retry-After"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        // Echo the caller's request id on every response, refusals included, so a gateway can
        // match what it saw to this server's request log.
        if (const std::string id = request_id_of(req); !id.empty()) {
            res.set_header(std::string(kClientRequestIdHeader), id);
        } else if (req.has_header(std::string(kClientRequestIdHeader))) {
            warn_rejected_request_id();
        }
        if (req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            req.get_header_value("Authorization") == ("Bearer " + options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!options_.api_key.empty() && !bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_messages_error(res, error);
            } else {
                write_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        if (req.method == "POST") {
            auto admission = admission_.acquire();
            if (!admission.accepted) {
                ApiError error{.status = 429, .type = "rate_limit_error",
                               .message = "server admission limit reached; retry later",
                               .code = "rate_limit_exceeded"};
                res.set_header("Retry-After", std::to_string(admission.retry_after));
                if (req.path.rfind("/v1/messages", 0) == 0) { write_messages_error(res, error); }
                else { write_error(res, error); }
                return httplib::Server::HandlerResponse::Handled;
            }
            if (admission.permit) { res.hold_resource(std::move(admission.permit)); }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                write_error(res, e.error());
            } catch (const std::exception& e) { write_exception(res, e); } catch (...) {
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                write_error(res, error);
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        // An engine whose worker has died refuses every request, so the server is not healthy,
        // however alive its HTTP side is: a health check, a registry or a load balancer must
        // take it out. The process also exits (server/main.cpp); this covers the moments until.
        const std::vector<std::string> failed = failed_models();
        if (!failed.empty()) {
            res.status = 503;
            res.set_content(nlohmann::json{{"status", "unavailable"},
                                           {"error", "the inference engine has stopped"},
                                           {"models", failed}}
                                .dump(),
                            "application/json");
            return;
        }
        res.set_content(nlohmann::json{{"status", "ok"}}.dump(), "application/json");
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get("/kv_stats", [this](const httplib::Request& req, httplib::Response& res) {
        handle_kv_stats(req, res);
    });
    server_.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        handle_metrics(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    // The same handler: a body carrying `tokens` uses them as the prompt, and this
    // path is the one an RL client posts that body to. It is a separate route
    // rather than a flag because that is the contract the client already speaks.
    server_.Post("/v1/chat/completions/tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/tokenize", [this](const httplib::Request& req, httplib::Response& res) {
        handle_tokenize(req, res);
    });
    server_.Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handle_completions(req, res);
    });
    // Decisions v1 (stable; see decisions_schema.h): one shared state, several single-token
    // questions, with OpenRouter's field names. /v1/decisions is the stable path; OpenRouter's
    // own /api/alpha/decisions and /api/v1/decisions answer identically, so a client that swaps
    // the base URL for this host keeps working. A future protocol version gets its own path.
    for (const char* path : {"/v1/decisions", "/api/alpha/decisions", "/api/v1/decisions",
                             "/v1/systemone", "/api/v1/systemone"}) {
        server_.Post(path, [this](const httplib::Request& req, httplib::Response& res) {
            handle_decisions(req, res);
        });
    }
    if (!openrouter_catalog_.is_null()) {
        server_.Get("/openrouter/v1/models", [this](const httplib::Request&, httplib::Response& res) {
            res.set_header("Cache-Control", "no-store");
            res.set_content(openrouter_catalog_.dump(), "application/json");
        });
        for (const char* path : {"/openrouter/v1/decisions", "/openrouter/api/alpha/decisions"}) {
            server_.Post(path, [this](const httplib::Request& req, httplib::Response& res) {
                handle_decisions(req, res);
            });
        }
    }
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    // vLLM's sleep-mode routes: sleeping releases VRAM (state parked in host
    // RAM), waking restores it in about the PCIe copy time. Generation while
    // asleep is refused by the engine with a 503 naming /wake_up.
    server_.Post("/sleep", [this](const httplib::Request& req, httplib::Response& res) {
        if (!options_.enable_sleep_mode) {
            res.status = 400;
            res.set_content("the server was started without --enable-sleep-mode", "text/plain");
            return;
        }
        const std::string level = req.get_param_value("level");
        if (!level.empty() && level != "1") {
            res.status = 400;
            res.set_content("only sleep level 1 is implemented (weights parked in host RAM); "
                            "level 2 (discard weights) is not",
                            "text/plain");
            return;
        }
        try {
            on_every_engine(routed_management_services(req), [](GenerationService& target) { target.sleep(); });
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
            return;
        }
        log_line("sleep: model asleep, VRAM released");
        res.set_content("{\"is_sleeping\": true}", "application/json");
    });
    server_.Post("/wake_up", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            on_every_engine(routed_management_services(req), [](GenerationService& target) { target.wake_up(); });
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
            return;
        }
        log_line("sleep: model awake");
        res.set_content("{\"is_sleeping\": false}", "application/json");
    });
    server_.Get("/is_sleeping", [this](const httplib::Request& req, httplib::Response& res) {
        // Asleep only once every replica is: a caller waiting to reuse the memory must not
        // proceed while one still holds it.
        const auto targets = routed_management_services(req);
        const bool asleep  = std::all_of(targets.begin(), targets.end(),
                                         [](const GenerationService* target) { return target->is_sleeping(); });
        res.set_content(asleep ? "{\"is_sleeping\": true}" : "{\"is_sleeping\": false}",
                        "application/json");
    });

    // vLLM's runtime adapter management routes, same request bodies.
    // Registered at both paths. The RL client strips the version prefix from its
    // admin base url and posts to the root, which is where vLLM's own extension
    // routes live; the versioned path is what everything else already calls.
    const auto load_lora_adapter_handler = [this](const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (const std::exception&) {
            res.status = 400;
            res.set_content("request body is not valid JSON", "text/plain");
            return;
        }
        if (!body.is_object() ||
            (body.contains("lora_name") && !body.at("lora_name").is_string()) ||
            (body.contains("lora_path") && !body.at("lora_path").is_string())) {
            res.status = 400;
            res.set_content("both lora_name and lora_path must be strings", "text/plain");
            return;
        }
        const std::string name = body.value("lora_name", "");
        const std::string path = body.value("lora_path", "");
        if (name.empty() || path.empty()) {
            res.status = 400;
            res.set_content("both lora_name and lora_path are required", "text/plain");
            return;
        }
        try {
            const std::vector<GenerationService*> targets = routed_management_services(req);
            GenerationService& target = *targets.front();
            std::lock_guard namespace_lock(adapter_management_mutex_);
            // The flat namespace holds at runtime too: refuse a name any other
            // service already answers to.
            if (name == public_model_id_ || extra_services_.count(name) != 0) {
                throw std::invalid_argument("'" + name + "' is a served model id");
            }
            if (service_->lora_slot(name) >= 0 && &target != service_) {
                throw std::invalid_argument("adapter '" + name + "' already exists on the primary");
            }
            for (auto& [id, extra] : extra_services_) {
                if (extra->lora_slot(name) >= 0 && &target != extra) {
                    throw std::invalid_argument("adapter '" + name + "' already exists on '" +
                                                id + "'");
                }
            }
            load_lora_adapter_everywhere(targets, name, path);
        } catch (const sinfer::RequestError& e) {
            write_error(res, request_error_to_api_error(e));
            return;
        } catch (const std::exception& e) {
            const std::string message = "lora: adapter '" + name + "' was not loaded: " + e.what();
            write_console_log(ConsoleLogLevel::Warning, message);
            res.status = 400;
            res.set_content(message, "text/plain");
            return;
        }
        log_line("lora: loaded adapter '" + name + "' from " + path);
        res.set_content("Success: LoRA adapter '" + name + "' added successfully", "text/plain");
    };
    server_.Post("/v1/load_lora_adapter", load_lora_adapter_handler);
    server_.Post("/load_lora_adapter", load_lora_adapter_handler);
    // Registered at both paths. The RL client strips the version prefix from its
    // admin base url and posts to the root, which is where vLLM's own extension
    // routes live; the versioned path is what everything else already calls.
    const auto unload_lora_adapter_handler = [this](const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (const std::exception&) {
            res.status = 400;
            res.set_content("request body is not valid JSON", "text/plain");
            return;
        }
        if (!body.is_object() ||
            (body.contains("lora_name") && !body.at("lora_name").is_string())) {
            res.status = 400;
            res.set_content("lora_name must be a string", "text/plain");
            return;
        }
        const std::string name = body.value("lora_name", "");
        if (name.empty()) {
            res.status = 400;
            res.set_content("lora_name is required", "text/plain");
            return;
        }
        try {
            std::lock_guard namespace_lock(adapter_management_mutex_);
            on_every_engine(routed_management_services(req),
                            [&name](GenerationService& target) { target.unload_lora_adapter(name); });
        } catch (const sinfer::RequestError& e) {
            write_error(res, request_error_to_api_error(e));
            return;
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(e.what(), "text/plain");
            return;
        }
        log_line("lora: unloaded adapter '" + name + "'");
        res.set_content("Success: LoRA adapter '" + name + "' removed successfully", "text/plain");
    };
    server_.Post("/v1/unload_lora_adapter", unload_lora_adapter_handler);
    server_.Post("/unload_lora_adapter", unload_lora_adapter_handler);
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
}

nlohmann::json HttpServer::model_listing() const {
    const auto created = unix_time_now();
    auto listing = nlohmann::json::parse(
        make_models_list(public_model_id_, created, service_->lora_adapter_names()));
    for (const auto& [name, service] : extra_services_) {
        const auto extra = nlohmann::json::parse(
            make_models_list(name, created, service->lora_adapter_names()));
        for (const auto& model : extra.at("data")) {
            listing["data"].push_back(model);
        }
    }
    return listing;
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(model_listing().dump(), "application/json");
}

void HttpServer::handle_kv_stats(const httplib::Request&, httplib::Response& res) const {
    // Every field here comes from the executor's published stats snapshot, never from
    // Engine::memory_summary(): that takes the execution lock, which a busy engine holds for a
    // whole round, so polling it piles handler threads up until the HTTP pool is exhausted and
    // the server stops answering anything at all. The static geometry is cached at attach time.
    const auto model_json = [this](const std::string& name, const GenerationService& service) {
        const sinfer::RuntimeStats stats = service.runtime_stats();
        const auto found                 = attached_memory_.find(&service);
        const auto bytes                 = [&](std::uint32_t pages) {
            return static_cast<std::uint64_t>(pages) * stats.kv_page_bytes;
        };
        return nlohmann::json{
            {"model", name},
            {"sleeping", service.is_sleeping()},
            {"running_requests", stats.running_requests},
            {"waiting_requests", stats.waiting_requests},
            {"kv_capacity_tokens",
             found == attached_memory_.end() ? 0U : found->second.kv_capacity},
            {"weights_bytes",
             found == attached_memory_.end() ? 0UL : found->second.weights.capacity_bytes},
            {"page_bytes", stats.kv_page_bytes},
            {"granule_pages", stats.kv_granule_pages},
            {"pages", stats.kv_pages},
            {"pages_entitled", stats.kv_pages_entitled},
            {"pages_in_use", stats.kv_pages_in_use},
            {"pages_resident_at_granule", stats.kv_pages_resident_at_granule},
            {"pages_mapped", stats.kv_pages_mapped},
            {"admission_unblocked_heads", stats.admission_unblocked_heads},
            {"pool_bytes", bytes(stats.kv_pages)},
            {"in_use_bytes", bytes(stats.kv_pages_in_use)},
            {"resident_at_granule_bytes", bytes(stats.kv_pages_resident_at_granule)},
            {"mapped_bytes", bytes(stats.kv_pages_mapped)},
        };
    };

    nlohmann::json models = nlohmann::json::array();
    // Data-parallel replicas share the primary's id; each reports its own pool and says which
    // replica it is.
    for (std::size_t i = 0; i < replicas_.size(); ++i) {
        nlohmann::json model = model_json(public_model_id_, *replicas_[i]);
        if (replicas_.size() > 1) { model["replica"] = i; }
        models.push_back(std::move(model));
    }
    for (const auto& [name, service] : extra_services_) {
        models.push_back(model_json(name, *service));
    }
    const nlohmann::json out{
        {"unix_time", unix_time_now()},
        {"device_free_bytes", sinfer::device_free_bytes(device_)},
        {"models", models},
    };
    res.set_content(out.dump(), "application/json");
}

void HttpServer::handle_metrics(const httplib::Request&, httplib::Response& res) const {
    // Same source as `/kv_stats`: the executor's published snapshot, never
    // `Engine::memory_summary()`, which takes the execution lock a busy engine holds for a
    // whole round. A scrape every fifteen seconds must never queue behind a decode.
    std::string out;
    out.reserve(4096);
    const auto label = [](std::string_view value) {
        std::string escaped;
        escaped.reserve(value.size() + 2);
        for (const char c : value) {
            if (c == '\\' || c == '"') { escaped.push_back('\\'); }
            if (c == '\n') { escaped += "\\n"; continue; }
            escaped.push_back(c);
        }
        return escaped;
    };
    const auto help = [&out](std::string_view name, std::string_view kind, std::string_view what) {
        out += "# HELP surogate_";
        out += name;
        out += ' ';
        out += what;
        out += "\n# TYPE surogate_";
        out += name;
        out += ' ';
        out += kind;
        out += '\n';
    };
    // `labels` is a row's label set: `model="..."`, plus `replica="N"` for a data-parallel replica.
    const auto metric = [&out](std::string_view name, const std::string& labels,
                               std::uint64_t value) {
        out += "surogate_";
        out += name;
        out += '{';
        out += labels;
        out += "} ";
        out += std::to_string(value);
        out += '\n';
    };

    struct Row {
        std::string labels;
        sinfer::RuntimeStats stats;
        bool sleeping = false;
        std::uint64_t kv_capacity = 0;
        std::uint64_t weights_bytes = 0;
    };
    std::vector<Row> rows;
    const auto collect = [&](const std::string& name, const GenerationService& service,
                             std::optional<std::size_t> replica = std::nullopt) {
        const auto found = attached_memory_.find(&service);
        std::string labels = "model=\"" + label(name) + '"';
        if (replica) { labels += ",replica=\"" + std::to_string(*replica) + '"'; }
        rows.push_back(Row{
            std::move(labels), service.runtime_stats(), service.is_sleeping(),
            found == attached_memory_.end() ? 0U : found->second.kv_capacity,
            found == attached_memory_.end() ? 0UL : found->second.weights.capacity_bytes,
        });
    };
    for (std::size_t i = 0; i < replicas_.size(); ++i) {
        collect(public_model_id_, *replicas_[i],
                replicas_.size() > 1 ? std::optional<std::size_t>(i) : std::nullopt);
    }
    for (const auto& [name, service] : extra_services_) { collect(name, *service); }

    help("up", "gauge", "1 when the server is answering.");
    out += "surogate_up 1\n";
    help("device_free_bytes", "gauge", "Free device memory on the serving GPU that this server may use (under --gpu-memory-limit-mib, what is left of its limit).");
    out += "surogate_device_free_bytes " + std::to_string(sinfer::device_free_bytes(device_)) +
           "\n";

    // Counters first: these are what a rate() is taken over, and the two that matter are the
    // prompt tokens actually computed (prefix hits excluded) and the tokens decode committed.
    help("prefill_tokens_total", "counter", "Prompt tokens evaluated by prefill.");
    for (const Row& r : rows) { metric("prefill_tokens_total", r.labels, r.stats.computed_prefill_tokens); }
    help("decode_tokens_total", "counter", "Tokens committed by decode rounds.");
    for (const Row& r : rows) { metric("decode_tokens_total", r.labels, r.stats.committed_decode_tokens); }
    help("decode_rounds_total", "counter", "Decode batch executions.");
    for (const Row& r : rows) { metric("decode_rounds_total", r.labels, r.stats.decode_rounds); }
    help("decode_rows_total", "counter", "Summed batch size over decode rounds; over rounds it is the mean batch.");
    for (const Row& r : rows) { metric("decode_rows_total", r.labels, r.stats.decode_row_rounds); }
    help("packed_prefill_rounds_total", "counter",
         "Prefill rounds that packed several waiting prompts while nothing was decoding.");
    for (const Row& r : rows) { metric("packed_prefill_rounds_total", r.labels, r.stats.packed_prefill_rounds); }
    help("packed_prefill_prompts_total", "counter",
         "Prompts advanced by packed prefill rounds; over rounds it is the mean packing.");
    for (const Row& r : rows) { metric("packed_prefill_prompts_total", r.labels, r.stats.packed_prefill_prompts); }
    help("admission_unblocked_heads_total", "counter",
         "Times the queue head was refused a lane although the admission ledger found it "
         "unblocked (KV committed to a retained GPU prefix); a benign retry.");
    for (const Row& r : rows) {
        metric("admission_unblocked_heads_total", r.labels, r.stats.admission_unblocked_heads);
    }

    help("requests", "gauge", "Requests in each scheduler state.");
    for (const Row& r : rows) {
        const auto state = [&](std::string_view which, std::uint64_t value) {
            out += "surogate_requests{" + r.labels + ",state=\"";
            out += which;
            out += "\"} " + std::to_string(value) + "\n";
        };
        state("running", r.stats.running_requests);
        state("prefilling", r.stats.prefilling_requests);
        state("decode_ready", r.stats.decode_ready_requests);
        state("waiting", r.stats.waiting_requests);
        state("reserving", r.stats.reserving_requests);
    }

    // The KV pool is the resource that decides whether a request queues, so it is exported in
    // pages *and* bytes: pages are what the planner reasons in, bytes are what an operator has
    // a budget for.
    help("kv_pages", "gauge", "KV pages by kind: the pool, what is entitled, live demand, resident, mapped.");
    for (const Row& r : rows) {
        const auto pages = [&](std::string_view kind, std::uint64_t value) {
            out += "surogate_kv_pages{" + r.labels + ",kind=\"";
            out += kind;
            out += "\"} " + std::to_string(value) + "\n";
        };
        pages("pool", r.stats.kv_pages);
        pages("entitled", r.stats.kv_pages_entitled);
        pages("in_use", r.stats.kv_pages_in_use);
        pages("resident_at_granule", r.stats.kv_pages_resident_at_granule);
        pages("mapped", r.stats.kv_pages_mapped);
    }
    help("kv_page_bytes", "gauge", "Bytes in one KV page.");
    for (const Row& r : rows) { metric("kv_page_bytes", r.labels, r.stats.kv_page_bytes); }
    help("kv_bytes", "gauge", "KV pool bytes by kind.");
    for (const Row& r : rows) {
        const auto bytes = [&](std::string_view kind, std::uint32_t page_count) {
            out += "surogate_kv_bytes{" + r.labels + ",kind=\"";
            out += kind;
            out += "\"} " +
                   std::to_string(static_cast<std::uint64_t>(page_count) * r.stats.kv_page_bytes) +
                   "\n";
        };
        bytes("pool", r.stats.kv_pages);
        bytes("in_use", r.stats.kv_pages_in_use);
        bytes("resident_at_granule", r.stats.kv_pages_resident_at_granule);
        bytes("mapped", r.stats.kv_pages_mapped);
    }
    help("kv_capacity_tokens", "gauge", "Tokens the KV pool was sized for.");
    for (const Row& r : rows) { metric("kv_capacity_tokens", r.labels, r.kv_capacity); }
    help("weights_bytes", "gauge", "Device bytes this model's weights occupy.");
    for (const Row& r : rows) { metric("weights_bytes", r.labels, r.weights_bytes); }
    help("sleeping", "gauge", "1 while a model's weights are released to host memory.");
    for (const Row& r : rows) { metric("sleeping", r.labels, r.sleeping ? 1U : 0U); }

    res.set_content(out, "text/plain; version=0.0.4; charset=utf-8");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto listing = model_listing();
    for (const auto& model : listing.at("data")) {
        if (model.at("id") == id) {
            res.set_content(model.dump(), "application/json");
            return;
        }
    }
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.code    = "model_not_found";
    error.message = "model '" + id + "' not found";
    write_error(res, error);
}

PreparationGate HttpServer::wake_gate() {
    // The route's reservation rides in the gate: the service runs it once it has admitted the
    // request, when its own count takes over, and destroying the gate releases it on any path
    // that never gets that far.
    // Shared by every copy of the gate, so running any one of them releases it.
    auto reservation = std::make_shared<std::shared_ptr<void>>(std::move(t_route_reservation));
    t_route_reservation.reset();
    if (scheduler_ == nullptr && *reservation == nullptr) { return {}; }
    return [scheduler = scheduler_, service = &svc(), reservation](const PreparationControl& control) {
        if (scheduler != nullptr) { scheduler->ensure_awake(service, control); }
        reservation->reset();
    };
}

std::function<bool()> HttpServer::request_cancelled(const httplib::Request& request) {
    return [this, &request] {
        return stopping_.load(std::memory_order_relaxed) ||
               (request.is_connection_alive && !request.is_connection_alive());
    };
}

// What a prompt tokenises to, without generating anything.
// A multi-turn RL client uses this to stitch turns: it asks what the next prompt
// renders to so it can check that the ids it already holds are a prefix of it. So
// the answer has to be this engine's own tokenisation of its own template, which
// is exactly what preparing the prompt produces.
void HttpServer::handle_tokenize(const httplib::Request& req, httplib::Response& res) {
    t_routed_service = nullptr;
    t_route_reservation.reset();
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_error(res, error);
        return;
    }
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        // `max_tokens` is meaningless here and a caller need not send one, so the
        // body is completed before it is parsed as a generation request.
        nlohmann::json completed = body;
        if (!completed.contains("messages") && !completed.contains("prompt")) {
            ApiError error;
            error.status  = 400;
            error.message = "tokenize needs `messages` or `prompt`";
            write_error(res, error);
            return;
        }
        GenerationRequest request =
            completed.contains("messages")
                ? parse_chat_completion_request(completed, limits)
                : parse_completion_request(completed, limits);
        t_routed_service = &route_model(request.model, &request.lora_adapter, &req,
                                        conversation_prefix_hashes(request));
        const std::vector<sinfer::TokenId> ids =
            svc().tokenize(request, request_cancelled(req), wake_gate());
        nlohmann::json out{{"count", ids.size()},
                           {"max_model_len", svc().max_context()},
                           {"tokens", ids}};
        if (body.value("with_token_strings", false)) {
            out["token_strs"] = svc().token_texts(ids);
        }
        res.set_content(out.dump(), "application/json");
    } catch (const ApiException& e) {
        write_error(res, e.error());
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 400;
        error.message = e.what();
        write_error(res, error);
    }
}

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    t_routed_service = nullptr; // keep-alive threads must not inherit a route
    t_route_reservation.reset();
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_error(res, error);
        return;
    }

    GenerationRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_chat_completion_request(body, limits);
        request.client_request_id = request_id_of(req);
        // `model` selects a served model or one of the primary's adapters.
        t_routed_service = &route_model(request.model, &request.lora_adapter, &req,
                                        conversation_prefix_hashes(request));
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = svc().prepare(
            request, request_cancelled(req), wake_gate());
    } catch (const ApiException& e) {
        log_request_rejected(make_request_rejection_log_context(req_id, "openai_chat_completions",
                                                                request, e.error()));
        write_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        log_request_rejected(
            make_request_rejection_log_context(req_id, "openai_chat_completions", request, error));
        write_error(res, error);
        return;
    }

    const std::string id       = new_chat_completion_id();
    const std::int64_t created = unix_time_now();
    const std::string model    = request.model;

    const RequestLogContext log_context =
        make_request_log_context(req_id, "openai_chat_completions", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = svc().run(prepared, nullptr, [this, &req] {
                return stopping_.load(std::memory_order_relaxed) ||
                       (req.is_connection_alive && !req.is_connection_alive());
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage = completion_usage(outcome);
            TokenDetail detail;
            detail.include_token_ids = request.return_token_ids;
            detail.include_logprobs = request.want_logprobs;
            detail.prompt_token_ids = outcome.prompt_token_ids;
            detail.completion_token_ids = outcome.completion_token_ids;
            detail.logprobs = outcome.token_logprobs;
            detail.texts = outcome.token_texts;
            detail.prompt_scores = outcome.prompt_scores;
            detail.completion_scores = outcome.completion_scores;
            detail.score_texts = outcome.score_texts;
            std::string response_body;
            if (!outcome.tool_calls.empty()) {
                response_body = make_chat_completion_tool_response(
                    id, model, created, outcome.text, outcome.reasoning, outcome.tool_calls, usage, detail);
            } else {
                response_body = make_chat_completion_response(
                    id, model, created, outcome.text, outcome.reasoning,
                    finish_reason_wire(outcome.finish_reason), usage, detail);
            }
            if (!outcome.parallel_decoding_details.empty()) {
                auto payload = nlohmann::json::parse(response_body);
                payload["parallel_decoding"] = nlohmann::json::parse(outcome.parallel_decoding_details);
                response_body = payload.dump();
            }
            set_owned_content(res, std::move(response_body), prepared.lifetime);
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            throw;
        }
        return;
    }

    auto stream              = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool include_usage = stream->prepared.include_usage;
    const bool tool_capable  = stream->prepared.tool_capable;

    // SSE hints: disable client/proxy caching and reverse-proxy response buffering
    // so tokens flush immediately. Content-Type is set by the chunked provider.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    GenerationService* const routed = &svc();
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, created, model, include_usage, tool_capable, routed,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            bool done_logged    = false;
            try {
                // The role chunk goes out with the first streamed item, not at acceptance:
                // that is when the OpenAI and vLLM servers send theirs, and it is what a
                // client's time-to-first-token clock stops on (vllm bench stamps TTFT on the
                // first chunk carrying `choices`). Sent up front it read as a 50 ms TTFT
                // for every request.
                bool role_sent   = false;
                auto ensure_role = [&] {
                    if (role_sent) { return; }
                    role_sent = true;
                    offer_stream_item(sink, *stream,
                                      make_chat_chunk_role(id, model, created, include_usage));
                };
                StreamSink output;
                output.on_content = [&](const std::string& text) {
                    ensure_role();
                    offer_stream_item(
                        sink, *stream,
                        make_chat_chunk_content(id, model, created, text, include_usage));
                };
                output.on_reasoning = [&](const std::string& text) {
                    ensure_role();
                    offer_stream_item(
                        sink, *stream,
                        make_chat_chunk_reasoning(id, model, created, text, include_usage));
                };
                output.on_scores = [&](const GenerationOutcome& scored) {
                    ensure_role();
                    TokenDetail detail;
                    detail.include_logprobs = stream->prepared.want_logprobs;
                    detail.logprobs = scored.token_logprobs;
                    detail.texts = scored.token_texts;
                    detail.prompt_scores = scored.prompt_scores;
                    detail.completion_scores = scored.completion_scores;
                    detail.score_texts = scored.score_texts;
                    offer_stream_item(sink, *stream, make_chat_chunk_token_detail(id, model, created, detail, include_usage));
                };
                output.is_cancelled = [&] {
                    return stopping_.load(std::memory_order_relaxed) ||
                           stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = routed->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                done_logged = true;
                // A client that left mid-stream has its usage in that record; nobody is reading.
                if (stream->cancelled.load(std::memory_order_acquire)) { return false; }
                ensure_role();
                if (stream->prepared.return_token_ids) {
                    TokenDetail detail;
                    detail.include_token_ids = stream->prepared.return_token_ids;
                    detail.include_logprobs = false;
                    detail.prompt_token_ids = outcome.prompt_token_ids;
                    detail.completion_token_ids = outcome.completion_token_ids;
                    detail.logprobs = outcome.token_logprobs;
                    detail.texts = outcome.token_texts;

                    detail.completion_scores = outcome.completion_scores;
                    detail.score_texts = outcome.score_texts;
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_token_detail(id, model, created, detail, include_usage));
                }
                const std::string_view remaining = outcome.unstreamed_content;
                if (!outcome.tool_calls.empty()) {
                    if (!remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_tool_calls(
                                          id, model, created, outcome.tool_calls, include_usage));
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_final(id, model, created, "tool_calls", include_usage));
                } else {
                    if (tool_capable && !remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_final(id, model, created,
                                              finish_reason_wire(outcome.finish_reason),
                                              include_usage, outcome.parallel_decoding_details));
                }
                if (include_usage) {
                    const CompletionUsage usage = completion_usage(outcome);
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_usage(id, model, created, usage));
                }
                write_stream_item(sink, *stream, sse_done());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                // After request_done, a failed final write adds nothing a caller can settle from.
                if (!done_logged) { log_request_error(log_context, e.what()); }
                return false;
            } catch (const ApiException& e) {
                if (!done_logged && e.error().code == "client_disconnected" &&
                    (stream->cancelled.load(std::memory_order_acquire) ||
                     (sink.is_writable && !sink.is_writable()))) {
                    // Generation refused as cancelled because the client left, which is how
                    // parallel decoding reports a drop: settled like any dropped stream.
                    log_cancelled_stream(log_context, stream->prepared);
                    return false;
                }
                log_stream_failure(log_context, e.error().message, done_logged);
                try {
                    write_stream_item(sink, *stream, sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_stream_failure(log_context, e.what(), done_logged);
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = e.what();
                try {
                    write_stream_item(sink, *stream, sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [this, stream, routed, log_context](bool) {
            stream->cancelled.store(true, std::memory_order_release);
            if (!stream->started) {
                stream->started = true;
                settle_abandoned_stream(*routed, stream->prepared, log_context);
            }
        });
}

static TokenDetail completion_detail(const GenerationOutcome& outcome, const PreparedRequest& prepared) {
    TokenDetail detail;
    detail.include_token_ids = prepared.return_token_ids;
    detail.include_logprobs = prepared.want_logprobs;
    detail.prompt_token_ids = outcome.prompt_token_ids;
    detail.completion_token_ids = outcome.completion_token_ids;
    detail.logprobs = outcome.token_logprobs;
    detail.texts = outcome.token_texts;
    detail.prompt_scores = outcome.prompt_scores;
    detail.completion_scores = outcome.completion_scores;
    detail.score_texts = outcome.score_texts;
    return detail;
}

void HttpServer::handle_completions(const httplib::Request& req, httplib::Response& res) {
    t_routed_service = nullptr; // keep-alive threads must not inherit a route
    t_route_reservation.reset();
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_error(res, error);
        return;
    }

    GenerationRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_completion_request(body, limits);
        request.client_request_id = request_id_of(req);
        t_routed_service          = &route_model(request.model, &request.lora_adapter, &req,
                                                 conversation_prefix_hashes(request));
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = svc().prepare(
            request, request_cancelled(req), wake_gate());
    } catch (const ApiException& e) {
        log_request_rejected(
            make_request_rejection_log_context(req_id, "openai_completions", request, e.error()));
        write_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        log_request_rejected(
            make_request_rejection_log_context(req_id, "openai_completions", request, error));
        write_error(res, error);
        return;
    }

    const std::string id       = new_completion_id();
    const std::int64_t created = unix_time_now();
    const std::string model    = request.model;

    const RequestLogContext log_context =
        make_request_log_context(req_id, "openai_completions", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = svc().run(prepared, nullptr, [this, &req] {
                return stopping_.load(std::memory_order_relaxed) ||
                       (req.is_connection_alive && !req.is_connection_alive());
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage = completion_usage(outcome);
            // A completion has no assistant turn and so no reasoning channel to split off:
            // whatever the model continued with is the text.
            set_owned_content(res,
                              make_completion_response(id, model, created, outcome.text,
                                                       finish_reason_wire(outcome.finish_reason),
                                                       usage, completion_detail(outcome, prepared)),
                              prepared.lifetime);
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            throw;
        }
        return;
    }

    auto stream              = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool include_usage = stream->prepared.include_usage;

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    GenerationService* const routed = &svc();
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, created, model, include_usage, routed,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            bool done_logged    = false;
            try {
                StreamSink output;
                output.on_content = [&](const std::string& text) {
                    offer_stream_item(
                        sink, *stream,
                        make_completion_chunk_text(id, model, created, text, include_usage));
                };
                std::size_t score_text_offset = 0;
                output.on_scores = [&](const GenerationOutcome& scored) {
                    auto detail = completion_detail(scored, stream->prepared);
                    detail.include_token_ids = false;
                    detail.text_offset = score_text_offset;
                    offer_stream_item(sink, *stream, make_completion_chunk_token_detail(id, model, created, detail, include_usage));
                    for (const auto& text : scored.token_texts) {
                        for (unsigned char byte : text) score_text_offset += (byte & 0xc0) != 0x80;
                    }
                };
                output.is_cancelled = [&] {
                    return stopping_.load(std::memory_order_relaxed) ||
                           stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = routed->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                done_logged = true;
                // A client that left mid-stream has its usage in that record; nobody is reading.
                if (stream->cancelled.load(std::memory_order_acquire)) { return false; }
                if (stream->prepared.return_token_ids) {
                    auto detail = completion_detail(outcome, stream->prepared);
                    detail.include_logprobs = false;
                    detail.prompt_scores.clear();
                    write_stream_item(sink, *stream, make_completion_chunk_token_detail(id, model, created, detail, include_usage));
                }
                write_stream_item(sink, *stream,
                                  make_completion_chunk_final(
                                      id, model, created,
                                      finish_reason_wire(outcome.finish_reason), include_usage));
                if (include_usage) {
                    const CompletionUsage usage = completion_usage(outcome);
                    write_stream_item(sink, *stream,
                                      make_completion_chunk_usage(id, model, created, usage));
                }
                write_stream_item(sink, *stream, sse_done());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                // After request_done, a failed final write adds nothing a caller can settle from.
                if (!done_logged) { log_request_error(log_context, e.what()); }
                return false;
            } catch (const ApiException& e) {
                if (!done_logged && e.error().code == "client_disconnected" &&
                    (stream->cancelled.load(std::memory_order_acquire) ||
                     (sink.is_writable && !sink.is_writable()))) {
                    // Generation refused as cancelled because the client left, which is how
                    // parallel decoding reports a drop: settled like any dropped stream.
                    log_cancelled_stream(log_context, stream->prepared);
                    return false;
                }
                log_stream_failure(log_context, e.error().message, done_logged);
                try {
                    write_stream_item(sink, *stream, sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_stream_failure(log_context, e.what(), done_logged);
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = e.what();
                try {
                    write_stream_item(sink, *stream, sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [this, stream, routed, log_context](bool) {
            stream->cancelled.store(true, std::memory_order_release);
            if (!stream->started) {
                stream->started = true;
                settle_abandoned_stream(*routed, stream->prepared, log_context);
            }
        });
}

void HttpServer::handle_decisions(const httplib::Request& req, httplib::Response& res) {
    t_routed_service = nullptr; // keep-alive threads must not inherit a route
    t_route_reservation.reset();
    const std::uint64_t req_id = ++request_seq_;
    DecisionsRequest request;
    try {
        request          = parse_decisions_request(req.body);
        t_routed_service = &route_model(request.model, &request.lora_adapter, &req);
    } catch (const ApiException& e) {
        // Unlike a chat request, a decisions request is fully validated here rather than in
        // preparation, so most refusals are this one: log them like any other rejection.
        RequestRejectionLogContext rejection;
        rejection.id                = req_id;
        rejection.protocol          = "decisions";
        rejection.model             = request.model;
        rejection.client_request_id = request_id_of(req);
        rejection.error             = e.error();
        log_request_rejected(rejection);
        write_error(res, e.error());
        return;
    }
    if (request.thinking) {
        auto admission = thinking_admission_.acquire();
        if (!admission.accepted) {
            res.set_header("Retry-After", "1");
            write_error(res, ApiError{.status = 429, .type = "rate_limit_error",
                .message = "thinking request capacity reached; retry later", .code = "thinking_limit_exceeded"});
            return;
        }
        if (admission.permit) { res.hold_resource(std::move(admission.permit)); }
    }
    if (!request.images.empty()) {
        auto admission = image_admission_.acquire();
        if (!admission.accepted) {
            res.set_header("Retry-After", "1");
            write_error(res, ApiError{.status = 429, .type = "rate_limit_error",
                .message = "image request capacity reached; retry later", .code = "image_limit_exceeded"});
            return;
        }
        if (admission.permit) { res.hold_resource(std::move(admission.permit)); }
    }
    // The same start/done/rejected records every other protocol writes, console and JSONL,
    // with the decisions-specific counts beside them.
    RequestLogContext context;
    context.id                      = req_id;
    context.protocol                = "decisions";
    context.model                   = request.model;
    context.message_count           = 2; // one system and one user turn per question
    context.media_item_count        = request.images.size();
    context.requested_output_tokens = static_cast<int>(request.questions.size());
    context.enable_thinking         = false;
    context.question_count          = request.questions.size();
    context.decision_temperature    = svc().options().decision_temperature;
    // Off unless asked for, so a request without it logs exactly what it always did.
    context.decision_thinking       = request.thinking;
    context.decision_order_averaging = request.order_averaging;
    if (request.order_averaging) {
        context.decision_mirrored_questions = decision_mirrored_questions(request).size();
    }
    context.client_request_id       = request_id_of(req);
    log_request_start(context);
    try {
        DecisionsOutcome outcome = svc().decide(request, request_cancelled(req), wake_gate(),
            [&](std::uint32_t attempt, const std::string& reason) {
                // Kept current here, so the error and rejection records of a request that fails
                // on a later attempt say how many times it ran.
                context.decision_attempts = attempt;
                write_console_log(ConsoleLogLevel::Warning, "[req " + std::to_string(req_id) + "] " + reason);
            },
            [&](const std::string& reason) {
                write_console_log(ConsoleLogLevel::Warning, "[req " + std::to_string(req_id) + "] " + reason);
            });
        context.shared_prefix_tokens = outcome.shared_prefix_tokens;
        context.decision_attempts    = outcome.attempts;
        context.decision_thinking_questions = outcome.thinking_questions;
        context.decision_reasoning_tokens   = outcome.reasoning_tokens;
        context.decision_thinking_attempts  = outcome.thinking_attempts;
        GenerationOutcome record;
        record.prompt_tokens           = outcome.input_tokens;
        record.completion_tokens       = outcome.output_tokens;
        record.reasoning_tokens        = outcome.reasoning_tokens;
        record.finish_reason           = sinfer::FinishReason::StopToken;
        record.metrics.prepare_seconds = outcome.prepare_seconds;
        record.metrics.prefill_seconds = outcome.prefill_seconds;
        record.metrics.decode_seconds  = outcome.decode_seconds;
        record.metrics.total_seconds   = outcome.total_seconds;
        record.metrics.ttft_seconds    = outcome.total_seconds;
        log_request_done(context, record);
        OrderedJson body       = OrderedJson::object();
        body["id"]             = new_decision_id();
        body["model"]          = request.model;
        body["provider"]       = "surogate";
        body["answers"]        = std::move(outcome.answers);
        OrderedJson usage      = OrderedJson::object();
        usage["input_tokens"]  = outcome.input_tokens;
        usage["output_tokens"] = outcome.output_tokens;
        // Thinking only: the thought tokens, which output_tokens includes. A request without
        // thinking keeps v1's usage object exactly.
        if (request.thinking) {
            usage["reasoning_tokens"] = outcome.reasoning_tokens;
        }
        usage["cost"]          = 0;
        body["usage"]          = std::move(usage);
        res.set_content(body.dump(), "application/json");
    } catch (const ApiException& e) {
        RequestRejectionLogContext rejection;
        rejection.id                      = req_id;
        rejection.protocol                = context.protocol;
        rejection.model                   = request.model;
        rejection.message_count           = context.message_count;
        rejection.media_item_count        = context.media_item_count;
        rejection.requested_output_tokens = context.requested_output_tokens;
        rejection.question_count          = context.question_count;
        rejection.decision_attempts       = context.decision_attempts;
        rejection.client_request_id       = context.client_request_id;
        rejection.error                   = e.error();
        log_request_rejected(rejection);
        write_error(res, e.error());
    } catch (const std::exception& e) {
        log_request_error(context, e.what());
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        write_error(res, error);
    }
}

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    t_routed_service = nullptr;
    t_route_reservation.reset();
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }
    try {
        RequestLimits limits;
        limits.default_max_tokens       = options_.default_max_tokens;
        const GenerationRequest request = parse_messages_request(body, limits);
        // Match /v1/messages soft routing before constructing its wake gate.
        const auto extra = extra_services_.find(request.model);
        t_routed_service = extra != extra_services_.end()
                               ? extra->second
                               : &route_primary(&req, conversation_prefix_hashes(request));
        const int input_tokens          = svc().count_prompt_tokens(
            request, request_cancelled(req), wake_gate());
        res.set_content(make_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        write_messages_error(res, error);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    t_routed_service = nullptr;
    t_route_reservation.reset();
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }

    GenerationRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        // The Anthropic endpoint accepts any `model` string (Claude Code sends real
        // Claude model names) and echoes it back; it never 404s on model id.
        request = parse_messages_request(body, limits);
        request.client_request_id = request_id_of(req);
        // Soft routing: an extra's served id selects it; anything else stays on
        // the primary (one of its replicas), preserving this endpoint's never-404 contract.
        const auto extra = extra_services_.find(request.model);
        t_routed_service = extra != extra_services_.end()
                               ? extra->second
                               : &route_primary(&req, conversation_prefix_hashes(request));
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        write_messages_error(res, error);
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = svc().prepare(
            request, request_cancelled(req), wake_gate());
    } catch (const ApiException& e) {
        log_request_rejected(
            make_request_rejection_log_context(req_id, "anthropic_messages", request, e.error()));
        write_messages_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        log_request_rejected(
            make_request_rejection_log_context(req_id, "anthropic_messages", request, error));
        write_messages_error(res, error);
        return;
    }

    const std::string id    = new_message_id();
    const std::string model = request.model; // echo the requested model
    const int input_tokens  = prepared.prompt_tokens;

    const RequestLogContext log_context =
        make_request_log_context(req_id, "anthropic_messages", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = svc().run(prepared, nullptr, [this, &req] {
                return stopping_.load(std::memory_order_relaxed) ||
                       (req.is_connection_alive && !req.is_connection_alive());
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage = completion_usage(outcome);
            const char* stop_reason =
                messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
            set_owned_content(res,
                              make_messages_response(id, model, outcome.text, outcome.reasoning,
                                                     outcome.tool_calls, stop_reason, usage, outcome.stop_sequence),
                              prepared.lifetime);
        } catch (const ApiException& e) {
            log_request_error(log_context, e.error().message);
            write_messages_error(res, e.error());
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = e.what();
            write_messages_error(res, error);
        }
        return;
    }

    auto stream             = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool tool_capable = stream->prepared.tool_capable;

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    GenerationService* const routed = &svc();
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, model, input_tokens, tool_capable, routed,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;

            // message_start waits for the prompt's prefill, when the cache split is known: gateways
            // take the input tokens from this event, so reporting the whole prompt as uncached
            // here would bill its cached part twice. It goes out on prompt_ready, which comes
            // before the first content. A stream that completes without it sends it after run();
            // one that fails first sends only the error event. Offered, not written, like every
            // event sent from inside generation: a client already gone is noticed by the engine
            // at its next step, and the request still ends in request_done.
            bool message_started = false;
            const auto start_message = [&](std::uint64_t reused_prompt_tokens) {
                if (message_started) { return; }
                message_started = true;
                offer_stream_item(sink, *stream,
                                  make_message_start(id, model,
                                                     completion_usage(input_tokens, 0, reused_prompt_tokens)));
            };
            // The blocks are written from the output callbacks, inside generation.
            MessagesStreamBlocks blocks([&](const std::string& event) {
                start_message(0); // never a content block before the message it belongs to
                offer_stream_item(sink, *stream, event);
            });
            bool done_logged = false;
            try {
                StreamSink output;
                output.on_prompt_ready = [&](std::uint32_t reused) { start_message(reused); };
                output.on_reasoning = [&](const std::string& text) {
                    blocks.append(OutputChannel::Reasoning, text);
                };
                output.on_content = [&](const std::string& text) {
                    blocks.append(OutputChannel::Content, text);
                };
                output.is_cancelled = [&] {
                    return stopping_.load(std::memory_order_relaxed) ||
                           stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = routed->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                done_logged = true;
                // A client that left mid-stream has its usage in that record; nobody is reading.
                if (stream->cancelled.load(std::memory_order_acquire)) { return false; }
                start_message(completion_usage(outcome).cached_tokens);
                const std::string_view remaining = outcome.unstreamed_content;

                blocks.close();
                int next_index = blocks.next_index();

                if (tool_capable) {
                    if (!remaining.empty()) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream, make_content_block_start_text(idx));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_text(idx, std::string(remaining)));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                    for (const ToolCall& call : outcome.tool_calls) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_tool_use(idx, call));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_tool_json(idx, call.arguments_json));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                }

                if (next_index == 0) {
                    const int idx = next_index++;
                    write_stream_item(sink, *stream, make_content_block_start_text(idx));
                    write_stream_item(sink, *stream, make_content_block_stop(idx));
                }

                const char* stop_reason =
                    messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
                // The final, cumulative usage: the prompt's cache split is known only now.
                write_stream_item(sink, *stream,
                                  make_message_delta(stop_reason, completion_usage(outcome), outcome.stop_sequence));
                write_stream_item(sink, *stream, make_message_stop());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                // After request_done, a failed final write adds nothing a caller can settle from.
                if (!done_logged) { log_request_error(log_context, e.what()); }
                return false;
            } catch (const ApiException& e) {
                if (!done_logged && e.error().code == "client_disconnected" &&
                    (stream->cancelled.load(std::memory_order_acquire) ||
                     (sink.is_writable && !sink.is_writable()))) {
                    // Generation refused as cancelled because the client left, which is how
                    // parallel decoding reports a drop: settled like any dropped stream.
                    log_cancelled_stream(log_context, stream->prepared);
                    return false;
                }
                log_stream_failure(log_context, e.error().message, done_logged);
                try {
                    write_stream_item(sink, *stream, messages_sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_stream_failure(log_context, e.what(), done_logged);
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = e.what();
                try {
                    write_stream_item(sink, *stream, messages_sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [this, stream, routed, log_context](bool) {
            stream->cancelled.store(true, std::memory_order_release);
            if (!stream->started) {
                stream->started = true;
                settle_abandoned_stream(*routed, stream->prepared, log_context);
            }
        });
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

std::vector<std::string> HttpServer::failed_models() const {
    std::vector<std::string> failed;
    for (std::size_t i = 0; i < replicas_.size(); ++i) {
        if (!replicas_[i]->healthy()) {
            failed.push_back(replicas_.size() == 1 ? public_model_id_
                                                   : public_model_id_ + " (replica " + std::to_string(i) + ")");
        }
    }
    for (const auto& [name, service] : extra_services_) {
        if (!service->healthy()) { failed.push_back(name); }
    }
    return failed;
}

thread_local GenerationService* HttpServer::t_routed_service = nullptr;
thread_local std::shared_ptr<void> HttpServer::t_route_reservation;

void HttpServer::attach_replica(GenerationService& service) {
    if (service_ == nullptr) { throw std::logic_error("attach the primary before its replicas"); }
    const std::string id = resolve_public_model_id(service.options(), service.load_summary().model_id);
    if (id != public_model_id_) {
        throw std::logic_error("data-parallel replica serves '" + id + "', not '" + public_model_id_ + "'");
    }
    attached_memory_.emplace(&service, service.memory_summary());
    replicas_.push_back(&service);
    router_ = std::make_unique<ReplicaRouter>(
        replicas_.size(),
        [this](std::size_t replica) { return replicas_[replica]->active_requests(); },
        [this](std::size_t replica) {
            return replicas_[replica]->healthy() && !replicas_[replica]->is_sleeping();
        });
}

bool HttpServer::is_primary_replica(const GenerationService* service) const {
    return std::find(replicas_.begin(), replicas_.end(), service) != replicas_.end();
}

GenerationService& HttpServer::route_primary(const httplib::Request* http,
                                             const std::vector<std::uint64_t>& prefixes) {
    if (router_ == nullptr) { return *service_; }
    // vLLM's data-parallel pin: a client that keeps its own affinity (one rank per rollout)
    // names the replica outright.
    const std::string rank =
        http != nullptr ? http->get_header_value("X-data-parallel-rank") : std::string();
    ReplicaRouter::Route route;
    if (rank.empty()) {
        route = router_->pick(prefixes);
    } else {
        std::size_t replica = 0;
        const auto [end, error] = std::from_chars(rank.data(), rank.data() + rank.size(), replica);
        if (error != std::errc() || end != rank.data() + rank.size() || replica >= replicas_.size()) {
            ApiError api_error;
            api_error.status  = 400;
            api_error.type    = "invalid_request_error";
            api_error.code    = "invalid_data_parallel_rank";
            api_error.message = "X-data-parallel-rank must be a replica index below " +
                                std::to_string(replicas_.size()) + ", got '" + rank + "'";
            throw ApiException(std::move(api_error));
        }
        route = router_->pin(replica, prefixes);
    }
    t_route_reservation = std::move(route.reservation);
    return *replicas_[route.replica];
}

void HttpServer::attach_extra(GenerationService& service) {
    const sinfer::LoadSummary load = service.load_summary();
    const std::string id           = service.options().model_id_override.value_or(load.model_id);
    if (id == public_model_id_ || extra_services_.count(id) != 0 ||
        service_->lora_slot(id) >= 0) {
        throw std::logic_error("multi-model: served id '" + id + "' is not unique");
    }
    for (const std::string& adapter : service.lora_adapter_names()) {
        if (adapter == public_model_id_ || service_->lora_slot(adapter) >= 0) {
            throw std::logic_error("multi-model: adapter '" + adapter +
                                   "' collides with another served name");
        }
    }
    attached_memory_.emplace(&service, service.memory_summary());
    extra_services_.emplace(id, &service);
}

std::vector<GenerationService*> HttpServer::routed_management_services(const httplib::Request& req) {
    GenerationService& target = routed_management_service(req);
    if (&target == service_) { return replicas_; }
    return {&target};
}

GenerationService& HttpServer::routed_management_service(const httplib::Request& req) {
    const std::string model = req.get_param_value("model");
    if (model.empty()) { return *service_; }
    const auto extra = extra_services_.find(model);
    if (extra != extra_services_.end()) { return *extra->second; }
    if (model == public_model_id_) { return *service_; }
    throw std::invalid_argument("unknown model '" + model + "'");
}

GenerationService& HttpServer::route_model(const std::string& model, std::string* lora_adapter,
                                          const httplib::Request* http,
                                          const std::vector<std::uint64_t>& prefixes) {
    if (model == public_model_id_) { return route_primary(http, prefixes); }
    const auto extra = extra_services_.find(model);
    if (extra != extra_services_.end()) { return *extra->second; }
    // Adapter names share one flat namespace across every service (uniqueness
    // is enforced at startup and at runtime load), so the first owner is the
    // only owner. The primary's replicas all hold the primary's adapters.
    if (service_->lora_slot(model) >= 0) {
        if (lora_adapter != nullptr) { *lora_adapter = model; }
        return route_primary(http, prefixes);
    }
    for (auto& [name, service] : extra_services_) {
        if (service->lora_slot(model) >= 0) {
            if (lora_adapter != nullptr) { *lora_adapter = model; }
            return *service;
        }
    }
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.code    = "model_not_found";
    error.message = "model '" + model + "' not found";
    throw ApiException(std::move(error));
}

void HttpServer::attach_scheduler(ModelScheduler& scheduler) { scheduler_ = &scheduler; }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const sinfer::LoadSummary load = service.load_summary();
    const auto id = resolve_public_model_id(options_, load.model_id);
    if (service.lora_slot(id) >= 0) {
        throw std::invalid_argument("served model id '" + id + "' collides with an adapter name");
    }
    public_model_id_               = id;
    service_                       = &service;
    replicas_                      = {&service};
    const sinfer::MemorySummary memory = service.memory_summary();
    device_                            = memory.device;
    attached_memory_.emplace(&service, memory);
    request_jsonl_.write_server_start(options_, service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (!openrouter_catalog_.is_null()) {
        auto ids = service_->lora_adapter_names();
        ids.push_back(public_model_id_);
        for (const auto& [id, service] : extra_services_) { ids.push_back(id); }
        validate_openrouter_catalog_models(openrouter_catalog_, ids);
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() {
    // Order matters: raise the flag before closing the listener, so a generation
    // already running sees it on its next token rather than after cpp-httplib has
    // begun waiting for that handler to return.
    stopping_.store(true, std::memory_order_relaxed);
    server_.stop();
}

} // namespace sinfer::serve
