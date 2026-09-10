#include "serve/http_server.h"

#include "serve/lora_registry.h"
#include "serve/model_scheduler.h"

#include "serve/anthropic_schema.h"
#include "serve/console_log.h"
#include "serve/openai_schema.h"
#include "serve/request_log.h"
#include "serve/translate.h"

#include "core/sleep.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include "ops/linear/marlin/marlin_plane.h"
#include "ops/linear/w8a8/w8fp8_plane.h"
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

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.scheduler.running_requests != 0 ||
           report.scheduler.waiting_requests != 0;
}

std::string_view unstreamed_content(const GenerationOutcome& outcome) {
    if (outcome.streamed_content_bytes > outcome.text.size()) {
        throw std::logic_error("streamed content exceeds terminal content");
    }
    return std::string_view(outcome.text).substr(outcome.streamed_content_bytes);
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

HttpServer::HttpServer(ServeOptions options)
    : options_(std::move(options)),
      response_store_(options_.response_store_max_records, options_.response_store_max_bytes),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path) {
    std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    for (const auto& extra : options_.extra_models) {
        queued_requests += (extra.max_num_seqs != 0 ? extra.max_num_seqs : options_.max_concurrency)
                           + static_cast<std::size_t>(options_.max_pending_requests);
    }
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, queued_requests);
    };
    server_.set_payload_max_length(options_.max_request_bytes);
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

void HttpServer::log_throughput(const ThroughputReport& report) {
    log_line(format_throughput(report));
    if (std::getenv("SUROGATE_SERVE_MEM_TRACE") != nullptr) {
        std::size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        const std::size_t derived = sinfer::ops::detail::w8_derived_plane_bytes();
        const std::size_t marlin  = sinfer::ops::detail::marlin_plane_bytes();
        std::ostringstream trace;
        trace << "mem-trace derived-planes=" << (derived >> 20) << " MiB marlin-planes="
              << (marlin >> 20) << " MiB free=" << (free_bytes >> 20) << " MiB";
        log_line(trace.str());
    }
    request_jsonl_.write_throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    sinfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_for(lock, interval, [this] { return stats_stopping_; })) { break; }
        }

        const sinfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { log_throughput(report); }
        previous      = current;
        previous_time = now;
    }

    const sinfer::RuntimeStats current = service_->runtime_stats();
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
             {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            req.get_header_value("Authorization") == ("Bearer " + options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
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

    server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
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
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
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
            routed_management_service(req).sleep();
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
            routed_management_service(req).wake_up();
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(e.what(), "text/plain");
            return;
        }
        log_line("sleep: model awake");
        res.set_content("{\"is_sleeping\": false}", "application/json");
    });
    server_.Get("/is_sleeping", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_content(routed_management_service(req).is_sleeping()
                            ? "{\"is_sleeping\": true}"
                            : "{\"is_sleeping\": false}",
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
        const std::string name = body.value("lora_name", "");
        const std::string path = body.value("lora_path", "");
        if (name.empty() || path.empty()) {
            res.status = 400;
            res.set_content("both lora_name and lora_path are required", "text/plain");
            return;
        }
        try {
            GenerationService& target = routed_management_service(req);
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
            target.load_lora_adapter(name, path);
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
        const std::string name = body.value("lora_name", "");
        if (name.empty()) {
            res.status = 400;
            res.set_content("lora_name is required", "text/plain");
            return;
        }
        try {
            routed_management_service(req).unload_lora_adapter(name);
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

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    std::vector<std::string> additional = service_->lora_adapter_names();
    for (const auto& [name, service] : extra_services_) {
        additional.push_back(name);
        for (const std::string& adapter : service->lora_adapter_names()) {
            additional.push_back(adapter);
        }
    }
    res.set_content(make_models_list(public_model_id_, unix_time_now(), additional),
                    "application/json");
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
            {"pool_bytes", bytes(stats.kv_pages)},
            {"in_use_bytes", bytes(stats.kv_pages_in_use)},
            {"resident_at_granule_bytes", bytes(stats.kv_pages_resident_at_granule)},
            {"mapped_bytes", bytes(stats.kv_pages_mapped)},
        };
    };

    nlohmann::json models = nlohmann::json::array();
    models.push_back(model_json(public_model_id_, *service_));
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
    const auto metric = [&out](std::string_view name, const std::string& model,
                               std::uint64_t value) {
        out += "surogate_";
        out += name;
        out += "{model=\"";
        out += model;
        out += "\"} ";
        out += std::to_string(value);
        out += '\n';
    };

    struct Row {
        std::string model;
        sinfer::RuntimeStats stats;
        bool sleeping = false;
        std::uint64_t kv_capacity = 0;
        std::uint64_t weights_bytes = 0;
    };
    std::vector<Row> rows;
    const auto collect = [&](const std::string& name, const GenerationService& service) {
        const auto found = attached_memory_.find(&service);
        rows.push_back(Row{
            label(name), service.runtime_stats(), service.is_sleeping(),
            found == attached_memory_.end() ? 0U : found->second.kv_capacity,
            found == attached_memory_.end() ? 0UL : found->second.weights.capacity_bytes,
        });
    };
    collect(public_model_id_, *service_);
    for (const auto& [name, service] : extra_services_) { collect(name, *service); }

    help("up", "gauge", "1 when the server is answering.");
    out += "surogate_up 1\n";
    help("device_free_bytes", "gauge", "Free device memory on the serving GPU.");
    out += "surogate_device_free_bytes " + std::to_string(sinfer::device_free_bytes(device_)) +
           "\n";

    // Counters first: these are what a rate() is taken over, and the two that matter are the
    // prompt tokens actually computed (prefix hits excluded) and the tokens decode committed.
    help("prefill_tokens_total", "counter", "Prompt tokens evaluated by prefill.");
    for (const Row& r : rows) { metric("prefill_tokens_total", r.model, r.stats.computed_prefill_tokens); }
    help("decode_tokens_total", "counter", "Tokens committed by decode rounds.");
    for (const Row& r : rows) { metric("decode_tokens_total", r.model, r.stats.committed_decode_tokens); }
    help("decode_rounds_total", "counter", "Decode batch executions.");
    for (const Row& r : rows) { metric("decode_rounds_total", r.model, r.stats.decode_rounds); }
    help("decode_rows_total", "counter", "Summed batch size over decode rounds; over rounds it is the mean batch.");
    for (const Row& r : rows) { metric("decode_rows_total", r.model, r.stats.decode_row_rounds); }

    help("requests", "gauge", "Requests in each scheduler state.");
    for (const Row& r : rows) {
        const auto state = [&](std::string_view which, std::uint64_t value) {
            out += "surogate_requests{model=\"" + r.model + "\",state=\"";
            out += which;
            out += "\"} " + std::to_string(value) + "\n";
        };
        state("running", r.stats.running_requests);
        state("prefilling", r.stats.prefilling_requests);
        state("decode_ready", r.stats.decode_ready_requests);
        state("waiting", r.stats.waiting_requests);
    }

    // The KV pool is the resource that decides whether a request queues, so it is exported in
    // pages *and* bytes: pages are what the planner reasons in, bytes are what an operator has
    // a budget for.
    help("kv_pages", "gauge", "KV pages by kind: the pool, what is entitled, live demand, resident, mapped.");
    for (const Row& r : rows) {
        const auto pages = [&](std::string_view kind, std::uint64_t value) {
            out += "surogate_kv_pages{model=\"" + r.model + "\",kind=\"";
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
    for (const Row& r : rows) { metric("kv_page_bytes", r.model, r.stats.kv_page_bytes); }
    help("kv_bytes", "gauge", "KV pool bytes by kind.");
    for (const Row& r : rows) {
        const auto bytes = [&](std::string_view kind, std::uint32_t page_count) {
            out += "surogate_kv_bytes{model=\"" + r.model + "\",kind=\"";
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
    for (const Row& r : rows) { metric("kv_capacity_tokens", r.model, r.kv_capacity); }
    help("weights_bytes", "gauge", "Device bytes this model's weights occupy.");
    for (const Row& r : rows) { metric("weights_bytes", r.model, r.weights_bytes); }
    help("sleeping", "gauge", "1 while a model's weights are released to host memory.");
    for (const Row& r : rows) { metric("sleeping", r.model, r.sleeping ? 1U : 0U); }

    res.set_content(out, "text/plain; version=0.0.4; charset=utf-8");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_ && service_->lora_slot(id) < 0 &&
        extra_services_.count(id) == 0) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now()), "application/json");
}

// What a prompt tokenises to, without generating anything.
//
// A multi-turn RL client uses this to stitch turns: it asks what the next prompt
// renders to so it can check that the ids it already holds are a prefix of it. So
// the answer has to be this engine's own tokenisation of its own template, which
// is exactly what preparing the prompt produces.
void HttpServer::handle_tokenize(const httplib::Request& req, httplib::Response& res) {
    t_routed_service = nullptr;
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
        t_routed_service = &route_model(request.model, &request.lora_adapter);
        if (scheduler_ != nullptr) { scheduler_->ensure_awake(t_routed_service); }
        const std::vector<sinfer::TokenId> ids = svc().tokenize(request);
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
        // `model` selects a served model or one of the primary's adapters.
        t_routed_service = &route_model(request.model, &request.lora_adapter);
        // Overcommit: a sleeping model is woken here (evicting idle neighbours
        // for room); the request waits instead of failing.
        if (scheduler_ != nullptr) { scheduler_->ensure_awake(t_routed_service); }
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = svc().prepare(
            request, [this, &req] {
                return stopping_.load(std::memory_order_relaxed) ||
                       (req.is_connection_alive && !req.is_connection_alive());
            });
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
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            TokenDetail detail;
            detail.include_token_ids = request.return_token_ids;
            detail.include_logprobs = request.want_logprobs;
            detail.prompt_token_ids = outcome.prompt_token_ids;
            detail.completion_token_ids = outcome.completion_token_ids;
            detail.logprobs = outcome.token_logprobs;
            detail.texts = outcome.token_texts;
            std::string response_body;
            if (!outcome.tool_calls.empty()) {
                response_body = make_chat_completion_tool_response(
                    id, model, created, outcome.text, outcome.reasoning, outcome.tool_calls, usage, detail);
            } else {
                response_body = make_chat_completion_response(
                    id, model, created, outcome.text, outcome.reasoning,
                    finish_reason_wire(outcome.finish_reason), usage, detail);
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
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_role(id, model, created, include_usage));
                };
                StreamSink output;
                output.on_content = [&](const std::string& text) {
                    ensure_role();
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_content(id, model, created, text, include_usage));
                };
                output.on_reasoning = [&](const std::string& text) {
                    ensure_role();
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_reasoning(id, model, created, text, include_usage));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = routed->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                ensure_role();
                if (stream->prepared.want_logprobs || stream->prepared.return_token_ids) {
                    TokenDetail detail;
                    detail.include_token_ids = stream->prepared.return_token_ids;
                    detail.include_logprobs = stream->prepared.want_logprobs;
                    detail.prompt_token_ids = outcome.prompt_token_ids;
                    detail.completion_token_ids = outcome.completion_token_ids;
                    detail.logprobs = outcome.token_logprobs;
                    detail.texts = outcome.token_texts;
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_token_detail(id, model, created, detail, include_usage));
                }
                const std::string_view remaining = unstreamed_content(outcome);
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
                                              include_usage));
                }
                if (include_usage) {
                    const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_usage(id, model, created, usage));
                }
                write_stream_item(sink, *stream, sse_done());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
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
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

void HttpServer::handle_completions(const httplib::Request& req, httplib::Response& res) {
    t_routed_service = nullptr; // keep-alive threads must not inherit a route
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
        t_routed_service          = &route_model(request.model, &request.lora_adapter);
        if (scheduler_ != nullptr) { scheduler_->ensure_awake(t_routed_service); }
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = svc().prepare(
            request, [this, &req] {
                return stopping_.load(std::memory_order_relaxed) ||
                       (req.is_connection_alive && !req.is_connection_alive());
            });
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
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            // A completion has no assistant turn and so no reasoning channel to split off:
            // whatever the model continued with is the text.
            set_owned_content(res,
                              make_completion_response(id, model, created, outcome.text,
                                                       finish_reason_wire(outcome.finish_reason),
                                                       usage),
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
            try {
                StreamSink output;
                output.on_content = [&](const std::string& text) {
                    write_stream_item(
                        sink, *stream,
                        make_completion_chunk_text(id, model, created, text, include_usage));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = routed->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                write_stream_item(sink, *stream,
                                  make_completion_chunk_final(
                                      id, model, created,
                                      finish_reason_wire(outcome.finish_reason), include_usage));
                if (include_usage) {
                    const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
                    write_stream_item(sink, *stream,
                                      make_completion_chunk_usage(id, model, created, usage));
                }
                write_stream_item(sink, *stream, sse_done());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
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
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    t_routed_service = nullptr;
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
        const int input_tokens          = svc().count_prompt_tokens(
            request, [this, &req] {
                return stopping_.load(std::memory_order_relaxed) ||
                       (req.is_connection_alive && !req.is_connection_alive());
            });
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
        // Soft routing: an extra's served id selects it; anything else stays on
        // the primary, preserving this endpoint's never-404 contract.
        const auto extra = extra_services_.find(request.model);
        if (extra != extra_services_.end()) { t_routed_service = extra->second; }
        if (scheduler_ != nullptr) { scheduler_->ensure_awake(&svc()); }
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
            request, [this, &req] {
                return stopping_.load(std::memory_order_relaxed) ||
                       (req.is_connection_alive && !req.is_connection_alive());
            });
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
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            const char* stop_reason =
                messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
            set_owned_content(res,
                              make_messages_response(id, model, outcome.text, outcome.reasoning,
                                                     outcome.tool_calls, stop_reason, usage),
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

            int next_index     = 0;
            bool thinking_open = false;
            int thinking_index = -1;
            bool text_open     = false;
            int text_index     = -1;
            try {
                write_stream_item(sink, *stream, make_message_start(id, model, input_tokens));

                StreamSink output;
                output.on_reasoning = [&](const std::string& text) {
                    if (!thinking_open) {
                        thinking_index = next_index++;
                        thinking_open  = true;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_thinking(thinking_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_thinking(thinking_index, text));
                };
                output.on_content = [&](const std::string& text) {
                    if (thinking_open) {
                        write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                        thinking_open = false;
                    }
                    if (!text_open) {
                        text_index = next_index++;
                        text_open  = true;
                        write_stream_item(sink, *stream, make_content_block_start_text(text_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_text(text_index, text));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = routed->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                const std::string_view remaining = unstreamed_content(outcome);

                if (thinking_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                    thinking_open = false;
                }
                if (text_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(text_index));
                    text_open = false;
                }

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
                write_stream_item(sink, *stream,
                                  make_message_delta(stop_reason, outcome.completion_tokens));
                write_stream_item(sink, *stream, make_message_stop());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, messages_sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
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
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

thread_local GenerationService* HttpServer::t_routed_service = nullptr;

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

GenerationService& HttpServer::routed_management_service(const httplib::Request& req) {
    const std::string model = req.get_param_value("model");
    if (model.empty()) { return *service_; }
    const auto extra = extra_services_.find(model);
    if (extra != extra_services_.end()) { return *extra->second; }
    if (model == public_model_id_) { return *service_; }
    throw std::invalid_argument("unknown model '" + model + "'");
}

GenerationService& HttpServer::route_model(const std::string& model, std::string* lora_adapter) {
    if (model == public_model_id_) { return *service_; }
    const auto extra = extra_services_.find(model);
    if (extra != extra_services_.end()) { return *extra->second; }
    // Adapter names share one flat namespace across every service (uniqueness
    // is enforced at startup and at runtime load), so the first owner is the
    // only owner.
    if (service_->lora_slot(model) >= 0) {
        if (lora_adapter != nullptr) { *lora_adapter = model; }
        return *service_;
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
    public_model_id_               = resolve_public_model_id(options_, load.model_id);
    service_                       = &service;
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
