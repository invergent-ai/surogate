#include "serve/http_server.h"
#include "serve/model_scheduler.h"

#include "serve/openai_schema.h"
#include "serve/responses_schema.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sinfer::serve {
namespace {

using Json = nlohmann::json;

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

struct StreamingResponse {
    PreparedRequest prepared;
    ResponsesRequest request;
    RequestLogContext log_context;
    std::unique_ptr<ResponsesEventStream> encoder;
    std::atomic<bool> cancelled{false};
    bool started = false;
};

void write_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

ApiError responses_error(ApiError error) {
    if (error.param == "messages") { error.param = "input"; }
    return error;
}

ApiError internal_error(const std::exception& exception) {
    ApiError error;
    error.status  = 500;
    error.type    = "server_error";
    error.message = exception.what();
    return error;
}

void validate_model(const std::string& requested, const std::string& available) {
    if (requested == available) { return; }
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "model";
    error.code    = "model_not_found";
    error.message = "model '" + requested + "' not found";
    throw ApiException(std::move(error));
}

Json parse_json_body(const httplib::Request& request) {
    try {
        return Json::parse(request.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.type    = "invalid_request_error";
        error.message = "request body is not valid JSON";
        throw ApiException(std::move(error));
    }
}

bool disconnected(const httplib::Request& request) {
    return request.is_connection_alive && !request.is_connection_alive();
}

void write_stream_item(httplib::DataSink& sink, StreamingResponse& request,
                       const std::string& item) {
    if (request.cancelled.load(std::memory_order_acquire) ||
        (sink.is_writable && !sink.is_writable()) || !sink.write(item.data(), item.size())) {
        request.cancelled.store(true, std::memory_order_release);
        throw ClientDisconnected();
    }
}

void write_stream_items(httplib::DataSink& sink, StreamingResponse& request,
                        std::vector<std::string> items) {
    for (const std::string& item : items) { write_stream_item(sink, request, item); }
}

void set_owned_content(httplib::Response& response, std::string body,
                       std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.hold_resource(std::move(lifetime));
}

ResponsesRuntimeValues runtime_values(const PreparedRequest& prepared,
                                      const GenerationOutcome* outcome = nullptr) {
    ResponsesRuntimeValues runtime;
    runtime.temperature = prepared.sampling.temperature;
    runtime.top_p       = prepared.sampling.top_p;
    if (outcome != nullptr) {
        runtime.cached_input_tokens = static_cast<int>(outcome->metrics.prefix_cache_hit_tokens);
    }
    return runtime;
}

} // namespace

void HttpServer::handle_responses(const httplib::Request& req, httplib::Response& res) {
    ResponsesRequest request;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request                   = parse_responses_request(parse_json_body(req), limits);
        t_routed_service = nullptr;
        if (request.generation.model != public_model_id_) {
            const auto extra = extra_services_.find(request.generation.model);
            if (extra == extra_services_.end()) {
                validate_model(request.generation.model, public_model_id_);
            } else {
                t_routed_service = extra->second;
            }
        }
        if (scheduler_ != nullptr) { scheduler_->ensure_awake(&svc()); }
    } catch (const ApiException& exception) {
        write_error(res, responses_error(exception.error()));
        return;
    } catch (const std::exception& exception) {
        write_error(res, internal_error(exception));
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    PreparedRequest prepared;
    try {
        prepared = svc().prepare(request.generation, [&req] { return disconnected(req); });
    } catch (const ApiException& exception) {
        const ApiError error = responses_error(exception.error());
        log_request_rejected(make_request_rejection_log_context(req_id, "openai_responses",
                                                                request.generation, error));
        write_error(res, error);
        return;
    } catch (const std::exception& exception) {
        const ApiError error = internal_error(exception);
        log_request_rejected(make_request_rejection_log_context(req_id, "openai_responses",
                                                                request.generation, error));
        write_error(res, error);
        return;
    }

    const std::string id       = new_response_id();
    const std::int64_t created = unix_time_now();
    const RequestLogContext log_context =
        make_request_log_context(req_id, "openai_responses", request.generation, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome =
                svc().run(prepared, nullptr, [&req] { return disconnected(req); });
            const ResponsesRuntimeValues runtime = runtime_values(prepared, &outcome);
            BuiltResponse response = make_response_object(id, created, request, runtime, outcome);
            log_request_done(log_context, outcome);
            set_owned_content(res, response.body.dump(), prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = responses_error(exception.error());
            log_request_error(log_context, error.message);
            write_error(res, error);
        } catch (const std::exception& exception) {
            log_request_error(log_context, exception.what());
            write_error(res, internal_error(exception));
        }
        return;
    }

    auto stream              = std::make_shared<StreamingResponse>();
    stream->prepared         = std::move(prepared);
    stream->request          = std::move(request);
    stream->log_context      = log_context;
    stream->encoder          = std::make_unique<ResponsesEventStream>(id, created, stream->request,
                                                                      runtime_values(stream->prepared));

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    GenerationService* const routed = &svc();
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, routed](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            try {
                write_stream_items(sink, *stream, stream->encoder->start());
                StreamSink output;
                output.on_reasoning = [&](const std::string& text) {
                    write_stream_items(sink, *stream, stream->encoder->reasoning_delta(text));
                };
                output.on_content = [&](const std::string& text) {
                    write_stream_items(sink, *stream, stream->encoder->content_delta(text));
                };
                output.on_scores = [&](const GenerationOutcome& scored) {
                    write_stream_items(sink, *stream, stream->encoder->scores_delta(scored));
                };
                output.is_cancelled = [&] {
                    return stream->cancelled.load(std::memory_order_acquire) ||
                           (sink.is_writable && !sink.is_writable());
                };

                const GenerationOutcome outcome = routed->run(stream->prepared, &output);
                ResponsesStreamFinish finished  = stream->encoder->finish(outcome);
                write_stream_items(sink, *stream, std::move(finished.events_before_terminal));
                log_request_done(stream->log_context, outcome);
                write_stream_item(sink, *stream, stream->encoder->terminal(finished.response));
                sink.done();
                return true;
            } catch (const ClientDisconnected& exception) {
                log_request_error(stream->log_context, exception.what());
                return false;
            } catch (const ApiException& exception) {
                const ApiError error = responses_error(exception.error());
                log_request_error(stream->log_context, error.message);
                try {
                    write_stream_item(sink, *stream, stream->encoder->failed(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& exception) {
                const ApiError error = internal_error(exception);
                log_request_error(stream->log_context, error.message);
                try {
                    write_stream_item(sink, *stream, stream->encoder->failed(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

void HttpServer::handle_response_input_tokens(const httplib::Request& req, httplib::Response& res) {
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        ResponsesRequest request =
            parse_response_input_tokens_request(parse_json_body(req), limits);
        t_routed_service = nullptr;
        if (request.generation.model != public_model_id_) {
            const auto extra = extra_services_.find(request.generation.model);
            if (extra == extra_services_.end()) {
                validate_model(request.generation.model, public_model_id_);
            } else {
                t_routed_service = extra->second;
            }
        }
        if (scheduler_ != nullptr) { scheduler_->ensure_awake(&svc()); }
        const int tokens =
            svc().count_prompt_tokens(request.generation, [&req] { return disconnected(req); });
        res.set_content(make_response_input_tokens_body(tokens), "application/json");
    } catch (const ApiException& exception) {
        write_error(res, responses_error(exception.error()));
    } catch (const std::exception& exception) { write_error(res, internal_error(exception)); }
}

void HttpServer::handle_response_compact(const httplib::Request&, httplib::Response& res) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.param   = "context_management";
    error.code    = "compaction_not_supported";
    error.message = "Responses compaction is not supported";
    write_error(res, error);
}

} // namespace sinfer::serve
