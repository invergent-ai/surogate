#pragma once

// Human-readable request summaries and the optional full-precision JSONL event log used for
// measurement. The HTTP layer owns request ids; this module owns one stable JSON schema and
// serializes concurrent writes from non-streaming handlers and streaming workers.

#include "serve/generation_service.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>

namespace sinfer::serve {

inline constexpr int kRequestLogSchemaVersion        = 10;
inline constexpr const char* kRequestLogArtifactType = "sinfer_serve_request_log";

struct RequestLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history                  = false;
    bool enable_thinking                   = true;
    bool preserve_thinking                 = false;
    sinfer::ResolvedSamplingParameters sampling;
    double acquisition_seconds = 0.0;
    sinfer::PromptPreparationStats preparation;
    /// Set for the `decisions` protocol: how many questions the request carried and, once it
    /// has run, how many tokens of shared prefix were prefilled once for all of them.
    std::size_t question_count       = 0;
    std::size_t shared_prefix_tokens = 0;
    /// How many times the decisions request ran (--decision-attempts); logged when above 1.
    std::uint32_t decision_attempts  = 1;
    /// The calibration temperature the answers were read at (`--decision-temperature`).
    double decision_temperature = 1.0;
    /// Thinking (decisions_thinking.h): whether the request asked for it (when not, nothing
    /// below is logged); once it has run, how many questions thought, the thought tokens their
    /// answers were read after, and the thinking rounds (above 1 after a non-finite thinking
    /// readout).
    bool decision_thinking = false;
    std::size_t decision_thinking_questions = 0;
    int decision_reasoning_tokens           = 0;
    std::uint32_t decision_thinking_attempts = 0;
    /// The caller's X-Request-Id (see client_request_id()); empty when none was sent.
    std::string client_request_id;
};

// A parsed generation request that failed during synchronous preparation. It intentionally has a
// separate shape from RequestLogContext: sampler and prompt semantics are not guaranteed to have
// resolved when preparation rejects the request.
struct RequestRejectionLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;
    bool stream                             = false;
    std::size_t message_count               = 0;
    std::size_t media_item_count            = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    std::size_t question_count = 0; // `decisions` protocol
    /// Decisions: how many times the request ran before it was refused (--decision-attempts).
    std::uint32_t decision_attempts = 1;
    std::string client_request_id;  // the caller's X-Request-Id, empty when none was sent
    ApiError error;
};

struct ServerLogEnvironment {
    int device = 0;
    std::string gpu_name;
    std::string gpu_uuid;
    std::uint64_t total_device_memory_bytes = 0;
    int compute_capability_major            = 0;
    int compute_capability_minor            = 0;
    std::string cuda_compile_version;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
};

struct ThroughputReport {
    double interval_seconds               = 0.0;
    std::uint64_t computed_prefill_tokens = 0;
    std::uint64_t committed_decode_tokens = 0;
    std::uint64_t decode_rounds           = 0;
    std::uint64_t decode_row_rounds       = 0;
    sinfer::RuntimeStats scheduler;
};

/// The header a caller (a gateway) names its request with. The server echoes it on the response
/// and writes it into every request_start, request_done, request_rejected and request_error
/// record, so a caller can settle a request -- a stream it dropped included -- from the log.
inline constexpr std::string_view kClientRequestIdHeader = "X-Request-Id";

/// The value of an X-Request-Id header if it is safe to log and echo: 1 to 128 visible ASCII
/// characters (0x21-0x7E). Anything else -- empty, longer, spaces, control characters, non-ASCII
/// -- is ignored and yields an empty string, so a header cannot inject text into a log line.
[[nodiscard]] std::string client_request_id(std::string_view header);

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const PreparedRequest& prepared);
RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              ApiError error);

// Compact console records retained for operator visibility.
std::string format_request_start(const RequestLogContext& context);
std::string format_request_rejected(const RequestRejectionLogContext& context);
std::string format_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
std::string format_request_error(const RequestLogContext& context, const std::string& message);
std::string format_throughput(const ThroughputReport& report);

// Pure JSON formatters are public to repository tests. Each return value is one complete JSON
// object without a trailing newline.
std::string format_server_start_json(const std::string& server_instance_id,
                                     std::uint64_t timestamp_unix_ms, const ServeOptions& options,
                                     const sinfer::ModelSamplingDefaults& sampling_defaults,
                                     const std::string& public_model_id,
                                     const sinfer::LoadSummary& load,
                                     const sinfer::MemorySummary& memory,
                                     const ServerLogEnvironment& environment,
                                     std::optional<std::uint64_t> artifact_size_bytes);
std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context);
std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp_unix_ms,
                                         const RequestRejectionLogContext& context);
std::string format_request_done_json(const std::string& server_instance_id,
                                     std::uint64_t timestamp_unix_ms,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome);
std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context, const std::string& message);
std::string format_throughput_json(const std::string& server_instance_id,
                                   std::uint64_t timestamp_unix_ms, const ThroughputReport& report);

ServerLogEnvironment query_server_log_environment(int device);

// Opens in append mode so one campaign file can contain multiple independently started MTP/model
// blocks. Every line carries server_instance_id because request ids restart at one per process.
class JsonlRequestLog {
public:
    explicit JsonlRequestLog(const std::string& path,
                             const std::string& protected_artifact_path = {});

    JsonlRequestLog(const JsonlRequestLog&)            = delete;
    JsonlRequestLog& operator=(const JsonlRequestLog&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return output_.is_open(); }

    [[nodiscard]] const std::string& server_instance_id() const noexcept {
        return server_instance_id_;
    }

    void write_server_start(const ServeOptions& options,
                            const sinfer::ModelSamplingDefaults& sampling_defaults,
                            const std::string& public_model_id, const sinfer::LoadSummary& load,
                            const sinfer::MemorySummary& memory);
    void write_request_start(const RequestLogContext& context);
    void write_request_rejected(const RequestRejectionLogContext& context);
    void write_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void write_request_error(const RequestLogContext& context, const std::string& message);
    void write_throughput(const ThroughputReport& report);

private:
    void append(std::string record);

    std::string path_;
    std::string server_instance_id_;
    std::ofstream output_;
    std::mutex mutex_;
    bool failed_ = false;
};

} // namespace sinfer::serve
