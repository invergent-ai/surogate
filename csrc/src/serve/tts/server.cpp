// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#include "frontend.h"
#include "runtime.h"
#include "../serve/api_key_file.h"
#include "../serve/audio_metrics.h"
#include "../serve/audio_http.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <system_error>
#include <set>
#include <thread>

namespace {
using namespace sinfer::tts;
using namespace sinfer::serve::audio;
// Set by the signal handler, read by request, synthesis and shutdown threads (lock-free, so
// signal-safe).
std::atomic<int> interrupted{0};
static_assert(std::atomic<int>::is_always_lock_free);

void interrupt(int) { interrupted = 1; }

const char* usage =
    "surogate-tts NATIVE_PACKAGE [options]\n"
    "  --host HOST                 bind address (default 127.0.0.1)\n"
    "  --port PORT                 HTTP port (default 8080)\n"
    "  --device cpu|N              CPU synthesis (default), or CUDA device N; a GPU needs the\n"
    "                              package's GPU variant (its lib/ holds the CUDA runtime)\n"
    "  --served-model-name NAME    public model ID\n"
    "  --api-key KEY               bearer authentication\n"
    "  --api-key-file PATH         read the key from a file, keeping it off the command line\n"
    "  --cpu-kernels auto|optimized|reference   CPU kernel selection (default auto)\n"
    "  --threads N                 CPU compute threads (default 4)\n"
    "  --codec-threads N           audio codec threads (0/default: --threads)\n"
    "  --voice NAME                default voice from voices.json\n"
    "  --max-num-seqs N            requests synthesized at once, each by a worker with its own\n"
    "                              copy of the model (default 1, maximum 16)\n"
    "  --max-pending-requests N    requests queued or still being delivered (default 8, maximum 128)\n"
    "  --request-timeout SECONDS   queue plus synthesis deadline (default 300)\n"
    "  --max-input-characters N    longest input, synthesized sentence by sentence (default 4096,\n"
    "                              maximum 16384)\n"
    "Use surogate serve --tts surogate/amami-357m-ro to download and verify assets.\n";

std::string base64(std::string_view bytes) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < bytes.size(); i += 3) {
        const uint32_t v = (uint8_t(bytes[i]) << 16) | (uint8_t(bytes[i + 1]) << 8) | uint8_t(bytes[i + 2]);
        out += {alphabet[v >> 18], alphabet[(v >> 12) & 63], alphabet[(v >> 6) & 63], alphabet[v & 63]};
    }
    if (i + 1 == bytes.size()) {
        const uint32_t v = uint8_t(bytes[i]) << 16;
        out += {alphabet[v >> 18], alphabet[(v >> 12) & 63], '=', '='};
    } else if (i + 2 == bytes.size()) {
        const uint32_t v = (uint8_t(bytes[i]) << 16) | (uint8_t(bytes[i + 1]) << 8);
        out += {alphabet[v >> 18], alphabet[(v >> 12) & 63], alphabet[(v >> 6) & 63], '='};
    }
    return out;
}

// Invalid UTF-8 (in an error message) is replaced rather than thrown.
std::string sse(const json& event) {
    return "data: " + event.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

/// What a stream's synthesis hands its delivery: the audio produced and not yet sent, and how
/// synthesis ended. Synthesis runs on a thread of its own, so the worker never waits for the
/// client: it is handed back as soon as the last chunk is made, while delivery goes on at the
/// client's pace. The request keeps its place in the queue until its response ends, so slow
/// listeners are bounded by --max-pending-requests like any other request.
struct Production {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> backlog;
    bool finished = false;
    std::optional<HttpError> error; ///< why synthesis failed, once finished
    std::atomic<bool> gone{false};  ///< the client has left: synthesis stops
    size_t pcm_bytes = 0;
    bool counted     = false; ///< the request's outcome is recorded
};

/// A streamed /v1/audio/speech response (SUROGATE-CHANGES #6). The request takes its place at the
/// worker first, so a full queue or a queue timeout is still an HTTP error; the headers then go
/// out at once, and the audio follows chunk by chunk as the runtime produces it. X-Usage-Characters
/// is therefore sent before synthesis ends: a stream that fails partway ends without its final
/// chunk (raw audio) or without speech.audio.done (SSE), and was not delivered.
void stream_speech(const httplib::Request& q, httplib::Response& r, Runtime& runtime,
                   Metrics& metrics, std::vector<std::vector<int32_t>> chunks, const Voice& voice,
                   int seed, const std::string& format, bool events, size_t characters) {
    auto lease = std::shared_ptr<Runtime::Lease>(runtime.acquire([&q] {
        return interrupted != 0 || (q.is_connection_alive && !q.is_connection_alive());
    }).release());
    r.set_header("X-Usage-Characters", std::to_string(characters));
    r.set_header("X-Audio-Sample-Rate", "22050");
    r.set_header("Cache-Control", "no-store");
    r.set_header("X-Accel-Buffering", "no");
    const std::string content_type =
        events ? "text/event-stream" : (format == "wav" ? "audio/wav" : "audio/pcm");
    auto state = std::make_shared<Production>();
    r.set_chunked_content_provider(
        content_type,
        [&runtime, &metrics, state, lease, chunks = std::move(chunks), voice, seed, format, events,
         characters, started = false](size_t, httplib::DataSink& sink) mutable {
            if (started) {
                sink.done();
                return true;
            }
            started        = true;
            const auto put = [&](std::string_view bytes) { return sink.write(bytes.data(), bytes.size()); };
            const auto fail = [&](const HttpError& error) {
                metrics.request_done("speech", false);
                state->counted = true;
                if (events && put(sse({{"type", "error"},
                                       {"error", {{"message", error.what()},
                                                  {"type", error.status < 500 ? "invalid_request_error"
                                                                              : "server_error"}}}}))) {
                    sink.done();
                    return true;
                }
                return false; // raw audio: end the connection without the final chunk
            };
            // Synthesis: it hands the worker back the moment it ends; the queue place (the lease
            // itself) lasts until the response ends.
            std::thread producer;
            try {
                producer = std::thread([&runtime, state, lease, &chunks, &voice, seed, format] {
                    bool header = format == "wav";
                    std::optional<HttpError> error;
                    try {
                        runtime.stream(
                            *lease, chunks, voice, seed,
                            [&] { return interrupted != 0 || state->gone.load(); },
                            [&](std::string_view pcm) {
                                std::string out = header ? wav_header(22050, 0xFFFFFFFFu) : std::string();
                                header          = false;
                                out.append(pcm);
                                {
                                    const std::lock_guard lock(state->mutex);
                                    state->backlog.push_back(std::move(out));
                                    state->pcm_bytes += pcm.size();
                                }
                                state->changed.notify_one();
                                return !state->gone.load();
                            });
                    } catch (const HttpError& e) {
                        error = e;
                    } catch (const std::exception& e) {
                        error = HttpError(503, e.what());
                    }
                    lease->release_worker(); // the worker is free for the next request
                    {
                        const std::lock_guard lock(state->mutex);
                        state->finished = true;
                        state->error    = std::move(error);
                    }
                    state->changed.notify_one();
                });
            } catch (const std::exception&) { // std::system_error, or std::bad_alloc
                return fail(HttpError(503, "Cannot start TTS synthesis; retry the request"));
            }
            // Delivery, at the client's pace. The producer is always joined before returning.
            struct Join {
                std::thread& thread;
                Production& state;
                ~Join() {
                    state.gone = true;
                    thread.join();
                }
            } join{producer, *state};
            try {
                for (;;) {
                    if (interrupted) { // shutting down: no more waiting on this client
                        state->gone = true;
                        return fail(HttpError(503, "TTS server is shutting down"));
                    }
                    std::unique_lock lock(state->mutex);
                    state->changed.wait_for(lock, std::chrono::milliseconds(25),
                                            [&] { return !state->backlog.empty() || state->finished; });
                    if (!state->backlog.empty()) {
                        auto out = std::move(state->backlog.front());
                        state->backlog.pop_front();
                        lock.unlock();
                        const bool sent =
                            events ? put(sse({{"type", "speech.audio.delta"}, {"audio", base64(out)}}))
                                   : put(out);
                        if (!sent) {
                            state->gone = true; // the client has gone: synthesis stops
                            return fail(HttpError(503, "TTS request cancelled"));
                        }
                        continue;
                    }
                    if (state->finished) break;
                    lock.unlock();
                    // Nothing to send: notice a client that leaves while synthesis is under way.
                    if (sink.is_writable && !sink.is_writable()) {
                        state->gone = true;
                        return fail(HttpError(503, "TTS request cancelled"));
                    }
                }
            } catch (const std::exception& e) { // an exception must not leave a content provider
                state->gone = true;
                return fail(HttpError(503, e.what()));
            }
            if (state->error) return fail(*state->error);
            const double seconds = static_cast<double>(state->pcm_bytes) / (2.0 * 22050.0);
            if (events &&
                !put(sse({{"type", "speech.audio.done"},
                          {"usage", {{"input_characters", characters}, {"audio_seconds", seconds}}}})))
                return fail(HttpError(503, "TTS request cancelled"));
            metrics.add_audio_seconds(seconds);
            metrics.add_characters(characters);
            metrics.request_done("speech", true);
            state->counted = true;
            sink.done();
            return true;
        },
        // A response that ends before its stream started (the client left right after the
        // headers) still counts as a failed request.
        [&metrics, state](bool) {
            if (!state->counted) metrics.request_done("speech", false);
        });
}

int integer(const std::string& flag, const std::string& value) {
    try {
        size_t used = 0;
        const int n = std::stoi(value, &used);
        if (used == value.size()) return n;
    } catch (const std::exception&) {
    }
    throw std::invalid_argument(flag + " must be an integer");
}

void validate_policy(const json& policy) {
    if (policy.value("max_steps", 0) != 900 || policy.value("topk", 0) != 80 ||
        policy.value("longform_mode", std::string()) != "auto")
        throw std::invalid_argument("Unsupported TTS decoding policy");
    for (auto field : {"temperature", "cfg_scale"}) {
        double v = policy.at(field).get<double>();
        if (!std::isfinite(v) || !std::isfinite(static_cast<float>(v)) || v <= 0)
            throw std::invalid_argument("Invalid TTS decoding value");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact, host = "127.0.0.1", name, default_voice, kernels = "auto", device = "cpu";
        std::optional<std::string> key_flag, key_file; // --api-key, --api-key-file (last one wins)
        bool default_set = false;
        int port = 8080, max_pending = 8, threads = 4, codec_threads = 0, max_num_seqs = 1;
        int max_characters = static_cast<int>(default_max_characters);
        double timeout = 300;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::cout << usage;
                return 0;
            }
            auto value = [&]() {
                if (++i >= argc) throw std::invalid_argument(arg + " needs a value");
                return std::string(argv[i]);
            };
            if (arg == "--host")
                host = value();
            else if (arg == "--cpu-kernels")
                kernels = value();
            else if (arg == "--threads")
                threads = integer(arg, value());
            else if (arg == "--codec-threads")
                codec_threads = integer(arg, value());
            else if (arg == "--port")
                port = integer(arg, value());
            else if (arg == "--device") {
                device = value();
                if (device != "cpu" && (device.empty() ||
                                        device.find_first_not_of("0123456789") != std::string::npos ||
                                        device.size() > 3))
                    throw std::invalid_argument("--device must be cpu or a CUDA device index");
                if (device != "cpu") device = std::to_string(std::stoul(device)); // "00" is 0
            } else if (arg == "--served-model-name")
                name = value();
            else if (arg == "--api-key")
                key_flag = value();
            else if (arg == "--api-key-file")
                key_file = value();
            else if (arg == "--voice") {
                default_voice = value();
                default_set   = true;
            } else if (arg == "--max-pending-requests")
                max_pending = integer(arg, value());
            else if (arg == "--max-num-seqs")
                max_num_seqs = integer(arg, value());
            else if (arg == "--max-input-characters")
                max_characters = integer(arg, value());
            else if (arg == "--request-timeout") {
                auto s      = value();
                size_t used = 0;
                try {
                    timeout = std::stod(s, &used);
                } catch (const std::exception&) {
                    used = std::string::npos;
                }
                if (used != s.size())
                    throw std::invalid_argument("--request-timeout must be a number of seconds");
            } else if (!arg.starts_with('-') && artifact.empty())
                artifact = arg;
            else
                throw std::invalid_argument("Unknown option: " + arg);
        }
        if (max_num_seqs < 1 || max_num_seqs > 16)
            throw std::invalid_argument("--max-num-seqs must be between 1 and 16");
        if (artifact.empty() || host.empty() || port < 1 || port > 65535 || max_pending < 0 ||
            max_pending > 128 || !std::isfinite(timeout) || timeout <= 0 || timeout > 3600)
            throw std::invalid_argument(usage);
        if (kernels != "auto" && kernels != "optimized" && kernels != "reference")
            throw std::invalid_argument("--cpu-kernels must be auto, optimized or reference");
        if (threads < 1 || threads > 256 || codec_threads < 0 || codec_threads > 256)
            throw std::invalid_argument("Thread counts must be between 1 and 256");
        // The spoken text is at most 16384 characters (frontend.h), about 24 minutes of speech,
        // within the 96 MiB of audio a request may produce.
        if (max_characters < 1 || max_characters > 16384)
            throw std::invalid_argument("--max-input-characters must be between 1 and 16384");
        if (!codec_threads) codec_threads = threads;
        const std::string key = sinfer::serve::resolve_api_key(key_flag, key_file);
        auto root = std::filesystem::canonical(artifact);
        if (root.filename() == "voices.json") root = root.parent_path();
        std::ifstream file(root / "voices.json");
        json profile = json::parse(file);
        if (profile.value("schema", 0) != 1 ||
            profile.value("method", std::string()) != "experimental_magpie_native_cpu" ||
            nlohmann::json(profile.at("tokenizer")) !=
                nlohmann::json{{"profile", "v2607"}, {"offset", 96}, {"eos", 3358}})
            throw std::invalid_argument("Unsupported native TTS package");
        auto policy = profile.at("decoding");
        validate_policy(policy);
        std::vector<Voice> voices;
        std::set<std::string> names;
        for (auto& [label, entry] : profile.at("voices").items()) {
            auto decoding = policy;
            if (entry.contains("decoding")) decoding.update(entry.at("decoding"));
            validate_policy(decoding);
            if (label.empty() || !names.insert(casefold(label)).second ||
                !entry.at("id").is_number_integer() || entry.at("id").get<int64_t>() < 0 ||
                entry.at("id").get<int64_t>() > INT32_MAX)
                throw std::invalid_argument("Invalid voice profile");
            voices.push_back({label, entry.at("id").get<int>(),
                              decoding.at("temperature").get<float>(),
                              decoding.at("cfg_scale").get<float>()});
        }
        if (voices.empty()) throw std::invalid_argument("No named voices in TTS package");
        // The generator GGUF is named by the profile; packages before the field used model.gguf.
        const std::string model_file = profile.value("model", std::string("model.gguf"));
        if (model_file.empty() || model_file.find('/') != std::string::npos || model_file.starts_with(".") ||
            !model_file.ends_with(".gguf") || !profile.at("files").contains(model_file))
            throw std::invalid_argument("Invalid model file in TTS package");
        auto voice = [&](const std::string& label) -> const Voice& {
            for (const auto& v : voices)
                if (casefold(v.name) == casefold(label)) return v;
            throw std::invalid_argument("Unknown voice; see /v1/audio/voices");
        };
        if (!default_set) default_voice = voices.front().name;
        default_voice = voice(default_voice).name;
        if (name.empty()) name = artifact;
        Runtime runtime(root, max_pending, timeout, threads, codec_threads, kernels, device, max_num_seqs,
                        model_file);
        const std::string device_name = device == "cpu" ? "cpu" : "cuda:" + device;
        // /metrics: requests in flight (queued ones included) and what was served -- what a
        // supervisor drains this server by.
        sinfer::serve::audio::Metrics metrics(name, {.characters = true});
        metrics.declare_endpoint("speech");
        httplib::Server server;
        // 16 bytes a character: room for JSON escapes of any input within the limit.
        configure(server, key, std::max<size_t>(64 * 1024, 16 * size_t(max_characters)), &metrics);
        metrics.serve(server);
        // A thread for every request that may run, wait or be delivered (whole recordings and
        // streams keep their queue place until delivered), and spare ones for GETs and idle
        // keep-alive connections, which hold a thread until they time out (1 s here).
        const size_t http_threads = size_t(max_pending) + size_t(max_num_seqs) + 16;
        server.new_task_queue     = [http_threads] {
            return new httplib::ThreadPool(http_threads, http_threads);
        };
        server.set_keep_alive_timeout(1);
        server.Get("/health", [&](const auto&, auto& r) {
            // Ready while any worker can serve; "workers" says how many of them can.
            const int ready = runtime.ready_workers();
            response(r,
                     {{"status", ready == runtime.workers() ? "ok" : ready ? "degraded" : "unavailable"},
                      {"device", device_name},
                      {"workers", {{"ready", ready}, {"total", runtime.workers()}}}},
                     ready ? 200 : 503);
        });
        server.Get("/v1/models", [&](const auto&, auto& r) {
            response(
                r,
                {{"object", "list"},
                 {"data",
                  json::array({{{"id", name}, {"object", "model"}, {"owned_by", "surogate"}}})}});
        });
        server.Get("/v1/audio/voices", [&](const auto&, auto& r) {
            auto data = json::array();
            for (auto& v : voices)
                data.push_back({{"id", v.name}, {"name", v.name}, {"language", "ro"}});
            response(r, {{"object", "list"}, {"default_voice", default_voice}, {"data", data}});
        });
        server.Post("/v1/audio/speech", [&](const httplib::Request& q, httplib::Response& r) {
            try {
                // A part of a recording is neither synthesized nor billed separately.
                if (q.has_header("Range")) throw HttpError(416, "Range requests are not supported");
                auto content_type = q.get_header_value("Content-Type");
                if (content_type.substr(0, content_type.find(';')) != "application/json")
                    throw HttpError(415, "Content-Type must be application/json");
                auto body = json::parse(q.body);
                if (!body.is_object()) throw std::invalid_argument("Expected a JSON object");
                const std::set<std::string> fields = {"model", "input", "voice", "response_format",
                                                      "speed", "seed", "stream_format"};
                for (auto& [field, value] : body.items())
                    if (!fields.contains(field))
                        throw std::invalid_argument("Unsupported speech field: " + field);
                if (body.value("model", name) != name)
                    throw std::invalid_argument("Unknown model; see /v1/models");
                if (!body.contains("input") || !body.at("input").is_string())
                    throw std::invalid_argument("input must contain text");
                if (body.contains("speed") &&
                    (!body["speed"].is_number() || body["speed"].get<double>() != 1))
                    throw std::invalid_argument("This model supports speed=1 only");
                int64_t seed = 9;
                if (body.contains("seed")) {
                    if (!body["seed"].is_number_integer() || body["seed"].get<double>() < 0 ||
                        body["seed"].get<double>() > INT32_MAX)
                        throw std::invalid_argument("seed must be an integer between 0 and 2147483647");
                    seed = body["seed"].get<int64_t>();
                }
                auto format = body.value("response_format", std::string("wav"));
                if (format != "wav" && format != "pcm")
                    throw std::invalid_argument("response_format must be wav or pcm");
                // Absent: the whole recording in one body, as before. "audio": the recording's bytes
                // as they are synthesized (a WAV header of unknown length first); "sse": the same
                // bytes base64-encoded in speech.audio.delta events, then speech.audio.done.
                std::optional<std::string> stream_format;
                if (body.contains("stream_format")) {
                    if (!body["stream_format"].is_string() ||
                        (body["stream_format"] != "audio" && body["stream_format"] != "sse"))
                        throw std::invalid_argument("stream_format must be audio or sse");
                    stream_format = body["stream_format"].get<std::string>();
                }
                const auto& selected = voice(body.value("voice", default_voice));
                const auto& input    = body.at("input").get_ref<const std::string&>();
                auto chunks          = tokenize(input, size_t(max_characters));
                // Counted like the input limit. A whole recording carries it only once synthesis has
                // succeeded, so an error response never carries a billable count; a stream carries it
                // from the start and is billable only when it completes (see stream_speech).
                const auto characters = input_characters(input);
                if (stream_format) {
                    stream_speech(q, r, runtime, metrics, std::move(chunks), selected, int(seed), format,
                                  *stream_format == "sse", characters);
                    return;
                }
                const auto cancelled = [&] {
                    return interrupted || (q.is_connection_alive && !q.is_connection_alive());
                };
                // A whole recording keeps its queue place until it has been delivered, as a stream
                // does, so --max-pending-requests bounds the responses being written as well; its
                // worker goes to the next request as soon as synthesis ends.
                auto lease = std::shared_ptr<Runtime::Lease>(runtime.acquire(cancelled).release());
                std::string pcm;
                const int rate = runtime.stream(*lease, chunks, selected, int(seed), cancelled,
                                                [&](std::string_view part) {
                                                    pcm.append(part);
                                                    return true;
                                                });
                lease->release_worker();
                // 16-bit mono at 22,050 Hz.
                const double seconds = static_cast<double>(pcm.size()) / (2.0 * 22050.0);
                auto data            = std::make_shared<std::string>();
                if (format == "wav") {
                    *data = wav_header(static_cast<uint32_t>(rate), static_cast<uint32_t>(pcm.size()));
                }
                data->append(pcm);
                pcm = std::string();
                r.set_header("X-Usage-Characters", std::to_string(characters));
                r.set_header("X-Audio-Sample-Rate", "22050");
                r.set_header("Cache-Control", "no-store");
                r.set_content_provider(
                    data->size(), format == "wav" ? "audio/wav" : "audio/pcm",
                    [data, lease](size_t offset, size_t length, httplib::DataSink& sink) {
                        return sink.write(data->data() + offset, std::min(length, data->size() - offset));
                    });
                metrics.add_audio_seconds(seconds);
                metrics.add_characters(characters);
                metrics.request_done("speech", true);
            } catch (...) {
                metrics.request_done("speech", false);
                throw;
            }
        });
        std::signal(SIGPIPE, SIG_IGN);
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
        if (!server.bind_to_port(host, port)) throw std::runtime_error("Cannot bind TTS server");
        std::jthread shutdown([&](std::stop_token token) {
            while (!token.stop_requested()) {
                if (interrupted) {
                    runtime.shut_down(); // a cancelled request's worker is killed, not drained
                    server.stop();
                    return;
                }
                runtime.revive(); // a worker that died while idle, once every worker has loaded
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        });
        // Every worker loads its model and synthesizes once before the server answers.
        const auto warmup_chunks = tokenize("Bună.");
        runtime.start([&](Runtime::Lease& lease) {
            runtime.stream(lease, warmup_chunks, voice(default_voice), 9,
                           [] { return interrupted != 0; }, [](std::string_view) { return true; });
        });
        if (interrupted) return 0;
        std::cerr << "TTS ready on " << device_name << " at http://" << host << ':' << port << '\n';
        if (!server.listen_after_bind() && !interrupted)
            throw std::runtime_error("TTS HTTP server failed");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "surogate-tts: " << e.what() << '\n';
        return interrupted ? 0 : 1;
    }
}
