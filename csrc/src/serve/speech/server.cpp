// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#include "model.h"
#include "audio.h"
#include "transcription.h"
#include "../serve/api_key_file.h"
#include "../serve/audio_metrics.h"
#include "../serve/websocket.h"
#include "../serve/audio_http.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>

namespace {
using namespace sinfer::speech;
const char* usage =
    "surogate-stt MODEL [--host HOST] [--port PORT] [--device N|cpu]\n"
    "  --cpu-kernels auto|optimized|reference   CPU kernel selection (default auto)\n"
    "  --threads N               CPU compute threads (default 4)\n"
    "  --served-model-name NAME   public model ID\n"
    "  --api-key KEY             authenticate requests\n"
    "  --api-key-file PATH       read the key from a file, keeping it off the command line\n"
    "  --max-num-seqs N          live streams (default 8)\n"
    "surogate serve --stt MODEL --lm PATH prepares a local NeMo checkpoint.\n";

using sinfer::serve::audio::response;
using sinfer::serve::audio::error;
} // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact, host = "127.0.0.1", device = "0", name, kernels = "auto";
        std::optional<std::string> key_flag, key_file; // --api-key, --api-key-file (last one wins)
        int port = 8080, limit = 8, threads = 4;
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
            else if (arg == "--threads") {
                auto text   = value();
                size_t used = 0;
                threads     = std::stoi(text, &used);
                if (used != text.size())
                    throw std::invalid_argument("--threads must be an integer");
            } else if (arg == "--port")
                port = std::stoi(value());
            else if (arg == "--device")
                device = value();
            else if (arg == "--served-model-name")
                name = value();
            else if (arg == "--api-key")
                key_flag = value();
            else if (arg == "--api-key-file")
                key_file = value();
            else if (arg == "--max-num-seqs")
                limit = std::stoi(value());
            else if (!arg.starts_with('-') && artifact.empty())
                artifact = arg;
            else
                throw std::invalid_argument("unknown option: " + arg);
        }
        if (artifact.empty() || port < 1 || port > 65535 || limit < 1 || limit > 128 ||
            threads < 1 || threads > 256)
            throw std::invalid_argument(usage);
        if (name.empty()) name = artifact;
        const std::string key = sinfer::serve::resolve_api_key(key_flag, key_file);
        at::set_num_threads(threads);
        at::set_num_interop_threads(1);
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
        c10::InferenceMode inference;
        if (kernels != "auto" && kernels != "optimized" && kernels != "reference")
            throw std::invalid_argument("--cpu-kernels must be auto, optimized or reference");
        auto mode = kernels == "reference"   ? CpuKernels::Reference
                    : kernels == "optimized" ? CpuKernels::Optimized
                                             : CpuKernels::Auto;
        Model model(artifact, device, mode);
        if (device == "cpu")
            std::cerr << "STT CPU kernels: "
                      << (model.optimized_cpu() ? "optimized FP32" : "reference")
                      << "; threads=" << threads << '\n';
        std::mutex mutex;

        struct Session {
            std::unique_ptr<Stream> stream;
            std::chrono::steady_clock::time_point used;
            /// A WebSocket's stream: its connection owns it (no expiry, no HTTP access).
            bool websocket = false;
        };

        std::map<std::string, Session> sessions;
        std::random_device random;
        // /metrics: requests in flight, realtime streams open, and what was served -- what a
        // supervisor drains this server by.
        sinfer::serve::audio::Metrics metrics(name, {.streams = true});
        metrics.declare_endpoint("transcriptions");
        metrics.declare_endpoint("streams");
        httplib::Server server;
        sinfer::serve::audio::configure(server, key, 64 * 1024 * 1024, &metrics);
        // A WebSocket stream holds an HTTP thread for its whole life, and so does an idle keep-alive
        // connection: room for every live stream, as many requests beside them, and more; requests
        // beyond that wait in an unbounded queue rather than being dropped.
        const size_t http_threads =
            std::max<size_t>(CPPHTTPLIB_THREAD_POOL_COUNT, 2 * size_t(limit) + 8);
        server.new_task_queue = [http_threads] { return new httplib::ThreadPool(http_threads); };
        server.Get("/health", [](const auto&, auto& r) { response(r, {{"status", "ok"}}); });
        server.Get("/v1/models", [&](const auto&, auto& r) {
            response(
                r,
                {{"object", "list"},
                 {"data",
                  json::array({{{"id", name}, {"object", "model"}, {"owned_by", "surogate"}}})}});
        });
        server.Post("/v1/audio/transcriptions", [&](const auto& q, auto& r) {
            try {
                if (!q.has_file("file"))
                    throw std::invalid_argument("multipart audio file is required");
                std::string format = "json";
                for (auto& [field, value] : q.files) {
                    if (field == "model" && value.content != name)
                        throw std::invalid_argument("unknown model");
                    if (field == "language" && value.content != "ro")
                        throw std::invalid_argument("this model supports Romanian (ro)");
                    if (field == "response_format") format = value.content;
                    if (field != "file" && field != "model" && field != "language" &&
                        field != "response_format")
                        throw std::invalid_argument("unsupported transcription field: " + field);
                }
                if (format != "json" && format != "text" && format != "verbose_json")
                    throw std::invalid_argument("response_format must be json, text or verbose_json");
                auto pcm = decode_audio(q.get_file_value("file").content);
                std::lock_guard lock(mutex);
                // OpenMP/MKL thread settings belong to the HTTP worker thread.
                at::set_num_threads(threads);
                c10::InferenceMode guard;
                std::vector<json> events;
                if (model.streaming()) {
                    Stream stream(model, artifact);
                    events = stream.accept(pcm, true);
                } else {
                    // Full-context features and attention see the complete file.
                    // In particular, do not run VAD or normalize separate chunks.
                    auto samples    = at::from_blob(pcm.data(), {int64_t(pcm.size())}, at::kFloat);
                    auto transcript = pcm.size() < 320
                                          ? std::string()
                                          : model.beam(model.ctc(model.encode(model.mel(samples))));
                    events.push_back({{"type", "final"}, {"text", transcript}});
                }
                std::string text;
                for (auto& event : events)
                    if (event["type"] == "final") {
                        auto segment = event["text"].template get<std::string>();
                        auto first   = segment.find_first_not_of(" \t\n\r");
                        if (first == std::string::npos) continue;
                        segment =
                            segment.substr(first, segment.find_last_not_of(" \t\n\r") - first + 1);
                        if (!text.empty()) text += ' ';
                        text += segment;
                    }
                // Every format reports the billed seconds (usage in JSON, and a header on all).
                auto reply = sinfer::speech::transcription_reply(format, text, pcm.size());
                for (auto& [header, value] : reply.headers) r.set_header(header, value);
                r.status = 200;
                r.set_content(std::move(reply.body), reply.content_type);
                metrics.add_audio_seconds(sinfer::speech::transcription_seconds(pcm.size()));
                metrics.request_done("transcriptions", true);
            } catch (...) {
                metrics.request_done("transcriptions", false);
                throw;
            }
        });
        auto prune = [&]() {
            auto now = std::chrono::steady_clock::now();
            for (auto it = sessions.begin(); it != sessions.end();)
                if (!it->second.websocket && now - it->second.used > std::chrono::minutes(2))
                    it = sessions.erase(it);
                else
                    ++it;
            metrics.set_open_streams(sessions.size());
        };
        // Expired streams stop counting even while no stream request arrives to prune them --
        // exactly when a supervisor is draining -- but a scrape never waits behind inference.
        metrics.serve(server, [&] {
            std::unique_lock lock(mutex, std::try_to_lock);
            if (lock.owns_lock()) prune();
        });
        server.Post("/v1/audio/streams", [&](const auto& q, auto& r) {
            try {
                if (!model.streaming())
                    throw std::invalid_argument(
                        "this model needs complete audio; use /v1/audio/transcriptions or serve the "
                        "streaming sibling for live audio");
                if (!q.body.empty() && q.body != "{}")
                    throw std::invalid_argument("stream creation takes an empty body; send mono 16 kHz "
                                                "PCM16 to the returned stream");
                std::lock_guard lock(mutex);
                // OpenMP/MKL thread settings belong to the HTTP worker thread.
                at::set_num_threads(threads);
                c10::InferenceMode guard;
                prune();
                if (sessions.size() >= size_t(limit)) {
                    metrics.request_done("streams", false);
                    error(r, "maximum live speech streams reached", 429);
                    return;
                }
                std::ostringstream id;
                for (int i = 0; i < 4; ++i) id << std::hex << random();
                sessions.emplace(id.str(), Session{std::make_unique<Stream>(model, artifact),
                                                   std::chrono::steady_clock::now()});
                metrics.set_open_streams(sessions.size());
                response(r,
                         {{"id", id.str()},
                          {"sample_rate", 16000},
                          {"channels", 1},
                          {"encoding", "pcm_s16le"}},
                         201);
                metrics.request_done("streams", true);
            } catch (...) {
                metrics.request_done("streams", false);
                throw;
            }
        });
        server.Post(R"(/v1/audio/streams/([0-9a-f]+))", [&](const auto& q, auto& r) {
            if (q.body.size() > 320000 || q.body.size() % 2)
                throw std::invalid_argument(
                    "send at most 10 seconds of little-endian PCM16 per chunk");
            bool finish = q.has_param("finish");
            if (finish && q.get_param_value("finish") != "true")
                throw std::invalid_argument("finish must be true");
            std::vector<float> pcm(q.body.size() / 2);
            for (size_t i = 0; i < pcm.size(); ++i) {
                uint16_t u = static_cast<unsigned char>(q.body[2 * i]) |
                             (static_cast<unsigned char>(q.body[2 * i + 1]) << 8);
                pcm[i] = static_cast<int16_t>(u) / 32768.f;
            }
            std::lock_guard lock(mutex);
            // OpenMP/MKL thread settings belong to the HTTP worker thread.
            at::set_num_threads(threads);
            c10::InferenceMode guard;
            prune();
            auto it = sessions.find(q.matches[1]);
            if (it == sessions.end() || it->second.websocket) {
                error(r, "unknown or expired speech stream", 404);
                return;
            }
            it->second.used = std::chrono::steady_clock::now();
            try {
                auto events = it->second.stream->accept(pcm, finish);
                metrics.add_audio_seconds(sinfer::speech::transcription_seconds(pcm.size()));
                if (finish) sessions.erase(it);
                metrics.set_open_streams(sessions.size());
                response(r, {{"events", events}});
            } catch (...) {
                sessions.erase(it);
                metrics.set_open_streams(sessions.size());
                throw;
            }
        });
        // Realtime transcription over one WebSocket (SUROGATE-CHANGES #9): the same stream as the HTTP
        // interface, without a round trip per chunk. Binary messages carry PCM16 chunks; each event
        // the HTTP interface would return comes back as one JSON text message; the text message
        // {"type": "finish"} drains the audio, sends the last events and {"type": "done"}, and closes.
        server.Get("/v1/audio/streams", [&](const httplib::Request& q, httplib::Response& r) {
            namespace ws = sinfer::serve::websocket;
            if (const auto problem = ws::handshake_error(q)) {
                error(r, problem->message + "; open a realtime stream as a WebSocket, or POST to create one",
                      problem->status);
                ws::refuse_handshake(q, r);
                return;
            }
            if (!model.streaming()) {
                error(r, "this model needs complete audio; use /v1/audio/transcriptions or serve the "
                         "streaming sibling for live audio", 400);
                return;
            }
            std::string id;
            try {
                std::lock_guard lock(mutex);
                at::set_num_threads(threads);
                c10::InferenceMode guard;
                prune();
                if (sessions.size() >= size_t(limit)) {
                    metrics.request_done("streams", false);
                    error(r, "maximum live speech streams reached", 429);
                    return;
                }
                do { // a fresh id, never one in use
                    std::ostringstream name_stream;
                    for (int i = 0; i < 4; ++i) name_stream << std::hex << random();
                    id = name_stream.str();
                } while (sessions.contains(id));
                sessions.emplace(id, Session{std::make_unique<Stream>(model, artifact),
                                             std::chrono::steady_clock::now(), true});
                metrics.set_open_streams(sessions.size());
            } catch (...) {
                metrics.request_done("streams", false);
                throw;
            }
            metrics.request_done("streams", true);
            // The stream ends with the session, or with the response if the session never runs (the
            // client left during the handshake): whichever releases this last.
            const auto owner = std::shared_ptr<void>(nullptr, [&, id](void*) {
                std::lock_guard lock(mutex);
                sessions.erase(id);
                metrics.set_open_streams(sessions.size());
            });
            ws::accept(q, r, [&, id, owner](httplib::Stream& connection) {
                const auto forget = [&] {
                    std::lock_guard lock(mutex);
                    sessions.erase(id);
                    metrics.set_open_streams(sessions.size());
                };
                // Invalid UTF-8 in an event (a stray byte from the model) is replaced, not thrown.
                const auto text = [](const json& message) {
                    return message.dump(-1, ' ', false, json::error_handler_t::replace);
                };
                const auto send_events = [&](const std::vector<json>& events) {
                    for (const auto& event : events)
                        if (!ws::send(connection, ws::Opcode::Text, text(event))) return false;
                    return true;
                };
                // The stream is given back before the close handshake, which may wait on the peer
                // -- unless a transcription holds the lock: then the close goes first (it waits for
                // the peer at most a round trip) and the stream follows, rather than the peer
                // waiting on someone else's audio for its close.
                const auto close_and_forget = [&](uint16_t code, const std::string& reason) {
                    std::unique_lock lock(mutex, std::try_to_lock);
                    if (lock.owns_lock()) {
                        sessions.erase(id);
                        metrics.set_open_streams(sessions.size());
                        lock.unlock();
                        ws::close(connection, code, reason);
                        return;
                    }
                    ws::close(connection, code, reason);
                    forget();
                };
                const auto fail = [&](const std::string& message, uint16_t code) {
                    (void)ws::send(connection, ws::Opcode::Text,
                                   text({{"type", "error"}, {"error", {{"message", message}}}}));
                    close_and_forget(code, message);
                };
                // Inference errors are logged, and the client gets a fixed message: what() may
                // carry a backtrace with source paths (c10::Error).
                const auto failed_inference = [&](const std::exception& e) {
                    std::cerr << "surogate-stt: speech stream: " << e.what() << '\n';
                    fail(dynamic_cast<const std::invalid_argument*>(&e) ? e.what() : "speech inference failed",
                         1011);
                };
                try {
                    (void)ws::send(connection, ws::Opcode::Text,
                                   text({{"type", "ready"}, {"sample_rate", 16000}, {"channels", 1},
                                         {"encoding", "pcm_s16le"}}));
                    for (;;) {
                        ws::Failure failure;
                        auto message = ws::receive(connection, 320000, &failure);
                        if (!message) {
                            if (failure.code != 0) {
                                fail(failure.reason, failure.code); // the client broke the protocol
                            } else {
                                // Broken, or silent past the read timeout: say so if it can hear.
                                close_and_forget(1001, "connection lost or idle");
                            }
                            return;
                        }
                        if (message->opcode == ws::Opcode::Close) {
                            close_and_forget(1000, "");
                            return;
                        }
                        bool finish = false;
                        std::vector<float> pcm;
                        if (message->opcode == ws::Opcode::Text) {
                            const auto control = json::parse(message->payload, nullptr, false);
                            if (!control.is_object() || !control.contains("type") ||
                                !control["type"].is_string() || control["type"] != "finish") {
                                fail("send PCM16 audio as binary messages, or {\"type\": \"finish\"}", 1003);
                                return;
                            }
                            finish = true;
                        } else {
                            const auto& body = message->payload;
                            if (body.size() % 2) {
                                fail("send whole little-endian PCM16 samples, at most 10 seconds per message",
                                     1007);
                                return;
                            }
                            pcm.resize(body.size() / 2);
                            for (size_t i = 0; i < pcm.size(); ++i) {
                                uint16_t u = static_cast<unsigned char>(body[2 * i]) |
                                             (static_cast<unsigned char>(body[2 * i + 1]) << 8);
                                pcm[i] = static_cast<int16_t>(u) / 32768.f;
                            }
                        }
                        std::vector<json> events;
                        bool ended = false;
                        try {
                            std::lock_guard lock(mutex);
                            at::set_num_threads(threads);
                            c10::InferenceMode guard;
                            auto it = sessions.find(id);
                            if (it == sessions.end()) { // cannot happen while the connection holds it
                                ended = true;
                            } else {
                                it->second.used = std::chrono::steady_clock::now();
                                events          = it->second.stream->accept(pcm, finish);
                                metrics.add_audio_seconds(sinfer::speech::transcription_seconds(pcm.size()));
                                if (finish) {
                                    sessions.erase(it);
                                    metrics.set_open_streams(sessions.size());
                                }
                            }
                        } catch (const std::exception& e) {
                            failed_inference(e); // outside the lock: fail() takes it again
                            return;
                        }
                        if (ended) {
                            fail("speech stream ended", 1011);
                            return;
                        }
                        if (!send_events(events)) {
                            forget();
                            return;
                        }
                        if (finish) {
                            (void)ws::send(connection, ws::Opcode::Text, text({{"type", "done"}}));
                            ws::close(connection, 1000);
                            return;
                        }
                    }
                } catch (const std::exception& e) {
                    failed_inference(e);
                }
            });
        });
        server.Delete(R"(/v1/audio/streams/([0-9a-f]+))", [&](const auto& q, auto& r) {
            std::lock_guard lock(mutex);
            const auto it = sessions.find(q.matches[1]);
            if (it == sessions.end() || it->second.websocket) {
                error(r, "unknown speech stream", 404);
                return;
            }
            sessions.erase(it);
            metrics.set_open_streams(sessions.size());
            response(r, {{"deleted", true}});
        });
        std::cerr << "surogate-stt: serving " << name << " on " << host << ':' << port << '\n';
        if (!server.listen(host, port)) throw std::runtime_error("cannot bind speech server");
    } catch (const std::exception& e) {
        std::cerr << "surogate-stt: " << e.what() << '\n';
        return 1;
    }
}
