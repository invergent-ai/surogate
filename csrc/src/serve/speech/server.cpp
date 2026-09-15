// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#include "model.h"
#include "audio.h"
#include <httplib.h>
#include <ATen/Parallel.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>

namespace {
using namespace sinfer::speech;
const char* usage = "surogate-stt MODEL [--host HOST] [--port PORT] [--device N|cpu]\n"
                    "  --served-model-name NAME   public model ID\n"
                    "  --api-key KEY             authenticate requests\n"
                    "  --max-num-seqs N          live streams (default 8)\n"
                    "surogate serve --stt MODEL --lm PATH prepares a local NeMo checkpoint.\n";

void response(httplib::Response& r, const json& body, int code = 200) {
    r.status = code;
    r.set_content(body.dump(), "application/json");
}

void error(httplib::Response& r, const std::string& message, int code) {
    response(
        r,
        {{"error",
          {{"message", message}, {"type", code < 500 ? "invalid_request_error" : "server_error"}}}},
        code);
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact, host = "127.0.0.1", device = "0", name, key;
        int port = 8080, limit = 8;
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
            else if (arg == "--port")
                port = std::stoi(value());
            else if (arg == "--device")
                device = value();
            else if (arg == "--served-model-name")
                name = value();
            else if (arg == "--api-key")
                key = value();
            else if (arg == "--max-num-seqs")
                limit = std::stoi(value());
            else if (!arg.starts_with('-') && artifact.empty())
                artifact = arg;
            else
                throw std::invalid_argument("unknown option: " + arg);
        }
        if (artifact.empty() || port < 1 || port > 65535 || limit < 1 || limit > 128)
            throw std::invalid_argument(usage);
        if (name.empty()) name = artifact;
        at::set_num_threads(4);
        at::set_num_interop_threads(1);
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
        c10::InferenceMode inference;
        Model model(artifact, device);
        std::mutex mutex;

        struct Session {
            std::unique_ptr<Stream> stream;
            std::chrono::steady_clock::time_point used;
        };

        std::map<std::string, Session> sessions;
        std::random_device random;
        httplib::Server server;
        server.set_payload_max_length(64 * 1024 * 1024);
        server.set_read_timeout(60);
        server.set_pre_routing_handler([&](const httplib::Request& q, httplib::Response& r) {
            if (!key.empty() && q.get_header_value("Authorization") != "Bearer " + key) {
                error(r, "invalid API key", 401);
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server.set_exception_handler([](const auto&, auto& r, std::exception_ptr exception) {
            try {
                std::rethrow_exception(exception);
            } catch (const std::invalid_argument& e) {
                error(r, e.what(), 400);
            } catch (const std::exception& e) {
                std::cerr << "speech request: " << e.what() << '\n';
                error(r, "speech inference failed", 500);
            }
        });
        server.Get("/health", [](const auto&, auto& r) { response(r, {{"status", "ok"}}); });
        server.Get("/v1/models", [&](const auto&, auto& r) {
            response(
                r,
                {{"object", "list"},
                 {"data",
                  json::array({{{"id", name}, {"object", "model"}, {"owned_by", "surogate"}}})}});
        });
        server.Post("/v1/audio/transcriptions", [&](const auto& q, auto& r) {
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
            c10::InferenceMode guard;
            Stream stream(model, artifact);
            auto events = stream.accept(pcm, true);
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
            if (format == "text")
                r.set_content(text, "text/plain; charset=utf-8");
            else if (format == "verbose_json")
                response(r, {{"text", text},
                             {"language", "romanian"},
                             {"duration", pcm.size() / 16000.},
                             {"task", "transcribe"}});
            else
                response(r, {{"text", text}});
        });
        auto prune = [&]() {
            auto now = std::chrono::steady_clock::now();
            for (auto it = sessions.begin(); it != sessions.end();)
                if (now - it->second.used > std::chrono::minutes(2))
                    it = sessions.erase(it);
                else
                    ++it;
        };
        server.Post("/v1/audio/streams", [&](const auto& q, auto& r) {
            if (!q.body.empty() && q.body != "{}")
                throw std::invalid_argument("stream creation takes an empty body; send mono 16 kHz "
                                            "PCM16 to the returned stream");
            std::lock_guard lock(mutex);
            c10::InferenceMode guard;
            prune();
            if (sessions.size() >= size_t(limit)) {
                error(r, "maximum live speech streams reached", 429);
                return;
            }
            std::ostringstream id;
            for (int i = 0; i < 4; ++i) id << std::hex << random();
            sessions.emplace(id.str(), Session{std::make_unique<Stream>(model, artifact),
                                               std::chrono::steady_clock::now()});
            response(r,
                     {{"id", id.str()},
                      {"sample_rate", 16000},
                      {"channels", 1},
                      {"encoding", "pcm_s16le"}},
                     201);
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
            c10::InferenceMode guard;
            prune();
            auto it = sessions.find(q.matches[1]);
            if (it == sessions.end()) {
                error(r, "unknown or expired speech stream", 404);
                return;
            }
            it->second.used = std::chrono::steady_clock::now();
            try {
                auto events = it->second.stream->accept(pcm, finish);
                if (finish) sessions.erase(it);
                response(r, {{"events", events}});
            } catch (...) {
                sessions.erase(it);
                throw;
            }
        });
        server.Delete(R"(/v1/audio/streams/([0-9a-f]+))", [&](const auto& q, auto& r) {
            std::lock_guard lock(mutex);
            if (!sessions.erase(q.matches[1])) {
                error(r, "unknown speech stream", 404);
                return;
            }
            response(r, {{"deleted", true}});
        });
        std::cerr << "surogate-stt: serving " << name << " on " << host << ':' << port << '\n';
        if (!server.listen(host, port)) throw std::runtime_error("cannot bind speech server");
    } catch (const std::exception& e) {
        std::cerr << "surogate-stt: " << e.what() << '\n';
        return 1;
    }
}
