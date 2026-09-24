// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#include "frontend.h"
#include "runtime.h"
#include "../serve/api_key_file.h"
#include "../serve/audio_metrics.h"
#include "../serve/audio_http.h"
#include <atomic>
#include <cmath>
#include <csignal>
#include <fstream>
#include <set>
#include <thread>

namespace {
using namespace sinfer::tts;
using namespace sinfer::serve::audio;
volatile std::sig_atomic_t interrupted = 0;

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
    "  --max-pending-requests N    queued requests (default 8, maximum 128)\n"
    "  --request-timeout SECONDS   queue plus synthesis deadline (default 300)\n"
    "Use surogate serve --tts surogate/surogate-ro-tts to download and verify assets.\n";

int integer(const std::string& value) {
    size_t used = 0;
    int n       = std::stoi(value, &used);
    if (used != value.size()) throw std::invalid_argument("Invalid integer option");
    return n;
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
        int port = 8080, max_pending = 8, threads = 4, codec_threads = 0;
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
                threads = integer(value());
            else if (arg == "--codec-threads")
                codec_threads = integer(value());
            else if (arg == "--port")
                port = integer(value());
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
                max_pending = integer(value());
            else if (arg == "--request-timeout") {
                auto s = value();
                size_t used;
                timeout = std::stod(s, &used);
                if (used != s.size()) throw std::invalid_argument("Invalid request timeout");
            } else if (!arg.starts_with('-') && artifact.empty())
                artifact = arg;
            else
                throw std::invalid_argument("Unknown option: " + arg);
        }
        if (artifact.empty() || host.empty() || port < 1 || port > 65535 || max_pending < 0 ||
            max_pending > 128 || !std::isfinite(timeout) || timeout <= 0 || timeout > 3600)
            throw std::invalid_argument(usage);
        if (kernels != "auto" && kernels != "optimized" && kernels != "reference")
            throw std::invalid_argument("--cpu-kernels must be auto, optimized or reference");
        if (threads < 1 || threads > 256 || codec_threads < 0 || codec_threads > 256)
            throw std::invalid_argument("Thread counts must be between 1 and 256");
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
        auto voice = [&](const std::string& label) -> const Voice& {
            for (const auto& v : voices)
                if (casefold(v.name) == casefold(label)) return v;
            throw std::invalid_argument("Unknown voice; see /v1/audio/voices");
        };
        if (!default_set) default_voice = voices.front().name;
        default_voice = voice(default_voice).name;
        if (name.empty()) name = artifact;
        Runtime runtime(root, max_pending, timeout, threads, codec_threads, kernels, device);
        const std::string device_name = device == "cpu" ? "cpu" : "cuda:" + device;
        // /metrics: requests in flight (queued ones included) and what was served -- what a
        // supervisor drains this server by.
        sinfer::serve::audio::Metrics metrics(name, {.characters = true});
        metrics.declare_endpoint("speech");
        httplib::Server server;
        configure(server, key, 64 * 1024, &metrics);
        metrics.serve(server);
        const size_t workers  = size_t(max_pending) + 4;
        server.new_task_queue = [workers] { return new httplib::ThreadPool(workers, workers); };
        server.Get("/health", [&](const auto&, auto& r) {
            bool ready = runtime.healthy();
            response(r, {{"status", ready ? "ok" : "unavailable"}, {"device", device_name}},
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
                auto content_type = q.get_header_value("Content-Type");
                if (content_type.substr(0, content_type.find(';')) != "application/json")
                    throw HttpError(415, "Content-Type must be application/json");
                auto body = json::parse(q.body);
                if (!body.is_object()) throw std::invalid_argument("Expected a JSON object");
                const std::set<std::string> fields = {"model",           "input", "voice",
                                                      "response_format", "speed", "seed"};
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
                const auto& selected = voice(body.value("voice", default_voice));
                const auto& input    = body.at("input").get_ref<const std::string&>();
                auto chunks          = tokenize(input);
                // Counted like the 4096-character limit. Set only once synthesis has succeeded, so an
                // error response never carries a billable count.
                const auto characters = input_characters(input);
                auto data             = runtime.synthesize(chunks, selected, int(seed), [&] {
                    return interrupted || (q.is_connection_alive && !q.is_connection_alive());
                });
                // 16-bit mono at 22,050 Hz after the 44-byte WAV header.
                const double seconds =
                    static_cast<double>(data.size() > 44 ? data.size() - 44 : 0) / (2.0 * 22050.0);
                if (format == "pcm") data.erase(0, 44);
                r.set_header("X-Usage-Characters", std::to_string(characters));
                r.set_header("X-Audio-Sample-Rate", "22050");
                r.set_header("Cache-Control", "no-store");
                r.set_content(std::move(data), format == "wav" ? "audio/wav" : "audio/pcm");
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
                    server.stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        });
        runtime.synthesize(tokenize("Bună."), voice(default_voice), 9,
                           [] { return interrupted != 0; });
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
