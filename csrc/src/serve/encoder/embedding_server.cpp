// An OpenAI-compatible /v1/embeddings server for the EmbeddingGemma encoder.
//
// Small on purpose. The engine's HTTP layer is built around a generation
// request -- sampling, streaming, stop policy, tool calls, a request lifetime
// that spans many rounds -- and an embedding request has none of that. It
// arrives, runs one forward, and returns a vector.
//
// `input` accepts text (`string` or `[string]`) and token ids (`[int]` or
// `[[int]]`), which is the whole of the OpenAI schema. Text is encoded by the
// tokenizer the artifact itself ships, so text and ids cannot disagree about
// what the weights were trained on. Ids remain useful for benchmarking against
// another engine, where they keep the measurement on the model rather than on
// two different tokenizers.
//
//   surogate-embed <model> [--host H] [--port 8413] [--device 0|cpu]
//
// The model is positional, as it is for the generative engine. Users reach this
// through `surogate serve --embed <model>`, which resolves the model spec and
// execs this binary.
//
// CPU serving wants OMP_WAIT_POLICY=ACTIVE in the environment. Everything —
// oneDNN and the encoder's own kernels — runs on one OpenMP team, so the spin
// covers the microsecond gaps between phases instead of starving a second team.
// It must be an environment variable because libgomp reads it before main.

#include "core/device.h"
#include "encoder/cpu/cpu_gemma_embedding.h"
#include "encoder/gemma_embedding.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using json = nlohmann::json;

/// All four `input` forms: one string, a list of strings, one list of ids, or a
/// list of those. A bare list of integers is the single-sequence id form.
std::vector<std::vector<std::int32_t>> parse_input(const json& input,
                                                   const sinfer::encoder::GemmaTokenizer& tokenizer) {
    std::vector<std::vector<std::int32_t>> out;
    if (input.is_string()) {
        out.push_back(tokenizer.encode(input.get<std::string>()));
        return out;
    }
    if (!input.is_array() || input.empty()) {
        throw std::invalid_argument("input must be a string, or a non-empty array");
    }
    if (input.front().is_number_integer()) {
        out.push_back(input.get<std::vector<std::int32_t>>());
        return out;
    }
    for (const json& element : input) {
        if (element.is_string()) {
            out.push_back(tokenizer.encode(element.get<std::string>()));
            continue;
        }
        if (!element.is_array() || element.empty()) {
            throw std::invalid_argument("each input must be a string or a non-empty array of ids");
        }
        out.push_back(element.get<std::vector<std::int32_t>>());
    }
    return out;
}

json error_body(const std::string& message, const char* type) {
    return json{{"error", {{"message", message}, {"type", type}, {"code", nullptr}}}};
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact;
        std::string host   = "127.0.0.1";
        std::string device = "0";
        int port           = 8413;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto next = [&](const char* what) -> std::string {
                if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + what); }
                return argv[i];
            };
            if (!arg.empty() && arg.front() != '-') {
                if (!artifact.empty()) {
                    throw std::invalid_argument("unexpected extra argument: " + arg);
                }
                artifact = arg;
            } else if (arg == "--host") {
                host = next("--host");
            } else if (arg == "--port") {
                port = std::stoi(next("--port"));
            } else if (arg == "--device") {
                device = next("--device");
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }
        if (artifact.empty()) { throw std::invalid_argument("a model is required"); }

        // One of the two encoders, behind the same three calls the handler uses.
        std::unique_ptr<sinfer::DeviceContext> gpu_device;
        std::unique_ptr<sinfer::encoder::GemmaEmbedding> gpu;
        std::unique_ptr<sinfer::encoder::cpu::CpuGemmaEmbedding> host_model;
        if (device == "cpu") {
            host_model = std::make_unique<sinfer::encoder::cpu::CpuGemmaEmbedding>(
                sinfer::encoder::cpu::CpuGemmaEmbedding::load(artifact));
            std::fprintf(stderr, "cpu: %.0f MB on %d threads, gemm backend %s\n",
                         static_cast<double>(host_model->weight_bytes()) / 1e6,
                         host_model->threads(), sinfer::encoder::cpu::gemm_backend_name());
        } else {
            gpu_device = std::make_unique<sinfer::DeviceContext>(std::stoi(device));
            gpu        = std::make_unique<sinfer::encoder::GemmaEmbedding>(
                sinfer::encoder::GemmaEmbedding::load(artifact, *gpu_device));
            std::fprintf(stderr, "gpu %s: %.0f MB\n", device.c_str(),
                         static_cast<double>(gpu->weight_bytes()) / 1e6);
        }
        const auto& tokenizer = host_model ? host_model->tokenizer() : gpu->tokenizer();
        const auto embed_batch =
            [&](const std::vector<std::vector<std::int32_t>>& sequences) {
                return host_model ? host_model->embed_batch(sequences)
                                  : gpu->embed_batch(sequences);
            };

        // Every forward runs on ONE dedicated thread, not on whichever httplib
        // worker carried the request. OpenMP keeps its team per master thread,
        // so a changing master means a fresh team per request — measured as the
        // difference between 69 ms in a CLI loop and over a second through the
        // server. The handler posts work here and waits for its result.
        std::mutex queue_mutex;
        std::condition_variable queue_wake;
        std::function<void()> pending;
        bool stopping = false;
        std::thread runner([&] {
            while (true) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_wake.wait(lock, [&] { return stopping || pending; });
                    if (stopping && !pending) { return; }
                    job = std::move(pending);
                    pending = nullptr;
                    // Wake posters blocked on the slot being occupied.
                    queue_wake.notify_all();
                }
                job();
            }
        });
        const auto run_on_model_thread = [&](std::function<void()> job) {
            std::mutex done_mutex;
            std::condition_variable done_wake;
            bool done = false;
            {
                // The slot holds one job. A second poster must wait for the
                // runner to take the first — overwriting it would drop that job
                // on the floor and leave its requester waiting on a result that
                // can never come, which is exactly how the first concurrent
                // benchmark hung one request for its full HTTP timeout.
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_wake.wait(lock, [&] { return !pending; });
                pending = [&] {
                    job();
                    std::lock_guard<std::mutex> done_lock(done_mutex);
                    done = true;
                    done_wake.notify_one();
                };
            }
            queue_wake.notify_all();
            std::unique_lock<std::mutex> lock(done_mutex);
            done_wake.wait(lock, [&] { return done; });
        };

        httplib::Server server;
        server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
            response.set_content(json{{"status", "ok"}}.dump(), "application/json");
        });

        server.Post("/v1/embeddings", [&](const httplib::Request& request,
                                          httplib::Response& response) {
            const auto began = std::chrono::steady_clock::now();
            std::vector<std::vector<std::int32_t>> sequences;
            std::string model_name = "embeddinggemma";
            try {
                const json body = json::parse(request.body);
                if (body.contains("model") && body["model"].is_string()) {
                    model_name = body["model"].get<std::string>();
                }
                sequences = parse_input(body.at("input"), tokenizer);
            } catch (const std::exception& error) {
                response.status = 400;
                response.set_content(error_body(error.what(), "invalid_request_error").dump(),
                                     "application/json");
                return;
            }

            json data = json::array();
            std::size_t prompt_tokens = 0;
            for (const std::vector<std::int32_t>& sequence : sequences) {
                prompt_tokens += sequence.size();
            }
            try {
                // One forward for the whole request, executed on the model
                // thread: OpenMP keeps its team per master thread, and running
                // the forward from whichever httplib worker carried the request
                // rebuilt the team every time — the difference between 69 ms in
                // a CLI loop and over a second through the server.
                std::vector<std::vector<float>> vectors;
                std::exception_ptr failure;
                run_on_model_thread([&] {
                    try {
                        vectors = embed_batch(sequences);
                    } catch (...) { failure = std::current_exception(); }
                });
                if (failure) { std::rethrow_exception(failure); }
                for (std::size_t index = 0; index < vectors.size(); ++index) {
                    data.push_back(json{{"object", "embedding"},
                                        {"index", index},
                                        {"embedding", vectors[index]}});
                }
            } catch (const std::exception& error) {
                response.status = 500;
                response.set_content(error_body(error.what(), "internal_error").dump(),
                                     "application/json");
                return;
            }

            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            response.set_content(
                json{{"object", "list"},
                     {"data", data},
                     {"model", model_name},
                     {"usage", {{"prompt_tokens", prompt_tokens},
                                {"total_tokens", prompt_tokens}}},
                     {"timings", {{"seconds", seconds}}}}
                    .dump(),
                "application/json");
        });

        std::fprintf(stderr, "listening on %s:%d\n", host.c_str(), port);
        const bool bound = server.listen(host, port);
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stopping = true;
        }
        queue_wake.notify_one();
        runner.join();
        if (!bound) {
            throw std::runtime_error("could not bind " + host + ":" + std::to_string(port));
        }
        return 0;
    } catch (const std::exception& error) {
        const char* program = argc > 0 ? argv[0] : "surogate-embed";
        if (const char* slash = std::strrchr(program, '/')) { program = slash + 1; }
        std::fprintf(stderr, "%s: %s\n", program, error.what());
        return 1;
    }
}
