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

#include "api/ops/batch_invariant.h"
#include "core/device.h"
#include "encoder/options.h"
#include "encoder/embedding_request.h"
#include "encoder/cpu/cpu_text_embedding.h"
#include "encoder/text_embedding.h"
#include "serve/http_socket.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using json = nlohmann::json;

json error_body(const std::string& message, const char* type) {
    return json{{"error", {{"message", message}, {"type", type}, {"code", nullptr}}}};
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = sinfer::encoder::parse_options(argc, argv);
        if (options.help_requested) {
            std::fputs(sinfer::encoder::usage_text(argv[0]).c_str(), stdout);
            return 0;
        }
        const auto& artifact = options.artifact;
        const auto& host = options.host;
        const auto& device = options.device;
        const int port = options.port;
        // Before the model loads: the GPU encoder decides at load whether to make the BF16
        // copies its fastest route needs, and that route is the one that is not invariant.
        sinfer::ops::set_batch_invariant(options.batch_invariant);
        if (options.batch_invariant) { std::fputs("batch-invariant numerics enabled\n", stderr); }

        // One of the two encoders, behind the same three calls the handler uses.
        std::unique_ptr<sinfer::DeviceContext> gpu_device;
        std::unique_ptr<sinfer::encoder::TextEmbedding> gpu;
        std::unique_ptr<sinfer::encoder::cpu::CpuTextEmbedding> host_model;
        if (device == "cpu") {
            host_model = std::make_unique<sinfer::encoder::cpu::CpuTextEmbedding>(
                sinfer::encoder::cpu::CpuTextEmbedding::load(artifact));
            std::fprintf(stderr, "cpu: %.0f MB on %d threads, gemm backend %s\n",
                         static_cast<double>(host_model->weight_bytes()) / 1e6,
                         host_model->threads(), sinfer::encoder::cpu::gemm_backend_name());
        } else {
            gpu_device = std::make_unique<sinfer::DeviceContext>(options.device_index);
            gpu        = std::make_unique<sinfer::encoder::TextEmbedding>(
                sinfer::encoder::TextEmbedding::load(artifact, *gpu_device));
            std::fprintf(stderr, "gpu %s: %.0f MB\n", device.c_str(),
                         static_cast<double>(gpu->weight_bytes()) / 1e6);
        }
        const auto& tokenizer = host_model ? host_model->tokenizer() : gpu->tokenizer();
        const auto& config = host_model ? host_model->config() : gpu->config();
        const auto& model_name = options.served_model_name;
        const auto embed_batch =
            [&](const std::vector<std::vector<std::int32_t>>& sequences) {
                return host_model ? host_model->embed_batch(sequences)
                                  : gpu->embed_batch(sequences);
            };

        // Every forward runs on ONE dedicated thread, not on whichever httplib
        // worker carried the request. OpenMP keeps its team per master thread,
        // so a changing master means a fresh team per request — measured as the
        // difference between 69 ms in a CLI loop and over a second through the
        // server. Handlers queue their request here and wait for its result.
        //
        // The thread also batches ACROSS requests: whatever queued while a forward
        // ran goes into the next one together. One short text is a forward of a
        // few dozen tokens whose ~500 launches cost the same as one of thousands,
        // so sixteen clients sending one text each took sixteen forwards where one
        // does. Nothing waits to gather a batch: an idle server runs a request the
        // moment it arrives, and requests merge only when they queued anyway.
        struct Job {
            const std::vector<std::vector<std::int32_t>>* sequences = nullptr;
            std::size_t tokens = 0;
            std::vector<std::vector<float>> vectors;
            std::exception_ptr failure;
            bool done = false;
        };
        // A round is at most one forward's worth of tokens, so a request never
        // waits behind a forward it is not in. The GPU encoder fits
        // max_batch_tokens in a forward; the CPU one packs up to max_tokens.
        const auto round_budget = static_cast<std::size_t>(
            host_model ? config.max_tokens : config.max_batch_tokens);
        std::mutex queue_mutex;
        std::condition_variable queue_wake;
        std::condition_variable done_wake;
        std::deque<Job*> queue;
        bool stopping = false;
        const auto run_alone = [&](Job& job) {
            try {
                job.vectors = embed_batch(*job.sequences);
            } catch (...) { job.failure = std::current_exception(); }
        };
        const auto run_round = [&](const std::vector<Job*>& round) {
            if (round.size() == 1) {
                run_alone(*round.front());
                return;
            }
            std::vector<std::vector<std::int32_t>> merged;
            for (const Job* job : round) {
                merged.insert(merged.end(), job->sequences->begin(), job->sequences->end());
            }
            try {
                std::vector<std::vector<float>> vectors = embed_batch(merged);
                auto next = vectors.begin();
                for (Job* job : round) {
                    const auto count = static_cast<std::ptrdiff_t>(job->sequences->size());
                    job->vectors.assign(std::make_move_iterator(next),
                                        std::make_move_iterator(next + count));
                    next += count;
                }
            } catch (...) {
                // Run each request on its own so a failure stays with the
                // request that caused it rather than failing its neighbours.
                for (Job* job : round) {
                    job->vectors.clear();
                    run_alone(*job);
                }
            }
        };
        std::thread runner([&] {
            while (true) {
                std::vector<Job*> round;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_wake.wait(lock, [&] { return stopping || !queue.empty(); });
                    if (queue.empty()) { return; }
                    // First come, first served; a request larger than the budget
                    // still runs, alone.
                    std::size_t tokens = 0;
                    while (!queue.empty() &&
                           (round.empty() || tokens + queue.front()->tokens <= round_budget)) {
                        tokens += queue.front()->tokens;
                        round.push_back(queue.front());
                        queue.pop_front();
                    }
                }
                run_round(round);
                {
                    // A job lives on its handler's stack: once `done` is set
                    // under the lock, the handler may return and free it.
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    for (Job* job : round) { job->done = true; }
                }
                done_wake.notify_all();
            }
        });

        httplib::Server server;
        sinfer::serve::disable_nagle(server);
        server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
            response.set_content(json{{"status", "ok"}}.dump(), "application/json");
        });

        server.Get("/v1/models", [&](const httplib::Request&, httplib::Response& response) {
            response.set_content(json{{"object", "list"}, {"data", json::array({
                {{"id", model_name}, {"object", "model"}, {"created", 0}, {"owned_by", "surogate"}}
            })}}.dump(), "application/json");
        });

        server.Post("/v1/embeddings", [&](const httplib::Request& request,
                                          httplib::Response& response) {
            const auto began = std::chrono::steady_clock::now();
            sinfer::encoder::EmbeddingRequest parsed;
            try {
                parsed = sinfer::encoder::parse_embedding_request(
                    json::parse(request.body), model_name, config.vocab, config.max_tokens,
                    config.hidden, [&](const std::string& text) { return tokenizer.encode(text); });
            } catch (const sinfer::encoder::EmbeddingRequestError& error) {
                response.status = error.status;
                auto body = error_body(error.what(), "invalid_request_error");
                body["error"]["param"] = error.param;
                body["error"]["code"] = error.code;
                response.set_content(body.dump(), "application/json");
                return;
            } catch (const std::exception& error) {
                response.status = 400;
                response.set_content(error_body(error.what(), "invalid_request_error").dump(),
                                     "application/json");
                return;
            }
            const auto& sequences = parsed.sequences;

            json data = json::array();
            std::size_t prompt_tokens = 0;
            for (const std::vector<std::int32_t>& sequence : sequences) {
                prompt_tokens += sequence.size();
            }
            try {
                // Executed on the model thread, possibly in one forward with
                // other requests that queued alongside it.
                Job job;
                job.sequences = &sequences;
                job.tokens    = prompt_tokens;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    queue.push_back(&job);
                }
                queue_wake.notify_one();
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    done_wake.wait(lock, [&] { return job.done; });
                }
                if (job.failure) { std::rethrow_exception(job.failure); }
                std::vector<std::vector<float>>& vectors = job.vectors;
                for (std::size_t index = 0; index < vectors.size(); ++index) {
                    data.push_back(json{{"object", "embedding"},
                                        {"index", index},
                                        {"embedding", sinfer::encoder::encode_embedding(
                                            std::move(vectors[index]), parsed.dimensions, parsed.base64)}});
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
