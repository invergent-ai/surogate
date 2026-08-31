// An OpenAI-compatible /v1/embeddings server for the EmbeddingGemma encoder.
//
// Small on purpose. The engine's HTTP layer is built around a generation
// request -- sampling, streaming, stop policy, tool calls, a request lifetime
// that spans many rounds -- and an embedding request has none of that. It
// arrives, runs one forward, and returns a vector.
//
// `input` accepts token ids (`[int]` or `[[int]]`), which the OpenAI schema
// permits. It does not yet accept text: the engine's tokenizer requires
// ByteLevel BPE with a Qwen pretokenizer, and Gemma's is BPE with SentencePiece
// normalisation and byte fallback, so text input waits on a second tokenizer
// model. Taking ids also makes a comparison against another engine measure the
// model rather than two different tokenizers.
//
//   sinfer_embedding_server --artifact model.sinfer [--port 8413] [--device 0]

#include "core/device.h"
#include "encoder/gemma_embedding.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

/// `input` is one sequence of ids or a list of them; both are accepted, and a
/// bare list of integers is the single-sequence form.
std::vector<std::vector<std::int32_t>> parse_input(const json& input) {
    std::vector<std::vector<std::int32_t>> out;
    if (!input.is_array() || input.empty()) {
        throw std::invalid_argument("input must be a non-empty array of token ids");
    }
    if (input.front().is_number_integer()) {
        out.push_back(input.get<std::vector<std::int32_t>>());
        return out;
    }
    for (const json& element : input) {
        if (!element.is_array() || element.empty()) {
            throw std::invalid_argument("each input must be a non-empty array of token ids");
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
        std::string host = "127.0.0.1";
        int port         = 8413;
        int device_id    = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto next = [&](const char* what) -> std::string {
                if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + what); }
                return argv[i];
            };
            if (arg == "--artifact") {
                artifact = next("--artifact");
            } else if (arg == "--host") {
                host = next("--host");
            } else if (arg == "--port") {
                port = std::stoi(next("--port"));
            } else if (arg == "--device") {
                device_id = std::stoi(next("--device"));
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }
        if (artifact.empty()) { throw std::invalid_argument("--artifact is required"); }

        sinfer::DeviceContext device(device_id);
        auto model = sinfer::encoder::GemmaEmbedding::load(artifact, device);
        const auto& config = model.config();
        std::fprintf(stderr, "loaded %.0f MB, %d layers, %d hidden, max %d tokens\n",
                     static_cast<double>(model.weight_bytes()) / 1e6, config.layers, config.hidden,
                     config.max_tokens);

        // One model, one CUDA stream, one arena: requests are serialised. A
        // single 512-token sequence is already 512 columns of GEMM, so the card
        // is not idle while it runs; batching several sequences into one forward
        // is the next thing worth doing, not a lock to remove.
        std::mutex model_mutex;

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
                sequences = parse_input(body.at("input"));
            } catch (const std::exception& error) {
                response.status = 400;
                response.set_content(error_body(error.what(), "invalid_request_error").dump(),
                                     "application/json");
                return;
            }

            json data = json::array();
            std::size_t prompt_tokens = 0;
            try {
                const std::lock_guard<std::mutex> lock(model_mutex);
                for (std::size_t index = 0; index < sequences.size(); ++index) {
                    prompt_tokens += sequences[index].size();
                    const std::vector<float> vector = model.embed(sequences[index]);
                    data.push_back(json{{"object", "embedding"},
                                        {"index", index},
                                        {"embedding", vector}});
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
        if (!server.listen(host, port)) {
            throw std::runtime_error("could not bind " + host + ":" + std::to_string(port));
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "sinfer_embedding_server: %s\n", error.what());
        return 1;
    }
}
