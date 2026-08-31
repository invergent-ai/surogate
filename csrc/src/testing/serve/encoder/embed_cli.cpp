// Run one sequence through the EmbeddingGemma encoder and print the vector.
//
// Tokenisation stays in Python for now: this takes token ids so the numerical
// path can be compared against the reference before a C++ tokenizer exists to
// disagree with it.
//
//   sinfer_embed_cli --artifact model.sinfer --tokens 2,105,4368,...

#include "core/device.h"
#include "encoder/gemma_embedding.h"

#include <charconv>
#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::int32_t> parse_tokens(std::string_view raw) {
    std::vector<std::int32_t> out;
    std::size_t begin = 0;
    while (begin <= raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string_view piece =
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin);
        if (!piece.empty()) {
            std::int32_t value = 0;
            const auto result  = std::from_chars(piece.data(), piece.data() + piece.size(), value);
            if (result.ec != std::errc()) {
                throw std::invalid_argument("not a token id: " + std::string(piece));
            }
            out.push_back(value);
        }
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (out.empty()) { throw std::invalid_argument("--tokens is empty"); }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact;
        std::vector<std::int32_t> tokens;
        int repeat = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            const auto next = [&](const char* what) -> std::string_view {
                if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + what); }
                return argv[i];
            };
            if (arg == "--artifact") {
                artifact = std::string(next("--artifact"));
            } else if (arg == "--tokens") {
                tokens = parse_tokens(next("--tokens"));
            } else if (arg == "--repeat") {
                repeat = std::stoi(std::string(next("--repeat")));
            } else {
                throw std::invalid_argument("unknown argument: " + std::string(arg));
            }
        }
        if (artifact.empty()) { throw std::invalid_argument("--artifact is required"); }
        if (tokens.empty()) { throw std::invalid_argument("--tokens is required"); }

        sinfer::DeviceContext device(0);
        auto model = sinfer::encoder::GemmaEmbedding::load(artifact, device);
        std::fprintf(stderr, "loaded %.0f MB of weights; %zu tokens\n",
                     static_cast<double>(model.weight_bytes()) / 1e6, tokens.size());

        std::vector<float> embedding;
        for (int i = 0; i < repeat; ++i) { embedding = model.embed(tokens); }

        std::printf("[");
        for (std::size_t i = 0; i < embedding.size(); ++i) {
            std::printf("%s%.6f", i == 0 ? "" : ",", embedding[i]);
        }
        std::printf("]\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "sinfer_embed_cli: %s\n", error.what());
        return 1;
    }
}
