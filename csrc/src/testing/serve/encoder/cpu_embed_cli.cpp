// The EmbeddingGemma encoder on host cores: text or ids in, one vector out.
//
//   sinfer_cpu_embed_cli --artifact model.sinfer --text "..." [--threads N] [--repeat N]

#include "encoder/cpu/cpu_gemma_embedding.h"

#include <charconv>
#include <chrono>
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
            std::from_chars(piece.data(), piece.data() + piece.size(), value);
            out.push_back(value);
        }
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact, text;
        std::vector<std::int32_t> tokens;
        bool have_text = false;
        int repeat     = 1;
        sinfer::encoder::cpu::ThreadPlan plan;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            const auto next = [&](const char* what) -> std::string_view {
                if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + what); }
                return argv[i];
            };
            if (arg == "--artifact") {
                artifact = std::string(next("--artifact"));
            } else if (arg == "--text") {
                text = std::string(next("--text"));
                have_text = true;
            } else if (arg == "--tokens") {
                tokens = parse_tokens(next("--tokens"));
            } else if (arg == "--threads") {
                plan.threads = std::stoi(std::string(next("--threads")));
            } else if (arg == "--repeat") {
                repeat = std::stoi(std::string(next("--repeat")));
            } else {
                throw std::invalid_argument("unknown argument: " + std::string(arg));
            }
        }
        if (artifact.empty()) { throw std::invalid_argument("--artifact is required"); }

        const auto began = std::chrono::steady_clock::now();
        auto model = sinfer::encoder::cpu::CpuGemmaEmbedding::load(artifact, plan);
        const double load_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        if (have_text) { tokens = model.tokenizer().encode(text); }
        if (tokens.empty()) { throw std::invalid_argument("one of --text or --tokens is required"); }
        std::fprintf(stderr, "cpu: %.1f GB decoded in %.1f s on %d threads; %zu tokens\n",
                     static_cast<double>(model.weight_bytes()) / 1e9, load_seconds,
                     model.threads(), tokens.size());

        std::vector<float> embedding;
        const auto ran = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i) { embedding = model.embed(tokens); }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - ran).count() / repeat;
        std::fprintf(stderr, "cpu: %.1f ms per forward\n", seconds * 1e3);

        std::printf("[");
        for (std::size_t i = 0; i < embedding.size(); ++i) {
            std::printf("%s%.6f", i == 0 ? "" : ",", embedding[i]);
        }
        std::printf("]\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "sinfer_cpu_embed_cli: %s\n", error.what());
        return 1;
    }
}
