#include "cli/options.h"
#include "encoder/options.h"
#include "serve/serve_options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
template <class Parse>
auto parse(Parse parser, std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) { argv.push_back(arg.data()); }
    return parser(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <class Parse>
void rejects(Parse parser, std::vector<std::string> args) {
    try { (void)parse(parser, args); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid arguments accepted: " + args.back());
}
} // namespace

int main() {
    try {
        using namespace sinfer;
        const auto gen = cli::parse_options;
        const auto server = serve::parse_serve_options;
        const auto embed = encoder::parse_options;
        for (const auto flag : {"--response-store-max-records", "--response-store-max-mib"}) {
            rejects(server, {"serve", "model.sinfer", flag, "1"});
        }
        for (const auto flag : {"--cpu-moe-share", "--cpu-moe-prefill-share"}) {
            for (const auto value : {"junk", "0.5suffix", "nan", "inf", "-0.1", "1.1", ""}) {
                rejects(gen, {"generate", "model.sinfer", "--prompt", "x", flag, value});
                rejects(server, {"serve", "model.sinfer", flag, value});
            }
        }
        for (const auto value : {"-1", " -1", "18446744073709551616", "1x"}) {
            rejects(gen, {"generate", "model.sinfer", "--prompt", "x", "--seed", value});
            rejects(server, {"serve", "model.sinfer", "--seed", value});
        }
        for (const auto value : {"", "0,", "0,,1", "0,0", "-1"}) {
            rejects(gen, {"generate", "model.sinfer", "--prompt", "x", "--devices", value});
            rejects(server, {"serve", "model.sinfer", "--devices", value});
        }
        for (const auto flag : {"--gpu-layers", "-ngl", "--n-gpu-layers"}) {
            const auto g = parse(gen, {"generate", "model.sinfer", "--prompt", "x", flag, "0"});
            const auto s = parse(server, {"serve", "model.sinfer", flag, "0"});
            require(g.gpu_layers == EngineOptions::kGpuLayersNone && g.gpu_layers == s.gpu_layers,
                    "zero GPU layers did not offload every layer");
            require(parse(gen, {"generate", "model.sinfer", "--prompt", "x", flag, "all"}).gpu_layers == 0,
                    "all GPU layers did not restore default residency");
        }
        const auto offload_gen = parse(gen, {"generate", "model.sinfer", "--prompt", "x",
            "--offload-vision", "--offload-embeddings", "--offload-output-head"});
        const auto offload_server = parse(server, {"serve", "model.sinfer",
            "--offload-vision", "--offload-embeddings", "--offload-output-head"});
        require(offload_gen.offload_vision && offload_gen.offload_embeddings && offload_gen.offload_output_head &&
                    offload_server.offload_vision && offload_server.offload_embeddings && offload_server.offload_output_head,
                "component offload options lost");
        const auto vision_dflash = parse(gen,
                                         {"generate",
                                          "model.sinfer",
                                          "--prompt",
                                          "x",
                                          "--vision",
                                          "--spec",
                                          "dflash",
                                          "--draft-tokens",
                                          "15",
                                          "--kv-dtype",
                                          "fp8"});
        require(vision_dflash.enable_vision && vision_dflash.speculative.backend == SpeculativeBackend::DFlash &&
                    vision_dflash.speculative.draft_tokens == 15,
                "DFlash vision generation options lost");
        const auto g = parse(gen, {"generate", "model.sinfer", "--prompt", "--embed",
            "--host-expert-bank", "q4", "--host-expert-bank", "auto", "--host-moe-layers", "0",
            "--expert-slots", "0", "--cpu-moe-min-tokens", "0", "--cpu-moe-share", "auto",
            "--cpu-moe-prefill-share", "0", "--max-context", "256", "--kv-capacity", "512",
            "--prefill-chunk", "128", "--max-new", "7", "--device", "2",
            "--kv-dtype", "fp8_e4m3", "--spec", "mtp", "--draft-tokens", "2",
            "--spec-max-lanes", "0", "--lm-head-draft", "--raw-output", "--print-token-ids",
            "--prefill-warmup", "--no-thinking", "--no-cuda-graph", "--vision",
            "--stop-token-id", "0", "--stop-token-id", "9", "--stop", "done",
            "--reasoning-stop", "end", "--temperature", "0.7", "--top-p", "0.8",
            "--top-k", "0", "--min-p", "0.1", "--presence-penalty", "-1",
            "--frequency-penalty", "1", "--seed", "18446744073709551615"});
        require(g.prompt == "--embed" && g.host_expert_bank == EngineOptions::HostExpertBank::Auto,
                "literal input or repeated host-bank selection lost");
        require(g.host_moe_layers == 0 && g.expert_slots == 0 && g.cpu_moe_min_tokens == 0 &&
                    g.cpu_moe_share == -1 && g.cpu_moe_prefill_share == 0, "offload options lost");
        require(g.max_context == 256 && g.kv_capacity.explicit_tokens == 512 &&
                    g.prefill_chunk == 128 && g.max_new == 7 && g.device == 2, "resource options lost");
        require(g.kv_cache == KvCacheStorage::Fp8E4M3 && g.speculative.backend == SpeculativeBackend::Mtp &&
                    g.speculative.draft_tokens == 2 && g.speculative.proposal_head == ProposalHead::Optimized,
                "cache/speculation options lost");
        require(g.raw_output && g.print_token_ids && g.prefill_warmup && !g.enable_thinking &&
                    !g.use_cuda_graph && g.enable_vision, "generation switches lost");
        require(g.stop_token_ids == std::vector<TokenId>{0, 9} && g.stop_strings.size() == 2 &&
                    g.stop_strings[1].channel == OutputChannel::Reasoning, "repeatable stops lost");
        require(g.sampling.temperature == .7F && g.sampling.top_p == .8F && g.sampling.top_k == 0 &&
                    g.sampling.min_p == .1F && g.sampling.presence_penalty == -1 &&
                    g.sampling.frequency_penalty == 1 && g.sampling.seed == UINT64_MAX,
                "sampling overrides lost");

        for (const auto value : {"other=m.sinfer,kv-tokens=-1", "other=m.sinfer,kv-tokens=1junk",
                "other=m.sinfer,max-num-seqs=4294967297",
                "other=m.sinfer,max-model-len=0", "other=m.sinfer,draft-tokens=2",
                "other=m.sinfer,spec=mtp,draft-tokens=6", "other=m.sinfer,spec-max-lanes=all",
                "other=m.sinfer,kv-tokens=128,max-model-len=256"}) {
            rejects(server, {"serve", "model.sinfer", "--model", value});
        }
        rejects(server, {"serve", "model.sinfer", "--max-loras", "0", "--model", "other=m.sinfer,lora=a:b"});
        rejects(server, {"serve", "model.sinfer", "--served-model-name", "other", "--model", "other=m.sinfer"});
        const auto s = parse(server, {"serve", "model.sinfer", "--host-expert-bank", "w8",
            "--host-expert-bank", "auto", "--model", "other=m.sinfer,spec=mtp,draft-tokens=2"});
        require(s.host_expert_bank == EngineOptions::HostExpertBank::Auto &&
                    s.extra_models.front().speculative.draft_tokens == 2, "server override lost");
        require(parse(embed, {"embed", "--help"}).help_requested, "embedding help loads a model");
        const auto e = parse(embed, {"embed", "--host", "localhost", "--device", "cpu",
                                     "model.sinfer", "--port", "9000"});
        require(e.host == "localhost" && e.port == 9000 && e.device == "cpu", "embedding options lost");
        for (const auto value : {"0", "65536", "1junk", "-1"}) {
            rejects(embed, {"embed", "model.sinfer", "--port", value});
        }
        for (const auto value : {"-1", "1junk", "cpu1", "2147483648"}) {
            rejects(embed, {"embed", "model.sinfer", "--device", value});
        }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
