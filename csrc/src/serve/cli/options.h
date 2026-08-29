#pragma once

#include "api/types.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t expert_slots = 0; // --expert-slots N
    EngineOptions::HostExpertBank host_expert_bank   = EngineOptions::HostExpertBank::Auto; // --host-expert-bank w8|q4 (default: q4 when the slot cache is on)
    float cpu_moe_share = 0.0F; // --cpu-moe-share F
    std::uint32_t cpu_moe_min_tokens = 0; // --cpu-moe-min-tokens N
    float cpu_moe_prefill_share      = -1.0F; // --cpu-moe-prefill-share F (default 0.5 with the split; 0 = off)
    // surogate vendor patch (PATCHES.md #13): 2048 measured +16% prefill at
// ~1.9k-token prompts on RTX 5090 (K=1024 GEMM tiles amortize better);
// activation workspace stays small at qwen3.5-0.8b widths.
std::uint32_t prefill_chunk  = 2048;
    int device                   = 0;
    std::vector<int> devices;    // --devices a,b,c

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    bool use_cuda_graph = true;

    // surogate vendor patch (PATCHES.md #20): run one discarded warmup
    // request first so measured runs exclude one-time lazy work (FP8 plane
    // derivation). Off by default; benches enable it.
    bool prefill_warmup = false;

    bool raw_output      = false;
    bool print_token_ids = false;
    bool enable_thinking = true;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy = false;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
