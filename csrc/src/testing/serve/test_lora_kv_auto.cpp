// Opt-in real-model regression: an automatic KV capacity counts the adapter banks.
//
// Supply a prepared artifact through SUROGATE_LORA_KV_TEST_ARTIFACT and reserve one device
// with CUDA_VISIBLE_DEVICES; SUROGATE_LORA_KV_TEST_SPEC=mtp serves with that draft head,
// SUROGATE_LORA_KV_TEST_{SEQS,CONTEXT,RANK,SLOTS} override 43 lanes, 40,960 tokens, rank 16
// and one slot.
//
// An engine started with --enable-lora allocates every --max-loras slot at --max-lora-rank on
// every adaptable module: 2.7 GiB for one rank-16 slot on the experts of a 35B-A3B. Those banks
// were allocated after the automatic capacity had been sized against free memory, so the pool
// spent them a second time; without MTP the 1 GiB of headroom could not absorb them and the
// engine refused to start (the conversation checkpoints found no room). So:
//
//   - the engine with adapters starts at an automatic capacity;
//   - without a draft head its sizing read free memory with the banks already allocated: what
//     it saw after the weights is what the engine without adapters saw, less the banks, and its
//     pool is smaller by the pages those bytes would have bought, no more and no less;
//   - with MTP it is sized as before, against what the engine without adapters saw
//     (runtime::kv_sizing_free_bytes: the MTP graph allowance holds the banks).
//
// Each engine is built in a child process of its own: an engine destroyed in this process leaves
// a few hundred MiB behind (CUDA context state, library workspaces), which a second engine here
// would read as less free memory.
#include "api/engine.h"
#include "api/ops/lora_store.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

using namespace sinfer;

namespace {

bool failed = false;
void check(bool ok, const std::string& what) {
    std::cout << (ok ? "ok    " : "FAILED") << "  " << what << std::endl;
    failed |= !ok;
}

std::uint32_t env_or(const char* name, std::uint32_t fallback) {
    const char* raw = std::getenv(name);
    return raw != nullptr && *raw != '\0' ? static_cast<std::uint32_t>(std::stoul(raw)) : fallback;
}

std::string mib(std::size_t bytes) { return std::to_string(bytes >> 20) + " MiB"; }

struct Sized {
    bool started                    = false;
    unsigned long long kv           = 0;
    unsigned long long groups       = 0;
    unsigned long long max_groups   = 0;
    unsigned long long stride       = 0;
    unsigned long long after_weights = 0;
    unsigned long long after_startup = 0;
    unsigned long long banks        = 0;
    unsigned long long minimum      = 0;
};

EngineOptions options_for(bool adapters) {
    EngineOptions options;
    options.artifact_path   = std::getenv("SUROGATE_LORA_KV_TEST_ARTIFACT");
    options.max_context     = env_or("SUROGATE_LORA_KV_TEST_CONTEXT", 40960);
    options.max_concurrency = env_or("SUROGATE_LORA_KV_TEST_SEQS", 43);
    options.prefill_chunk   = 4096;
    options.kv_cache        = KvCacheStorage::Fp8E4M3;
    options.kv_capacity     = KvCapacityPolicy::automatic();
    if (const char* spec = std::getenv("SUROGATE_LORA_KV_TEST_SPEC");
        spec != nullptr && std::string_view(spec) == "mtp") {
        options.speculative.backend      = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 2;
        options.speculative.max_lanes    = kSpeculateAtAnyWidth;
    }
    if (adapters) {
        options.lora_enable   = true;
        options.lora_slots    = env_or("SUROGATE_LORA_KV_TEST_SLOTS", 1);
        options.lora_max_rank = env_or("SUROGATE_LORA_KV_TEST_RANK", 16);
    }
    return options;
}

// The child: one engine, its sizing on one line.
int child(bool adapters) {
    try {
        Engine engine(options_for(adapters));
        const MemorySummary m = engine.memory_summary();
        std::size_t banks     = 0;
        for (int device : engine.lora_stores().devices()) {
            banks += engine.lora_stores().peek(device)->device_bytes();
        }
        std::printf("SIZED %u %u %u %zu %zu %zu %zu %zu\n", m.kv_capacity, m.kv_capacity_page_groups,
                    m.kv_capacity_max_page_groups, m.kv_capacity_increment_bytes,
                    m.available_after_weights_bytes, m.available_after_startup_bytes, banks,
                    m.minimum_runtime_reservation_bytes);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& error) {
        std::printf("REFUSED %s\n", error.what());
        std::fflush(stdout);
        return 1;
    }
}

Sized run_child(const char* self, bool adapters, std::string& refusal) {
    const std::string command = std::string(self) + (adapters ? " --child-adapters" : " --child-plain") + " 2>/dev/null";
    Sized out;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) { refusal = "popen failed"; return out; }
    char line[4096];
    while (std::fgets(line, sizeof line, pipe) != nullptr) {
        const std::string_view view(line);
        if (view.starts_with("SIZED ")) {
            out.started = std::sscanf(line, "SIZED %llu %llu %llu %llu %llu %llu %llu %llu", &out.kv,
                                      &out.groups, &out.max_groups, &out.stride, &out.after_weights,
                                      &out.after_startup, &out.banks, &out.minimum) == 8;
        } else if (view.starts_with("REFUSED ")) {
            refusal = std::string(view.substr(8));
        }
    }
    (void)pclose(pipe);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (std::getenv("SUROGATE_LORA_KV_TEST_ARTIFACT") == nullptr) { return 77; }
    if (argc > 1 && std::string_view(argv[1]) == "--child-plain") { return child(false); }
    if (argc > 1 && std::string_view(argv[1]) == "--child-adapters") { return child(true); }

    std::string refusal;
    const Sized plain = run_child(argv[0], false, refusal);
    if (!plain.started) {
        std::cout << "FAILED  the engine without adapters did not start: " << refusal << std::endl;
        return 1;
    }
    std::cout << "without adapters: KV " << plain.kv << " tokens, " << mib(plain.after_weights)
              << " after weights, " << mib(plain.after_startup) << " after startup" << std::endl;
    const Sized adapted = run_child(argv[0], true, refusal);
    check(adapted.started, "engine with adapters starts at an automatic capacity" +
                               (adapted.started ? std::string() : ": " + refusal));
    if (!adapted.started) { return 1; }
    std::cout << "with adapters: banks " << mib(adapted.banks) << ", KV " << adapted.kv << " tokens, "
              << mib(adapted.after_weights) << " after weights, " << mib(adapted.after_startup)
              << " after startup" << std::endl;
    check(adapted.banks > 0, "the adapter banks are allocated");

    // Free memory after the weights is a driver reading: two loads of the same artifact on an
    // idle card agree to the allocator's granularity, a few MiB.
    constexpr std::size_t kTolerance = std::size_t{64} << 20;
    const bool counted     = options_for(false).speculative.backend == SpeculativeBackend::None;
    const std::size_t seen = adapted.after_weights;
    const std::size_t expected =
        counted ? (plain.after_weights > adapted.banks ? plain.after_weights - adapted.banks : 0)
                : plain.after_weights;
    const std::size_t distance = seen > expected ? seen - expected : expected - seen;
    check(distance <= kTolerance,
          std::string(counted ? "the sizing read free memory after the banks: "
                              : "with a draft head the sizing read free memory as before: ") +
              mib(seen) + " against " + mib(expected) +
              (counted ? " (without adapters, less the banks)" : " (without adapters)"));

    // The pool gives up what the banks cost and nothing else: the page groups the difference in
    // what each engine saw would have bought, plus what the armed plan's larger minimum (graphs
    // for both round flavors, adapter scratch) takes, within one group either way (and the
    // tolerance).
    if (counted && adapted.stride != 0 && plain.groups < plain.max_groups) {
        const std::size_t lost   = plain.groups > adapted.groups ? plain.groups - adapted.groups : 0;
        const std::size_t gap    = (plain.after_weights > seen ? plain.after_weights - seen : 0) +
                                (adapted.minimum > plain.minimum ? adapted.minimum - plain.minimum : 0);
        const std::size_t want   = gap / adapted.stride;
        const std::size_t slack  = 1 + kTolerance / adapted.stride;
        check(lost + slack >= want && lost <= want + slack,
              "the pool shrank by " + std::to_string(lost) + " page groups for " + mib(gap) +
                  " of banks and minimum (" + std::to_string(want) + " expected)");
    }
    if (failed) {
        std::cout << "adapter KV sizing: FAILED\n";
        return 1;
    }
    std::cout << (counted ? "adapter KV sizing: the automatic capacity counts the adapter banks\n"
                          : "adapter KV sizing: with a draft head the automatic capacity is sized as before\n");
}
