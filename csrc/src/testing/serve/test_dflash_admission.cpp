#include "api/family/round_state.h"
#include "core/device.h"
#include "targets/registry.h"

#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {
const void* ingress_source = nullptr;
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
}

// Observe the source used by the real eager copy or captured memcpy node.
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                                                  cudaMemcpyKind kind, cudaStream_t stream) {
    static const auto real = reinterpret_cast<decltype(&cudaMemcpyAsync)>(dlsym(RTLD_NEXT, "cudaMemcpyAsync"));
    if (kind == cudaMemcpyHostToDevice && bytes == sizeof(sinfer::family::DFlashDecodeIngress)) {
        ingress_source = src;
    }
    return real(dst, src, bytes, kind, stream);
}

int main() {
    const auto* artifact = std::getenv("SUROGATE_DFLASH_ADMISSION_ARTIFACT");
    if (!artifact) { return 77; }
    try {
        sinfer::DeviceContext device(0);
        sinfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context = 256;
        options.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(768);
        options.max_concurrency = 3;
        options.prefill_chunk = 128;
        options.elastic_kv = false;
        options.kv_cache = sinfer::KvCacheStorage::BFloat16;
        options.speculative.backend = sinfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = 3;
        options.use_cuda_graph = !std::getenv("SUROGATE_DFLASH_ADMISSION_EAGER");
        auto target = sinfer::targets::construct_target(options, device);
        auto& instance = *std::get<std::unique_ptr<sinfer::targets::Qwen3_5Instance>>(target.active);
        auto& program = *instance.program;
        const auto start = [&](std::uint32_t lane, bool deferred) {
            auto prompt = instance.loaded->frontend.prepare_tokens(std::vector<sinfer::TokenId>(32 + lane * 8, 100));
            sinfer::runtime::ResolvedExecutionOptions request;
            request.requested_output_tokens = 16;
            request.sampling.temperature = 0;
            auto base = program.plan_request_base(prompt, request);
            auto plan = program.plan_request_for_lane(lane, prompt, base);
            require(program.can_admit_lane(lane, plan), "admission test request must fit");
            const auto summary = plan.summary();
            instance.request_memory.activate(summary.transient_bytes, summary.transient_alignment);
            auto step = program.start_prefill_lane(lane, std::move(prompt), std::move(plan),
                                                   instance.request_memory.region(), deferred);
            if (deferred) {
                require(!step.complete && step.processed_prompt_tokens == 0, "admission ran a prefill chunk");
            } else {
                while (!step.complete) { step = program.advance_prefill_lane(lane); }
                program.resolve_prefill_lane(lane, false);
                instance.request_memory.deactivate();
            }
        };
        std::vector<sinfer::TokenId> baseline;
        for (bool admit : {false, true}) {
            start(0, false);
            const std::array<std::uint32_t, 1> lanes{0};
            const std::array<sinfer::runtime::RoundBudget, 1> budgets{{{15}}};
            const auto handle = program.launch_decode_round(lanes, budgets);
            require(ingress_source != nullptr, "no DFlash DMA source was observed");
            std::vector<std::byte> before(sizeof(sinfer::family::DFlashDecodeIngress));
            std::memcpy(before.data(), ingress_source, before.size());
            if (admit) {
                start(1, true);
                require(std::memcmp(before.data(), ingress_source, before.size()) == 0,
                        "admission rewrote a launched DFlash round's DMA source");
            }
            const auto round = program.consume_decode_round(handle);
            const auto count = static_cast<std::uint32_t>(round.row_counts[0]);
            const std::vector<sinfer::TokenId> tokens(round.tokens.begin(), round.tokens.begin() + count);
            if (!admit) { baseline = tokens; }
            else { require(tokens == baseline, "concurrent admission changed DFlash output"); }
            const std::array<std::uint32_t, 1> accepted{count};
            const std::array<std::uint8_t, 1> done{1}, cancelled{0};
            program.resolve_pending_batch(lanes, accepted, done, cancelled);
            program.evict_retained_lane(0);
            if (admit) {
                auto step = program.advance_prefill_lane(1);
                while (!step.complete) { step = program.advance_prefill_lane(1); }
                require(!step.round.tokens.empty(), "admitted DFlash prompt did not finish");
                program.resolve_prefill_lane(1, true);
                program.evict_retained_lane(1);
                instance.request_memory.deactivate();
            }
        }
        std::cout << "DFlash admission preserves launched DMA source and generation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
