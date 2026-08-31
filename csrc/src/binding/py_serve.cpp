// _surogate_serve — nanobind module exposing the serve engine to Python.
//
// Deliberately a SEPARATE extension from _surogate: the serve engine is
// arch-gated (sm_120a today, the RTX ladder per PATCHES.md #26), its kernel
// payload is large, and serving must never couple to the training build's
// lifecycle. The module binds the high-level product API (api/engine.h):
// blocking generate() with an optional streaming callback, token counting,
// and load/memory summaries. Quant profile and checkpoint-deferral knobs
// remain the documented SUROGATE_SERVE_* environment variables.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "api/engine.h"
#include "product/prompt_input/prompt_input.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

const char* finish_reason_name(sinfer::FinishReason reason) {
    switch (reason) {
    case sinfer::FinishReason::None: return "none";
    case sinfer::FinishReason::OutputLimit: return "output_limit";
    case sinfer::FinishReason::ContextCapacity: return "context_capacity";
    case sinfer::FinishReason::StopToken: return "stop_token";
    case sinfer::FinishReason::StopString: return "stop_string";
    case sinfer::FinishReason::Cancelled: return "cancelled";
    }
    return "unknown";
}

// Streams deltas into a Python callable; the engine publishes from its own
// thread, so every call re-acquires the GIL.
class CallbackSink final : public sinfer::OutputSink {
public:
    explicit CallbackSink(nb::object callback) : callback_(std::move(callback)) {}

    void publish(sinfer::OutputDelta delta) override {
        nb::gil_scoped_acquire gil;
        callback_(delta.channel == sinfer::OutputChannel::Content ? "content" : "reasoning",
                  delta.text);
    }

private:
    nb::object callback_;
};

class PyEngine {
public:
    PyEngine(const std::string& artifact, int device, std::uint32_t max_context,
             std::optional<std::uint32_t> kv_capacity, std::uint32_t prefill_chunk,
             std::uint32_t max_concurrency, bool use_cuda_graph) {
        sinfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.device          = device;
        options.max_context     = max_context;
        options.kv_capacity     = kv_capacity.has_value()
                                      ? sinfer::KvCapacityPolicy::explicit_capacity(*kv_capacity)
                                      : sinfer::KvCapacityPolicy::automatic();
        options.prefill_chunk   = prefill_chunk;
        options.max_concurrency = max_concurrency;
        options.use_cuda_graph  = use_cuda_graph;
        nb::gil_scoped_release release;
        engine_.emplace(std::move(options));
    }

    std::uint32_t count_tokens(const std::string& prompt, bool enable_thinking) const {
        return engine().count_tokens(
            sinfer::product::prompt_from_text(prompt, enable_thinking));
    }

    nb::dict generate(const std::string& prompt, bool enable_thinking, std::uint32_t max_new,
                      bool greedy, std::optional<float> temperature, std::optional<float> top_p,
                      std::optional<std::int32_t> top_k, std::optional<float> min_p,
                      std::optional<std::uint64_t> seed, const std::vector<std::string>& stop,
                      nb::object on_delta) {
        sinfer::RequestOptions request;
        request.execution.requested_output_tokens = max_new;
        if (greedy) { request.execution.sampling.temperature = 0.0F; }
        if (temperature) { request.execution.sampling.temperature = *temperature; }
        if (top_p) { request.execution.sampling.top_p = *top_p; }
        if (top_k) { request.execution.sampling.top_k = *top_k; }
        if (min_p) { request.execution.sampling.min_p = *min_p; }
        if (seed) { request.execution.sampling.seed = *seed; }
        for (const std::string& text : stop) {
            request.stop.strings.push_back(sinfer::StopString{.text = text});
        }

        std::optional<CallbackSink> sink;
        if (!on_delta.is_none()) { sink.emplace(std::move(on_delta)); }

        sinfer::PromptInput input =
            sinfer::product::prompt_from_text(prompt, enable_thinking);
        sinfer::GenerationResult result = [&] {
            nb::gil_scoped_release release;
            return engine().generate(engine().prepare(std::move(input)), std::move(request),
                                     sink ? &*sink : nullptr);
        }();

        nb::dict out;
        out["content"]              = result.content;
        out["reasoning"]            = result.reasoning;
        out["token_ids"]            = result.generated_token_ids;
        out["finish_reason"]        = finish_reason_name(result.finish_reason);
        out["prompt_tokens"]        = result.prompt.prompt_tokens;
        out["reused_prompt_tokens"] = result.reused_prompt_tokens;
        nb::dict timings;
        timings["prepare_seconds"]     = result.timings.prepare_seconds;
        timings["first_token_seconds"] = result.timings.first_token_seconds;
        timings["prefill_seconds"]     = result.timings.prefill_seconds;
        timings["decode_seconds"]      = result.timings.decode_seconds;
        timings["total_seconds"]       = result.timings.total_seconds;
        out["timings"] = timings;
        return out;
    }

    nb::dict memory_summary() const {
        const sinfer::MemorySummary memory = engine().memory_summary();
        nb::dict out;
        out["weights_bytes"]   = memory.weights.used_bytes;
        out["sequence_bytes"]  = memory.sequence.used_bytes;
        out["workspace_bytes"] = memory.workspace.capacity_bytes;
        out["kv_capacity"]     = memory.kv_capacity;
        return out;
    }

private:
    sinfer::Engine& engine() const {
        if (!engine_.has_value()) { throw std::runtime_error("engine is not loaded"); }
        return const_cast<sinfer::Engine&>(*engine_);
    }

    mutable std::optional<sinfer::Engine> engine_;
};

} // namespace

NB_MODULE(_surogate_serve, m) {
    m.doc() = "surogate serve engine (RTX inference; see csrc/src/serve)";

    nb::class_<PyEngine>(m, "Engine")
        .def(nb::init<const std::string&, int, std::uint32_t, std::optional<std::uint32_t>,
                      std::uint32_t, std::uint32_t, bool>(),
             nb::arg("artifact"), nb::arg("device") = 0, nb::arg("max_context") = 4096,
             nb::arg("kv_capacity") = nb::none(), nb::arg("prefill_chunk") = 2048,
             nb::arg("max_concurrency") = 1, nb::arg("use_cuda_graph") = true)
        .def("count_tokens", &PyEngine::count_tokens, nb::arg("prompt"),
             nb::arg("enable_thinking") = true)
        .def("generate", &PyEngine::generate, nb::arg("prompt"),
             nb::arg("enable_thinking") = true, nb::arg("max_new") = 512,
             nb::arg("greedy") = false, nb::arg("temperature") = nb::none(),
             nb::arg("top_p") = nb::none(), nb::arg("top_k") = nb::none(),
             nb::arg("min_p") = nb::none(), nb::arg("seed") = nb::none(),
             nb::arg("stop") = std::vector<std::string>{}, nb::arg("on_delta") = nb::none())
        .def("memory_summary", &PyEngine::memory_summary);
}
