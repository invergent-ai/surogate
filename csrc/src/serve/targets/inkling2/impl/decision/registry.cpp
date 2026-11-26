// The process's decision adapters (api/targets/inkling2/decision.h): runners built over the served inkling2 base when
// its frontend is made (package.cpp `bind_decision_adapters`), looked up by the decisions endpoint by name.

#include "targets/inkling2/impl/decision/registry.h"

#include "targets/inkling2/impl/decision/runner.h"

#include "core/device.h"
#include <api/targets/inkling2/decision.h>

#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <stdexcept>

namespace sinfer::targets::inkling2::detail::decision {

namespace {

struct Entry {
    std::unique_ptr<Runner> runner;
    int device          = 0;
    cudaStream_t stream = nullptr;
};

std::mutex& mutex() {
    static std::mutex m;
    return m;
}

std::map<std::string, Entry, std::less<>>& table() {
    static std::map<std::string, Entry, std::less<>> t;
    return t;
}

Entry& find(std::string_view model) {
    auto& t      = table();
    const auto it = t.find(model);
    if (it == t.end()) { throw std::invalid_argument("no decision adapter named " + std::string(model)); }
    return it->second;
}

} // namespace

void install(const std::string& name, const LoadedModelData& base, const std::filesystem::path& dir,
             std::int32_t max_tokens) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    Entry e;
    e.runner = std::make_unique<Runner>(base, dir, max_tokens);
    e.device = device;
    CUDA_CHECK(cudaStreamCreateWithFlags(&e.stream, cudaStreamNonBlocking));
    std::lock_guard lock(mutex());
    if (table().count(name) != 0) { throw std::invalid_argument("decision adapter " + name + " is loaded twice"); }
    table().emplace(name, std::move(e));
}

} // namespace sinfer::targets::inkling2::detail::decision

namespace sinfer::inkling2_decision {

using sinfer::targets::inkling2::detail::decision::Answer;

bool available(std::string_view model) {
    std::lock_guard lock(targets::inkling2::detail::decision::mutex());
    return targets::inkling2::detail::decision::table().count(model) != 0;
}

std::vector<std::string> names() {
    std::lock_guard lock(targets::inkling2::detail::decision::mutex());
    std::vector<std::string> out;
    for (const auto& [name, e] : targets::inkling2::detail::decision::table()) { out.push_back(name); }
    return out;
}

Info info(std::string_view model) {
    std::lock_guard lock(targets::inkling2::detail::decision::mutex());
    const auto& e = targets::inkling2::detail::decision::find(model);
    Info out;
    out.max_loops        = e.runner->max_loops();
    out.max_tokens       = e.runner->max_tokens();
    out.max_state_tokens = e.runner->meta().value("max_state_tokens", 8192);
    out.special_tokens   = e.runner->special_tokens();
    for (const char* mode : {"fast", "balanced", "strict"}) {
        try {
            out.modes[mode] = e.runner->mode_threshold(mode);
        } catch (const std::exception&) {
        }
    }
    out.modes["full"] = 1.01;
    return out;
}

Result run(std::string_view model, const Question& question, double threshold, int max_loops) {
    // One question at a time per process: a runner's device state is a single question's.
    std::lock_guard lock(targets::inkling2::detail::decision::mutex());
    auto& e = targets::inkling2::detail::decision::find(model);
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    CUDA_CHECK(cudaSetDevice(e.device));
    targets::inkling2::detail::decision::Question q;
    q.ids        = question.ids;
    q.decide_idx = question.decide_idx;
    q.option_idx = question.option_idx;
    targets::inkling2::detail::decision::Policy policy;
    policy.threshold = threshold;
    policy.max_loops = max_loops;
    Answer a;
    try {
        a = e.runner->run(q, policy, e.stream);
    } catch (...) {
        cudaSetDevice(previous);
        throw;
    }
    CUDA_CHECK(cudaSetDevice(previous));
    Result r;
    r.loops_used    = a.loops_used;
    r.logits        = a.logits;
    r.temperature   = a.temperature;
    r.probabilities = a.probs;
    r.halt_score    = a.halt_score;
    r.escalate_prob = a.escalate_prob;
    r.state_drift   = a.state_drift;
    r.l1_pred       = a.l1_pred;
    r.aux           = a.aux;
    r.seconds       = a.seconds;
    return r;
}

} // namespace sinfer::inkling2_decision
