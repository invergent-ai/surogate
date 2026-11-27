// The process's decision adapters (api/targets/inkling2/decision.h): runners built over the served inkling2 base when
// its frontend is made (package.cpp `bind_decision_adapters`), looked up by the decisions endpoint by name.

#include "targets/inkling2/impl/decision/registry.h"

#include "targets/inkling2/impl/decision/runner.h"

#include "core/device.h"
#include <api/targets/inkling2/decision.h>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

namespace {

// SUROGATE_SERVE_INKLING_DECISION_CAPTURE=<dir> (head training / parity work; off by default): every question runs
// with every pass traced, and the hidden rows the heads read are written to <dir>/c<pid>-<n>.cap:
//   "S15C", u32 version (1), u32 length of a JSON header, the header, then per pass the FP32 blocks
//   core [1 + K, D], exit [1 + K, D], pooled [pooled_rows, D] (row-major, little endian).
// The header holds the question's token ids and marker positions and each pass's readouts. The answer returned is the
// halting policy's, as without the capture; a traced question costs every pass.
std::string write_capture(const std::string& dir, const Question& q, const Answer& a) {
    static std::atomic<std::uint64_t> counter{0};
    const std::string name = "c" + std::to_string(static_cast<long>(::getpid())) + "-" + std::to_string(counter++) + ".cap";
    nlohmann::json h;
    h["ids"]        = q.ids;
    h["decide_idx"] = q.decide_idx;
    h["option_idx"] = q.option_idx;
    h["options"]    = q.option_idx.size();
    h["loops_used"] = a.loops_used;
    h["passes"]     = nlohmann::json::array();
    std::int64_t width = 0;
    for (const auto& p : a.passes) {
        nlohmann::json j;
        j["gate_logits"] = p.gate_logits;
        j["halt"]        = p.halt ? nlohmann::json(*p.halt) : nlohmann::json();
        j["max_halt"]    = p.max_halt;
        j["drift"]       = p.drift;
        j["exit_logits"] = p.exit_logits;
        j["escalate"]    = p.escalate ? nlohmann::json(*p.escalate) : nlohmann::json();
        j["core_floats"]   = p.cap_core.size();
        j["exit_floats"]   = p.cap_exit.size();
        j["pooled_rows"]   = p.cap_pooled_rows;
        j["pooled_floats"] = p.cap_pooled.size();
        if (!p.cap_core.empty()) { width = static_cast<std::int64_t>(p.cap_core.size() / (q.option_idx.size() + 1)); }
        h["passes"].push_back(std::move(j));
    }
    h["hidden"] = width;
    const std::string head = h.dump();
    const std::string tmp  = dir + "/" + name + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) { throw std::runtime_error("decision capture: cannot write " + tmp); }
        const std::uint32_t version = 1, len = static_cast<std::uint32_t>(head.size());
        f.write("S15C", 4);
        f.write(reinterpret_cast<const char*>(&version), 4);
        f.write(reinterpret_cast<const char*>(&len), 4);
        f.write(head.data(), static_cast<std::streamsize>(head.size()));
        for (const auto& p : a.passes) {
            for (const auto* v : {&p.cap_core, &p.cap_exit, &p.cap_pooled}) {
                f.write(reinterpret_cast<const char*>(v->data()), static_cast<std::streamsize>(v->size() * sizeof(float)));
            }
        }
        if (!f) { throw std::runtime_error("decision capture: short write to " + tmp); }
    }
    if (std::rename(tmp.c_str(), (dir + "/" + name).c_str()) != 0) { throw std::runtime_error("decision capture: rename failed"); }
    return name;
}

} // namespace

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
    const char* capture_dir = std::getenv("SUROGATE_SERVE_INKLING_DECISION_CAPTURE");
    const bool capture      = capture_dir != nullptr && capture_dir[0] != '\0';
    Answer a;
    try {
        a = e.runner->run(q, policy, e.stream, capture);
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
    if (capture) { r.capture = write_capture(capture_dir, question, a); }
    return r;
}

} // namespace sinfer::inkling2_decision
