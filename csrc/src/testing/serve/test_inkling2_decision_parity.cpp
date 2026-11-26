// An Inkling-2 decision adapter run end to end on the engine (targets/inkling2/impl/decision/runner.cpp) over
// pre-tokenised questions, every pass traced: the per-pass halting scores, joint logits / probabilities and escalation
// probabilities, written as JSON lines for comparison with the System 1.5 reference trace (scratch tooling).
//
// usage: sinfer_inkling2_decision_parity BASE.sinfer ADAPTER_DIR QUESTIONS.jsonl OUT.jsonl [mode] [limit]
//   QUESTIONS.jsonl: {"id", "question", "ids": [...], "decide_idx", "opt_idx": [...]} per line.

#include "targets/inkling2/impl/decision/runner.h"
#include "targets/inkling2/impl/load/bindings.h"
#include "targets/inkling2/impl/model.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include <api/family/startup_features.h>
#include <api/targets/inkling2/package.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

using namespace sinfer;
using namespace sinfer::targets::inkling2;
using namespace sinfer::targets::inkling2::detail;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::printf("SKIP: usage %s BASE.sinfer ADAPTER_DIR QUESTIONS.jsonl OUT.jsonl [mode] [limit]\n", argv[0]);
        return 0;
    }
    const std::string mode = argc > 5 ? argv[5] : "balanced";
    const long limit       = argc > 6 ? std::atol(argv[6]) : -1;
    CUDA_CHECK(cudaSetDevice(0));
    auto context = std::make_unique<DeviceContext>(0);
    auto reader  = std::make_unique<artifact::Reader>(argv[1]);
    (void)Package::resolve_weights(reader->identity());
    artifact::Binder binder(*reader);
    load::BindOptions bind;
    bind.draft = false;
    EngineOptions options;
    ArtifactLoadPlan plan;
    plan.bindings.plan     = load::bind_artifact(binder, bind);
    plan.bindings.geometry = visit_geometry(*reader);
    plan.bindings.features = family::startup_features(options);
    plan.materialization   = binder.finish();
    auto materialized      = artifact::materialize(*reader, plan.materialization, *context);
    auto data = std::make_unique<LoadedModelData>(std::move(plan.bindings), std::move(materialized));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("base loaded\n");

    decision::Runner runner(*data, argv[2], 16384);
    decision::Policy policy;
    policy.threshold = mode == "full" ? 1.01 : runner.mode_threshold(mode);
    policy.max_loops = runner.max_loops();
    std::printf("adapter loaded: mode %s threshold %.3f, max %d loops\n", mode.c_str(), policy.threshold,
                policy.max_loops);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::ifstream in(argv[3]);
    std::ofstream out(argv[4]);
    std::string line;
    long n = 0;
    double seconds = 0;
    while (std::getline(in, line) && (limit < 0 || n < limit)) {
        const nlohmann::json q = nlohmann::json::parse(line);
        decision::Question question;
        question.ids        = q.at("ids").get<std::vector<std::int32_t>>();
        question.decide_idx = q.at("decide_idx").get<int>();
        question.option_idx = q.at("opt_idx").get<std::vector<std::int32_t>>();
        const decision::Answer a = runner.run(question, policy, stream, /*trace=*/true);
        seconds += a.seconds;
        nlohmann::json passes = nlohmann::json::array();
        for (const auto& p : a.passes) {
            passes.push_back({{"halt", p.halt ? nlohmann::json(*p.halt) : nlohmann::json(nullptr)},
                              {"max_halt", p.max_halt},
                              {"drift", p.drift},
                              {"gate_logits", p.gate_logits},
                              {"exit_logits", p.exit_logits},
                              {"probs", p.probs},
                              {"escalate", p.escalate ? nlohmann::json(*p.escalate) : nlohmann::json(nullptr)}});
        }
        out << nlohmann::json{{"id", q.at("id")}, {"question", q.at("question")}, {"loops_used", a.loops_used},
                              {"probs", a.probs}, {"aux", a.aux}, {"passes", passes}, {"seconds", a.seconds}}
                   .dump()
            << "\n";
        out.flush();
        if (++n % 25 == 0) { std::printf("%ld questions, %.2f s each (traced)\n", n, seconds / n); }
    }
    std::printf("%ld questions, %.2f s each (traced)\n", n, n ? seconds / n : 0.0);
    return 0;
}
