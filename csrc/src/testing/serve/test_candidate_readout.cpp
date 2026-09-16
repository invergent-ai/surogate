#include "api/engine.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>

int main() {
    const char* artifact = std::getenv("SUROGATE_PCD_TEST_ARTIFACT");
    if (!artifact) return 77;
    sinfer::EngineOptions config;
    config.artifact_path = artifact;
    config.max_context = 1024;
    config.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(4096);
    config.max_concurrency = 4;
    config.prefill_chunk = 128;
    if (std::getenv("SUROGATE_PCD_TEST_MTP")) {
        config.speculative.backend = sinfer::SpeculativeBackend::Mtp;
        config.speculative.draft_tokens = 3;
    }
    if (std::getenv("SUROGATE_PCD_TEST_PIPELINE")) config.devices = {0, 1};
    sinfer::Engine engine(config);
    auto prefix = engine.prepare_text("Classify the text using true or false. ").token_ids();
    const auto fragment = engine.encode_fragment("Useful facts about classification. ");
    while (prefix.size() < 256) prefix.insert(prefix.end(), fragment.begin(), fragment.end());
    // Keep cold and cached recurrent prefill chunk boundaries identical.
    prefix.resize(256);
    sinfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature = 0;
    options.execution.sampling.top_k = 0;
    options.execution.sampling.top_p = 1;
    options.stop.include_model_defaults = false;
    std::vector<sinfer::TokenId> candidates(256);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::vector<std::vector<sinfer::TokenId>> prompts;
    std::vector<std::vector<float>> cold;
    for (int i = 0; i < 4; ++i) {
        auto ids = prefix;
        const auto tail = engine.encode_fragment("{\"field_" + std::to_string(i) + "\":");
        ids.insert(ids.end(), tail.begin(), tail.end());
        prompts.push_back(ids);
        auto opt = options;
        opt.execution.allow_prefix_reuse = false;
        opt.execution.next_token_candidates = candidates;
        cold.push_back(engine.generate(engine.prepare_tokens(ids), opt).next_token_logits);
        assert(cold.back().size() == 256);
    }
    auto warm = options;
    warm.execution.cache_prompt = true;
    engine.generate(engine.prepare_tokens(prefix), warm);
    std::vector<sinfer::GenerationHandle> handles;
    for (const auto& ids : prompts) {
        auto opt = options;
        opt.execution.next_token_candidates = candidates;
        handles.push_back(engine.submit(engine.prepare_tokens(ids), opt));
    }
    float worst = 0;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        const auto result = handles[i].wait();
        assert(result.reused_prompt_tokens >= prefix.size());
        assert(result.next_token_logits.size() == 256);
        for (std::size_t j = 0; j < 256; ++j) {
            const auto difference = std::abs(result.next_token_logits[j] - cold[i][j]);
            worst = std::max(worst, difference);
            assert(difference <= .125f);
        }
    }
    // Compare the readout with independent full-vocabulary log probabilities.
    // Their normalizer cancels; bias chooses a token but must not enter its score.
    float scores[2];
    for (int i = 0; i < 2; ++i) {
        auto opt = options;
        opt.execution.allow_prefix_reuse = false;
        opt.execution.top_logprobs = 0;
        opt.execution.sampling.logit_bias[100 + i] = 100;
        const auto result = engine.generate(engine.prepare_tokens(prompts[0]), opt);
        assert(result.generated_token_ids.at(0) == 100 + i);
        scores[i] = result.completion_logprobs.at(0).selected.logprob;
    }
    assert(std::abs((scores[0] - scores[1]) - (cold[0][100] - cold[0][101])) < 1e-4);
    std::cout << "256-candidate readout, shared prefix, concurrent/serial parity and scalar scoring passed; max logit delta=" << worst << '\n';
}
