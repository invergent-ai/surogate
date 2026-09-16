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
    if (std::getenv("SUROGATE_PCD_TEST_DFLASH")) {
        config.speculative.backend = sinfer::SpeculativeBackend::DFlash;
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
        opt.execution.save_gpu_prefix = std::make_shared<sinfer::GpuPrefixKey>();
        cold.push_back(engine.generate(engine.prepare_tokens(ids), opt).next_token_logits);
        assert(cold.back().size() == 256);
    }
    auto warm = options;
    auto root = std::make_shared<sinfer::GpuPrefixKey>();
    root->state_slots = 5;
    warm.execution.save_gpu_prefix = root;
    engine.generate(engine.prepare_tokens(prefix), warm);
    std::vector<std::vector<float>> serial;
    for (const auto& ids : prompts) {
        auto opt = options;
        opt.execution.gpu_prefix = root;
        opt.execution.next_token_candidates = candidates;
        serial.push_back(engine.generate(engine.prepare_tokens(ids), opt).next_token_logits);
    }
    std::vector<sinfer::PreparedPrompt> batch;
    std::vector<sinfer::RequestOptions> batch_options;
    std::vector<std::shared_ptr<const sinfer::GpuPrefixKey>> branches;
    for (const auto& ids : prompts) {
        auto opt = options;
        opt.execution.next_token_candidates = candidates;
        opt.execution.gpu_prefix = root;
        branches.push_back(std::make_shared<sinfer::GpuPrefixKey>());
        opt.execution.save_gpu_prefix = branches.back();
        batch.push_back(engine.prepare_tokens(ids));
        batch_options.push_back(opt);
    }
    auto handles = engine.submit_batch(std::move(batch), std::move(batch_options));
    float worst = 0;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        const auto result = handles[i].wait();
        assert(result.reused_prompt_tokens == prefix.size());
        assert(result.next_token_logits.size() == 256);
        float serial_delta = 0, batch_delta = 0;
        for (std::size_t j = 0; j < 256; ++j) {
            const auto difference = std::abs(result.next_token_logits[j] - cold[i][j]);
            worst = std::max(worst, difference);
            serial_delta = std::max(serial_delta, std::abs(serial[i][j] - cold[i][j]));
            batch_delta = std::max(batch_delta, std::abs(serial[i][j] - result.next_token_logits[j]));
        }
        assert(serial_delta <= .125f);
        assert(batch_delta <= .125f);
    }
    // Four branch tails occupy four extra pages. The 256-token prefix is
    // resident once, even with all four saved branch frontiers still alive.
    assert(engine.runtime_stats().kv_pages_in_use == 8);
    handles.clear();
    auto invalid = options;
    invalid.execution.gpu_prefix = root;
    invalid.execution.next_token_candidates = candidates;
    auto wrong_prefix = prompts[0];
    wrong_prefix[0] = (wrong_prefix[0] + 1) % 256;
    bool refused = false;
    try { engine.generate(engine.prepare_tokens(wrong_prefix), invalid); }
    catch (const std::invalid_argument&) { refused = true; }
    assert(refused);
    invalid.execution.save_gpu_prefix = std::make_shared<sinfer::GpuPrefixKey>();
    refused = false;
    try { engine.generate(engine.prepare_tokens(prompts[0]), invalid); }
    catch (const sinfer::RequestError& error) { refused = error.kind() == sinfer::RequestErrorKind::Overloaded; }
    assert(refused);
    invalid.execution.gpu_prefix.reset();
    // Nested branches must resume from their immediate parent, including its
    // partially filled KV page and recurrent state, with the root handle gone.
    warm.execution.save_gpu_prefix.reset();
    root.reset();
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        auto ids = prompts[i];
        const auto tail = engine.encode_fragment("true, ");
        ids.insert(ids.end(), tail.begin(), tail.end());
        auto opt = options;
        opt.execution.gpu_prefix = branches[i];
        opt.execution.next_token_candidates = candidates;
        const auto nested = engine.generate(engine.prepare_tokens(ids), opt);
        assert(nested.reused_prompt_tokens == prompts[i].size());
        opt.execution.gpu_prefix.reset();
        opt.execution.allow_prefix_reuse = false;
        // Recompute the parent independently, preserving the same recurrent
        // chunk boundaries before appending the final continuation.
        auto reference = std::make_shared<sinfer::GpuPrefixKey>();
        opt.execution.save_gpu_prefix = reference;
        engine.generate(engine.prepare_tokens(prompts[i]), opt);
        opt.execution.save_gpu_prefix.reset();
        opt.execution.gpu_prefix = reference;
        const auto fresh = engine.generate(engine.prepare_tokens(ids), opt);
        for (std::size_t j = 0; j < candidates.size(); ++j)
            assert(std::abs(nested.next_token_logits[j] - fresh.next_token_logits[j]) <= .125f);
    }
    branches.clear();
    // Compare the readout with independent full-vocabulary log probabilities.
    // Their normalizer cancels; bias chooses a token but must not enter its score.
    float scores[2];
    for (int i = 0; i < 2; ++i) {
        auto opt = options;
        opt.execution.allow_prefix_reuse = false;
        opt.execution.save_gpu_prefix = std::make_shared<sinfer::GpuPrefixKey>();
        opt.execution.top_logprobs = 0;
        opt.execution.sampling.logit_bias[100 + i] = 100;
        const auto result = engine.generate(engine.prepare_tokens(prompts[0]), opt);
        assert(result.generated_token_ids.at(0) == 100 + i);
        scores[i] = result.completion_logprobs.at(0).selected.logprob;
    }
    assert(std::abs((scores[0] - scores[1]) - (cold[0][100] - cold[0][101])) < 1e-4);
    std::cout << "256-candidate readout, shared prefix, concurrent/serial parity and scalar scoring passed; max logit delta=" << worst << '\n';
}
