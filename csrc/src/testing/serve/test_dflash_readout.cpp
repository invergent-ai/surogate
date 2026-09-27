// Optional GPU regression: use a paired target/DFlash artifact. The ordinary and
// speculative engines load sequentially, so only one model occupies the device.
#include "api/engine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace {
sinfer::RequestOptions readout_options(std::size_t count) {
    sinfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.allow_prefix_reuse = false;
    options.execution.sampling.temperature = 0;
    options.execution.sampling.top_k = 0;
    options.execution.sampling.top_p = 1;
    options.execution.next_token_candidates.resize(count);
    std::iota(options.execution.next_token_candidates.begin(),
              options.execution.next_token_candidates.end(), 0);
    options.stop.include_model_defaults = false;
    return options;
}

void check_readout(const sinfer::GenerationResult& result, std::size_t count) {
    assert(result.generated_token_ids.size() == 1);
    assert(result.next_token_logits.size() == count);
    for (float logit : result.next_token_logits) { assert(std::isfinite(logit)); }
    assert(!result.speculative.enabled);
    assert(result.speculative.backend == sinfer::SpeculativeBackend::None);
    assert(result.speculative.draft_window == 0);
    assert(result.speculative.drafted_tokens == 0);
    assert(result.speculative.rounds == 0);
}

void compare(const std::vector<float>& reference, const std::vector<float>& actual,
             float tolerance) {
    assert(reference.size() == actual.size());
    float worst = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        assert(std::isfinite(actual[i]));
        worst = std::max(worst, std::abs(reference[i] - actual[i]));
    }
    if (worst > tolerance) {
        throw std::runtime_error("candidate logits differ by " + std::to_string(worst));
    }
}

sinfer::PromptInput image_prompt(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot read image fixture"); }
    sinfer::MessagePart image;
    image.kind = sinfer::MessagePartKind::Media;
    image.media.bytes.assign(std::istreambuf_iterator<char>(file), {});
    image.media.source_name = path;
    sinfer::ChatMessage message;
    message.parts.push_back(std::move(image));
    message.parts.push_back({.text = "Classify the image using true or false."});
    sinfer::PromptInput input;
    input.options.enable_thinking = false;
    input.messages.push_back(std::move(message));
    return input;
}
} // namespace

int main() {
    const char* artifact = std::getenv("SUROGATE_DFLASH_READOUT_ARTIFACT");
    if (!artifact) { return 77; }
    const char* image_path = std::getenv("SUROGATE_DFLASH_READOUT_IMAGE");
    sinfer::EngineOptions config;
    config.artifact_path = artifact;
    config.max_context = 2048;
    config.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(8192);
    config.max_concurrency = 4;
    config.prefill_chunk = 256;
    config.enable_vision = image_path != nullptr;
    config.gemma_image_tokens = image_path ? 1120 : 0;
    config.use_cuda_graph = !std::getenv("SUROGATE_DFLASH_READOUT_EAGER");

    // Both sides of the candidate-only head threshold, plus a full readout.
    const std::vector<std::size_t> counts{2, 16, 17, 256};
    const std::vector<std::size_t> lengths{37, 127, 257, 511};
    std::vector<std::vector<sinfer::TokenId>> prompts;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        prompts.emplace_back(lengths[i], static_cast<sinfer::TokenId>(100 + i));
    }
    std::vector<std::vector<float>> reference;
    std::vector<float> image_reference;
    {
        sinfer::Engine ordinary(config);
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            auto result = ordinary.generate(ordinary.prepare_tokens(prompts[i]), readout_options(counts[i]));
            check_readout(result, counts[i]);
            reference.push_back(std::move(result.next_token_logits));
        }
        if (image_path) {
            const auto result = ordinary.generate(ordinary.prepare(image_prompt(image_path)), readout_options(2));
            check_readout(result, 2);
            image_reference = result.next_token_logits;
        }
    }
    config.speculative.backend = sinfer::SpeculativeBackend::DFlash;
    config.speculative.draft_tokens = 3;
    sinfer::Engine engine(config);
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const auto result = engine.generate(engine.prepare_tokens(prompts[i]), readout_options(counts[i]));
        check_readout(result, counts[i]);
        compare(reference[i], result.next_token_logits, 0);
    }
    if (image_path) {
        const auto result = engine.generate(engine.prepare(image_prompt(image_path)), readout_options(2));
        check_readout(result, 2);
        compare(image_reference, result.next_token_logits, 0);
    }

    // First fill all lanes with readouts, then interleave three readouts with a
    // real generation. Repeat with images to exercise transient-buffer admission.
    for (bool images : {false, true}) {
        if (images && !image_path) { continue; }
        for (bool thinking : {false, true}) {
            std::vector<sinfer::PreparedPrompt> batch;
            std::vector<sinfer::RequestOptions> options;
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                const bool thought = thinking && i == 0;
                batch.push_back(images && !thought ? engine.prepare(image_prompt(image_path))
                                                   : engine.prepare_tokens(prompts[i]));
                auto request = readout_options(images ? 2 : counts[i]);
                if (thought) {
                    request.execution.next_token_candidates.clear();
                    request.execution.requested_output_tokens = 32;
                }
                options.push_back(std::move(request));
            }
            auto handles = engine.submit_batch(std::move(batch), std::move(options));
            for (std::size_t i = 0; i < handles.size(); ++i) {
                const auto result = handles[i].wait();
                if (thinking && i == 0) {
                    assert(result.generated_token_ids.size() == 32);
                    assert(result.speculative.enabled);
                    assert(result.speculative.backend == sinfer::SpeculativeBackend::DFlash);
                    assert(result.speculative.drafted_tokens > 0);
                    assert(result.speculative.rounds > 0);
                } else {
                    check_readout(result, images ? 2 : counts[i]);
                    // BF16 GEMMs use different widths in a pack. Match the existing
                    // candidate-readout tolerance for serial versus packed execution.
                    compare(images ? image_reference : reference[i], result.next_token_logits, .125f);
                }
            }
            assert(engine.healthy());
        }
    }
    std::cout << "DFlash readout parity, packed decisions and mixed generation passed"
              << (image_path ? " (including images)\n" : " (image fixture absent)\n");
}
