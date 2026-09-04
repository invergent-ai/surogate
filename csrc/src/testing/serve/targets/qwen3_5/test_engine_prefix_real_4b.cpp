// Qwen3.5-4B engine prefix/rewrite-checkpoint test (surogate; modeled on the
// 27B sibling). Two ctest registrations run this binary in both
// checkpoint modes:
//   (default)  rewrite checkpoints are captured at prefill and replays must
//              restore them (RestoreResponseCheckpoint);
//   "defer"    SUROGATE_SERVE_DEFER_REWRITE_CHECKPOINT=1 — the capture is
//              skipped (PATCHES.md #24) and replays must still SUCCEED by
//              falling back to prefix recompute, never by erroring. This is
//              the gate for ever defaulting the deferral on.
// Artifact comes from SINFER_QWEN3_5_4B_WEIGHTS; absent resources SKIP (77).

#include "api/engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

sinfer::EngineOptions engine_options(const char* artifact) {
    sinfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 4096;
    options.kv_capacity   = sinfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk = 1024;
    return options;
}

sinfer::ChatMessage text_message(sinfer::ChatRole role, std::string text) {
    sinfer::ChatMessage message;
    message.role = role;
    message.parts.push_back(sinfer::MessagePart{
        .kind = sinfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

sinfer::PromptInput input_with_history(int completed_responses, bool preserve_thinking) {
    auto assistant_call = [](std::string reasoning, std::string id, std::string key) {
        sinfer::ChatMessage message = text_message(sinfer::ChatRole::Assistant, "");
        message.reasoning_content   = std::move(reasoning);
        message.tool_calls.push_back(sinfer::ToolCall{
            .id = std::move(id), .name = "lookup", .arguments_json = "{\"key\":\"" + key + "\"}"});
        return message;
    };
    sinfer::PromptInput input;
    input.messages.push_back(text_message(
        sinfer::ChatRole::User,
        "Use the lookup results to determine the deterministic checkpoint value."));
    if (completed_responses >= 1) {
        input.messages.push_back(
            assistant_call("The first lookup should be alpha.", "call_alpha", "alpha"));
        sinfer::ChatMessage tool =
            text_message(sinfer::ChatRole::Tool, "{\"value\":17,\"next\":\"beta\"}");
        tool.tool_call_id = "call_alpha";
        input.messages.push_back(std::move(tool));
    }
    input.options.preserve_thinking = preserve_thinking;
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
    return input;
}

sinfer::RequestOptions request_options(bool reuse) {
    sinfer::RequestOptions result;
    result.execution.requested_output_tokens = 4;
    result.execution.sampling.temperature    = 0.0F;
    result.execution.allow_prefix_reuse      = reuse;
    result.stop.include_model_defaults       = false;
    return result;
}

int exercise_checkpoints(sinfer::Engine& engine, bool deferred) {
    const sinfer::GenerationResult first =
        engine.generate(engine.prepare(input_with_history(0, true)), request_options(false));
    if (first.generated_token_ids.size() != 4 ||
        first.prefix_reuse_path != sinfer::PrefixReusePath::FullReset) {
        std::cerr << "cold request did not complete from a fresh lane\n";
        return 1;
    }

    // Re-render the same conversation without thinking preservation: the
    // frontier rewrite. With capture this restores the response checkpoint;
    // with deferral it must still complete via recompute.
    const sinfer::GenerationResult replay =
        engine.generate(engine.prepare(input_with_history(0, false)), request_options(true));
    if (replay.generated_token_ids.size() != 4) {
        std::cerr << "rewrite replay did not generate\n";
        return 1;
    }
    if (!deferred) {
        if (replay.prefix_reuse_path != sinfer::PrefixReusePath::RestoreResponseCheckpoint ||
            replay.reused_prompt_tokens == 0) {
            std::cerr << "capture mode: response checkpoint was not restored (path="
                      << static_cast<int>(replay.prefix_reuse_path) << ")\n";
            return 1;
        }
    } else {
        if (replay.prefix_reuse_path == sinfer::PrefixReusePath::RestoreResponseCheckpoint) {
            std::cerr << "defer mode: a checkpoint restore happened although capture was "
                         "deferred\n";
            return 1;
        }
    }

    // Continue the conversation with the first tool round completed.
    const sinfer::GenerationResult continued =
        engine.generate(engine.prepare(input_with_history(1, true)), request_options(true));
    if (continued.generated_token_ids.size() != 4) {
        std::cerr << "tool-continuation request did not generate\n";
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const bool deferred = argc > 1 && std::strcmp(argv[1], "defer") == 0;
    if (deferred) { setenv("SUROGATE_SERVE_DEFER_REWRITE_CHECKPOINT", "1", 1); }

    const char* artifact = std::getenv("SINFER_QWEN3_5_4B_WEIGHTS");
    if (artifact == nullptr || !std::ifstream(artifact).good()) {
        std::fprintf(stderr,
                     "SKIP: set SINFER_QWEN3_5_4B_WEIGHTS to a qwen3_5_4b .sinfer artifact\n");
        return 77;
    }
    try {
        sinfer::Engine engine(engine_options(artifact));
        if (engine.count_tokens(input_with_history(0, true)) == 0) {
            std::cerr << "tokenizer produced an empty prompt\n";
            return 1;
        }
        if (const int result = exercise_checkpoints(engine, deferred); result != 0) {
            return result;
        }
    } catch (const std::exception& error) {
        std::cerr << "qwen3_5_4b prefix test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK qwen3_5_4b engine prefix/" << (deferred ? "defer" : "capture") << '\n';
    return 0;
}
