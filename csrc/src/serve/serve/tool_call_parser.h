#pragma once

#include "serve/request.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::serve {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

// Function/parameter XML (or JSON) inside `<tool_call>`, with the argument values converted as vLLM 0.31's
// qwen3_coder / qwen3_xml parser converts them (serve/vllm_tool_arguments.h). `constrained`: the call was generated
// under the tool grammar, so a value vLLM's reading would take out of the tool's schema keeps the schema's reading.
ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const std::vector<ToolDefinition>& tools = {},
                                                 bool constrained = false);

ParsedToolCallOutput parse_spark_tool_call_output(const std::string& text,
                                                  std::size_t max_tool_name_length,
                                                  const std::vector<ToolDefinition>& tools = {});

ParsedToolCallOutput parse_muse_tool_call_output(const std::string& text,
    std::size_t max_tool_name_length, const std::vector<ToolDefinition>& tools = {});

// Gemma 4: `<|tool_call>call:NAME{key:<|"|>text<|"|>,count:2}<tool_call|>`, its chat template's own
// argument notation, read back into JSON arguments.
ParsedToolCallOutput parse_gemma4_tool_call_output(const std::string& text,
    std::size_t max_tool_name_length, const std::vector<ToolDefinition>& tools = {});

// Incrementally publishes text outside a possible tool-call suffix. At terminal time, a valid tool response discards the
// buffered tool region; malformed/non-tool output flushes it verbatim.
class ToolCallStreamFilter {
public:
    explicit ToolCallStreamFilter(bool json_tools = false, bool muse_tools = false, bool gemma_tools = false)
        : json_tools_(json_tools), muse_tools_(muse_tools), gemma_tools_(gemma_tools) {}
    std::string feed(std::string_view text);
    std::string finish(bool is_tool_call_response);

    [[nodiscard]] std::size_t emitted_bytes() const noexcept { return emitted_bytes_; }

private:
    bool json_tools_ = false;
    bool muse_tools_ = false;
    bool gemma_tools_ = false;
    std::string pending_;
    std::string tool_region_;
    std::size_t emitted_bytes_ = 0;
    bool saw_tool_marker_      = false;
    bool finished_             = false;
};

} // namespace sinfer::serve
