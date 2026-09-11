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

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const std::vector<ToolDefinition>& tools = {});

ParsedToolCallOutput parse_spark_tool_call_output(const std::string& text,
                                                  std::size_t max_tool_name_length,
                                                  const std::vector<ToolDefinition>& tools = {});

// Incrementally publishes text outside a possible tool-call suffix. At terminal time, a valid tool response discards the
// buffered tool region; malformed/non-tool output flushes it verbatim.
class ToolCallStreamFilter {
public:
    explicit ToolCallStreamFilter(bool json_tools = false) : json_tools_(json_tools) {}
    std::string feed(std::string_view text);
    std::string finish(bool is_tool_call_response);

    [[nodiscard]] std::size_t emitted_bytes() const noexcept { return emitted_bytes_; }

private:
    bool json_tools_ = false;
    std::string pending_;
    std::string tool_region_;
    std::size_t emitted_bytes_ = 0;
    bool saw_tool_marker_      = false;
    bool finished_             = false;
};

} // namespace sinfer::serve
