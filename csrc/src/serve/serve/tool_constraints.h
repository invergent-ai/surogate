#pragma once
#include "serve/request.h"
#include "serve/output_parsers.h"
namespace sinfer::serve {
[[nodiscard]] std::string make_tool_constraint(const GenerationRequest& request, ToolCallFormat format, bool thinking);
[[nodiscard]] bool tool_arguments_match_schema(const ToolDefinition& tool, const std::string& arguments);
} // namespace sinfer::serve
