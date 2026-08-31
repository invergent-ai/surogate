#pragma once

#include "api/types.h"

#include <filesystem>
#include <string>

namespace sinfer::product {

[[nodiscard]] PromptInput prompt_from_text(std::string text, bool enable_thinking);
[[nodiscard]] PromptInput prompt_from_messages(const std::filesystem::path& path,
                                               bool enable_thinking, bool vision_enabled);

} // namespace sinfer::product
