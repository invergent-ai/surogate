#pragma once

#include <chrono>
#include <string>
#include <string_view>

namespace ninfer::serve {

enum class ConsoleLogLevel {
    Info,
    Warning,
    Error,
};

[[nodiscard]] std::string format_console_log_prefix(std::chrono::system_clock::time_point timestamp,
                                                    ConsoleLogLevel level);

[[nodiscard]] std::string current_console_log_prefix(ConsoleLogLevel level);

void write_console_log(ConsoleLogLevel level, std::string_view message);

} // namespace ninfer::serve
