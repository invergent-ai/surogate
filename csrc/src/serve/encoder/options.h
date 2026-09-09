#pragma once

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sinfer::encoder {

struct Options {
    bool help_requested = false;
    std::string artifact;
    std::string host = "127.0.0.1";
    std::string device = "0";
    int port = 8413;
    int device_index = 0;
};

inline int parse_integer(std::string_view value, const char* flag, int minimum, int maximum) {
    int parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed < minimum || parsed > maximum) {
        throw std::invalid_argument(std::string(flag) + " must be in [" +
                                    std::to_string(minimum) + "," + std::to_string(maximum) + "]");
    }
    return parsed;
}

inline Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto next = [&]() -> std::string {
            if (++i >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            return argv[i];
        };
        if (arg == "--help" || arg == "-h") {
            options.help_requested = true;
            return options;
        } else if (arg == "--host") {
            options.host = next();
        } else if (arg == "--port") {
            options.port = parse_integer(next(), "--port", 1, 65535);
        } else if (arg == "--device") {
            options.device = next();
            if (options.device != "cpu") {
                options.device_index = parse_integer(options.device, "--device", 0, 2147483647);
            }
        } else if (!arg.empty() && arg.front() != '-' && options.artifact.empty()) {
            options.artifact = arg;
        } else {
            throw std::invalid_argument("unexpected argument: " + arg);
        }
    }
    if (options.artifact.empty()) { throw std::invalid_argument("a model is required"); }
    if (options.host.empty()) { throw std::invalid_argument("--host must not be empty"); }
    return options;
}

inline std::string usage_text(const char* program) {
    return std::string("usage: ") + program +
           " <model.sinfer> [--host H] [--port N] [--device N|cpu]\n"
           "Serve /v1/embeddings. Defaults: host 127.0.0.1, port 8413, device 0.\n";
}

} // namespace sinfer::encoder
