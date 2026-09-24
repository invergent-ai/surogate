#pragma once

#include <sys/stat.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

namespace sinfer::serve {

/// Longest key file read; anything larger is refused rather than read into memory.
inline constexpr std::size_t kMaxApiKeyFileBytes = 4096;

/// The API key held in `path` (`--api-key-file`), so the key never has to appear on a command
/// line, where /proc/<pid>/cmdline shows it to every local user. The file holds the key on one
/// line of visible ASCII; a leading UTF-8 byte-order mark and surrounding whitespace, including a
/// trailing newline, are ignored. Throws std::invalid_argument, never with the key in the
/// message, when the file cannot be read, is a directory, is larger than 4 KiB, holds no key, or
/// holds anything but one key on one line. A file other users can read is used, with a warning.
inline std::string read_api_key_file(const std::string& path) {
    struct stat info {};
    if (::stat(path.c_str(), &info) != 0) {
        throw std::invalid_argument("cannot read --api-key-file '" + path + "': " +
                                    std::strerror(errno));
    }
    if (S_ISDIR(info.st_mode)) {
        throw std::invalid_argument("--api-key-file '" + path + "' is a directory");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::invalid_argument("cannot read --api-key-file '" + path + "': " +
                                    std::strerror(errno));
    }
    std::string text(kMaxApiKeyFileBytes + 1, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(in.gcount()));
    if (text.size() > kMaxApiKeyFileBytes) {
        throw std::invalid_argument("--api-key-file '" + path + "' is larger than " +
                                    std::to_string(kMaxApiKeyFileBytes) + " bytes");
    }
    if (text.rfind("\xEF\xBB\xBF", 0) == 0) { text.erase(0, 3); }
    const auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    std::size_t first = 0, last = text.size();
    while (first < last && blank(text[first])) { ++first; }
    while (last > first && blank(text[last - 1])) { --last; }
    if (first == last) { throw std::invalid_argument("--api-key-file '" + path + "' holds no key"); }
    for (std::size_t i = first; i < last; ++i) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (c < 0x21 || c > 0x7e) {
            throw std::invalid_argument("--api-key-file '" + path +
                                        "' must hold one key of visible ASCII characters on one line");
        }
    }
    if ((info.st_mode & (S_IROTH | S_IWOTH)) != 0) {
        std::fprintf(stderr,
                     "warning: --api-key-file '%s' is readable by other users (mode %03o); "
                     "chmod 600 keeps the key to its owner\n",
                     path.c_str(), static_cast<unsigned>(info.st_mode & 0777));
    }
    return text.substr(first, last - first);
}

/// The key a server authenticates with, from `--api-key KEY` or `--api-key-file PATH` as given on
/// its command line (nullopt: not given; a repeated flag keeps its last value). Giving both is
/// refused, and a key file is always read, so an empty path is an error rather than a server
/// without authentication. Neither given: no key.
inline std::string resolve_api_key(const std::optional<std::string>& key,
                                   const std::optional<std::string>& key_file) {
    if (key.has_value() && key_file.has_value()) {
        throw std::invalid_argument("give --api-key or --api-key-file, not both");
    }
    if (key_file.has_value()) { return read_api_key_file(*key_file); }
    return key.value_or("");
}

} // namespace sinfer::serve
