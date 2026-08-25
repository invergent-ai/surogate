#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <string_view>

namespace ninfer::product {

enum class LoadProgressOutputMode : std::uint8_t {
    Interactive,
    Log,
};

struct LoadProgressRendererOptions {
    LoadProgressOutputMode mode                    = LoadProgressOutputMode::Log;
    std::chrono::milliseconds min_refresh_interval = std::chrono::seconds(10);
    std::function<std::string()> line_prefix;
};

[[nodiscard]] LoadProgressRendererOptions stderr_load_progress_options() noexcept;

// Owns product-side terminal rendering for Engine load progress. The renderer must
// outlive the Engine whose LoadProgress callback references it.
class LoadProgressRenderer {
public:
    LoadProgressRenderer(std::ostream& output, LoadProgressRendererOptions options);
    ~LoadProgressRenderer();

    LoadProgressRenderer(const LoadProgressRenderer&)            = delete;
    LoadProgressRenderer& operator=(const LoadProgressRenderer&) = delete;
    LoadProgressRenderer(LoadProgressRenderer&&)                 = delete;
    LoadProgressRenderer& operator=(LoadProgressRenderer&&)      = delete;

    [[nodiscard]] LoadProgress callback();

    // Ends an open interactive line. This is idempotent and is also called by
    // the destructor so an Engine construction failure cannot corrupt later output.
    void finish() noexcept;

private:
    using Clock = std::chrono::steady_clock;

    void update(std::string_view phase, std::uint64_t done, std::uint64_t total);
    void render(std::uint64_t done, std::uint64_t total, Clock::time_point now, bool final);

    std::ostream* output_ = nullptr;
    LoadProgressRendererOptions options_;
    bool has_phase_ = false;
    bool line_open_ = false;
    std::string phase_;
    std::uint64_t last_done_    = 0;
    std::uint64_t last_total_   = 0;
    std::size_t terminal_width_ = 0;
    Clock::time_point phase_started_;
    Clock::time_point last_rendered_;
};

} // namespace ninfer::product
