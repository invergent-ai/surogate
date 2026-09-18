// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace sinfer::tts {
struct Voice {
    std::string name;
    int id;
    float temperature;
    float cfg;
};

class Runtime {
public:
    Runtime(std::filesystem::path root, int max_pending, double timeout, int threads = 4,
            int codec_threads = 4, std::string kernels = "auto");
    ~Runtime();
    Runtime(const Runtime&)            = delete;
    Runtime& operator=(const Runtime&) = delete;
    bool healthy() const;
    std::string synthesize(const std::vector<std::vector<int32_t>>& chunks, const Voice& voice,
                           int seed, const std::function<bool()>& cancelled);
private:
    using Clock = std::chrono::steady_clock;
    void spawn();
    void stop() noexcept;
    void check(Clock::time_point deadline, const std::function<bool()>& cancelled);
    void wait_fd(int fd, short event, Clock::time_point deadline,
                 const std::function<bool()>& cancelled);
    std::filesystem::path root_, directory_;
    int max_pending_;
    int threads_, codec_threads_;
    std::string kernels_;
    double timeout_;
    std::atomic<int> pid_{-1}, pending_{0};
    int input_ = -1, output_ = -1;
    std::timed_mutex mutex_;
    uint64_t counter_ = 0;
    std::string buffered_;
};
} // namespace sinfer::tts
