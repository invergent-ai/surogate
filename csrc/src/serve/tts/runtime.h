// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::tts {
struct Voice {
    std::string name;
    int id;
    float temperature;
    float cfg;
};

/// The canonical 44-byte header of 16-bit mono PCM; `data_bytes` 0xFFFFFFFF for a stream of
/// unknown length.
std::string wav_header(uint32_t sample_rate, uint32_t data_bytes);

class Runtime {
public:
    /// A request's place at the worker: its queue slot, then the worker itself, until destroyed.
    /// Taken before a response starts, so a full queue or a queue timeout is still an HTTP error.
    /// It may be released on another thread than the one that took it (a stream hands its
    /// worker back as soon as synthesis ends, while the audio is still being delivered).
    class Lease {
    public:
        ~Lease();
        /// Hands the worker to the next request while this one keeps its place in the queue: a
        /// stream whose synthesis has ended, still being delivered, frees the worker but still
        /// counts against the queue until its response ends.
        void release_worker();
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;

    private:
        friend class Runtime;
        Lease() = default;
        Runtime* owner_ = nullptr;
        bool held_      = false; ///< the worker is this request's
        std::chrono::steady_clock::time_point deadline_{};
    };

    /// `device`: "cpu", or the CUDA device index the worker synthesizes on.
    Runtime(std::filesystem::path root, int max_pending, double timeout, int threads = 4,
            int codec_threads = 4, std::string kernels = "auto", std::string device = "cpu");
    ~Runtime();
    Runtime(const Runtime&)            = delete;
    Runtime& operator=(const Runtime&) = delete;
    bool healthy() const;
    /// Waits for a place at the worker: 429 when the queue is full, 504 past the deadline, 503
    /// when `cancelled` says the client has gone.
    std::unique_ptr<Lease> acquire(const std::function<bool()>& cancelled);
    /// Synthesizes on the leased worker, handing each PCM chunk (16-bit mono) to `on_pcm` as the
    /// runtime produces it; `on_pcm` returning false cancels. Returns the sample rate.
    int stream(Lease& lease, const std::vector<std::vector<int32_t>>& chunks, const Voice& voice,
               int seed, const std::function<bool()>& cancelled,
               const std::function<bool(std::string_view)>& on_pcm);
    /// The whole recording as a WAV file.
    std::string synthesize(const std::vector<std::vector<int32_t>>& chunks, const Voice& voice,
                           int seed, const std::function<bool()>& cancelled);
    /// The server is shutting down: a cancelled request kills its worker instead of draining it.
    void shut_down() { stopping_ = true; }

private:
    using Clock = std::chrono::steady_clock;
    void spawn();
    void stop() noexcept;
    /// Cancels request `identity` on the worker and reads and drops the rest of its output
    /// (`frame_left` bytes of an audio frame first), so the worker can serve the next request.
    /// False when that fails or takes longer than 30 s; the caller then replaces the worker.
    bool drain(const std::string& identity, size_t frame_left) noexcept;
    void check(Clock::time_point deadline, const std::function<bool()>& cancelled);
    void wait_fd(int fd, short event, Clock::time_point deadline,
                 const std::function<bool()>& cancelled);
    std::filesystem::path root_;
    int max_pending_;
    int threads_, codec_threads_;
    std::string kernels_;
    std::string device_;
    double timeout_;
    std::atomic<int> pid_{-1}, pending_{0};
    std::atomic<bool> stopping_{false};
    int input_ = -1, output_ = -1;
    std::mutex assign_mutex_; ///< guards leased_
    std::condition_variable freed_;
    bool leased_ = false;
    uint64_t counter_ = 0;
    std::string buffered_;
};
} // namespace sinfer::tts
