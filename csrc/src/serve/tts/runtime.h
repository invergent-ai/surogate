// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <set>
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
    /// One worker process: its own copy of the model, synthesizing one request at a time.
    struct Worker;

    /// A request's place at a worker: its queue slot, then a worker of its own, until destroyed.
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
        Worker* worker_ = nullptr;
        std::chrono::steady_clock::time_point deadline_{};
    };

    /// `device`: "cpu", or the CUDA device index the workers synthesize on. `workers`: how many
    /// requests are synthesized at once, each by its own worker process with its own copy of the
    /// model (SUROGATE-CHANGES #7); `max_pending` more may wait.
    Runtime(std::filesystem::path root, int max_pending, double timeout, int threads = 4,
            int codec_threads = 4, std::string kernels = "auto", std::string device = "cpu",
            int workers = 1);
    ~Runtime();
    Runtime(const Runtime&)            = delete;
    Runtime& operator=(const Runtime&) = delete;
    /// Workers that can serve: running with their model loaded (their "0 ready" line has come), or
    /// loading in place of one that had loaded; and leased ones that could when leased (a request
    /// replaces its worker before handing it back if it failed). A worker retrying after it died
    /// before loading does not count until it has loaded.
    int ready_workers() const;
    /// Restarts workers that died while idle, and retries those that could not load, each at most
    /// every 10 s; notes which idle workers have finished loading. Call it periodically.
    void revive();
    /// Starts every worker and synthesizes `warmup` on each, so all load before serving.
    void start(const std::function<void(Lease&)>& warmup);
    /// Waits for a place at a free worker, in arrival order: 429 when the queue is full, 504 past
    /// the deadline, 503 when `cancelled` says the client has gone.
    std::unique_ptr<Lease> acquire(const std::function<bool()>& cancelled);
    /// Synthesizes on the leased worker, handing each PCM chunk (16-bit mono) to `on_pcm` as the
    /// runtime produces it; `on_pcm` returning false cancels. Returns the sample rate.
    int stream(Lease& lease, const std::vector<std::vector<int32_t>>& chunks, const Voice& voice,
               int seed, const std::function<bool()>& cancelled,
               const std::function<bool(std::string_view)>& on_pcm);
    /// The whole recording as a WAV file.
    std::string synthesize(const std::vector<std::vector<int32_t>>& chunks, const Voice& voice,
                           int seed, const std::function<bool()>& cancelled);
    int workers() const { return static_cast<int>(workers_.size()); }
    /// The server is shutting down: a cancelled request's worker is stopped, not drained, and a
    /// stopped worker is not restarted.
    void shut_down() { stopping_ = true; }

private:
    using Clock = std::chrono::steady_clock;
    void spawn(Worker& worker);
    static void stop(Worker& worker) noexcept;
    static bool alive(const Worker& worker);
    /// For a worker nobody leases (assign_mutex_ held): reads what it wrote without waiting, and
    /// notes its "0 ready" line.
    static void poll_ready(Worker& worker) noexcept;
    static bool ready(const Worker& worker);
    void note_failed_start(Worker& worker) const noexcept;
    /// Notes the "0 ready" lines at the front of `worker.buffered`, removing them.
    static void take_ready_lines(Worker& worker) noexcept;
    /// Cancels request `identity` on `worker` and reads and drops the rest of its output
    /// (`frame_left` bytes of an audio frame first), so the worker can serve the next request.
    /// False when that fails or takes longer than 30 s; the caller then replaces the worker.
    bool drain(Worker& worker, const std::string& identity, size_t frame_left) noexcept;
    void check(Clock::time_point deadline, const std::function<bool()>& cancelled);
    void wait_fd(int fd, short event, Clock::time_point deadline,
                 const std::function<bool()>& cancelled);
    std::filesystem::path root_;
    int max_pending_;
    int threads_, codec_threads_;
    std::string kernels_;
    std::string device_;
    double timeout_;
    std::atomic<int> pending_{0};
    std::atomic<bool> stopping_{false};
    std::atomic<bool> started_{false}; ///< start() has loaded every worker
    std::vector<std::unique_ptr<Worker>> workers_;
    mutable std::mutex assign_mutex_; ///< guards Worker::leased and waiting_
    std::condition_variable freed_;
    /// The queue, first come first served: the tickets of the requests waiting for a worker.
    std::set<uint64_t> waiting_;
    uint64_t next_ticket_ = 0;
};

struct Runtime::Worker {
    std::atomic<int> pid{-1};
    int input = -1, output = -1;
    bool leased = false;
    std::atomic<bool> warm{false};   ///< has synthesized since it was started
    std::atomic<bool> loaded{false}; ///< its "0 ready" line has come since it was started
    bool lease_loaded = false;       ///< guarded by assign_mutex_: ready() when it was leased
    bool start_failed = false;       ///< guarded by assign_mutex_: its last start died unloaded
    std::chrono::steady_clock::time_point retry_at{}; ///< a failed start is retried from then
    uint64_t counter = 0;
    std::string buffered;
};
} // namespace sinfer::tts
