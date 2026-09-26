// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#include "runtime.h"
#include "../serve/audio_http.h"
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>
extern char** environ;

namespace sinfer::tts {
using sinfer::serve::audio::HttpError;

namespace {
constexpr size_t max_audio = 96 * 1024 * 1024; // 38 minutes of 22,050 Hz 16-bit audio

/// The next field of a worker line as a byte count: decimal digits only (no sign, no space), so a
/// malformed frame is refused rather than read as a huge or wrapped size.
bool byte_count(std::istringstream& fields, size_t& bytes) {
    std::string text;
    if (!(fields >> text) || text.empty() || text.size() > 12 ||
        text.find_first_not_of("0123456789") != std::string::npos)
        return false;
    bytes = static_cast<size_t>(std::stoull(text));
    return true;
}

/// The CUDA_VISIBLE_DEVICES the worker gets: none on the CPU; for `--device N`, the N-th card this
/// process may use, so N means what it means to the LLM server (an ordinal among the visible
/// devices), whatever CUDA_VISIBLE_DEVICES the server itself was started with.
std::string worker_visible_devices(const std::string& device) {
    if (device == "cpu") return {};
    const char* inherited = std::getenv("CUDA_VISIBLE_DEVICES");
    if (inherited == nullptr) return device;
    // Set but empty hides every device, as it does for CUDA itself.
    std::vector<std::string> visible;
    std::string entry;
    for (std::istringstream list(inherited); std::getline(list, entry, ',');) {
        const auto first = entry.find_first_not_of(" \t");
        const auto last  = entry.find_last_not_of(" \t");
        if (first == std::string::npos)
            throw std::invalid_argument(std::string("CUDA_VISIBLE_DEVICES=") + inherited +
                                        " has an empty entry");
        visible.push_back(entry.substr(first, last - first + 1));
    }
    const auto index = static_cast<size_t>(std::stoul(device));
    if (index >= visible.size())
        throw std::invalid_argument("--device " + device + " is not among CUDA_VISIBLE_DEVICES=" +
                                    inherited);
    return visible[index];
}

} // namespace

Runtime::Runtime(std::filesystem::path root, int max_pending, double timeout, int threads,
                 int codec_threads, std::string kernels, std::string device, int workers,
                 std::string model_file)
    : root_(std::move(root)), model_file_(std::move(model_file)), max_pending_(max_pending), threads_(threads),
      codec_threads_(codec_threads), kernels_(std::move(kernels)), device_(std::move(device)),
      timeout_(timeout) {
    (void)worker_visible_devices(device_); // refused at startup, not at the first request
    if (workers < 1) throw std::invalid_argument("at least one TTS worker is needed");
    for (int i = 0; i < workers; ++i) workers_.push_back(std::make_unique<Worker>());
}

Runtime::~Runtime() {
    for (auto& worker : workers_) stop(*worker);
}

bool Runtime::alive(const Worker& worker) {
    auto pid = worker.pid.load();
    if (pid <= 0) return false;
    siginfo_t info{};
    return waitid(P_PID, pid, &info, WEXITED | WNOHANG | WNOWAIT) == 0 && info.si_pid == 0;
}

void Runtime::take_ready_lines(Worker& worker) noexcept {
    static constexpr std::string_view ready = "0 ready\n";
    while (std::string_view(worker.buffered).starts_with(ready)) {
        worker.buffered.erase(0, ready.size());
        worker.loaded = true;
    }
}

void Runtime::poll_ready(Worker& worker) noexcept {
    if (worker.loaded || worker.output < 0) return;
    char block[256];
    while (worker.buffered.size() <= 4096) {
        const auto n = read(worker.output, block, sizeof(block));
        if (n <= 0) break; // nothing more yet (EAGAIN), or it stopped
        worker.buffered.append(block, static_cast<size_t>(n));
    }
    take_ready_lines(worker);
    if (worker.loaded) worker.start_failed = false;
}

bool Runtime::ready(const Worker& worker) {
    // Loaded, or loading in place of one that had loaded (a crash in synthesis): not one that is
    // retrying after dying before its model loaded -- that load is likely to fail again.
    return alive(worker) && (worker.loaded || !worker.start_failed);
}

void Runtime::note_failed_start(Worker& worker) const noexcept {
    // A worker that stopped before its model loaded, after startup, failed to start: revive()
    // tries it again in 10 s.
    if (started_ && !worker.start_failed && !worker.loaded && !alive(worker)) {
        worker.start_failed = true;
        worker.retry_at     = Clock::now() + std::chrono::seconds(10);
    }
}

int Runtime::ready_workers() const {
    const std::lock_guard lock(assign_mutex_);
    int count = 0;
    for (const auto& worker : workers_) {
        // A leased worker counts if it was ready when leased: its request replaces it before
        // handing it back if it fails.
        if (worker->leased) {
            count += worker->lease_loaded || worker->loaded;
            continue;
        }
        poll_ready(*worker);
        note_failed_start(*worker);
        count += ready(*worker);
    }
    return count;
}

void Runtime::revive() {
    // Off the request path: a request never waits for a model load while a loaded worker is free.
    if (!started_ || stopping_) return;
    for (auto& worker : workers_) {
        const auto now = Clock::now();
        {
            const std::lock_guard lock(assign_mutex_);
            if (worker->leased) continue;
            poll_ready(*worker);
            if (alive(*worker)) continue;
            // One that died after loading is restarted at once; one that died loading is retried
            // every 10 s, and requests go elsewhere meanwhile (acquire()).
            note_failed_start(*worker);
            if (worker->start_failed && now < worker->retry_at) continue;
            worker->leased       = true; // held while it restarts
            worker->lease_loaded = false;
        }
        try {
            spawn(*worker);
        } catch (...) {
        }
        {
            const std::lock_guard lock(assign_mutex_);
            worker->retry_at = now + std::chrono::seconds(10);
            worker->leased   = false;
        }
        freed_.notify_all();
    }
}

void Runtime::start(const std::function<void(Lease&)>& warmup) {
    // Every worker loads its model and synthesizes once before the server listens. The leases
    // are held together, so each warm-up lands on a different worker; each is taken just before
    // its warm-up, so each has the whole request timeout to itself.
    std::vector<std::unique_ptr<Lease>> leases;
    for (size_t i = 0; i < workers_.size(); ++i) {
        leases.push_back(acquire([this] { return stopping_.load(); }));
        try {
            warmup(*leases.back());
        } catch (const std::exception& error) {
            if (stopping_) throw;
            const auto* failure = dynamic_cast<const HttpError*>(&error);
            std::string hint;
            if (failure != nullptr && failure->status == 504)
                hint = ". Loading took longer than --request-timeout; raise it";
            else if (i > 0)
                hint = ". Each of the --max-num-seqs workers holds its own copy of the model; try "
                       "fewer";
            throw std::runtime_error("TTS worker " + std::to_string(i + 1) + " of " +
                                     std::to_string(workers_.size()) +
                                     " failed to load and synthesize: " + error.what() + hint);
        }
    }
    started_ = true;
}

void Runtime::stop(Worker& worker) noexcept {
    if (worker.input >= 0) {
        close(worker.input);
        worker.input = -1;
    }
    if (worker.output >= 0) {
        close(worker.output);
        worker.output = -1;
    }
    int pid = worker.pid.exchange(-1);
    if (pid > 0) {
        // A cancelled request must finish before another voice can use this worker.
        kill(pid, SIGKILL);
        while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
    }
    worker.buffered.clear();
    // Whatever comes next starts afresh: a start that fails before its process runs is then seen
    // as a failed start (and retried every 10 s), not as the loaded worker it replaces.
    worker.loaded = false;
    worker.warm   = false;
}

void Runtime::spawn(Worker& worker) {
    if (alive(worker)) return;
    stop(worker);
    int in[2] = {-1, -1}, out[2] = {-1, -1};
    if (pipe2(in, O_CLOEXEC) || pipe2(out, O_CLOEXEC)) {
        for (int fd : {in[0], in[1], out[0], out[1]})
            if (fd >= 0) close(fd);
        throw HttpError(503, "Cannot create native TTS pipes");
    }
    auto program =
        std::filesystem::read_symlink("/proc/self/exe").parent_path() / "surogate-tts-worker";
    // Also permits a protocol worker for lifecycle tests, without loading weights.
    if (const char* override = std::getenv("SUROGATE_TTS_WORKER_BIN")) program = override;
    std::vector<std::string> arguments = {program.string(),
                                          (root_ / model_file_).string(),
                                          (root_ / "codec.gguf").string(),
                                          "/dev/stdin",
                                          "-", // stream mode: audio frames on stdout, no files
                                          std::to_string(threads_),
                                          std::to_string(codec_threads_),
                                          kernels_,
                                          device_ == "cpu" ? "cpu" : "cuda"};
    std::vector<char*> argv;
    for (auto& s : arguments) argv.push_back(s.data());
    argv.push_back(nullptr);
    std::vector<std::string> environment;
    for (char** p = environ; *p; ++p) {
        std::string s(*p);
        if (!s.starts_with("CUDA_VISIBLE_DEVICES=") && !s.starts_with("LD_LIBRARY_PATH=") &&
            !s.starts_with("OMP_NUM_THREADS=") && !s.starts_with("OPENBLAS_NUM_THREADS="))
            environment.push_back(std::move(s));
    }
    // The worker sees only the card it serves on (as its device 0), or none on the CPU.
    environment.insert(environment.end(),
                       {"CUDA_VISIBLE_DEVICES=" + worker_visible_devices(device_),
                        "LD_LIBRARY_PATH=" + (root_ / "lib").string(),
                        "OMP_NUM_THREADS=" + std::to_string(threads_), "OPENBLAS_NUM_THREADS=1"});
    std::vector<char*> env;
    for (auto& s : environment) env.push_back(s.data());
    env.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, in[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, in[1]);
    posix_spawn_file_actions_addclose(&actions, out[0]);
    // The runtime libraries carry empty RUNPATH entries, which the loader reads as the working
    // directory: run the worker from / so nothing there can be loaded in their place.
    posix_spawn_file_actions_addchdir_np(&actions, "/");
    pid_t pid;
    int status = posix_spawn(&pid, argv[0], &actions, nullptr, argv.data(), env.data());
    posix_spawn_file_actions_destroy(&actions);
    close(in[0]);
    close(out[1]);
    if (status) {
        close(in[1]);
        close(out[0]);
        throw HttpError(503, "Cannot start native TTS worker");
    }
    worker.warm   = false;
    worker.loaded = false;
    worker.pid    = pid;
    worker.input  = in[1];
    worker.output = out[0];
    if (fcntl(worker.input, F_SETFL, O_NONBLOCK) < 0 || fcntl(worker.output, F_SETFL, O_NONBLOCK) < 0)
        throw HttpError(503, "Cannot configure native TTS pipes");
}

void Runtime::check(Clock::time_point deadline, const std::function<bool()>& cancelled) {
    if (cancelled()) throw HttpError(503, "TTS request cancelled");
    if (Clock::now() >= deadline) throw HttpError(504, "TTS request timed out");
}

void Runtime::wait_fd(int fd, short event, Clock::time_point deadline,
                      const std::function<bool()>& cancelled) {
    for (;;) {
        check(deadline, cancelled);
        pollfd poll_fd{fd, event, 0};
        int n = poll(&poll_fd, 1, 25);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 || (poll_fd.revents & (POLLERR | POLLNVAL)))
            throw HttpError(503, "Native TTS worker failed; retry the request");
        if (poll_fd.revents & event) return;
        if (poll_fd.revents & POLLHUP)
            throw HttpError(503, "Native TTS worker stopped; retry the request");
    }
}

std::string wav_header(uint32_t sample_rate, uint32_t data_bytes) {
    std::string header;
    header.reserve(44);
    const auto put = [&](uint32_t value, int bytes) {
        for (int i = 0; i < bytes; ++i) header.push_back(static_cast<char>((value >> (8 * i)) & 255));
    };
    // An unknown length (a stream) is written as the largest one, which players read as "until
    // the end of the data".
    const bool open_ended = data_bytes == 0xFFFFFFFFu;
    header += "RIFF";
    put(open_ended ? 0xFFFFFFFFu : 36 + data_bytes, 4);
    header += "WAVEfmt ";
    put(16, 4);
    put(1, 2);
    put(1, 2);
    put(sample_rate, 4);
    put(sample_rate * 2, 4);
    put(2, 2);
    put(16, 2);
    header += "data";
    put(data_bytes, 4);
    return header;
}

void Runtime::Lease::release_worker() {
    if (owner_ == nullptr || worker_ == nullptr) return;
    {
        const std::lock_guard lock(owner_->assign_mutex_);
        if (worker_->loaded) worker_->start_failed = false;
        worker_->leased = false;
        worker_         = nullptr;
    }
    owner_->freed_.notify_all(); // waiters take workers in arrival order
}

Runtime::Lease::~Lease() {
    if (owner_ == nullptr) return;
    release_worker();
    owner_->pending_.fetch_sub(1);
}

std::unique_ptr<Runtime::Lease> Runtime::acquire(const std::function<bool()>& cancelled) {
    auto lease       = std::unique_ptr<Lease>(new Lease());
    lease->deadline_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                          std::chrono::duration<double>(timeout_));
    const int pending = pending_.fetch_add(1);
    lease->owner_     = this;
    if (pending >= max_pending_ + workers()) throw HttpError(429, "TTS request queue is full");
    std::unique_lock lock(assign_mutex_);
    // First come, first served: the oldest waiting request takes the next free worker: a loaded
    // one first (one that has already synthesized before others), then one loading afresh, then a
    // stopped one it starts itself. A worker whose last start died before loading (a GPU whose
    // memory was taken) is taken only when no other worker is running or leased, and a stopped one
    // only once its retry time has come: the request would likely fail, and another worker will
    // be free soon. When no worker is running or can start, the request is refused at once.
    const uint64_t ticket = next_ticket_++;
    waiting_.insert(ticket);
    struct Leave { // leaves the queue on every exit, a timeout or a cancel included
        Runtime& runtime;
        std::unique_lock<std::mutex>& lock;
        uint64_t ticket;
        ~Leave() {
            if (!lock.owns_lock()) lock.lock();
            runtime.waiting_.erase(ticket);
            runtime.freed_.notify_all(); // the next in line may go
        }
    };
    {
        const Leave leave{*this, lock, ticket};
        for (;;) {
            if (*waiting_.begin() == ticket) {
                const auto now = Clock::now();
                int live       = 0; // workers running or leased
                for (auto& worker : workers_) {
                    if (!worker->leased) {
                        poll_ready(*worker);
                        note_failed_start(*worker);
                    }
                    live += worker->leased || alive(*worker);
                }
                Worker* chosen = nullptr;
                int best       = -1;
                for (auto& worker : workers_) {
                    if (worker->leased) continue;
                    const bool running = alive(*worker);
                    const bool alone   = live - int(running) == 0;
                    int rank           = -1;
                    if (running && worker->loaded) rank = worker->warm ? 5 : 4;
                    else if (running && !worker->start_failed) rank = 3;
                    else if (!running && !worker->start_failed) rank = 2;
                    else if (running) rank = alone ? 1 : -1;
                    else if (alone && now >= worker->retry_at) rank = 0;
                    if (rank > best) {
                        chosen = worker.get();
                        best   = rank;
                    }
                }
                if (chosen != nullptr) {
                    chosen->leased       = true;
                    chosen->lease_loaded = ready(*chosen);
                    lease->worker_       = chosen;
                    break;
                }
                if (live == 0) throw HttpError(503, "No TTS worker can load its model; retry the request later");
            }
            freed_.wait_for(lock, std::chrono::milliseconds(25));
            lock.unlock();
            check(lease->deadline_, cancelled);
            lock.lock();
        }
    }
    lock.unlock();
    // A queued timeout must not stop the request that owns the worker.
    check(lease->deadline_, cancelled);
    return lease;
}

int Runtime::stream(Lease& lease, const std::vector<std::vector<int32_t>>& chunks, const Voice& voice,
                    int seed, const std::function<bool()>& cancelled,
                    const std::function<bool(std::string_view)>& on_pcm) {
    if (lease.owner_ != this || lease.worker_ == nullptr) throw std::logic_error("TTS worker not leased");
    Worker& worker      = *lease.worker_;
    const auto deadline = lease.deadline_;
    // What a cancelled GPU request needs to hand its worker back intact (see drain()).
    bool cancel_requested = false, job_sent = false;
    size_t frame_left = 0;
    std::string identity;
    const std::function<bool()> watched = [&] {
        if (!cancelled()) return false;
        cancel_requested = true;
        return true;
    };
    try {
        spawn(worker);
        identity = std::to_string(++worker.counter) + "-v" + std::to_string(voice.id);
        std::ostringstream job;
        job.imbue(std::locale::classic());
        job << identity << '\t' << voice.id << '\t' << seed << '\t' << voice.temperature << '\t'
            << voice.cfg << '\t';
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i) job << ';';
            for (size_t j = 0; j < chunks[i].size(); ++j) {
                if (j) job << ' ';
                job << chunks[i][j];
            }
        }
        job << '\n';
        auto command = job.str();
        for (size_t written = 0; written < command.size();) {
            wait_fd(worker.input, POLLOUT, deadline, watched);
            auto n = write(worker.input, command.data() + written, command.size() - written);
            if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (n <= 0) throw HttpError(503, "Native TTS worker stopped");
            written += n;
        }
        job_sent = true;
        std::string& buffered = worker.buffered;
        // Reads until `buffered` holds `bytes` bytes.
        const auto fill = [&](size_t bytes) {
            while (buffered.size() < bytes) {
                wait_fd(worker.output, POLLIN, deadline, watched);
                char block[65536];
                auto n = read(worker.output, block, sizeof(block));
                if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
                if (n <= 0) throw HttpError(503, "Native TTS worker stopped");
                buffered.append(block, n);
            }
        };
        size_t produced = 0;
        for (;;) {
            size_t newline;
            while ((newline = buffered.find('\n')) == std::string::npos) {
                if (buffered.size() > 4096) throw HttpError(503, "Invalid native TTS output");
                fill(buffered.size() + 1);
            }
            const auto line = buffered.substr(0, newline);
            buffered.erase(0, newline + 1);
            if (!line.starts_with(identity + " ")) {
                if (line == "0 ready") worker.loaded = true; // its model finished loading
                continue;
            }
            std::istringstream fields(line.substr(identity.size() + 1));
            fields.imbue(std::locale::classic());
            std::string kind;
            fields >> kind;
            if (kind == "pcm") {
                size_t bytes = 0;
                if (!byte_count(fields, bytes) || bytes == 0 || bytes % 2 || bytes > max_audio - produced)
                    throw HttpError(503, "Native TTS returned invalid or oversized audio");
                frame_left = bytes;
                fill(bytes);
                produced += bytes;
                const bool wanted = on_pcm(std::string_view(buffered).substr(0, bytes));
                buffered.erase(0, bytes);
                frame_left = 0;
                if (!wanted) {
                    cancel_requested = true;
                    throw HttpError(503, "TTS request cancelled");
                }
                check(deadline, watched);
                continue;
            }
            int sample_rate = 0;
            if (kind != "done" || !(fields >> sample_rate) || sample_rate != 22050 || produced == 0)
                throw HttpError(503, "Native TTS returned invalid audio");
            worker.loaded = true; // also for a worker that sends no ready line
            worker.warm   = true;
            return sample_rate;
        }
    } catch (...) {
        // A request whose client left gives its worker back intact: the worker is told to stop
        // at its next audio chunk, and what it wrote before that is read and dropped. The next
        // request then pays no new model load (on a GPU, no new CUDA context and upload, of
        // memory a neighbouring server could take in between). Anything else replaces the
        // worker, at once unless it died before its model loaded; a cancel at shutdown only
        // stops it.
        if (cancel_requested && job_sent && !stopping_ && drain(worker, identity, frame_left))
            throw;
        // A worker that ended by itself before its model loaded failed to start. One the server
        // stops here (a timeout, a client that left, a failed drain) was healthy, as was one that
        // had loaded: those are replaced at once.
        const bool died_loading = !worker.loaded && !alive(worker);
        stop(worker);
        if (started_ && !stopping_) {
            if (!died_loading) {
                {
                    const std::lock_guard lock(assign_mutex_);
                    worker.start_failed = false;
                }
                try {
                    spawn(worker);
                } catch (...) {
                    // Left stopped: it counts as a failed start, retried every 10 s.
                }
            } else {
                // Retried in 10 s by revive(), not by every request that comes in the meantime.
                const std::lock_guard lock(assign_mutex_);
                worker.start_failed = true;
                worker.retry_at     = Clock::now() + std::chrono::seconds(10);
            }
        }
        throw;
    }
}

bool Runtime::drain(Worker& worker, const std::string& identity, size_t frame_left) noexcept {
    try {
        const auto until = Clock::now() + std::chrono::seconds(30);
        // A shutdown during the drain ends it; the worker is then killed.
        const std::function<bool()> stopping = [this] { return stopping_.load(); };
        static constexpr std::string_view cancel = "cancel\n";
        for (size_t written = 0; written < cancel.size();) {
            wait_fd(worker.input, POLLOUT, until, stopping);
            auto n = write(worker.input, cancel.data() + written, cancel.size() - written);
            if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (n <= 0) return false;
            written += static_cast<size_t>(n);
        }
        std::string& buffered = worker.buffered;
        const auto fill = [&](size_t bytes) {
            while (buffered.size() < bytes) {
                wait_fd(worker.output, POLLIN, until, stopping);
                char block[65536];
                auto n = read(worker.output, block, sizeof(block));
                if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
                if (n <= 0) throw HttpError(503, "Native TTS worker stopped");
                buffered.append(block, n);
            }
        };
        fill(frame_left);
        buffered.erase(0, frame_left);
        for (;;) {
            size_t newline;
            while ((newline = buffered.find('\n')) == std::string::npos) {
                if (buffered.size() > 4096) return false;
                fill(buffered.size() + 1);
            }
            const auto line = buffered.substr(0, newline);
            buffered.erase(0, newline + 1);
            if (!line.starts_with(identity + " ")) {
                if (line == "0 ready") worker.loaded = true;
                continue;
            }
            std::istringstream fields(line.substr(identity.size() + 1));
            fields.imbue(std::locale::classic());
            std::string kind;
            fields >> kind;
            if (kind == "done" || kind == "cancelled") return true;
            size_t bytes = 0;
            if (kind != "pcm" || !byte_count(fields, bytes) || bytes > max_audio) return false;
            fill(bytes);
            buffered.erase(0, bytes);
        }
    } catch (...) {
        return false;
    }
}

std::string Runtime::synthesize(const std::vector<std::vector<int32_t>>& chunks, const Voice& voice,
                                int seed, const std::function<bool()>& cancelled) {
    auto lease = acquire(cancelled);
    std::string pcm;
    const int rate = stream(*lease, chunks, voice, seed, cancelled, [&](std::string_view part) {
        pcm.append(part);
        return true;
    });
    // The canonical 44-byte header of 16-bit mono PCM.
    return wav_header(static_cast<uint32_t>(rate), static_cast<uint32_t>(pcm.size())) + pcm;
}
} // namespace sinfer::tts
