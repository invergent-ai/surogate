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
                 int codec_threads, std::string kernels, std::string device)
    : root_(std::move(root)), max_pending_(max_pending), threads_(threads),
      codec_threads_(codec_threads), kernels_(std::move(kernels)), device_(std::move(device)),
      timeout_(timeout) {
    (void)worker_visible_devices(device_); // refused at startup, not at the first request
}

Runtime::~Runtime() { stop(); }

bool Runtime::healthy() const {
    auto pid = pid_.load();
    if (pid <= 0) return false;
    siginfo_t info{};
    return waitid(P_PID, pid, &info, WEXITED | WNOHANG | WNOWAIT) == 0 && info.si_pid == 0;
}

void Runtime::stop() noexcept {
    if (input_ >= 0) {
        close(input_);
        input_ = -1;
    }
    if (output_ >= 0) {
        close(output_);
        output_ = -1;
    }
    int pid = pid_.exchange(-1);
    if (pid > 0) {
        // A cancelled request must finish before another voice can use this worker.
        kill(pid, SIGKILL);
        while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
    }
    buffered_.clear();
}

void Runtime::spawn() {
    if (healthy()) return;
    stop();
    int in[2] = {-1, -1}, out[2] = {-1, -1};
    if (pipe2(in, O_CLOEXEC) || pipe2(out, O_CLOEXEC)) {
        for (int fd : {in[0], in[1], out[0], out[1]})
            if (fd >= 0) close(fd);
        throw HttpError(503, "Cannot create native TTS pipes");
    }
    auto worker =
        std::filesystem::read_symlink("/proc/self/exe").parent_path() / "surogate-tts-worker";
    // Also permits a protocol worker for lifecycle tests, without loading weights.
    if (const char* override = std::getenv("SUROGATE_TTS_WORKER_BIN")) worker = override;
    std::vector<std::string> arguments = {worker.string(),
                                          (root_ / "model.gguf").string(),
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
    pid_    = pid;
    input_  = in[1];
    output_ = out[0];
    if (fcntl(input_, F_SETFL, O_NONBLOCK) < 0 || fcntl(output_, F_SETFL, O_NONBLOCK) < 0)
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
    if (owner_ == nullptr || !held_) return;
    {
        const std::lock_guard lock(owner_->assign_mutex_);
        owner_->leased_ = false;
        held_           = false;
    }
    owner_->freed_.notify_one();
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
    if (pending >= max_pending_ + 1) throw HttpError(429, "TTS request queue is full");
    std::unique_lock lock(assign_mutex_);
    while (leased_) {
        freed_.wait_for(lock, std::chrono::milliseconds(25));
        lock.unlock();
        check(lease->deadline_, cancelled);
        lock.lock();
    }
    leased_       = true;
    lease->held_  = true;
    lock.unlock();
    // A queued timeout must not stop the request that owns the worker.
    check(lease->deadline_, cancelled);
    return lease;
}

int Runtime::stream(Lease& lease, const std::vector<std::vector<int32_t>>& chunks, const Voice& voice,
                    int seed, const std::function<bool()>& cancelled,
                    const std::function<bool(std::string_view)>& on_pcm) {
    if (lease.owner_ != this || !lease.held_) throw std::logic_error("TTS worker not leased");
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
        spawn();
        identity = std::to_string(++counter_) + "-v" + std::to_string(voice.id);
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
            wait_fd(input_, POLLOUT, deadline, watched);
            auto n = write(input_, command.data() + written, command.size() - written);
            if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (n <= 0) throw HttpError(503, "Native TTS worker stopped");
            written += n;
        }
        job_sent = true;
        // Reads until `buffered_` holds `bytes` bytes.
        const auto fill = [&](size_t bytes) {
            while (buffered_.size() < bytes) {
                wait_fd(output_, POLLIN, deadline, watched);
                char block[65536];
                auto n = read(output_, block, sizeof(block));
                if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
                if (n <= 0) throw HttpError(503, "Native TTS worker stopped");
                buffered_.append(block, n);
            }
        };
        size_t produced = 0;
        for (;;) {
            size_t newline;
            while ((newline = buffered_.find('\n')) == std::string::npos) {
                if (buffered_.size() > 4096) throw HttpError(503, "Invalid native TTS output");
                fill(buffered_.size() + 1);
            }
            const auto line = buffered_.substr(0, newline);
            buffered_.erase(0, newline + 1);
            if (!line.starts_with(identity + " ")) continue;
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
                const bool wanted = on_pcm(std::string_view(buffered_).substr(0, bytes));
                buffered_.erase(0, bytes);
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
            return sample_rate;
        }
    } catch (...) {
        // A request whose client left gives its worker back intact: the worker is told to stop
        // at its next audio chunk, and what it wrote before that is read and dropped. The next
        // request then pays no new model load (on a GPU, no new CUDA context and upload, of
        // memory a neighbouring server could take in between). Anything else replaces the
        // worker, and so does a cancel at shutdown, which must not wait for the worker.
        if (cancel_requested && job_sent && !stopping_ && drain(identity, frame_left)) throw;
        stop();
        throw;
    }
}

bool Runtime::drain(const std::string& identity, size_t frame_left) noexcept {
    try {
        const auto until = Clock::now() + std::chrono::seconds(30);
        // A shutdown during the drain ends it; the worker is then killed.
        const std::function<bool()> stopping = [this] { return stopping_.load(); };
        static constexpr std::string_view cancel = "cancel\n";
        for (size_t written = 0; written < cancel.size();) {
            wait_fd(input_, POLLOUT, until, stopping);
            auto n = write(input_, cancel.data() + written, cancel.size() - written);
            if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (n <= 0) return false;
            written += static_cast<size_t>(n);
        }
        const auto fill = [&](size_t bytes) {
            while (buffered_.size() < bytes) {
                wait_fd(output_, POLLIN, until, stopping);
                char block[65536];
                auto n = read(output_, block, sizeof(block));
                if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
                if (n <= 0) throw HttpError(503, "Native TTS worker stopped");
                buffered_.append(block, n);
            }
        };
        fill(frame_left);
        buffered_.erase(0, frame_left);
        for (;;) {
            size_t newline;
            while ((newline = buffered_.find('\n')) == std::string::npos) {
                if (buffered_.size() > 4096) return false;
                fill(buffered_.size() + 1);
            }
            const auto line = buffered_.substr(0, newline);
            buffered_.erase(0, newline + 1);
            if (!line.starts_with(identity + " ")) continue;
            std::istringstream fields(line.substr(identity.size() + 1));
            fields.imbue(std::locale::classic());
            std::string kind;
            fields >> kind;
            if (kind == "done" || kind == "cancelled") return true;
            size_t bytes = 0;
            if (kind != "pcm" || !byte_count(fields, bytes) || bytes > max_audio) return false;
            fill(bytes);
            buffered_.erase(0, bytes);
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
