// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#include "runtime.h"
#include "../serve/audio_http.h"
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <sstream>
extern char** environ;

namespace sinfer::tts {
using sinfer::serve::audio::HttpError;

namespace {
constexpr size_t max_audio = 64 * 1024 * 1024;

uint32_t le(const std::string& data, size_t offset, size_t count) {
    uint32_t result = 0;
    for (size_t i = 0; i < count; ++i)
        result |= uint32_t(static_cast<unsigned char>(data.at(offset + i))) << (8 * i);
    return result;
}
} // namespace

Runtime::Runtime(std::filesystem::path root, int max_pending, double timeout, int threads,
                 int codec_threads, std::string kernels)
    : root_(std::move(root)), max_pending_(max_pending), threads_(threads),
      codec_threads_(codec_threads), kernels_(std::move(kernels)), timeout_(timeout) {}

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
    std::error_code error;
    if (!directory_.empty()) std::filesystem::remove_all(directory_, error);
    directory_.clear();
    buffered_.clear();
}

void Runtime::spawn() {
    if (healthy()) return;
    stop();
    auto pattern = (std::filesystem::temp_directory_path() / "surogate-tts-XXXXXX").string();
    if (!mkdtemp(pattern.data())) throw HttpError(503, "Cannot create native TTS workspace");
    directory_ = pattern;
    std::filesystem::create_symlink("/dev/null", directory_ / "stats.jsonl");
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
                                          directory_.string(),
                                          std::to_string(threads_),
                                          std::to_string(codec_threads_),
                                          kernels_};
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
    environment.insert(environment.end(),
                       {"CUDA_VISIBLE_DEVICES=", "LD_LIBRARY_PATH=" + (root_ / "lib").string(),
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
    pid_t pid;
    int status = posix_spawn(&pid, argv[0], &actions, nullptr, argv.data(), env.data());
    posix_spawn_file_actions_destroy(&actions);
    close(in[0]);
    close(out[1]);
    if (status) {
        close(in[1]);
        close(out[0]);
        throw HttpError(503, "Cannot start native CPU TTS worker");
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
            throw HttpError(503, "Native CPU TTS worker failed; retry the request");
        if (poll_fd.revents & event) return;
        if (poll_fd.revents & POLLHUP)
            throw HttpError(503, "Native CPU TTS worker stopped; retry the request");
    }
}

std::string Runtime::synthesize(const std::vector<std::vector<int32_t>>& chunks, const Voice& voice,
                                int seed, const std::function<bool()>& cancelled) {
    auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                       std::chrono::duration<double>(timeout_));
    int pending = pending_.fetch_add(1);

    struct Release {
        std::atomic<int>& n;

        ~Release() { --n; }
    } release{pending_};

    if (pending >= max_pending_ + 1) throw HttpError(429, "TTS request queue is full");
    std::unique_lock lock(mutex_, std::defer_lock);
    while (!lock.try_lock_for(std::chrono::milliseconds(25))) check(deadline, cancelled);
    // A queued timeout must not stop the request that owns the worker.
    check(deadline, cancelled);
    try {
        spawn();
        auto identity = std::to_string(++counter_) + "-v" + std::to_string(voice.id);
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
            wait_fd(input_, POLLOUT, deadline, cancelled);
            auto n = write(input_, command.data() + written, command.size() - written);
            if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (n <= 0) throw HttpError(503, "Native CPU TTS worker stopped");
            written += n;
        }
        bool complete = false;
        while (!complete) {
            auto newline = buffered_.find('\n');
            if (newline != std::string::npos) {
                auto line = buffered_.substr(0, newline);
                buffered_.erase(0, newline + 1);
                complete = line.starts_with(identity + " ");
                continue;
            }
            wait_fd(output_, POLLIN, deadline, cancelled);
            char block[4096];
            auto n = read(output_, block, sizeof(block));
            if (n < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (n <= 0) throw HttpError(503, "Native CPU TTS worker stopped");
            buffered_.append(block, n);
            if (buffered_.size() > 1024 * 1024) throw HttpError(503, "Invalid native TTS output");
        }
        check(deadline, cancelled);
        auto audio = directory_ / (identity + ".wav");
        if (!std::filesystem::is_regular_file(audio) ||
            std::filesystem::file_size(audio) > max_audio)
            throw HttpError(503, "Native TTS returned missing or oversized audio");
        std::ifstream stream(audio, std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(stream)), {});
        std::filesystem::remove(audio);
        // This worker emits a canonical 44-byte PCM WAV header.
        if (data.size() <= 44 || data.compare(0, 4, "RIFF") || data.compare(8, 8, "WAVEfmt ") ||
            le(data, 16, 4) != 16 || le(data, 20, 2) != 1 || le(data, 22, 2) != 1 ||
            le(data, 24, 4) != 22050 || le(data, 34, 2) != 16 || data.compare(36, 4, "data") ||
            le(data, 40, 4) != data.size() - 44 || (data.size() - 44) % 2)
            throw HttpError(503, "Native TTS returned invalid audio");
        return data;
    } catch (...) {
        stop();
        throw;
    }
}
} // namespace sinfer::tts
