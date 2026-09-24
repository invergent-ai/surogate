// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Adapter for the pinned native model runtime. Resolved after its library loads.
#include "vendor/magpie_runtime.h"
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void le(std::ostream& out, uint32_t value, int bytes) {
    for (int index = 0; index < bytes; ++index)
        out.put(static_cast<char>((value >> (8 * index)) & 255));
}

// Writes all of `data` to `fd`; false when the reader has gone.
static bool write_all(int fd, const std::string& data) {
    for (size_t written = 0; written < data.size();) {
        const auto n = write(fd, data.data() + written, data.size() - written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        written += static_cast<size_t>(n);
    }
    return true;
}

static void wav(const std::string& path, const std::vector<uint8_t>& audio, int rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open output WAV");
    out.write("RIFF", 4);
    le(out, 36 + audio.size(), 4);
    out.write("WAVEfmt ", 8);
    le(out, 16, 4);
    le(out, 1, 2);
    le(out, 1, 2);
    le(out, rate, 4);
    le(out, rate * 2, 4);
    le(out, 2, 2);
    le(out, 16, 2);
    out.write("data", 4);
    le(out, audio.size(), 4);
    out.write(reinterpret_cast<const char*>(audio.data()), audio.size());
}

extern "C" int surogate_tts_worker_main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: batch MAGPIE CODEC JOBS OUTPUT_DIR THREADS CODEC_THREADS cpu|cuda\n";
        return 2;
    }
    const std::string device = argv[7];
    if (device != "cpu" && device != "cuda") return 2;
    // CUDA: the worker sees only the card it serves on (CUDA_VISIBLE_DEVICES), as device 0.
    const bool cuda = device == "cuda";
    using namespace nemo_speech::tts;
    MagpieRuntimeConfig config;
    config.magpie_model  = argv[1];
    config.codec_model   = argv[2];
    config.threads       = std::stoi(argv[5]);
    config.codec_threads = std::stoi(argv[6]);
    if (config.threads < 1 || config.codec_threads < 1 || config.threads > 256 ||
        config.codec_threads > 256)
        return 2;
    config.magpie_cpu           = !cuda;
    config.codec_cpu            = !cuda;
    config.lt_fp32              = true;
    config.lt_backend           = cuda ? MagpieBackendPreference::Cuda : MagpieBackendPreference::Cpu;
    config.sampling_backend     = cuda ? MagpieBackendPreference::Cuda : MagpieBackendPreference::Cpu;
    // Device memory only: not unified memory, whatever GGML_CUDA_ENABLE_UNIFIED_MEMORY says.
    config.uma_mode             = MagpieUmaMode::Off;
    config.longform_mode        = MagpieLongformMode::Auto;
    config.seed                 = 9;
    config.steps                = 900;
    config.top_k                = 80;
    config.temperature          = .5f;
    config.override_temperature = true;
    config.cfg_scale            = 1.5f;
    config.override_cfg_scale   = true;
    // OUTPUT_DIR "-" or "fd:N": stream mode. Audio goes to stdout (or to descriptor N) as it is
    // produced, one frame per PCM chunk ("ID pcm N\n" and N bytes of 16-bit mono PCM), then
    // "ID done RATE AUDIO_S ELAPSED_S TTFA_MS\n"; no files. A "cancel" line on stdin while a
    // request is synthesized stops it at its next audio chunk, which is then answered with
    // "ID cancelled\n" instead; a "cancel" line between requests (one that came too late) is
    // skipped. Otherwise each request's WAV and stats go to OUTPUT_DIR.
    const std::string output = argv[4];
    const bool stream        = output == "-" || output.starts_with("fd:");
    // The frames get a descriptor of their own, and anything else that writes to stdout (the
    // runtime's libraries included) lands on stderr, so a stray line can never split a frame's
    // header from its audio. surogate-tts-worker sets this up before it loads the runtime and
    // passes "fd:N"; "-" does it here.
    int protocol = -1;
    if (output.starts_with("fd:")) {
        protocol = std::atoi(output.c_str() + 3);
        if (protocol < 3) return 2;
    } else if (stream) {
        std::cout.flush();
        std::fflush(stdout);
        protocol = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
        if (protocol < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
            std::cerr << "Cannot set up the worker's output\n";
            return 3;
        }
    }
    try {
        MagpieTtsRuntime synth(config);
        // The model is loaded: the server counts this worker as ready from now on, not while a
        // load that may fail is still running (SUROGATE-CHANGES #7).
        if (stream && !write_all(protocol, "0 ready\n")) throw std::runtime_error("Output closed");
        std::ifstream jobs(argv[3]);
        std::ofstream report;
        if (!stream) report.open(std::string(argv[4]) + "/stats.jsonl");
        if (!jobs || (!stream && !report)) throw std::runtime_error("Cannot open jobs or stats");
        std::string line;
        while (std::getline(jobs, line)) {
            if (stream && line == "cancel") continue;
            std::istringstream fields(line);
            std::string id, voice_text, seed_text, temperature_text, guidance_text, encoded;
            std::getline(fields, id, '\t');
            std::getline(fields, voice_text, '\t');
            std::getline(fields, seed_text, '\t');
            std::getline(fields, temperature_text, '\t');
            std::getline(fields, guidance_text, '\t');
            std::getline(fields, encoded);
            if (id.empty() || id.find_first_not_of("0123456789-v") != std::string::npos)
                throw std::runtime_error("Invalid corpus ID");
            MagpieSynthesisOptions options;
            options.speaker     = std::stoi(voice_text);
            options.seed        = std::stoi(seed_text);
            options.temperature = std::stof(temperature_text);
            options.cfg_scale   = std::stof(guidance_text);
            if (!std::isfinite(options.temperature) || options.temperature <= 0 ||
                !std::isfinite(options.cfg_scale) || options.cfg_scale <= 0)
                throw std::runtime_error("Invalid fixed sampling policy");
            options.override_temperature = true;
            options.override_cfg_scale   = true;
            std::vector<std::vector<int32_t>> chunks;
            std::istringstream pieces(encoded);
            std::string piece;
            while (std::getline(pieces, piece, ';')) {
                std::istringstream values(piece);
                std::vector<int32_t> tokens;
                int32_t token;
                while (values >> token) tokens.push_back(token);
                if (tokens.empty()) throw std::runtime_error("Empty input chunk");
                chunks.push_back(std::move(tokens));
            }
            if (chunks.empty()) throw std::runtime_error("Empty request");
            if (stream) {
                bool produced = false, cancelled = false;
                MagpieSynthesisStats stats{};
                try {
                    stats = synth.synthesize(chunks, options, [&](const std::string& pcm) {
                        // Anything on stdin now is the server's cancel: the next job only comes
                        // after this one is answered.
                        if (jobs.rdbuf()->in_avail() > 0) {
                            cancelled = true;
                            return false;
                        }
                        if (pcm.empty()) return true;
                        produced = true;
                        return write_all(protocol, id + " pcm " + std::to_string(pcm.size()) + "\n" + pcm);
                    });
                } catch (const std::exception&) {
                    if (!cancelled) throw;
                }
                if (cancelled) {
                    if (!std::getline(jobs, line) || line != "cancel")
                        throw std::runtime_error("Expected a cancel line");
                    if (!write_all(protocol, id + " cancelled\n")) throw std::runtime_error("Output closed");
                    continue;
                }
                if (!produced) throw std::runtime_error("Empty audio output");
                std::ostringstream done;
                done.imbue(std::locale::classic());
                done << id << " done " << stats.sample_rate << ' ' << stats.audio_s << ' '
                     << stats.elapsed_s << ' ' << stats.ttfa_ms << '\n';
                if (!write_all(protocol, done.str())) throw std::runtime_error("Output closed");
                continue;
            }
            const std::string file = std::string(argv[4]) + "/" + id + ".wav";
            if (std::ifstream(file))
                throw std::runtime_error("Output exists; do not overwrite audio");
            std::vector<uint8_t> audio;
            auto stats = synth.synthesize(chunks, options, [&](const std::string& pcm) {
                audio.insert(audio.end(), pcm.begin(), pcm.end());
                return true;
            });
            if (audio.empty()) throw std::runtime_error("Empty audio output");
            wav(file, audio, stats.sample_rate);
            report << "{\"id\":\"" << id << "\",\"voice\":" << options.speaker
                   << ",\"seed\":" << options.seed << ",\"audio_s\":" << stats.audio_s
                   << ",\"elapsed_s\":" << stats.elapsed_s << ",\"rtf\":" << stats.rtf
                   << ",\"ttfa_ms\":" << stats.ttfa_ms
                   << ",\"temperature\":" << options.temperature
                   << ",\"cfg_scale\":" << options.cfg_scale
                   << ",\"frames\":" << stats.generated_frames
                   << ",\"input_chunks\":" << chunks.size()
                   << ",\"sample_rate\":" << stats.sample_rate << "}\n"
                   << std::flush;
            std::cout << id << " " << stats.audio_s << " seconds audio in " << stats.elapsed_s
                      << " seconds\n"
                      << std::flush;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 3;
    }
    return 0;
}
