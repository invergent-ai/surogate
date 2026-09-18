// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Adapter for the pinned native model runtime. Resolved after its library loads.
#include "vendor/magpie_runtime.h"
#include <sstream>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void le(std::ostream& out, uint32_t value, int bytes) {
    for (int index = 0; index < bytes; ++index)
        out.put(static_cast<char>((value >> (8 * index)) & 255));
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
    if (argc != 7) {
        std::cerr << "Usage: batch MAGPIE CODEC JOBS OUTPUT_DIR THREADS CODEC_THREADS\n";
        return 2;
    }
    using namespace nemo_speech::tts;
    MagpieRuntimeConfig config;
    config.magpie_model  = argv[1];
    config.codec_model   = argv[2];
    config.threads       = std::stoi(argv[5]);
    config.codec_threads = std::stoi(argv[6]);
    if (config.threads < 1 || config.codec_threads < 1 || config.threads > 256 ||
        config.codec_threads > 256)
        return 2;
    config.magpie_cpu           = true;
    config.codec_cpu            = true;
    config.lt_fp32              = true;
    config.lt_backend           = MagpieBackendPreference::Cpu;
    config.sampling_backend     = MagpieBackendPreference::Cpu;
    config.longform_mode        = MagpieLongformMode::Auto;
    config.seed                 = 9;
    config.steps                = 900;
    config.top_k                = 80;
    config.temperature          = .5f;
    config.override_temperature = true;
    config.cfg_scale            = 1.5f;
    config.override_cfg_scale   = true;
    try {
        MagpieTtsRuntime synth(config);
        std::ifstream jobs(argv[3]);
        std::ofstream report(std::string(argv[4]) + "/stats.jsonl");
        if (!jobs || !report) throw std::runtime_error("Cannot open jobs or stats");
        std::string line;
        while (std::getline(jobs, line)) {
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
