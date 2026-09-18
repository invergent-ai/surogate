// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Stage timings for real CPU speech artifacts. Warm-up is reported separately.
#include "model.h"
#include "audio.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace sinfer::speech;
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "speech-cpu-bench MODEL THREADS REPEATS AUDIO...\n";
        return 2;
    }
    try {
        c10::InferenceMode guard;
        at::set_num_threads(std::stoi(argv[2]));
        at::set_num_interop_threads(1);
        bool reference = std::getenv("SUROGATE_CPU_REFERENCE") != nullptr;
        Model model(argv[1], "cpu", reference ? CpuKernels::Reference : CpuKernels::Optimized);
        for (int i = 4; i < argc; ++i) {
            std::ifstream file(argv[i], std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(file)), {});
            auto pcm = decode_audio(bytes);
            auto samples = at::from_blob(pcm.data(), {int64_t(pcm.size())}, at::kFloat);
            for (int repeat = 0; repeat < std::stoi(argv[3]); ++repeat) {
                auto now = [] {
                    return std::chrono::steady_clock::now();
                };
                auto ms = [](auto begin, auto end) {
                    return std::chrono::duration<double, std::milli>(end - begin).count();
                };
                const char* profile = std::getenv("SUROGATE_CPU_PROFILE");
                if (profile && repeat == 1) {
                    using namespace torch::autograd::profiler;
                    ProfilerConfig config(ProfilerState::KINETO, true);
                    prepareProfiler(config, {ActivityType::CPU});
                    enableProfiler(config, {ActivityType::CPU});
                }
                auto a = now();
                auto mel = model.mel(samples);
                auto b = now();
                auto encoded = model.encode(mel);
                auto c = now();
                auto ctc = model.ctc(encoded);
                auto d = now();
                auto text = model.beam(ctc);
                auto e = now();
                if (profile && repeat == 1) torch::autograd::profiler::disableProfiler()->save(profile);
                std::cout << json{{"audio", argv[i]},
                                  {"audio_s", pcm.size() / 16000.},
                                  {"repeat", repeat},
                                  {"threads", std::stoi(argv[2])},
                                  {"mel_ms", ms(a, b)},
                                  {"encode_ms", ms(b, c)},
                                  {"ctc_ms", ms(c, d)},
                                  {"beam_ms", ms(d, e)},
                                  {"total_ms", ms(a, e)},
                                  {"text", text}}
                                 .dump()
                          << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
