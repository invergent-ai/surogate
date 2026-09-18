// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Stage timings for real CPU speech artifacts. Warm-up is reported separately.
#include "model.h"
#include "audio.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <algorithm>
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
                if (std::getenv("SUROGATE_CPU_STREAMING")) {
                    if (!model.streaming()) throw std::runtime_error("Streaming timing needs a streaming model");
                    auto begin = now();
                    Stream stream(model, argv[1]);
                    auto ready = now();
                    std::vector<double> chunks;
                    json events = json::array();
                    constexpr size_t packet_samples = 2560;  // 160 ms of mono 16 kHz audio.
                    for (size_t offset = 0; offset < pcm.size(); offset += packet_samples) {
                        auto packet_begin = now();
                        auto part = stream.accept(
                            {pcm.begin() + offset, pcm.begin() + std::min(pcm.size(), offset + packet_samples)});
                        chunks.push_back(ms(packet_begin, now()));
                        for (auto& event : part)
                            events.push_back(std::move(event));
                    }
                    auto final_begin = now();
                    for (auto& event : stream.accept({}, true))
                        events.push_back(std::move(event));
                    auto end = now();
                    std::sort(chunks.begin(), chunks.end());
                    json transcripts = json::array();
                    for (auto& event : events)
                        transcripts.push_back({{"type", event.at("type")}, {"text", event.at("text")}});
                    std::cout << json{{"audio", argv[i]},
                                      {"audio_s", pcm.size() / 16000.},
                                      {"repeat", repeat},
                                      {"threads", std::stoi(argv[2])},
                                      {"stream_init_ms", ms(begin, ready)},
                                      {"total_ms", ms(ready, end)},
                                      {"finalize_ms", ms(final_begin, end)},
                                      {"max_packet_ms", chunks.empty() ? 0 : chunks.back()},
                                      {"median_packet_ms", chunks.empty() ? 0 : chunks[chunks.size() / 2]},
                                      {"transcripts", transcripts}}
                                     .dump()
                              << std::endl;
                    continue;
                }
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
