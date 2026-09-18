// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Compare the CPU kernel path with the original FP32 ATen model and decoders.
#include "model.h"
#include "audio.h"
#include <ATen/Parallel.h>
#include <fstream>
#include <iostream>
using namespace sinfer::speech;
int main(int argc, char** argv) {
    if (argc != 4) return 77;
    try {
        c10::InferenceMode guard;
        at::set_num_threads(4);
        at::set_num_interop_threads(1);
        Model reference_model(argv[1], "cpu", CpuKernels::Reference);
        Model model(argv[1], "cpu", CpuKernels::Optimized);
        json cases;
        std::ifstream(argv[2]) >> cases;
        std::ofstream report(argv[3]);
        int failures = 0;
        for (auto& row : cases) {
            std::ifstream file(row.at("audio").get<std::string>(), std::ios::binary);
            if (!file) throw std::runtime_error("Missing control recording");
            std::string bytes((std::istreambuf_iterator<char>(file)), {});
            auto pcm = decode_audio(bytes);
            auto samples = at::from_blob(pcm.data(), {int64_t(pcm.size())}, at::kFloat);
            auto mel = reference_model.mel(samples);
            auto original = reference_model.encode(mel);
            auto reference = reference_model.ctc(original);
            auto reference_text = reference_model.beam(reference);
            PredictorState ref_state;
            auto reference_tdt = reference_model.tdt(original, ref_state);
            auto optimized = model.encode(mel);
            auto candidate = model.ctc(optimized);
            auto candidate_text = model.beam(candidate);
            PredictorState opt_state;
            auto candidate_tdt = model.tdt(optimized, opt_state);
            auto relative_rms = [](const Tensor& a, const Tensor& b) {
                return ((a - b).square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-12)).item<double>();
            };
            double encoder_relative_rms = relative_rms(original, optimized),
                   ctc_relative_rms = relative_rms(reference, candidate);
            // Different FP32 reduction kernels need not be bit-identical. Bound
            // aggregate drift and require both independent decoders to agree.
            bool numeric = encoder_relative_rms < 1e-5 && ctc_relative_rms < 1e-5;
            bool strict =
                at::allclose(original, optimized, 5e-5, 2e-4) && at::allclose(reference, candidate, 5e-5, 1e-3);
            bool transcripts = reference_text == candidate_text && reference_tdt == candidate_tdt;
            if (!numeric || !transcripts) ++failures;
            auto result = json{{"id", row.at("id")},
                               {"voice", row.at("voice")},
                               {"encoder_max_error", (original - optimized).abs().max().item<float>()},
                               {"ctc_max_error", (reference - candidate).abs().max().item<float>()},
                               {"encoder_relative_rms", encoder_relative_rms},
                               {"ctc_relative_rms", ctc_relative_rms},
                               {"strict_elementwise_pass", strict},
                               {"numeric_pass", numeric},
                               {"transcript_match", transcripts},
                               {"reference", reference_text},
                               {"candidate", candidate_text},
                               {"reference_tdt", reference_tdt},
                               {"candidate_tdt", candidate_tdt}};
            report << result.dump() << '\n' << std::flush;
            if (!numeric || !transcripts) std::cout << result.dump() << std::endl;
        }
        std::cout << "Compared " << cases.size() << " recordings; failures=" << failures << std::endl;
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
