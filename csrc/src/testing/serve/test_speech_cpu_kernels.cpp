// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Compare the CPU kernel path with the original FP32 ATen model and decoders.
#include "model.h"
#include "audio.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace sinfer::speech;
int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) return 77;
    try {
        c10::InferenceMode guard;
        int threads = argc == 5 ? std::stoi(argv[4]) : 4;
        if (threads < 1 || threads > 256) throw std::invalid_argument("Invalid CPU thread count");
        at::set_num_threads(threads);
        at::set_num_interop_threads(1);
        Model reference_model(argv[1], "cpu", CpuKernels::Reference);
        Model model(argv[1], "cpu", CpuKernels::Optimized);
        json cases;
        std::ifstream(argv[2]) >> cases;
        if (!cases.is_array() || cases.empty()) throw std::runtime_error("Provide a nonempty recording manifest");
        std::ofstream report(argv[3]);
        if (!report) throw std::runtime_error("Cannot open the comparison report");
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
            bool exact = at::equal(original.contiguous().view(at::kInt), optimized.contiguous().view(at::kInt)) &&
                         at::equal(reference.contiguous().view(at::kInt), candidate.contiguous().view(at::kInt));
            bool stream_exact = true, stream_numeric = true, stream_transcripts = true;
            int stream_chunks = 0;
            if (model.streaming()) {
                EncoderState ref_encoder, opt_encoder;
                PredictorState ref_predictor, opt_predictor;
                for (int64_t offset = 0; offset < mel.size(2); ++stream_chunks) {
                    int count = std::min<int64_t>((offset ? 8 : 1) + 8 * model.right, mel.size(2) - offset);
                    auto chunk = mel.slice(2, offset, offset + count);
                    auto history = mel.slice(2, offset ? std::max<int64_t>(0, offset - 9) : 0, offset);
                    if (offset && history.size(2) < 9)
                        history = at::constant_pad_nd(history, {9 - history.size(2), 0}, 0);
                    auto signal = at::cat({history, chunk}, 2);
                    auto a = reference_model.encode(signal, &ref_encoder);
                    auto b = model.encode(signal, &opt_encoder);
                    stream_exact &= at::equal(a.contiguous().view(at::kInt), b.contiguous().view(at::kInt));
                    stream_numeric &= at::allclose(a, b, 5e-5, 2e-4) && (!a.numel() || relative_rms(a, b) < 1e-5);
                    offset += count;
                    if (offset < mel.size(2)) {
                        a = a.slice(2, 0, model.right + 1);
                        b = b.slice(2, 0, model.right + 1);
                    }
                    if (a.size(2))
                        stream_transcripts &= reference_model.tdt(a, ref_predictor) == model.tdt(b, opt_predictor);
                }
            }
            if (!numeric || !strict || !transcripts || !stream_numeric || !stream_transcripts) ++failures;
            auto result = json{{"id", row.at("id")},
                               {"threads", threads},
                               {"voice", row.at("voice")},
                               {"encoder_max_error", (original - optimized).abs().max().item<float>()},
                               {"ctc_max_error", (reference - candidate).abs().max().item<float>()},
                               {"encoder_relative_rms", encoder_relative_rms},
                               {"ctc_relative_rms", ctc_relative_rms},
                               {"strict_elementwise_pass", strict},
                               {"bit_identical", exact},
                               {"numeric_pass", numeric},
                               {"transcript_match", transcripts},
                               {"stream_chunks", stream_chunks},
                               {"stream_bit_identical", stream_exact},
                               {"stream_numeric_pass", stream_numeric},
                               {"stream_transcript_match", stream_transcripts},
                               {"reference", reference_text},
                               {"candidate", candidate_text},
                               {"reference_tdt", reference_tdt},
                               {"candidate_tdt", candidate_tdt}};
            report << result.dump() << '\n' << std::flush;
            if (!numeric || !strict || !transcripts || !stream_numeric || !stream_transcripts)
                std::cout << result.dump() << std::endl;
        }
        std::cout << "Compared " << cases.size() << " recordings; failures=" << failures << std::endl;
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
