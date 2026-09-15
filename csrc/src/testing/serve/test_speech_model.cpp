// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Independent NeMo oracle: scripts/serve_speech_reference.py.
#include "model.h"
#include <ATen/Parallel.h>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace sinfer::speech;
int main(int argc, char** argv) {
    if (argc < 3) return 77;
    try {
        c10::InferenceMode guard;
        at::set_num_threads(4);
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
        Model model(argv[1], argc > 3 ? argv[3] : "cpu");
        Weights reference(argv[2], model.device());
        auto metadata = std::filesystem::path(argv[2]);
        metadata.replace_extension(".json");
        json expected;
        std::ifstream(metadata) >> expected;
        auto compare = [](const std::string& name, const Tensor& a, const Tensor& b, float tolerance) {
            if (a.sizes() != b.sizes()) throw std::runtime_error(name + ": wrong shape");
            float error = (a - b).abs().max().item<float>();
            std::cout << name << " max error " << error << std::endl;
            // Allow the rounding differences between CPU and CUDA GEMM/FFT,
            // including large negative log probabilities, while bounding error
            // near zero. Both decoders must also match the oracle text exactly.
            if (!std::isfinite(error) || !at::allclose(a, b, 5e-5, tolerance))
                throw std::runtime_error(name + ": numerical disagreement");
        };
        auto mel = model.mel(reference["pcm"]);
        // CPU MKL FFT and cuFFT differ slightly in low-energy log-mel bins.
        compare("mel", mel, reference["mel"], model.device().is_cpu() ? 2e-4 : 3e-5);
        auto encoded = model.encode(mel);
        compare("encoded", encoded, reference["encoded"], 2e-4);
        auto ctc = model.ctc(encoded);
        compare("ctc", ctc, reference["ctc"], 1e-3);
        if (model.beam(reference["ctc"]) != expected["final"].get<std::string>() ||
            model.beam(ctc) != expected["final"].get<std::string>())
            throw std::runtime_error("CTC+LM transcript disagrees with NeMo");
        EncoderState state;
        PredictorState predictor;
        for (size_t i = 0; i < expected["partials"].size(); ++i) {
            auto prefix = "chunk_" + std::to_string(i);
            auto x = model.encode(reference[prefix], &state);
            compare(prefix, x, reference["chunk_encoded_" + std::to_string(i)], 2e-4);
            if (model.tdt(x, predictor) != expected["partials"][i].get<std::string>())
                throw std::runtime_error(prefix + ": TDT transcript disagrees with NeMo");
        }
        std::cout << "NeMo features, encoder, TDT partials and CTC+LM final agree" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
