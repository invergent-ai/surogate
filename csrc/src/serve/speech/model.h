// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <nlohmann/json.hpp>
#include <sentencepiece_processor.h>
#include <torch/script.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sinfer::speech {
using at::Tensor;
using json = nlohmann::json;

class Weights {
    struct Mapping;
    std::shared_ptr<Mapping> mapping_;
    std::map<std::string, Tensor> tensors_;
public:
    Weights(const std::string& path, at::Device device);

    const Tensor& operator[](const std::string& name) const { return tensors_.at(name); }
};

struct EncoderState {
    std::vector<Tensor> attention, convolution;
    int valid = 0, step = 0;
};

struct PredictorState {
    Tensor h, c, prediction;
    std::vector<int> tokens;
    int time_jump = 0;
};

class Model {
    at::Device device_;
    Weights w_, lm_;
    sentencepiece::SentencePieceProcessor tokenizer_;
    Tensor linear(const Tensor&, const std::string&, bool bias = true) const;
    Tensor norm(const Tensor&, const std::string&) const;
    Tensor conv(const Tensor&, const std::string&, int groups = 1) const;
    std::pair<Tensor, Tensor> predict(int, const Tensor&, const Tensor&) const;
public:
    json config;
    int hidden, heads, layers, left, right, kernel, vocab;
    Model(const std::string& directory, const std::string& device);
    Tensor mel(const Tensor& samples) const;
    Tensor mel_frames(const Tensor& emphasized) const;
    Tensor encode(const Tensor& features, EncoderState* state = nullptr) const;
    Tensor ctc(const Tensor& encoded) const;
    std::string tdt(const Tensor&, PredictorState&) const;
    std::string beam(const Tensor&) const;
    std::string text(const std::vector<int>& ids) const;

    at::Device device() const { return device_; }

    Tensor empty_features() const {
        return at::empty({1, 80, 0}, at::TensorOptions().device(device_));
    }
};

class Stream {
    Model& model_;
    torch::jit::Module vad_;
    EncoderState encoder_;
    PredictorState predictor_;
    Tensor fft_, pending_, history_;
    std::vector<float> audio_, packet_;
    int64_t samples_ = 0, frames_ = 0, total_ = 0, start_ = 0, quiet_ = 0;
    float previous_ = 0;
    bool speech_ = false, closed_ = false;
    std::string partial_, displayed_;
    void reset();
    void features(const Tensor&, bool final);
    void drain(bool final);
    void packet(const float*, size_t, std::vector<json>&);
    void finish_segment(const char*, std::vector<json>&);
public:
    Stream(Model&, const std::string& directory);
    std::vector<json> accept(const std::vector<float>& pcm, bool finish = false);
};
} // namespace sinfer::speech
